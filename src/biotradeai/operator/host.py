"""Agent host: the always-on process that fires schedules and runs jobs.

    biotradeai host            # run in the foreground (Ctrl+C to stop)

Every few seconds it:
  1. writes a heartbeat (the UI shows "Agent host: running");
  2. queues any schedule that is due (local time, missed runs within 2 hours);
  3. starts queued runs, one per job at a time, each as its own process with its
     output in ``data/logs/runs/run-<id>.log``;
  4. finishes runs that exited, and stops runs you cancelled;
  5. expires approve-later orders nobody decided in time.

Only one host runs at a time (a fresh heartbeat from another live process blocks a
second one).
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import IO, Any, Protocol

from biotradeai.operator import jobs
from biotradeai.operator.db import OperatorDB, utcnow
from biotradeai.paths import REPO_ROOT, data_dir

HEARTBEAT_KEY = "operator.host"
STALE_AFTER = timedelta(seconds=30)
TICK_SECONDS = 5.0


def log_dir() -> Path:
    return data_dir() / "logs" / "runs"


class Proc(Protocol):
    pid: int

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


Spawner = Callable[[list[str], Path], Proc]


def spawn(argv: list[str], log_path: Path) -> Proc:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh: IO[bytes] = log_path.open("ab")
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(  # noqa: S603 - fixed module entry, validated argv
        [sys.executable, "-m", "biotradeai", *argv], cwd=REPO_ROOT, stdout=fh,
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env, creationflags=flags)
    fh.close()  # the child keeps its own handle
    return proc


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel = ctypes.windll.kernel32  # type: ignore[attr-defined,unused-ignore]
        h = kernel.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = kernel.GetExitCodeProcess(h, ctypes.byref(code))
        kernel.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@dataclass(frozen=True)
class HostStatus:
    running: bool
    pid: int | None
    heartbeat: str | None
    started_at: str | None


def host_status(db: OperatorDB) -> HostStatus:
    raw = db.store.kv_get(HEARTBEAT_KEY)
    if not raw:
        return HostStatus(False, None, None, None)
    d = json.loads(raw)
    beat = datetime.fromisoformat(d["heartbeat"])
    alive = utcnow() - beat < STALE_AFTER and pid_alive(int(d["pid"]))
    return HostStatus(alive, d["pid"], d["heartbeat"], d.get("started_at"))


class AgentHost:
    def __init__(self, db: OperatorDB | None = None, *, spawner: Spawner = spawn,
                 now_local: Callable[[], datetime] = lambda: datetime.now().astimezone(),
                 log_root: Path | None = None) -> None:
        self.db = db or OperatorDB()
        self._spawn = spawner
        self._now = now_local
        self._log_root = log_root or log_dir()
        self._procs: dict[int, Proc] = {}
        self._started = utcnow().isoformat()
        self._stop = False

    # ------------------------------------------------------------------ lifecycle

    def claim(self) -> None:
        st = host_status(self.db)
        if st.running and st.pid != os.getpid():
            raise RuntimeError(f"another agent host is running (pid {st.pid})")
        self._beat()
        # Runs left 'running' by a host that died: their process is gone or orphaned.
        for r in self.db.runs(status=("running",), limit=200):
            if r.pid and pid_alive(r.pid):
                _terminate_pid(r.pid)
            self.db.mark_finished(r.id, status="failed", exit_code=None,
                                  error="agent host restarted while this run was active")
        self.db.store.append_audit("operator.host_started", {"pid": os.getpid()})

    def _beat(self) -> None:
        self.db.store.kv_set(HEARTBEAT_KEY, json.dumps({
            "pid": os.getpid(), "heartbeat": utcnow().isoformat(),
            "started_at": self._started}))

    def stop(self, *_: Any) -> None:
        self._stop = True

    def serve(self, tick: float = TICK_SECONDS) -> None:
        self.claim()
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        while not self._stop:
            self.tick()
            time.sleep(tick)
        self.db.store.kv_set(HEARTBEAT_KEY, "")
        self.db.store.append_audit("operator.host_stopped", {"pid": os.getpid(),
                                                             "active_runs": list(self._procs)})

    # ------------------------------------------------------------------ one pass

    def tick(self) -> None:
        self._beat()
        self._fire_schedules()
        self._reap()
        self._start_queued()
        self._expire_pending()

    def _fire_schedules(self) -> None:
        now = self._now()
        for job_id, d in jobs.JOBS.items():
            s = jobs.schedule_for(self.db, job_id)
            slot = jobs.due_fire(s, now)
            if slot is None:
                continue
            self.db.mark_fired(job_id, slot)
            if self.db.runs(job=job_id, status=("queued", "running"), limit=1):
                self.db.store.append_audit("operator.schedule_skipped", {
                    "job": job_id, "slot": slot.isoformat(), "reason": "already running"})
                continue
            p = d.validate(s.params)
            self.db.enqueue(job_id, trigger="schedule", by="schedule", params=p,
                            argv=d.argv(p))

    def _start_queued(self) -> None:
        busy = {r.job for r in self.db.runs(status=("running",), limit=200)}
        for r in reversed(self.db.runs(status=("queued",), limit=200)):  # oldest first
            if r.cancel_requested:
                self.db.mark_finished(r.id, status="cancelled", exit_code=None,
                                      error="cancelled before it started")
                continue
            if r.job in busy:
                continue
            log = self._log_root / f"run-{r.id}.log"
            try:
                log.parent.mkdir(parents=True, exist_ok=True)
                proc = self._spawn(r.argv, log)
            except OSError as exc:
                self.db.mark_finished(r.id, status="failed", exit_code=None,
                                      error=f"could not start: {exc}")
                continue
            self._procs[r.id] = proc
            busy.add(r.job)
            self.db.mark_started(r.id, pid=proc.pid, log_path=str(log))
            self.db.store.append_audit("operator.run_started", {"run": r.id, "job": r.job,
                                                                "trigger": r.trigger})

    def _reap(self) -> None:
        for r in self.db.runs(status=("running",), limit=200):
            proc = self._procs.get(r.id)
            if proc is None:  # not ours (should not happen after claim): leave it alone
                continue
            code = proc.poll()
            if code is None and r.cancel_requested:
                proc.terminate()
                for _ in range(20):
                    if (code := proc.poll()) is not None:
                        break
                    time.sleep(0.1)
                else:
                    proc.kill()
                self.db.mark_finished(r.id, status="cancelled", exit_code=code,
                                      error="cancelled by the operator")
            elif code is not None:
                status = "succeeded" if code == 0 else "failed"
                self.db.mark_finished(r.id, status=status, exit_code=code,
                                      error=None if code == 0 else _tail_error(r.log_path))
            else:
                continue
            self._procs.pop(r.id, None)
            done = self.db.get_run(r.id)
            self.db.store.append_audit("operator.run_finished", {
                "run": r.id, "job": r.job, "status": done.status if done else "?",
                "exit_code": code})

    def _expire_pending(self) -> None:
        from biotradeai.operator.trades import expire_stale  # local: avoids import cycle

        expire_stale(self.db)


def _terminate_pid(pid: int) -> None:
    with contextlib.suppress(OSError):
        os.kill(pid, signal.SIGTERM)


def _tail_error(log_path: str | None, lines: int = 3) -> str:
    if not log_path or not Path(log_path).exists():
        return "exited with an error"
    text = Path(log_path).read_text(encoding="utf-8", errors="replace").strip().splitlines()
    tail = [x for x in text[-lines:] if x.strip()]
    return " | ".join(tail)[-400:] or "exited with an error"


def read_log(log_path: str | None, max_chars: int = 20_000) -> str:
    if not log_path:
        return ""
    p = Path(log_path)
    if not p.exists():
        return ""
    data = p.read_text(encoding="utf-8", errors="replace")
    return data[-max_chars:]


def start_detached() -> int:
    """Start the agent host in the background (used by the UI's Start button)."""
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe")
    runner = str(pyw if os.name == "nt" and pyw.exists() else exe)
    flags = 0
    if os.name == "nt":
        flags = (getattr(subprocess, "DETACHED_PROCESS", 0)
                 | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                 | getattr(subprocess, "CREATE_NO_WINDOW", 0))
    log = data_dir() / "logs" / "host.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as fh:
        proc = subprocess.Popen(  # noqa: S603 - fixed module entry
            [runner, "-m", "biotradeai", "host"], cwd=REPO_ROOT, stdout=fh,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags,
            start_new_session=os.name != "nt")
    return proc.pid
