"""Operator app commands.

    biotradeai host                 # agent host: fires schedules, runs jobs (keep it running)
    biotradeai ui                   # operator app in its own window (starts the host if needed)
    biotradeai ui --browser         # same app in your default browser instead
"""

from __future__ import annotations

import argparse

from biotradeai.operator.host import AgentHost


def cmd_host(args: argparse.Namespace) -> int:
    host = AgentHost()
    print("BioTradeAI agent host running. Ctrl+C to stop.")
    host.serve(tick=args.tick)
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    try:
        from biotradeai.operator.desktop import launch
    except ImportError as exc:  # the UI extras are optional
        raise PermissionError(f"the operator app needs its extras: pip install -e \".[ui]\" "
                              f"({exc.name} is missing)") from exc
    return launch(browser=args.browser, port=args.port, start_host=not args.no_host)


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    h = sub.add_parser("host", help="agent host: schedules and runs the agents")
    h.add_argument("--tick", type=float, default=5.0, help=argparse.SUPPRESS)
    h.set_defaults(fn=cmd_host)
    u = sub.add_parser("ui", help="open the operator app")
    u.add_argument("--browser", action="store_true", help="open in the default browser")
    u.add_argument("--port", type=int, default=0, help="local port (default: any free port)")
    u.add_argument("--no-host", action="store_true", help="don't start the agent host")
    u.set_defaults(fn=cmd_ui)
