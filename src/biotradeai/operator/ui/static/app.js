// BioTradeAI Operator: small progressive enhancements. Every action works without JS.
(function () {
  "use strict";

  // Flash messages: tidy the URL and allow dismiss.
  const url = new URL(window.location.href);
  if (url.searchParams.has("ok") || url.searchParams.has("err")) {
    url.searchParams.delete("ok");
    url.searchParams.delete("err");
    history.replaceState(null, "", url.pathname + (url.search || "") + url.hash);
  }
  document.querySelectorAll(".flash .x").forEach((b) =>
    b.addEventListener("click", () => b.parentElement.remove()));
  document.querySelectorAll(".flash.ok").forEach((f) => setTimeout(() => f.remove(), 7000));

  // Dialogs: <button data-open="dialog-id">, <button data-close>.
  document.querySelectorAll("[data-open]").forEach((b) =>
    b.addEventListener("click", (e) => {
      e.preventDefault();
      const d = document.getElementById(b.dataset.open);
      if (d) { d.showModal(); const i = d.querySelector("input,textarea"); if (i) i.focus(); }
    }));
  document.querySelectorAll("dialog [data-close]").forEach((b) =>
    b.addEventListener("click", (e) => { e.preventDefault(); b.closest("dialog").close(); }));

  // Typed confirmation: <input data-match="ONCO"> enables the form's submit button.
  document.querySelectorAll("input[data-match]").forEach((inp) => {
    const btn = inp.form.querySelector("button[type=submit]");
    const want = inp.dataset.match.trim().toUpperCase();
    const sync = () => { btn.disabled = inp.value.trim().toUpperCase() !== want; };
    inp.addEventListener("input", sync);
    sync();
  });
  // Required text (reasons): <textarea data-required> enables submit when non-empty.
  document.querySelectorAll("[data-required]").forEach((inp) => {
    const btn = inp.form.querySelector("button[type=submit]");
    const sync = () => { btn.disabled = inp.value.trim().length < 3; };
    inp.addEventListener("input", sync);
    sync();
  });
  // Prevent double submits.
  document.querySelectorAll("form").forEach((f) =>
    f.addEventListener("submit", () => {
      f.querySelectorAll("button[type=submit]").forEach((b) => {
        setTimeout(() => { b.disabled = true; }, 0);
      });
    }));

  // Clickable rows.
  document.querySelectorAll("tr[data-href]").forEach((tr) =>
    tr.addEventListener("click", (e) => {
      if (e.target.closest("a,button,form,input")) return;
      window.location.href = tr.dataset.href;
    }));

  // Live status in the sidebar and header.
  const setCount = (name, n) => {
    document.querySelectorAll(`[data-count="${name}"]`).forEach((el) => {
      el.textContent = n > 0 ? String(n) : "";
      if (n > 0) el.removeAttribute("data-zero"); else el.setAttribute("data-zero", "");
    });
  };
  async function poll() {
    try {
      const r = await fetch("/api/status", { credentials: "same-origin" });
      if (!r.ok) return;
      const s = await r.json();
      setCount("trades", s.trades);
      setCount("watch", s.watch);
      const host = document.getElementById("host-pill");
      if (host) {
        host.className = "pill " + (s.host ? "ok" : "warn");
        host.querySelector(".t").textContent = s.host ? "Agent host running" : "Agent host stopped";
      }
      const run = document.getElementById("running-pill");
      if (run) {
        run.hidden = s.running.length === 0;
        run.querySelector(".t").textContent = s.running.length === 1
          ? s.running[0].job + " running" : s.running.length + " agents running";
      }
      const page = document.body.dataset.watchRunning;
      if (page !== undefined && String(s.running.length) !== page && !busy()) {
        window.location.reload();
      }
    } catch (_) { /* server closed */ }
  }
  const busy = () => document.querySelector("dialog[open]") ||
    (document.activeElement && document.activeElement.matches("input,textarea,select"));
  poll();
  setInterval(poll, 5000);

  // Live run log.
  const log = document.getElementById("log");
  if (log && log.dataset.run && log.dataset.active === "1") {
    const id = log.dataset.run;
    const tick = async () => {
      try {
        const r = await fetch(`/api/runs/${id}`, { credentials: "same-origin" });
        const s = await r.json();
        const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 20;
        log.textContent = s.log || "Waiting for output…";
        if (atEnd) log.scrollTop = log.scrollHeight;
        const d = document.getElementById("run-duration");
        if (d) d.textContent = s.duration;
        if (!s.active) { window.location.reload(); return; }
      } catch (_) { /* ignore */ }
      setTimeout(tick, 2000);
    };
    log.scrollTop = log.scrollHeight;
    setTimeout(tick, 1500);
  }
})();
