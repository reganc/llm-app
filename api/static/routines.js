// Routines: saved tasks that run on a schedule, and the inbox of results.
import { renderMd } from "./md.js";

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const EVERY_DAY = [0, 1, 2, 3, 4, 5, 6];
const WEEKDAYS = [0, 1, 2, 3, 4];
const POLL_MS = 60000;        // unread badge
const RUNNING_POLL_MS = 4000; // while a run is in progress and the panel is open

const STATUS = {
  ok: ["Sourced", "good"],
  unsourced: ["Unsourced", "warn"],
  error: ["Failed", "bad"],
  interrupted: ["Interrupted", "bad"],
  missed: ["Missed", "warn"],
  unchanged: ["Nothing new", ""],
  running: ["Running…", "live"],
};

const same = (a, b) => a.length === b.length && a.every((v, i) => v === b[i]);

export function describeSchedule(s) {
  if (s.kind === "interval") return s.hours === 1 ? "Every hour" : `Every ${s.hours} hours`;
  const days = s.days;
  const when = same(days, EVERY_DAY) ? "Every day"
    : same(days, WEEKDAYS) ? "Weekdays"
    : same(days, [5, 6]) ? "Weekends"
    : days.map(d => DAYS[d]).join(", ");
  return `${when} at ${s.time}`;
}

export function fmtWhen(iso, now = new Date()) {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const time = d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((day(d) - day(now)) / 86400000);
  if (diff === 0) return `today ${time}`;
  if (diff === 1) return `tomorrow ${time}`;
  if (diff === -1) return `yesterday ${time}`;
  return `${d.toLocaleDateString([], { weekday: "short", month: "short", day: "numeric" })} ${time}`;
}

// Sources come from web results; only link what is really a web address.
export const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : "");

export function initRoutines({ $, el, api, toast }) {
  const call = {
    list: () => api.json("/v1/routines"),
    inbox: () => api.json("/v1/routines/inbox?limit=50"),
    get: (id) => api.json(`/v1/routines/${id}`),
    create: (b) => api.json("/v1/routines", { method: "POST", body: JSON.stringify(b) }),
    update: (id, b) => api.json(`/v1/routines/${id}`, { method: "PATCH", body: JSON.stringify(b) }),
    remove: (id) => api.json(`/v1/routines/${id}`, { method: "DELETE" }),
    run: (id) => api.json(`/v1/routines/${id}/run`, { method: "POST", body: "{}" }),
    read: (id, run) => api.json(`/v1/routines/${id}/runs/${run}/read`, { method: "POST", body: "{}" }),
    forget: (id) => api.json(`/v1/routines/${id}/forget`, { method: "POST", body: "{}" }),
    draft: (text) => api.json("/v1/routines/draft", { method: "POST", body: JSON.stringify({ text }) }),
    telegram: () => api.json("/v1/notify/telegram/status"),
    readAll: () => api.json("/v1/routines/inbox/read", { method: "POST", body: "{}" }),
  };
  // view: {name:"inbox"} | {name:"routine", id} | {name:"edit", id|null, draft?}
  const state = { routines: [], unread: 0, tz: "", scheduler: true, view: { name: "inbox" }, timer: 0 };
  const modal = $("#routinesModal");
  const detail = $("#routinesDetail");

  const badge = ({ status, sources }) => {
    let [label, tone] = STATUS[status] || [status, "warn"];
    if (status === "ok" && !sources?.length) label = "Done"; // no tools, nothing to cite
    return el("span", { class: `run-badge ${tone}`, text: label });
  };
  const fail = (what, e) => toast(`${what}: ${e.message}`, "error", 5000);

  function setUnread(n) {
    state.unread = n;
    const node = $("#routinesCount");
    node.textContent = n > 99 ? "99+" : String(n);
    node.hidden = n === 0;
  }

  async function refreshUnread() {
    if (document.hidden) return;
    try { setUnread((await call.list()).unread); } catch { /* badge only */ }
  }

  async function refreshList() {
    const data = await call.list();
    state.routines = data.routines;
    state.tz = data.default_timezone;
    state.scheduler = data.scheduler_enabled;
    setUnread(data.unread);
    renderList();
  }

  function renderList() {
    const list = $("#routinesList");
    list.replaceChildren();
    const row = (active, title, meta, icon, onclick) => el("button", {
      type: "button", class: "library-row" + (active ? " active" : ""), onclick,
    }, [
      el("div", { class: "lib-row-icon", text: icon }),
      el("div", { class: "lib-row-body" }, [
        el("div", { class: "lib-row-title", title, text: title }),
        el("div", { class: "lib-row-meta", text: meta }),
      ]),
    ]);
    const v = state.view;
    list.append(row(v.name === "inbox", "Inbox",
      state.unread ? `${state.unread} unread` : "All caught up", "📥", () => show({ name: "inbox" })));
    if (!state.routines.length) {
      list.append(el("div", { class: "library-list-empty",
        text: "No routines yet. Create one to have a task run on a schedule." }));
    }
    for (const r of state.routines) {
      const running = r.last_run?.status === "running";
      const meta = running ? "Running…"
        : r.enabled ? `${describeSchedule(r.schedule)} · next ${fmtWhen(r.next_run_at)}` : "Paused";
      list.append(row(v.id === r.id, r.name, meta, running ? "◌" : r.enabled ? "⏱" : "⏸",
        () => show({ name: "routine", id: r.id })));
    }
    $("#routinesStats").textContent = state.scheduler
      ? `${state.routines.length} ${state.routines.length === 1 ? "routine" : "routines"}`
      : "Scheduler is off (ROUTINES_ENABLED=false)";
  }

  // ── results ────────────────────────────────────────────────────────────────
  function runCard(run, { routineId, title, onRead }) {
    const body = el("div", { class: "run-body md" });
    if (run.content) body.innerHTML = renderMd(run.content);
    const sources = (run.sources || []).map(s => {
      const label = `[${s.marker}] ${s.title}${s.preview ? " (search preview only)" : ""}`;
      const url = safeUrl(s.url);
      return el("li", {}, [url
        ? el("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: label })
        : el("span", { text: label })]);
    });
    const children = [
      el("div", { class: "run-head" }, [
        badge(run),
        ...(title ? [el("strong", { class: "run-title", text: title })] : []),
        el("span", { class: "muted-small", text: fmtWhen(run.started_at) + (run.trigger === "manual" ? " · run by hand" : "") }),
      ]),
    ];
    if (run.error) children.push(el("p", { class: "run-note bad", text: run.error }));
    if (run.notified?.startsWith("failed")) {
      children.push(el("p", { class: "run-note", text: `Telegram notification ${run.notified}` }));
    }
    if (run.note) children.push(el("p", { class: "run-note", text: run.note }));
    if (run.content) children.push(body);
    if (sources.length) children.push(el("details", { class: "run-sources" }, [
      el("summary", { text: `${sources.length} source${sources.length === 1 ? "" : "s"}` }),
      el("ul", {}, sources),
    ]));
    if (run.removed?.length) children.push(el("details", { class: "run-sources" }, [
      el("summary", { text: `${run.removed.length} removed` }),
      el("ul", {}, run.removed.map(p => el("li", { text: `${p.text} (${p.reason})` }))),
    ]));
    const card = el("article", { class: "run-card" + (run.read || run.status === "running" ? "" : " unread") }, children);
    if (!run.read && run.status !== "running") {
      card.querySelector(".run-head").append(el("button", {
        type: "button", class: "ghost-btn run-read", text: "Mark read",
        onclick: async (e) => {
          e.target.disabled = true;
          try {
            await call.read(routineId, run.id);
            run.read = true;
            card.classList.remove("unread");
            e.target.remove();
            setUnread(Math.max(0, state.unread - 1));
            renderList();
            onRead?.();
          } catch (err) { e.target.disabled = false; fail("Could not mark as read", err); }
        },
      }));
    }
    return card;
  }

  const head = (title, meta, actions) => el("div", { class: "library-detail-head" }, [
    el("div", { class: "lib-detail-title", text: title }),
    ...(meta ? [el("div", { class: "lib-detail-url muted-small", text: meta })] : []),
    ...(actions?.length ? [el("div", { class: "routine-actions" }, actions)] : []),
  ]);
  const btn = (text, onclick, cls = "ghost-btn") => el("button", { type: "button", class: cls, text, onclick });

  async function renderInbox(view) {
    const { runs, unread } = await call.inbox();
    if (state.view !== view) return;
    setUnread(unread);
    const actions = unread ? [btn("Mark all read", async () => {
      try { await call.readAll(); await refresh(); } catch (e) { fail("Could not mark as read", e); }
    })] : [];
    const body = el("div", { class: "library-content run-list" });
    if (!runs.length) {
      body.append(el("div", { class: "library-empty" }, [
        el("div", { class: "library-empty-mark", text: "📥" }),
        el("p", { text: "Results from your routines will show up here." }),
        btn("New routine", () => show({ name: "edit", id: null }), "primary-btn"),
      ]));
    }
    for (const run of runs) body.append(runCard(run, { routineId: run.routine_id, title: run.routine_name }));
    detail.replaceChildren(head("Inbox", "Latest results from every routine", actions), body);
  }

  async function renderRoutine(view) {
    const r = await call.get(view.id);
    if (state.view !== view) return;
    const running = r.runs[0]?.status === "running";
    const runBtn = btn(running ? "Running…" : "Run now", async () => {
      runBtn.disabled = true;
      try { await call.run(r.id); toast(`Running “${r.name}”`, "info"); await refresh(); }
      catch (e) { runBtn.disabled = false; fail("Could not start", e); }
    }, "primary-btn");
    runBtn.disabled = running;
    const remove = btn("Delete", async () => {
      if (!confirm(`Delete “${r.name}” and its results? This cannot be undone.`)) return;
      try { await call.remove(r.id); toast("Routine deleted", "success", 1500); await show({ name: "inbox" }); }
      catch (e) { fail("Could not delete", e); }
    });
    remove.classList.add("danger");
    const actions = [
      runBtn,
      btn("Edit", () => show({ name: "edit", id: r.id })),
      btn(r.enabled ? "Pause" : "Resume", async () => {
        try { await call.update(r.id, { enabled: !r.enabled }); await refresh(); }
        catch (e) { fail("Could not update", e); }
      }),
      remove,
    ];
    if (r.remembers) {
      actions.splice(3, 0, btn("Forget earlier results", async () => {
        try { await call.forget(r.id); toast("It will compare from scratch", "success", 1800); await refresh(); }
        catch (e) { fail("Could not clear", e); }
      }));
    }
    const extras = [r.only_new && "only new", r.notify !== "off" && `Telegram ${r.notify}`].filter(Boolean);
    const tools = [r.sources.web && (r.sources.news ? "recent news" : "web"), r.sources.library && "library"].filter(Boolean).join(" + ") || "no tools";
    const searches = r.search_phrase && `searches “${r.search_phrase}”`;
    const next = r.enabled ? `next ${fmtWhen(r.next_run_at)}` : "paused";
    const body = el("div", { class: "library-content run-list" }, [
      el("blockquote", { class: "routine-prompt", text: r.prompt }),
    ]);
    if (!r.runs.length) body.append(el("p", { class: "muted", text: "No results yet. Use Run now to try it." }));
    for (const run of r.runs) body.append(runCard(run, { routineId: r.id }));
    detail.replaceChildren(head(r.name, `${describeSchedule(r.schedule)} (${r.timezone}) · ${[next, tools, searches, ...extras].filter(Boolean).join(" · ")}`, actions), body);
  }

  // ── form ───────────────────────────────────────────────────────────────────
  async function renderForm(view) {
    const r = view.id ? await call.get(view.id) : view.draft?.routine || null;
    const telegram = await call.telegram().catch(() => ({ ready: false }));
    if (state.view !== view) return;
    const form = $("#routineFormTpl").content.firstElementChild.cloneNode(true);
    const f = (name) => form.elements[name];
    const s = r?.schedule || { kind: "daily", time: "07:00", days: WEEKDAYS };
    const dayBox = form.querySelector(".day-picker");
    DAYS.forEach((label, i) => dayBox.append(el("label", { class: "day" }, [
      el("input", { type: "checkbox", name: "day", value: String(i), ...((s.days || EVERY_DAY).includes(i) ? { checked: "" } : {}) }),
      el("span", { text: label }),
    ])));
    f("name").value = r?.name || "";
    f("prompt").value = r?.prompt || "";
    f("query").value = r?.query || "";
    f("kind").value = s.kind;
    f("time").value = s.time || "07:00";
    f("hours").value = s.hours || 6;
    f("timezone").value = r?.timezone || state.tz;
    f("web").checked = r ? r.sources.web : true;
    f("library").checked = r ? r.sources.library : false;
    f("news").checked = !!r?.sources.news;
    f("extra_sites").checked = !!r?.sources.extra_sites;
    if (r?.search_phrase && !r.query) f("query").placeholder = r.search_phrase;
    f("only_new").checked = !!r?.only_new;
    f("notify").value = r?.notify || "off";
    form.querySelector(".notify-hint").hidden = telegram.ready;
    form.querySelector(".notify-full-hint").hidden = true;
    if (view.draft) {
      const box = form.querySelector(".draft-note");
      box.hidden = false;
      box.querySelector("strong").textContent = `${describeSchedule(r.schedule)}.`;
      box.querySelector("ul").replaceChildren(...view.draft.warnings.map(w => el("li", { text: w })));
      const send = form.querySelector(".send-instead");
      send.hidden = !view.draft.sendInstead;
      send.onclick = () => { modal.close(); view.draft.sendInstead(); };
    }
    const sync = () => {
      form.querySelector(".daily-fields").hidden = f("kind").value !== "daily";
      form.querySelector(".interval-fields").hidden = f("kind").value !== "interval";
      form.querySelectorAll(".query-field").forEach(n => { n.hidden = !f("web").checked; });
      form.querySelector(".extra-sites-field").hidden = !f("web").checked || !f("news").checked;
      form.querySelector(".notify-full-hint").hidden = f("notify").value !== "full";
      f("hours").disabled = f("kind").value !== "interval";
    };
    f("kind").onchange = f("web").onchange = f("news").onchange = f("notify").onchange = sync;
    sync();
    form.querySelector(".cancel").onclick = () => show(view.id ? { name: "routine", id: view.id } : { name: "inbox" });
    form.onsubmit = async (e) => {
      e.preventDefault();
      const days = [...form.querySelectorAll("input[name=day]:checked")].map(x => +x.value);
      const daily = f("kind").value === "daily";
      if (daily && !days.length) { toast("Choose at least one day", "error"); return; }
      const body = {
        name: f("name").value.trim(),
        prompt: f("prompt").value.trim(),
        query: f("query").value.trim() || null,
        timezone: f("timezone").value.trim(),
        schedule: daily ? { kind: "daily", time: f("time").value, days }
          : { kind: "interval", hours: parseInt(f("hours").value, 10) },
        sources: { web: f("web").checked, library: f("library").checked, news: f("news").checked,
          extra_sites: f("news").checked && f("extra_sites").checked },
        only_new: f("only_new").checked,
        notify: f("notify").value,
      };
      const save = form.querySelector("[type=submit]");
      save.disabled = true;
      try {
        const saved = view.id ? await call.update(view.id, body) : await call.create(body);
        toast(view.id ? "Routine saved" : "Routine created", "success", 1800);
        await show({ name: "routine", id: saved.id });
      } catch (err) { save.disabled = false; fail("Could not save", err); }
    };
    detail.replaceChildren(head(view.id ? "Edit routine" : view.draft ? "Check this routine" : "New routine",
      "Runs on its own with read-only tools. It can search, summarize and notify you. It cannot change anything or contact anyone else."), form);
    f("name").focus();
  }

  // ── navigation ─────────────────────────────────────────────────────────────
  async function renderView() {
    const view = state.view;
    const draw = { inbox: renderInbox, routine: renderRoutine, edit: renderForm }[view.name];
    try { await draw(view); } catch (e) {
      if (state.view !== view) return;
      if (view.id && e.message.startsWith("404")) { await show({ name: "inbox" }); return; }
      detail.replaceChildren(el("div", { class: "library-empty" }, [el("p", { text: "Could not load: " + e.message })]));
    }
  }

  async function refresh() {
    try { await refreshList(); } catch (e) { fail("Routines unavailable", e); }
    if (!["edit", "drafting"].includes(state.view.name)) await renderView();
    schedulePoll();
  }

  async function show(view) {
    state.view = view;
    detail.replaceChildren(el("div", { class: "library-loading muted", text: "Loading…" }));
    try { await refreshList(); } catch (e) { fail("Routines unavailable", e); }
    await renderView();
    schedulePoll();
  }

  // Follow a run while it is in progress; stop when the panel closes.
  function schedulePoll() {
    clearTimeout(state.timer);
    if (!modal.open || ["edit", "drafting"].includes(state.view.name)) return;
    if (!state.routines.some(r => r.last_run?.status === "running")) return;
    state.timer = setTimeout(refresh, RUNNING_POLL_MS);
  }

  $("#routinesNewBtn").onclick = () => show({ name: "edit", id: null });
  modal.addEventListener("close", () => clearTimeout(state.timer));
  const open = () => { if (!modal.open) modal.showModal(); return show({ name: "inbox" }); };
  $("#routinesBtn").onclick = open;
  if (!window.__routinesPoll) window.__routinesPoll = setInterval(refreshUnread, POLL_MS);
  // Plain language → a filled-in form. Nothing is saved until the user presses Save.
  async function openDraft(text, sendInstead) {
    if (!modal.open) modal.showModal();
    if (!text) return show({ name: "edit", id: null });
    const view = { name: "drafting" };
    state.view = view;
    detail.replaceChildren(el("div", { class: "library-loading muted", text: "Reading your request…" }));
    try {
      const draft = await call.draft(text);
      if (state.view === view) await show({ name: "edit", id: null, draft: { ...draft, sendInstead } });
    } catch (e) {
      fail("Could not draft the routine", e);
      if (state.view === view) await show({ name: "edit", id: null });
    }
  }
  return { open, openDraft, refreshUnread };
}
