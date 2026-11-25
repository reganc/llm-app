// Watched pages: shown inside the Routines panel. Alerts quote the page exactly.
import { fmtWhen, safeUrl } from "./routines.js";

const EVERY = { 15: "15 minutes", 30: "30 minutes", 60: "hour", 180: "3 hours", 360: "6 hours", 720: "12 hours", 1440: "day" };

export function hostOf(url) {
  try {
    const u = new URL(url);
    return /^https?:$/.test(u.protocol) ? u.hostname.replace(/\.$/, "") : "";
  } catch { return ""; }
}

export function describeRule(r) {
  if (r.kind === "number") return r.when === "changes" ? `“${r.label}” changes` : `“${r.label}” goes ${r.when} ${r.limit}`;
  if (r.kind === "phrase") return `“${r.phrase}” ${{ appears: "appears", disappears: "disappears", either: "appears or disappears" }[r.when]}`;
  return r.only ? `lines with “${r.only}” change` : "anything changes";
}

export function initWatches({ $, el, api, toast }, { detail, show, head, btn, fail, refresh }) {
  const call = {
    list: () => api.json("/v1/watches"),
    get: (id) => api.json(`/v1/watches/${id}`),
    create: (b) => api.json("/v1/watches", { method: "POST", body: JSON.stringify(b) }),
    update: (id, b) => api.json(`/v1/watches/${id}`, { method: "PATCH", body: JSON.stringify(b) }),
    remove: (id) => api.json(`/v1/watches/${id}`, { method: "DELETE" }),
    check: (id) => api.json(`/v1/watches/${id}/check`, { method: "POST", body: "{}" }),
    read: (id) => api.json(`/v1/watches/${id}/read`, { method: "POST", body: "{}" }),
    preview: (b) => api.json("/v1/watches/preview", { method: "POST", body: JSON.stringify(b) }),
    telegram: () => api.json("/v1/notify/telegram/status"),
  };
  const why = (e) => {
    try {
      const d = JSON.parse(e.message.slice(e.message.indexOf("{"))).detail;
      return typeof d === "string" ? d : d?.[0]?.msg || e.message;
    } catch { return e.message; }
  };

  function row(w, active) {
    const meta = !w.enabled ? "Paused" : w.error ? "Could not be read" : w.shown || "Not checked yet";
    return el("button", { type: "button", class: "library-row" + (active ? " active" : ""), onclick: () => show({ name: "watch", id: w.id }) }, [
      el("div", { class: "lib-row-icon", text: w.unread ? "●" : w.error ? "!" : "👁" }),
      el("div", { class: "lib-row-body" }, [
        el("div", { class: "lib-row-title", title: w.name, text: w.name }),
        el("div", { class: "lib-row-meta", text: meta }),
      ]),
    ]);
  }

  function alertCard(a) {
    const children = [
      el("div", { class: "run-head" }, [
        el("span", { class: `run-badge ${a.kind === "error" ? "bad" : "good"}`, text: a.kind === "error" ? "Problem" : "Changed" }),
        el("span", { class: "muted-small", text: fmtWhen(a.at) }),
      ]),
      el("pre", { class: "watch-text", text: a.text }),
    ];
    if (a.notified?.startsWith("failed")) children.push(el("p", { class: "run-note", text: `Telegram notification ${a.notified}` }));
    return el("article", { class: "run-card" + (a.read ? "" : " unread") }, children);
  }

  async function renderWatch(view, isCurrent) {
    const w = await call.get(view.id);
    if (!isCurrent()) return;
    const check = btn("Check now", async () => {
      check.disabled = true;
      check.textContent = "Checking…";
      try { await call.check(w.id); } catch (e) { fail("Could not check", e); }
      await refresh();
    }, "primary-btn");
    const remove = btn("Delete", async () => {
      if (!confirm(`Stop watching “${w.name}” and delete its alerts?`)) return;
      try { await call.remove(w.id); await show({ name: "inbox" }); } catch (e) { fail("Could not delete", e); }
    });
    remove.classList.add("danger");
    const url = safeUrl(w.url);
    const body = el("div", { class: "library-content run-list" }, [
      el("blockquote", { class: "routine-prompt", text: w.error ? `${w.error} (${w.failures} in a row)` : w.shown || "Not checked yet." }),
      ...(url ? [el("p", { class: "muted-small" }, [el("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: w.url })])] : []),
      ...w.alerts.map(alertCard),
      ...(w.alerts.length ? [] : [el("p", { class: "muted", text: "No changes yet. The first check only sets the starting point." })]),
    ]);
    detail.replaceChildren(head(w.name, [
      `Tell me when ${describeRule(w.rule)}`, `every ${EVERY[w.every_minutes]}`,
      w.enabled ? (w.last_checked_at ? `checked ${fmtWhen(w.last_checked_at)}` : "first check soon") : "paused",
      ...(w.notify ? ["Telegram"] : []),
    ].join(" · "), [
      check, btn("Edit", () => show({ name: "watch-edit", id: w.id })),
      btn(w.enabled ? "Pause" : "Resume", async () => {
        try { await call.update(w.id, { enabled: !w.enabled }); await refresh(); } catch (e) { fail("Could not update", e); }
      }), remove,
    ]), body);
    if (w.unread) call.read(w.id).catch(() => {});
  }

  async function renderForm(view, isCurrent) {
    const w = view.id ? await call.get(view.id) : null;
    const telegram = await call.telegram().catch(() => ({ ready: false }));
    if (!isCurrent()) return;
    const form = $("#watchFormTpl").content.firstElementChild.cloneNode(true);
    const f = (name) => form.elements[name];
    const r = w?.rule || { kind: "number", when: "changes" };
    f("name").value = w?.name || "";
    f("url").value = w?.url || "";
    f("kind").value = r.kind;
    f("label").value = r.label || "";
    f("numberWhen").value = r.kind === "number" ? r.when : "changes";
    f("limit").value = r.limit ?? "";
    f("phrase").value = r.phrase || "";
    f("phraseWhen").value = r.kind === "phrase" ? r.when : "appears";
    f("only").value = r.only || "";
    f("every").value = w?.every_minutes || 60;
    f("notify").checked = w ? w.notify : telegram.ready;
    const out = form.querySelector(".watch-preview");
    const sync = () => {
      const kind = f("kind").value, host = hostOf(f("url").value);
      form.querySelectorAll("[class*=kind-]").forEach(n => { n.hidden = !n.classList.contains(`kind-${kind}`); });
      form.querySelector(".limit-field").hidden = f("numberWhen").value === "changes";
      f("label").required = kind === "number";
      f("limit").required = kind === "number" && f("numberWhen").value !== "changes";
      f("phrase").required = kind === "phrase";
      form.querySelector(".approve-text").textContent = host ? `Let llm-app read ${host} on schedule` : "Let llm-app read this site on schedule";
      f("approve").checked = !!w && host === w.approved_host;
      out.hidden = true;
    };
    f("kind").onchange = f("numberWhen").onchange = f("url").oninput = sync;
    sync();
    const rule = () => {
      const kind = f("kind").value;
      if (kind === "number") {
        const when = f("numberWhen").value;
        return { kind, label: f("label").value.trim(), when, limit: when === "changes" ? null : parseFloat(f("limit").value) };
      }
      if (kind === "phrase") return { kind, phrase: f("phrase").value.trim(), when: f("phraseWhen").value };
      return { kind, only: f("only").value.trim() };
    };
    form.querySelector(".cancel").onclick = () => show(view.id ? { name: "watch", id: view.id } : { name: "inbox" });
    const preview = form.querySelector(".preview");
    preview.onclick = async () => {
      if (!form.reportValidity()) return;
      preview.disabled = true;
      out.hidden = false;
      out.replaceChildren(el("p", { class: "muted", text: "Reading the page…" }));
      try {
        const p = await call.preview({ url: f("url").value.trim(), rule: rule() });
        out.replaceChildren(el("p", {}, [el("strong", { text: "Found: " }), el("span", { text: p.shown })]),
          ...(p.sample.length ? [el("pre", { class: "watch-text", text: p.sample.join("\n") })] : []));
      } catch (e) { out.replaceChildren(el("p", { class: "run-note bad", text: why(e) })); }
      preview.disabled = false;
    };
    form.onsubmit = async (e) => {
      e.preventDefault();
      const host = hostOf(f("url").value);
      if (!f("approve").checked) { toast(`Tick the box to let llm-app read ${host || "this site"}`, "error", 4000); return; }
      const body = { name: f("name").value.trim(), url: f("url").value.trim(), approved_host: host, rule: rule(),
        every_minutes: parseInt(f("every").value, 10), notify: f("notify").checked };
      const save = form.querySelector("[type=submit]");
      save.disabled = true;
      try {
        const saved = view.id ? await call.update(view.id, body) : await call.create(body);
        toast(view.id ? "Watch saved" : "Watching this page", "success", 1800);
        await show({ name: "watch", id: saved.id });
      } catch (err) { save.disabled = false; toast(why(err), "error", 6000); }
    };
    detail.replaceChildren(head(view.id ? "Edit watch" : "Watch a page",
      "Code compares the page, so alerts quote it exactly. No model is involved."), form);
    f("name").focus();
  }

  return { list: call.list, row, renderWatch, renderForm };
}
