// Apps: every app in ~/apps/fleet.yaml with its health and card (GET /v1/hub).
// Read-only. Titles have already passed each app's privacy rule on the server.
import { safeUrl } from "./routines.js";

const LEVEL = { alert: "Needs attention", warn: "Warning", info: "OK" };

export function appState(a) {
  if (a.up === false) return { cls: "alert", text: "Down" };
  if (a.level === "alert") return { cls: "alert", text: LEVEL.alert };
  if (a.level === "warn") return { cls: "warn", text: a.problem && !a.items.some((i) => i.severity !== "info") ? a.problem : LEVEL.warn };
  if (a.up === true || a.items.length) return { cls: "ok", text: LEVEL.info };
  return { cls: "none", text: "No health check" };
}

const dueText = (d) => {
  const t = new Date(`${d}T00:00:00`);
  return Number.isNaN(t.getTime()) ? "" : `due ${t.toLocaleDateString(undefined, { month: "short", day: "numeric" })}`;
};

export function initApps({ $, el, api }) {
  const modal = $("#appsModal"), list = $("#appsList"), stamp = $("#appsStamp"), badge = $("#appsCount");
  let loading = null;

  function card(a) {
    const st = appState(a);
    const ui = safeUrl(a.ui);
    const head = el("div", { class: "app-head" }, [
      el("span", { class: `app-dot ${st.cls}`, title: st.text }),
      el("span", { class: "app-title", text: a.title }),
      el("span", { class: "muted-small", text: st.text }),
      ui ? el("a", { class: "ghost-btn app-open", href: ui, target: "_blank", rel: "noopener noreferrer", text: "Open ↗" }) : null,
    ].filter(Boolean));
    const lines = a.items.map((i) => {
      const link = safeUrl(i.link);
      const title = link ? el("a", { href: link, target: "_blank", rel: "noopener noreferrer", text: i.title }) : el("span", { text: i.title });
      return el("li", { class: `app-item ${i.severity}` }, [title, i.due ? el("span", { class: "muted-small", text: ` · ${dueText(i.due)}` }) : null].filter(Boolean));
    });
    const notes = [];
    if (a.problem && st.text !== a.problem) notes.push(a.problem);
    if (a.withheld) notes.push(`${a.withheld} line${a.withheld === 1 ? "" : "s"} withheld by its privacy rule`);
    return el("article", { class: `app-card ${st.cls}` }, [
      head,
      lines.length ? el("ul", { class: "app-items" }, lines) : null,
      notes.length ? el("p", { class: "muted-small app-note", text: notes.join(" · ") }) : null,
    ].filter(Boolean));
  }

  function render(apps) {
    const order = { alert: 0, warn: 1, ok: 2, none: 3 };
    const sorted = [...apps].sort((x, y) => order[appState(x).cls] - order[appState(y).cls] || x.title.localeCompare(y.title));
    list.replaceChildren(...(sorted.length ? sorted.map(card) : [el("p", { class: "muted", text: "No apps are listed in fleet.yaml." })]));
    stamp.textContent = `Checked ${new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}`;
  }

  function setBadge(apps) {
    const n = apps.filter((a) => ["alert", "warn"].includes(appState(a).cls)).length;
    badge.textContent = String(n);
    badge.hidden = !n;
    badge.title = `${n} app${n === 1 ? "" : "s"} need${n === 1 ? "s" : ""} a look`;
  }

  async function refresh() {
    loading ??= api.json("/v1/hub").then((r) => r.apps || []).finally(() => { loading = null; });
    const apps = await loading;
    setBadge(apps);
    return apps;
  }

  async function load() {
    list.setAttribute("aria-busy", "true");
    if (!list.childElementCount) list.replaceChildren(el("p", { class: "muted", text: "Checking the apps…" }));
    try {
      render(await refresh());
    } catch (e) {
      list.replaceChildren(el("p", { class: "muted", text: `Couldn't load the app list: ${e.message}` }));
    } finally {
      list.removeAttribute("aria-busy");
    }
  }

  function open() {
    modal.showModal();
    load();
  }

  $("#appsBtn").onclick = open;
  $("#appsRefresh").onclick = load;
  return { open, refresh: () => refresh().catch(() => []) };
}
