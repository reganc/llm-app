// Actions: what the bot or an MCP client proposed, and what became of it (GET /v1/actions).
// Reject only. Approving is a tap on the Telegram card: MCP clients hold this page's API key,
// so an Approve button here would let a client approve its own proposal.

const LABEL = {
  pending: "Waiting for you", confirming: "Waiting for your second tap",
  approved: "Running…", running: "Running…", done: "Done", failed: "Failed",
  rejected: "Rejected", expired: "Expired", undelivered: "Card not delivered",
};
const OPEN = new Set(["pending", "confirming"]);

export function actionState(a) {
  const cls = OPEN.has(a.status) ? "warn" : a.status === "failed" ? "alert"
    : a.status === "done" ? "ok" : a.status === "approved" || a.status === "running" ? "ok" : "none";
  return { cls, text: LABEL[a.status] || a.status, open: OPEN.has(a.status) };
}

export function sourceText(a) {
  if (a.source === "mcp") return `MCP client “${a.client || "unnamed"}” (what it read before is unknown)`;
  return `Telegram chat${a.after_reading ? ", after reading web pages or your library" : ""}`;
}

const clock = (iso) => {
  const t = new Date(iso);
  return Number.isNaN(t.getTime()) ? "" : t.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
};

export function initActions({ $, el, api, toast }) {
  const modal = $("#actionsModal"), list = $("#actionsList"), badge = $("#actionsCount");
  let loading = null, timer = null;

  async function reject(a, button) {
    button.disabled = true;
    try {
      await api.json(`/v1/actions/${encodeURIComponent(a.id)}/reject`, { method: "POST" });
      toast?.("Rejected. Nothing was changed.");
    } catch (e) {
      toast?.(`Couldn't reject: ${e.message}`);
    }
    await load();
  }

  function card(a) {
    const st = actionState(a);
    const head = el("div", { class: "app-head" }, [
      el("span", { class: `app-dot ${st.cls}`, title: st.text }),
      el("span", { class: "app-title", text: a.title }),
      a.risk === "high" ? el("span", { class: "action-risk", text: "high risk" }) : null,
      el("span", { class: "muted-small", text: st.text }),
    ].filter(Boolean));
    const body = [head, el("ul", { class: "app-items" }, (a.details || []).map((d) => el("li", { class: "app-item", text: d })))];
    if (a.result) body.push(el("p", { class: "action-result", text: a.result }));
    const meta = [`From ${sourceText(a)}`, `proposed ${clock(a.created_at)}`];
    if (st.open) meta.push(`expires ${clock(a.expires_at)}`);
    body.push(el("p", { class: "muted-small app-note", text: `Why (the model's words): “${a.reason || "no reason given"}”` }));
    body.push(el("p", { class: "muted-small app-note", text: meta.join(" · ") }));
    if (st.open) {
      const button = el("button", { type: "button", class: "ghost-btn", text: "Reject" });
      button.onclick = () => reject(a, button);
      body.push(el("div", { class: "action-foot" }, [
        el("span", { class: "muted-small", text: "Approve it on your phone, on its Telegram card." }), button]));
    }
    return el("article", { class: `app-card ${st.cls}` }, body);
  }

  function render(rows) {
    list.replaceChildren(...(rows.length ? rows.map(card) : [el("p", { class: "muted",
      text: "Nothing has been proposed. Ask the bot in Telegram, or an MCP client, to run, pause or restart something." })]));
  }

  function setBadge(waiting) {
    badge.textContent = String(waiting);
    badge.hidden = !waiting;
    badge.title = `${waiting} action${waiting === 1 ? "" : "s"} waiting for your approval`;
  }

  async function refresh() {
    timer ??= setInterval(() => { refresh().catch(() => {}); }, 60_000);   // keeps the badge current
    loading ??= api.json("/v1/actions").finally(() => { loading = null; });
    const data = await loading;
    setBadge(data.waiting || 0);
    return data.actions || [];
  }

  async function load() {
    list.setAttribute("aria-busy", "true");
    if (!list.childElementCount) list.replaceChildren(el("p", { class: "muted", text: "Loading…" }));
    try {
      render(await refresh());
    } catch (e) {
      list.replaceChildren(el("p", { class: "muted", text: `Couldn't load actions: ${e.message}` }));
    } finally {
      list.removeAttribute("aria-busy");
    }
  }

  function open() {
    modal.showModal();
    load();
  }

  $("#actionsBtn").onclick = open;
  $("#actionsRefresh").onclick = load;
  return { open, refresh: () => refresh().catch(() => []) };
}
