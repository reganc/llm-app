// Local AI — single-page app
// No build step, no framework. Vanilla ES modules.
import { escHtml, renderMd } from "./md.js";
import { routineRequest } from "./routine-phrases.js";
import { initRoutines } from "./routines.js";
import { initSkills } from "./skills.js";

// ─────────────────────── State ───────────────────────
const State = {
  apiKey: localStorage.getItem("apiKey") || "",
  models: [],
  prompts: {},
  settings: {},
  conversations: [],          // [{id, name, updated_at, turn_count}]
  activeConvId: null,         // null = quick (ephemeral) chat
  thread: [],                 // current quick-chat in-memory transcript
  streaming: false,
  searchOn: true,            // default ON — visual matches SEARCH_ALWAYS=true
  sidebarCollapsed: false,
  sourcesOpen: false,
  lastSearch: null,          // {query, source, retrieved_at, results: [...]}
};

// ─────────────────────── Helpers ───────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const el = (tag, opts = {}, children = []) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(opts)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function")
      node.addEventListener(k.slice(2).toLowerCase(), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return node;
};

// ─────────────────────── Toast ───────────────────────
function toast(msg, type = "info", ms = 3500) {
  const host = $("#toastHost");
  if (!host) return;
  const node = el("div", { class: `toast ${type}` }, [msg]);
  host.appendChild(node);
  // trigger transition
  requestAnimationFrame(() => node.classList.add("show"));
  setTimeout(() => {
    node.classList.remove("show");
    setTimeout(() => node.remove(), 250);
  }, ms);
}

function toastAction(msg, actionLabel, onAction, type = "info", ms = 7000) {
  const host = $("#toastHost");
  if (!host) return;
  const dismiss = () => {
    node.classList.remove("show");
    setTimeout(() => node.remove(), 250);
  };
  const btn = el("button", {
    class: "toast-action",
    type: "button",
    onclick: () => { onAction(); dismiss(); },
  }, [actionLabel]);
  const node = el("div", { class: `toast ${type} with-action` }, [
    el("span", { class: "toast-msg" }, [msg]), btn,
  ]);
  host.appendChild(node);
  requestAnimationFrame(() => node.classList.add("show"));
  setTimeout(dismiss, ms);
}

const fmtDate = (iso) => {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  const now = new Date();
  const diff = (now - d) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return d.toLocaleDateString();
};

const groupConvs = (convs) => {
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const yest = new Date(today.getTime() - 86400000);
  const week = new Date(today.getTime() - 7 * 86400000);
  const groups = { Today: [], Yesterday: [], "This week": [], Older: [] };
  for (const c of convs) {
    const d = new Date(c.updated_at);
    if (d >= today) groups.Today.push(c);
    else if (d >= yest) groups.Yesterday.push(c);
    else if (d >= week) groups["This week"].push(c);
    else groups.Older.push(c);
  }
  return groups;
};

// ─────────────────────── API client ───────────────────────
const headers = (extra = {}) => ({
  "Authorization": `Bearer ${State.apiKey}`,
  ...extra,
});

const api = {
  async json(path, init = {}) {
    const r = await fetch(path, {
      ...init,
      headers: { "Content-Type": "application/json", ...headers(init.headers) },
    });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${(await r.text()).slice(0, 200)}`);
    return r.json();
  },
  async raw(path, init = {}) {
    return fetch(path, { ...init, headers: { ...headers(init.headers || {}) } });
  },
  health: () => fetch("/health").then(r => r.json()),
  models: () => api.json("/v1/models"),
  prompts: () => api.json("/v1/system-prompts"),
  settings: () => api.json("/v1/settings"),
  patchSettings: (body) => api.json("/v1/settings", { method: "PATCH", body: JSON.stringify(body) }),
  listConvs: () => api.json("/v1/conversations"),
  getConv: (id) => api.json(`/v1/conversations/${id}`),
  createConv: (body) => api.json("/v1/conversations", { method: "POST", body: JSON.stringify(body) }),
  deleteConv: (id) => api.json(`/v1/conversations/${id}`, { method: "DELETE" }),
  memStats: () => api.json("/v1/memory/stats"),
  memSources: (params = {}) => {
    const qs = new URLSearchParams(params).toString();
    return api.json(`/v1/memory/sources${qs ? "?" + qs : ""}`);
  },
  memSource: (sid) => api.json(`/v1/memory/source/${encodeURIComponent(sid)}`),
  memDelete: (sid) => api.json(`/v1/memory/source/${encodeURIComponent(sid)}`, { method: "DELETE" }),
  auxList: () => api.json("/v1/search/auxiliary-sites"),
  auxAdd: (url) => api.json("/v1/search/auxiliary-sites", { method: "POST", body: JSON.stringify({ url }) }),
  auxRemove: (url) => api.json(`/v1/search/auxiliary-sites?url=${encodeURIComponent(url)}`, { method: "DELETE" }),
  tgStatus: () => api.json("/v1/notify/telegram/status"),
  tgChats: () => api.json("/v1/notify/telegram/chats"),
  tgChoose: (id) => api.json("/v1/notify/telegram/chat", { method: "POST", body: JSON.stringify({ chat_id: id }) }),
  tgOptions: (b) => api.json("/v1/notify/telegram/options", { method: "POST", body: JSON.stringify(b) }),
  briefing: () => api.json("/v1/briefing"),
  briefingSet: (b) => api.json("/v1/briefing", { method: "PATCH", body: JSON.stringify(b) }),
  briefingPreview: () => api.json("/v1/briefing/preview", { method: "POST", body: "{}" }),
  briefingSend: () => api.json("/v1/briefing/send", { method: "POST", body: "{}" }),
  tgTest: () => api.json("/v1/notify/telegram/test", { method: "POST", body: "{}" }),
  xStatus: () => api.json("/v1/search/x/status"),
  distillStatus: () => api.json("/v1/distill/status"),
  distillRun: () => api.json("/v1/distill/run", { method: "POST", body: "{}" }),
  ingestUrl: (url, name) => api.json("/v1/ingest/url/conversation",
    { method: "POST", body: JSON.stringify({ url, conversation_name: name }) }),
  ingestSite: (body) => api.json("/v1/ingest/site",
    { method: "POST", body: JSON.stringify(body) }),
  ingestSiteStatus: (jobId) => api.json(`/v1/ingest/site/${encodeURIComponent(jobId)}`),
  ingestDoc: (file) => {
    const fd = new FormData();
    fd.append("file", file);
    return api.raw("/v1/ingest/document/conversation", { method: "POST", body: fd })
      .then(async r => { if (!r.ok) throw new Error(await r.text()); return r.json(); });
  },
  // Memory-only ingestion (no conversation created).
  memIngestFile: (file, title) => {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("title", title || file.name);
    fd.append("source_type", "document");
    return api.raw("/v1/memory/ingest", { method: "POST", body: fd })
      .then(async r => { if (!r.ok) throw new Error(await r.text()); return r.json(); });
  },
};

// ─────────────────────── Rendering ───────────────────────
function renderConvList() {
  const list = $("#convList");
  list.innerHTML = "";
  const filter = $("#convFilter").value.toLowerCase();
  const filtered = State.conversations.filter(c => !filter || c.name.toLowerCase().includes(filter));
  const groups = groupConvs(filtered);
  for (const [name, convs] of Object.entries(groups)) {
    if (!convs.length) continue;
    list.appendChild(el("div", { class: "conv-group", text: name }));
    for (const c of convs) {
      const item = el("button", {
        class: "conv-item" + (c.id === State.activeConvId ? " active" : ""),
        title: c.name,
        onclick: () => loadConversation(c.id),
      }, [c.name]);
      const del = el("span", {
        class: "conv-del", title: "Delete",
        onclick: (e) => { e.stopPropagation(); deleteConv(c.id); },
      }, ["×"]);
      item.appendChild(del);
      list.appendChild(item);
    }
  }
}

function renderThread() {
  const thread = $("#thread");
  const nearBottom = _isNearBottom(thread);
  thread.innerHTML = "";
  const empty = $("#empty");
  if (!State.thread.length) {
    empty.style.display = "flex";
    return;
  }
  empty.style.display = "none";
  for (const m of State.thread) {
    if (m.role === "system") continue;
    thread.appendChild(renderMessage(m));
  }
  if (nearBottom) thread.scrollTop = thread.scrollHeight;
}

function _isNearBottom(el, threshold = 120) {
  return (el.scrollHeight - el.scrollTop - el.clientHeight) < threshold;
}

// In-place paint of just the streaming message's content node, rAF-batched.
// Avoids the full renderThread rebuild per-token that caused the visible flicker.
let _streamPaintPending = false;
function paintStreamingDelta(assistant) {
  if (_streamPaintPending) return;
  _streamPaintPending = true;
  requestAnimationFrame(() => {
    _streamPaintPending = false;
    const node = assistant && assistant._contentNode;
    if (!node) return;
    node.innerHTML = renderMd(assistant.content || "") + '<span class="cursor"></span>';
    const thread = $("#thread");
    if (_isNearBottom(thread)) thread.scrollTop = thread.scrollHeight;
  });
}

function renderMessage(m) {
  const avatar = el("div", { class: "msg-avatar", text: m.role === "user" ? "U" : "AI" });
  const meta = el("div", { class: "msg-meta" });
  meta.appendChild(el("span", { text: m.role === "user" ? "You" : "Assistant" }));
  if (m.web_search) {
    const sources = m.web_search.source || "";
    const signals = m.web_search.intent_signals || [];
    const sigText = signals.length ? ` (${signals.slice(0, 3).join(", ")})` : "";
    const xCount = (m.web_search.results || []).filter(r => (r.engine || "").includes("x_twitter") || (r.url || "").includes("x.com/")).length;
    const webCount = (m.web_search.results || []).length - xCount;
    const openRailWith = () => openSourcesWith(m.web_search);
    if (sources.includes("x_twitter")) {
      const b = el("button", {
        class: "badge x clickable",
        title: signals.length
          ? `Triggered by: ${signals.join(", ")}\nClick to view fetched X posts`
          : "Click to view fetched X posts",
        text: `X · ${xCount || m.web_search.results?.length || 0} posts${sigText}`,
        onClick: openRailWith,
      });
      meta.appendChild(b);
    }
    if (sources.includes("searxng") || sources.includes("duckduckgo")) {
      const b = el("button", {
        class: "badge search clickable",
        title: signals.length
          ? `Triggered by: ${signals.join(", ")}\nClick to view fetched web sources`
          : "Click to view fetched web sources",
        text: `Web · ${webCount || m.web_search.results?.length || 0} sources${sigText}`,
        onClick: openRailWith,
      });
      meta.appendChild(b);
    }
  }
  if (m.memory_used) {
    meta.appendChild(el("span", { class: "badge memory", text: `Memory · ${m.memory_used} chunks` }));
  }
  const content = el("div", { class: "msg-content" });
  content.innerHTML = renderMd(m.content || "") + (m.streaming ? '<span class="cursor"></span>' : "");
  if (m.streaming) m._contentNode = content;
  const children = [meta, content];
  if (m.role === "assistant" && !m.streaming && m.message_id) {
    children.push(renderFeedback(m));
  }
  const body = el("div", { class: "msg-body" }, children);
  return el("div", { class: `msg ${m.role}` }, [avatar, body]);
}

function renderFeedback(m) {
  const cur = m.feedback?.rating || null;
  const wrap = el("div", { class: "msg-feedback" });
  const make = (rating, label, icon) => {
    const active = cur === rating;
    const btn = el("button", {
      class: `fb-btn${active ? " active" : ""}`,
      title: active ? `Remove ${label}` : label,
      "aria-pressed": String(active),
      onClick: async () => {
        const next = active ? null : rating;
        try {
          await sendFeedback(m, next);
          m.feedback = next ? { rating: next } : null;
          renderThread();
        } catch (e) {
          toast(`Feedback failed: ${e.message}`, "error", 4000);
        }
      },
    });
    btn.innerHTML = icon;
    return btn;
  };
  const up = `<svg viewBox="0 0 16 16" width="13" height="13"><path d="M2 7.5h2.5V14H2zM5.5 14h6.2c.7 0 1.3-.4 1.5-1.1l1.6-5c.3-.9-.4-1.8-1.5-1.8h-3.7l.6-2.6c.2-.8-.4-1.5-1.2-1.5-.5 0-1 .3-1.2.8L5.5 7.4z" stroke="currentColor" stroke-width="1.2" fill="none" stroke-linejoin="round"/></svg>`;
  const down = `<svg viewBox="0 0 16 16" width="13" height="13"><path d="M2 8.5h2.5V2H2zM5.5 2h6.2c.7 0 1.3.4 1.5 1.1l1.6 5c.3.9-.4 1.8-1.5 1.8h-3.7l.6 2.6c.2.8-.4 1.5-1.2 1.5-.5 0-1-.3-1.2-.8L5.5 8.6z" stroke="currentColor" stroke-width="1.2" fill="none" stroke-linejoin="round"/></svg>`;
  wrap.appendChild(make("up", "Helpful", up));
  wrap.appendChild(make("down", "Not helpful", down));
  return wrap;
}

async function sendFeedback(m, rating) {
  const body = { message_id: m.message_id };
  if (State.activeConvId) body.conversation_id = State.activeConvId;
  if (rating) {
    body.rating = rating;
    await api.json("/v1/feedback", { method: "POST", body: JSON.stringify(body) });
  } else {
    await api.json("/v1/feedback", { method: "DELETE", body: JSON.stringify(body) });
  }
}

function setStreaming(on, label = "") {
  State.streaming = on;
  $("#streamStatus").textContent = on ? (label || "Generating…") : "";
  $("#sendBtn").disabled = on || !$("#composerInput").value.trim();
}

// ─────────────────────── Conversation ops ───────────────────────
async function refreshConvs() {
  try {
    const r = await api.listConvs();
    State.conversations = r.conversations;
    renderConvList();
  } catch (e) { console.error("listConvs", e); }
}

async function loadConversation(id) {
  try {
    const conv = await api.getConv(id);
    State.activeConvId = id;
    State.thread = conv.messages.map(m => ({
      role: m.role,
      content: m.content,
      message_id: m.message_id,
      feedback: m.feedback || null,
    }));
    $("#convTitle").textContent = conv.name;
    renderConvList();
    renderThread();
  } catch (e) { alert("Failed to load conversation: " + e.message); }
}

function newQuickChat() {
  State.activeConvId = null;
  State.thread = [];
  $("#convTitle").textContent = "Quick chat";
  renderConvList();
  renderThread();
  $("#composerInput").focus();
}

async function deleteConv(id) {
  if (!confirm("Delete this conversation?")) return;
  await api.deleteConv(id);
  if (State.activeConvId === id) newQuickChat();
  await refreshConvs();
}

// ─────────────────────── Send message (streaming) ───────────────────────
async function sendMessage(text, { asMessage = false } = {}) {
  if (!text.trim() || State.streaming) return;
  const routine = asMessage ? null : routineRequest(text);
  if (routine) {
    $("#composerInput").value = "";
    autoresize($("#composerInput"));
    return Routines.openDraft(routine.text,
      routine.explicit ? null : () => sendMessage(text, { asMessage: true }));
  }
  const userMsg = { role: "user", content: text };
  State.thread.push(userMsg);
  const assistant = { role: "assistant", content: "", streaming: true };
  State.thread.push(assistant);
  renderThread();
  $("#composerInput").value = "";
  autoresize($("#composerInput"));
  setStreaming(true, hintFor(text));

  const isUrlAlone = /^https?:\/\/\S+$/.test(text.trim());
  // URL alone → treat as ingest into a new conversation
  if (isUrlAlone && !State.activeConvId) {
    return ingestUrlAndOpen(text.trim(), assistant);
  }

  try {
    if (State.activeConvId) {
      await streamConversation(text, assistant);
    } else {
      await streamQuickChat(text, assistant);
    }
  } catch (e) {
    assistant.content += `\n\n_Error: ${e.message}_`;
  } finally {
    assistant.streaming = false;
    setStreaming(false);
    renderThread();
    if (State.activeConvId) refreshConvs();
  }
}

function hintFor(text) {
  const t = text.trim().toLowerCase();
  if (t.startsWith("/x ")) return "Searching X…";
  if (t.startsWith("/search ")) return "Searching the web…";
  if (State.searchOn) return "Searching the web…";
  return "Generating…";
}

async function streamQuickChat(userText, assistant) {
  const messages = State.thread
    .filter(m => m.role !== "system" && !(m === assistant))
    .map(({ role, content }) => ({ role, content }));
  // Search pill is now a hint to the backend's auto-search heuristic, not a force.
  // Use explicit "/search ..." or "/x ..." commands when you want to force a web/X query.
  await streamSSE("/v1/chat/completions", {
    model: State.settings.default_model,
    messages,
    stream: true,
    system_prompt_key: State.settings.default_system_prompt_key,
  }, assistant);
}

async function streamConversation(userText, assistant) {
  await streamSSE(`/v1/conversations/${State.activeConvId}/messages`, {
    content: userText,
    model: State.settings.default_model,
    stream: true,
  }, assistant);
}

async function streamSSE(url, body, assistant) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers() },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 200)}`);
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let currentEvent = null;       // SSE event name (default = "message")
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    // SSE messages are separated by blank lines; process line-by-line
    const lines = buf.split("\n");
    buf = lines.pop();
    for (const line of lines) {
      if (line === "") { currentEvent = null; continue; }
      if (line.startsWith("event:")) { currentEvent = line.slice(6).trim(); continue; }
      if (!line.startsWith("data:")) continue;
      const data = line.slice(5).trim();
      if (!data || data === "[DONE]") continue;
      if (currentEvent === "llm.web_search") {
        try {
          const summary = JSON.parse(data);
          assistant.web_search = summary;
          State.lastSearch = summary;
          renderLastSearch();
        } catch {}
        continue;
      }
      if (currentEvent === "llm.message_id") {
        try {
          const obj = JSON.parse(data);
          if (obj.id) assistant.message_id = obj.id;
        } catch {}
        continue;
      }
      if (currentEvent === "llm.replace") {
        // Server corrected the streamed answer (e.g. it cited a source it
        // was never shown): swap in the fixed text.
        try {
          const obj = JSON.parse(data);
          if (typeof obj.content === "string") {
            assistant.content = obj.content;
            renderThread();
          }
        } catch {}
        continue;
      }
      if (currentEvent === "llm.memory") {
        try {
          const summary = JSON.parse(data);
          assistant.memory = summary;
          assistant.memory_used = summary.used;
          renderThread();
        } catch {}
        continue;
      }
      try {
        const obj = JSON.parse(data);
        const delta = obj.choices?.[0]?.delta?.content || "";
        if (delta) {
          assistant.content += delta;
          paintStreamingDelta(assistant);
        }
      } catch {}
    }
  }
}

async function ingestUrlAndOpen(url, assistant) {
  try {
    const r = await api.ingestUrl(url, null);
    if (assistant) {
      assistant.content = `Loaded **${r.name}** into a new conversation.\n\nAsk follow-up questions in the chat.`;
      assistant.streaming = false;
      renderThread();
      setStreaming(false);
    }
    toast(`Saved "${r.name}" to memory`, "success");
    await refreshConvs();
    await refreshHealth();
    await loadConversation(r.conversation_id);
  } catch (e) {
    if (assistant) {
      assistant.content = `_Failed to ingest URL: ${e.message}_`;
      assistant.streaming = false;
      renderThread();
      setStreaming(false);
    }
    toast(`URL ingestion failed: ${e.message}`, "error", 5000);
    throw e;
  }
}

async function ingestFile(file) {
  setStreaming(true, `Ingesting ${file.name}…`);
  try {
    const r = await api.ingestDoc(file);
    toast(`Ingested "${file.name}" — saved to memory`, "success");
    await refreshConvs();
    await refreshHealth();
    await loadConversation(r.conversation_id);
  } catch (e) {
    toast(`Ingest failed: ${e.message}`, "error", 5000);
  } finally {
    setStreaming(false);
  }
}

// ─────────────────────── URL ingest dialog ───────────────────────
function openUrlIngest(prefill = "") {
  const dlg = $("#urlIngestModal");
  $("#urlIngestInput").value = prefill;
  $("#urlIngestName").value = "";
  dlg.showModal();
  setTimeout(() => $("#urlIngestInput").focus(), 30);
}

async function submitUrlIngest(e) {
  e.preventDefault();
  const url = $("#urlIngestInput").value.trim();
  if (!url) return;
  const name = $("#urlIngestName").value.trim() || null;
  $("#urlIngestModal").close();
  setStreaming(true, "Reading page…");
  try {
    const r = await api.ingestUrl(url, name);
    toast(`Saved "${r.name}" to memory`, "success");
    await refreshConvs();
    await refreshHealth();
    await loadConversation(r.conversation_id);
  } catch (err) {
    toast(`Could not read URL: ${err.message}`, "error", 5000);
  } finally {
    setStreaming(false);
  }
}

// ─────────────────────── Site crawl dialog ───────────────────────
function openSiteIngest(prefill = "") {
  const dlg = $("#siteIngestModal");
  $("#siteIngestUrl").value = prefill;
  $("#siteIngestPages").value = "25";
  $("#siteIngestDepth").value = "2";
  $("#siteIngestLocal").checked = false;
  const progress = $("#siteIngestProgress");
  progress.hidden = true;
  progress.textContent = "";
  $("#siteIngestSubmit").disabled = false;
  dlg.showModal();
  setTimeout(() => $("#siteIngestUrl").focus(), 30);
}

async function submitSiteIngest(e) {
  e.preventDefault();
  const url = $("#siteIngestUrl").value.trim();
  if (!url) return;
  const max_pages = clampInt($("#siteIngestPages").value, 1, 500, 25);
  const max_depth = clampInt($("#siteIngestDepth").value, 0, 6, 2);
  const force_local = $("#siteIngestLocal").checked;
  const progress = $("#siteIngestProgress");
  const submit = $("#siteIngestSubmit");
  submit.disabled = true;
  progress.hidden = false;
  progress.textContent = "Starting crawl…";
  try {
    const job = await api.ingestSite({ url, max_pages, max_depth, force_local });
    toast(`Crawl started (${job.strategy}) — ${max_pages} pages max`, "info");
    await pollSiteIngest(job.job_id);
  } catch (err) {
    progress.textContent = `Failed to start: ${err.message}`;
    toast(`Crawl failed: ${err.message}`, "error", 5000);
  } finally {
    submit.disabled = false;
  }
}

function clampInt(raw, min, max, fallback) {
  const n = parseInt(raw, 10);
  if (Number.isNaN(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

async function pollSiteIngest(jobId) {
  const progress = $("#siteIngestProgress");
  const dlg = $("#siteIngestModal");
  const deadline = Date.now() + 1000 * 60 * 20; // 20 minutes
  let consecutive404s = 0;
  while (Date.now() < deadline) {
    let status;
    try {
      status = await api.ingestSiteStatus(jobId);
      consecutive404s = 0;
    } catch (err) {
      // Tolerate transient 404s (DB persistence is async + best-effort
      // during early job creation). Bail with a clear message after a
      // few consecutive misses — the job genuinely went missing then.
      if (/\b404\b/.test(err.message)) {
        consecutive404s += 1;
        if (consecutive404s < 3) {
          await new Promise(r => setTimeout(r, 2500));
          continue;
        }
        progress.textContent =
          "Crawl job no longer tracked — the API likely restarted mid-crawl. " +
          "Re-run if you didn't see a completion toast.";
        toast("Crawl job vanished — try again", "error", 6000);
        return;
      }
      progress.textContent = `Polling error: ${err.message}`;
      return;
    }
    progress.textContent =
      `Status: ${status.status} · ${status.pages_crawled}/${status.max_pages} pages crawled · ` +
      `${status.pages_ingested} ingested · ${status.chunks_stored} chunks · ` +
      `${status.elapsed_seconds}s`;
    if (status.status === "interrupted") {
      const errs = status.errors || [];
      progress.innerHTML =
        `<strong>Interrupted:</strong> ${status.pages_ingested} pages, ` +
        `${status.chunks_stored} chunks before the API restarted.<br>` +
        errs.map(e => `<span class="muted-small">⚠ ${escHtml(e)}</span>`).join("<br>");
      toast("Crawl interrupted by API restart — re-run to continue", "error", 6000);
      await refreshHealth();
      return;
    }
    if (status.status === "completed" || status.status === "failed") {
      const errs = status.errors || [];
      const ok = status.pages_ingested > 0;
      // Always show the last error line in the modal — that is where the
      // actionable diagnostic lives (e.g. "HTTP 403", "JS-rendered SPA").
      if (errs.length) {
        progress.innerHTML =
          `<strong>${status.status === "completed" ? "Done" : "Failed"}:</strong> ` +
          `${status.pages_ingested} pages, ${status.chunks_stored} chunks.<br>` +
          errs.map(e => `<span class="muted-small">⚠ ${escHtml(e)}</span>`).join("<br>");
      }
      if (status.status === "completed" && ok) {
        const summary =
          `Crawl finished — ${status.pages_ingested} pages, ${status.chunks_stored} chunks` +
          (errs.length ? ` (${errs.length} warning${errs.length > 1 ? "s" : ""})` : "");
        toastAction(summary, "View in Library", openLibrary, "success", 8000);
        // Auto-refresh Library if it's already open.
        if ($("#libraryModal").open) refreshLibrary().catch(() => {});
      } else if (status.status === "completed") {
        // Completed but ingested nothing — most common case the user is hitting.
        toast(
          `Crawl finished but ingested 0 pages — see modal for diagnostics`,
          "error", 8000,
        );
      } else {
        toast(`Crawl failed${errs.length ? ` (${errs.length} error${errs.length > 1 ? "s" : ""})` : ""}`,
          "error", 6000);
      }
      await refreshHealth();
      if (status.status === "completed" && ok && !errs.length && dlg.open) {
        setTimeout(() => dlg.open && dlg.close(), 1200);
      }
      return;
    }
    await new Promise(r => setTimeout(r, 2500));
  }
  progress.textContent = "Timed out waiting for crawl to finish — it may still be running.";
}

// ─────────────────────── Composer behavior ───────────────────────
function autoresize(ta) {
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 240) + "px";
}

function bindComposer() {
  const ta = $("#composerInput");
  const form = $("#composer");
  ta.addEventListener("input", () => {
    autoresize(ta);
    $("#sendBtn").disabled = State.streaming || !ta.value.trim();
  });
  ta.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      form.requestSubmit();
    } else if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    sendMessage(ta.value);
  });
  $("#attachBtn").addEventListener("click", () => $("#fileInput").click());
  $("#addUrlBtn").addEventListener("click", () => openUrlIngest());
  $("#crawlSiteBtn").addEventListener("click", () => openSiteIngest());
  $("#fileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) ingestFile(file);
    e.target.value = "";
  });
  $("#urlIngestForm").addEventListener("submit", submitUrlIngest);
  $("#siteIngestForm").addEventListener("submit", submitSiteIngest);
  // Quick-action cards in empty state
  $$(".quick-action").forEach(b => b.addEventListener("click", () => {
    const action = b.dataset.action;
    if (action === "attach") $("#fileInput").click();
    else if (action === "add-url") openUrlIngest();
    else if (action === "crawl-site") openSiteIngest();
  }));
}

// ─────────────────────── Drag & drop ───────────────────────
function bindDnD() {
  const overlay = $("#dropOverlay");
  let counter = 0;
  window.addEventListener("dragenter", (e) => {
    if (!e.dataTransfer?.types.includes("Files")) return;
    counter++; overlay.hidden = false;
  });
  window.addEventListener("dragleave", () => {
    counter = Math.max(0, counter - 1);
    if (counter === 0) overlay.hidden = true;
  });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    counter = 0;
    overlay.hidden = true;
    const file = e.dataTransfer?.files?.[0];
    if (file) ingestFile(file);
  });
}

// ─────────────────────── Settings modal ───────────────────────
async function openSettings() {
  await ensureSettings();
  const sel = $("#settingModel");
  sel.innerHTML = "";
  for (const m of State.models) sel.appendChild(el("option", { value: m.id, text: m.id }));
  sel.value = State.settings.default_model;

  const ps = $("#settingPrompt");
  ps.innerHTML = "";
  for (const [k, p] of Object.entries(State.prompts)) {
    ps.appendChild(el("option", { value: k, text: `${k} — ${p.description}` }));
  }
  ps.value = State.settings.default_system_prompt_key;
  $("#settingPromptDesc").textContent = State.prompts[ps.value]?.preview || "";
  ps.onchange = () => $("#settingPromptDesc").textContent = State.prompts[ps.value]?.preview || "";

  $("#settingSearchAlways").checked = !!State.settings.search_always;
  $("#settingApiKey").value = State.apiKey;

  await refreshXStatus();
  await refreshTelegram();
  await refreshBriefing();
  await refreshDistillStatus();
  $("#distillRunBtn").onclick = runDistillNow;

  $("#settingsModal").showModal();
  $("#settingsModal").addEventListener("close", saveSettings, { once: true });
}

function _fmtRelative(iso) {
  if (!iso) return "never";
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.round(diff / 3600)}h ago`;
  return `${Math.round(diff / 86400)}d ago`;
}

async function refreshDistillStatus() {
  const node = $("#distillStatus");
  if (!node) return;
  try {
    const s = await api.distillStatus();
    const parts = [
      `Last run: ${_fmtRelative(s.last_run_at)}`,
      `${s.total_distilled || 0} distilled`,
      `${s.pending || 0} pending`,
    ];
    if (s.interval_hours > 0) parts.push(`auto every ${s.interval_hours}h`);
    if (s.last_error) parts.push(`last error: ${s.last_error}`);
    node.textContent = parts.join(" · ");
  } catch (e) {
    node.textContent = `status unavailable: ${e.message}`;
  }
}

async function runDistillNow() {
  const btn = $("#distillRunBtn");
  if (!btn || btn.disabled) return;
  btn.disabled = true;
  const original = btn.textContent;
  btn.textContent = "Distilling…";
  try {
    const r = await api.distillRun();
    if (r.distilled > 0) {
      toast(`Distilled ${r.distilled} answer${r.distilled === 1 ? "" : "s"} into the library`, "success");
    } else if (r.skipped > 0) {
      toast(`Nothing to distill (${r.skipped} skipped — too short or unparseable)`, "info");
    } else {
      toast("Nothing to distill — rate some answers with 👍 first", "info");
    }
    await refreshDistillStatus();
    await refreshHealth();
  } catch (e) {
    toast(`Distill failed: ${e.message}`, "error", 5000);
  } finally {
    btn.disabled = false;
    btn.textContent = original;
  }
}

async function saveSettings() {
  const newKey = $("#settingApiKey").value.trim();
  if (newKey && newKey !== State.apiKey) {
    State.apiKey = newKey;
    localStorage.setItem("apiKey", newKey);
  }
  // Send only fields that were loaded and changed. If settings never loaded
  // (e.g. no API key yet — the usual reason to open this dialog) the
  // dropdowns are empty, and PATCHing them once blanked the server's model.
  const loaded = State.models.length && Object.keys(State.prompts).length;
  const patch = {};
  if (loaded) {
    const model = $("#settingModel").value;
    const prompt = $("#settingPrompt").value;
    const searchAlways = $("#settingSearchAlways").checked;
    if (model && model !== State.settings.default_model) patch.default_model = model;
    if (prompt && prompt !== State.settings.default_system_prompt_key) {
      patch.default_system_prompt_key = prompt;
    }
    if (searchAlways !== !!State.settings.search_always) patch.search_always = searchAlways;
  }
  try {
    if (Object.keys(patch).length) await api.patchSettings(patch);
    await refreshSettings();
  } catch (e) { console.warn("save settings:", e); }
}

async function refreshAuxList() {
  try {
    const r = await api.auxList();
    const box = $("#auxRailList");
    if (!box) return;
    box.innerHTML = "";
    if (!r.sites.length) {
      box.appendChild(el("p", { class: "muted",
        text: "Add URLs that should be fetched alongside every web search (e.g. your favourite news sources)." }));
      return;
    }
    for (const s of r.sites) {
      const row = el("div", { class: "aux-row" }, [
        el("a", { class: "url", href: s.url, target: "_blank", rel: "noopener", text: s.url }),
        el("button", {
          class: "rm", type: "button", title: "Remove", text: "×",
          onclick: async (e) => {
            e.preventDefault();
            await api.auxRemove(s.url);
            refreshAuxList();
          },
        }),
      ]);
      box.appendChild(row);
    }
  } catch (e) { console.error("aux list", e); }
}

function openSourcesWith(webSearch) {
  if (webSearch) {
    State.lastSearch = webSearch;
    renderLastSearch();
  }
  State.sourcesOpen = true;
  $("#app").classList.add("with-sources");
  $("#sources").hidden = false;
}

function renderLastSearch() {
  const meta = $("#lastSearchMeta");
  const list = $("#lastSearchList");
  if (!list) return;
  list.innerHTML = "";
  const ls = State.lastSearch;
  if (!ls) {
    meta.textContent = "no search yet";
    list.appendChild(el("p", { class: "muted",
      text: "Sources used in the most recent answer will appear here." }));
    return;
  }
  meta.textContent = `${ls.results.length} sources · ${ls.source} · ${ls.retrieved_at || ""}`;
  if (!ls.results.length) {
    list.appendChild(el("p", { class: "muted", text: "No sources returned." }));
    return;
  }
  for (const r of ls.results) {
    const isX = (r.engine || "").includes("x_twitter");
    const isAux = (r.engine || "").includes("auxiliary");
    const card = el("div", { class: "source-card" }, [
      el("div", { class: "src-title", text: r.title || r.url }),
      el("a", { class: "src-url", href: r.url, target: "_blank", rel: "noopener", text: r.url }),
      el("div", { class: "src-meta",
        text: `${isX ? "X · " : isAux ? "Always-fetch · " : "Web · "}${r.char_count || 0} chars` }),
    ]);
    list.appendChild(card);
  }
}

const apiDetail = (e) => {
  try { return JSON.parse(e.message.slice(e.message.indexOf("{"))).detail || e.message; } catch { return e.message; }
};

async function refreshTelegram() {
  const node = $("#tgStatus"), find = $("#tgFindBtn"), test = $("#tgTestBtn"), list = $("#tgChats");
  list.replaceChildren();
  let s;
  try { s = await api.tgStatus(); } catch { node.textContent = "Status unavailable."; return; }
  node.textContent = s.ready ? "Connected."
    : s.token_set ? "Open your bot in Telegram, press Start, then choose your chat."
    : "Not set up: put TELEGRAM_BOT_TOKEN in llm-app/.env and recreate the api container.";
  find.hidden = !s.token_set;
  find.textContent = s.chat_set ? "Change chat" : "Find my chat";
  test.hidden = !s.ready;
  const options = $("#tgOptions"), chat = $("#tgChatToggle"), library = $("#tgLibraryToggle");
  options.hidden = !s.ready;
  chat.checked = s.chat_enabled;
  library.checked = s.library;
  library.disabled = !s.chat_enabled;
  const set = async (body) => {
    try { await api.tgOptions(body); } catch (e) { toast(apiDetail(e), "error", 5000); }
    await refreshTelegram();
  };
  chat.onchange = () => set({ chat: chat.checked });
  library.onchange = () => set({ library: library.checked });
  test.onclick = async () => {
    test.disabled = true;
    try { await api.tgTest(); toast("Test sent", "success", 2000); }
    catch (e) { toast(apiDetail(e), "error", 6000); }
    test.disabled = false;
  };
  find.onclick = async () => {
    find.disabled = true;
    try {
      const { chats } = await api.tgChats();
      list.replaceChildren(...chats.map(c => el("button", {
        type: "button", class: "ghost-btn",
        text: `Use ${c.name || "chat"}${c.username ? ` (@${c.username})` : ""}`,
        onclick: async () => {
          try { await api.tgChoose(c.id); toast("Telegram chat saved", "success", 2000); await refreshTelegram(); }
          catch (e) { toast(apiDetail(e), "error", 6000); }
        },
      })));
      if (!chats.length) toast("No messages yet. Send your bot any message, then try again.", "info", 5000);
    } catch (e) { toast(apiDetail(e), "error", 6000); }
    find.disabled = false;
  };
}

async function refreshBriefing() {
  const node = $("#briefStatus"), toggle = $("#briefToggle"), time = $("#briefTime"), out = $("#briefPreview");
  let s;
  try { s = await api.briefing(); } catch { node.textContent = "Status unavailable."; return; }
  toggle.checked = s.enabled;
  time.value = s.time;
  const parts = [
    s.calendar_set ? "Calendar connected." : "Calendar not set up: put Google Calendar's “Secret address in iCal format” in llm-app/.env as GOOGLE_CALENDAR_ICS_URL and recreate the api container.",
    s.telegram_ready ? "" : "Telegram isn't connected, so it will only be saved here.",
    s.enabled && s.next_at ? `Next: ${new Date(s.next_at).toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" })}.` : "",
    s.last ? `Last: ${new Date(s.last.at).toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" })}, ${s.last.notified.replace(/\.$/, "")}.` : "",
  ];
  node.textContent = parts.filter(Boolean).join(" ");
  const save = async (body) => {
    try { await api.briefingSet(body); } catch (e) { toast(apiDetail(e), "error", 5000); }
    await refreshBriefing();
  };
  toggle.onchange = () => save({ enabled: toggle.checked, time: time.value });
  time.onchange = () => { if (time.value) save({ time: time.value }); };
  const show = async (btn, call, done) => {
    btn.disabled = true;
    out.hidden = false;
    out.textContent = "Putting it together…";
    try { const r = await call(); out.textContent = r.text; if (done) done(r); }
    catch (e) { out.textContent = apiDetail(e); }
    btn.disabled = false;
  };
  $("#briefPreviewBtn").onclick = (e) => show(e.target, api.briefingPreview);
  $("#briefSendBtn").onclick = (e) => show(e.target, api.briefingSend, async (r) => {
    toast(r.notified === "sent" ? "Briefing sent" : `Saved, ${r.notified}`, r.notified === "sent" ? "success" : "info", 4000);
    await refreshBriefing();
  });
}

async function refreshXStatus() {
  try {
    const s = await api.xStatus();
    $("#xStatus").textContent = s.ready
      ? (s.auth_ok ? "Ready — /x <query> to search."
        : "Set up — the login is checked on the next X search.")
      : s.auth_ok === false ? (s.last_error || "X login has expired.")
      : `Not ready (binary: ${s.binary_found ? "✓" : "✗"}, creds: ${s.creds_found ? "✓" : "✗"}, enabled: ${s.enabled ? "✓" : "✗"}). Run x-cli auth login on host and set X_SEARCH_ENABLED=true.`;
  } catch { $("#xStatus").textContent = "Status unavailable."; }
}

// ─────────────────────── Library ───────────────────────
const Library = {
  query: "",
  type: "",
  selectedId: null,
  total: 0,
  bound: false,
  searchTimer: 0,
  cache: { sources: [], types: [] },
};

function libraryTypeLabel(t) {
  const map = {
    web: "Web", firecrawl: "Web", duckduckgo: "Web",
    youtube_transcript: "YouTube",
    pdf: "PDF", docx: "Word",
    txt: "Text", md: "Markdown", csv: "CSV",
    rtf: "RTF", json: "JSON", html: "HTML",
    document: "Document", manual: "Manual",
    conversation: "Chat",
  };
  return map[t] || (t ? t[0].toUpperCase() + t.slice(1) : "Unknown");
}

function libraryTypeIcon(t) {
  if (["web", "firecrawl", "duckduckgo"].includes(t)) return "🌐";
  if (t === "youtube_transcript") return "▶";
  if (["pdf", "docx", "txt", "md", "csv", "rtf", "html", "json", "xml", "log", "document"].includes(t)) return "📄";
  if (t === "conversation") return "💬";
  return "•";
}

async function openLibrary() {
  bindLibrary();
  Library.selectedId = null;
  $("#libraryDetail").innerHTML = `
    <div class="library-empty">
      <div class="library-empty-mark">📚</div>
      <p>Select a source to read its full content.</p>
    </div>`;
  $("#libraryModal").showModal();
  await refreshLibrary();
}

async function refreshLibrary() {
  const params = {};
  if (Library.query) params.q = Library.query;
  if (Library.type) params.source_type = Library.type;
  let data;
  try {
    data = await api.memSources(params);
  } catch (e) {
    $("#libraryList").innerHTML = "";
    $("#libraryList").appendChild(el("div", { class: "muted", text: "Library unavailable: " + e.message }));
    return;
  }
  Library.cache.sources = data.sources || [];
  Library.cache.types = data.types || [];
  Library.total = data.total || 0;
  renderLibraryStats(data);
  renderLibraryTypes(data.types || []);
  renderLibraryList(data.sources || []);
}

function renderLibraryStats(data) {
  const total = data.total || 0;
  const filtered = (Library.query || Library.type) && data.sources;
  const txt = filtered
    ? `${data.sources.length} of ${total} sources match`
    : `${total} ${total === 1 ? "source" : "sources"}`;
  $("#libraryStats").textContent = txt;
}

function renderLibraryTypes(types) {
  const box = $("#libraryTypes");
  box.innerHTML = "";
  const all = el("button", {
    type: "button",
    class: "type-pill" + (Library.type === "" ? " active" : ""),
    onclick: () => { Library.type = ""; refreshLibrary(); },
  }, ["All"]);
  box.appendChild(all);
  for (const t of types) {
    const pill = el("button", {
      type: "button",
      class: "type-pill" + (Library.type === t ? " active" : ""),
      onclick: () => { Library.type = (Library.type === t ? "" : t); refreshLibrary(); },
    }, [libraryTypeLabel(t)]);
    box.appendChild(pill);
  }
}

function renderLibraryList(sources) {
  const list = $("#libraryList");
  list.innerHTML = "";
  if (!sources.length) {
    list.appendChild(el("div", { class: "library-list-empty" }, [
      Library.query || Library.type
        ? "No sources match this filter."
        : "Library is empty. Add a URL or upload a document to start.",
    ]));
    return;
  }
  for (const s of sources) {
    const row = el("button", {
      type: "button",
      class: "library-row" + (Library.selectedId === s.source_id ? " active" : ""),
      onclick: () => selectLibrarySource(s.source_id),
    }, [
      el("div", { class: "lib-row-icon", text: libraryTypeIcon(s.source_type) }),
      el("div", { class: "lib-row-body" }, [
        el("div", { class: "lib-row-title", title: s.title || s.identifier, text: s.title || s.identifier || "(untitled)" }),
        el("div", { class: "lib-row-meta", text: `${libraryTypeLabel(s.source_type)} · ${s.chunk_count} chunk${s.chunk_count === 1 ? "" : "s"} · ${(s.char_count || 0).toLocaleString()} chars · ${fmtDate(s.timestamp)}` }),
      ]),
    ]);
    list.appendChild(row);
  }
}

async function selectLibrarySource(sid) {
  Library.selectedId = sid;
  // re-render list to highlight
  renderLibraryList(Library.cache.sources);
  const detail = $("#libraryDetail");
  detail.innerHTML = `<div class="library-loading muted">Loading…</div>`;
  let data;
  try {
    data = await api.memSource(sid);
  } catch (e) {
    detail.innerHTML = "";
    detail.appendChild(el("div", { class: "muted", text: "Failed to load: " + e.message }));
    return;
  }
  renderLibraryDetail(data);
}

function renderLibraryDetail(s) {
  const detail = $("#libraryDetail");
  detail.innerHTML = "";
  const isUrl = /^https?:\/\//i.test(s.identifier || "");

  const head = el("div", { class: "library-detail-head" }, [
    el("div", { class: "lib-detail-title", text: s.title || s.identifier || "(untitled)" }),
    el("div", { class: "lib-detail-meta" }, [
      el("span", { class: "type-pill active small", text: libraryTypeLabel(s.source_type) }),
      el("span", { class: "muted-small", text: `${(s.char_count || 0).toLocaleString()} chars` }),
      el("span", { class: "muted-small", text: `${s.chunk_count} chunks` }),
      el("span", { class: "muted-small", text: fmtDate(s.timestamp) }),
    ]),
  ]);
  if (s.identifier) {
    head.appendChild(
      isUrl
        ? el("a", { class: "lib-detail-url", href: s.identifier, target: "_blank", rel: "noopener", text: s.identifier })
        : el("div", { class: "lib-detail-url muted-small", text: s.identifier })
    );
  }

  const actions = el("div", { class: "library-actions" }, [
    el("button", {
      type: "button", class: "ghost-btn",
      onclick: () => libraryAskInChat(s),
    }, ["Ask in new chat"]),
    el("button", {
      type: "button", class: "ghost-btn",
      onclick: () => libraryCopyText(s),
    }, ["Copy text"]),
    isUrl
      ? el("a", { class: "ghost-btn", href: s.identifier, target: "_blank", rel: "noopener", text: "Open original ↗" })
      : el("span", {}, []),
    el("button", {
      type: "button", class: "ghost-btn danger",
      onclick: () => libraryDelete(s),
    }, ["Delete"]),
  ]);

  const content = el("div", { class: "library-content" });
  content.innerHTML = renderMd(s.text || "");

  detail.appendChild(head);
  detail.appendChild(actions);
  detail.appendChild(content);
}

function libraryAskInChat(s) {
  $("#libraryModal").close();
  newQuickChat();
  const ta = $("#composerInput");
  ta.value = `About "${s.title || s.identifier}": `;
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
  autoresize(ta);
  toast(`Memory will be searched for "${s.title || s.identifier}" automatically.`, "info");
}

async function libraryCopyText(s) {
  try {
    await navigator.clipboard.writeText(s.text || "");
    toast("Copied to clipboard", "success", 1500);
  } catch (e) {
    toast("Copy failed: " + e.message, "error");
  }
}

async function libraryDelete(s) {
  if (!confirm(`Delete "${s.title || s.identifier}" from memory? This cannot be undone.`)) return;
  try {
    await api.memDelete(s.source_id);
    toast("Source deleted", "success", 1500);
    Library.selectedId = null;
    $("#libraryDetail").innerHTML = `
      <div class="library-empty">
        <div class="library-empty-mark">📚</div>
        <p>Select a source to read its full content.</p>
      </div>`;
    await refreshLibrary();
    await refreshHealth();
  } catch (e) {
    toast("Delete failed: " + e.message, "error");
  }
}

function bindLibrary() {
  if (Library.bound) return;
  Library.bound = true;
  $("#librarySearch").addEventListener("input", (e) => {
    Library.query = e.target.value;
    clearTimeout(Library.searchTimer);
    Library.searchTimer = setTimeout(refreshLibrary, 200);
  });
  $("#libraryRefreshBtn").addEventListener("click", refreshLibrary);
  $("#libraryAddUrlBtn").addEventListener("click", () => {
    $("#libraryModal").close();
    openUrlIngest();
  });
  $("#libraryCrawlSiteBtn").addEventListener("click", () => {
    $("#libraryModal").close();
    openSiteIngest();
  });
  $("#libraryAddFileBtn").addEventListener("click", () => $("#libraryAddFileInput").click());
  $("#libraryAddFileInput").addEventListener("change", async (e) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
      await api.memIngestFile(file, file.name);
      toast(`Saved "${file.name}" to memory`, "success");
      await refreshLibrary();
      await refreshHealth();
    } catch (err) {
      toast(`Ingest failed: ${err.message}`, "error", 5000);
    }
  });
}

// ─────────────────────── Command palette ───────────────────────
const PALETTE_COMMANDS = [
  { ico: "+", label: "New chat", desc: "⌘N", run: () => newQuickChat() },
  { ico: "📎", label: "Upload document", desc: "PDF, DOCX, TXT, MD, CSV → memory", run: () => $("#fileInput").click() },
  { ico: "🔗", label: "Read a webpage", desc: "URL → memory + chat", run: () => openUrlIngest() },
  { ico: "🕸", label: "Crawl an entire site", desc: "Recursively ingest every page", run: () => openSiteIngest() },
  { ico: "🔍", label: "Search the web", desc: "/search <q>", run: () => prefillComposer("/search ") },
  { ico: "𝕏", label: "Search X / Twitter", desc: "/x <q>", run: () => prefillComposer("/x ") },
  { ico: "📚", label: "Library", desc: "search, read, manage saved sources", run: () => openLibrary() },
  { ico: "⏱", label: "Routines", desc: "scheduled tasks and their results", run: () => Routines.open() },
  { ico: "⏱", label: "New routine from a sentence", desc: "/routine <what and when>", run: () => prefillComposer("/routine ") },
  { ico: "👁", label: "Watch a page", desc: "alert me when it changes", run: () => Routines.watchPage() },
  { ico: "⚡", label: "Skills", desc: "/name <message>", run: () => Skills.open() },
  { ico: "⚙", label: "Settings", desc: "model, prompts, key", run: () => openSettings() },
];

function openPalette() {
  const input = $("#paletteInput");
  input.value = "";
  renderPalette("");
  $("#paletteModal").showModal();
  setTimeout(() => input.focus(), 30);
}

function renderPalette(filter) {
  const list = $("#paletteList");
  list.innerHTML = "";
  const items = PALETTE_COMMANDS.filter(c => c.label.toLowerCase().includes(filter.toLowerCase()));
  items.forEach((cmd, i) => {
    const li = el("li", {
      class: i === 0 ? "active" : "",
      onclick: () => { cmd.run(); $("#paletteModal").close(); },
    }, [
      el("span", { class: "ico", text: cmd.ico }),
      el("span", { text: cmd.label }),
      el("span", { class: "desc", text: cmd.desc }),
    ]);
    list.appendChild(li);
  });
}

function bindPalette() {
  $("#paletteInput").addEventListener("input", (e) => renderPalette(e.target.value));
  $("#paletteInput").addEventListener("keydown", (e) => {
    const items = $$("#paletteList li");
    let active = items.findIndex(i => i.classList.contains("active"));
    if (e.key === "ArrowDown") {
      e.preventDefault();
      items[active]?.classList.remove("active");
      items[Math.min(items.length - 1, active + 1)]?.classList.add("active");
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      items[active]?.classList.remove("active");
      items[Math.max(0, active - 1)]?.classList.add("active");
    } else if (e.key === "Enter") {
      e.preventDefault();
      items[active]?.click();
    }
  });
}

function prefillComposer(prefix) {
  const ta = $("#composerInput");
  ta.value = prefix + ta.value;
  ta.focus();
  ta.setSelectionRange(ta.value.length, ta.value.length);
  autoresize(ta);
}

// ─────────────────────── Settings + bootstrap ───────────────────────
async function refreshSettings() {
  try {
    const [s, ms, ps] = await Promise.all([api.settings(), api.models(), api.prompts()]);
    State.settings = s;
    State.models = ms.data;
    State.prompts = ps.prompts;
    $("#modelLabel").textContent = s.default_model;
  } catch (e) { console.error("refreshSettings", e); }
}

async function ensureSettings() {
  if (!State.models.length || !Object.keys(State.prompts).length) await refreshSettings();
}

async function refreshHealth() {
  try {
    const h = await api.health();
    $("#memoryCount").textContent = h.memory?.unique_sources ?? 0;
  } catch {}
}

// ─────────────────────── Top bar wiring ───────────────────────
function bindTopbar() {
  $("#newChatBtn").onclick = newQuickChat;
  $("#commandBtn").onclick = openPalette;
  $("#settingsBtn").onclick = openSettings;
  $("#libraryBtn").onclick = openLibrary;
  $("#convFilter").oninput = renderConvList;
  $("#toggleSidebar").onclick = () => {
    const isMobile = window.matchMedia("(max-width: 720px)").matches;
    const cls = isMobile ? "sidebar-open" : "sidebar-collapsed";
    const nowOn = $("#app").classList.toggle(cls);
    if (!isMobile) State.sidebarCollapsed = nowOn;
  };
  $("#modelPicker").onclick = openSettings;
  $("#searchToggle").onclick = (e) => {
    State.searchOn = !State.searchOn;
    $("#searchToggle").dataset.on = String(State.searchOn);
  };
  $("#sourcesBtn").onclick = () => {
    State.sourcesOpen = !State.sourcesOpen;
    $("#app").classList.toggle("with-sources", State.sourcesOpen);
    $("#sources").hidden = !State.sourcesOpen;
  };
  $("#closeSources").onclick = () => {
    State.sourcesOpen = false;
    $("#app").classList.remove("with-sources");
    $("#sources").hidden = true;
  };
  $("#auxRailForm").onsubmit = async (e) => {
    e.preventDefault();
    const input = $("#auxRailUrl");
    const url = input.value.trim();
    if (!url) return;
    try { await api.auxAdd(url); input.value = ""; refreshAuxList(); }
    catch (err) { alert("Add failed: " + err.message); }
  };
  $$(".suggest-btn").forEach(b => b.onclick = () => {
    sendMessage(b.dataset.q);
  });
  document.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
      e.preventDefault(); openPalette();
    } else if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "n") {
      e.preventDefault(); newQuickChat();
    } else if (e.key === "Escape") {
      // close any open dialogs
      $$("dialog[open]").forEach(d => d.close());
    }
  });
}

// ─────────────────────── API key bootstrap ───────────────────────
async function ensureApiKey() {
  // Try to access /v1/models with current key — if fail, prompt
  try {
    await api.models();
    return true;
  } catch (e) {
    if (!String(e.message).startsWith("401")) throw e;
    return false;
  }
}

function promptApiKey() {
  const dlg = $("#apiKeyModal");
  $("#apiKeyInput").value = State.apiKey || "";
  dlg.showModal();
  $("#apiKeyForm").addEventListener("submit", () => {
    const v = $("#apiKeyInput").value.trim();
    if (!v) return;
    State.apiKey = v;
    localStorage.setItem("apiKey", v);
    dlg.close();
    boot();
  }, { once: true });
}

// ─────────────────────── Dialog dismiss ───────────────────────
// Backdrop click + [data-close] button + Escape all close the active modal.
// Idempotent — safe to call repeatedly.
function bindDialogDismiss() {
  $$("dialog").forEach(dlg => {
    if (dlg.dataset.dismissBound) return;
    dlg.dataset.dismissBound = "1";
    dlg.addEventListener("click", (e) => {
      // Click landed on the backdrop (the dialog itself, not its inner content)
      if (e.target === dlg) dlg.close();
    });
  });
  $$("[data-close]").forEach(b => {
    if (b.dataset.dismissBound) return;
    b.dataset.dismissBound = "1";
    b.addEventListener("click", (e) => {
      e.preventDefault();
      e.stopPropagation();
      const id = b.getAttribute("data-close");
      $("#" + id)?.close();
    });
  });
}

// ─────────────────────── Main ───────────────────────
const Skills = initSkills({ $, el, api, toast });
const Routines = initRoutines({ $, el, api, toast, skills: Skills });

async function boot() {
  bindTopbar();
  bindComposer();
  bindDnD();
  bindPalette();
  bindDialogDismiss();
  if (!State.apiKey) {
    promptApiKey();
    return;
  }
  if (!(await ensureApiKey())) {
    promptApiKey();
    return;
  }
  await Promise.all([refreshSettings(), refreshConvs(), refreshHealth(), refreshAuxList(),
    Routines.refreshUnread()]);
  renderLastSearch();
  newQuickChat();
}

boot();
