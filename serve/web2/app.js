// serve/web2/app.js - the SlenderEngine web app: chats and projects in the sidebar, the chat, settings.
// Talks to this server only (/health, /mcp, /v1/chat/completions). Chats stay in the browser (store.js).
"use strict";
const $ = (id) => document.getElementById(id);
const {icon} = SlenderIcons;
const {markdown, esc} = SlenderMarkdown;
const fmt = (n, d = 0) => (n == null || Number.isNaN(+n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`);

// ------------------------------------------------------------------ browser storage (small keys only)
const ls = {
  get(k, d) { try { const v = localStorage.getItem("slender." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("slender." + k, JSON.stringify(v)); return true; } catch (e) { return false; } },
};

// ------------------------------------------------------------------ settings
const DEFAULTS = {system: "", effort: "high", enterSends: true, card: true, temperature: 1.0, top_p: 0.95, top_k: 20,
                  max: "", seed: "", mcp: true, typing: true, thinkOpen: true, math: true};
let settings = {...DEFAULTS, ...ls.get("settings", {})};
const saveSettings = () => ls.set("settings", settings);
// the classic page's key is taken over once, so a server with a key keeps working after the switch
let apiKey = ls.get("apikey", null);
if (apiKey == null) { try { apiKey = JSON.parse(localStorage.getItem("strata.apikey") || '""') || ""; } catch (e) { apiKey = ""; } }
function headers(json = false) {
  const h = {};
  if (apiKey) h.Authorization = "Bearer " + apiKey;
  if (json) h["Content-Type"] = "application/json";
  return h;
}

// ------------------------------------------------------------------ toasts
function toast(title, text = "", {kind = "", ms = 4000, action = null} = {}) {
  const el = document.createElement("div");
  el.className = `toast${kind ? " toast--" + kind : ""}`;
  el.innerHTML = `<span><b>${esc(title)}</b>${text ? " " + esc(text) : ""}</span>`;
  if (action) {
    const b = document.createElement("button");
    b.type = "button"; b.textContent = action.label;
    b.onclick = () => { action.run(); el.remove(); };
    el.append(b);
  }
  $("toasts").append(el);
  setTimeout(() => el.remove(), ms);
}
async function copyText(text, btn) {
  try { await navigator.clipboard.writeText(text); } catch (e) { toast("Could not copy", "The browser refused the clipboard.", {kind: "error"}); return; }
  if (!btn) return;
  const old = btn.innerHTML;
  btn.innerHTML = icon("check");
  setTimeout(() => { btn.innerHTML = old; }, 1200);
}

// ------------------------------------------------------------------ state
const store = SlenderStore.createStore({onProblem: (k) => toast(k === "memory" ? "Chats are not saved" : "A chat could not be saved",
  k === "memory" ? "This browser keeps no storage for the page (a private window?). Chats last until you close it." : "The browser refused the write (storage full?).",
  {kind: "error", ms: 8000})});
let chats = [];                 // every chat record, messages included
let projects = [];
let chat = null;                // the open chat (not in `chats` until its first message)
let busy = null;                // {controller, msg, chat}
let health = {model: "", max_context: 0, images: false, loaded: null};
let mcpInfo = {servers: [], tools: 0};
let attachments = [];           // {kind: "image"|"file", name, url|text}
let searchQuery = "";
const openProjects = new Set(ls.get("openProjects", []));

const isNew = (c) => !chats.includes(c);
function newChat(projectId = null) {
  return {id: uid(), title: "New chat", projectId, pinned: false, created: Date.now(), updated: Date.now(), messages: []};
}
function titleFrom(text) {
  const t = text.replace(/\s+/g, " ").trim();
  return t ? (t.length > 60 ? t.slice(0, 57).trimEnd() + "…" : t) : "New chat";
}
function persist(c) { if (!isNew(c)) store.putChat(c); }

// ------------------------------------------------------------------ sidebar
function dateGroup(t) {
  const d = new Date(t), now = new Date();
  const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((day(now) - day(d)) / 86400000);
  if (diff <= 0) return "Today";
  if (diff === 1) return "Yesterday";
  if (diff < 7) return "Previous 7 days";
  if (diff < 30) return "Previous 30 days";
  return d.toLocaleDateString(undefined, {month: "long", year: "numeric"});
}
function chatMatches(c, q) {
  if (!q) return true;
  if (c.title.toLowerCase().includes(q)) return true;
  return c.messages.some((m) => (m.text || "").toLowerCase().includes(q));
}
function chatRow(c) {
  const row = document.createElement("div");
  row.className = `row${chat && c.id === chat.id ? " is-active" : ""}`;
  row.tabIndex = 0;
  row.setAttribute("role", "button");
  row.dataset.chat = c.id;
  row.innerHTML = `${c.pinned ? '<span class="pin-dot" title="Pinned"></span>' : ""}<span class="row__title">${esc(c.title)}</span>` +
    `<button class="row__more" type="button" aria-label="Chat options" aria-haspopup="menu" aria-expanded="false">${icon("more", "ico--sm")}</button>`;
  row.onclick = (e) => { if (!e.target.closest(".row__more, input")) openChat(c.id); };
  row.onkeydown = (e) => { if (e.key === "Enter" && e.target === row) openChat(c.id); };
  row.querySelector(".row__more").onclick = (e) => { e.stopPropagation(); chatMenu(c, e.currentTarget); };
  return row;
}
function renderSidebar() {
  const q = searchQuery.trim().toLowerCase();
  const sorted = [...chats].sort((a, b) => b.updated - a.updated);
  // projects
  const pl = $("project-list");
  pl.replaceChildren();
  const ps = [...projects].sort((a, b) => a.created - b.created);
  if (!ps.length && !q) {
    const e = document.createElement("div");
    e.className = "side-empty"; e.textContent = "No projects yet.";
    pl.append(e);
  }
  for (const p of ps) {
    const inside = sorted.filter((c) => c.projectId === p.id && chatMatches(c, q));
    if (q && !inside.length && !p.name.toLowerCase().includes(q)) continue;
    const open = openProjects.has(p.id) || (q && inside.length > 0);
    const row = document.createElement("div");
    row.className = "row row--project";
    row.tabIndex = 0;
    row.setAttribute("role", "button");
    row.setAttribute("aria-expanded", String(!!open));
    row.dataset.project = p.id;
    row.innerHTML = `${icon(open ? "folder-open" : "folder", "ico--sm")}<span class="row__title">${esc(p.name)}</span>` +
      `<button class="row__more" type="button" aria-label="Project options" aria-haspopup="menu" aria-expanded="false">${icon("more", "ico--sm")}</button>`;
    const toggle = () => {
      if (openProjects.has(p.id)) openProjects.delete(p.id); else openProjects.add(p.id);
      ls.set("openProjects", [...openProjects]);
      renderSidebar();
    };
    row.onclick = (e) => { if (!e.target.closest(".row__more, input")) toggle(); };
    row.onkeydown = (e) => { if (e.key === "Enter" && e.target === row) toggle(); };
    row.querySelector(".row__more").onclick = (e) => { e.stopPropagation(); projectMenu(p, e.currentTarget); };
    pl.append(row);
    if (open) {
      const box = document.createElement("div");
      box.className = "project-chats";
      const add = document.createElement("div");
      add.className = "row"; add.tabIndex = 0; add.setAttribute("role", "button");
      add.innerHTML = `${icon("plus", "ico--sm")}<span class="row__title">New chat in ${esc(p.name)}</span>`;
      add.onclick = () => startChat(p.id);
      add.onkeydown = (e) => { if (e.key === "Enter") startChat(p.id); };
      box.append(add, ...inside.map(chatRow));
      pl.append(box);
    }
  }
  // chats outside projects (searching shows every match)
  const cl = $("chat-list");
  cl.replaceChildren();
  const loose = sorted.filter((c) => (q ? true : !c.projectId) && chatMatches(c, q));
  const groups = new Map();
  if (q) groups.set(loose.length ? "Results" : "No chats match", loose);
  else {
    const pinned = loose.filter((c) => c.pinned);
    if (pinned.length) groups.set("Pinned", pinned);
    for (const c of loose.filter((x) => !x.pinned)) {
      const g = dateGroup(c.updated);
      if (!groups.has(g)) groups.set(g, []);
      groups.get(g).push(c);
    }
  }
  if (!q && !loose.length) {
    const h = document.createElement("div");
    h.className = "side-group"; h.textContent = "Chats";
    const e = document.createElement("div");
    e.className = "side-empty"; e.textContent = "Your chats will show up here.";
    cl.append(h, e);
  }
  for (const [name, list] of groups) {
    const h = document.createElement("div");
    h.className = "side-group"; h.textContent = name;
    cl.append(h, ...list.map(chatRow));
  }
}

// a small menu next to a button (chat and project options)
let menuFor = null;
function closeMenu() {
  const m = $("ctx-menu");
  m.hidden = true;
  if (menuFor) { menuFor.setAttribute("aria-expanded", "false"); menuFor = null; }
}
function openMenu(anchor, items) {
  closeMenu();
  const m = $("ctx-menu");
  m.replaceChildren();
  for (const it of items) {
    if (it === "-") { m.append(document.createElement("hr")); continue; }
    if (it.label && !it.run) { const l = document.createElement("div"); l.className = "menu-label"; l.textContent = it.label; m.append(l); continue; }
    const b = document.createElement("button");
    b.type = "button"; b.className = `opt${it.danger ? " opt--danger" : ""}`; b.setAttribute("role", "menuitem");
    b.innerHTML = `${icon(it.icon || "chevron-right", "ico--sm")}<span>${esc(it.text)}</span>${it.checked ? `<span class="check" style="visibility:visible;margin-left:auto">${icon("check", "ico--sm")}</span>` : ""}`;
    b.onclick = (e) => { e.stopPropagation(); closeMenu(); it.run(); };
    m.append(b);
  }
  m.hidden = false;
  const r = anchor.getBoundingClientRect(), mr = m.getBoundingClientRect();
  let x = r.right - mr.width, y = r.bottom + 4;
  if (x < 8) x = Math.min(r.left, innerWidth - mr.width - 8);
  if (y + mr.height > innerHeight - 8) y = Math.max(8, r.top - mr.height - 4);
  m.style.left = `${Math.max(8, x)}px`; m.style.top = `${y}px`;
  menuFor = anchor; anchor.setAttribute("aria-expanded", "true");
  const first = m.querySelector("button");
  if (first) first.focus({preventScroll: true});
}
document.addEventListener("click", (e) => { if (!e.target.closest("#ctx-menu")) closeMenu(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape") { closeMenu(); closeEffort(); } });

function renameInline(rowSel, current, commit) {
  const row = document.querySelector(rowSel);
  if (!row) return;
  const title = row.querySelector(".row__title");
  const input = document.createElement("input");
  input.className = "rename"; input.value = current; input.setAttribute("aria-label", "New name");
  title.replaceWith(input);
  input.focus(); input.select();
  let done = false;
  const finish = (save) => {
    if (done) return; done = true;
    const v = input.value.trim();
    if (save && v && v !== current) commit(v); else renderSidebar();
  };
  input.onkeydown = (e) => { if (e.key === "Enter") finish(true); if (e.key === "Escape") finish(false); e.stopPropagation(); };
  input.onblur = () => finish(true);
  input.onclick = (e) => e.stopPropagation();
}
function chatMenu(c, anchor) {
  const items = [
    {text: "Rename", icon: "rename", run: () => renameInline(`[data-chat="${c.id}"]`, c.title, (v) => { c.title = v; persist(c); renderSidebar(); })},
    {text: c.pinned ? "Unpin" : "Pin", icon: "pin", run: () => { c.pinned = !c.pinned; persist(c); renderSidebar(); }},
  ];
  if (projects.length) {
    items.push("-", {label: "Move to project"});
    for (const p of projects) items.push({text: p.name, icon: "folder", checked: c.projectId === p.id,
      run: () => { c.projectId = p.id; openProjects.add(p.id); ls.set("openProjects", [...openProjects]); persist(c); renderSidebar(); renderChatHead(); }});
    if (c.projectId) items.push({text: "Out of the project", icon: "move", run: () => { c.projectId = null; persist(c); renderSidebar(); renderChatHead(); }});
  }
  items.push("-", {text: "Delete", icon: "trash", danger: true, run: () => deleteChat(c)});
  openMenu(anchor, items);
}
function deleteChat(c) {
  if (busy && busy.chat === c) busy.controller.abort();
  chats = chats.filter((x) => x !== c);
  store.deleteChat(c.id);
  if (chat === c) startChat(c.projectId);
  renderSidebar();
  toast("Chat deleted", "", {ms: 6000, action: {label: "Undo", run: () => { chats.push(c); store.putChat(c); renderSidebar(); }}});
}
function projectMenu(p, anchor) {
  openMenu(anchor, [
    {text: "New chat here", icon: "plus", run: () => startChat(p.id)},
    {text: "Rename", icon: "rename", run: () => renameInline(`[data-project="${p.id}"]`, p.name, (v) => { p.name = v; store.putProject(p); renderSidebar(); renderChatHead(); })},
    "-",
    {text: "Delete project", icon: "trash", danger: true, run: () => deleteProject(p)},
  ]);
}
function deleteProject(p) {
  const inside = chats.filter((c) => c.projectId === p.id);
  const note = inside.length ? `\n\nIts ${inside.length} chat${inside.length > 1 ? "s" : ""} move to your chat list.` : "";
  if (!confirm(`Delete the project "${p.name}"?${note}`)) return;
  for (const c of inside) { c.projectId = null; store.putChat(c); }
  if (chat && chat.projectId === p.id) chat.projectId = null;
  projects = projects.filter((x) => x !== p);
  store.deleteProject(p.id);
  renderSidebar(); renderChatHead();
}
function newProject() {
  const p = {id: uid(), name: "New project", created: Date.now()};
  projects.push(p);
  store.putProject(p);
  openProjects.add(p.id); ls.set("openProjects", [...openProjects]);
  renderSidebar();
  renameInline(`[data-project="${p.id}"]`, p.name, (v) => { p.name = v; store.putProject(p); renderSidebar(); });
}

// ------------------------------------------------------------------ opening chats
function startChat(projectId = null) {
  chat = newChat(projectId);
  ls.set("current", null);
  attachments = []; renderAttachments();
  renderChat(); renderSidebar();
  closeSideOnMobile();
  showTab("chat");
  $("input").focus();
}
function openChat(id) {
  const c = chats.find((x) => x.id === id);
  if (!c) return;
  chat = c;
  ls.set("current", c.id);
  renderChat(); renderSidebar();
  closeSideOnMobile();
  showTab("chat");
}

// ------------------------------------------------------------------ messages
function renderChatHead() {
  const p = chat && chat.projectId ? projects.find((x) => x.id === chat.projectId) : null;
  $("chat-head").hidden = !p;
  if (p) $("chat-head-project").innerHTML = `${icon("folder", "ico--sm")}<span>${esc(p.name)}</span>`;
}
function userMsgEl(m) {
  const el = document.createElement("div");
  el.className = "msg msg--user";
  const imgs = (m.images || []).filter((i) => i.url);
  const files = m.files || [];
  let html = "";
  if (imgs.length || files.length) {
    html += `<div class="files">${imgs.map((i) => `<img src="${esc(i.url)}" alt="${esc(i.name || "image")}">`).join("")}` +
      files.map((f) => `<span class="chip">${icon("file", "ico--sm")}<span>${esc(f.name)}</span></span>`).join("") + "</div>";
  }
  if (m.text) html += `<div class="bubble">${esc(m.text)}</div>`;
  html += `<div class="msg__actions"><button class="icon-btn" type="button" data-act="copy" aria-label="Copy message" title="Copy">${icon("copy")}</button></div>`;
  el.innerHTML = html;
  el.querySelector('[data-act="copy"]').onclick = (e) => copyText(m.text || "", e.currentTarget);
  return el;
}
function assistantMsgEl(m) {
  const el = document.createElement("div");
  el.className = "msg msg--assistant";
  el.innerHTML = `<details class="think" hidden><summary>${icon("brain", "ico--sm")}<span class="think__label"></span>${icon("chevron-down", "ico--sm chev")}</summary>` +
    `<div class="think__body"></div></details><div class="answer"></div><div class="msg__error" hidden></div>` +
    `<div class="msg__foot"><div class="msg__actions">` +
    `<button class="icon-btn" type="button" data-act="copy" aria-label="Copy answer" title="Copy">${icon("copy")}</button>` +
    `<button class="icon-btn" type="button" data-act="regen" aria-label="Answer again" title="Answer again">${icon("refresh")}</button>` +
    `</div><span class="msg__meta"></span></div>`;
  const details = el.querySelector(".think");
  details.querySelector("summary").onclick = (e) => {
    e.preventDefault();
    m.thinkOpen = !details.open;
    details.open = m.thinkOpen;
    if (details.open) {
      el._rlen = -1;
      el._twT = SlenderTyping.newState(Infinity);   // the thinking so far appears at once; only new words fade in
      updateAssistant(el, m, busy && busy.msg === m);
    }
  };
  el.querySelector('[data-act="copy"]').onclick = (e) => copyText(m.text || "", e.currentTarget);
  el.querySelector('[data-act="regen"]').onclick = () => regenerate(m);
  el.querySelector(".answer").addEventListener("click", (e) => {
    const b = e.target.closest("[data-copy-code]");
    if (b) copyText(b.closest(".code").querySelector("pre").textContent, b);
    const t = e.target.closest(".tool > summary");
    if (t) {
      e.preventDefault();
      const tool = (m.tools || [])[+t.parentElement.dataset.tool];
      if (tool) { tool.open = !tool.open; el._html = null; updateAssistant(el, m, busy && busy.msg === m); }
    }
  });
  // what is on the screen now counts as already shown: no typing effect for it
  el._twA = SlenderTyping.newState(Infinity);
  el._twT = SlenderTyping.newState(Infinity);
  return el;
}
const TOOL_STATE = {writing: "Writing", running: "Running", done: "Done", error: "Error", skipped: "Not run"};
function toolHtml(t, k) {
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2);
  let body = "";
  if (t.open) {
    body = `<pre>${esc(args || "(being written)")}</pre>`;
    if (t.result != null) body += `<pre>${esc(t.result)}</pre>`;
  }
  return `<details class="tool" data-tool="${k}" data-state="${esc(t.state)}"${t.open ? " open" : ""}><summary>${icon("tool", "ico--sm")}` +
    `<span class="tool__name">${esc(t.tool || t.name || "tool")}</span>${t.server ? `<span class="msg__meta">${esc(t.server)}</span>` : ""}` +
    `<span class="tool__state">${esc(TOOL_STATE[t.state] || t.state)}${t.ms != null && t.state !== "skipped" ? ` · ${fmt(t.ms / 1000, 1)} s` : ""}</span></summary>${body}</details>`;
}
function textHtml(text) {
  return settings.math && window.StrataMath ? StrataMath.markdown(text, markdown) : markdown(text);
}
function answerHtml(m) {
  if (!m.tools || !m.tools.length) return textHtml(m.text || "");
  let html = "", pos = 0;
  m.tools.forEach((t, k) => {
    const at = Math.min(Math.max(t.at || 0, pos), m.text.length);
    if (at > pos) html += textHtml(m.text.slice(pos, at));
    pos = at;
    html += toolHtml(t, k);
  });
  return html + textHtml(m.text.slice(pos));
}
function onTool(m, x) {
  if (x.event === "limit") { m.limit = x.max_rounds; return; }
  m.tools = m.tools || [];
  let t = m.tools.find((y) => y.id === x.id);
  if (!t) { t = {id: x.id, name: x.name, at: m.text.length, rat: m.reasoning.length, state: "writing"}; m.tools.push(t); }
  if (x.event === "call") {
    Object.assign(t, {name: x.name, server: x.server, tool: x.tool, arguments: x.arguments, round: x.round, state: "running"});
  } else if (x.event === "result") {
    Object.assign(t, {result: x.text, ok: x.ok, chars: x.chars, truncated: x.truncated, ms: x.ms, images: x.images || [],
                      vision: x.vision === true, state: x.skipped ? "skipped" : x.ok ? "done" : "error"});
  }
}
function updateAssistant(el, m, streaming) {
  // also on the last render after the stream ends, so the final words finish their fade instead of snapping in;
  // a message drawn from history has nothing young to animate (its typing state starts as "all shown")
  const animate = SlenderTyping.allowed(settings.typing);
  // thinking
  const details = el.querySelector(".think");
  if (m.reasoning) {
    details.hidden = false;
    const live = streaming && !m.text && !(m.tools || []).some((t) => t.state === "running");
    const label = details.querySelector(".think__label");
    label.classList.toggle("is-live", live);
    label.textContent = live ? "Thinking…" : m.thinkSecs != null ? `Thought for ${m.thinkSecs < 10 ? fmt(m.thinkSecs, 1) : fmt(m.thinkSecs)} s` : "Thoughts";
    const want = m.thinkOpen != null ? m.thinkOpen : (streaming && !m.text && settings.thinkOpen);
    if (details.open !== want) details.open = want;
    const body = details.querySelector(".think__body");
    if (details.open && el._rlen !== m.reasoning.length) {
      const follow = body.scrollHeight - body.scrollTop - body.clientHeight < 30;
      body.textContent = m.reasoning;
      el._rlen = m.reasoning.length;
      SlenderTyping.apply(body, el._twT, animate);
      if (follow) body.scrollTop = body.scrollHeight;
    }
  } else {
    details.hidden = true;
  }
  // the answer
  const answer = el.querySelector(".answer");
  if (streaming && !m.text && !m.reasoning && !(m.tools || []).length) {
    if (!answer.querySelector(".wait")) answer.innerHTML = '<span class="wait" aria-label="Waiting for the answer"><i></i><i></i><i></i></span>';
    el._html = null;
  } else {
    const html = answerHtml(m);
    if (html !== el._html) {
      answer.innerHTML = html;
      el._html = html;
      SlenderTyping.apply(answer, el._twA, animate);
      if (settings.math && window.StrataMath) StrataMath.hydrate(answer);
    }
  }
  const err = el.querySelector(".msg__error");
  err.hidden = !m.error;
  err.textContent = m.error || "";
  el.querySelector(".msg__meta").textContent = m.meta || "";
  el.querySelector(".msg__foot").hidden = streaming;
}
const elOf = new WeakMap();        // message -> its element on the screen (never stored with the chat)
function renderChat() {
  const box = $("messages");
  box.replaceChildren();
  const msgs = chat.messages;
  $("view-chat").classList.toggle("is-empty", !msgs.length);
  renderChatHead();
  const lastAssistant = msgs.length && msgs[msgs.length - 1].role === "assistant" ? msgs[msgs.length - 1] : null;
  for (const m of msgs) {
    if (m.role === "user") { box.append(userMsgEl(m)); continue; }
    const el = assistantMsgEl(m);
    if (m === lastAssistant) el.classList.add("msg--last");
    el.querySelector('[data-act="regen"]').hidden = m !== lastAssistant;
    elOf.set(m, el);
    box.append(el);
    updateAssistant(el, m, !!(busy && busy.msg === m));
  }
  setBusyUi();
  stick = true;
  scrollDown(true);
}

// ------------------------------------------------------------------ scrolling: follow the answer unless the reader scrolled up
let stick = true;
const thread = $("thread");
thread.addEventListener("scroll", () => {
  stick = thread.scrollHeight - thread.scrollTop - thread.clientHeight < 60;
  $("jump").hidden = stick || !chat.messages.length;
}, {passive: true});
function scrollDown(force) {
  if (force || stick) thread.scrollTop = thread.scrollHeight;
}
$("jump").onclick = () => { stick = true; thread.scrollTo({top: thread.scrollHeight, behavior: "smooth"}); };

// ------------------------------------------------------------------ the request
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|ipynb|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function userText(m) {
  return [m.text, ...(m.files || []).filter((f) => f.text != null).map(fileBlock)].filter((s) => s).join("\n\n");
}
function assistantMessages(m) {
  const ran = (m.tools || []).filter((t) => t.round != null && t.result != null && t.state !== "skipped");
  if (!ran.length) {
    const msg = {role: "assistant", content: m.text || ""};
    if (!m.text && m.reasoning) msg.reasoning_content = m.reasoning;
    return [msg];
  }
  const out = [];
  let pos = 0;
  for (const r of [...new Set(ran.map((t) => t.round))]) {
    const calls = ran.filter((t) => t.round === r);
    const at = Math.min(Math.max(pos, calls[0].at || 0), m.text.length);
    out.push({role: "assistant", content: m.text.slice(pos, at).trim(),
              tool_calls: calls.map((t) => ({id: t.id, type: "function", function: {name: t.name, arguments: JSON.stringify(t.arguments || {})}}))});
    for (const t of calls) out.push({role: "tool", tool_call_id: t.id, content: t.result});
    pos = at;
  }
  const rest = m.text.slice(pos).trim();
  if (rest) out.push({role: "assistant", content: rest});
  return out;
}
function apiMessages(c, current) {
  const out = [];
  const sys = (settings.system || "").trim();
  if (sys) out.push({role: "system", content: sys});
  for (const m of c.messages) {
    if (m === current) continue;
    if (m.role === "user") {
      const imgs = (m.images || []).filter((i) => i.url);
      const text = userText(m);
      out.push({role: "user", content: imgs.length ? [{type: "text", text}, ...imgs.map((i) => ({type: "image_url", image_url: {url: i.url}}))] : text});
    } else {
      out.push(...assistantMessages(m));
    }
  }
  return out;
}

function setBusyUi() {
  const mine = !!(busy && chat && busy.chat === chat);
  $("send-btn").hidden = mine;
  $("stop-btn").hidden = !mine;
  paintDot();
}
async function send() {
  const text = $("input").value.trim();
  if (!text && !attachments.length) return;
  if (busy) { toast("One answer at a time", busy.chat === chat ? "Stop this answer or wait for it to finish." : "Another chat is still being answered."); return; }
  const m = {role: "user", text, images: attachments.filter((a) => a.kind === "image"), files: attachments.filter((a) => a.kind === "file"), time: Date.now()};
  if (isNew(chat)) {
    chat.title = titleFrom(text || (m.files[0] && m.files[0].name) || "Pictures");
    chats.push(chat);
    ls.set("current", chat.id);
  }
  chat.messages.push(m);
  attachments = []; renderAttachments();
  $("input").value = ""; autosize();
  return generateReply(chat);
}
function regenerate(m) {
  if (busy) return;
  const i = chat.messages.indexOf(m);
  if (i < 0 || i !== chat.messages.length - 1) return;
  chat.messages.splice(i, 1);
  generateReply(chat);
}

async function generateReply(c) {
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  c.messages.push(m);
  c.updated = Date.now();
  persist(c);
  renderSidebar();
  const controller = new AbortController();
  busy = {controller, msg: m, chat: c};
  renderChat();
  const elNow = () => (chat === c ? elOf.get(m) : null);
  if (elNow()) { elNow()._twA = SlenderTyping.newState(0); elNow()._twT = SlenderTyping.newState(0); }

  const body = {model: health.model || "slender", messages: apiMessages(c, m), stream: true, reasoning_effort: settings.effort};
  if (!settings.card) {
    if (!(+settings.temperature > 0)) body.temperature = 0;
    else Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, top_k: +settings.top_k});
  }
  if (settings.seed !== "" && settings.seed != null) body.seed = +settings.seed;
  if (settings.max) body.max_tokens = +settings.max;
  if (settings.mcp !== false && mcpInfo.tools > 0) body.strata_mcp = true;

  let firstAt = null, thinkStart = null, usage = null, timings = null, frame = 0, lastSave = performance.now();
  const paint = () => {
    frame = 0;
    const el = elNow();
    if (el) { updateAssistant(el, m, true); scrollDown(); }
    if (performance.now() - lastSave > 2500) { lastSave = performance.now(); persist(c); }
  };
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), body: JSON.stringify(body), signal: controller.signal});
    if (!r.ok) {
      let msg = `HTTP ${r.status}`;
      try { msg = (await r.json()).error.message || msg; } catch (e) { /* not json */ }
      if (r.status === 401) msg = "This server needs an API key: add it in Settings > Connection.";
      throw new Error(msg);
    }
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith("data:")) continue;          // ": keep-alive" while a long prompt is read
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        let j;
        try { j = JSON.parse(data); } catch (e) { continue; }
        if (j.error) throw new Error(j.error.message || "the engine reported an error");
        if (j.usage) usage = j.usage;
        if (j.timings) timings = j.timings;
        if (j.strata_mcp) onTool(m, j.strata_mcp);
        const d = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        const lastTool = m.tools && m.tools.length ? m.tools[m.tools.length - 1] : null;
        if (d.reasoning_content) {
          if (!firstAt) firstAt = performance.now();
          if (!thinkStart) thinkStart = performance.now();
          if (lastTool && m.reasoning && lastTool.rat === m.reasoning.length) m.reasoning += "\n\n";
          m.reasoning += d.reasoning_content;
        }
        if (d.content) {
          if (!firstAt) firstAt = performance.now();
          if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
          if (lastTool && m.text && lastTool.at === m.text.length) m.text += "\n\n";
          m.text += d.content;
        }
        if (!frame) frame = requestAnimationFrame(paint);
      }
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else m.error = e.message || String(e);
  }
  if (frame) cancelAnimationFrame(frame);
  if (thinkStart && m.thinkSecs == null) m.thinkSecs = (performance.now() - thinkStart) / 1000;
  for (const t of m.tools || []) if (t.state === "writing" || t.state === "running") { t.state = "skipped"; t.ms = null; }
  const parts = [];
  const n = usage ? usage.completion_tokens : timings ? timings.predicted_n : null;
  if (n) parts.push(`${fmt(n)} tokens`);
  if (timings && timings.predicted_ms > 0 && timings.predicted_n > 1) parts.push(`${fmt(timings.predicted_n / (timings.predicted_ms / 1000), 1)} tok/s`);
  else if (n && firstAt) { const s = (performance.now() - firstAt) / 1000; if (s > 0.25) parts.push(`${fmt(n / s, 1)} tok/s`); }
  if (timings && timings.prompt_n > 0 && timings.prompt_ms > 0) parts.push(`read ${fmt(timings.prompt_n)} in ${fmt(timings.prompt_ms / 1000, 1)} s`);
  const ran = (m.tools || []).filter((t) => t.state === "done" || t.state === "error").length;
  if (ran) parts.push(`${ran} tool call${ran > 1 ? "s" : ""}`);
  if (m.stopped) parts.push("stopped");
  if (m.limit) parts.push(`stopped at ${m.limit} tool rounds`);
  m.meta = parts.join(" · ");
  busy = null;
  c.updated = Date.now();
  persist(c);
  const el = elNow();
  if (el) { updateAssistant(el, m, false); scrollDown(); }
  setBusyUi();
  renderSidebar();
  if (m.error && chat !== c) toast("An answer failed", m.error, {kind: "error", ms: 7000});
}

// ------------------------------------------------------------------ composer
const input = $("input");
function autosize() { input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight, innerHeight * 0.4)}px`; }
input.addEventListener("input", autosize);
input.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || e.isComposing) return;
  if (settings.enterSends ? !e.shiftKey : (e.ctrlKey || e.metaKey)) { e.preventDefault(); send(); }
});
$("composer").onsubmit = (e) => { e.preventDefault(); send(); };
$("stop-btn").onclick = () => { if (busy) busy.controller.abort(); };

function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      if (!health.images) { toast("Pictures are off", "This model was started for text only."); continue; }
      if (f.size > 20e6) { toast("Picture too large", `${f.name} is over 20 MB.`); continue; }
      const r = new FileReader();
      r.onload = () => { attachments.push({kind: "image", name: f.name || "pasted image", url: r.result}); renderAttachments(); };
      r.readAsDataURL(f);
      continue;
    }
    const textLike = f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) || TEXT_EXT.test(f.name) ||
      /(^|[\\/])(makefile|dockerfile|readme|license)$/i.test(f.name);
    if (!textLike) { toast("Not supported yet", `${f.name}: PDFs and documents come with the file tools. Text files and pictures work now.`); continue; }
    if (f.size > 512 * 1024) { toast("File too large", `${f.name} is over 512 KB.`); continue; }
    const r = new FileReader();
    r.onload = () => {
      const text = String(r.result);
      if (text.includes("\u0000")) { toast("Not a text file", `${f.name} looks like a binary file.`); return; }
      attachments.push({kind: "file", name: f.name, text});
      renderAttachments();
    };
    r.readAsText(f);
  }
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.replaceChildren();
  attachments.forEach((a, i) => {
    const el = document.createElement("div");
    el.className = "att";
    el.innerHTML = (a.kind === "image" ? `<img src="${esc(a.url)}" alt="">` : icon("file")) + `<span>${esc(a.name)}</span>` +
      `<button type="button" aria-label="Remove ${esc(a.name)}">${icon("x", "ico--sm")}</button>`;
    el.querySelector("button").onclick = () => { attachments.splice(i, 1); renderAttachments(); };
    box.append(el);
  });
}
$("attach-btn").onclick = () => $("file-input").click();
$("file-input").onchange = (e) => { addFiles([...e.target.files]); e.target.value = ""; };
input.addEventListener("paste", (e) => {
  const files = [...(e.clipboardData ? e.clipboardData.files : [])];
  if (files.length) { e.preventDefault(); addFiles(files); }
});
const viewChat = $("view-chat");
viewChat.addEventListener("dragover", (e) => { if (e.dataTransfer.types.includes("Files")) { e.preventDefault(); $("composer").classList.add("is-drop"); } });
viewChat.addEventListener("dragleave", (e) => { if (!viewChat.contains(e.relatedTarget)) $("composer").classList.remove("is-drop"); });
viewChat.addEventListener("drop", (e) => {
  if (!e.dataTransfer.files.length) return;
  e.preventDefault(); $("composer").classList.remove("is-drop");
  addFiles([...e.dataTransfer.files]);
});

// ------------------------------------------------------------------ reasoning effort
const EFFORTS = [
  {v: "none", label: "Off", note: "Answers straight away, no thinking"},
  {v: "low", label: "Low", note: "A short think first"},
  {v: "medium", label: "Medium", note: "Thinks as much as the task needs"},
  {v: "high", label: "High", note: "Thinks it through (the model's default)"},
];
function renderEffort() {
  const cur = EFFORTS.find((e) => e.v === settings.effort) || EFFORTS[3];
  $("effort-label").textContent = cur.label;
  const menu = $("effort-menu");
  menu.innerHTML = '<div class="menu-label" style="padding:6px 10px 4px;font-size:12px;color:var(--ink-3);text-transform:uppercase;letter-spacing:.4px">Reasoning effort</div>' +
    EFFORTS.map((e) => `<button type="button" class="opt" role="option" data-v="${e.v}" aria-selected="${e.v === cur.v}">` +
      `<span>${esc(e.label)}<small>${esc(e.note)}</small></span><span class="check">${icon("check", "ico--sm")}</span></button>`).join("");
  for (const b of menu.querySelectorAll(".opt")) b.onclick = () => { settings.effort = b.dataset.v; saveSettings(); renderEffort(); closeEffort(); input.focus(); };
}
function closeEffort() { $("effort-menu").hidden = true; $("effort-btn").setAttribute("aria-expanded", "false"); }
$("effort-btn").onclick = (e) => {
  e.stopPropagation();
  const open = $("effort-menu").hidden;
  closeMenu();
  $("effort-menu").hidden = !open;
  $("effort-btn").setAttribute("aria-expanded", String(open));
  if (open) { const sel = $("effort-menu").querySelector('[aria-selected="true"]'); if (sel) sel.focus(); }
};
$("effort-menu").addEventListener("keydown", (e) => {
  if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
  e.preventDefault();
  const opts = [...$("effort-menu").querySelectorAll(".opt")];
  const i = opts.indexOf(document.activeElement);
  opts[(i + (e.key === "ArrowDown" ? 1 : opts.length - 1)) % opts.length].focus();
});
document.addEventListener("click", (e) => { if (!e.target.closest("#effort")) closeEffort(); });

// ------------------------------------------------------------------ settings window
function setSwitch(id, on) { $(id).setAttribute("aria-checked", String(!!on)); }
function bindSwitch(id, key, after) {
  $(id).onclick = () => { settings[key] = !settings[key]; setSwitch(id, settings[key]); saveSettings(); if (after) after(); };
}
function fillSettings() {
  $("s-system").value = settings.system;
  setSwitch("s-enter", settings.enterSends);
  setSwitch("s-card", settings.card);
  $("sampling").classList.toggle("is-off", settings.card);
  for (const [id, key, d] of [["temp", "temperature", 2], ["topp", "top_p", 2], ["topk", "top_k", 0]]) {
    $(`s-${id}`).value = settings[key];
    $(`o-${id}`).textContent = fmt(settings[key], d);
  }
  $("s-max").value = settings.max;
  $("s-seed").value = settings.seed;
  setSwitch("s-mcp", settings.mcp);
  $("s-mcp").disabled = !mcpInfo.tools;
  $("mcp-note").textContent = mcpInfo.tools
    ? `${mcpInfo.tools} tool${mcpInfo.tools > 1 ? "s" : ""} from ${mcpInfo.servers.length} server${mcpInfo.servers.length > 1 ? "s" : ""}; the server runs them for the model.`
    : "No MCP servers are set up in the run config.";
  setSwitch("s-typing", settings.typing);
  setSwitch("s-think", settings.thinkOpen);
  setSwitch("s-math", settings.math);
  $("s-key").value = apiKey;
  const facts = [["Model", health.model || "–"], ["Context", health.max_context ? `${fmt(health.max_context)} tokens` : "–"],
                 ["Pictures", health.images ? "Yes" : "No"], ["Loaded", health.loaded == null ? "–" : health.loaded ? "Yes" : "No (loads on the first message)"],
                 ["Chats saved", `${chats.length}`]];
  $("facts").innerHTML = facts.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
}
function showPane(name) {
  for (const b of $("settings-nav").children) b.setAttribute("aria-selected", String(b.dataset.pane === name));
  for (const p of document.querySelectorAll(".pane")) p.hidden = p.dataset.pane !== name;
}
$("settings-nav").onclick = (e) => { const b = e.target.closest("button[data-pane]"); if (b) showPane(b.dataset.pane); };
$("open-settings").onclick = () => { fillSettings(); $("settings").showModal(); closeSideOnMobile(); };
$("settings").addEventListener("click", (e) => { if (e.target === $("settings")) $("settings").close(); });   // a click on the backdrop
$("s-system").oninput = () => { settings.system = $("s-system").value; saveSettings(); };
bindSwitch("s-enter", "enterSends");
bindSwitch("s-card", "card", () => $("sampling").classList.toggle("is-off", settings.card));
for (const [id, key, d] of [["temp", "temperature", 2], ["topp", "top_p", 2], ["topk", "top_k", 0]]) {
  $(`s-${id}`).oninput = () => { settings[key] = +$(`s-${id}`).value; $(`o-${id}`).textContent = fmt(settings[key], d); saveSettings(); };
}
$("s-max").oninput = () => { settings.max = $("s-max").value; saveSettings(); };
$("s-seed").oninput = () => { settings.seed = $("s-seed").value; saveSettings(); };
bindSwitch("s-mcp", "mcp");
bindSwitch("s-typing", "typing");
bindSwitch("s-think", "thinkOpen");
bindSwitch("s-math", "math", () => { if (window.StrataMath) StrataMath.reset(); renderChat(); });
$("s-key").onchange = () => {
  apiKey = $("s-key").value.trim();
  toast(ls.set("apikey", apiKey) ? "API key saved" : "API key used for this visit only", ls.get("apikey") === apiKey ? "Kept in this browser only." : "Browser storage is unavailable.");
  loadHealth(); loadMcp();
};
$("s-wipe").onclick = async () => {
  if (!chats.length) { toast("No chats to delete"); return; }
  if (!confirm(`Delete all ${chats.length} chats? This cannot be undone.`)) return;
  if (busy) busy.controller.abort();
  chats = [];
  await store.clearChats();
  startChat();
  fillSettings();
  toast("Every chat was deleted");
};

// ------------------------------------------------------------------ the server's state
function paintDot() {
  const dot = $("model-dot");
  dot.className = "dot";
  if (busy) dot.classList.add("is-busy");
  else if (health.loaded === false) { /* idle grey: loads on the first message */ }
  else if (health.ok === false) dot.classList.add("is-off");
  else if (health.ok) dot.classList.add("is-on");
  $("model-pill").title = health.ok === false ? "The server does not answer" : busy ? "Answering" :
    health.loaded === false ? "Not loaded: it loads on the first message" : "Loaded and ready";
}
async function loadHealth() {
  try {
    const r = await fetch("health", {headers: headers(), cache: "no-store"});
    const h = await r.json();
    health = {...h, ok: true};
    $("model-name").textContent = h.model || "Model";
  } catch (e) {
    health = {...health, ok: false};
    $("model-name").textContent = "Server not reachable";
  }
  paintDot();
}
async function loadMcp() {
  try {
    const r = await fetch("mcp", {headers: headers()});
    if (r.ok) mcpInfo = await r.json();
  } catch (e) { /* no tools */ }
}

// ------------------------------------------------------------------ tabs and sidebar
function showTab(name) {
  for (const t of document.querySelectorAll(".tab")) t.setAttribute("aria-selected", String(t.dataset.tab === name));
  $("view-chat").hidden = name !== "chat";
  $("view-analytics").hidden = name !== "analytics";
}
for (const t of document.querySelectorAll(".tab")) t.onclick = () => showTab(t.dataset.tab);
const narrow = matchMedia("(max-width: 860px)");
function closeSideOnMobile() { if (narrow.matches) { $("app").classList.remove("side-open"); $("scrim").hidden = true; } }
$("side-open").onclick = () => {
  if (narrow.matches) { $("app").classList.add("side-open"); $("scrim").hidden = false; }
  else { $("app").classList.remove("side-collapsed"); ls.set("sideCollapsed", false); }
};
$("side-close").onclick = () => {
  if (narrow.matches) closeSideOnMobile();
  else { $("app").classList.add("side-collapsed"); ls.set("sideCollapsed", true); }
};
$("scrim").onclick = closeSideOnMobile;
$("new-chat").onclick = () => startChat();
$("top-new").onclick = () => startChat();
$("new-project").onclick = newProject;
$("chat-search").oninput = (e) => { searchQuery = e.target.value; renderSidebar(); };
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); startChat(); }
});

// ------------------------------------------------------------------ start
async function start() {
  SlenderIcons.fill();
  if (ls.get("sideCollapsed", false)) $("app").classList.add("side-collapsed");
  renderEffort();
  await store.start();
  [chats, projects] = await Promise.all([store.chats(), store.projects()]);
  for (const c of chats) {                       // an answer cut off by a closed page: keep what was written
    const last = c.messages[c.messages.length - 1];
    if (last && last.role === "assistant" && !last.meta && !last.error && !last.stopped && (last.text || last.reasoning)) last.meta = "interrupted";
  }
  const cur = ls.get("current", null);
  chat = chats.find((c) => c.id === cur) || newChat();
  renderSidebar();
  renderChat();
  input.focus();
  await Promise.all([loadHealth(), loadMcp()]);
  setInterval(loadHealth, 5000);
}
start();
