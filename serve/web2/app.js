// serve/web2/app.js - the SlenderEngine web app: chats and projects in the sidebar, the chat, settings.
// Talks to this server only (/health, /mcp, /v1/chat/completions, /v1/files, /v1/search). Chats stay in the browser.
//
// Smooth under load: the engine keeps the GPU and most CPU cores busy while it answers, so the page does as little
// as it can per frame. A streaming answer is drawn in two parts: finished paragraphs are rendered once and never
// touched again, only the paragraph being written is rebuilt. The thinking text is appended, not rebuilt. Paints are
// spaced out when a paint gets slow (a starved CPU), and every animation is transform/opacity only.
"use strict";
const $ = (id) => document.getElementById(id);
const {icon} = SlenderIcons;
const {markdown, esc} = SlenderMarkdown;
const fmt = (n, d = 0) => (n == null || Number.isNaN(+n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
const uid = () => (crypto.randomUUID ? crypto.randomUUID() : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`);
const estTokens = (s) => Math.ceil((s ? s.length : 0) / 3.6);   // a rough count for planning; the server's usage wins
const INLINE_CHARS = 20000;    // an attached document up to this size goes into the message; longer ones are read by tools

// ------------------------------------------------------------------ browser storage (small keys only)
const ls = {
  get(k, d) { try { const v = localStorage.getItem("slender." + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem("slender." + k, JSON.stringify(v)); return true; } catch (e) { return false; } },
};

// ------------------------------------------------------------------ settings
const DEFAULTS = {system: "", effort: "high", enterSends: true, card: true, temperature: 1.0, top_p: 0.95, top_k: 20,
                  max: "", seed: "", mcp: true, typing: true, thinkOpen: true, math: true, search: false,
                  autoCompact: true, compactAt: 80, theme: "black", font: "system", corners: "soft", textSize: "normal"};
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
function download(name, text, type) {
  const url = URL.createObjectURL(new Blob([text], {type}));
  const a = document.createElement("a");
  a.href = url; a.download = name;
  document.body.append(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// ------------------------------------------------------------------ state
const store = SlenderStore.createStore({onProblem: (k) => toast(k === "memory" ? "Chats are not saved" : "A chat could not be saved",
  k === "memory" ? "This browser keeps no storage for the page (a private window?). Chats last until you close it." : "The browser refused the write (storage full?).",
  {kind: "error", ms: 8000})});
let chats = [];                 // every chat record, messages included
let projects = [];
let chat = null;                // the open chat (not in `chats` until its first message)
let busy = null;                // {controller, msg, chat, compacting}
let health = {model: "", max_context: 0, images: false, loaded: null};
let mcpInfo = {servers: [], tools: 0};
let builtins = false;           // the server has the web app's own tools (files, search)
let searchInfo = {up: false, can_start: false, starting: false, error: null, url: null};
let attachments = [];           // {kind: "image"|"file"|"doc", name, url|text|id, busy}
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
const projectOf = (c) => (c && c.projectId ? projects.find((p) => p.id === c.projectId) : null);

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
const DRAG = "text/x-slender-chat";
function moveChat(id, projectId) {
  const c = chats.find((x) => x.id === id);
  if (!c || (c.projectId || null) === (projectId || null)) return;
  c.projectId = projectId || null;
  if (projectId) { openProjects.add(projectId); ls.set("openProjects", [...openProjects]); }
  persist(c);
  renderSidebar(); renderChatHead();
}
function dropTarget(el, projectId) {
  el.addEventListener("dragover", (e) => {
    if (!e.dataTransfer.types.includes(DRAG)) return;
    e.preventDefault(); e.stopPropagation();
    e.dataTransfer.dropEffect = "move";
    el.classList.add("is-drop");
  });
  el.addEventListener("dragleave", (e) => { if (!el.contains(e.relatedTarget)) el.classList.remove("is-drop"); });
  el.addEventListener("drop", (e) => {
    const id = e.dataTransfer.getData(DRAG);
    if (!id) return;
    e.preventDefault(); e.stopPropagation();
    el.classList.remove("is-drop");
    moveChat(id, projectId);
  });
}
function chatRow(c) {
  const row = document.createElement("div");
  row.className = `row${chat && c.id === chat.id ? " is-active" : ""}`;
  row.tabIndex = 0;
  row.setAttribute("role", "button");
  row.draggable = true;
  row.dataset.chat = c.id;
  row.innerHTML = `${c.pinned ? '<span class="pin-dot" title="Pinned"></span>' : ""}<span class="row__title">${esc(c.title)}</span>` +
    `<button class="row__more" type="button" aria-label="Chat options" aria-haspopup="menu" aria-expanded="false">${icon("more", "ico--sm")}</button>`;
  row.onclick = (e) => { if (!e.target.closest(".row__more, input")) openChat(c.id); };
  row.onkeydown = (e) => { if (e.key === "Enter" && e.target === row) openChat(c.id); };
  row.ondragstart = (e) => { e.dataTransfer.setData(DRAG, c.id); e.dataTransfer.effectAllowed = "move"; };
  row.querySelector(".row__more").onclick = (e) => { e.stopPropagation(); chatMenu(c, e.currentTarget); };
  return row;
}
function renderSidebar() {
  const q = searchQuery.trim().toLowerCase();
  const sorted = [...chats].sort((a, b) => b.updated - a.updated);
  const pl = $("project-list");
  pl.replaceChildren();
  const ps = [...projects].sort((a, b) => a.created - b.created);
  if (!ps.length && !q) {
    const e = document.createElement("div");
    e.className = "side-empty"; e.textContent = "No projects yet. Drag chats onto one to file them.";
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
    row.title = p.instructions ? `${p.name}: has instructions` : p.name;
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
    dropTarget(row, p.id);
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
      dropTarget(box, p.id);
      pl.append(box);
    }
  }
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
dropTarget($("chat-list"), null);

// a small menu next to a button (chat and project options)
let menuFor = null;
function closeMenu() {
  $("ctx-menu").hidden = true;
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
    if (it.disabled) b.disabled = true;
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
  const first = m.querySelector("button:not(:disabled)");
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
function chatMarkdown(c) {
  const out = [`# ${c.title}`, ""];
  for (const m of c.messages) {
    out.push(m.role === "user" ? "**You:**" : "**Assistant:**", "", m.role === "user" ? userText(m) : (m.text || ""), "");
  }
  return out.join("\n");
}
function chatMenu(c, anchor) {
  const items = [
    {text: "Rename", icon: "rename", run: () => renameInline(`[data-chat="${c.id}"]`, c.title, (v) => { c.title = v; persist(c); renderSidebar(); })},
    {text: c.pinned ? "Unpin" : "Pin", icon: "pin", run: () => { c.pinned = !c.pinned; persist(c); renderSidebar(); }},
    {text: "Export as Markdown", icon: "download", run: () => download(`${c.title.replace(/[\\/:*?"<>|]+/g, " ").slice(0, 60) || "chat"}.md`, chatMarkdown(c), "text/markdown")},
  ];
  if (projects.length) {
    items.push("-", {label: "Move to project"});
    for (const p of projects) items.push({text: p.name, icon: "folder", checked: c.projectId === p.id, run: () => moveChat(c.id, p.id)});
    if (c.projectId) items.push({text: "Out of the project", icon: "move", run: () => moveChat(c.id, null)});
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
    {text: "Name and instructions…", icon: "rename", run: () => projectDialog(p)},
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
// the project window: name and instructions, for a new project (p = null) or an existing one. Saved from the form's
// submit (the Save button or Enter in the name), which every browser sends at once; Cancel, Escape or the X just close.
let projectEditing = undefined;
function projectDialog(p) {
  projectEditing = p || null;
  $("proj-title").textContent = p ? "Project" : "New project";
  $("proj-name").value = p ? p.name : "";
  $("proj-instr").value = p ? p.instructions || "" : "";
  $("proj-dialog").showModal();
  $("proj-name").focus();
}
function saveProject() {
  const p = projectEditing;
  projectEditing = undefined;
  if (p === undefined) return;
  const name = $("proj-name").value.trim() || "Untitled project";
  const instructions = $("proj-instr").value.trim();
  if (p) { p.name = name; p.instructions = instructions; store.putProject(p); }
  else {
    const np = {id: uid(), name, instructions, created: Date.now()};
    projects.push(np); store.putProject(np);
    openProjects.add(np.id); ls.set("openProjects", [...openProjects]);
  }
  renderSidebar(); renderChatHead();
}
for (const b of $("proj-dialog").querySelectorAll("[data-close]")) b.onclick = () => { projectEditing = undefined; $("proj-dialog").close(); };
$("proj-form").addEventListener("submit", (e) => {
  const v = e.submitter ? e.submitter.value : "save";
  if (v === "save") saveProject(); else projectEditing = undefined;
});

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
  const p = projectOf(chat);
  $("chat-head").hidden = !p;
  if (p) $("chat-head-project").innerHTML = `${icon("folder", "ico--sm")}<span>${esc(p.name)}</span>`;
}
function userMsgEl(m) {
  const el = document.createElement("div");
  el.className = "msg msg--user";
  const imgs = (m.images || []).filter((i) => i.url);
  const files = [...(m.files || []), ...(m.docs || [])];
  let html = "";
  if (imgs.length || files.length) {
    html += `<div class="files">${imgs.map((i) => `<img src="${esc(i.url)}" alt="${esc(i.name || "image")}" loading="lazy" decoding="async">`).join("")}` +
      files.map((f) => `<span class="chip" title="${esc(f.name)}">${icon("file", "ico--sm")}<span>${esc(f.name)}</span>` +
        (f.pages > 1 ? `<small>${fmt(f.pages)} parts</small>` : "") + "</span>").join("") + "</div>";
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
    if (details.open) { el._th = {len: -1, live: [], n: 0}; updateAssistant(el, m, !!(busy && busy.msg === m)); }
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
      if (tool) { tool.open = !tool.open; el._full = null; updateAssistant(el, m, !!(busy && busy.msg === m)); }
    }
  });
  // what is on the screen now counts as already shown: no typing effect for it
  el._twA = SlenderTyping.newState(Infinity);
  el._th = {len: -1, live: [], n: 0};
  return el;
}
const TOOL_STATE = {writing: "Writing", running: "Running", done: "Done", error: "Error", skipped: "Not run"};
function toolLabel(t) {
  const a = t.arguments || {};
  const name = t.tool || t.name || "tool";
  if (name === "web_search") return ["Web search", a.query || ""];
  if (name === "fetch_url") { try { return ["Reading", new URL(a.url).host]; } catch (e) { return ["Reading", a.url || ""]; } }
  if (name === "read_document") return ["Reading the document", a.pages ? `pages ${a.pages}` : ""];
  if (name === "search_document") return ["Searching the document", a.query || ""];
  return [name, ""];
}
function toolHtml(t, k) {
  const args = t.arguments == null ? "" : JSON.stringify(t.arguments, null, 2);
  let body = "";
  if (t.open) {
    body = `<pre>${esc(args || "(being written)")}</pre>`;
    if (t.result != null) body += `<pre>${esc(t.result)}</pre>`;
  }
  const [label, detail] = toolLabel(t);
  const server = t.server && t.server !== "built-in" ? t.server : "";
  return `<details class="tool" data-tool="${k}" data-state="${esc(t.state)}"${t.open ? " open" : ""}><summary>${icon(t.server === "built-in" && /search|fetch/.test(t.tool || "") ? "globe" : "tool", "ico--sm")}` +
    `<span class="tool__name">${esc(label)}</span>${detail ? `<span class="tool__detail">${esc(detail.slice(0, 80))}</span>` : ""}${server ? `<span class="tool__detail">${esc(server)}</span>` : ""}` +
    `<span class="tool__state">${esc(TOOL_STATE[t.state] || t.state)}${t.ms != null && t.state !== "skipped" ? ` · ${fmt(t.ms / 1000, 1)} s` : ""}</span></summary>${body}</details>`;
}
function textHtml(text) {
  return settings.math && window.StrataMath ? StrataMath.markdown(text, markdown) : markdown(text);
}
function hydrate(node) { if (settings.math && window.StrataMath) StrataMath.hydrate(node); }
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

// The last place a streaming answer can be cut into "finished" and "still being written": a blank line outside a
// code fence or a $$ formula, scanned from the last cut only (so the cost stays with the tail, not the whole answer).
function safeCut(text, from) {
  let i = from, fence = false, math = false, best = -1;
  while (i < text.length) {
    const nl = text.indexOf("\n", i);
    if (nl < 0) break;
    const line = text.slice(i, nl);
    if (/^\s*(```|~~~)/.test(line)) fence = !fence;
    else if (!fence && ((line.match(/\$\$/g) || []).length % 2)) math = !math;
    if (!fence && !math && !line.trim() && i > from) best = nl + 1;
    i = nl + 1;
  }
  return best;
}
const SETTLE_MS = 650;   // a finished paragraph moves out of the live tail once its last words have faded in
function paintAnswer(el, m, streaming, animate) {
  const answer = el.querySelector(".answer");
  if (m.tools && m.tools.length) {                 // answers with tool calls: drawn whole (rare, and short between calls)
    el._inc = null;
    const html = answerHtml(m);
    if (html !== el._full) {
      answer.innerHTML = html;
      el._full = html;
      SlenderTyping.apply(answer, el._twA, animate);
      hydrate(answer);
    }
    return;
  }
  let st = el._inc;
  if (!st) {
    answer.innerHTML = '<div class="ans-part"></div><div class="ans-part"></div>';
    st = el._inc = {settled: 0, cuts: [], tail: null, done: answer.firstChild, live: answer.lastChild};
    el._full = null;
  }
  const text = m.text || "";
  if (streaming) {
    // each cut point is remembered with the time it appeared; the newest one old enough (its last words have
    // faded in) becomes the new edge of the finished part. A fast stream of short paragraphs moves the newest
    // cut on every paint, so waiting for one cut to stay put would never settle anything.
    const now = performance.now();
    const cut = safeCut(text, st.settled);
    if (cut > st.settled && (!st.cuts.length || cut > st.cuts[st.cuts.length - 1].pos)) st.cuts.push({pos: cut, t: now});
    let edge = -1;
    while (st.cuts.length && now - st.cuts[0].t > SETTLE_MS) edge = st.cuts.shift().pos;
    if (edge > st.settled) {
      const part = document.createElement("div");        // its own box, so math is drawn in it alone
      part.className = "ans-chunk";
      part.innerHTML = textHtml(text.slice(st.settled, edge));
      SlenderTyping.shift(el._twA, SlenderTyping.textLength(part));
      st.done.append(part);
      hydrate(part);
      st.settled = edge; st.tail = null;
    }
  }
  const html = textHtml(text.slice(st.settled));
  if (html !== st.tail) {
    st.live.innerHTML = html;
    st.tail = html;
    SlenderTyping.apply(st.live, el._twA, animate);
    hydrate(st.live);
  }
}
// the thinking text is plain: new text is appended (new words as fading spans that become plain text again)
function paintThinking(el, m, streaming, animate) {
  const details = el.querySelector(".think");
  if (!m.reasoning) { details.hidden = true; return; }
  details.hidden = false;
  const live = streaming && !m.text && !(m.tools || []).some((t) => t.state === "running");
  const label = details.querySelector(".think__label");
  label.classList.toggle("is-live", live);
  const want = live ? "Thinking…" : m.thinkSecs != null ? `Thought for ${m.thinkSecs < 10 ? fmt(m.thinkSecs, 1) : fmt(m.thinkSecs)} s` : "Thoughts";
  if (label.textContent !== want) label.textContent = want;
  const open = m.thinkOpen != null ? m.thinkOpen : (streaming && !m.text && settings.thinkOpen);
  if (details.open !== open) { details.open = open; if (open) el._th = {len: -1, live: [], n: 0}; }
  if (!details.open) return;
  const body = details.querySelector(".think__body"), th = el._th;
  const follow = body.scrollHeight - body.scrollTop - body.clientHeight < 30;
  if (th.len < 0 || m.reasoning.length < th.len) {
    body.textContent = m.reasoning;
    th.len = m.reasoning.length; th.live = [];
  } else if (m.reasoning.length > th.len) {
    const chunk = m.reasoning.slice(th.len);
    th.len = m.reasoning.length;
    if (animate) {
      const now = performance.now(), frag = document.createDocumentFragment();
      for (const part of chunk.split(/(\s+)/)) {
        if (!part) continue;
        if (/^\s+$/.test(part)) { frag.append(part); continue; }
        const s = document.createElement("span");
        s.className = "tw"; s.textContent = part;
        frag.append(s); th.live.push({s, t: now});
      }
      body.append(frag);
    } else {
      body.append(chunk);
    }
  }
  const old = performance.now() - SlenderTyping.LIFE;
  while (th.live.length && th.live[0].t < old) { const x = th.live.shift(); x.s.replaceWith(x.s.textContent); }
  if (++th.n % 400 === 0) body.normalize();
  if (follow) body.scrollTop = body.scrollHeight;
}
function updateAssistant(el, m, streaming) {
  // also on the last render after the stream ends, so the final words finish their fade instead of snapping in;
  // a message drawn from history has nothing young to animate (its typing state starts as "all shown")
  const animate = SlenderTyping.allowed(settings.typing);
  paintThinking(el, m, streaming, animate);
  const answer = el.querySelector(".answer");
  if (streaming && !m.text && !m.reasoning && !(m.tools || []).length) {
    if (!answer.querySelector(".wait")) answer.innerHTML = '<span class="wait" aria-label="Waiting for the answer"><i></i><i></i><i></i></span>';
    el._inc = null; el._full = null;
  } else {
    paintAnswer(el, m, streaming, animate);
  }
  const err = el.querySelector(".msg__error");
  err.hidden = !m.error;
  if (err.textContent !== (m.error || "")) err.textContent = m.error || "";
  const meta = el.querySelector(".msg__foot .msg__meta");
  if (meta.textContent !== (m.meta || "")) meta.textContent = m.meta || "";
  el.querySelector(".msg__foot").hidden = streaming;
}
const elOf = new WeakMap();        // message -> its element on the screen (never stored with the chat)
function summaryEl(c, live) {
  const d = document.createElement("details");
  d.className = `summary${live ? " summary--busy" : ""}`;
  if (live) {
    d.innerHTML = `<summary>${icon("compress", "ico--sm")}<span>Compacting: summarising the earlier messages…</span></summary>`;
    return d;
  }
  const s = c.summary;
  d.innerHTML = `<summary>${icon("compress", "ico--sm")}<span>Earlier messages are summarised for the model` +
    `${s.tokensBefore ? ` (about ${fmt(s.tokensBefore)} tokens before)` : ""}</span>${icon("chevron-down", "ico--sm chev")}</summary>` +
    `<div class="summary__body"></div>`;
  d.querySelector(".summary__body").textContent = s.text;
  return d;
}
function renderChat() {
  const box = $("messages");
  box.replaceChildren();
  const msgs = chat.messages;
  $("view-chat").classList.toggle("is-empty", !msgs.length);
  renderChatHead();
  const lastAssistant = msgs.length && msgs[msgs.length - 1].role === "assistant" ? msgs[msgs.length - 1] : null;
  const upTo = chat.summary ? chat.summary.upTo : 0;
  const frag = document.createDocumentFragment();
  msgs.forEach((m, i) => {
    if (i === upTo && chat.summary) frag.append(summaryEl(chat));
    let el;
    if (m.role === "user") el = userMsgEl(m);
    else {
      el = assistantMsgEl(m);
      if (m === lastAssistant) el.classList.add("msg--last");
      el.querySelector('[data-act="regen"]').hidden = m !== lastAssistant;
      elOf.set(m, el);
      updateAssistant(el, m, !!(busy && busy.msg === m));
    }
    if (i < upTo) el.classList.add("msg--archived");
    frag.append(el);
  });
  if (chat.summary && upTo >= msgs.length) frag.append(summaryEl(chat));
  if (busy && busy.compacting && busy.chat === chat) frag.append(summaryEl(chat, true));
  box.append(frag);
  setBusyUi();
  renderCtx();
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

// ------------------------------------------------------------------ what the model is sent
const TEXT_EXT = /\.(txt|md|markdown|rst|tex|py|pyi|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|toml|ini|cfg|conf|env|xml|html?|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|dockerfile|gitignore|proto|graphql)$/i;
function fileBlock(f) {
  const longest = Math.max(2, ...(f.text.match(/`+/g) || []).map((s) => s.length));
  const fence = "`".repeat(longest + 1);
  return `File: ${f.name}\n${fence}\n${f.text}\n${fence}`;
}
function docNote(d) {
  return `Attached document: ${d.name} (id ${d.id}; ${fmt(d.pages)} part${d.pages === 1 ? "" : "s"}, ${fmt(d.chars)} characters). ` +
    "It is too long to include here: read it with read_document (a few parts at a time) and find things in it with search_document.";
}
function userText(m) {
  return [m.text, ...(m.files || []).filter((f) => f.text != null).map(fileBlock), ...(m.docs || []).map(docNote)].filter((s) => s).join("\n\n");
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
// one system message: the system prompt, the project's instructions, and the summary of compacted messages
function systemText(c) {
  const p = projectOf(c);
  const parts = [(settings.system || "").trim()];
  if (p && p.instructions) parts.push(`Instructions for this project (${p.name}):\n${p.instructions.trim()}`);
  if (c.summary) parts.push(`Summary of the earlier part of this conversation (the messages before it were compacted):\n${c.summary.text}`);
  return parts.filter((s) => s).join("\n\n");
}
function apiMessages(c, current) {
  const out = [];
  const sys = systemText(c);
  if (sys) out.push({role: "system", content: sys});
  const from = c.summary ? c.summary.upTo : 0;
  c.messages.forEach((m, i) => {
    if (i < from || m === current) return;
    if (m.role === "user") {
      const imgs = (m.images || []).filter((x) => x.url);
      const text = userText(m);
      out.push({role: "user", content: imgs.length ? [{type: "text", text}, ...imgs.map((x) => ({type: "image_url", image_url: {url: x.url}}))] : text});
    } else {
      out.push(...assistantMessages(m));
    }
  });
  return out;
}
function builtinTools(c) {
  if (!builtins) return [];
  const out = [];
  if (settings.search) out.push("web_search", "fetch_url");
  if (c.messages.some((m) => (m.docs || []).length)) out.push("read_document", "search_document");
  return out;
}

// ------------------------------------------------------------------ the context meter and compaction
function ctxTokens(c) {
  if (!c || !c.messages.length) return 0;
  if (c.ctx && c.ctx.tokens) return c.ctx.tokens;
  return estTokens(systemText(c)) + c.messages.reduce((n, m) => n + estTokens(m.role === "user" ? userText(m) : m.text), 0);
}
function renderCtx() {
  const btn = $("ctx-btn");
  const max = health.max_context;
  if (!max || !chat || !chat.messages.length) { btn.hidden = true; return; }
  const used = ctxTokens(chat), pct = Math.min(100, (used / max) * 100);
  btn.hidden = false;
  $("ctx-fill").style.strokeDashoffset = String(94.25 * (1 - pct / 100));
  $("ctx-pct").textContent = pct >= 10 ? Math.round(pct) : "";
  btn.classList.toggle("is-warn", pct >= settings.compactAt - 10 && pct < settings.compactAt);
  btn.classList.toggle("is-full", pct >= settings.compactAt);
  btn.title = `Context: about ${fmt(used)} of ${fmt(max)} tokens (${fmt(pct)}%)`;
}
$("ctx-btn").onclick = (e) => {
  e.stopPropagation();
  const used = ctxTokens(chat), max = health.max_context;
  openMenu(e.currentTarget, [
    {label: `About ${fmt(used)} of ${fmt(max)} tokens`},
    {text: "Compact now", icon: "compress", disabled: !!busy || chat.messages.length < 2, run: () => compact(chat, "by hand")},
    {text: settings.autoCompact ? `Compacts by itself at ${settings.compactAt}%` : "Automatic compaction is off", icon: "settings",
     run: () => { fillSettings(); showPane("context"); $("settings").showModal(); }},
  ]);
};
const SUMMARY_PROMPT = "You compress a conversation so it can go on in a smaller context. Write a dense summary of everything " +
  "the assistant needs to continue it well: the user's goals, preferences and constraints; facts, numbers, names and decisions; " +
  "code, commands and file names (keep exact identifiers and short essential snippets); documents that were attached (with their ids); " +
  "what was being worked on last and what is still open. Use short bullet points under a few headings. No commentary, no preamble.";
function transcript(c, from, to) {
  const cap = (s, n) => (s.length > n ? s.slice(0, n) + `\n[... ${fmt(s.length - n)} more characters]` : s);
  const out = [];
  for (let i = from; i < to; i++) {
    const m = c.messages[i];
    if (m.role === "user") out.push(`User: ${cap(userText(m), 6000)}`);
    else {
      const tools = (m.tools || []).filter((t) => t.result != null).map((t) => `[tool ${t.tool || t.name}(${JSON.stringify(t.arguments || {})}) -> ${cap(t.result, 1200)}]`);
      out.push(`Assistant: ${tools.length ? tools.join("\n") + "\n" : ""}${cap(m.text || "", 6000)}`);
    }
  }
  return out.join("\n\n");
}
async function compact(c, why) {
  if (busy) return false;
  const max = health.max_context || 32768;
  const msgs = c.messages, from = c.summary ? c.summary.upTo : 0;
  // keep the last one or two exchanges as they are (if they fit in 30% of the context); summarise what is before them
  const users = msgs.map((m, i) => (m.role === "user" && i >= from ? i : -1)).filter((i) => i >= 0);
  let keepFrom = msgs.length, kept = 0;
  for (let j = users.length - 1; j >= 0 && kept < 2; j--) {
    const tok = msgs.slice(users[j]).reduce((n, m) => n + estTokens(m.role === "user" ? userText(m) : m.text), 0);
    if (tok > 0.3 * max) break;
    keepFrom = users[j]; kept++;
  }
  if (keepFrom <= from) keepFrom = msgs.length;
  if (keepFrom <= from) { toast("Nothing to compact yet"); return false; }
  let text = transcript(c, from, keepFrom);
  const room = Math.floor(0.55 * max * 3.6);                     // the summary request must fit in the context itself
  if (text.length > room) text = "[... the oldest part is left out; the earlier summary covers it]\n\n" + text.slice(text.length - room);
  const prompt = (c.summary ? `Summary so far:\n${c.summary.text}\n\n` : "") + `Conversation to add to it:\n\n${text}`;
  const controller = new AbortController();
  busy = {controller, msg: null, chat: c, compacting: true};
  if (chat === c) renderChat();
  const before = ctxTokens(c);
  let summary = "";
  try {
    const r = await fetch("v1/chat/completions", {method: "POST", headers: headers(true), signal: controller.signal, body: JSON.stringify({
      model: health.model || "slender", stream: false, reasoning_effort: "none", max_tokens: 1500,
      messages: [{role: "system", content: SUMMARY_PROMPT}, {role: "user", content: prompt}]})});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || `HTTP ${r.status}`);
    summary = ((j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) || "").trim();
    if (!summary) throw new Error("the model wrote no summary");
  } catch (e) {
    busy = null;
    if (chat === c) renderChat();
    if (e.name !== "AbortError") toast("Compaction failed", e.message || String(e), {kind: "error", ms: 7000});
    return false;
  }
  busy = null;
  c.summary = {text: summary, upTo: keepFrom, at: Date.now(), tokensBefore: before, why};
  c.ctx = {tokens: estTokens(systemText(c)) + msgs.slice(keepFrom).reduce((n, m) => n + estTokens(m.role === "user" ? userText(m) : m.text), 0)};
  persist(c);
  if (chat === c) renderChat();
  toast("Chat compacted", `About ${fmt(before)} → ${fmt(c.ctx.tokens)} tokens.`);
  return true;
}
function needsCompaction(c, extraText) {
  const max = health.max_context;
  if (!settings.autoCompact || !max || c.messages.length < 2) return false;
  return ctxTokens(c) + estTokens(extraText) + 1024 > (settings.compactAt / 100) * max;
}

// ------------------------------------------------------------------ sending
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
  const reading = attachments.find((a) => a.busy);
  if (reading) { toast("Still reading a file", `${reading.name} is being read; send again in a moment.`); return; }
  const m = {role: "user", text, images: attachments.filter((a) => a.kind === "image"), time: Date.now(),
             files: attachments.filter((a) => a.kind === "file").map(({kind, busy: _, ...f}) => f),
             docs: attachments.filter((a) => a.kind === "doc").map(({kind, busy: _, ...d}) => d)};
  const c = chat;
  if (needsCompaction(c, userText(m))) {
    const ok = await compact(c, "automatic");
    if (!ok && busy) return;
  }
  if (isNew(c)) {
    c.title = titleFrom(text || (m.files[0] && m.files[0].name) || (m.docs[0] && m.docs[0].name) || "Pictures");
    chats.push(c);
    ls.set("current", c.id);
  }
  c.messages.push(m);
  if (c.ctx) c.ctx.tokens += estTokens(userText(m));
  attachments = []; renderAttachments();
  $("input").value = ""; autosize();
  return generateReply(c);
}
function regenerate(m) {
  if (busy) return;
  const i = chat.messages.indexOf(m);
  if (i < 0 || i !== chat.messages.length - 1) return;
  chat.messages.splice(i, 1);
  generateReply(chat);
}

async function generateReply(c, retried = false) {
  const m = {role: "assistant", text: "", reasoning: "", time: Date.now()};
  c.messages.push(m);
  c.updated = Date.now();
  persist(c);
  renderSidebar();
  const controller = new AbortController();
  busy = {controller, msg: m, chat: c};
  if (chat === c) renderChat();
  const elNow = () => (chat === c ? elOf.get(m) : null);
  if (elNow()) { elNow()._twA = SlenderTyping.newState(0); elNow()._th = {len: 0, live: [], n: 0}; }

  const body = {model: health.model || "slender", messages: apiMessages(c, m), stream: true, reasoning_effort: settings.effort};
  if (!settings.card) {
    if (!(+settings.temperature > 0)) body.temperature = 0;
    else Object.assign(body, {temperature: +settings.temperature, top_p: +settings.top_p, top_k: +settings.top_k});
  }
  if (settings.seed !== "" && settings.seed != null) body.seed = +settings.seed;
  if (settings.max) body.max_tokens = +settings.max;
  if (settings.mcp !== false && mcpInfo.tools > 0) body.strata_mcp = true;
  const own = builtinTools(c);
  if (own.length) body.strata_builtin = own;

  let firstAt = null, thinkStart = null, usage = null, timings = null, lastSave = performance.now();
  // paints follow the screen's frames, spaced out when one gets slow: a starved CPU paints less often, not later.
  // A timer backs the frame request up: a browser can stop sending frames while the page still counts as visible
  // (a covered window, a starved compositor), and the answer must still appear as it is written.
  let frame = 0, fallback = 0, timer = 0, nextAt = 0, cost = 0;
  const arm = () => { frame = requestAnimationFrame(run); fallback = setTimeout(run, 250); };
  const run = () => {
    if (frame) { cancelAnimationFrame(frame); frame = 0; }
    if (fallback) { clearTimeout(fallback); fallback = 0; }
    const t0 = performance.now();
    if (t0 < nextAt) { timer = setTimeout(() => { timer = 0; arm(); }, nextAt - t0); return; }
    const el = elNow();
    if (el) { updateAssistant(el, m, true); scrollDown(); }
    const t1 = performance.now();
    cost = cost * 0.7 + (t1 - t0) * 0.3;
    nextAt = t1 + Math.min(220, cost * 3);
    if (t1 - lastSave > 5000) { lastSave = t1; persist(c); }
  };
  const schedule = () => { if (!frame && !fallback && !timer) arm(); };
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
        schedule();
      }
    }
  } catch (e) {
    if (e.name === "AbortError") m.stopped = true;
    else m.error = e.message || String(e);
  }
  if (frame) cancelAnimationFrame(frame);
  if (fallback) clearTimeout(fallback);
  if (timer) clearTimeout(timer);
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
  if (usage && usage.prompt_tokens) c.ctx = {tokens: usage.prompt_tokens + (usage.completion_tokens || 0)};
  busy = null;
  c.updated = Date.now();
  persist(c);
  const el = elNow();
  if (el) { updateAssistant(el, m, false); scrollDown(); }
  setBusyUi();
  renderSidebar();
  renderCtx();
  // a conversation that no longer fits: compact it and answer again, once
  if (m.error && !retried && settings.autoCompact && /too long|context|exceed|n_ctx|tokens? (limit|budget)/i.test(m.error) && c.messages.length > 2) {
    c.messages.splice(c.messages.indexOf(m), 1);
    if (await compact(c, "the conversation no longer fit")) return generateReply(c, true);
    c.messages.push(m);
    if (chat === c) renderChat();
  }
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

function readAs(f, how) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = () => reject(r.error || new Error("the browser could not read the file"));
    r[how](f);
  });
}
// a document (PDF, Word, Excel, PowerPoint, a long text...) goes to the server, which keeps its text for the model
async function uploadFile(f) {
  const att = {kind: "doc", name: f.name, busy: true};
  attachments.push(att); renderAttachments();
  try {
    if (f.size > 64e6) throw new Error(`${f.name} is over 64 MB`);
    const data = String(await readAs(f, "readAsDataURL"));
    const r = await fetch("v1/files", {method: "POST", headers: headers(true),
      body: JSON.stringify({name: f.name, data: data.slice(data.indexOf(",") + 1), inline_chars: INLINE_CHARS})});
    const j = await r.json();
    if (!r.ok) throw new Error((j.error && j.error.message) || `HTTP ${r.status}`);
    if (j.text != null) Object.assign(att, {kind: "file", text: j.text, id: j.id, pages: j.pages});
    else Object.assign(att, {kind: "doc", id: j.id, pages: j.pages, chars: j.chars, type: j.kind});
    att.busy = false;
  } catch (e) {
    attachments = attachments.filter((a) => a !== att);
    toast("Could not read the file", `${f.name}: ${e.message || e}`, {kind: "error", ms: 7000});
  }
  renderAttachments();
}
async function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/") && health.images) {
      if (f.size > 20e6) { toast("Picture too large", `${f.name} is over 20 MB.`); continue; }
      attachments.push({kind: "image", name: f.name || "pasted image", url: String(await readAs(f, "readAsDataURL"))});
      renderAttachments();
      continue;
    }
    if (builtins) { uploadFile(f); continue; }
    // a server without the file tools: text files go into the message, as before
    const textLike = f.type.startsWith("text/") || /json|xml|javascript|yaml|toml|x-sh|x-python/.test(f.type) || TEXT_EXT.test(f.name);
    if (!textLike) { toast("Not supported here", `${f.name}: this server cannot read documents; text files and pictures work.`); continue; }
    if (f.size > 512 * 1024) { toast("File too large", `${f.name} is over 512 KB.`); continue; }
    const text = String(await readAs(f, "readAsText"));
    if (text.includes("\u0000")) { toast("Not a text file", `${f.name} looks like a binary file.`); continue; }
    attachments.push({kind: "file", name: f.name, text});
    renderAttachments();
  }
}
function renderAttachments() {
  const box = $("attachments");
  box.hidden = !attachments.length;
  box.replaceChildren();
  attachments.forEach((a) => {
    const el = document.createElement("div");
    el.className = `att${a.busy ? " att--busy" : ""}`;
    const extra = a.busy ? "" : a.kind === "doc" ? `<small>${fmt(a.pages)} parts · read by tools</small>` : a.pages > 1 ? `<small>${fmt(a.pages)} parts</small>` : "";
    el.innerHTML = (a.kind === "image" ? `<img src="${esc(a.url)}" alt="">` : icon("file")) + `<span>${esc(a.name)}</span>${extra}` +
      `<button type="button" aria-label="Remove ${esc(a.name)}">${icon("x", "ico--sm")}</button>`;
    el.querySelector("button").onclick = () => { attachments = attachments.filter((x) => x !== a); renderAttachments(); };
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

// ------------------------------------------------------------------ web search
function searchStatusText() {
  if (!builtins) return ["is-off", "This server has no web search (it was started without the web app's tools)."];
  if (searchInfo.up) return ["is-on", `SearXNG is running at ${searchInfo.url}.`];
  if (searchInfo.starting) return ["is-busy", "Starting SearXNG…"];
  if (searchInfo.error) return ["is-off", searchInfo.error];
  return ["", searchInfo.can_start ? "SearXNG is not running; switching search on starts it." :
    `SearXNG does not answer at ${searchInfo.url || "its address"}. Start it yourself, or set "searxng_start" in the run config.`];
}
function renderSearch() {
  const b = $("search-btn");
  b.hidden = !builtins;
  b.setAttribute("aria-pressed", String(!!settings.search));
  b.title = settings.search ? (searchInfo.up ? "Web search is on" : "Web search is on, but SearXNG is not running yet") : "Web search is off";
  setSwitch("s-search", settings.search);
  $("s-search").disabled = !builtins;
  const [cls, text] = searchStatusText();
  $("search-status").innerHTML = `<span class="dot ${cls}"></span><span>${esc(text)}</span>`;
}
async function loadSearch() {
  try {
    const r = await fetch("v1/search", {headers: headers(), cache: "no-store"});
    builtins = r.ok;
    if (r.ok) searchInfo = await r.json();
  } catch (e) { builtins = false; }
  renderSearch();
}
let searchStarting = null;
async function ensureSearch() {
  if (searchStarting) return searchStarting;
  searchStarting = (async () => {
    try {
      let r = await fetch("v1/search/start", {method: "POST", headers: headers(true), body: "{}"});
      if (r.ok) searchInfo = await r.json();
      renderSearch();
      for (let i = 0; i < 30 && !searchInfo.up && (searchInfo.starting || i === 0) && settings.search; i++) {
        await new Promise((res) => setTimeout(res, 1500));
        r = await fetch("v1/search", {headers: headers(), cache: "no-store"});
        if (r.ok) searchInfo = await r.json();
        renderSearch();
      }
      if (settings.search && !searchInfo.up) toast("Web search is not running", searchStatusText()[1], {kind: "error", ms: 8000});
    } catch (e) { /* the status line says it */ }
    searchStarting = null;
  })();
  return searchStarting;
}
function setSearch(on) {
  settings.search = !!on; saveSettings(); renderSearch();
  if (on && !searchInfo.up) ensureSearch();
}
$("search-btn").onclick = () => setSearch(!settings.search);

// ------------------------------------------------------------------ settings window
function setSwitch(id, on) { $(id).setAttribute("aria-checked", String(!!on)); }
function bindSwitch(id, key, after) {
  $(id).onclick = () => { settings[key] = !settings[key]; setSwitch(id, settings[key]); saveSettings(); if (after) after(); };
}
// ------------------------------------------------------------------ appearance: theme, font, corners, text size
// (index.html applies the saved look before the first paint; this keeps it in step with the settings)
const lightScheme = matchMedia("(prefers-color-scheme: light)");
function applyAppearance() {
  const d = document.documentElement;
  const theme = settings.theme === "system" ? (lightScheme.matches ? "light" : "black") : settings.theme;
  const changed = d.dataset.theme !== theme;
  d.dataset.theme = theme;
  d.dataset.font = settings.font;
  d.dataset.corners = settings.corners;
  d.dataset.size = settings.textSize;
  if (changed) document.dispatchEvent(new CustomEvent("slender:theme", {detail: theme}));
}
lightScheme.addEventListener("change", () => { if (settings.theme === "system") applyAppearance(); });
function markChoice(id, value) {
  for (const b of $(id).querySelectorAll("[data-v]")) b.setAttribute("aria-checked", String(b.dataset.v === value));
}
function bindChoice(id, key) {
  $(id).addEventListener("click", (e) => {
    const b = e.target.closest("[data-v]");
    if (!b) return;
    settings[key] = b.dataset.v; saveSettings(); markChoice(id, settings[key]); applyAppearance();
  });
}
bindChoice("s-theme", "theme");
bindChoice("s-font", "font");
bindChoice("s-corners", "corners");
bindChoice("s-size", "textSize");
function fillSettings() {
  markChoice("s-theme", settings.theme);
  markChoice("s-font", settings.font);
  markChoice("s-corners", settings.corners);
  markChoice("s-size", settings.textSize);
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
  setSwitch("s-compact", settings.autoCompact);
  $("s-compact-at").value = settings.compactAt;
  $("o-compact").textContent = `${settings.compactAt}%`;
  $("s-key").value = apiKey;
  renderSearch();
  const facts = [["Model", health.model || "–"], ["Context", health.max_context ? `${fmt(health.max_context)} tokens` : "–"],
                 ["Pictures", health.images ? "Yes" : "No"], ["Loaded", health.loaded == null ? "–" : health.loaded ? "Yes" : "No (loads on the first message)"],
                 ["Documents and search", builtins ? "Yes" : "No"], ["Chats saved", `${chats.length}`]];
  $("facts").innerHTML = facts.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
}
function showPane(name) {
  for (const b of $("settings-nav").children) b.setAttribute("aria-selected", String(b.dataset.pane === name));
  for (const p of document.querySelectorAll("#settings .pane")) p.hidden = p.dataset.pane !== name;
  if (name === "search") loadSearch();
}
$("settings-nav").onclick = (e) => { const b = e.target.closest("button[data-pane]"); if (b) showPane(b.dataset.pane); };
$("open-settings").onclick = () => { fillSettings(); $("settings").showModal(); closeSideOnMobile(); };
$("settings").addEventListener("click", (e) => { if (e.target === $("settings")) $("settings").close(); });   // a click on the backdrop
$("proj-dialog").addEventListener("click", (e) => { if (e.target === $("proj-dialog")) $("proj-dialog").close(); });
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
bindSwitch("s-compact", "autoCompact", renderCtx);
$("s-compact-at").oninput = () => { settings.compactAt = +$("s-compact-at").value; $("o-compact").textContent = `${settings.compactAt}%`; saveSettings(); renderCtx(); };
$("s-search").onclick = () => setSearch(!settings.search);
$("s-key").onchange = () => {
  apiKey = $("s-key").value.trim();
  toast(ls.set("apikey", apiKey) ? "API key saved" : "API key used for this visit only", ls.get("apikey") === apiKey ? "Kept in this browser only." : "Browser storage is unavailable.");
  loadHealth(); loadMcp(); loadSearch();
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
$("s-export").onclick = () => {
  const data = {app: "SlenderEngine", version: 1, exported: new Date().toISOString(), projects, chats};
  download(`slender-chats-${new Date().toISOString().slice(0, 10)}.json`, JSON.stringify(data), "application/json");
};
$("s-import").onclick = () => $("import-input").click();
$("import-input").onchange = async (e) => {
  const f = e.target.files[0];
  e.target.value = "";
  if (!f) return;
  let data;
  try { data = JSON.parse(await readAs(f, "readAsText")); } catch (err) { toast("Not an export file", "The file is not JSON.", {kind: "error"}); return; }
  if (!data || !Array.isArray(data.chats)) { toast("Not an export file", "It has no chats list.", {kind: "error"}); return; }
  let added = 0, replaced = 0, ps = 0;
  for (const p of Array.isArray(data.projects) ? data.projects : []) {
    if (!p || typeof p.id !== "string" || typeof p.name !== "string" || projects.some((x) => x.id === p.id)) continue;
    const np = {id: p.id, name: p.name, instructions: typeof p.instructions === "string" ? p.instructions : "", created: +p.created || Date.now()};
    projects.push(np); store.putProject(np); ps++;
  }
  for (const c of data.chats) {
    if (!c || typeof c.id !== "string" || !Array.isArray(c.messages)) continue;
    const clean = {id: c.id, title: String(c.title || "Imported chat"), projectId: typeof c.projectId === "string" ? c.projectId : null,
                   pinned: !!c.pinned, created: +c.created || Date.now(), updated: +c.updated || Date.now(),
                   messages: c.messages.filter((m) => m && (m.role === "user" || m.role === "assistant")),
                   ...(c.summary && typeof c.summary.text === "string" ? {summary: c.summary} : {})};
    const i = chats.findIndex((x) => x.id === c.id);
    if (i < 0) { chats.push(clean); added++; }
    else if (clean.updated > chats[i].updated && !(busy && busy.chat === chats[i])) { chats[i] = clean; replaced++; }
    else continue;
    store.putChat(clean);
  }
  renderSidebar(); fillSettings();
  toast("Imported", `${added} new chat${added === 1 ? "" : "s"}, ${replaced} updated, ${ps} project${ps === 1 ? "" : "s"}.`);
};

// ------------------------------------------------------------------ the server's state
function paintDot() {
  const dot = $("model-dot");
  const cls = busy ? "dot is-busy" : health.loaded === false ? "dot" : health.ok === false ? "dot is-off" : health.ok ? "dot is-on" : "dot";
  if (dot.className !== cls) dot.className = cls;
  $("model-pill").title = health.ok === false ? "The server does not answer" : busy ? "Answering" :
    health.loaded === false ? "Not loaded: it loads on the first message" : "Loaded and ready";
}
async function loadHealth() {
  try {
    const r = await fetch("health", {headers: headers(), cache: "no-store"});
    const h = await r.json();
    health = {...h, ok: true};
    if ($("model-name").textContent !== (h.model || "Model")) $("model-name").textContent = h.model || "Model";
  } catch (e) {
    health = {...health, ok: false};
    $("model-name").textContent = "Server not reachable";
  }
  paintDot();
  renderCtx();
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
  document.dispatchEvent(new CustomEvent("slender:tab", {detail: name}));
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
$("new-project").onclick = () => projectDialog(null);
let searchTimer = 0;
$("chat-search").oninput = (e) => {                 // searching every chat's text: wait for a pause in typing
  searchQuery = e.target.value;
  clearTimeout(searchTimer);
  searchTimer = setTimeout(renderSidebar, 120);
};
document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key.toLowerCase() === "o") { e.preventDefault(); startChat(); }
});

// ------------------------------------------------------------------ start
async function start() {
  SlenderIcons.fill();
  applyAppearance();
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
  await Promise.all([loadHealth(), loadMcp(), loadSearch()]);
  if (settings.search && builtins && !searchInfo.up) ensureSearch();
  // the model's state every 5 s; not while the tab is in the background
  setInterval(() => { if (!document.hidden) loadHealth(); }, 5000);
}
start();
