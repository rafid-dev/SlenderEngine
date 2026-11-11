// serve/web2/markdown.js - a small Markdown renderer for the answers (ported from serve/web/app.js and extended).
// The text is escaped first and then formatted, so a model's HTML is shown as text and never runs.
// A fenced block still being written (no closing fence yet) is shown as code as it streams.
"use strict";
(function (root) {
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
  const icon = (n) => (root.SlenderIcons ? root.SlenderIcons.icon(n) : "");

  function inline(s) {
    const codes = [];
    s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return `\u0000${codes.length - 1}\u0000`; });
    s = esc(s)
      .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
      .replace(/__([^_\n]+)__/g, "<strong>$1</strong>")
      .replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, "$1<em>$2</em>")
      .replace(/~~([^~\n]+)~~/g, "<del>$1</del>")
      .replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>')
      .replace(/(^|[\s(])(https?:\/\/[^\s<)]+[^\s<).,;:!?'"])/g, '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>');
    return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code class="inline">${esc(codes[+i])}</code>`);
  }

  function codeBlock(lang, code) {
    return `<div class="code"><div class="code__head"><span>${esc(lang || "code")}</span>` +
      `<button class="icon-btn" type="button" data-copy-code aria-label="Copy code" title="Copy code">${icon("copy")}</button></div>` +
      `<pre><code>${esc(code)}</code></pre></div>`;
  }

  function blocks(text) {
    const out = [], lines = text.split("\n");
    let para = [], list = null;
    const flushPara = () => { if (para.length) out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; };
    const flushList = () => {
      if (list) out.push(`<${list.tag}${list.start > 1 ? ` start="${list.start}"` : ""}>${list.items.map((i) => `<li>${inline(i)}</li>`).join("")}</${list.tag}>`);
      list = null;
    };
    for (let i = 0; i < lines.length; i++) {
      const l = lines[i];
      let m;
      if (!l.trim()) { flushPara(); flushList(); continue; }
      if ((m = l.match(/^(#{1,6})\s+(.*)$/))) {
        flushPara(); flushList();
        const tag = m[1].length === 1 ? "h2" : m[1].length === 2 ? "h3" : "h4";
        out.push(`<${tag}>${inline(m[2].replace(/\s+#+\s*$/, ""))}</${tag}>`);
        continue;
      }
      if (/^\s*([-*_])\s*\1\s*\1[\s\1]*$/.test(l)) { flushPara(); flushList(); out.push("<hr>"); continue; }
      if ((m = l.match(/^>\s?(.*)$/))) { flushPara(); flushList(); out.push(`<blockquote>${inline(m[1])}</blockquote>`); continue; }
      if (/^\s*\|.*\|\s*$/.test(l) && i + 1 < lines.length && /^\s*\|?[\s:-]+\|[\s|:-]*$/.test(lines[i + 1])) {
        flushPara(); flushList();
        const cells = (row) => row.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
        let html = `<table><thead><tr>${cells(l).map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>`;
        i += 2;
        while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) html += `<tr>${cells(lines[i++]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
        i--;
        out.push(html + "</tbody></table>");
        continue;
      }
      if ((m = l.match(/^\s*(?:[-*+]|(\d+)[.)])\s+(.*)$/))) {
        flushPara();
        const tag = m[1] ? "ol" : "ul";
        if (!list || list.tag !== tag) { flushList(); list = {tag, items: [], start: m[1] ? +m[1] : 1}; }
        list.items.push(m[2]);
        continue;
      }
      if (list && /^\s{2,}\S/.test(l)) { list.items[list.items.length - 1] += " " + l.trim(); continue; }
      flushList();
      para.push(l);
    }
    flushPara(); flushList();
    return out.join("");
  }

  function markdown(text) {
    let html = "", rest = text;
    for (;;) {
      const m = rest.match(/(^|\n)[ \t]*```([^\n`]*)\n/);
      if (!m) { html += blocks(rest); break; }
      html += blocks(rest.slice(0, m.index));
      rest = rest.slice(m.index + m[0].length);
      const end = rest.match(/(^|\n)[ \t]*```[ \t]*(\n|$)/);
      if (!end) { html += codeBlock(m[2].trim(), rest); break; }
      html += codeBlock(m[2].trim(), rest.slice(0, end.index));
      rest = rest.slice(end.index + end[0].length);
    }
    return html;
  }

  root.SlenderMarkdown = {markdown, inline, esc};
})(typeof globalThis !== "undefined" ? globalThis : this);
