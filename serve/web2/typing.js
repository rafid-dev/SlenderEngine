// serve/web2/typing.js - the typing effect: words that just arrived fade in with a small wobble.
//
// A streaming answer is rebuilt from its Markdown on every frame, so the effect cannot live on the elements: they are new
// each time. Instead each container remembers WHEN each stretch of its visible text first appeared (in offsets of the
// rendered text). After a rebuild, every word younger than the animation is wrapped in <span class="tw"> with a negative
// animation-delay equal to its age, so it carries on from where it was instead of starting over. Older words stay plain
// text, so only the last few dozen words are ever wrapped and long answers cost nothing extra.
"use strict";
(function (root) {
  const LIFE = 520;   // ms: a little longer than the CSS animation (460 ms)
  const SKIP = "[data-math], .code__head, .tool, .katex, button, .wait";

  function textNodes(container) {
    const out = [];
    const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
      acceptNode(n) {
        const p = n.parentElement;
        if (!p || !n.nodeValue) return NodeFilter.FILTER_REJECT;
        return p.closest(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
      },
    });
    for (let n = walker.nextNode(); n; n = walker.nextNode()) out.push(n);
    return out;
  }

  // state: {len, births: [{start, t}]} kept by the caller per container (one per answer, one per thinking block)
  function newState(alreadyShown = 0) { return {len: alreadyShown, births: []}; }

  function apply(container, state, enabled) {
    const now = performance.now();
    const nodes = textNodes(container);
    let total = 0;
    for (const n of nodes) total += n.nodeValue.length;
    if (total > state.len) { state.births.push({start: state.len, t: now}); state.len = total; }
    else if (total < state.len) { state.len = total; state.births = state.births.filter((b) => b.start < total); }
    state.births = state.births.filter((b) => now - b.t < LIFE);
    if (!enabled || !state.births.length) return;
    const births = state.births, aliveFrom = births[0].start;
    const birthAt = (pos) => { let b = null; for (const x of births) { if (x.start <= pos) b = x; else break; } return b; };
    let offset = 0;
    for (const node of nodes) {
      const text = node.nodeValue, start = offset;
      offset += text.length;
      if (offset <= aliveFrom) continue;
      const cut = Math.max(0, aliveFrom - start);
      const frag = document.createDocumentFragment();
      if (cut > 0) frag.append(text.slice(0, cut));
      let pos = start + cut;
      for (const part of text.slice(cut).split(/(\s+)/)) {
        if (!part) continue;
        const b = /^\s+$/.test(part) ? null : birthAt(pos);
        const age = b ? now - b.t : LIFE;
        if (age < LIFE) {
          const span = document.createElement("span");
          span.className = "tw";
          span.style.animationDelay = `${-Math.round(age)}ms`;
          span.textContent = part;
          frag.append(span);
        } else {
          frag.append(part);
        }
        pos += part.length;
      }
      node.replaceWith(frag);
    }
  }

  // The app's own switch decides. Windows reports "reduced motion" whenever its animation effects are off (a common
  // speed tweak), so following the system would hide the effect from people who asked for it. Settings > Interface
  // turns it off.
  const reduced = root.matchMedia ? root.matchMedia("(prefers-reduced-motion: reduce)") : {matches: false};
  const allowed = (setting) => !!setting;

  // the visible text length of a node, counted as apply() counts it
  function textLength(node) { let n = 0; for (const t of textNodes(node)) n += t.nodeValue.length; return n; }
  // `moved` characters left the front of the container (a finished paragraph moved out of the live tail)
  function shift(state, moved) {
    if (!moved) return;
    state.len = Math.max(0, state.len - moved);
    state.births = state.births.map((b) => ({start: Math.max(0, b.start - moved), t: b.t}));
  }

  root.SlenderTyping = {apply, newState, allowed, LIFE, textLength, shift, systemReduced: () => reduced.matches};
})(typeof globalThis !== "undefined" ? globalThis : this);
