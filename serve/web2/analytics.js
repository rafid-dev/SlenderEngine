// serve/web2/analytics.js - the Analytics tab: decode and prefill dials, the hardware and each disk, with a minute of
// history. Polls GET /v1/live once a second, only while the tab is open and the page is in front. Light on purpose:
// the server samples the hardware anyway (serve/telemetry.py), the dials move with CSS transforms (composited), the
// small graphs are canvases redrawn once a second, and text is only written when it changed.
"use strict";
(function () {
  const $a = (id) => document.getElementById(id);
  const fmtA = (n, d = 0) => (n == null || Number.isNaN(+n) ? "–" : Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d}));
  const gib = (b) => b / 1073741824;
  let open = false, timer = 0, inflight = false, failures = 0;
  let maxDecode = 60, maxPrefill = 1000, prevRead = null;
  const recent = {decode: [], prefill: []};    // what the page saw this visit, for the dials' "peak"

  // ---------------------------------------------------------------- dials
  function niceMax(v, floor) {
    const x = Math.max(floor, v * 1.15);
    const p = Math.pow(10, Math.floor(Math.log10(x)));
    for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (m * p >= x) return m * p;
    return 10 * p;
  }
  function ticks(svg) {
    const g = svg.querySelector(".gauge__ticks");
    if (g.childNodes.length) return;
    let s = "";
    for (let i = 0; i <= 10; i++) {
      const a = Math.PI * (1 - i / 10), r1 = 66, r2 = i % 5 ? 70 : 73;
      s += `<line x1="${100 + r1 * Math.cos(a)}" y1="${106 - r1 * Math.sin(a)}" x2="${100 + r2 * Math.cos(a)}" y2="${106 - r2 * Math.sin(a)}"/>`;
    }
    g.innerHTML = s;
  }
  function setText(el, v) { if (el.textContent !== v) el.textContent = v; }
  // value null: not known yet ("…", the needle stays where it is); held: a past reading shown on purpose (dimmed)
  function dial(id, value, max, foot, held = false, caption = "tok/s") {
    const card = $a(id);
    ticks(card.querySelector("svg"));
    card.classList.toggle("is-held", held);
    setText(card.querySelector(".gauge__value span"), caption);
    if (value == null) {
      setText(card.querySelector(".gauge__value b"), "…");
      const footEl = card.querySelector(".gauge-card__foot");
      const html = foot.map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join("");
      if (footEl._html !== html) { footEl.innerHTML = html; footEl._html = html; }
      return;
    }
    const f = Math.max(0, Math.min(1, (value || 0) / max));
    card.querySelector(".gauge__needle").style.transform = `rotate(${-90 + 180 * f}deg)`;
    const arc = card.querySelector(".gauge__arc");
    arc.style.strokeDasharray = `${(f * 100).toFixed(2)} 100`;
    arc.style.opacity = f > 0.002 ? "1" : "0";        // a round cap would leave a dot at zero
    setText(card.querySelector(".gauge__value b"), value ? fmtA(value, value < 100 ? 1 : 0) : "0");
    setText(card.querySelector(".gauge__max"), fmtA(max));
    const footEl = card.querySelector(".gauge-card__foot");
    const html = foot.map(([k, v]) => `<span>${k} <b>${v}</b></span>`).join("");
    if (footEl._html !== html) { footEl.innerHTML = html; footEl._html = html; }
  }

  // ---------------------------------------------------------------- cards
  const cards = new Map();
  let ink = null;
  function colors() {
    if (ink) return ink;
    const cs = getComputedStyle(document.documentElement);
    ink = {accent: cs.getPropertyValue("--accent").trim() || "#8e97f2", warn: cs.getPropertyValue("--warn").trim() || "#f4c97c",
           line: cs.getPropertyValue("--line").trim() || "#252948"};
    return ink;
  }
  function card(box, key, label, iconName) {
    let c = cards.get(key);
    if (c && c.el.isConnected) return c;
    const el = document.createElement("div");
    el.className = "card";
    el.innerHTML = `<div class="card__label">${SlenderIcons.icon(iconName)}<span></span></div><div class="card__value"></div>` +
      `<div class="card__sub"></div><canvas aria-hidden="true"></canvas>`;
    el.querySelector(".card__label span").textContent = label;
    box.append(el);
    c = {el, value: el.querySelector(".card__value"), sub: el.querySelector(".card__sub"), canvas: el.querySelector("canvas"), stamp: ""};
    cards.set(key, c);
    return c;
  }
  function spark(c, values, max) {
    const pts = (values || []).filter((v) => v != null);
    const stamp = pts.length + ":" + pts.slice(-3).join(",") + ":" + max;
    if (stamp === c.stamp) return;
    c.stamp = stamp;
    const cv = c.canvas, dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = cv.clientWidth, h = cv.clientHeight;
    if (!w || !h) return;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (pts.length < 2) return;
    const top = max || Math.max(1, ...pts) * 1.1, n = 60, step = w / (n - 1), x0 = w - (pts.length - 1) * step;
    const y = (v) => h - 2 - (Math.min(v, top) / top) * (h - 4);
    const col = colors().accent;
    ctx.beginPath();
    pts.forEach((v, i) => (i ? ctx.lineTo(x0 + i * step, y(v)) : ctx.moveTo(x0, y(v))));
    ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.lineJoin = "round"; ctx.stroke();
    ctx.lineTo(x0 + (pts.length - 1) * step, h); ctx.lineTo(x0, h); ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, col + "55"); grad.addColorStop(1, col + "00");
    ctx.fillStyle = grad; ctx.fill();
  }
  function show(c, value, unit, sub, {hot = false, na = false} = {}) {
    const html = value == null ? "–" : `${value}${unit ? `<small>${unit}</small>` : ""}`;
    if (c.value._html !== html) { c.value.innerHTML = html; c.value._html = html; }
    setText(c.sub, sub || "");
    c.el.classList.toggle("is-hot", hot);
    c.el.classList.toggle("is-na", na || value == null);
  }

  // ---------------------------------------------------------------- one reading
  function render(d) {
    const live = d.live || {}, hw = d.hardware || {}, hist = d.history || {}, st = d.static || {}, last = d.last || {};
    // state strip
    const state = live.state || "idle";
    const names = {idle: "Idle", reading: "Reading the prompt", generating: "Writing", unloaded: "Not loaded"};
    let stateText = names[state] || state;
    if (state === "reading" && live.prompt_total) stateText += ` · ${fmtA((100 * (live.prompt_read || 0)) / live.prompt_total)}%`;
    if (live.queued) stateText += ` · ${live.queued} waiting`;
    setText($a("an-state-text"), stateText);
    const dot = $a("an-state").querySelector(".dot");
    const cls = state === "generating" || state === "reading" ? "dot is-busy" : state === "unloaded" ? "dot" : "dot is-on";
    if (dot.className !== cls) dot.className = cls;
    const t = d.totals || {};
    setText($a("an-facts"), [d.model, d.max_context ? `${fmtA(d.max_context)} context` : null,
      t.output_tokens ? `${fmtA(t.output_tokens)} tokens written since start` : null].filter(Boolean).join(" · "));

    // decode
    const lastDecode = last.predicted_ms > 0 && last.predicted_n > 1 ? last.predicted_n / (last.predicted_ms / 1000) : null;
    const decode = state === "generating" ? live.tok_s || 0 : 0;
    if (decode) recent.decode.push(decode);
    const peakD = Math.max(0, ...recent.decode.slice(-600), lastDecode || 0);
    maxDecode = niceMax(Math.max(peakD, maxDecode / 1.15), 30);
    const avgDecode = t.decode_ms > 0 ? t.output_tokens / (t.decode_ms / 1000) : null;
    dial("g-decode", decode, maxDecode, [["Last answer", lastDecode ? `${fmtA(lastDecode, 1)} tok/s` : "–"],
      ["Average", avgDecode ? `${fmtA(avgDecode, 1)} tok/s` : "–"], ...(live.tok_s_mean && state === "generating" ? [["This answer", `${fmtA(live.tok_s_mean, 1)}`]] : [])]);

    // prefill: the prompt read so far over the time between two readings, or the engine's own average
    let prefill = 0;
    if (state === "reading") {
      if (live.prompt_read != null && prevRead && prevRead.total === live.prompt_total && live.prompt_read >= prevRead.read) {
        const dt = (d.time - prevRead.time);
        if (dt > 0.2) prefill = (live.prompt_read - prevRead.read) / dt;
      }
      if (!prefill && live.prefill_tok_s_mean) prefill = live.prefill_tok_s_mean;
      prevRead = live.prompt_read != null ? {read: live.prompt_read, total: live.prompt_total, time: d.time} : null;
    } else prevRead = null;
    if (prefill) recent.prefill.push(prefill);
    const lastPrefill = last.prompt_ms > 0 && last.prompt_n > 0 ? last.prompt_n / (last.prompt_ms / 1000) : null;
    const peakP = Math.max(0, ...recent.prefill.slice(-600), lastPrefill || 0);
    maxPrefill = niceMax(Math.max(peakP, maxPrefill / 1.15), 500);
    const avgPrefill = t.prompt_ms > 0 ? (t.prompt_tokens - (t.reused || 0)) / (t.prompt_ms / 1000) : null;
    // the engine reports reading progress in large steps, so a short prompt often has no rate until it is read:
    // while reading the dial says "…" until a rate is known; afterwards it holds the last prompt's speed (dimmed)
    const footP = [["Last prompt", lastPrefill ? `${fmtA(last.prompt_n)} at ${fmtA(lastPrefill)} tok/s` : "–"],
      ["Average", avgPrefill > 0 ? `${fmtA(avgPrefill)} tok/s` : "–"]];
    if (state === "reading") dial("g-prefill", prefill || null, maxPrefill, footP, false, "tok/s, reading");
    else dial("g-prefill", lastPrefill || 0, maxPrefill, footP, !!lastPrefill, lastPrefill ? "tok/s, last prompt" : "tok/s");

    // hardware
    const box = $a("an-cards");
    let c = card(box, "gpu", "GPU load", "chart");
    show(c, hw.gpu_util != null ? fmtA(hw.gpu_util) : null, "%", st.gpu_name || "GPU");
    spark(c, hist.gpu_util, 100);
    c = card(box, "vram", "VRAM", "layers");
    // NVML reports bytes; the static total (AMD on Windows, Apple) is in MiB
    const vt = hw.gpu_mem_total != null ? hw.gpu_mem_total : st.gpu_vram_total_mib ? st.gpu_vram_total_mib * 1048576 : null;
    show(c, hw.gpu_mem_used != null ? fmtA(gib(hw.gpu_mem_used), 1) : null, vt ? `/ ${fmtA(gib(vt), 1)} GB` : "GB",
         hw.gpu_mem_used != null && vt ? `${fmtA((100 * hw.gpu_mem_used) / vt)}% used` : "");
    spark(c, hist.gpu_mem_used, vt || null);
    c = card(box, "gtemp", "GPU temperature", "sparkle");
    show(c, hw.gpu_temp != null ? fmtA(hw.gpu_temp) : null, "°C", hw.gpu_power != null ? `${fmtA(hw.gpu_power)} W${hw.gpu_power_limit ? ` of ${fmtA(hw.gpu_power_limit)} W` : ""}` : "",
         {hot: hw.gpu_temp >= 83});
    spark(c, hist.gpu_temp, 100);
    c = card(box, "cpu", "CPU load", "settings");
    show(c, hw.cpu != null ? fmtA(hw.cpu) : null, "%", st.cpu_name ? `${st.cpu_name}${st.threads ? ` · ${st.threads} threads` : ""}` : "");
    spark(c, hist.cpu, 100);
    c = card(box, "ctemp", "CPU temperature", "sparkle");
    show(c, hw.cpu_temp != null ? fmtA(hw.cpu_temp) : null, "°C",
         hw.cpu_temp != null ? `from ${hw.cpu_temp_source || "sensors"}` : "needs LibreHardwareMonitor's web server", {hot: hw.cpu_temp >= 90});
    spark(c, hist.cpu_temp, 105);
    c = card(box, "ram", "RAM", "database");
    show(c, hw.ram_used != null ? fmtA(gib(hw.ram_used), 1) : null, hw.ram_total ? `/ ${fmtA(gib(hw.ram_total), 0)} GB` : "GB",
         hw.ram_used != null && hw.ram_total ? `${fmtA((100 * hw.ram_used) / hw.ram_total)}% used` : "", {hot: hw.ram_total && hw.ram_used / hw.ram_total > 0.93});
    spark(c, hist.ram_used, hw.ram_total || null);
    if (hw.gpu_pcie_rx_mb != null) {
      c = card(box, "pcie", "PCIe to the GPU", "arrow-down");
      show(c, fmtA(hw.gpu_pcie_rx_mb / 1024, 2), "GB/s", hw.gpu_pcie_gen ? `Gen ${hw.gpu_pcie_gen} x${hw.gpu_pcie_width || "?"}` : "");
      spark(c, hist.gpu_pcie_rx_mb, null);
    }

    // disks
    const disks = hw.disks || [];
    $a("an-disks-h").hidden = !disks.length;
    const dbox = $a("an-disks");
    for (const dk of disks) {
      c = card(dbox, "disk:" + dk.key, `${dk.name}${dk.letters && dk.letters.length ? ` (${dk.letters.join(" ")})` : ""}`, "database");
      const fill = dk.total ? ` · ${fmtA((100 * dk.used) / dk.total)}% full` : "";
      show(c, fmtA(dk.busy), "% busy", `${dk.model ? dk.model + " · " : ""}R ${fmtA(dk.read_mb, dk.read_mb < 10 ? 1 : 0)} · W ${fmtA(dk.write_mb, dk.write_mb < 10 ? 1 : 0)} MB/s${fill}`);
      spark(c, hist[`disk_rw:${dk.key}`], null);
    }
    const notes = [];
    if (hw.cpu_temp == null) notes.push("CPU temperature: Windows gives none without administrator rights. Run LibreHardwareMonitor and switch on Options > Remote Web Server (port 8085) to see it here.");
    if (st.gpu_note) notes.push(st.gpu_note);
    setText($a("an-note"), notes.join(" "));
  }

  // ---------------------------------------------------------------- polling
  async function poll() {
    timer = 0;
    if (!open || document.hidden) return;
    if (!inflight) {
      inflight = true;
      try {
        const r = await fetch("v1/live", {headers: headers(), cache: "no-store"});
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        render(await r.json());
        failures = 0;
      } catch (e) {
        failures++;
        setText($a("an-state-text"), failures > 2 ? "The server does not answer" : "Connecting…");
        $a("an-state").querySelector(".dot").className = failures > 2 ? "dot is-off" : "dot";
      }
      inflight = false;
    }
    if (open && !document.hidden) timer = setTimeout(poll, failures > 2 ? 3000 : 1000);
  }
  function start() { if (!timer) poll(); }
  function stop() { if (timer) { clearTimeout(timer); timer = 0; } }
  document.addEventListener("slender:tab", (e) => { open = e.detail === "analytics"; if (open) start(); else stop(); });
  document.addEventListener("visibilitychange", () => { if (open && !document.hidden) start(); else stop(); });
})();
