// serve/web2/boot.js - the saved look, applied before the first paint (no flash of another theme). It is a file, not an
// inline script, so the page's Content-Security-Policy can forbid inline scripts (script-src 'self'). app.js keeps the
// look in step with the settings afterwards.
"use strict";
(function () {
  try {
    var s = JSON.parse(localStorage.getItem("slender.settings") || "{}"), d = document.documentElement;
    var t = s.theme || "black";
    if (t === "system") t = matchMedia("(prefers-color-scheme: light)").matches ? "light" : "black";
    d.dataset.theme = t; d.dataset.font = s.font || "system";
    d.dataset.corners = s.corners || "soft"; d.dataset.size = s.textSize || "normal";
  } catch (e) { /* storage blocked: the defaults in the <html> tag stay */ }
})();
