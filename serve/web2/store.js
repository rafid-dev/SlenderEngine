// serve/web2/store.js - chats and projects, kept in this browser (IndexedDB database "slender": stores "chats" and
// "projects", one record each). The server never sees them. The classic page's database ("strata") is left alone.
// Where the browser keeps nothing (a private window, a blocked store) everything lives in memory for the visit and
// onProblem(kind) says so once: "memory" or "failed".
"use strict";
(function (root) {
  function createStore({onProblem = () => {}} = {}) {
    let db = null, told = false;
    const mem = {chats: new Map(), projects: new Map()};
    const problem = (k) => { if (!told) { told = true; onProblem(k); } };

    function open() {
      return new Promise((resolve) => {
        let r;
        try { r = root.indexedDB.open("slender", 1); } catch (e) { resolve(null); return; }
        r.onupgradeneeded = () => {
          const d = r.result;
          if (!d.objectStoreNames.contains("chats")) d.createObjectStore("chats", {keyPath: "id"});
          if (!d.objectStoreNames.contains("projects")) d.createObjectStore("projects", {keyPath: "id"});
        };
        r.onsuccess = () => resolve(r.result);
        r.onerror = () => resolve(null);
        r.onblocked = () => resolve(null);
      });
    }
    async function start() {
      db = await open();
      if (!db) problem("memory");
      return !!db;
    }
    function all(store) {
      if (!db) return Promise.resolve([...mem[store].values()]);
      return new Promise((resolve) => {
        const req = db.transaction(store, "readonly").objectStore(store).getAll();
        req.onsuccess = () => resolve(req.result || []);
        req.onerror = () => resolve([]);
      });
    }
    function write(store, fn) {
      if (!db) return Promise.resolve(true);
      return new Promise((resolve) => {
        let tx;
        try { tx = db.transaction(store, "readwrite"); fn(tx.objectStore(store)); } catch (e) { problem("failed"); resolve(false); return; }
        tx.oncomplete = () => resolve(true);
        tx.onerror = tx.onabort = () => { problem("failed"); resolve(false); };
      });
    }
    // records are cloned on the way in: a chat being streamed into keeps changing in the page
    const clone = (x) => JSON.parse(JSON.stringify(x));
    return {
      start,
      chats: () => all("chats"),
      projects: () => all("projects"),
      putChat: (c) => { const v = clone(c); mem.chats.set(v.id, v); return write("chats", (s) => s.put(v)); },
      deleteChat: (id) => { mem.chats.delete(id); return write("chats", (s) => s.delete(id)); },
      clearChats: () => { mem.chats.clear(); return write("chats", (s) => s.clear()); },
      putProject: (p) => { const v = clone(p); mem.projects.set(v.id, v); return write("projects", (s) => s.put(v)); },
      deleteProject: (id) => { mem.projects.delete(id); return write("projects", (s) => s.delete(id)); },
    };
  }
  root.SlenderStore = {createStore};
})(typeof globalThis !== "undefined" ? globalThis : this);
