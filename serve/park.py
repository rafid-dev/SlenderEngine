"""serve/park.py - disk parking: conversations the engine let go of, kept on the SSD as session files (opt-in).

The engine holds one conversation.  A request that does not continue it (an agent's subagent, a title, a second
chat) overwrites it, and so does an engine restart; the next turn of the first conversation then reads its whole
history again (a 150K-token history took about two minutes here).  The engine can already write the conversation it
holds to a file and read it back (`SAVE` / `RESTORE`, StrataEngine.session_file: the K/V cells, the running state, the
turn checkpoint and the token ids, checked against the model's fingerprint).  This module decides when:

* before a request that leaves a long conversation, that conversation is saved (unless its file is up to date);
* before a request that a saved file matches better than what the engine holds, by more than the restore costs,
  the file is restored and the engine reads only what follows;
* after a minute idle, a conversation that grew is saved again, so a restart or a crash loses little;
* an opening that several new conversations share (an agent's system prompt and tools) is saved once on its own,
  so a new subagent restores it instead of reading it;
* while several requests wait, the one that continues the held conversation goes first, then the one with the
  least to read (no one waits longer than `wait_cap_s` before plain arrival order takes over).

Files are kept only while they can save time (retention rules R1-R10 in the fork's notes): lifetimes by kind from
the last use, a byte budget, a free-space floor, a daily write cap.  The index holds lengths, times and hashes of
256-token blocks, never token ids or text; the files themselves hold the conversation (its token ids), so the folder
belongs outside every repository.

Nothing here changes what the engine computes: a restored conversation continues exactly as the live one would
(the engine's own session-file tests), and every failure falls back to reading the prompt as before.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from array import array

BLOCK = 256                     # tokens per hashed block (the index matches prefixes at this grain)
INDEX = "index.json"
CLEAR_MARK = ".clear_on_stop"   # a folder with this file is deleted on a clean stop (and by stop scripts)
SPACES = ("host", "sandbox")


def _safe_name(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "model")).strip("._") or "model"
    return s[:80]


def block_hashes(ids, upto: int | None = None) -> list[str]:
    """Chained hashes of the full 256-token blocks of ids[:upto]: block i's hash covers every token before its end."""
    n = len(ids) if upto is None else min(upto, len(ids))
    out, prev = [], b""
    for i in range(0, n - n % BLOCK, BLOCK):
        h = hashlib.blake2b(prev + array("i", ids[i:i + BLOCK]).tobytes(), digest_size=8).digest()
        out.append(h.hex())
        prev = h
    return out


def blocks_shared(a: list[str], b: list[str]) -> int:
    """How many leading blocks two chains share (chained, so the first difference ends it)."""
    lo, hi = 0, min(len(a), len(b))
    while lo < hi:                                  # chained hashes: equal at i means equal before i too
        mid = (lo + hi + 1) // 2
        if a[mid - 1] == b[mid - 1]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def common_prefix(a, b) -> int:
    """Exact common prefix of two token lists (binary search on slice equality, which runs in C)."""
    lo, hi = 0, min(len(a), len(b))
    if hi and a[:hi] == b[:hi]:
        return hi
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


class ParkStore:
    """The files of one model in one folder, their index and the retention rules.  Not thread-safe on its own:
    Parker calls it under its lock."""

    def __init__(self, root: str, model_name: str, *, budget_gib: float = 24.0, min_free_gb: float = 50.0,
                 ttl_sub_s: float = 1800.0, ttl_main_s: float = 86400.0, ttl_prefix_s: float = 7 * 86400.0,
                 ttl_sandbox_s: float = 1800.0, ttl_scale: float = 1.0, write_cap_gb: float = 200.0,
                 sub_tokens: int = 32768, clear_on_stop: bool = False, clock=time.time, disk_free=None):
        self.dir = os.path.join(os.path.abspath(root), _safe_name(model_name))
        self.budget = int(budget_gib * (1 << 30))
        self.min_free = int(min_free_gb * 1e9)
        self.ttl = {"sub": ttl_sub_s * ttl_scale, "main": ttl_main_s * ttl_scale, "prefix": ttl_prefix_s * ttl_scale}
        self.ttl_sandbox = ttl_sandbox_s * ttl_scale
        self.write_cap = int(write_cap_gb * 1e9)
        self.sub_tokens = sub_tokens
        self.clear_on_stop = clear_on_stop
        self.clock = clock
        self.disk_free = disk_free or (lambda d: shutil.disk_usage(d).free)
        os.makedirs(self.dir, exist_ok=True)
        mark = os.path.join(self.dir, CLEAR_MARK)
        if clear_on_stop:
            open(mark, "a").close()
        elif os.path.exists(mark):
            os.remove(mark)
        self.files: dict[str, dict] = {}
        self.day, self.written_today = "", 0
        self.counters = {"saves": 0, "restores": 0, "restore_fail": 0, "save_fail": 0, "dropped": 0,
                         "bytes_written": 0, "bytes_read": 0, "save_ms": 0.0, "restore_ms": 0.0}
        self._load()

    # ---- index ----
    def _load(self) -> None:
        try:
            with open(os.path.join(self.dir, INDEX), encoding="utf-8") as f:
                d = json.load(f)
            if d.get("version") == 1 and isinstance(d.get("files"), dict):
                self.files = {k: v for k, v in d["files"].items()
                              if re.fullmatch(r"[0-9a-f]{16}", k) and isinstance(v, dict)}
                self.day, self.written_today = str(d.get("day", "")), int(d.get("written_today", 0))
        except (OSError, ValueError, TypeError):
            self.files = {}

    def save_index(self) -> None:
        tmp = os.path.join(self.dir, INDEX + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": 1, "day": self.day, "written_today": self.written_today, "files": self.files}, f)
        os.replace(tmp, os.path.join(self.dir, INDEX))

    def path(self, fid: str) -> str:
        return os.path.join(self.dir, fid + ".ses")

    @staticmethod
    def new_id() -> str:
        return secrets.token_hex(8)

    # ---- retention ----
    def kind_of(self, tokens: int, prefix: bool = False) -> str:
        return "prefix" if prefix else ("sub" if tokens < self.sub_tokens else "main")

    def lifetime(self, e: dict) -> float:
        return self.ttl_sandbox if e.get("space") == "sandbox" else self.ttl.get(e.get("kind", "main"), self.ttl["main"])

    def expired(self, e: dict, now: float | None = None) -> bool:
        now = self.clock() if now is None else now
        return now - float(e.get("last_used", 0)) > self.lifetime(e)

    def sweep(self, header_ok=None) -> int:
        """R8: temp files, expired files, files that fail `header_ok(path)`, index entries without a file and files
        without an entry.  -> how many were removed."""
        removed = 0
        try:
            names = os.listdir(self.dir)
        except OSError:
            return 0
        for n in names:
            p = os.path.join(self.dir, n)
            if n.endswith(".tmp") or (n.startswith(".") and n != CLEAR_MARK):
                removed += _rm(p)                       # the engine's own temp files are hidden ".<name>.<hex>.tmp"
            elif n.endswith(".ses") and n[:-4] not in self.files:
                removed += _rm(p)
        for fid in list(self.files):
            e, p = self.files[fid], self.path(fid)
            if not os.path.exists(p) or self.expired(e) or (header_ok is not None and not header_ok(p)):
                self.drop(fid, count=False)
                removed += 1
        self.save_index()
        return removed

    def drop(self, fid: str, count: bool = True) -> None:
        self.files.pop(fid, None)
        _rm(self.path(fid))
        if count:
            self.counters["dropped"] += 1

    def used_bytes(self) -> int:
        return sum(int(e.get("bytes", 0)) for e in self.files.values())

    def enforce(self, protect=(), incoming: int = 0) -> None:
        """R2/R3: drop expired files, then the largest size x idle time until the budget and the free-space floor
        allow `incoming` more bytes.  Never a file in `protect`."""
        now = self.clock()
        for fid in [f for f, e in self.files.items() if f not in protect and self.expired(e, now)]:
            self.drop(fid)

        def over() -> bool:
            if self.used_bytes() + incoming > self.budget:
                return True
            try:
                return self.disk_free(self.dir) - incoming < self.min_free
            except OSError:
                return False
        while over():
            cands = [(int(e.get("bytes", 0)) * max(1.0, now - float(e.get("last_used", 0))), f)
                     for f, e in self.files.items() if f not in protect]
            if not cands:
                break
            self.drop(max(cands)[1])
        self.save_index()

    def room_for(self, incoming: int, protect=()) -> bool:
        self.enforce(protect, incoming)
        try:
            free_ok = self.disk_free(self.dir) - incoming >= self.min_free
        except OSError:
            free_ok = True
        return self.used_bytes() + incoming <= self.budget and free_ok

    # ---- writes (R4) ----
    def _roll_day(self) -> None:
        today = time.strftime("%Y-%m-%d", time.localtime(self.clock()))
        if today != self.day:
            self.day, self.written_today = today, 0

    def write_allowed(self) -> bool:
        self._roll_day()
        return self.written_today < self.write_cap

    def note_written(self, n: int) -> None:
        self._roll_day()
        self.written_today += int(n)
        self.counters["bytes_written"] += int(n)

    # ---- entries ----
    def put(self, fid: str, *, space: str, kind: str, tokens: int, size: int, blocks: list[str]) -> None:
        now = self.clock()
        old = self.files.get(fid, {})
        self.files[fid] = {"space": space, "kind": kind, "tokens": int(tokens), "bytes": int(size),
                           "blocks": blocks, "created": old.get("created", now), "last_used": now}
        self.save_index()

    def touch(self, fid: str) -> None:
        if fid in self.files:
            self.files[fid]["last_used"] = self.clock()
            self.save_index()

    def listing(self) -> list[dict]:
        now = self.clock()
        return [{"id": f, "space": e.get("space"), "kind": e.get("kind"), "tokens": e.get("tokens"),
                 "bytes": e.get("bytes"), "idle_s": round(now - float(e.get("last_used", now)), 1),
                 "expires_in_s": round(self.lifetime(e) - (now - float(e.get("last_used", now))), 1)}
                for f, e in sorted(self.files.items(), key=lambda kv: -float(kv[1].get("last_used", 0)))]

    def clear(self) -> int:
        n = len(self.files)
        for fid in list(self.files):
            self.drop(fid, count=False)
        self.save_index()
        return n

    def remove_folder(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def _rm(p: str) -> int:
    try:
        os.remove(p)
        return 1
    except OSError:
        return 0


class Parker:
    """The decisions.  `svc` is the server's Service (its engine, fifo, status); every engine call happens with the
    caller holding svc.fifo, except idle_tick, which takes it itself without waiting."""

    def __init__(self, svc, store: ParkStore, *, park_min: int = 4096, park_gain: int = 4096, idle_s: float = 60.0,
                 read_ms_per_tok: float = 0.6, restore_ms_base: float = 200.0, restore_us_per_tok: float = 17.0,
                 prefix_min: int = 2048, prefix_max: int = 16384, turn_token: int | None = None,
                 wait_cap_s: float = 20.0, log=print, monotonic=time.monotonic):
        self.svc, self.store = svc, store
        self.park_min, self.park_gain, self.idle_s = park_min, park_gain, idle_s
        self.read_ms, self.restore_base, self.restore_us = read_ms_per_tok, restore_ms_base, restore_us_per_tok
        self.prefix_min, self.prefix_max, self.turn_token = prefix_min, prefix_max, turn_token
        self.wait_cap_s = wait_cap_s
        self.log, self.mono = log, monotonic
        self.lock = threading.RLock()
        self.held: dict | None = None               # {"ids", "space", "epoch", "file", "saved_len", "images"}
        self.held_version = 0
        self.recent_starts: list[tuple[str, list[int]]] = []   # (space, first tokens) of recent new conversations
        self.cv = threading.Condition()
        self.waiting: dict[object, dict] = {}       # queue order (rules 10-11)
        self.released_at = self.mono()
        self.last_done = self.mono()
        self.stop_evt = threading.Event()
        self.thread: threading.Thread | None = None

    # ---- engine identity: a restart or an unload drops what the engine held ----
    def _epoch(self):
        e = self.svc.engine
        tag = getattr(e, "park_epoch", None)
        return tag if tag is not None else id(getattr(e, "proc", None))

    def _held_now(self) -> dict | None:
        h = self.held
        if h is not None and h["epoch"] != self._epoch():
            self.held = None
            self.held_version += 1
            return None
        return h

    # ---- engine calls ----
    def _session(self, action: str, path: str) -> dict:
        """One SAVE / RESTORE, no retries: a refusal (RAM, space, another model) means this park is skipped, never that a
        request waits.  An engine started with --session-save-reclaim frees its older in-RAM checkpoints first when a
        save needs the room (its RAM floor stays), which a parking setup wants: the conversation goes to disk anyway."""
        return self.svc.engine.session_file(action, path)

    def _save_held(self, why: str) -> bool:
        h = self.held
        if h is None or h["images"] or len(h["ids"]) < min(self.park_min, self.prefix_min) or not self.store.write_allowed():
            return False
        st = self.store
        fid = h["file"] or st.new_id()
        est = int(120e6 + 17600 * len(h["ids"]))
        if not st.room_for(est, protect={fid}):
            self.log(f"[strata] park: no room for a {est / 1e9:.1f} GB file (budget or free-space floor); not saved")
            return False
        try:
            r = self._session("save", st.path(fid))
        except Exception as e:  # noqa: BLE001 - parking is an optimization: the request goes on without it
            st.counters["save_fail"] += 1
            self.log(f"[strata] park: save failed ({type(e).__name__}: {e}); continuing without it")
            self._after_failure()
            return False
        n = int(r.get("tokens", len(h["ids"])))
        st.put(fid, space=h["space"], kind=st.kind_of(n), tokens=n, size=int(r.get("bytes", 0)),
               blocks=block_hashes(h["ids"], n))
        st.note_written(int(r.get("bytes", 0)))
        st.counters["saves"] += 1
        st.counters["save_ms"] += float(r.get("ms", 0))
        h["file"], h["saved_len"] = fid, n
        self.log(f"[strata] park: saved {n} tokens ({why}) in {float(r.get('ms', 0)):.0f} ms, "
                 f"{int(r.get('bytes', 0)) / 1e9:.2f} GB")
        return True

    def _restore(self, fid: str, ids, space: str, shared: int) -> bool:
        st = self.store
        try:
            r = self._session("restore", st.path(fid))
        except Exception as e:  # noqa: BLE001
            st.counters["restore_fail"] += 1
            self.log(f"[strata] park: restore failed ({type(e).__name__}: {e}); dropping the file, reading normally")
            st.drop(fid)
            self.held = None
            self.held_version += 1
            self._after_failure()
            return False
        e = st.files.get(fid, {})
        st.touch(fid)
        st.counters["restores"] += 1
        st.counters["restore_ms"] += float(r.get("ms", 0))
        st.counters["bytes_read"] += int(r.get("bytes", 0))
        prefix = e.get("kind") == "prefix"
        self.held = {"ids": list(ids[:shared]), "space": space, "epoch": self._epoch(),
                     "file": None if prefix else fid, "saved_len": int(r.get("tokens", 0)), "images": False}
        self.held_version += 1
        self.log(f"[strata] park: restored {int(r.get('tokens', 0))} tokens ({e.get('kind')}) in "
                 f"{float(r.get('ms', 0)):.0f} ms")
        return True

    def _after_failure(self) -> None:
        """A session exchange can end the engine (out of step, FATAL): start it again for the request that waits."""
        try:
            if not self.svc.loaded():
                self.svc.ensure_loaded()
        except Exception as e:  # noqa: BLE001
            self.log(f"[strata] park: the engine did not start again ({e})")

    # ---- matching ----
    def _best_file(self, ids, space: str, blocks: list[str] | None = None):
        """-> (fid, shared tokens) of the parked file that saves the most time, or (None, 0)."""
        blocks = blocks if blocks is not None else block_hashes(ids)
        best, best_gain, best_shared = None, 0.0, 0
        for fid, e in self.store.files.items():
            if e.get("space") != space or self.store.expired(e):
                continue
            shared = min(blocks_shared(blocks, e.get("blocks", [])) * BLOCK, int(e.get("tokens", 0)), len(ids))
            if shared <= 0:
                continue
            gain = shared * self.read_ms - (self.restore_base + self.restore_us * int(e.get("tokens", 0)) / 1000.0)
            if gain > best_gain:
                best, best_gain, best_shared = fid, gain, shared
        return best, best_shared

    def estimate(self, ids, space: str, timeout: float = -1) -> tuple[int, bool] | None:
        """(tokens left to read, continues the held conversation) - for the queue order.  None when the lock was
        busy for longer than `timeout` (a save or restore running): the caller keeps its last estimate."""
        if not self.lock.acquire(timeout=timeout):
            return None
        try:
            h = self._held_now()
            ll = common_prefix(ids, h["ids"]) if h is not None else 0
            cont = h is not None and ll >= max(0, len(h["ids"]) - self.park_min)
            fid, shared = self._best_file(ids, space)
            return len(ids) - max(ll, shared), cont
        finally:
            self.lock.release()

    # ---- the request path ----
    def before(self, ids, space: str = "host", images: bool = False, cancel=None) -> None:
        """Called holding svc.fifo, the engine loaded, right before a request's prompt goes to the engine."""
        with self.lock:
            if space not in SPACES:
                space = "host"
            h = self._held_now()
            ll = common_prefix(ids, h["ids"]) if h is not None else 0
            leaving = h is not None and ll < len(h["ids"]) - self.park_min
            if leaving and len(h["ids"]) - h["saved_len"] >= self.park_gain:
                self._save_held("switch")
            if images:
                if leaving or h is None:
                    self.held = None
                    self.held_version += 1
                return
            blocks = block_hashes(ids)
            fid, shared = self._best_file(ids, space, blocks)
            restored = False
            if fid is not None and shared - ll > 0:
                gain = (shared - ll) * self.read_ms - (self.restore_base
                                                       + self.restore_us * int(self.store.files[fid]["tokens"]) / 1000.0)
                if gain > 0:
                    restored = self._restore(fid, ids, space, shared)
            if restored:
                return
            # a new conversation (nothing held, or it leaves the held one early, at most an opening's length in):
            # rule 12 may save the opening it shares with an earlier new conversation
            if h is None or (ll < len(h["ids"]) - 512 and ll <= self.prefix_max):
                self._maybe_prefix(ids, space, cancel)

    def _maybe_prefix(self, ids, space: str, cancel) -> None:
        """Rule 12: a new conversation that shares an opening of at least prefix_min tokens with an earlier new one
        gets that opening read on its own and saved, then reads the rest from there."""
        starts = [s for sp, s in self.recent_starts if sp == space]
        self.recent_starts.append((space, list(ids[:32768])))
        del self.recent_starts[:-8]
        # shared with ANOTHER conversation: both go on past the shared part (an earlier turn of this same
        # conversation is a prefix of it, so it never counts)
        shared = [c for c, s in ((common_prefix(ids, s), s) for s in starts) if c < min(len(ids), len(s)) - 1024]
        c = min(max(shared, default=0), self.prefix_max)
        if c < self.prefix_min or self.turn_token is None or not hasattr(self.svc.engine, "session_file"):
            return
        cut = 0
        for i in range(min(c, len(ids)) - 1, self.prefix_min - 1, -1):    # the last turn start inside the shared part
            if ids[i] == self.turn_token:
                cut = i
                break
        if cut < self.prefix_min or cut >= len(ids):
            return
        if any(e.get("kind") == "prefix" and e.get("space") == space
               and blocks_shared(block_hashes(ids, cut), e.get("blocks", [])) * BLOCK >= cut - BLOCK
               for e in self.store.files.values()):
            return
        ev = cancel if cancel is not None else threading.Event()
        try:
            for _ in self.svc.engine.generate(list(ids[:cut]), 0, {}, ev, prefill_only=True):
                pass
        except TypeError:
            return                                      # an engine without prefill_only (tests' mocks)
        except Exception as e:  # noqa: BLE001
            self.log(f"[strata] park: reading the shared opening failed ({e}); continuing")
            self._after_failure()
            return
        self.held = {"ids": list(ids[:cut]), "space": space, "epoch": self._epoch(), "file": None,
                     "saved_len": 0, "images": False}
        self.held_version += 1
        fid = self.store.new_id()
        self.held["file"] = fid
        if self._save_held("shared opening"):
            e = self.store.files.get(fid)
            if e is not None:
                e["kind"] = "prefix"
                self.store.save_index()
        self.held["file"] = None                        # the conversation that follows gets its own file

    def after(self, prompt, generated, space: str = "host", images: bool = False) -> None:
        """A pass ended: the engine holds the prompt and what it generated."""
        with self.lock:
            h = self._held_now()
            ids = list(prompt) + list(generated)
            cont = h is not None and common_prefix(ids, h["ids"]) >= max(0, len(h["ids"]) - self.park_min)
            images = bool(images or (cont and h["images"]))
            keep = cont and h["file"] is not None and not images
            self.held = {"ids": ids, "space": space if space in SPACES else "host", "epoch": self._epoch(),
                         "file": h["file"] if keep else None, "saved_len": h["saved_len"] if keep else 0,
                         "images": images}
            self.held_version += 1
            self.last_done = self.mono()

    def forget(self) -> None:
        with self.lock:
            self.held = None
            self.held_version += 1

    # ---- idle save (rule 5) ----
    def idle_tick(self) -> bool:
        if self.idle_s <= 0 or self.mono() - self.last_done < self.idle_s:
            return False
        h = self.held
        if h is None or h["images"] or len(h["ids"]) < self.park_min or len(h["ids"]) - h["saved_len"] < self.park_gain:
            return False
        if self.waiting or not self.store.write_allowed():
            return False
        svc = self.svc
        if not svc.fifo.acquire(blocking=False):
            return False
        try:
            if not svc.loaded():
                return False
            with self.lock:
                if self._held_now() is None:
                    return False
                return self._save_held("idle")
        finally:
            svc.fifo.release()
            self.notify()

    def start(self) -> None:
        def loop():
            while not self.stop_evt.wait(5.0):
                try:
                    self.idle_tick()
                    if int(self.mono()) % 600 < 5:      # retention every ten minutes
                        with self.lock:
                            self.store.enforce(protect={self.held["file"]} if self.held and self.held["file"] else ())
                except Exception as e:  # noqa: BLE001 - never take the server down
                    self.log(f"[strata] park: background step failed ({e})")
        self.thread = threading.Thread(target=loop, name="park", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.stop_evt.set()
        if self.store.clear_on_stop:
            self.store.remove_folder()

    # ---- queue order (rules 10-11) ----
    def enter(self, key, ids, space: str) -> None:
        with self.cv:
            self.waiting[key] = {"t": self.mono(), "ids": ids, "space": space, "v": -1, "est": (len(ids), False)}

    def leave(self, key) -> None:
        with self.cv:
            self.waiting.pop(key, None)
            self.cv.notify_all()

    def notify(self) -> None:
        with self.cv:
            self.released_at = self.mono()
            self.cv.notify_all()

    def wait(self, timeout: float) -> None:
        with self.cv:
            self.cv.wait(timeout)

    def is_next(self, key) -> bool:
        with self.cv:
            if key not in self.waiting:
                return True
            if len(self.waiting) == 1 or self.mono() - self.released_at > 1.0:
                return True                             # nobody else, or the turn sat free: whoever asks takes it
            now = self.mono()
            items = list(self.waiting.items())
        late = [(w["t"], k) for k, w in items if now - w["t"] > self.wait_cap_s]
        if late:
            return min(late)[1] is key
        order = []
        for k, w in items:
            if w["v"] != self.held_version:
                est = self.estimate(w["ids"], w["space"], timeout=0.05)
                if est is not None:
                    w["est"], w["v"] = est, self.held_version
            left, cont = w["est"]
            order.append((0 if cont else 1, left, w["t"], id(k), k))
        return min(order)[4] is key

    def stats(self) -> dict:
        with self.lock:
            st = self.store
            h = self.held
            return {"dir": st.dir, "files": len(st.files), "bytes": st.used_bytes(), "budget_bytes": st.budget,
                    "written_today": st.written_today, "write_cap": st.write_cap, **st.counters,
                    "held_tokens": len(h["ids"]) if h else 0, "held_saved": h["saved_len"] if h else 0,
                    "waiting": len(self.waiting)}
