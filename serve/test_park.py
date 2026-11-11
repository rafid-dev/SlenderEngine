"""Disk parking (serve/park.py): the store's retention rules, the decisions, the queue order, and the hooks in
Service.run, on a fake engine that keeps one conversation and writes it to small session files.

    python -m unittest serve.test_park -v
"""
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.park import BLOCK, CLEAR_MARK, Parker, ParkStore, block_hashes, blocks_shared, common_prefix
from serve.server import ByteTokenizer, MockEngine, Service, SessionRefused

ROOT = Path(__file__).resolve().parents[1]
TURN = 7                                            # the fake chat's turn-start token


class FakeSessionEngine(MockEngine):
    """Holds one conversation (`live`).  A request reuses its common prefix with `live` and reads the rest;
    SAVE / RESTORE write and read `live` with the session file's magic in front."""

    def __init__(self, tok, script="ok"):
        super().__init__(tok, script)
        self.live: list[int] = []
        self.read = []                              # tokens read per request
        self.saves = self.restores = 0
        self.park_epoch = 1
        self.fail_restore = False

    def generate(self, ids, max_new, sampling, cancel, embeddings=None, prefill_only=False):
        reuse = common_prefix(list(ids), self.live)
        self.read.append(len(ids) - reuse)
        if prefill_only:
            self.live = list(ids)
            return
        self.live = list(ids)                       # (kept current token by token: a caller may close us early)
        for t in super().generate(ids, max_new, sampling, cancel, embeddings):
            self.live.append(t)
            yield t

    def session_file(self, action, path):
        if action == "save":
            body = b"STRSESS\x01" + json.dumps(self.live).encode()
            with open(path, "wb") as f:
                f.write(body)
            self.saves += 1
            return {"tokens": len(self.live), "bytes": len(body), "ms": 1.0}
        if self.fail_restore:
            raise SessionRefused("invalid", "session file: payload checksum mismatch", False)
        with open(path, "rb") as f:
            body = f.read()
        if body[:8] != b"STRSESS\x01":
            raise SessionRefused("invalid", "not a session file", False)
        self.live = json.loads(body[8:])
        self.restores += 1
        return {"tokens": len(self.live), "bytes": len(body), "ms": 1.0}


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def convo(seed: int, n: int, opening=()):
    """A token list: an optional shared opening, then tokens unique to `seed`, a turn token every 512."""
    body = [(seed * 7919 + i) % 200 + 20 if i % 512 else TURN for i in range(n)]
    return list(opening) + body


class StoreRules(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.clock = Clock()
        self.free = [10**15]
        self.st = ParkStore(self.dir, "m", budget_gib=1, min_free_gb=0, clock=self.clock,
                            disk_free=lambda d: self.free[0])

    def put(self, fid, tokens, size, kind=None, space="host"):
        open(self.st.path(fid), "wb").write(b"STRSESS\x01")
        self.st.put(fid, space=space, kind=kind or self.st.kind_of(tokens), tokens=tokens, size=size,
                    blocks=block_hashes(list(range(tokens))))

    def test_block_hashes_match_prefixes(self):
        a, b = list(range(3000)), list(range(1000)) + [9] * 2000
        self.assertEqual(blocks_shared(block_hashes(a), block_hashes(b)), 1000 // BLOCK)
        self.assertEqual(common_prefix(a, b), 1000)

    def test_lifetimes_by_kind(self):
        self.put("a" * 16, 1000, 10)                  # sub: 30 min
        self.put("b" * 16, 50000, 10)                 # main: 24 h
        self.put("c" * 16, 3000, 10, kind="prefix")   # prefix: 7 days
        self.put("d" * 16, 50000, 10, space="sandbox")  # sandbox: 30 min whatever the size
        self.clock.t += 31 * 60
        self.st.enforce()
        self.assertEqual(set(self.st.files), {"b" * 16, "c" * 16})
        self.clock.t += 24 * 3600
        self.st.enforce()
        self.assertEqual(set(self.st.files), {"c" * 16})

    def test_budget_drops_largest_idle_first_and_never_protected(self):
        g = 1 << 30
        self.put("a" * 16, 50000, int(0.5 * g))
        self.clock.t += 100
        self.put("b" * 16, 50000, int(0.4 * g))
        self.assertTrue(self.st.room_for(int(0.3 * g), protect={"a" * 16}))
        self.assertIn("a" * 16, self.st.files)
        self.assertNotIn("b" * 16, self.st.files)

    def test_free_space_floor(self):
        self.st.min_free = 100
        self.free[0] = 150
        self.put("a" * 16, 50000, 10)
        self.assertFalse(self.st.room_for(80, protect={"a" * 16}))
        self.assertTrue(self.st.room_for(40))

    def test_write_cap_rolls_daily(self):
        self.st.write_cap = 100
        self.st.note_written(150)
        self.assertFalse(self.st.write_allowed())
        self.clock.t += 86400
        self.assertTrue(self.st.write_allowed())

    def test_sweep_removes_temp_orphans_bad_and_missing(self):
        self.put("a" * 16, 1000, 10)
        self.put("b" * 16, 1000, 10)
        open(os.path.join(self.st.dir, ".x.abc.tmp"), "w").close()
        open(os.path.join(self.st.dir, "f" * 16 + ".ses"), "w").close()      # no index entry
        Path(self.st.path("b" * 16)).write_bytes(b"garbage")
        self.st.files["c" * 16] = {"space": "host", "kind": "sub", "tokens": 1, "bytes": 1, "last_used": self.clock.t}
        magic = lambda p: open(p, "rb").read(8) == b"STRSESS\x01"
        self.st.sweep(magic)
        self.assertEqual(set(self.st.files), {"a" * 16})
        self.assertEqual(sorted(os.listdir(self.st.dir)), sorted(["a" * 16 + ".ses", "index.json"]))

    def test_index_holds_no_token_ids(self):
        self.put("a" * 16, 1000, 10)
        text = Path(self.st.dir, "index.json").read_text()
        self.assertNotIn("[0, 1, 2", text)
        self.assertEqual(set(json.loads(text)["files"]["a" * 16]),
                         {"space", "kind", "tokens", "bytes", "blocks", "created", "last_used"})

    def test_clear_on_stop(self):
        st = ParkStore(self.dir, "private", clear_on_stop=True, min_free_gb=0)
        self.assertTrue(os.path.exists(os.path.join(st.dir, CLEAR_MARK)))
        st.remove_folder()
        self.assertFalse(os.path.exists(st.dir))


class Decisions(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.eng = FakeSessionEngine(self.tok)
        self.svc = Service(self.eng, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.dir = tempfile.mkdtemp()
        self.clock = Clock()
        self.mono = [0.0]
        self.store = ParkStore(self.dir, "m", min_free_gb=0, clock=self.clock)
        self.pk = Parker(self.svc, self.store, turn_token=TURN, log=lambda *a: None, monotonic=lambda: self.mono[0])
        self.svc.parker = self.pk

    def request(self, ids, space="host", images=False):
        """A request as Service.run makes it, without the HTTP side."""
        self.pk.before(ids, space, images=images)
        gen = list(self.eng.generate(ids, 4, {}, threading.Event()))
        self.pk.after(ids, gen, space, images=images)
        return self.eng.read[-1]

    def test_side_request_then_back_restores_instead_of_reading(self):
        main = convo(1, 20000)
        self.request(main)
        side = convo(2, 5000)
        self.request(side)
        self.assertEqual(self.eng.saves, 1)           # main was saved before the side request
        nxt = self.eng.live[:0] + main + [50, 51, 52]
        read = self.request(nxt)
        self.assertEqual(self.eng.restores, 1)
        self.assertLess(read, 300)                     # only what follows the saved part
        self.assertEqual(self.eng.saves, 2)           # the side conversation (5K) was saved when we left it

    def test_short_conversations_are_not_parked(self):
        self.request(convo(1, 2000))
        self.request(convo(2, 2000))
        self.assertEqual(self.eng.saves, 0)

    def test_restore_cost_counts_the_whole_file(self):
        big = convo(1, 150000, opening=[TURN] + [5] * 1023)
        self.request(big)
        self.request(convo(3, 8000))                   # leaves big: saved
        other = [TURN] + [5] * 1023 + convo(4, 3000)   # shares only 1K with the 150K file
        self.request(other)
        self.assertEqual(self.eng.restores, 0)

    def test_restart_restores_from_the_file(self):
        main = convo(1, 20000)
        self.request(main)
        self.request(convo(2, 6000))                    # main saved
        self.eng.park_epoch = 2                          # the engine restarted: it holds nothing
        self.eng.live = []
        read = self.request(main + [60, 61])
        self.assertEqual(self.eng.restores, 1)
        self.assertLess(read, 300)

    def test_a_refused_file_is_dropped_and_the_request_reads(self):
        main = convo(1, 20000)
        self.request(main)
        self.request(convo(2, 6000))
        self.eng.fail_restore = True
        read = self.request(main + [1])
        self.assertGreater(read, 19000)                  # read normally
        self.assertEqual(len([e for e in self.store.files.values() if e["tokens"] >= 20000]), 0)

    def test_images_are_neither_restored_nor_parked(self):
        main = convo(1, 20000)
        self.request(main, images=True)
        self.request(convo(2, 6000))
        self.assertEqual(self.eng.saves, 0)

    def test_sandbox_never_restores_a_host_file(self):
        main = convo(1, 20000)
        self.request(main)
        self.request(convo(2, 6000), space="sandbox")
        self.request(main + [1], space="sandbox")
        self.assertEqual(self.eng.restores, 0)

    def test_shared_opening_saved_once_and_restored_for_new_subagents(self):
        opening = convo(9, 3000) + [TURN]
        self.request(opening + convo(1, 2000))           # first subagent: nothing to share yet
        self.request(opening + convo(2, 2000))           # second: the opening is read alone and saved
        prefixes = [e for e in self.store.files.values() if e["kind"] == "prefix"]
        self.assertEqual(len(prefixes), 1)
        self.request(convo(5, 9000))                     # something else in between
        read = self.request(opening + convo(3, 2000))    # third: restores the opening
        self.assertGreaterEqual(self.eng.restores, 1)
        self.assertLess(read, 2300)

    def test_idle_save_after_a_minute(self):
        main = convo(1, 20000)
        self.request(main)
        self.assertFalse(self.pk.idle_tick())
        self.mono[0] += 61
        self.assertTrue(self.pk.idle_tick())
        self.assertEqual(self.eng.saves, 1)
        self.assertFalse(self.pk.idle_tick())            # up to date: nothing more

    def test_queue_prefers_continuing_then_shortest_then_arrival_cap(self):
        main = convo(1, 20000)
        self.request(main)
        self.mono[0] += 5
        self.pk.released_at = self.mono[0]
        a, b, c = object(), object(), object()
        self.pk.enter(a, convo(2, 9000), "host")         # new, 9K to read
        self.pk.enter(b, convo(3, 600), "host")          # new, 600 to read
        self.pk.enter(c, main + [1, 2, 3], "host")       # continues the held conversation
        self.assertTrue(self.pk.is_next(c))
        self.pk.leave(c)
        self.pk.released_at = self.mono[0]
        self.assertTrue(self.pk.is_next(b))
        self.assertFalse(self.pk.is_next(a))
        self.mono[0] += 25                                # a and b waited past the cap: arrival order
        self.assertTrue(self.pk.is_next(a))

    def test_through_service_run(self):
        """The hooks in Service.run: a side request between two turns costs a save and a restore, not a read."""
        main = convo(1, 20000)
        for ids in (main, convo(2, 6000)):
            list(self.svc.run(ids, False, None, 4, {}, threading.Event()))
        nxt = main + list(self.tok.encode("ok"))[:2] + [70]
        list(self.svc.run(nxt, False, None, 4, {}, threading.Event()))
        self.assertEqual(self.eng.restores, 1)
        self.assertLess(self.eng.read[-1], 300)
        self.assertEqual(self.svc.metrics()["park"]["restores"], 1)


if __name__ == "__main__":
    unittest.main()
