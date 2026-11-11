"""tools/test_tailscale_share.py - the Tailscale sharing tool without Tailscale: its config changes, its reading of
`tailscale serve status`, and that it leaves other mappings alone.

    python tools/test_tailscale_share.py
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tailscale_share as ts  # noqa: E402

NAME = "pc.example-tailnet.ts.net"


class Config(unittest.TestCase):
    def test_changes_are_added_once(self):
        new, said, key = ts.config_changes({"allowed_hosts": "relay"}, NAME, 8443, want_key=True)
        self.assertEqual(new["allowed_hosts"], ["relay", NAME])
        self.assertEqual(new["trusted_origins"], [f"https://{NAME}:8443"])
        self.assertTrue(key.startswith("sk-slender-") and len(key) > 30)
        self.assertEqual(len(said), 3)
        again, said2, key2 = ts.config_changes(new, NAME, 8443, want_key=True)
        self.assertEqual((again, said2, key2), (new, [], None))           # nothing twice, the key is kept

    def test_no_key_and_port_443(self):
        new, _, key = ts.config_changes({}, NAME, 443, want_key=False)
        self.assertIsNone(key)
        self.assertNotIn("api_key", new)
        self.assertEqual(new["trusted_origins"], [f"https://{NAME}"])


SERVE = {"TCP": {"443": {"HTTPS": True}, "8443": {"HTTPS": True}},
         "Web": {f"{NAME}:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:7860"}}},
                 f"{NAME}:8443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8080"}}},
                 "other.example-tailnet.ts.net:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:1"}}}}}


class Serve(unittest.TestCase):
    def test_reads_only_this_pc(self):
        with mock.patch.object(ts, "run", return_value=(0, json.dumps(SERVE))):
            m = ts.serve_map("tailscale", NAME)
        self.assertEqual(m, {443: {"/": "http://127.0.0.1:7860"}, 8443: {"/": "http://127.0.0.1:8080"}})

    def run_cmd(self, argv, mapping):
        calls = []
        state = {"running": True, "name": NAME, "magicdns": True, "https": True, "state": "Running"}

        def fake_run(argv, timeout=30.0):
            calls.append(argv)
            return 0, ""
        patches = [mock.patch.object(ts, "find_tailscale", return_value="tailscale"),
                   mock.patch.object(ts, "ts_state", return_value=state),
                   mock.patch.object(ts, "serve_map", return_value=mapping),
                   mock.patch.object(ts, "run", side_effect=fake_run),
                   mock.patch.object(ts, "health", return_value=None)]
        for p in patches:
            p.start()
        try:
            code = ts.main(argv)
        finally:
            for p in patches:
                p.stop()
        return code, calls

    def test_on_refuses_a_port_used_by_something_else(self):
        code, calls = self.run_cmd(["on", "--https-port", "443", "--yes"], {443: {"/": "http://127.0.0.1:7860"}})
        self.assertEqual(code, 1)
        self.assertFalse([c for c in calls if "--bg" in c])

    def test_dry_run_changes_nothing_and_on_writes_once(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Path(d) / "strata-x.json"
            cfg.write_text(json.dumps({"exe": "strata.exe", "port": 8080}, indent=2))
            before = cfg.read_text()
            code, calls = self.run_cmd(["on", "--config", str(cfg), "--dry-run"], {})
            self.assertEqual((code, cfg.read_text(), calls), (0, before, []))
            code, calls = self.run_cmd(["on", "--config", str(cfg), "--yes"], {})
            self.assertEqual(code, 0)
            saved = json.loads(cfg.read_text())
            self.assertIn(NAME, saved["allowed_hosts"])
            self.assertTrue(saved["api_key"].startswith("sk-slender-"))
            self.assertTrue((Path(d) / "strata-x.before-tailscale.json").is_file())
            self.assertIn(["tailscale", "serve", "--bg", "--yes", "--https=8443", "http://127.0.0.1:8080"], calls)

    def test_off_leaves_other_mappings(self):
        code, calls = self.run_cmd(["off", "--https-port", "443"], {443: {"/": "http://127.0.0.1:7860"}})
        self.assertEqual(code, 1)
        self.assertFalse([c for c in calls if "off" in c])
        code, calls = self.run_cmd(["off"], {8443: {"/": "http://127.0.0.1:8080"}})
        self.assertEqual(code, 0)
        self.assertIn(["tailscale", "serve", "--https=8443", "off"], calls)


if __name__ == "__main__":
    unittest.main()
