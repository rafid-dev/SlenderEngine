"""serve/test_searxng_setup.py - the SearXNG installer (serve/searxng_setup.py) and its HTTP routes, without
downloading or running SearXNG: the settings it writes, the checksum, the archive guard, who may install, the
Docker command sequence, and that a refused install answers at once.

    python -m unittest serve.test_searxng_setup -v
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import searxng_setup as ss  # noqa: E402
from serve.builtin_tools import Builtins, FileStore, Searx  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Settings(unittest.TestCase):
    def test_this_pc_only_json_on_secret_new_each_time(self):
        a, b = ss.settings_yml(8888), ss.settings_yml(8888)
        self.assertIn('bind_address: "127.0.0.1"', a)
        self.assertIn("port: 8888", a)
        self.assertIn("    - json", a)
        self.assertIn("limiter: false", a)
        self.assertNotEqual(a.split("secret_key:")[1].split()[0], b.split("secret_key:")[1].split()[0])
        self.assertIn('bind_address: "0.0.0.0"', ss.settings_yml(8080, bind="0.0.0.0"))   # inside a container only

    def test_pin_is_consistent(self):
        self.assertIn(ss.COMMIT, ss.SOURCE_URL)
        self.assertTrue(ss.VERSION.endswith(ss.COMMIT[:9]))
        self.assertRegex(ss.SOURCE_SHA256, r"^[0-9a-f]{64}$")
        self.assertRegex(ss.IMAGE, r"^searxng/searxng@sha256:[0-9a-f]{64}$")


class Download(unittest.TestCase):
    def test_checksum_mismatch_installs_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            s = ss.SearxngSetup(d, "http://127.0.0.1:1")
            fake = io.BytesIO(b"not the archive")
            fake.__enter__ = lambda *a: fake
            fake.__exit__ = lambda *a: None
            with mock.patch("urllib.request.urlopen", return_value=fake):
                with self.assertRaisesRegex(ss.SetupError, "checksum"):
                    s._install_native()
            self.assertFalse((Path(d) / "src").exists())
            self.assertFalse((Path(d) / "venv").exists())

    def test_archive_cannot_write_outside_its_folder(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("searxng-x/", "")
            z.writestr("searxng-x/searx/webapp.py", "ok")
            z.writestr("searxng-x/../../evil.txt", "no")
            z.writestr("searxng-x/requirements.txt", "")
        data = buf.getvalue()
        with tempfile.TemporaryDirectory() as d:
            s = ss.SearxngSetup(Path(d) / "inst", "http://127.0.0.1:1")
            with mock.patch.object(s, "_download_source", return_value=data), \
                    mock.patch.object(ss, "run", return_value=(1, "stop here")):
                with self.assertRaisesRegex(ss.SetupError, "Python environment"):
                    s._install_native()
            self.assertTrue((Path(d) / "inst" / "src" / "searx" / "webapp.py").is_file())
            self.assertFalse(any(Path(d).rglob("evil.txt")))


class Lifecycle(unittest.TestCase):
    def test_refused_install_answers_at_once(self):
        with tempfile.TemporaryDirectory() as d:
            s = ss.SearxngSetup(d, "http://127.0.0.1:1")
            done = []
            t = threading.Thread(target=lambda: done.append(s.start("native", may_install=False)))
            t.start()
            t.join(5)
            self.assertTrue(done, "start() must not hang when the install is refused")
            self.assertEqual(done[0]["state"], "failed")
            self.assertIn("only from this PC", done[0]["error"])
            with self.assertRaises(ValueError):
                s.start("curl | sh", may_install=True)

    def test_installed_needs_the_same_pin(self):
        with tempfile.TemporaryDirectory() as d:
            s = ss.SearxngSetup(d, "http://127.0.0.1:1")
            (Path(d) / "native.installed").write_text(json.dumps({"commit": "old"}))
            self.assertFalse(s.installed()["native"])
            (Path(d) / "native.installed").write_text(json.dumps({"commit": ss.COMMIT}))
            self.assertTrue(s.installed()["native"])

    def test_docker_commands(self):
        calls = []

        def fake_run(argv, timeout=60.0, **kw):
            calls.append([str(a) for a in argv])
            return 0, ""
        with tempfile.TemporaryDirectory() as d:
            s = ss.SearxngSetup(d, "http://127.0.0.1:8890")
            with mock.patch.object(ss, "run", side_effect=fake_run), mock.patch.object(ss.shutil, "which", return_value="docker"):
                s._install_docker()
                s._launch_docker()
            pull = next(c for c in calls if c[1] == "pull")
            self.assertEqual(pull[2], ss.IMAGE)                               # pinned by digest
            create = next(c for c in calls if c[1] == "create")
            self.assertIn("127.0.0.1:8890:8080", create)                       # this PC only
            self.assertTrue(any(x.startswith("type=bind,source=") and x.endswith(",target=/etc/searxng") for x in create))
            self.assertEqual(calls[-1], ["docker", "start", ss.CONTAINER])
            self.assertIn("    - json", (Path(d) / "docker" / "settings.yml").read_text())


class Routes(unittest.TestCase):
    """Who may install: a request from this PC itself, never one through a proxy (a Tailscale share)."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.dir = tempfile.TemporaryDirectory()
        cls.svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.setup = ss.SearxngSetup(Path(cls.dir.name) / "searxng", "http://127.0.0.1:1")
        cls.svc.builtins = Builtins(FileStore(Path(cls.dir.name) / "files"), Searx("http://127.0.0.1:1", setup=cls.setup))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.svc.telemetry.close()
        cls.dir.cleanup()

    def req(self, path, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})} if body is not None else dict(headers or {})
        r = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None, headers=h)
        try:
            with urllib.request.urlopen(r, timeout=10) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read() or b"{}")

    def test_status_says_who_may_install(self):
        code, j = self.req("/v1/search")
        self.assertEqual(code, 200)
        self.assertTrue(j["from_this_pc"])
        self.assertIn("setup", j)
        self.assertFalse(self.req("/v1/search", headers={"X-Forwarded-For": "100.64.0.9"})[1]["from_this_pc"])
        self.assertFalse(self.req("/v1/search", headers={"Tailscale-User-Login": "a@b.c"})[1]["from_this_pc"])

    def test_install_refused_through_a_proxy(self):
        with mock.patch.object(self.setup, "_work") as work:
            code, j = self.req("/v1/search/start", {"method": "native"}, {"X-Forwarded-For": "100.64.0.9"})
            time.sleep(0.2)
        self.assertEqual(code, 200)
        self.assertEqual(j["setup"]["state"], "failed")
        work.assert_not_called()
        self.assertEqual(self.req("/v1/search/uninstall", {"method": "native"}, {"Tailscale-User-Login": "a@b.c"})[0], 403)
        self.assertEqual(self.req("/v1/search/start", {"method": "native"}, {"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(self.req("/v1/search/start", {"method": "bash"})[0], 400)

    def test_install_starts_from_this_pc(self):
        started = threading.Event()
        with mock.patch.object(self.setup, "_work", side_effect=lambda m: started.set()):
            code, j = self.req("/v1/search/start", {"method": "native"})
            self.assertTrue(started.wait(5))
        self.assertEqual(code, 200)
        self.assertEqual(j["setup"]["method"], "native")


if __name__ == "__main__":
    unittest.main()
