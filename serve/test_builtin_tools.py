"""serve/test_builtin_tools.py - the web app's own tools (serve/builtin_tools.py), its file, search and live endpoints,
and the Analytics tab's disk and CPU-temperature readers (serve/telemetry.py). No GPU, no network beyond 127.0.0.1.

    python -m unittest serve.test_builtin_tools -v
"""
from __future__ import annotations

import base64
import http.server
import io
import json
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import builtin_tools as bt  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402
from serve.telemetry import _CpuTemp, _Disks  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def zipped(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


W = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'
S = 'xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"'


def docx(*paras):
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paras)
    return zipped({"word/document.xml": f"<w:document {W}><w:body>{body}</w:body></w:document>"})


def pptx(*slides):
    return zipped({f"ppt/slides/slide{i}.xml": f"<p:sld {A} xmlns:p='p'><a:p><a:r><a:t>{t}</a:t></a:r></a:p></p:sld>"
                   for i, t in enumerate(slides, 1)})


def xlsx():
    return zipped({
        "xl/sharedStrings.xml": f"<sst {S}><si><t>name</t></si><si><t>speed</t></si><si><t>engine</t></si></sst>",
        "xl/workbook.xml": f"<workbook {S}><sheets><sheet name='Runs'/></sheets></workbook>",
        "xl/worksheets/sheet1.xml": f"<worksheet {S}><sheetData><row><c r='A1' t='s'><v>0</v></c><c r='B1' t='s'><v>1</v></c></row>"
                                    f"<row><c r='A2' t='s'><v>2</v></c><c r='C2'><v>42.5</v></c></row></sheetData></worksheet>"})


def tiny_pdf(text: str) -> bytes:
    """A one-page PDF with `text` in Helvetica, offsets computed so readers accept it."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out, offs = bytearray(b"%PDF-1.4\n"), []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    x = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1) + b"".join(b"%010d 00000 n \n" % o for o in offs)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, x)
    return bytes(out)


PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + (640).to_bytes(4, "big") + (480).to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"


class Extract(unittest.TestCase):
    def test_office_and_text_formats(self):
        self.assertIn("Hello world", bt.extract("a.docx", docx("Hello world", "Second line"))["pages"][0])
        got = bt.extract("deck.pptx", pptx("First slide", "Second slide"))
        self.assertEqual(got["kind"], "slides")
        self.assertEqual(got["pages"], ["First slide", "Second slide"])
        sheet = bt.extract("runs.xlsx", xlsx())["pages"][0]
        self.assertIn("Sheet: Runs", sheet)
        self.assertIn("name\tspeed", sheet)
        self.assertIn("engine\t\t42.5", sheet)
        title, text = bt.html_to_text("<html><head><title>T</title><script>x()</script></head><body><h1>Head</h1><p>Body"
                                      "</p><ul><li>one</li></ul></body></html>")
        self.assertEqual(title, "T")
        self.assertNotIn("x()", text)
        self.assertIn("## Head", text)
        self.assertIn("- one", text)
        parts = bt.extract("long.txt", ("word " * 3000).encode())["pages"]
        self.assertGreater(len(parts), 2)
        self.assertTrue(all(len(p) <= bt.PART_CHARS for p in parts))

    def test_pictures_and_binaries(self):
        pic = bt.extract("p.png", PNG)
        self.assertEqual(pic["kind"], "image")
        self.assertEqual((pic["info"]["width"], pic["info"]["height"]), (640, 480))
        with self.assertRaises(bt.ExtractError):
            bt.extract("a.bin", b"\x00\x01\x02" * 100)
        with self.assertRaises(bt.ExtractError):
            bt.extract("broken.docx", b"not a zip")

    def test_pdf(self):
        try:
            import pypdf  # noqa: F401
        except ImportError:
            with self.assertRaisesRegex(bt.ExtractError, "pypdf"):
                bt.extract("a.pdf", tiny_pdf("Hello PDF"))
            return
        got = bt.extract("a.pdf", tiny_pdf("Hello PDF"))
        self.assertEqual(got["kind"], "pdf")
        self.assertIn("Hello PDF", got["pages"][0])


class Store(unittest.TestCase):
    def test_put_get_summary(self):
        with tempfile.TemporaryDirectory() as d:
            fs = bt.FileStore(d)
            doc = fs.put("..\\evil/name.docx", docx("Alpha", "Beta"))
            self.assertEqual(doc["name"], "name.docx")
            self.assertTrue((Path(d) / f"{doc['id']}.json").is_file())
            self.assertEqual(bt.FileStore(d).get(doc["id"])["pages"], doc["pages"])     # read back from disk
            self.assertIsNone(fs.get("../../etc"))
            self.assertIsNone(fs.get("0" * 20))
            s = bt.FileStore.summary(doc, inline_chars=1000)
            self.assertIn("Alpha", s["text"])
            self.assertNotIn("text", bt.FileStore.summary(doc, inline_chars=3))


class FakeSearx(http.server.BaseHTTPRequestHandler):
    pages = {}

    def do_GET(self):
        if self.path.startswith("/healthz"):
            body, ctype = b"OK", "text/plain"
        elif self.path.startswith("/search"):
            body = json.dumps({"results": [{"title": "One", "url": "https://example.com/1", "content": "first  snippet"},
                                           {"title": "Two", "url": "https://example.com/2", "content": ""}]}).encode()
            ctype = "application/json"
        elif self.path in self.pages:
            ctype, body = self.pages[self.path]
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class Tools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeSearx)
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.dir = tempfile.TemporaryDirectory()
        cls.files = bt.FileStore(cls.dir.name)
        FakeSearx.pages["/page"] = ("text/html; charset=utf-8", ("<title>Page</title><p>" + "x" * 30000 + "</p>").encode())

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.dir.cleanup()

    def text(self, res):
        return res["content"][0]["text"]

    def test_web_search(self):
        b = bt.Builtins(self.files, bt.Searx(self.base))
        r = b.call("web_search", {"query": "speed  test", "max_results": 1})
        self.assertFalse(r["isError"])
        self.assertIn("1. One\n   https://example.com/1\n   first snippet", self.text(r))
        self.assertNotIn("Two", self.text(r))
        down = bt.Builtins(self.files, bt.Searx("http://127.0.0.1:1"))
        r = down.call("web_search", {"query": "x"})
        self.assertTrue(r["isError"])
        self.assertIn("Settings > Search", self.text(r))
        self.assertTrue(b.searx.status()["up"])
        self.assertFalse(down.searx.status()["up"])
        self.assertIn("searxng_start", down.searx.start()["error"])

    def test_fetch_refuses_this_pc_and_reads_pages(self):
        b = bt.Builtins(self.files, bt.Searx(self.base))
        r = b.call("fetch_url", {"url": self.base + "/page"})
        self.assertTrue(r["isError"])
        self.assertIn("not a public web address", self.text(r))
        for url in ("file:///etc/passwd", "http://10.0.0.1/", "http://169.254.169.254/latest", "http://[::1]/"):
            self.assertTrue(b.call("fetch_url", {"url": url})["isError"], url)
        self.assertFalse(bt.public_host("localhost"))
        local = bt.Builtins(self.files, bt.Searx(self.base), check_host=lambda h: True)   # a test server stands in
        r = local.call("fetch_url", {"url": self.base + "/page"})
        t = self.text(r)
        self.assertFalse(r["isError"], t)
        self.assertIn("Title: Page", t)
        self.assertIn("call fetch_url with start=12000", t)
        t2 = self.text(local.call("fetch_url", {"url": self.base + "/page", "start": 24000}))
        self.assertNotIn("more characters", t2)

    def test_documents(self):
        doc = self.files.put("deck.pptx", pptx(*[f"Slide {i} talks about topic {i}" for i in range(1, 8)]))
        b = bt.Builtins(self.files, None)
        self.assertEqual(b.available(), ["read_document", "search_document"])
        t = self.text(b.call("read_document", {"id": doc["id"], "pages": "2-3"}))
        self.assertIn("--- slide 2 of 7 ---\nSlide 2", t)
        self.assertNotIn("Slide 4", t)
        t = self.text(b.call("search_document", {"id": doc["id"], "query": "topic 5"}))
        self.assertIn("- page 5: Slide 5", t.splitlines()[1])
        self.assertTrue(b.call("read_document", {"id": "nope"})["isError"])
        self.assertTrue(b.call("read_document", {"id": doc["id"], "pages": "x"})["isError"])
        self.assertTrue(b.call("read_document", {"idd": doc["id"]})["isError"])          # wrong argument name
        big = self.files.put("big.txt", ("line of text\n\n" * 6000).encode())
        t = self.text(b.call("read_document", {"id": big["id"]}))
        self.assertIn("ask for parts", t)

    def test_router_offers_only_what_was_asked(self):
        b = bt.Builtins(self.files, bt.Searx(self.base))
        r = bt.ToolRouter(None, b, ["web_search", "read_document", "made_up"])
        self.assertEqual([t["name"] for t in r.template_tools()], ["web_search", "read_document"])
        self.assertEqual(set(r.routes()), {"web_search", "read_document"})
        self.assertEqual([t["name"] for t in r.template_tools(exclude={"web_search"})], ["read_document"])
        out = r.call("fetch_url", {"url": "https://example.com"})                # not offered: not run
        self.assertFalse(out["ok"])
        out = r.call("web_search", {"query": "q"})
        self.assertTrue(out["ok"])
        self.assertEqual(out["server"], bt.BUILTIN_SERVER)
        r.settings["max_result_chars"] = 10
        self.assertTrue(r.call("web_search", {"query": "q"})["truncated"])


class Endpoints(unittest.TestCase):
    """The HTTP side: upload, file facts, search status, the live view, and a chat that reads a document."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.dir = tempfile.TemporaryDirectory()
        cls.files = bt.FileStore(cls.dir.name)
        doc = cls.files.put("notes.docx", docx("The secret number is 4711."))
        cls.doc_id = doc["id"]
        body = f"<parameter=id>\n{doc['id']}\n</parameter>\n"
        call = f"</think>\n\nLet me read it.\n\n<tool_call>\n<function=read_document>\n{body}</function>\n</tool_call>"
        cls.engine = MockEngine(tok, [call, "</think>\n\nIt says 4711."], max_context=8192)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.svc.builtins = bt.Builtins(cls.files, bt.Searx("http://127.0.0.1:1"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.svc.telemetry.close()
        cls.dir.cleanup()

    def req(self, path, body=None, headers=None):
        data = json.dumps(body).encode() if body is not None else None
        h = {"Content-Type": "application/json"} if body is not None else {}
        h.update(headers or {})
        r = urllib.request.Request(self.base + path, data=data, headers=h)
        try:
            with urllib.request.urlopen(r, timeout=20) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read()

    def test_upload_and_facts(self):
        code, body = self.req("/v1/files", {"name": "deck.pptx", "data": base64.b64encode(pptx("A", "B")).decode(),
                                            "inline_chars": 100})
        self.assertEqual(code, 200, body)
        j = json.loads(body)
        self.assertEqual((j["kind"], j["pages"], j["text"]), ("slides", 2, "A\n\nB"))
        code, body = self.req(f"/v1/files/{j['id']}")
        self.assertEqual(code, 200)
        self.assertNotIn("text", json.loads(body))
        self.assertEqual(self.req("/v1/files/" + "f" * 20)[0], 404)
        code, body = self.req("/v1/files", {"name": "x.bin", "data": base64.b64encode(b"\x00\x01" * 50).decode()})
        self.assertEqual(code, 422)
        self.assertEqual(self.req("/v1/files", {"name": "x", "data": "!!not base64"})[0], 400)
        code, _ = self.req("/v1/files", {"name": "a.txt", "data": "aGk="}, headers={"Origin": "https://evil.example"})
        self.assertEqual(code, 403)

    def test_search_status_and_live(self):
        code, body = self.req("/v1/search")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["up"], False)
        code, body = self.req("/v1/live")
        self.assertEqual(code, 200)
        j = json.loads(body)
        for k in ("live", "last", "hardware", "history", "static", "totals", "max_context"):
            self.assertIn(k, j)
        self.assertEqual(j["live"]["state"], "idle")

    def test_chat_reads_the_document(self):
        self.engine.turns = 0
        code, body = self.req("/v1/chat/completions", {"model": "m", "max_tokens": 200, "strata_builtin": ["read_document"],
                                                       "messages": [{"role": "user", "content": f"Read {self.doc_id}"}]})
        self.assertEqual(code, 200, body)
        j = json.loads(body)
        self.assertIn("4711", j["choices"][0]["message"]["content"])
        events = j["choices"][0]["message"].get("strata_mcp") or []
        result = [e for e in events if e.get("event") == "result"]
        self.assertTrue(result and result[0]["ok"], events)
        self.assertIn("The secret number is 4711.", result[0]["text"])


class Readers(unittest.TestCase):
    def test_cpu_temperature_from_librehardwaremonitor(self):
        tree = {"Text": "Sensor", "Children": [{"Text": "DESKTOP", "Children": [
            {"Text": "Intel Core i7", "ImageURL": "images_icon/cpu.png", "Children": [
                {"Text": "Temperatures", "Children": [
                    {"Text": "CPU Core #1", "Value": "51,0 °C", "SensorId": "/intelcpu/0/temperature/0", "Children": []},
                    {"Text": "CPU Package", "Value": "57.5 °C", "SensorId": "/intelcpu/0/temperature/8", "Children": []}]}]},
            {"Text": "NVIDIA", "ImageURL": "images_icon/nvidia.png", "Children": [
                {"Text": "GPU Core", "Value": "70.0 °C", "SensorId": "/gpu-nvidia/0/temperature/0", "Children": []}]}]}]}
        self.assertEqual(_CpuTemp._from_tree(tree), 57.5)
        self.assertIsNone(_CpuTemp._from_tree({"Text": "x", "Children": []}))

    def test_disks(self):
        C = SimpleNamespace
        counters = [{"PhysicalDrive0": C(read_bytes=0, write_bytes=0, read_time=0, write_time=0)},
                    {"PhysicalDrive0": C(read_bytes=200 << 20, write_bytes=0, read_time=900, write_time=900)}]
        ps = C(disk_io_counters=lambda perdisk=False: counters.pop(0), disk_partitions=lambda all=False: [])
        d = _Disks(ps)
        d.layout_at = 1e18                               # no layout lookup in the test
        self.assertIsNone(d.read())
        d.prev = (d.prev[0] - 1.0, d.prev[1])            # one second between the two readings
        out = d.read()
        self.assertEqual(out[0]["name"], "Disk 0")
        self.assertAlmostEqual(out[0]["read_mb"], 200, delta=5)
        self.assertEqual(out[0]["busy"], 100.0)          # 1.8 s of I/O in 1 s: capped


if __name__ == "__main__":
    unittest.main()
