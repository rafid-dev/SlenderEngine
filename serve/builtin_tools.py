"""serve/builtin_tools.py - the web app's own tools: web search, reading web pages, and reading attached documents.

They run in the same tool loop as MCP tools (serve/server.py run_with_mcp) but need no MCP server:
- `web_search` asks a SearXNG instance (default http://127.0.0.1:8888, its JSON API) and lists the results.
- `fetch_url` reads one public web page (or PDF) as text. Addresses on this PC or the local network are refused, so a
  page the model reads cannot make it fetch something private.
- `read_document` / `search_document` read a file attached in the web app: `FileStore` keeps the text extracted from
  it (PDF through pypdf when installed; Word, PowerPoint, Excel, OpenDocument and EPUB with the standard library;
  HTML and text as is; pictures as their size and format), so a long document is read a few pages at a time instead
  of filling the context.

A request offers them with `"strata_builtin": ["web_search", ...]` (the web app's switches); nothing is offered
otherwise. `ToolRouter` joins them with the MCP hub's tools for one request. Standard library only, plus pypdf.
"""
from __future__ import annotations

import hashlib
import html.parser
import io
import ipaddress
import json
import os
import re
import socket
import struct
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path
from xml.etree import ElementTree

BUILTIN_SERVER = "built-in"
PART_CHARS = 4000                 # a document without pages is cut into parts of about this size
READ_CAP = 15000                  # the most read_document returns in one call
FETCH_BYTES = 4 << 20             # the most fetch_url downloads
FILE_BYTES = 64 << 20             # the largest file that can be attached
USER_AGENT = "Mozilla/5.0 (compatible; SlenderEngine web reader)"


class ExtractError(ValueError):
    """A file that cannot be read: the sentence says why (shown to the user)."""


# ------------------------------------------------------------------------------------------------ HTML to text
class _Text(html.parser.HTMLParser):
    SKIP = {"script", "style", "noscript", "svg", "head", "template", "iframe", "canvas", "form", "button", "select"}
    BLOCK = {"p", "div", "section", "article", "main", "header", "footer", "aside", "nav", "li", "ul", "ol", "table",
             "tr", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "blockquote", "dd", "dt", "figure", "figcaption"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip, self.title, self._in_title, self._title_done = [], 0, "", False, False

    def handle_starttag(self, tag, attrs):
        if tag == "title" and not self._title_done:     # the page's title; an inline SVG's <title> later is not it
            self._in_title = True
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3") and not self.skip:
            self.out.append("## ")
        if tag == "li" and not self.skip:
            self.out.append("- ")
        if tag in ("td", "th") and not self.skip:
            self.out.append("\t")

    def handle_endtag(self, tag):
        if tag == "title" and self._in_title:
            self._in_title, self._title_done = False, True
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        if not self.skip:
            self.out.append(data)


def html_to_text(markup: str) -> tuple[str, str]:
    """(title, text) of an HTML page: scripts, styles and forms dropped, blocks on their own lines."""
    p = _Text()
    try:
        p.feed(markup)
        p.close()
    except Exception:  # noqa: BLE001 - a broken page still gives what was read
        pass
    text = "".join(p.out)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return " ".join(p.title.split()), text.strip()


# ------------------------------------------------------------------------------------------------ extraction
def _parts(text: str, size: int = PART_CHARS) -> list[str]:
    """Text cut into parts of about `size` characters at paragraph (or line) ends."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = text.rfind("\n\n", start + size // 2, end)
            if cut < 0:
                cut = text.rfind("\n", start + size // 2, end)
            if cut < 0:
                cut = text.rfind(" ", start + size // 2, end)
            end = cut if cut > start else end
        out.append(text[start:end].strip())
        start = end
    return [p for p in out if p]


def _xml(z: zipfile.ZipFile, name: str):
    with z.open(name) as f:
        return ElementTree.parse(f).getroot()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _docx(z: zipfile.ZipFile) -> list[str]:
    root = _xml(z, "word/document.xml")
    lines = []
    for el in root.iter():
        t = _local(el.tag)
        if t == "p":
            lines.append("".join(x.text or "" for x in el.iter() if _local(x.tag) == "t"))
    # a table's cells are paragraphs too, so they read as lines; good enough for reading
    return _parts("\n".join(lines))


def _pptx(z: zipfile.ZipFile) -> list[str]:
    names = sorted((n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)),
                   key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))
    pages = []
    for n in names:
        root = _xml(z, n)
        paras = []
        for p in root.iter():
            if _local(p.tag) == "p":
                s = "".join(x.text or "" for x in p.iter() if _local(x.tag) == "t").strip()
                if s:
                    paras.append(s)
        pages.append("\n".join(paras))
    return pages


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def _xlsx(z: zipfile.ZipFile) -> list[str]:
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in _xml(z, "xl/sharedStrings.xml"):
            shared.append("".join(x.text or "" for x in si.iter() if _local(x.tag) == "t"))
    names = []
    try:
        wb = _xml(z, "xl/workbook.xml")
        names = [s.get("name") for s in wb.iter() if _local(s.tag) == "sheet"]
    except KeyError:
        pass
    sheets = sorted((n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)),
                    key=lambda n: int(re.search(r"(\d+)\.xml$", n).group(1)))
    pages = []
    for i, n in enumerate(sheets):
        rows = []
        for row in _xml(z, n).iter():
            if _local(row.tag) != "row":
                continue
            cells = {}
            for c in row:
                if _local(c.tag) != "c":
                    continue
                v = next((x.text for x in c if _local(x.tag) == "v"), None)
                if v is None:
                    v = "".join(x.text or "" for x in c.iter() if _local(x.tag) == "t") or None
                if v is not None and c.get("t") == "s":
                    v = shared[int(v)] if v.isdigit() and int(v) < len(shared) else v
                if v is not None and c.get("r"):
                    cells[_col(c.get("r"))] = v
            if cells:
                rows.append("\t".join(cells.get(k, "") for k in range(max(cells) + 1)))
            if len(rows) >= 20000:
                rows.append("[... more rows not read]")
                break
        title = names[i] if i < len(names) else f"Sheet {i + 1}"
        pages.append(f"Sheet: {title}\n" + "\n".join(rows))
    return pages


def _odf(z: zipfile.ZipFile) -> list[str]:
    root = _xml(z, "content.xml")
    lines = ["".join(el.itertext()) for el in root.iter() if _local(el.tag) in ("p", "h")]
    return _parts("\n".join(lines))


def _epub(z: zipfile.ZipFile) -> list[str]:
    pages = []
    for n in sorted(z.namelist()):
        if n.lower().endswith((".xhtml", ".html", ".htm")):
            _, text = html_to_text(z.read(n).decode("utf-8", "replace"))
            if text:
                pages.extend(_parts(text))
    return pages


def image_info(data: bytes) -> dict | None:
    """Format and size of a PNG, JPEG, GIF, WebP or BMP picture from its header, or None."""
    try:
        if data[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", data[16:24])
            return {"format": "PNG", "width": w, "height": h}
        if data[:6] in (b"GIF87a", b"GIF89a"):
            w, h = struct.unpack("<HH", data[6:10])
            return {"format": "GIF", "width": w, "height": h}
        if data[:2] == b"BM":
            w, h = struct.unpack("<ii", data[18:26])
            return {"format": "BMP", "width": w, "height": abs(h)}
        if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            chunk = data[12:16]
            if chunk == b"VP8X":
                return {"format": "WebP", "width": 1 + int.from_bytes(data[24:27], "little"),
                        "height": 1 + int.from_bytes(data[27:30], "little")}
            if chunk == b"VP8L":
                b = int.from_bytes(data[21:25], "little")
                return {"format": "WebP", "width": (b & 0x3FFF) + 1, "height": ((b >> 14) & 0x3FFF) + 1}
            if chunk == b"VP8 ":
                w, h = struct.unpack("<HH", data[26:30])
                return {"format": "WebP", "width": w & 0x3FFF, "height": h & 0x3FFF}
        if data[:2] == b"\xff\xd8":
            i = 2
            while i + 9 < len(data):
                if data[i] != 0xFF:
                    i += 1
                    continue
                marker = data[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", data[i + 5:i + 9])
                    return {"format": "JPEG", "width": w, "height": h}
                i += 2 + struct.unpack(">H", data[i + 2:i + 4])[0]
    except (struct.error, IndexError, ValueError):
        return None
    return None


TEXT_LIKE = re.compile(r"\.(txt|md|markdown|rst|tex|py|pyi|js|mjs|cjs|ts|tsx|jsx|vue|svelte|json|jsonl|csv|tsv|log|ya?ml|"
                       r"toml|ini|cfg|conf|env|xml|css|scss|less|c|cc|cpp|cxx|h|hh|hpp|cu|cuh|rs|go|java|kt|kts|swift|rb|"
                       r"php|pl|lua|r|jl|scala|sql|sh|bash|zsh|fish|ps1|psm1|bat|cmd|diff|patch|gradle|cmake|mk|proto|"
                       r"graphql|srt|vtt|ipynb)$", re.I)


def extract(name: str, data: bytes) -> dict:
    """{"kind", "pages": [text, ...], "info"} for a file's bytes; raises ExtractError for what cannot be read."""
    low = name.lower()
    ext = os.path.splitext(low)[1]
    info: dict = {}
    if ext == ".pdf" or data[:5] == b"%PDF-":
        try:
            import pypdf
        except ImportError:
            raise ExtractError("reading PDFs needs the pypdf package in the server's Python "
                               "(pip install pypdf)") from None
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:  # noqa: BLE001
                    raise ExtractError("this PDF is protected by a password") from None
            pages = [(p.extract_text() or "").strip() for p in reader.pages]
        except ExtractError:
            raise
        except Exception as e:  # noqa: BLE001 - pypdf raises many kinds
            raise ExtractError(f"this PDF could not be read ({type(e).__name__})") from None
        if not any(pages):
            raise ExtractError("this PDF has no text layer (a scan?); attach it as pictures instead")
        return {"kind": "pdf", "pages": pages, "info": {"pages": len(pages)}}
    if ext in (".docx", ".pptx", ".xlsx", ".odt", ".ods", ".odp", ".epub", ".docm", ".pptm", ".xlsm"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                if ext in (".docx", ".docm"):
                    return {"kind": "word", "pages": _docx(z), "info": info}
                if ext in (".pptx", ".pptm"):
                    return {"kind": "slides", "pages": _pptx(z), "info": {"slides": len(z.namelist())}}
                if ext in (".xlsx", ".xlsm"):
                    return {"kind": "sheet", "pages": _xlsx(z), "info": info}
                if ext == ".epub":
                    return {"kind": "book", "pages": _epub(z), "info": info}
                return {"kind": "opendocument", "pages": _odf(z), "info": info}
        except (zipfile.BadZipFile, KeyError, ElementTree.ParseError) as e:
            raise ExtractError(f"this {ext[1:].upper()} file could not be read ({type(e).__name__})") from None
    pic = image_info(data)
    if pic:
        text = f"Picture {name}: {pic['format']}, {pic['width']} x {pic['height']} pixels, {len(data):,} bytes."
        return {"kind": "image", "pages": [text], "info": pic}
    if ext in (".html", ".htm", ".xhtml"):
        title, text = html_to_text(data.decode("utf-8", "replace"))
        return {"kind": "html", "pages": _parts(text), "info": {"title": title}}
    if b"\x00" in data[:8192]:
        raise ExtractError(f"{ext[1:].upper() or 'this'} files cannot be read (not text, not a known document type)")
    for enc in ("utf-8-sig", "utf-16") if data[:2] in (b"\xff\xfe", b"\xfe\xff") else ("utf-8-sig", "cp1252"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = data.decode("utf-8", "replace")
    return {"kind": "text", "pages": _parts(text), "info": info}


# ------------------------------------------------------------------------------------------------ the file store
ID_RE = re.compile(r"[0-9a-f]{20}")


class FileStore:
    """Attached files' extracted text, one JSON file per file in `folder` (named by the content's hash, so the same file
    attached twice is stored once). The bytes themselves are not kept."""

    def __init__(self, folder, max_bytes: int = 2 << 30, keep_days: float = 30.0):
        self.folder = Path(folder)
        self.max_bytes, self.keep_days = max_bytes, keep_days   # 0: no limit
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.lock = threading.Lock()

    def _files(self):
        try:
            return [(f, f.stat()) for f in self.folder.glob("*.json") if ID_RE.fullmatch(f.stem)]
        except OSError:
            return []

    def prune(self) -> int:
        """Drop files older than keep_days, then the oldest until the folder is under max_bytes. -> how many."""
        files = sorted(self._files(), key=lambda x: x[1].st_mtime)
        now, gone = time.time(), 0
        total = sum(st.st_size for _, st in files)
        for f, st in files:
            old = self.keep_days and now - st.st_mtime > self.keep_days * 86400
            big = self.max_bytes and total > self.max_bytes
            if not (old or big):
                continue
            try:
                f.unlink()
                gone += 1
                total -= st.st_size
                with self.lock:
                    self.cache.pop(f.stem, None)
            except OSError:
                pass
        return gone

    def clear(self) -> int:
        """Delete every kept file (the web app's Settings > Data). -> how many."""
        gone = 0
        for f, _ in self._files():
            try:
                f.unlink()
                gone += 1
            except OSError:
                pass
        with self.lock:
            self.cache.clear()
        return gone

    def usage(self) -> dict:
        files = self._files()
        return {"files": len(files), "bytes": sum(st.st_size for _, st in files), "folder": str(self.folder),
                "max_bytes": self.max_bytes, "keep_days": self.keep_days}

    def put(self, name: str, data: bytes) -> dict:
        if len(data) > FILE_BYTES:
            raise ExtractError(f"the file is larger than {FILE_BYTES >> 20} MB")
        name = os.path.basename(str(name or "file").replace("\\", "/"))[:200] or "file"
        fid = hashlib.sha256(data).hexdigest()[:20]
        got = extract(name, data)
        pages = got["pages"]
        doc = {"id": fid, "name": name, "kind": got["kind"], "pages": pages, "info": got["info"],
               "chars": sum(len(p) for p in pages), "bytes": len(data), "created": time.time()}
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = self.folder / f"{fid}.json.tmp"
        tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.folder / f"{fid}.json")
        self.prune()
        with self.lock:
            self.cache[fid] = doc
            self.cache.move_to_end(fid)
            while len(self.cache) > 8:
                self.cache.popitem(last=False)
        return doc

    def get(self, fid: str) -> dict | None:
        fid = str(fid or "").strip().lower()
        if not ID_RE.fullmatch(fid):
            return None
        with self.lock:
            if fid in self.cache:
                self.cache.move_to_end(fid)
                return self.cache[fid]
        f = self.folder / f"{fid}.json"
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            os.utime(f)                       # in use: the age limit counts from the last read
        except (OSError, ValueError):
            return None
        with self.lock:
            self.cache[fid] = doc
            while len(self.cache) > 8:
                self.cache.popitem(last=False)
        return doc

    @staticmethod
    def summary(doc: dict, inline_chars: int = 0) -> dict:
        """What the web app gets back: no page text, except the whole text when it is short enough to send inline."""
        out = {k: doc[k] for k in ("id", "name", "kind", "chars", "bytes", "info")}
        out["pages"] = len(doc["pages"])
        if inline_chars and doc["chars"] <= inline_chars:
            out["text"] = "\n\n".join(doc["pages"])
        return out


# ------------------------------------------------------------------------------------------------ fetching
def public_host(host: str) -> bool:
    """True when every address `host` resolves to is a public internet address (not this PC, the local network,
    link-local, carrier NAT or reserved)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return False
    addrs = {i[4][0] for i in infos}
    try:
        return bool(addrs) and all(ipaddress.ip_address(a.split("%")[0]).is_global for a in addrs)
    except ValueError:
        return False


def _is_public(addr: str) -> bool:
    try:
        return ipaddress.ip_address(addr.split("%")[0]).is_global
    except ValueError:
        return False


def _vetted_socket(host: str, port: int, timeout, check_host) -> socket.socket:
    """A connection to `host` that goes to an address checked here and now. Checking a name and then letting the HTTP
    library resolve it again leaves a gap (DNS rebinding: the second answer can be 127.0.0.1), so the check and the
    connection use the same resolved address. Every address the name has must be public (one private one refuses
    the name), unless `check_host` is not public_host (the tests' local servers)."""
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except (socket.gaierror, UnicodeError, OSError) as e:
        raise urllib.error.URLError(f"{host} could not be found ({e})") from None
    if check_host is public_host:
        if not infos or not all(_is_public(i[4][0]) for i in infos):
            raise urllib.error.URLError(f"{host} is not a public web address (this PC and the local network are not read)")
    elif not check_host(host):
        raise urllib.error.URLError(f"{host} is not an allowed address")
    last = None
    for family, kind, proto, _, addr in infos:
        sock = socket.socket(family, kind, proto)
        try:
            if timeout is not None and timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            sock.connect(addr)
            return sock
        except OSError as e:
            last = e
            sock.close()
    raise urllib.error.URLError(f"{host} did not answer ({last})")


def _opener(check_host):
    """urllib with every connection (redirects too) made by _vetted_socket, and no proxy from the environment (a proxy
    would resolve the name itself, out of reach of the check)."""
    import http.client
    import ssl

    class Plain(http.client.HTTPConnection):
        def connect(self):
            self.sock = _vetted_socket(self.host, self.port, self.timeout, check_host)

    class Secure(http.client.HTTPSConnection):
        def connect(self):
            sock = _vetted_socket(self.host, self.port, self.timeout, check_host)
            self.sock = self._context.wrap_socket(sock, server_hostname=self.host)

    class PlainHandler(urllib.request.HTTPHandler):
        def http_open(self, req):
            return self.do_open(Plain, req)

    class SecureHandler(urllib.request.HTTPSHandler):
        def https_open(self, req):
            return self.do_open(Secure, req, context=ssl.create_default_context())

    class Redirects(urllib.request.HTTPRedirectHandler):
        max_redirections = 5

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            u = urllib.parse.urlsplit(newurl)
            if u.scheme not in ("http", "https") or not u.hostname:
                raise urllib.error.URLError(f"a redirect to {newurl!r} was refused (only http and https)")
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    return urllib.request.build_opener(urllib.request.ProxyHandler({}), PlainHandler(), SecureHandler(), Redirects())


def fetch(url: str, timeout: float = 15.0, check_host=public_host) -> tuple[str, bytes, str]:
    """(final url, body, content type) of a public http(s) URL, at most FETCH_BYTES. Every hop (redirects included)
    connects only to an address that passed the check."""
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("only http:// and https:// addresses can be read")
    if u.username or u.password:
        raise ValueError("addresses with a user name or password are not read")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,"
                                               "text/plain,application/pdf;q=0.9,*/*;q=0.5", "Accept-Language": "en"})
    try:
        with _opener(check_host).open(req, timeout=timeout) as r:
            body = r.read(FETCH_BYTES + 1)
            return r.geturl(), body[:FETCH_BYTES], r.headers.get("Content-Type", "")
    except urllib.error.URLError as e:
        reason = e.reason
        while isinstance(reason, urllib.error.URLError):     # urllib wraps a connect error in another URLError
            reason = reason.reason
        reason = str(reason)
        if "not a public web address" in reason or "not an allowed address" in reason or "was refused" in reason:
            raise ValueError(reason) from None
        raise


# ------------------------------------------------------------------------------------------------ SearXNG
class Searx:
    """The SearXNG instance web_search uses, and how it gets started when the web app switches search on: the run
    config's "searxng_start" command when there is one (a SearXNG you set up yourself), else serve/searxng_setup.py,
    which installs it the first time (native venv, WSL or Docker) and starts it."""

    def __init__(self, url: str = "http://127.0.0.1:8888", start_cmd=None, setup=None, may_install: bool = True):
        self.url = (url or "http://127.0.0.1:8888").rstrip("/")
        self.start_cmd = start_cmd
        self.setup = setup                      # searxng_setup.SearxngSetup, or None
        self.may_install = may_install          # the config's "searxng_install" (false: never install)
        self.started_at = None
        self.last_error = None

    def up(self, timeout: float = 1.5) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/healthz", timeout=timeout) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001 - down, refused, timed out
            return False

    def status(self, with_methods: bool = False) -> dict:
        starting = self.started_at is not None and time.time() - self.started_at < 60
        out = {"url": self.url, "up": self.up(), "custom": bool(self.start_cmd),
               "can_start": bool(self.start_cmd or self.setup), "starting": starting, "error": self.last_error}
        if self.setup is not None and not self.start_cmd:
            st = self.setup.status()
            out["setup"] = st
            out["starting"] = st["state"] in ("installing", "starting")
            out["error"] = st["error"] if st["state"] == "failed" else None
            out["may_install"] = self.may_install
            if with_methods:
                out["methods"] = self.setup.methods()
        return out

    def start(self, method: str = "native", may_install: bool = False) -> dict:
        """Start SearXNG: the configured command, or the managed one (installing it only when `may_install`)."""
        if self.up():
            return self.status()
        if self.start_cmd:
            if self.started_at is None or time.time() - self.started_at > 60:
                cmd = self.start_cmd if isinstance(self.start_cmd, list) else [self.start_cmd]
                flags = 0x08000000 if os.name == "nt" else 0               # CREATE_NO_WINDOW: no console pops up
                try:
                    subprocess.Popen([str(c) for c in cmd], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True)
                    self.started_at, self.last_error = time.time(), None
                    print(f"[strata] starting SearXNG for web search: {' '.join(map(str, cmd))}", flush=True)
                except OSError as e:
                    self.last_error = f"the searxng_start command failed: {e}"
            return self.status()
        if self.setup is None:
            self.last_error = f"SearXNG does not answer at {self.url} and no searxng_start command is set in the config"
            return self.status()
        self.setup.start(method, may_install=may_install and self.may_install)
        return self.status()

    def autostart(self) -> threading.Thread:
        """Start SearXNG in the background as the server starts (the config's "searxng_autostart": true), so other
        apps on this PC (an agent that searches through the same SearXNG) have it without the web page. The run
        config is the PC owner's own, so this may install; the way already installed is used, else the Python one."""
        def go():
            method = "native"
            if self.setup is not None:
                method = next((m for m, ok in self.setup.installed().items() if ok), "native")
            self.start(method, may_install=True)
        t = threading.Thread(target=go, daemon=True, name="searxng-autostart")
        t.start()
        return t

    def stop(self) -> dict:
        if self.setup is not None and not self.start_cmd:
            self.setup.stop()
        return self.status()

    def uninstall(self, method: str) -> dict:
        if self.setup is None or self.start_cmd:
            raise ValueError("this server does not manage SearXNG (a searxng_start command is set)")
        self.setup.uninstall(method)
        return self.status()

    def search(self, query: str, n: int = 6, timeout: float = 12.0) -> list[dict]:
        q = urllib.parse.urlencode({"q": query, "format": "json", "safesearch": 0})
        req = urllib.request.Request(f"{self.url}/search?{q}", headers={"User-Agent": USER_AGENT,
                                                                         "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read(4 << 20).decode("utf-8", "replace"))
        out = []
        for x in data.get("results") or []:
            if x.get("url") and x.get("title"):
                out.append({"title": " ".join(str(x["title"]).split()), "url": x["url"],
                            "snippet": " ".join(str(x.get("content") or "").split())[:400],
                            "date": x.get("publishedDate")})
            if len(out) >= n:
                break
        return out


# ------------------------------------------------------------------------------------------------ the tools
TOOLS = [
    {"name": "web_search",
     "description": "Search the web. Returns titles, addresses and short snippets of the top results. Use it for "
                    "current events, facts you are unsure of, or anything after your training data; then read the "
                    "most useful pages with fetch_url.",
     "inputSchema": {"type": "object", "properties": {
         "query": {"type": "string", "description": "What to search for, as you would type it into a search engine"},
         "max_results": {"type": "integer", "description": "How many results (1-10, default 6)"}},
         "required": ["query"]}},
    {"name": "fetch_url",
     "description": "Read a public web page or PDF as plain text. Long pages come in pieces: call again with the "
                    "`start` the answer names to read on.",
     "inputSchema": {"type": "object", "properties": {
         "url": {"type": "string", "description": "The full http:// or https:// address"},
         "start": {"type": "integer", "description": "Character offset to start reading from (default 0)"}},
         "required": ["url"]}},
    {"name": "read_document",
     "description": "Read pages of a document the user attached (its id is given where it was attached). Ask for "
                    "a few pages at a time; the answer says how many pages there are.",
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string", "description": "The document's id"},
         "pages": {"type": "string", "description": "Pages to read, like \"1\", \"3-5\" or \"2,7\" (default: from the start)"}},
         "required": ["id"]}},
    {"name": "search_document",
     "description": "Find the passages of an attached document that match words or a phrase, with their page "
                    "numbers. Use it to locate something in a long document before reading those pages.",
     "inputSchema": {"type": "object", "properties": {
         "id": {"type": "string", "description": "The document's id"},
         "query": {"type": "string", "description": "Words or a phrase to look for"}},
         "required": ["id", "query"]}},
]
NAMES = [t["name"] for t in TOOLS]


def _pages(spec, n: int) -> list[int]:
    """1-based page numbers from "3", "3-5", "2,7" (clamped to the document); the first pages when empty."""
    if spec in (None, ""):
        return list(range(1, n + 1))
    out = []
    for part in str(spec).replace(" ", "").split(","):
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise ValueError(f"pages: {spec!r} is not like \"3\", \"3-5\" or \"2,7\"")
        a, b = int(m.group(1)), int(m.group(2) or m.group(1))
        out.extend(range(max(1, a), min(n, max(a, b)) + 1))
    return out


class Builtins:
    """The built-in tools. `call` answers like an MCP server's tools/call ({"content": [...], "isError"})."""
    name = BUILTIN_SERVER
    kind = "builtin"

    def __init__(self, files: FileStore | None, searx: Searx | None, check_host=public_host):
        self.files, self.searx, self.check_host = files, searx, check_host
        self.tools = TOOLS

    def available(self) -> list[str]:
        return [n for n in NAMES if (n in ("web_search", "fetch_url") and self.searx is not None)
                or (n in ("read_document", "search_document") and self.files is not None)]

    def call(self, tool: str, args: dict, timeout: float = 60.0, cancel=None) -> dict:
        try:
            text = getattr(self, "_" + tool)(**{k: v for k, v in (args or {}).items() if isinstance(k, str)})
            return {"content": [{"type": "text", "text": text}], "isError": False}
        except TypeError as e:
            return {"content": [{"type": "text", "text": f"wrong arguments for {tool}: {e}"}], "isError": True}
        except (ValueError, ExtractError, OSError, urllib.error.URLError) as e:
            return {"content": [{"type": "text", "text": str(getattr(e, "reason", None) or e)}], "isError": True}

    # each tool: keyword arguments as the model writes them; unknown ones are an error the model can read
    def _web_search(self, query, max_results=6):
        if self.searx is None:
            raise ValueError("web search is not set up on this server")
        q = " ".join(str(query).split())
        if not q:
            raise ValueError("query is empty")
        try:
            n = max(1, min(10, int(max_results)))
        except (TypeError, ValueError):
            n = 6
        try:
            results = self.searx.search(q, n)
        except (urllib.error.URLError, OSError, ValueError) as e:
            raise ValueError(f"web search is not running (SearXNG at {self.searx.url} did not answer: "
                             f"{getattr(e, 'reason', e)}). The user can switch it on in Settings > Search.") from None
        if not results:
            return f"No results for {q!r}."
        lines = [f"Results for {q!r}:"]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r['title']}\n   {r['url']}" + (f"\n   {r['snippet']}" if r["snippet"] else "")
                         + (f"\n   ({r['date'][:10]})" if r.get("date") else ""))
        return "\n".join(lines)

    def _fetch_url(self, url, start=0, max_chars=12000):
        try:
            start, max_chars = max(0, int(start or 0)), max(1000, min(20000, int(max_chars or 12000)))
        except (TypeError, ValueError):
            start, max_chars = 0, 12000
        final, body, ctype = fetch(str(url).strip(), check_host=self.check_host)
        # a page that only redirects with <meta http-equiv="refresh" content="0; url=...">: followed once
        refresh = re.search(rb'<meta[^>]+http-equiv=["\']?refresh["\']?[^>]*content=["\'][^"\']*url=([^"\'>\s]+)', body[:20000], re.I)
        if refresh and "html" in ctype:
            target = urllib.parse.urljoin(final, refresh.group(1).decode("utf-8", "replace"))
            final, body, ctype = fetch(target, check_host=self.check_host)
        title = ""
        if "pdf" in ctype or body[:5] == b"%PDF-":
            text = "\n\n".join(f"--- page {i} ---\n{p}" for i, p in enumerate(extract("page.pdf", body)["pages"], 1))
        elif "html" in ctype or body.lstrip()[:15].lower().startswith((b"<!doctype html", b"<html")):
            charset = re.search(r"charset=([\w-]+)", ctype)
            title, text = html_to_text(body.decode(charset.group(1) if charset else "utf-8", "replace"))
        elif ctype.startswith("text/") or "json" in ctype or "xml" in ctype:
            text = body.decode("utf-8", "replace")
        else:
            raise ValueError(f"{final} is {ctype or 'not text'}, which cannot be read as text")
        piece = text[start:start + max_chars]
        head = (f"Title: {title}\n" if title else "") + f"Address: {final}\n"
        rest = len(text) - (start + len(piece))
        tail = f"\n\n[... {rest:,} more characters: call fetch_url with start={start + len(piece)} to read on]" if rest > 0 else ""
        if not piece.strip():
            return head + ("\n(the page has no readable text)" if not start else "\n(nothing after that point)")
        return f"{head}\n{piece}{tail}"

    def _doc(self, id):
        if self.files is None:
            raise ValueError("attached documents are not set up on this server")
        doc = self.files.get(id)
        if doc is None:
            raise ValueError(f"there is no attached document with the id {id!r}")
        return doc

    def _read_document(self, id, pages=None):
        doc = self._doc(id)
        n = len(doc["pages"])
        unit = "slide" if doc["kind"] == "slides" else "sheet" if doc["kind"] == "sheet" else \
            "page" if doc["kind"] == "pdf" else "part"
        want = _pages(pages, n)
        out, used, last = [], 0, None
        for p in want:
            text = doc["pages"][p - 1]
            if used and used + len(text) > READ_CAP:
                break
            if len(text) > READ_CAP:
                text = text[:READ_CAP] + f"\n[... this {unit} goes on for {len(text) - READ_CAP:,} more characters]"
            out.append(f"--- {unit} {p} of {n} ---\n{text}")
            used += len(text)
            last = p
        head = f"{doc['name']} ({n} {unit}{'s' * (n != 1)}, {doc['chars']:,} characters)"
        more = [p for p in want if last is not None and p > last]
        tail = f"\n\n[... stopped at {unit} {last} to keep the answer short: ask for {unit}s {more[0]}-{more[-1]} next]" if more else ""
        return f"{head}\n\n" + "\n\n".join(out) + tail if out else f"{head}\n\n(no {unit}s in that range)"

    def _search_document(self, id, query, max_hits=8):
        doc = self._doc(id)
        words = [w for w in re.findall(r"\w+", str(query).lower()) if len(w) > 1]
        phrase = " ".join(str(query).lower().split())
        if not words:
            raise ValueError("query has no words to look for")
        hits = []
        for pi, page in enumerate(doc["pages"], 1):
            for para in re.split(r"\n\s*\n|(?<=[.!?])\s{2,}", page):
                low = para.lower()
                score = sum(low.count(w) for w in words) + (5 if phrase in low else 0)
                if score:
                    hits.append((score, pi, " ".join(para.split())))
        if not hits:
            return f"Nothing in {doc['name']} matches {query!r}."
        hits.sort(key=lambda h: -h[0])
        out = [f"Best matches for {query!r} in {doc['name']}:"]
        for score, pi, para in hits[:max(1, min(20, int(max_hits or 8)))]:
            out.append(f"- page {pi}: {para[:500]}{'…' if len(para) > 500 else ''}")
        return "\n".join(out)


# ------------------------------------------------------------------------------------------------ one request's tools
class ToolRouter:
    """The MCP hub's tools (when the request asked for them) and the built-in tools it asked for, behind the hub's
    interface (settings, routes, call, template_tools) so run_with_mcp drives both."""

    def __init__(self, hub, builtins: Builtins | None, builtin_names, external: bool = True):
        from serve.mcp import DEFAULTS, result_text
        self.hub = hub if external else None
        self.builtins = builtins
        self.names = [n for n in builtin_names if builtins is not None and n in builtins.available()]
        self.settings = dict(hub.settings) if hub is not None else dict(DEFAULTS)
        self.media_dir = getattr(hub, "media_dir", None)
        self._text = result_text

    def routes(self) -> dict:
        out = dict(self.hub.routes()) if self.hub is not None else {}
        for n in self.names:
            out[n] = (self.builtins, n)
        return out

    def template_tools(self, exclude=()) -> list[dict]:
        out = self.hub.template_tools(exclude=exclude) if self.hub is not None else []
        for t in TOOLS:
            if t["name"] in self.names and t["name"] not in exclude:
                out = [x for x in out if x["name"] != t["name"]]
                out.append({"name": t["name"], "description": t["description"], "parameters": t["inputSchema"]})
        return out

    def call(self, name: str, arguments: dict, cancel=None) -> dict:
        if name not in self.names:
            if self.hub is None:
                return {"server": None, "tool": name, "ok": False, "text": f"error: there is no tool named {name!r}",
                        "chars": 0, "truncated": False, "ms": 0}
            return self.hub.call(name, arguments, cancel)
        t0 = time.monotonic()
        res = self.builtins.call(name, arguments if isinstance(arguments, dict) else {})
        text, ok = self._text(res), not res.get("isError")
        if not ok:
            text = "error: " + (text or "the tool failed")
        cap, full = int(self.settings["max_result_chars"]), len(text)
        if cap > 0 and full > cap:
            text = text[:cap] + f"\n\n[... truncated: {full:,} characters; only the first {cap:,} are shown]"
        return {"server": BUILTIN_SERVER, "tool": name, "ok": ok, "text": text, "chars": full,
                "truncated": cap > 0 and full > cap, "ms": int((time.monotonic() - t0) * 1000)}
