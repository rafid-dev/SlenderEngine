"""tools/web2_mock.py - the web app (serve/web2) against a mock engine: no GPU, no model, for UI work.

    python tools/web2_mock.py [--port 8095] [--delay 0.004] [--ctx 32768]

The mock answers by looking at the prompt, one byte per token:
- a message that says "search"      -> a web_search call (a fake SearXNG answers it), then an answer
- a message with an attached document -> a read_document call, then an answer
- the web app's compaction request   -> a short summary
- a message that says "long answer" -> the reply below ten times (about 10K characters)
- anything else                     -> thinking, then Markdown with a list, a table, code and math
Attached files are kept in a temporary folder. The Analytics tab shows this PC's real hardware.
"""
from __future__ import annotations

import argparse
import http.server
import json
import re
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.builtin_tools import Builtins, FileStore, Searx  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import IM_END, ByteTokenizer, MockEngine, Service, serve  # noqa: E402

REPLY = """The user wants a quick overview. I should keep it short, show a list, a small table and one code block.
Let me also include a formula so the math drawing can be checked.
</think>

Here is a **short overview** of how the pieces fit together.

## What happens on each turn

1. The page sends the conversation to `/v1/chat/completions`.
2. The server renders the chat template and streams tokens back.
3. The page draws each new word with a *small fade-in*.

| Part | Where | Job |
|---|---|---|
| Engine | `src/` | Runs the model |
| Server | `serve/` | API and streaming |
| Page | `serve/web2/` | What you see |

```python
def speed(tokens, seconds):
    return tokens / seconds   # tokens per second
```

The decode rate is $r = \\frac{n}{t}$, and over a whole reply:

$$ \\bar r = \\frac{\\sum_i n_i}{\\sum_i t_i} $$

That's all - ask a follow-up to see the chat keep its history."""


def call(name, **params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"I should use a tool for this.\n</think>\n\n<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


class SmartMock(MockEngine):
    """Picks its reply from the prompt instead of a fixed script."""

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        text = bytes(t for t in ids if t < 256).decode("utf-8", "replace")
        last_user = text.rsplit("<|im_start|>user", 1)[-1]
        answered = "<tool_response>" in last_user
        doc = re.search(r"Attached document: .*?\(id ([0-9a-f]{20})", last_user)
        if "You compress a conversation" in text:
            reply = ("- The user is testing the SlenderEngine web app with the mock engine.\n- Earlier turns asked for "
                     "overviews of how the page, server and engine fit together.\n- Nothing is open.")
        elif "long answer" in last_user:     # about 10K characters: for checking that long answers stay smooth
            reply = REPLY + ("\n\n" + REPLY.split("</think>", 1)[1].strip()) * 9
        elif answered:
            reply = "I looked that up.\n</think>\n\nThe tool answered; here is what it found, summed up in two lines. " \
                    "Everything came back as expected, so the tool loop works end to end."
        elif doc:
            reply = call("read_document", id=doc.group(1), pages="1-2")
        elif re.search(r"\bsearch\b", last_user, re.I) and "web_search" in text:
            reply = call("web_search", query="SlenderEngine web app test")
        else:
            reply = REPLY
        self.script = self.tok.encode(reply, parse_special=True) + self.tok.encode(IM_END, parse_special=True)
        for t in self.script[:max_new]:
            if cancel.is_set():
                return
            if self.delay:
                time.sleep(self.delay)
            yield t


class FakeSearx(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/healthz"):
            body, ctype = b"OK", "text/plain"
        else:
            body = json.dumps({"results": [
                {"title": "SlenderEngine - a fast local engine", "url": "https://example.com/slender", "content": "A test result."},
                {"title": "Second result", "url": "https://example.org/two", "content": "Another snippet."}]}).encode()
            ctype = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--delay", type=float, default=0.004, help="seconds per byte (one byte = one token)")
    ap.add_argument("--ctx", type=int, default=32768, help="the context size the mock reports")
    ap.add_argument("--real-searxng", metavar="DIR", help="use the real SearXNG installer (serve/searxng_setup.py) with "
                    "this install folder instead of the fake search; for testing the install from the page")
    ap.add_argument("--searxng-url", default="http://127.0.0.1:8890", help="with --real-searxng: where it listens")
    a = ap.parse_args()
    tok = ByteTokenizer()
    svc = Service(SmartMock(tok, REPLY, max_context=a.ctx, delay_s=a.delay), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.model = "mock-model"
    if a.real_searxng:
        import atexit
        from serve.searxng_setup import SearxngSetup
        setup = SearxngSetup(a.real_searxng, a.searxng_url)
        atexit.register(setup.stop)
        search = Searx(a.searxng_url, setup=setup)
    else:
        searx = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeSearx)
        threading.Thread(target=searx.serve_forever, daemon=True).start()
        search = Searx(f"http://127.0.0.1:{searx.server_address[1]}")
    svc.builtins = Builtins(FileStore(tempfile.mkdtemp(prefix="web2-mock-files-")), search)
    httpd = serve(svc, port=a.port)
    print(f"web app on http://127.0.0.1:{httpd.server_address[1]}/  (classic: /classic)", flush=True)
    try:                                   # serve() already answers on its own thread
        threading.Event().wait()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
