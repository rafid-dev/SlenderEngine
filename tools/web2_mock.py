"""tools/web2_mock.py - the web app (serve/web2) against the mock engine: no GPU, no model, for UI work.

    python tools/web2_mock.py [--port 8095] [--delay 0.004]

Every answer replays the same scripted reply (thinking, then Markdown with a list, a table, code and math), one byte
per token, so the typing effect, the thinking block and the layout can be checked without loading the engine.
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8095)
    ap.add_argument("--delay", type=float, default=0.004, help="seconds per byte (one byte = one token)")
    a = ap.parse_args()
    tok = ByteTokenizer()
    svc = Service(MockEngine(tok, REPLY, max_context=32768, delay_s=a.delay), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    svc.model = "mock-model"
    httpd = serve(svc, port=a.port)
    print(f"web app on http://127.0.0.1:{httpd.server_address[1]}/  (classic: /classic)", flush=True)
    try:                                   # serve() already answers on its own thread
        threading.Event().wait()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()
