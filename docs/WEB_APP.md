# The web app

The server serves a chat page at its own address, for example `http://127.0.0.1:8080/`. It runs in your browser and
talks only to that server. The older Strata page is still at `/classic` until it is retired.

What it has: chats and project folders in the sidebar, a system prompt, a reasoning effort menu, a thinking block you
can open or close, context compaction for long chats, web search through SearXNG (off by default), attached documents
(PDF, Word, Excel, PowerPoint, OpenDocument, EPUB, HTML, text, pictures), and an Analytics tab with the engine's
speed and the hardware. Themes, fonts, corners and text size are in Settings > Appearance.

To open it from your phone or another computer, see [Sharing over Tailscale](TAILSCALE.md).

## Web search: set up for you

Search needs [SearXNG](https://github.com/searxng/searxng), a search engine that runs on your PC and asks the big
search engines for you. You do not have to install it. The first time you switch search on, the server installs it
and starts it; after that the switch only starts it. Settings > Search shows each step.

Pick how it runs in Settings > Search:

| Way | What it needs | RAM when running | First switch-on |
|---|---|---|---|
| **Python on this PC** (default) | nothing extra | about 100 MB | about 40 s (measured: 36-48 s) |
| **WSL** | WSL with a Linux distribution that has `python3-venv` | about 1 GB for WSL | similar |
| **Docker** | Docker Desktop, running | Docker's own | depends on the image download |

All three install the same pinned SearXNG version and check the download before they use it: the Python and WSL
ways compare the source archive's SHA-256, the Docker way uses the official image by its digest. SearXNG listens
on `127.0.0.1` only. The Python way installs into `SlenderEngine/searxng` (a Python environment of about 120 MB);
the WSL way into `~/.local/share/slenderengine-searxng` inside the distribution; the Docker way into a container
named `slenderengine-searxng`. Settings > Search has **Stop** (frees its memory) and **Remove** (deletes what was
installed). The server also stops the SearXNG it started when the server stops.

SearXNG does not support Windows officially. The Python way runs it anyway: the one Linux-only module it imports
(`pwd`, used only for an error message about a database a local setup does not have) gets a small stand-in. If a
future SearXNG version breaks this, the pinned version keeps working until the pin is changed in
`serve/searxng_setup.py`. WSL and Docker are never installed by SlenderEngine: they need administrator rights.

**Only from this PC.** SearXNG is installed only when the request comes from a browser on the PC that runs the
model. A phone or laptop using the page over the network or a Tailscale share can switch search on once SearXNG is
installed, but cannot install or remove it, with or without the API key.

**Your own SearXNG.** If you run SearXNG yourself, put its address in `"searxng_url"` and, if the server should
start it, its start command in `"searxng_start"`. With `"searxng_start"` set, the server never installs anything.
`"searxng_install": false` turns installing off entirely.

## Where your data is

| What | Where | How to remove it |
|---|---|---|
| Chats, projects, summaries of compacted chats | Your browser (IndexedDB database `slender`) | Settings > Data > Delete chats, or clear the site's data in the browser |
| Settings and the API key you typed | Your browser (`localStorage`, keys `slender.*`) | Clear the site's data in the browser |
| Text of attached documents | The server's folder next to its run config (`<config>.files`, or `"files_dir"`) | Settings > Data > Delete documents, or delete the folder |
| Pictures you attach | Your browser, inside the chat; sent to the server with each request, which does not keep them | Delete the chat |

Storage belongs to an address. `http://127.0.0.1:8080` and `https://your-pc.your-tailnet.ts.net:8443` are two
different sites to the browser, each with its own chats and its own saved key. Use Settings > Data > Export and
Import to move chats between them.

The server keeps only the extracted text of a document, never the file itself. The oldest documents are deleted
after 30 days without use (`"files_keep_days"`) or when the folder passes 2 GB (`"files_max_mb"`); 0 turns a limit
off.

## Security

The page is served by the same server as the API, so everything in [SECURITY.md](../SECURITY.md) applies: it listens
on `127.0.0.1` unless you change it, the Host and Origin checks guard against other web sites, and an `"api_key"`
protects everything when the server is reachable from other devices. This section covers what the web app adds.

### What the page may do

The page is sent with a strict Content-Security-Policy:

- Scripts only from the server's own files. The page has no inline script, so text that slips into the page as HTML
  cannot run.
- Pictures, fonts and connections only from the server itself (and `data:` pictures you attach). A chat that
  contains a link to an outside picture cannot load it, and a script cannot send your API key anywhere.
- No frames, and no other site may show the page in a frame (`frame-ancestors 'none'`, `X-Frame-Options: DENY`).
- `Referrer-Policy: no-referrer`: a link the model wrote does not tell the site where you came from.

Answers are escaped before they are formatted, so HTML in an answer is shown as text. Links work only for `http` and
`https`. Formulas are drawn by KaTeX with `trust: false`, so a formula cannot make links or load anything.

### The tools

The web app can offer the model four tools of its own. The server runs them, only for requests from its own page
(JSON, and no `Origin` of another site; with an API key, also the key).

| Tool | Offered when | What leaves your PC |
|---|---|---|
| `web_search` | Search is switched on (Settings > Search, or the globe button) | Your search words, sent by SearXNG to the search engines it uses |
| `fetch_url` | Search is switched on | A request from your PC to the page the model picked; the site sees your IP address |
| `read_document`, `search_document` | A chat has a document too long to include | Nothing |

`fetch_url` reads only public internet addresses. It refuses this PC, your local network, link-local and carrier
addresses, and addresses with a user name or password. The check happens when the connection is made, on the exact
address it connects to, for every redirect too, so a name that first points to the internet and then to your PC
(DNS rebinding) is still refused. It uses no proxy from the environment, reads at most 4 MB and waits at most 15 s.

Starting SearXNG runs the command in the run config's `"searxng_start"`, or the server's own installer. A request
can ask for one of three fixed ways (Python, WSL, Docker), never a command, and only a request from this PC itself
can cause an install.

### Prompt injection

Web pages and documents are text the model reads, and that text can try to steer it. A page could, for example,
tell the model to call `fetch_url` on an address that carries part of your chat in it. Every tool call is shown in
the chat, with its address when you open it. To be safe:

- Keep Search off while you work with private documents, or use separate chats for the two.
- Read what a tool call fetched before you trust an answer built on it.
- MCP tools (configured in the run config) run with your user's rights. Add only servers you trust.

### Checklist before you open the server to other devices

1. Set an `"api_key"` in the run config (the Tailscale tool does this for you).
2. Keep the server on `127.0.0.1` and put Tailscale in front of it ([TAILSCALE.md](TAILSCALE.md)), rather than
   `--host 0.0.0.0` on a network you do not control.
3. Do not put it on the public internet (Tailscale Funnel, a public tunnel) unless you accept that anyone who gets
   the key can use your GPU and the tools above.
4. Leave Search off unless you want the model to reach the internet.
5. Restart the server after you change the run config: it reads the key and host names only when it starts.

## Settings in the run config

| Key | Default | What it does |
|---|---|---|
| `"searxng_url"` | `http://127.0.0.1:8888` | Where SearXNG answers (the installed one listens on this port too) |
| `"searxng_start"` | none | Your own command that starts SearXNG; when set, the server installs nothing |
| `"searxng_install"` | true | false: never install SearXNG (search works only with one you run yourself) |
| `"searxng_dir"` | `SlenderEngine/searxng` | Where the Python way installs SearXNG |
| `"searxng_autostart"` | false | true: start SearXNG when the server starts, not only when the page asks (other apps on this PC that search through the same address get it too); it stops with the server |
| `"files_dir"` | `<config>.files` | Where attached documents' text is kept |
| `"files_max_mb"` | 2048 | The most the documents folder may hold before the oldest go (0: no limit) |
| `"files_keep_days"` | 30 | Days a document is kept after it was last read (0: no limit) |

CPU temperature in Analytics needs LibreHardwareMonitor with its web server on (Options > Remote Web Server, port
8085); `STRATA_LHM_URL` changes the address. Windows gives no CPU temperature to programs without administrator
rights.
