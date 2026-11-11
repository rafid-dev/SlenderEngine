"""serve/searxng_setup.py - installs and runs SearXNG for the web app's search, so nobody has to set it up by hand.

The first time the web app switches search on, the server installs SearXNG one of three ways and starts it; after
that the switch only starts it. Each way installs the same pinned version, checked before it is used:

- "native" (the default): SearXNG's source in a Python venv inside SlenderEngine/searxng. SearXNG does not support
  Windows officially, but it runs there: the one Unix-only module it imports (pwd, used only to word an error about
  a Valkey database, which a local setup does not have) gets a small stand-in. About 100 MB of RAM when running.
- "wsl": the same venv inside the default WSL distribution (~/.local/share/slenderengine-searxng). About 1 GB of RAM
  for WSL's virtual machine.
- "docker": the official image, pinned by digest, published on 127.0.0.1 only.

SearXNG listens on 127.0.0.1 only, with JSON results on (the web_search tool reads them) and no rate limiter (it is
for this PC). What can trigger an install is decided by the server (only a request from this PC itself; see
server.py), never by this module. Standard library only.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

WINDOWS = os.name == "nt"
NO_WINDOW = 0x08000000 if WINDOWS else 0           # CREATE_NO_WINDOW: no console flashes up

# The version installed: tested on Windows 11 (native, Python 3.14) on 2026-11-11. Change all three together.
VERSION = "2026.11.9+57ff753ad"
COMMIT = "57ff753ad416b7bf27e59d72ae8c7cac93b56be4"
SOURCE_URL = f"https://codeload.github.com/searxng/searxng/zip/{COMMIT}"
SOURCE_SHA256 = "0b22645df495343e39f151b9e29439c6cfb51a11af32363c4cc5848366463e3e"
SOURCE_BYTES = 7_276_876
IMAGE = "searxng/searxng@sha256:287af61f1a13e86d02974fcde91a3413a9945788de38b61ffbc491d81a19bcc2"   # 2026.11.9-57ff753ad
CONTAINER = "slenderengine-searxng"
WSL_DIR = "$HOME/.local/share/slenderengine-searxng"
METHODS = ("native", "wsl", "docker")

PWD_STANDIN = '''"""A stand-in for the Unix-only pwd module, written by SlenderEngine so SearXNG imports on Windows. SearXNG uses
it only to name the user in an error message when a Valkey database is set up and cannot be reached."""
import collections
import getpass
import os

struct_passwd = collections.namedtuple("struct_passwd", "pw_name pw_passwd pw_uid pw_gid pw_gecos pw_dir pw_shell")


def getpwuid(uid=0):
    return struct_passwd(getpass.getuser(), "x", 0, 0, "", os.path.expanduser("~"), "")
'''


def settings_yml(port: int, bind: str = "127.0.0.1") -> str:
    """SearXNG's settings: its defaults, plus a secret, this PC only, JSON results and no rate limiter."""
    return (f"# Written by SlenderEngine (serve/searxng_setup.py). SearXNG {VERSION}.\n"
            "use_default_settings: true\n"
            "general:\n  instance_name: SlenderEngine search\n"
            f"server:\n  secret_key: \"{secrets.token_hex(32)}\"\n  bind_address: \"{bind}\"\n  port: {port}\n"
            "  limiter: false\n  image_proxy: false\n"
            "search:\n  safe_search: 0\n  formats:\n    - html\n    - json\n")


def run(argv, timeout: float = 60.0, **kw) -> tuple[int, str]:
    try:
        r = subprocess.run([str(a) for a in argv], capture_output=True, text=True, errors="replace", timeout=timeout,
                           creationflags=NO_WINDOW, **kw)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


class SetupError(RuntimeError):
    """An install step that failed; the sentence is shown to the user."""


class SearxngSetup:
    def __init__(self, folder, url: str = "http://127.0.0.1:8888", python: str | None = None):
        self.folder = Path(folder)
        self.url = url.rstrip("/")
        self.port = urllib.parse.urlsplit(self.url).port or 8888
        self.python = python or sys.executable
        self.lock = threading.RLock()                  # status() is also called while it is held
        self.state = {"state": "idle", "step": None, "error": None, "method": None, "since": None}
        self.proc: subprocess.Popen | None = None
        self.thread: threading.Thread | None = None
        self._methods_cache = (0.0, None)

    # ---------------------------------------------------------------- what this PC offers
    def _wsl_distro(self) -> str | None:
        if not WINDOWS or not shutil.which("wsl.exe"):
            return None
        code, out = run(["wsl.exe", "-l", "-q"], timeout=15)
        names = [n.strip().replace("\x00", "") for n in out.replace("\x00", "").splitlines()]
        names = [n for n in names if n and not n.lower().startswith("docker-desktop")]
        return names[0] if code == 0 and names else None

    def methods(self) -> dict:
        """{method: {"ok", "note"}}: what can be used here (cached 30 s: asking WSL and Docker takes a moment)."""
        at, cached = self._methods_cache
        if cached and time.time() - at < 30:
            return cached
        py_ok = sys.version_info >= (3, 10)
        out = {"native": {"ok": py_ok, "note": "Python on this PC, about 100 MB of RAM" if py_ok else
                          "needs Python 3.10 or newer for the server"}}
        distro = self._wsl_distro()
        out["wsl"] = {"ok": bool(distro), "note": f"in WSL ({distro}), about 1 GB of RAM" if distro else
                      ("WSL is not installed (wsl --install, as administrator, then restart)" if WINDOWS else
                       "only on Windows")}
        if not shutil.which("docker"):
            out["docker"] = {"ok": False, "note": "Docker is not installed"}
        else:
            code, _ = run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=12)
            out["docker"] = {"ok": code == 0, "note": "the official image, in Docker" if code == 0 else
                             "Docker is installed but not running: start Docker Desktop"}
        self._methods_cache = (time.time(), out)
        return out

    def installed(self) -> dict:
        out = {}
        for m in METHODS:
            try:
                out[m] = json.loads((self.folder / f"{m}.installed").read_text(encoding="utf-8")).get("commit") == COMMIT
            except (OSError, ValueError):
                out[m] = False
        return out

    def status(self) -> dict:
        with self.lock:
            s = dict(self.state)
        s["installed"] = self.installed()
        s["version"] = VERSION
        return s

    def _step(self, text: str, state: str | None = None):
        with self.lock:
            self.state["step"] = text
            if state:
                self.state["state"] = state
        print(f"[strata] SearXNG: {text}", flush=True)

    def _up(self, timeout: float = 1.5) -> bool:
        try:
            with urllib.request.urlopen(self.url + "/healthz", timeout=timeout) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001 - not yet
            return False

    # ---------------------------------------------------------------- start / stop / remove
    def start(self, method: str, may_install: bool) -> dict:
        """Install (when allowed and needed) and start, in the background. -> status() at once."""
        if method not in METHODS:
            raise ValueError(f"method: one of {', '.join(METHODS)}")
        with self.lock:
            busy = self.thread is not None and self.thread.is_alive()
            if not busy:
                if not self.installed().get(method) and not may_install:
                    self.state.update(state="failed", step=None, method=method,
                                      error="SearXNG is not installed yet; it can be installed only from this PC itself")
                    return self.status()
                self.state.update(state="installing" if not self.installed().get(method) else "starting",
                                  step=None, error=None, method=method, since=time.time())
                self.thread = threading.Thread(target=self._work, args=(method,), daemon=True, name="searxng-setup")
                self.thread.start()
        return self.status()

    def _work(self, method: str):
        try:
            if not self.installed().get(method):
                getattr(self, f"_install_{method}")()
                (self.folder / f"{method}.installed").write_text(json.dumps(
                    {"commit": COMMIT, "version": VERSION, "at": time.time()}), encoding="utf-8")
            self._step("Starting SearXNG", "starting")
            getattr(self, f"_launch_{method}")()
            for _ in range(120):
                if self._up():
                    with self.lock:
                        self.state.update(state="running", step=None, error=None)
                    print(f"[strata] SearXNG is running at {self.url} ({method})", flush=True)
                    return
                if self.proc is not None and self.proc.poll() is not None:
                    where = f"{WSL_DIR}/searxng.log in WSL" if method == "wsl" else self.folder / "searxng.log"
                    raise SetupError(f"SearXNG stopped while starting; see {where}")
                time.sleep(0.5)
            raise SetupError(f"SearXNG did not answer at {self.url} within 60 s; see {self.folder / 'searxng.log'}")
        except Exception as e:  # noqa: BLE001 - every failure becomes a sentence in the page
            msg = str(e) if isinstance(e, SetupError) else f"{type(e).__name__}: {e}"
            with self.lock:
                self.state.update(state="failed", step=None, error=msg)
            print(f"[strata] SearXNG could not be set up: {msg}", flush=True)

    def stop(self) -> dict:
        method = self.state.get("method")
        if self.proc is not None and self.proc.poll() is None:
            if WINDOWS:   # the venv's python.exe is a launcher with a child: end the whole tree
                run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], timeout=15)
            else:
                self.proc.terminate()
        self.proc = None
        if method == "wsl" or self.installed().get("wsl"):
            distro = self._wsl_distro()
            if distro:
                run(["wsl.exe", "-d", distro, "-e", "sh", "-c", "pkill -f 'slenderengine-searxng/venv/bin/python -m searx.webapp' || true"],
                    timeout=20)
        if method == "docker" or self.installed().get("docker"):
            if shutil.which("docker"):
                run(["docker", "stop", CONTAINER], timeout=40)
        with self.lock:
            self.state.update(state="idle", step=None, error=None)
        return self.status()

    def uninstall(self, method: str) -> dict:
        if method not in METHODS:
            raise ValueError(f"method: one of {', '.join(METHODS)}")
        self.stop()
        if method == "native":
            for name in ("src", "venv", "tmp", "src.zip", "settings.yml", "searxng.log"):
                p = self.folder / name
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
        elif method == "wsl":
            distro = self._wsl_distro()
            if distro:
                run(["wsl.exe", "-d", distro, "-e", "sh", "-c", f'rm -rf "{WSL_DIR}"'], timeout=60)
        else:
            if shutil.which("docker"):
                run(["docker", "rm", "-f", CONTAINER], timeout=60)
                run(["docker", "image", "rm", IMAGE], timeout=60)
            shutil.rmtree(self.folder / "docker", ignore_errors=True)
        (self.folder / f"{method}.installed").unlink(missing_ok=True)
        return self.status()

    # ---------------------------------------------------------------- native: a venv in SlenderEngine/searxng
    def _download_source(self) -> bytes:
        self._step(f"Downloading SearXNG {VERSION} ({SOURCE_BYTES / 2**20:.0f} MB)", "installing")
        req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "SlenderEngine-searxng-setup"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read(64 << 20)
        if hashlib.sha256(data).hexdigest() != SOURCE_SHA256:
            raise SetupError("the downloaded SearXNG source does not match its checksum; nothing was installed "
                             "(GitHub may have changed the archive: update SOURCE_SHA256 in serve/searxng_setup.py after checking it)")
        return data

    def _install_native(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        data = self._download_source()
        self._step("Unpacking")
        src, tmp = self.folder / "src", self.folder / "src.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            top = z.namelist()[0].split("/")[0]
            for name in z.namelist():                      # the archive's own folder is dropped; no path escapes
                rel = name[len(top) + 1:]
                if not rel or rel.startswith(("/", "\\")) or ".." in Path(rel).parts:
                    continue
                target = tmp / rel
                if name.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(name))
        shutil.rmtree(src, ignore_errors=True)
        tmp.rename(src)
        self._step("Creating a Python environment")
        venv = self.folder / "venv"
        shutil.rmtree(venv, ignore_errors=True)
        code, out = run([self.python, "-m", "venv", venv], timeout=300)
        if code != 0:
            raise SetupError(f"could not create a Python environment: {out.strip()[-400:]}")
        vpy = venv / ("Scripts/python.exe" if WINDOWS else "bin/python")
        self._step("Installing SearXNG's packages (about 95 MB; a minute or two)")
        code, out = run([vpy, "-m", "pip", "install", "--disable-pip-version-check", "--no-input", "-q", "-r",
                         src / "requirements.txt"], timeout=1800)
        if code != 0:
            raise SetupError(f"installing SearXNG's packages failed: {out.strip()[-600:]}")
        if WINDOWS:
            code, purelib = run([vpy, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"])
            Path(purelib.strip()).joinpath("pwd.py").write_text(PWD_STANDIN, encoding="utf-8")
        (self.folder / "settings.yml").write_text(settings_yml(self.port), encoding="utf-8")

    def _launch_native(self):
        vpy = self.folder / "venv" / ("Scripts/python.exe" if WINDOWS else "bin/python")
        # its own temp folder: SearXNG keeps its caches there, and a shared one may hold another SearXNG's files
        tmp = self.folder / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "SEARXNG_SETTINGS_PATH": str(self.folder / "settings.yml"), "PYTHONIOENCODING": "utf-8",
               "TMP": str(tmp), "TEMP": str(tmp), "TMPDIR": str(tmp)}
        log = open(self.folder / "searxng.log", "ab")
        self.proc = subprocess.Popen([str(vpy), "-m", "searx.webapp"], cwd=str(self.folder / "src"), env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     creationflags=NO_WINDOW | (0x00000200 if WINDOWS else 0))   # + NEW_PROCESS_GROUP
        log.close()

    # ---------------------------------------------------------------- WSL: the same venv inside the distribution
    def _install_wsl(self):
        distro = self._wsl_distro()
        if not distro:
            raise SetupError("WSL is not installed (run wsl --install as administrator, then restart Windows)")
        self.folder.mkdir(parents=True, exist_ok=True)
        settings = settings_yml(self.port)
        script = f'''set -e
D="{WSL_DIR}"; mkdir -p "$D"; cd "$D"
command -v python3 >/dev/null || {{ echo "NEED: python3 (sudo apt install python3 python3-venv)"; exit 3; }}
echo "STEP: Downloading SearXNG {VERSION}"
python3 - <<'PY'
import hashlib, io, os, shutil, urllib.request, zipfile
data = urllib.request.urlopen(urllib.request.Request("{SOURCE_URL}", headers={{"User-Agent": "SlenderEngine-searxng-setup"}}), timeout=60).read()
if hashlib.sha256(data).hexdigest() != "{SOURCE_SHA256}":
    raise SystemExit("CHECKSUM: the downloaded SearXNG source does not match its checksum")
shutil.rmtree("src", ignore_errors=True)
z = zipfile.ZipFile(io.BytesIO(data)); top = z.namelist()[0].split("/")[0]
for n in z.namelist():
    rel = n[len(top) + 1:]
    if not rel or ".." in rel.split("/"): continue
    if n.endswith("/"): os.makedirs("src/" + rel, exist_ok=True)
    else:
        os.makedirs(os.path.dirname("src/" + rel) or ".", exist_ok=True)
        open("src/" + rel, "wb").write(z.read(n))
PY
echo "STEP: Creating a Python environment"
rm -rf venv
python3 -m venv venv || {{ echo "NEED: python3-venv (sudo apt install python3-venv)"; exit 3; }}
echo "STEP: Installing SearXNG's packages (a minute or two)"
venv/bin/pip install --disable-pip-version-check --no-input -q -r src/requirements.txt
cat > settings.yml <<'YML'
{settings}YML
echo "DONE"
'''
        proc = subprocess.Popen(["wsl.exe", "-d", distro, "-e", "bash", "-c", script], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, errors="replace", creationflags=NO_WINDOW)
        tail = []
        for line in proc.stdout:
            line = line.strip()
            tail = (tail + [line])[-8:]
            if line.startswith("STEP: "):
                self._step(line[6:] + " (in WSL)", "installing")
        proc.wait()
        if proc.returncode != 0 or "DONE" not in tail:
            need = next((t for t in tail if t.startswith(("NEED:", "CHECKSUM:"))), None)
            raise SetupError(f"installing SearXNG in WSL failed: {need or ' | '.join(tail[-3:])}")

    def _launch_wsl(self):
        distro = self._wsl_distro()
        if not distro:
            raise SetupError("WSL is not available")
        # its own temp folder: SearXNG keeps its caches in /tmp, where another SearXNG (another user) may own them
        cmd = (f'mkdir -p "{WSL_DIR}/tmp" && cd "{WSL_DIR}/src" && TMPDIR="{WSL_DIR}/tmp" '
               f'SEARXNG_SETTINGS_PATH="{WSL_DIR}/settings.yml" exec "{WSL_DIR}/venv/bin/python" -m searx.webapp >> ../searxng.log 2>&1')
        # the wsl.exe process stays open while SearXNG runs: that keeps the distribution running
        self.proc = subprocess.Popen(["wsl.exe", "-d", distro, "-e", "sh", "-c", cmd], stdin=subprocess.DEVNULL,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)

    # ---------------------------------------------------------------- Docker: the official image, pinned
    def _install_docker(self):
        if not shutil.which("docker"):
            raise SetupError("Docker is not installed")
        code, _ = run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=15)
        if code != 0:
            raise SetupError("Docker is installed but not running: start Docker Desktop, then switch search on again")
        conf = self.folder / "docker"
        conf.mkdir(parents=True, exist_ok=True)
        (conf / "settings.yml").write_text(settings_yml(8080, bind="0.0.0.0"), encoding="utf-8")
        self._step("Downloading the SearXNG image (Docker)", "installing")
        code, out = run(["docker", "pull", IMAGE], timeout=1800)
        if code != 0:
            raise SetupError(f"docker pull failed: {out.strip()[-400:]}")
        run(["docker", "rm", "-f", CONTAINER], timeout=60)
        self._step("Creating the container")
        code, out = run(["docker", "create", "--name", CONTAINER, "--restart", "no", "-p", f"127.0.0.1:{self.port}:8080",
                         "--mount", f"type=bind,source={conf},target=/etc/searxng", IMAGE], timeout=120)
        if code != 0:
            raise SetupError(f"docker create failed: {out.strip()[-400:]}")

    def _launch_docker(self):
        code, out = run(["docker", "start", CONTAINER], timeout=120)
        if code != 0:
            raise SetupError(f"docker start failed: {out.strip()[-400:]}")
