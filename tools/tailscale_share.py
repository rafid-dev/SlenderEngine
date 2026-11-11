"""tools/tailscale_share.py - open the web app to your own Tailscale devices, safely, in one step.

    python tools/tailscale_share.py status                     what is set up now (changes nothing)
    python tools/tailscale_share.py on  --config strata-x.json  share: HTTPS on your tailnet, with an API key
    python tools/tailscale_share.py off                        stop sharing

What `on` does (it shows the plan and asks first; --yes skips the question, --dry-run only shows it):
1. Checks that Tailscale is installed, signed in, and has MagicDNS and HTTPS certificates on.
2. Adds to the run config: an "api_key" (a new random one, unless the config has one), this PC's tailnet name to
   "allowed_hosts", and the shared address to "trusted_origins". A copy of the config is kept first
   (<name>.before-tailscale.json). The server reads these when it starts: restart it afterwards.
3. Runs `tailscale serve --bg --https=8443 http://127.0.0.1:<port>`. The server keeps listening on 127.0.0.1 only;
   Tailscale answers on https://<this-pc>.<tailnet>.ts.net:8443 inside your tailnet and passes requests on. No
   firewall rule, no router change, nothing on your local network or the internet.

It never touches another `tailscale serve` mapping (another HTTPS port, or the same port pointing elsewhere: it
stops instead), and it never uses `tailscale funnel` (the public internet). Standard library only.
See docs/TAILSCALE.md.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

WINDOWS = os.name == "nt"
OK, BAD, NOTE = "[ok]", "[!!]", "[..]"


# ------------------------------------------------------------------------------------------------ Tailscale
def find_tailscale() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    for guess in (r"C:\Program Files\Tailscale\tailscale.exe", "/usr/bin/tailscale", "/usr/local/bin/tailscale",
                  "/Applications/Tailscale.app/Contents/MacOS/Tailscale"):
        if Path(guess).is_file():
            return guess
    return None


def run(argv: list[str], timeout: float = 30.0) -> tuple[int, str]:
    try:
        r = subprocess.run(argv, capture_output=True, text=True, errors="replace", timeout=timeout,
                           creationflags=0x08000000 if WINDOWS else 0)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)


def ts_state(exe: str) -> dict:
    """This PC on the tailnet: {"running", "name", "magicdns", "https"} (never the other devices)."""
    code, out = run([exe, "status", "--json"])
    try:
        d = json.loads(out[out.index("{"):]) if "{" in out else {}
    except ValueError:
        d = {}
    me = d.get("Self") or {}
    name = str(me.get("DNSName") or "").rstrip(".").lower() or None
    return {"running": d.get("BackendState") == "Running", "state": d.get("BackendState") or "not running",
            "name": name, "magicdns": bool((d.get("CurrentTailnet") or {}).get("MagicDNSEnabled")),
            "https": bool(d.get("CertDomains"))}


def serve_map(exe: str, name: str | None) -> dict[int, dict[str, str]]:
    """This PC's `tailscale serve` HTTPS mappings: {port: {path: target}}."""
    code, out = run([exe, "serve", "status", "--json"])
    try:
        d = json.loads(out[out.index("{"):]) if "{" in out else {}
    except ValueError:
        return {}
    found: dict[int, dict[str, str]] = {}
    for hostport, web in (d.get("Web") or {}).items():
        host, _, port = hostport.rpartition(":")
        if name and host.lower().rstrip(".") != name:
            continue
        handlers = (web or {}).get("Handlers") or {}
        found[int(port or 443)] = {p: (h.get("Proxy") or h.get("Path") or h.get("Text") or "?") for p, h in handlers.items()}
    return found


# ------------------------------------------------------------------------------------------------ the server
def health(port: int, key: str | None = None, base: str | None = None) -> dict | None:
    url = (base or f"http://127.0.0.1:{port}") + "/health"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        with e:
            return {"http_status": e.code}
    except (OSError, ValueError):
        return None


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def as_list(v) -> list[str]:
    return [v] if isinstance(v, str) else list(v) if isinstance(v, list) else []


def origin_of(name: str, https_port: int) -> str:
    return f"https://{name}" if https_port == 443 else f"https://{name}:{https_port}"


def config_changes(cfg: dict, name: str, https_port: int, want_key: bool) -> tuple[dict, list[str], str | None]:
    """(new config, what changes in words, the new key or None)."""
    new, said, key = dict(cfg), [], None
    hosts = as_list(cfg.get("allowed_hosts"))
    if name not in [h.lower() for h in hosts]:
        new["allowed_hosts"] = hosts + [name]
        said.append(f'add "{name}" to "allowed_hosts" (the server answers to that name)')
    origins = as_list(cfg.get("trusted_origins"))
    origin = origin_of(name, https_port)
    if origin not in origins:
        new["trusted_origins"] = origins + [origin]
        said.append(f'add "{origin}" to "trusted_origins" (the shared page may use its tools and settings)')
    have = str(cfg.get("api_key") or "").strip()
    if want_key and (not have or have.lower() == "none"):
        key = "sk-slender-" + secrets.token_urlsafe(24)
        new["api_key"] = key
        said.append('set a new random "api_key" (every device must enter it once)')
    return new, said, key


def write_config(path: Path, cfg: dict):
    backup = path.with_name(path.stem + ".before-tailscale" + path.suffix)
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"{NOTE} kept a copy of the config: {backup}")
    raw = path.read_text(encoding="utf-8-sig")
    indent = 2 if "\n  \"" in raw else (4 if "\n    \"" in raw else None)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------------ commands
def need_tailscale() -> tuple[str, dict]:
    exe = find_tailscale()
    if not exe:
        sys.exit(f"{BAD} Tailscale is not installed. Install it from https://tailscale.com/download, sign in, and run this again.")
    st = ts_state(exe)
    if not st["running"] or not st["name"]:
        sys.exit(f"{BAD} Tailscale is installed but not connected ({st['state']}). Open Tailscale and sign in "
                 "(or run: tailscale up), then run this again.")
    return exe, st


def config_path(arg: str | None) -> Path | None:
    p = arg or os.environ.get("STRATA_LAUNCH_CONFIG")
    return Path(p).expanduser().resolve() if p else None


def cmd_status(a) -> int:
    exe = find_tailscale()
    print(f"{OK if exe else BAD} Tailscale: {'installed' if exe else 'not installed (https://tailscale.com/download)'}")
    st = ts_state(exe) if exe else {"running": False, "name": None, "magicdns": False, "https": False, "state": "-"}
    if exe:
        print(f"{OK if st['running'] else BAD} connected: {'yes' if st['running'] else st['state']}")
        print(f"{OK if st['magicdns'] else BAD} MagicDNS: {'on' if st['magicdns'] else 'off (turn it on in the admin console: DNS)'}")
        print(f"{OK if st['https'] else BAD} HTTPS certificates: {'on' if st['https'] else 'off (admin console: DNS > HTTPS Certificates)'}")
        if st["name"]:
            print(f"{NOTE} this PC on the tailnet: {st['name']}")
            mapping = serve_map(exe, st["name"])
            mine = mapping.get(a.https_port, {})
            target = f"http://127.0.0.1:{a.port}"
            if mine.get("/") == target:
                print(f"{OK} shared: {origin_of(st['name'], a.https_port)}/ -> {target}")
            elif mine:
                print(f"{BAD} HTTPS port {a.https_port} is used for something else: {mine}")
            else:
                print(f"{NOTE} not shared on HTTPS port {a.https_port} (run: on)")
            others = {p: m for p, m in mapping.items() if p != a.https_port}
            if others:
                print(f"{NOTE} other `tailscale serve` mappings on this PC (left alone): " +
                      ", ".join(f"{p} -> {m.get('/', m)}" for p, m in sorted(others.items())))
    h = health(a.port)
    if h is None:
        print(f"{NOTE} no server answers on 127.0.0.1:{a.port} (start it first)")
    elif "http_status" in h:
        print(f"{NOTE} the server on 127.0.0.1:{a.port} answered HTTP {h['http_status']}")
    else:
        print(f"{OK} server on 127.0.0.1:{a.port}: {h.get('model')}, API key {'on' if h.get('api_key') else 'OFF'}")
    cp = config_path(a.config)
    if cp and cp.is_file():
        cfg = load_config(cp)
        key = bool(str(cfg.get("api_key") or "").strip())
        print(f"{OK if key else BAD} config {cp.name}: api_key {'set' if key else 'not set'}"
              + (f", allowed_hosts {as_list(cfg.get('allowed_hosts'))}" if cfg.get("allowed_hosts") else ""))
    return 0


def cmd_on(a) -> int:
    exe, st = need_tailscale()
    name = st["name"]
    if not st["https"]:
        print(f"{BAD} HTTPS certificates are off for your tailnet. Turn them on once in the Tailscale admin console "
              "(DNS > HTTPS Certificates: Enable), then run this again.")
        return 1
    target = f"http://127.0.0.1:{a.port}"
    mapping = serve_map(exe, name)
    mine = mapping.get(a.https_port, {})
    if mine and mine.get("/") != target:
        print(f"{BAD} HTTPS port {a.https_port} on this PC already shares something else ({mine}). It is left alone: "
              f"pick another port with --https-port, or free it yourself (tailscale serve --https={a.https_port} off).")
        return 1
    cp = config_path(a.config)
    plan, new_cfg, key = [], None, None
    if cp:
        if not cp.is_file():
            print(f"{BAD} no config at {cp}")
            return 1
        cfg = load_config(cp)
        new_cfg, said, key = config_changes(cfg, name, a.https_port, want_key=not a.no_key)
        plan += [f"in {cp.name}: {x}" for x in said]
        if a.no_key and not str(cfg.get("api_key") or "").strip():
            print(f"{BAD} --no-key: anyone who can reach this PC on your tailnet can use the model and its tools. "
                  "Fine for a tailnet that is only you; not for a shared one.")
    else:
        print(f"{NOTE} no --config given (or STRATA_LAUNCH_CONFIG): the run config is not changed. Without an API key and "
              f'without "{name}" in "allowed_hosts", the server refuses the shared address (HTTP 403).')
    if not mine:
        plan.append(f"run: tailscale serve --bg --https={a.https_port} {target}")
    url = origin_of(name, a.https_port) + "/"
    if not plan:
        print(f"{OK} already shared at {url} with the config in place.")
        return 0
    print("This will:")
    for x in plan:
        print(f"  - {x}")
    print(f"Then the web app is at {url} on your Tailscale devices only.")
    if a.dry_run:
        print(f"{NOTE} dry run: nothing changed.")
        return 0
    if not a.yes:
        try:
            if input("Go ahead? [y/N] ").strip().lower() not in ("y", "yes"):
                print("Nothing changed.")
                return 1
        except EOFError:
            print("Nothing changed (no answer; use --yes to skip the question).")
            return 1
    changed_config = False
    if new_cfg is not None and new_cfg != load_config(cp):
        write_config(cp, new_cfg)
        changed_config = True
        print(f"{OK} config updated: {cp}")
    if not mine:
        code, out = run([exe, "serve", "--bg", "--yes", f"--https={a.https_port}", target], timeout=60)
        if code != 0 and "flag provided but not defined: -yes" in out:      # Tailscale before --yes existed
            code, out = run([exe, "serve", "--bg", f"--https={a.https_port}", target], timeout=60)
        if code != 0:
            print(f"{BAD} tailscale serve failed:\n{out.strip()}")
            if "access denied" in out.lower() or "operator" in out.lower():
                print(f"{NOTE} on Linux, allow your user once: sudo tailscale set --operator=$USER")
            return 1
        print(f"{OK} Tailscale shares {target} at {url}")
    h = health(a.port)
    if h is None:
        print(f"{NOTE} no server runs on 127.0.0.1:{a.port} yet: start it, then open {url}")
    elif changed_config:
        print(f"{BAD} restart the server now (close its window, start it again): it reads the key and the host name only "
              "when it starts. Until then the shared address is refused.")
    if key:
        print(f"\nAPI key (shown once; it is also in {cp.name}):\n\n    {key}\n")
        print("On each device: open the address, then Settings > Connection > API key, and paste it.")
    print(f"\nShared at: {url}\nStop sharing: python tools/tailscale_share.py off")
    return 0


def cmd_off(a) -> int:
    exe, st = need_tailscale()
    mine = serve_map(exe, st["name"]).get(a.https_port, {})
    target = f"http://127.0.0.1:{a.port}"
    if not mine:
        print(f"{OK} nothing is shared on HTTPS port {a.https_port}.")
    elif mine.get("/") != target and not a.force:
        print(f"{BAD} HTTPS port {a.https_port} shares something else ({mine}); left alone (--force turns it off anyway).")
        return 1
    else:
        code, out = run([exe, "serve", f"--https={a.https_port}", "off"])
        if code != 0:
            print(f"{BAD} tailscale serve could not stop it:\n{out.strip()}")
            return 1
        print(f"{OK} stopped sharing {origin_of(st['name'], a.https_port)}/")
    cp = config_path(a.config)
    if a.forget and cp and cp.is_file():
        cfg = load_config(cp)
        origin = origin_of(st["name"], a.https_port)
        new = dict(cfg)
        new["allowed_hosts"] = [h for h in as_list(cfg.get("allowed_hosts")) if h.lower() != st["name"]]
        new["trusted_origins"] = [o for o in as_list(cfg.get("trusted_origins")) if o != origin]
        for k in ("allowed_hosts", "trusted_origins"):
            if not new[k]:
                new.pop(k)
        if new != cfg:
            write_config(cp, new)
            print(f"{OK} removed the tailnet name from {cp.name} (the api_key stays). Restart the server to apply it.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Share the SlenderEngine web app with your own Tailscale devices (HTTPS, tailnet only).")
    ap.add_argument("command", nargs="?", default="status", choices=["status", "on", "off"])
    ap.add_argument("--port", type=int, default=8080, help="the server's port on 127.0.0.1 (default 8080)")
    ap.add_argument("--https-port", type=int, default=8443, help="the HTTPS port on the tailnet (default 8443; 443 if it is free)")
    ap.add_argument("--config", help="the run config the server starts with (default: $STRATA_LAUNCH_CONFIG)")
    ap.add_argument("--no-key", action="store_true", help="do not add an API key (only for a tailnet that is you alone)")
    ap.add_argument("--yes", action="store_true", help="do not ask before changing anything")
    ap.add_argument("--dry-run", action="store_true", help="show what would change, change nothing")
    ap.add_argument("--forget", action="store_true", help="off: also remove the tailnet name from the config")
    ap.add_argument("--force", action="store_true", help="off: turn the HTTPS port off even if it shares something else")
    a = ap.parse_args(argv)
    return {"status": cmd_status, "on": cmd_on, "off": cmd_off}[a.command](a)


if __name__ == "__main__":
    sys.exit(main())
