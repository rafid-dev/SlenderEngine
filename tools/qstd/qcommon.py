"""Q-STD shared pieces (docs/fork/QUALITY.md): the local config, the launcher's settings read from the launcher itself,
the engine's serve protocol, cache identities and what the engine's log says it did.

Config: a JSON file named by --config or the QSTD_CONFIG environment variable (template: tools/qstd/qstd.example.json):
  launcher      the .bat the daily server is started with (its `set "STRATA_X=..."` lines are the settings)
  daily_config  the server config whose "args" / "exe" / "cwd" / "lib_dirs" the engine runs with
  sol_cases     the production-replay corpus (cases.json: id, stratum, domain, prompt_ids, continuation_ids, spans)
  bf16_dir      the BF16 reference set (reference/manifest.json, reference/*.f32, corpus/all-cases.json, vocab.json)
  work_dir      where the current reference (work_dir/ref) and the runs (work_dir/runs/<name>) live
  runlog_dir    optional: a folder with runlog.py (a run ledger imported for its side effect)
"""
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time

SCHEMA = "qstd-2"     # bump when what is stored or how it is scored changes: cached references then rebuild
Q = 256               # scored positions of a production-replay case
SAME_ENV = {"STRATA_IQ_MT_MIN": "1", "STRATA_PREFILL_CPU_SHARE": "0"}   # identical-output settings
DROP_VALUE = {"--expert-cache", "--adapt-every", "--adapt-swaps", "--pcie-frac", "--prompt-cache", "--max-context", "--short-read",
              "--conversation-cache-mib", "--conversation-cache-min-free-mib"}
DROP_FLAG = {"--serve", "--vision", "--session-save-reclaim"}


def load_config(path=None):
    path = path or os.environ.get("QSTD_CONFIG")
    if not path:
        sys.exit("Q-STD needs a config: --config FILE or QSTD_CONFIG (template: tools/qstd/qstd.example.json)")
    cfg = json.load(open(path, encoding="utf-8"))
    for k in ("launcher", "daily_config", "sol_cases", "bf16_dir", "work_dir"):
        if k not in cfg:
            sys.exit(f"config {path}: '{k}' missing")
    cfg = {k: (pathlib.Path(v) if isinstance(v, str) and k != "engine" else v) for k, v in cfg.items()}
    if cfg.get("runlog_dir"):
        sys.path.insert(0, str(cfg["runlog_dir"]))
        try:
            import runlog  # noqa: F401  ledger, recorded by import
        except ImportError:
            pass
    return cfg


def launcher_env(bat):
    """The STRATA_* settings the launcher sets, in order, with %VAR% expanded from earlier lines (rem lines ignored)."""
    env, local = {}, {}
    for line in open(bat, encoding="utf-8", errors="replace"):
        s = line.strip()
        m = re.match(r'(?i)^if not defined (\w+) set "(\w+)=([^"]*)"', s)
        if m:
            local.setdefault(m.group(2), m.group(3))
            continue
        m = re.match(r'(?i)^set "(\w+)=([^"]*)"', s)
        if not m:
            continue
        k, v = m.group(1), re.sub(r"%(\w+)%", lambda x: local.get(x.group(1), env.get(x.group(1), x.group(0))), m.group(2))
        local[k] = v
        if k.startswith("STRATA_") and not k.startswith("STRATA_LAUNCH_") or k == "PYTHONUTF8":
            env[k] = v
    return env


def daily(cfg):
    return json.load(open(cfg["daily_config"], encoding="utf-8-sig"))


def engine_exe(cfg):
    return str(cfg.get("engine") or daily(cfg)["exe"])


def engine_args(cfg, exe, short_read, extra):
    base, out, i = list(daily(cfg)["args"]), [], 0
    while i < len(base):
        x = base[i]
        if x in DROP_FLAG:
            i += 1
        elif x in DROP_VALUE:
            i += 2
        else:
            out.append(x)
            i += 1
    return [exe, "--serve", *out, "--expert-cache", "auto", "--adapt-every", "1000000", "--adapt-swaps", "1", "--pcie-frac", "0",
            "--max-context", "262144", "--prompt-cache", "6", "--short-read", str(short_read), *extra]


def build(cfg, exe, env_extra, args_extra, short_read):
    """The arm's full environment and arguments (STRATA_* variables inherited from this shell are not passed on)."""
    base = {k: v for k, v in os.environ.items() if not k.startswith("STRATA_")}
    env = {**base, **launcher_env(cfg["launcher"]), **SAME_ENV, **env_extra}
    env["PATH"] = os.pathsep.join(list(daily(cfg).get("lib_dirs") or []) + [env.get("PATH", "")])
    return engine_args(cfg, exe, short_read, args_extra), env


def sha_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sha_obj(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity(cfg, exe, env_extra, args_extra, short_read, payload_sha, scorer):
    """What makes a cached reference valid: engine binary, arguments, STRATA_* settings, server config, the cases' full payload,
    the scorer contract (schema, TOPK, audit rule, sampler, vocabulary)."""
    args, env = build(cfg, exe, env_extra, args_extra, short_read)
    return sha_obj({"exe": sha_file(exe), "args": args[1:], "env": {k: env[k] for k in sorted(env) if k.startswith("STRATA_")},
                    "cfg": sha_obj(daily(cfg)), "payload": payload_sha, "scorer": scorer})


class Engine:
    """One fresh engine in serve mode, driven over stdin (GEN 1 ids -> DONE / ERR), logits to STRATA_LOGITS_DUMP."""

    def __init__(self, cfg, exe, env_extra, args_extra, dump, errpath, short_read):
        self.args, env = build(cfg, exe, env_extra, args_extra, short_read)
        env["STRATA_LOGITS_DUMP"] = str(dump)
        self.env_used = {k: env[k] for k in sorted(env) if k.startswith("STRATA_") and k != "STRATA_LOGITS_DUMP"}
        self.errf = open(errpath, "w", encoding="utf-8", errors="replace")
        self.p = subprocess.Popen(self.args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errf, text=True,
                                  cwd=daily(cfg).get("cwd"), env=env, bufsize=1)
        self.ready = False
        for line in self.p.stdout:
            if line.startswith("READY"):
                self.ready = True
                break

    def gen(self, ids):
        self.p.stdin.write("GEN 1 " + ",".join(map(str, ids)) + "\n")
        self.p.stdin.flush()
        for line in self.p.stdout:
            if line.startswith(("DONE", "ERR")):
                return line.strip()
        return None

    def close(self):
        try:
            self.p.stdin.write("QUIT\n")
            self.p.stdin.flush()
            self.p.wait(timeout=120)
        except Exception:
            self.p.kill()
        if self.p.poll() is None:
            self.p.kill()
        self.errf.close()
        time.sleep(6)


def other_engine():
    return "strata.exe" in subprocess.run(["tasklist", "/FI", "IMAGENAME eq strata.exe", "/FO", "CSV", "/NH"], capture_output=True, text=True).stdout


def log_facts(errtext):
    """What the engine actually did: slots and the auto budget, chunk plans, prompt reads, expert tiers, non-finite and fallback lines."""
    reads = [(int(a), int(b), int(c)) for a, b, c in re.findall(r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read", errtext)]
    gpu_hits = [int(x) for x in re.findall(r"expert tiers: GPU (\d+) hits this request", errtext)]
    cold = [int(x) for x in re.findall(r"layers with a blob to read \((\d+) such experts", errtext)]
    return {"slots": (re.search(r"expert cache (\d+) slots", errtext) or [None, None])[1],
            "auto_budget": (re.search(r"expert cache auto: .*?-> (\d+) slots", errtext) or [None, None])[1],
            "chunks_used": sorted(set(re.findall(r"this prompt reads in chunks of (\d+)", errtext))),
            "reads": reads,
            "gpu_expert_hits": sum(gpu_hits), "ssd_experts": sum(cold),
            "nonfinite": errtext.count("non-finite"),
            "fallback": [l[:160] for l in errtext.splitlines() if re.search(r"fall(s|ing)? back|fallback|not available here", l, re.I)][:5],
            "windows_up_to": (re.search(r"windows up to (\d+) tokens", errtext) or [None, None])[1]}
