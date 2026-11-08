"""tools/prompt_path_kl.py - teacher-forced KL of Qwen3.8-Flash-Next prompt-path settings (layer-major, dense MMQ, the
GEMM routes) against the chunk-major prompt path, on the same tokens: what a prompt-path setting does to the state a
prompt leaves.

Each arm is one `strata --serve` engine (the run config's exe and args, plus the arm's env / args). Request 1 reads the
first PREFIX tokens of the quant-quality set (data/quant_quality/v1, its five segments joined) through the batched
prompt path - the path under test. Request 2 sends the whole text: the PREFIX tokens are reused from the session and the
last SUFFIX tokens go through the verify windows (--short-read), where STRATA_LOGPOS writes the teacher-forced
log-probabilities. So every scored position depends on the state the prompt path left, and nothing else differs.

    python tools/prompt_path_kl.py --config strata-<model>.json --out-dir DIR [arm ...]
    python tools/quant_quality.py compare DIR/ref.tsv DIR/<arm>.tsv ...

Arms (ARMS below): ref = chunk-major prompt path, no MMQ, 512-token chunks; floor = the same with 256-token chunks (the
spread regrouping alone gives); sgemm = the GEMM inputs widened to FP32, an arithmetic MORE exact than the reference (how
far an equally valid rounding lands); lm, mmq, lm+mmq = the settings under test. Fixed in every arm, as quant_quality.py fixes
them: one GPU expert slot, no adaptation, no PCIe share, no suffix drafts, skip 0, CPU share 0.
"""
import argparse, json, os, pathlib, subprocess, sys, time

FORK = pathlib.Path(__file__).resolve().parent.parent
SET = FORK / "data" / "quant_quality" / "v1" / "set.ids.json"
PREFIX, SUFFIX = 4000, 1000
LM = {"STRATA_LM_HIP": "1", "STRATA_PREFILL_LAYER_MAJOR": "1", "STRATA_PREFILL_LM_TOKENS": "32768", "STRATA_PREFILL_LM_R": "host"}
ARMS = {
    "ref": ({"STRATA_PREFILL_LAYER_MAJOR": "0", "STRATA_DENSE_MMQ": "0"}, ["--prefill", "512"]),
    "floor": ({"STRATA_PREFILL_LAYER_MAJOR": "0", "STRATA_DENSE_MMQ": "0"}, ["--prefill", "256"]),
    "lm": ({**LM, "STRATA_DENSE_MMQ": "0"}, ["--prefill", "512"]),
    "mmq": ({"STRATA_PREFILL_LAYER_MAJOR": "0", "STRATA_DENSE_MMQ": "1"}, ["--prefill", "512"]),
    "lm+mmq": ({**LM, "STRATA_DENSE_MMQ": "1"}, ["--prefill", "512"]),
    "sgemm": ({"STRATA_PREFILL_LAYER_MAJOR": "0", "STRATA_DENSE_MMQ": "0", "STRATA_RDNA2_SGEMM": "1"}, ["--prefill", "512"]),
}
FIXED = ["--expert-cache", "1", "--adapt-every", "100000", "--pcie-frac", "0", "--suffix-draft", "0", "--conversation-cache-mib", "0"]
FIXED_ENV = {"STRATA_ROUTE_TAIL_SKIP": "0", "STRATA_PREFILL_CPU_SHARE": "0", "STRATA_PCIE_FRAC_DEFAULT": "old"}
DROP_VALUE = {"--expert-cache", "--adapt-every", "--pcie-frac", "--suffix-draft", "--max-context", "--prefill", "--short-read",
              "--conversation-cache-mib", "--draft-conf", "--spec"}
DROP_FLAG = {"--serve"}


def server_args(cfg):
    """The engine arguments serve/server.py builds for this config (drafter choice, user_args), without --serve."""
    sys.path.insert(0, str(FORK / "serve"))
    saved = sys.argv
    sys.argv = ["x"]
    try:
        import server   # noqa: E402 - serve/server.py, for its engine_args
    finally:
        sys.argv = saved
    return [x for x in server.engine_args(cfg, cfg["exe"], announce=False)]


def engine_cmd(cfg, extra):
    base, out, i = server_args(cfg), [], 0
    drop = DROP_VALUE | {extra[k] for k in range(0, len(extra), 2)}
    while i < len(base):
        if base[i] in DROP_FLAG:
            i += 1
        elif base[i] in drop:
            i += 2
        else:
            out.append(base[i]); i += 1
    return [cfg["exe"], "--serve", *out, *FIXED, *extra, "--max-context", str(PREFIX + SUFFIX + 1024),
            "--short-read", str(SUFFIX + 64), "--spec", "6"]


def gen(p, ids):
    p.stdin.write("GEN 1 " + ",".join(map(str, ids)) + "\n")
    p.stdin.flush()
    for line in p.stdout:
        if line.startswith(("DONE", "ERR")):
            return line.strip()
    return None


def run_arm(cfg, name, out_dir, ids):
    env_add, extra = ARMS[name]
    out = out_dir / f"{name}.tsv"
    if out.exists():
        out.unlink()
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    env.update(FIXED_ENV)
    env.update(env_add)
    env["STRATA_LOGPOS"] = str(out)
    env["STRATA_LOGPOS_TOPK"] = "64"
    path_add = os.pathsep.join(cfg.get("lib_dirs") or [])
    if path_add:
        env["PATH"] = path_add + os.pathsep + env.get("PATH", "")
    args = engine_cmd(cfg, extra)
    errf = open(str(out) + ".stderr.txt", "w")
    t0 = time.time()
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errf, text=True, cwd=cfg.get("cwd"),
                         env=env, bufsize=1)
    for line in p.stdout:
        if line.startswith("READY"):
            break
    else:
        raise SystemExit(f"{name}: the engine ended before READY; see {out}.stderr.txt")
    d1 = gen(p, ids[:PREFIX])
    lines1 = sum(1 for _ in open(out)) if out.exists() else 0
    d2 = gen(p, ids[:PREFIX + SUFFIX])
    lines2 = sum(1 for _ in open(out)) if out.exists() else 0
    p.stdin.write("QUIT\n"); p.stdin.flush()
    try:
        p.wait(timeout=300)
    except subprocess.TimeoutExpired:
        p.kill()
    errf.close()
    # keep only request 2's rows (request 1 is the batched path: it should write none)
    rows = open(out).read().splitlines()[lines1:] if out.exists() else []
    out.write_text("\n".join(rows) + "\n")
    (out_dir / f"{name}.tsv.seg.json").write_text(json.dumps({"segments": [{"name": "suffix", "first_line": 0, "end_line": len(rows)}]}))
    print(f"{name:8s} {time.time() - t0:6.0f} s | request 1: {d1} ({lines1} rows) | request 2: {d2} ({lines2 - lines1} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("arms", nargs="*", default=list(ARMS))
    a = ap.parse_args()
    cfg = json.load(open(a.config, encoding="utf-8-sig"))
    st = json.load(open(SET))
    ids = [t for s in st["segments"] for t in s["ids"]]
    if len(ids) < PREFIX + SUFFIX:
        raise SystemExit(f"the set has {len(ids)} tokens, {PREFIX + SUFFIX} needed")
    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in a.arms:
        run_arm(cfg, name, out_dir, ids)


if __name__ == "__main__":
    main()
