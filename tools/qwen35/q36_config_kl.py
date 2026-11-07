"""tools/qwen35/q36_config_kl.py - teacher-forced KL between strata-q36 configurations (environment switches and
--prefill), against a reference configuration, on the same tokens.

Each configuration reads the same prompt (in --prefill-sized chunks, so the prompt path runs), then the forced tokens;
--dump-full-logits writes the next-token logits of every forced step. The expert placement is pinned in every run
(--expert-slots, STRATA_Q36_FIXED_SLOTS=1, STRATA_Q36_XSLOTS=0), so a run differs from the reference only in what the
configuration changes. A configuration that only regroups the prompt (another --prefill) measures the floor: the
spread that the order of sums alone gives.

    python tools/qwen35/q36_config_kl.py --engine ENGINE --model GGUF --prompt IDS_FILE --force IDS_FILE

IDS files hold comma-separated token ids (3000 prompt tokens and 512 forced ones were used on 2026-11-07). Prints the
mean, 99th-percentile and largest KL(ref || cfg) over the forced steps and the top-1 agreement with the reference.
"""
import argparse
import os
import subprocess
import tempfile

import numpy as np

V = 248320
BASE = {"STRATA_Q36_FIXED_SLOTS": "1", "STRATA_Q36_XSLOTS": "0", "STRATA_Q8_PACKED": "0"}
CONFIGS = [   # (name, env, prompt chunk); the first is the reference
    ("ref: fp32 KV, FP16 GEMMs, chunk 2048", {"STRATA_Q36_KV_F16": "0", "STRATA_DENSE_MMQ": "0"}, 2048),
    ("floor: the same, chunk 1024", {"STRATA_Q36_KV_F16": "0", "STRATA_DENSE_MMQ": "0"}, 1024),
    ("f16 KV", {"STRATA_Q36_KV_F16": "1", "STRATA_DENSE_MMQ": "0"}, 2048),
    ("dense MMQ", {"STRATA_Q36_KV_F16": "0", "STRATA_DENSE_MMQ": "1"}, 2048),
    ("f16 KV + dense MMQ", {"STRATA_Q36_KV_F16": "1", "STRATA_DENSE_MMQ": "1"}, 2048),
]


def logsm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=1, keepdims=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="strata.exe / strata")
    ap.add_argument("--model", required=True, help="the Qwen3.6-35B-A3B GGUF")
    ap.add_argument("--prompt", required=True, help="file with the prompt's token ids")
    ap.add_argument("--force", required=True, help="file with the forced tokens' ids")
    ap.add_argument("--slots", type=int, default=4, help="--expert-slots for every run (4)")
    ap.add_argument("--ctx", type=int, default=8192)
    a = ap.parse_args()
    prompt = open(a.prompt).read().strip()
    force = open(a.force).read().strip()
    tmp = tempfile.mkdtemp(prefix="q36kl")
    ref = None
    for i, (name, env, chunk) in enumerate(CONFIGS):
        out = os.path.join(tmp, f"logits_{i}.bin")
        cmd = [a.engine, "--native", a.model, "--max-context", str(a.ctx), "--prefill", str(chunk), "--expert-slots",
               str(a.slots), "--tokens", prompt, "--force", force, "--dump-full-logits", out]
        r = subprocess.run(cmd, env=dict(os.environ, **BASE, **env), capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f"{name}: the engine failed\n{r.stderr[-2000:]}")
        q = logsm(np.fromfile(out, dtype=np.float32).reshape(-1, V))
        if ref is None:
            ref = q
            print(f"{name}: {len(q)} steps")
            continue
        n = min(len(q), len(ref))
        kl = (np.exp(ref[:n]) * (ref[:n] - q[:n])).sum(axis=1)
        top = (ref[:n].argmax(1) == q[:n].argmax(1)).mean()
        print(f"{name:32s} mean KL {kl.mean():.6f}  p99 {np.percentile(kl, 99):.5f}  max {kl.max():.4f}  "
              f"same top-1 {100 * top:.1f}%")


if __name__ == "__main__":
    main()
