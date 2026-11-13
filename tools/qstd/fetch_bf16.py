"""Q-STD: fetch the BF16 reference subset of
jyohukuchan/qwen38-flash-next-bf16-logits (Qwen Community License 1.0) into research/bf16ref and check every file
against the dataset's own manifest (byte size + SHA256). Resumes partial files. Only the files listed below.
usage: python tools/qstd/fetch_bf16.py OUTDIR   (first put the dataset's manifest.json in OUTDIR)
"""
import hashlib
import json
import os
import subprocess
import sys

BASE = "https://huggingface.co/datasets/jyohukuchan/qwen38-flash-next-bf16-logits/resolve/main/"
if len(sys.argv) < 2:
    sys.exit("usage: python tools/qstd/fetch_bf16.py OUTDIR   (first put the dataset's manifest.json in OUTDIR)")
OUT = sys.argv[1]
CASES = ["agent_code_513", "arithmetic_65", "code_4097", "en_prose_129", "ja_prose_129", "natural_4097", "python_code_513",
         "rust_code_257", "sql_code_1025", "structured_257", "zh_tech_257", "wiki2_r0", "wiki2_r1"]
WANT = ["README.md", "compare.py", "vocab.json", "corpus/all-cases.json", "corpus/wiki2-scored-inputs.json", "reference/manifest.json",
        "results/kl-xestrata-vs-bf16.json", "results/kl-cand-noise.json", "results/engine-crosscheck.json"] + [f"reference/{c}.f32" for c in CASES]
man = {f["path"]: f for f in json.load(open(os.path.join(OUT, "manifest.json")))["files"]}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


total = sum(man[p]["bytes"] for p in WANT if p in man)
print(f"{len(WANT)} files, {total / 1e9:.2f} GB (manifest sizes)", flush=True)
bad = []
for p in WANT:
    dst = os.path.join(OUT, *p.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    m = man.get(p)
    if m and os.path.exists(dst) and os.path.getsize(dst) == m["bytes"] and sha(dst) == m["sha256"]:
        print("ok (already)", p, flush=True)
        continue
    r = subprocess.run(["curl", "-sSL", "--retry", "5", "-C", "-", "-o", dst, BASE + p])
    if r.returncode not in (0, 33):
        print("curl failed", r.returncode, p, flush=True)
        bad.append(p)
        continue
    if m is None:
        print("fetched (not in manifest)", p, os.path.getsize(dst), flush=True)
        continue
    ok = os.path.getsize(dst) == m["bytes"] and sha(dst) == m["sha256"]
    print(("ok " if ok else "MISMATCH ") + p, os.path.getsize(dst), flush=True)
    if not ok:
        bad.append(p)
print("DONE, bad:", bad, flush=True)
sys.exit(1 if bad else 0)
