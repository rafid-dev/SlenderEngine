"""Q-STD: how close is the compact KL (R's top-1024 + one tail bucket) to the exact full-vocabulary KL?
Real reference rows (a full dump left from the harness check), candidates made by perturbing them: Gaussian noise on all
logits at several sizes (the realistic case), and noise on ranks past 1024 only (the case the compact form sees least).
usage: python tools/qstd/verify_compact.py DUMP VOCAB_JSON [n_rows]
  DUMP: a file written with STRATA_LOGITS_DUMP=<file> (records {magic, T, n_vocab, record, pos} + rows)
"""
import sys

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import qscore as qs  # noqa: E402
import json  # noqa: E402
import struct  # noqa: E402

dump = sys.argv[1]
vj = json.load(open(sys.argv[2], encoding="utf-8"))
V = np.asarray(vj["valid_token_ids"] if isinstance(vj, dict) else vj, dtype=np.int64)
n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
rows = []
H = struct.Struct("<iiiiq")
with open(dump, "rb") as f:
    while len(rows) < n:
        h = f.read(H.size)
        if len(h) < H.size:
            break
        _, T, nv, _, _ = H.unpack(h)
        a = np.frombuffer(f.read(T * nv * 4), dtype=np.float32).reshape(T, nv)
        rows.extend(a[t] for t in range(T))
rows = rows[:n]
rng = np.random.default_rng(7)
print(f"{len(rows)} real reference rows; KL exact (full valid vocabulary, FP64) vs compact (top-{qs.TOPK} + tail)")
for label, sigma, tail_only in (("noise 0.01", 0.01, False), ("noise 0.03", 0.03, False), ("noise 0.1", 0.1, False), ("noise 0.3", 0.3, False),
                                ("tail-only noise 0.5", 0.5, True), ("tail-only noise 2.0", 2.0, True)):
    ex, cp = [], []
    for r in rows:
        noise = rng.normal(0, sigma, r.shape).astype(np.float32)
        if tail_only:
            lp, _ = qs.logsm(r, V)
            top = np.argpartition(-lp, qs.TOPK)[:qs.TOPK]
            mask = np.ones(r.shape, bool)
            mask[V[top]] = False
            noise = noise * mask
        c = r + noise
        lp, _ = qs.logsm(r, V)
        lq, _ = qs.logsm(c, V)
        ex.append(float((np.exp(lp) * (lp - lq)).sum()))
        cp.append(qs.cmp_record(qs.ref_record(r, V, -1), c, V, -1)["kl"])
    ex, cp = np.array(ex), np.array(cp)
    rel = np.abs(ex - cp) / np.maximum(ex, 1e-30)
    print(f"  {label:22s} exact mean {ex.mean():.3e}  compact mean {cp.mean():.3e}  max abs diff {np.abs(ex - cp).max():.2e}  "
          f"median rel diff {np.median(rel) * 100:.3f}%  max rel diff {rel.max() * 100:.2f}%")
