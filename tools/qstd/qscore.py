"""Q-STD row scoring (docs/fork/QUALITY.md). Everything is computed in FP64 over the valid vocabulary (the BF16 set's vocab.json).

Reference (R), per scored position (compact, so nothing large is stored):
  - the top TOPK token ids with their log probabilities, and the log probability of all the other tokens together (the tail
    bucket, computed with a log-sum-exp over the discarded tokens, not as 1 - head);
  - R's top-1 id, top probability and top1-top2 gap, the NLL of the supplied next token, the invalid-id mass;
  - the sampler's actual distribution (temperature / top-k / top-p of the model card): its ids and probabilities;
  - a SHA-256 of the raw FP32 row (byte identity is decided on this, never on KL);
  - on AUDIT rows only: the full valid-vocabulary log-probability vector in FP16 (the head stays FP64).
Candidate (C), against R:
  - KL_compact(R||C): R's head + the tail bucket. By the chain rule this is a LOWER bound of the exact KL; the missing part is
    P(tail) * KL(P(.|tail) || Q(.|tail)) and is not bounded by the tail mass. It is a screening value;
  - KL_exact on audit rows (FP64 head + FP16 rest): how much the compact value missed, per candidate;
  - sampler TV: total variation between R's and C's actual sampler distributions (exact: both supports are small);
  - top-1 agreement, NLL of the supplied token, invalid-id mass, byte identity.
BF16 lane: KL(BF16 || C) exact over the valid vocabulary, from the full BF16 rows on disk.
"""
import hashlib

import numpy as np

TOPK = 2048   # 24 KB a position held in RAM while the engine runs; median compact error ~0.2% (verify_compact.py); audit rows check the rest
AUDIT_UNIFORM, AUDIT_TAIL = 4, 4        # audit rows per case: evenly spaced + the highest tail mass
SAMPLER = {"temperature": 0.6, "top_k": 20, "top_p": 0.95}   # the model card's sampling (Qwen3.8 thinking mode)


def contract():
    return {"topk": TOPK, "audit": [AUDIT_UNIFORM, AUDIT_TAIL], "sampler": SAMPLER, "estimator": "compact-head+tail-bucket, fp64; audit fp64 head + fp16 rest"}


def lse(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def logsm(row, vocab):
    x = row.astype(np.float64)
    lse_full = lse(x)
    v = x[vocab]
    lse_v = lse(v)
    return v - lse_v, float(1.0 - np.exp(lse_v - lse_full))


def target_col(vocab, tok):
    if tok is None or tok < 0:
        return -1
    c = int(np.searchsorted(vocab, tok))
    return c if c < len(vocab) and vocab[c] == tok else -1


def sampler_dist(lp):
    """The sampler's distribution over valid-vocabulary columns: temperature, then top-k, then top-p (at least one token)."""
    k = SAMPLER["top_k"]
    idx = np.argpartition(-lp, k)[:k]
    s = lp[idx] / SAMPLER["temperature"]
    order = np.argsort(-s)
    idx, s = idx[order], s[order]
    p = np.exp(s - lse(s))
    keep = int(np.searchsorted(np.cumsum(p), SAMPLER["top_p"]) + 1)
    idx, p = idx[:keep], p[:keep]
    return idx.astype(np.int32), p / p.sum()


def tv(ia, pa, ib, pb):
    d = dict(zip(ia.tolist(), pa.tolist()))
    for i, q in zip(ib.tolist(), pb.tolist()):
        d[i] = d.get(i, 0.0) - q
    return 0.5 * sum(abs(v) for v in d.values())


def ref_record(row, vocab, tok):
    lp, inv = logsm(row, vocab)
    top = np.argpartition(-lp, TOPK)[:TOPK]
    top = top[np.argsort(-lp[top])]
    mask = np.ones(len(lp), bool)
    mask[top] = False
    tail_lp = lse(lp[mask])
    p2 = np.exp(lp[top[:2]])
    c = target_col(vocab, tok)
    sid, sp = sampler_dist(lp)
    return {"ids": top.astype(np.int32), "lp": lp[top], "tail_lp": float(tail_lp), "top1": int(top[0]), "ptop": float(p2[0]),
            "gap": float(p2[0] - p2[1]), "nll": float(-lp[c]) if c >= 0 else np.nan, "inv": inv, "sid": sid, "sp": sp,
            "sha": hashlib.sha256(row.tobytes()).digest(), "_lp": lp}


def audit_vector(ref):
    return ref["_lp"].astype(np.float16)


def cmp_record(ref, row, vocab, tok, audit_lp16=None):
    lq, inv = logsm(row, vocab)
    ids, lpr = ref["ids"], ref["lp"]
    head = lq[ids]
    mask = np.ones(len(lq), bool)
    mask[ids] = False
    q_tail_lp = lse(lq[mask])
    kl = float((np.exp(lpr) * (lpr - head)).sum() + np.exp(ref["tail_lp"]) * (ref["tail_lp"] - q_tail_lp))
    c = target_col(vocab, tok)
    sid, sp = sampler_dist(lq)
    out = {"kl": max(kl, 0.0), "top": int(np.argmax(lq)) == ref["top1"], "ptop": ref["ptop"], "gap": ref["gap"], "nll_r": ref["nll"],
           "nll_c": float(-lq[c]) if c >= 0 else np.nan, "inv": inv, "ident": hashlib.sha256(row.tobytes()).digest() == ref["sha"],
           "tv": tv(ref["sid"], ref["sp"], sid, sp), "kl_audit": np.nan}
    if audit_lp16 is not None:
        lr = audit_lp16.astype(np.float64)
        lr[ids] = lpr                                   # the head exactly
        lr -= lse(lr)
        out["kl_audit"] = float(max((np.exp(lr) * (lr - lq)).sum(), 0.0))
    return out


def bf16_record(brow, row, vocab, tok):
    lb, _ = logsm(brow, vocab)
    lq, _ = logsm(row, vocab)
    pb = np.exp(lb)
    c = target_col(vocab, tok)
    return {"kl": float(max((pb * (lb - lq)).sum(), 0.0)), "top": int(np.argmax(lb)) == int(np.argmax(lq)), "ptop": float(pb.max()),
            "nll_b": float(-lb[c]) if c >= 0 else np.nan, "nll_c": float(-lq[c]) if c >= 0 else np.nan}


def save_ref(path, recs, audit):
    """recs: {(case, pos): ref_record}; audit: {(case, pos): fp16 vector}. One npz, replaced (never accumulated)."""
    keys = sorted(recs)
    ak = sorted(audit)
    S = max(len(recs[k]["sid"]) for k in keys)
    sid = np.full((len(keys), S), -1, np.int32)
    sp = np.zeros((len(keys), S))
    for n, k in enumerate(keys):
        sid[n, :len(recs[k]["sid"])] = recs[k]["sid"]
        sp[n, :len(recs[k]["sp"])] = recs[k]["sp"]
    tmp = str(path) + ".tmp.npz"
    np.savez_compressed(tmp, case=np.array([k[0] for k in keys], np.int32), pos=np.array([k[1] for k in keys], np.int64),
                        ids=np.stack([recs[k]["ids"] for k in keys]), lp=np.stack([recs[k]["lp"] for k in keys]),
                        tail_lp=np.array([recs[k]["tail_lp"] for k in keys]), top1=np.array([recs[k]["top1"] for k in keys], np.int32),
                        ptop=np.array([recs[k]["ptop"] for k in keys]), gap=np.array([recs[k]["gap"] for k in keys]),
                        nll=np.array([recs[k]["nll"] for k in keys]), inv=np.array([recs[k]["inv"] for k in keys]), sid=sid, sp=sp,
                        sha=np.frombuffer(b"".join(recs[k]["sha"] for k in keys), dtype=np.uint8).reshape(len(keys), 32),
                        audit_case=np.array([k[0] for k in ak], np.int32), audit_pos=np.array([k[1] for k in ak], np.int64),
                        audit_lp=np.stack([audit[k] for k in ak]) if ak else np.zeros((0, 1), np.float16))
    import os
    os.replace(tmp, path)


def load_ref(path):
    # Each z[name] decompresses the whole array again, so read every array exactly once (indexing z[name][i] per row held a
    # full copy per row: 14.7 GB for 768 rows on 2026-11-13). Rows are views into these single arrays.
    with np.load(path) as zf:
        z = {k: zf[k] for k in zf.files}
    out = {}
    for i in range(len(z["case"])):
        n = int((z["sid"][i] >= 0).sum())
        out[(int(z["case"][i]), int(z["pos"][i]))] = {"ids": z["ids"][i], "lp": z["lp"][i], "tail_lp": float(z["tail_lp"][i]),
                                                     "top1": int(z["top1"][i]), "ptop": float(z["ptop"][i]), "gap": float(z["gap"][i]),
                                                     "nll": float(z["nll"][i]), "inv": float(z["inv"][i]), "sid": z["sid"][i][:n],
                                                     "sp": z["sp"][i][:n], "sha": z["sha"][i].tobytes()}
    audit = {(int(c), int(p)): z["audit_lp"][j] for j, (c, p) in enumerate(zip(z["audit_case"], z["audit_pos"]))}
    return out, audit
