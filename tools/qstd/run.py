"""Q-STD runner (docs/fork/QUALITY.md). The engine's logits stream through a named pipe and are scored as they arrive: nothing
but compact results reaches the disk. A broken run is marked invalid and can never pass.

lanes:
  sol   production replay: the corpus' prompts through the prompt path, their 256 supplied tokens through verify windows
  bf16  every position of the BF16 reference cases through verify windows; also scored against the BF16 logits (exact)
arms: tools/qstd/arms.json (R = the accepted setup, R2 its repeat, P* prompt-chunk controls, D*/X* placement stress) plus
candidates from --cand FILE ({"name": {"env": {...}, "args": [...], "exe": optional, "unpinned": optional, "kind": ...}}).
R leaves the compact reference in WORK/ref/R.<lane>.ref.npz (replaced, never accumulated); every other arm is scored
against it into WORK/runs/<run>/<arm>.<lane>.npz.

usage: python tools/qstd/run.py RUN --lane sol|bf16 --arms R,R2,P1 [--cand cand.json] [--config cfg.json] [--cases a,b]
  RUN    a run name (WORK/runs/RUN) or a folder path
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcommon as qc  # noqa: E402
import qscore as qs  # noqa: E402
from pipe import PipeReader  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--lane", required=True, choices=["sol", "bf16"])
ap.add_argument("--arms", required=True)
ap.add_argument("--cand", default=None)
ap.add_argument("--config", default=None)
ap.add_argument("--ref", default=None, help="reference folder (default WORK/ref)")
ap.add_argument("--cases", default=None, help="comma list of case ids: a harness check only, never a verdict")
a = ap.parse_args()
os.environ.setdefault("RUNLOG_Q", "QSTD")
CFG = qc.load_config(a.config)
WORK = pathlib.Path(CFG["work_dir"])
OUT = pathlib.Path(a.run) if any(c in a.run for c in "/\\:") else WORK / "runs" / a.run
OUT, REF = OUT.resolve(), pathlib.Path(a.ref or WORK / "ref").resolve()
OUT.mkdir(parents=True, exist_ok=True)
REF.mkdir(parents=True, exist_ok=True)
ARMS = json.load(open(pathlib.Path(__file__).with_name("arms.json"), encoding="utf-8"))["arms"]
if a.cand:
    ARMS.update(json.load(open(a.cand, encoding="utf-8")))
BF16 = pathlib.Path(CFG["bf16_dir"])
VJ = json.load(open(BF16 / "vocab.json", encoding="utf-8"))
VOCAB = np.asarray(VJ["valid_token_ids"] if isinstance(VJ, dict) else VJ, dtype=np.int64)
assert np.all(VOCAB[1:] > VOCAB[:-1])

# ---- the lane's cases and, per (case, phase), the scored positions and their next tokens
if a.lane == "sol":
    CASES = json.load(open(CFG["sol_cases"], encoding="utf-8"))
    SHORT_READ = qc.Q + 64
else:
    inputs = {c["id"]: c for c in json.load(open(BF16 / "corpus" / "all-cases.json", encoding="utf-8"))["cases"]}
    man = {c["id"]: c for c in json.load(open(BF16 / "reference" / "manifest.json", encoding="utf-8"))["cases"]}
    have = [i for i in man if (BF16 / "reference" / man[i]["logits_file"]).exists() and i in inputs]
    CASES = [{"id": i, "ids": inputs[i]["token_ids"], "positions": man[i]["positions"], "file": man[i]["logits_file"],
              "shape": man[i]["shape"], "prefix": 55 if i.startswith("wiki2_") else 0} for i in have]
    SHORT_READ = max(len(c["ids"]) for c in CASES) + 64
if a.cases:
    keep = set(a.cases.split(","))
    CASES = [c for c in CASES if c["id"] in keep]
WANT = {}      # (case, phase) -> {position: next token}
for i, c in enumerate(CASES):
    if a.lane == "sol":
        P, cont = len(c["prompt_ids"]), c["continuation_ids"][:qc.Q]
        WANT[(i, 0)] = {P - 1: cont[0]}
        WANT[(i, 1)] = {p: cont[p - (P - 1)] for p in range(P, P + qc.Q - 1)}
    else:
        WANT[(i, 0)] = {p: (c["ids"][p + 1] if p + 1 < len(c["ids"]) else -1) for p in c["positions"] if p >= c["prefix"]}
EXPECTED = {(k[0], p) for k, v in WANT.items() for p in v}
# the full payload (token ids, scored positions, targets), not just the case names: changed contents invalidate the cache
PAYLOAD_SHA = qc.sha_obj({"lane": a.lane, "cases": [c["id"] for c in CASES], "want": {f"{k[0]}:{k[1]}": v for k, v in WANT.items()},
                          "ids": [c.get("prompt_ids", c.get("ids")) for c in CASES]})
SCORER = {"schema": qc.SCHEMA, **qs.contract(), "vocab": qc.sha_obj(VOCAB.tolist())}
UNIFORM = {}   # case -> evenly spaced audit positions
for i in range(len(CASES)):
    pos = sorted(p for (c, p) in EXPECTED if c == i)
    UNIFORM[i] = {pos[int(round(j))] for j in np.linspace(0, len(pos) - 1, qs.AUDIT_UNIFORM)}
BROW = {}


def brow(ci, p):
    if ci not in BROW:
        c = CASES[ci]
        BROW[ci] = (np.memmap(BF16 / "reference" / c["file"], dtype="<f4", mode="r", shape=tuple(c["shape"])), {q: k for k, q in enumerate(c["positions"])})
    m, idx = BROW[ci]
    return np.asarray(m[idx[p]], dtype=np.float32)


def check_reads(rec):
    """From the log: every request present; on the sol lane the 256 tokens read through the windows after a reuse."""
    reads = rec["facts"]["reads"]
    want = len(rec["cases"]) * (2 if a.lane == "sol" else 1)
    if len(reads) != want:
        rec["invalid"].append(f"{len(reads)} prompt-read lines in the log, {want} expected")
        return
    for k, c in enumerate(rec["cases"]):
        if a.lane == "sol":
            r1, r2 = reads[2 * k], reads[2 * k + 1]
            if r1[0] != c["P"]:
                rec["invalid"].append(f"{c['id']}: first request read {r1[0]} tokens, {c['P']} expected")
            if r2[0] != c["P"] + qc.Q or r2[1] < 1 or r2[2] > SHORT_READ:
                rec["invalid"].append(f"{c['id']}: second request {r2} (no reuse or not through the windows)")
        elif reads[k][0] != c["P"] or reads[k][2] > SHORT_READ:
            rec["invalid"].append(f"{c['id']}: request {reads[k]}")
    if rec["facts"]["nonfinite"]:
        rec["invalid"].append(f"{rec['facts']['nonfinite']} non-finite lines")


def run_arm(name, spec, is_ref, ref, audit_in):
    if qc.other_engine():
        sys.exit("another strata.exe is running: stopped before " + name)
    got, bgot, dup, audit, heaps = {}, {}, [], {}, {}

    def on_row(tag, pos, row):
        want = WANT.get(tag)
        if want is None or pos not in want:
            return                                    # an unscored row (P+255, a re-read prompt tail)
        key = (tag[0], pos)
        if key in got:
            dup.append(key)
            return
        tok = want[pos]
        if is_ref:
            r = qs.ref_record(row, VOCAB, tok)
            if pos in UNIFORM[tag[0]]:
                audit[key] = qs.audit_vector(r)
            else:                                     # keep the AUDIT_TAIL highest-tail rows of this case
                h = heaps.setdefault(tag[0], [])
                if len(h) < qs.AUDIT_TAIL or r["tail_lp"] > min(x[0] for x in h):
                    if len(h) >= qs.AUDIT_TAIL:
                        h.remove(min(h, key=lambda x: x[0]))
                    h.append((r["tail_lp"], pos, qs.audit_vector(r)))
            r.pop("_lp")
            got[key] = r
        else:
            got[key] = qs.cmp_record(ref[key], row, VOCAB, tok, audit_in.get(key))
        if a.lane == "bf16":
            bgot[key] = qs.bf16_record(brow(tag[0], pos), row, VOCAB, tok)

    errp = (REF if is_ref else OUT) / f"{name}.{a.lane}.err.txt"
    reader = PipeReader(f"qstd_{os.getpid()}_{name}_{a.lane}", on_row)
    exe = spec.get("exe", qc.engine_exe(CFG))
    eng = qc.Engine(CFG, exe, spec.get("env", {}), spec.get("args", []), reader.path, errp, SHORT_READ)
    rec = {"arm": name, "lane": a.lane, "spec": spec, "args": eng.args, "env": eng.env_used, "payload": PAYLOAD_SHA, "scorer": SCORER,
           "cases": [], "invalid": [], "t0": time.strftime("%Y-%m-%d %H:%M:%S")}
    t0 = time.time()
    if not eng.ready:
        rec["invalid"].append("engine ended before READY")
    else:
        for i, c in enumerate(CASES):
            reqs = [c["prompt_ids"], c["prompt_ids"] + c["continuation_ids"][:qc.Q]] if a.lane == "sol" else [c["ids"]]
            done = []
            for ph, ids in enumerate(reqs):
                reader.set_tag((i, ph))
                done.append(eng.gen(ids))
                reader.drain()
            rec["cases"].append({"id": c["id"], "P": len(c["prompt_ids"] if a.lane == "sol" else c["ids"]), "done": done})
            if any(d is None or d.startswith("ERR") for d in done) or reader.error:
                rec["invalid"].append(f"request failed on {c['id']}" + (f": {reader.error}" if reader.error else ""))
                break
    eng.close()
    reader.close()
    rec["seconds"] = round(time.time() - t0)
    rec["exe_sha256"] = qc.sha_file(exe)
    rec["facts"] = qc.log_facts(errp.read_text(errors="replace"))
    check_reads(rec)
    if reader.error:
        rec["invalid"].append("pipe: " + reader.error)
    if dup:
        rec["invalid"].append(f"{len(dup)} positions arrived twice, e.g. {dup[:3]}")
    missing = EXPECTED - set(got)
    if missing and not rec["invalid"]:
        rec["invalid"].append(f"{len(missing)} of {len(EXPECTED)} scored positions missing, e.g. {sorted(missing)[:3]}")
    for ci, h in heaps.items():
        for _, pos, vec in h:
            audit[(ci, pos)] = vec
    return rec, got, bgot, audit


def bf16_arrays(bgot):
    bk = sorted(bgot)
    return {"b_case": np.array([k[0] for k in bk]), "b_pos": np.array([k[1] for k in bk]),
            **{"b_" + f: np.array([bgot[k][f] for k in bk]) for f in ("kl", "top", "ptop", "nll_b", "nll_c")}}


for name in a.arms.split(","):
    spec = dict(ARMS[name])
    is_ref = name == "R"
    if not is_ref and not spec.get("unpinned") and "--expert-cache" not in spec.get("args", []):
        # placement pinned to R's auto budget, so an arm changes only what it means to. An explicit --expert-cache N is a budget
        # in largest-blob units packed into more slots, so the pin passes R's AUTO BUDGET, and the slot count is checked after.
        rmeta = REF / f"R.{a.lane}.json"
        if not rmeta.exists():
            sys.exit("run R first: the other arms pin its placement")
        rf = json.load(open(rmeta, encoding="utf-8"))["facts"]
        if not rf.get("auto_budget"):
            sys.exit("R's log has no auto sizing line: cannot pin its placement")
        spec = dict(spec, args=list(spec.get("args", [])) + ["--expert-cache", str(rf["auto_budget"])], pinned_slots=rf["slots"])
    exe = spec.get("exe", qc.engine_exe(CFG))
    ident = qc.identity(CFG, exe, spec.get("env", {}), [x for x in spec.get("args", [])], SHORT_READ, PAYLOAD_SHA, SCORER)
    meta, refnpz = REF / f"R.{a.lane}.json", REF / f"R.{a.lane}.ref.npz"
    if is_ref and meta.exists() and refnpz.exists():
        old = json.load(open(meta, encoding="utf-8"))
        if old.get("identity") == ident and not old["invalid"]:
            print(f"R ({a.lane}): the current reference matches this setup: reused ({refnpz})", flush=True)
            continue
        print(f"R ({a.lane}): the stored reference is for another setup, case set or scorer: replacing it", flush=True)
    ref, audit_in = None, {}
    if not is_ref:
        if not refnpz.exists():
            sys.exit("no reference: run R on this lane first")
        rj = json.load(open(meta, encoding="utf-8"))
        if rj["invalid"] or rj.get("payload") != PAYLOAD_SHA or rj.get("scorer") != SCORER:
            sys.exit("the stored R is invalid, for another case set, or from another scorer version: run R again")
        ref, audit_in = qs.load_ref(refnpz)
    rec, got, bgot, audit = run_arm(name, spec, is_ref, ref, audit_in)
    rec["identity"] = ident
    if spec.get("pinned_slots") and rec["facts"]["slots"] != spec["pinned_slots"]:
        rec["invalid"].append(f"placement not reproduced: {rec['facts']['slots']} slots, R has {spec['pinned_slots']}")
    print(f"{name} ({a.lane}): {len(rec['cases'])} cases, {len(got)} positions, {rec['seconds']} s, slots {rec['facts']['slots']}, "
          f"GPU expert hits {rec['facts']['gpu_expert_hits']}, chunks {rec['facts']['chunks_used']}, windows up to {rec['facts']['windows_up_to']}"
          + (f"  INVALID: {rec['invalid']}" if rec["invalid"] else ""), flush=True)
    target = REF if is_ref else OUT
    if is_ref and not rec["invalid"]:
        qs.save_ref(refnpz, got, audit)
        if a.lane == "bf16":
            np.savez_compressed(REF / "R.bf16.npz", **bf16_arrays(bgot))
            kb = np.array([v["kl"] for v in bgot.values()])
            print(f"   R vs BF16: mean KL {kb.mean():.5f}, KL99 {np.quantile(kb, .99):.4f}, top-1 {np.mean([v['top'] for v in bgot.values()]) * 100:.2f}%", flush=True)
        print(f"   reference: {len(got)} positions, {len(audit)} audit rows, {os.path.getsize(refnpz) / 1e6:.0f} MB", flush=True)
    json.dump(rec, open(target / f"{name}.{a.lane}.json", "w", encoding="utf-8"), indent=1)
    if is_ref or rec["invalid"]:
        continue
    keys = sorted(got)
    arrays = {"case": np.array([k[0] for k in keys]), "pos": np.array([k[1] for k in keys]),
              **{f: np.array([got[k][f] for k in keys]) for f in ("kl", "top", "ptop", "gap", "nll_r", "nll_c", "inv", "ident", "tv", "kl_audit")}}
    if bgot:
        arrays.update(bf16_arrays(bgot))
    np.savez_compressed(OUT / f"{name}.{a.lane}.npz", **arrays)
    kl, ka = arrays["kl"], arrays["kl_audit"]
    am = ~np.isnan(ka)
    print(f"   vs R: mean KL {kl.mean():.6f}, KL99 {np.quantile(kl, .99):.6f}, max {kl.max():.4f}, top-1 {arrays['top'].mean() * 100:.3f}%, "
          f"sampler TV mean {arrays['tv'].mean():.5f}, byte-identical {int(arrays['ident'].sum())}/{len(kl)}; audit rows {int(am.sum())}: "
          f"exact {ka[am].sum():.5f} vs compact {kl[am].sum():.5f}" + (f"; vs BF16: mean KL {arrays['b_kl'].mean():.5f}" if bgot else ""), flush=True)
