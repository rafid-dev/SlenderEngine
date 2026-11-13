"""Q-STD analyzer (docs/fork/QUALITY.md): one verdict per candidate, with its reasons, from the runner's compact results.

usage:
  python tools/qstd/analyze.py CAND --cal CALRUN --run RUN --kind exact|prompt|decode|placement|weights [--config cfg.json]
  python tools/qstd/analyze.py --cal CALRUN --calcheck
CALRUN: the calibration run (R2, P1-P3, S1, D1-D3, X1-X2 results); RUN: the candidate's run. Both are run names (WORK/runs/<name>)
or folders. Writes RUN/REPORT-<cand>.md (or CALRUN/CALCHECK.md) and prints it.

kinds:
  exact      promised byte-identical: identical on every scored position of both lanes -> PASS-EXACT, else FAIL
  prompt     a prompt-path reordering: judged against the prompt-chunk controls P1-P3 (the rounding envelope)
  decode     a decode-path change that is not byte-identical: no calibrated decode control exists -> HOLD (uncalibrated)
  placement  moves experts between CPU and GPU (a small precision change): judged on the BF16 lane with zero margin;
             never auto-passed before the correctness set and the deployment panel exist (Phase 2)
  weights    changes weights or precision on purpose: always the user's decision (HOLD), numbers reported
"""
import argparse
import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qcommon as qc  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("cand", nargs="?")
ap.add_argument("--cal", required=True)
ap.add_argument("--run")
ap.add_argument("--kind", default="prompt", choices=["exact", "prompt", "decode", "placement", "weights"])
ap.add_argument("--config", default=None)
ap.add_argument("--calcheck", action="store_true")
ap.add_argument("--ref", default=None, help="reference folder (default WORK/ref)")
a = ap.parse_args()
CFG = qc.load_config(a.config)
WORK = pathlib.Path(CFG["work_dir"])
resolve = lambda r: (pathlib.Path(r) if any(c in r for c in "/\\:") else WORK / "runs" / r).resolve()
CAL, REF = resolve(a.cal), pathlib.Path(a.ref or WORK / "ref").resolve()
PROMPT_CTRL = ["P1", "P2", "P3"]
STRESS = ["D1", "D2", "D3", "X1", "X2"]
ALLOW = 1.25                                   # provisional: within 1.25x the controls (upper 95% bound)
NLL_ALL, NLL_DOM = math.log(1.001), math.log(1.002)
AUDIT_RATIO = 1.10                             # the compact KL may miss at most 10% of the exact KL on the audit rows
NB, SEED = 4000, 1
SOL = json.load(open(CFG["sol_cases"], encoding="utf-8"))
STRATUM = np.array([c["stratum"] for c in SOL])
BF16_DOMAIN = {"en_prose_129": "prose", "ja_prose_129": "prose", "zh_tech_257": "prose", "natural_4097": "prose", "wiki2_r0": "prose",
               "wiki2_r1": "prose", "rust_code_257": "code", "python_code_513": "code", "sql_code_1025": "code", "code_4097": "code",
               "agent_code_513": "code", "structured_257": "structured", "arithmetic_65": "arithmetic"}


def load(d, arm, lane):
    f, j = d / f"{arm}.{lane}.npz", d / f"{arm}.{lane}.json"
    if not f.exists() or not j.exists():
        return None
    z = dict(np.load(f))
    z["_rec"] = json.load(open(j, encoding="utf-8"))
    return z


def per_case(z, key, n, fn, casekey="case"):
    return np.array([fn(z[key][z[casekey] == i]) if np.any(z[casekey] == i) else np.nan for i in range(n)])


tail = lambda v: np.sort(v)[-max(1, math.ceil(0.01 * len(v))):].mean()


def boot_upper(fn, sel, n_boot=NB):
    idx = np.flatnonzero(sel)
    rng = np.random.default_rng(SEED)
    vals = [fn(idx[rng.integers(0, len(idx), len(idx))]) for _ in range(n_boot)]
    return float(np.percentile(vals, 95)), float(np.percentile(vals, 5)), float(fn(idx))


def size_gates(cand, ctrls, sel):
    """Mean KL, KL99, per-case worst-1% mean and sampler-TV99 of cand vs 1.25x the controls' median, on the cases in sel."""
    n = len(SOL)
    out = {}
    for key, stat, nm in (("kl", np.mean, "mean KL"), ("kl", tail, "worst-1% mean")):
        c = per_case(cand, key, n, stat)
        f = np.median(np.stack([per_case(x, key, n, stat) for x in ctrls]), axis=0)
        z = c - ALLOW * f
        up, _, _ = boot_upper(lambda ix: z[ix].mean(), sel)
        out[nm] = (up, float(c[sel].mean()), float(f[sel].mean()))
    for key, nm in (("kl", "KL99"), ("tv", "sampler TV99")):
        bc = [cand[key][cand["case"] == i] for i in range(n)]
        bf = [[x[key][x["case"] == i] for i in range(n)] for x in ctrls]
        q = lambda cs, ix: np.quantile(np.concatenate([cs[i] for i in ix]), 0.99)
        full = np.flatnonzero(sel)
        up, _, _ = boot_upper(lambda ix: q(bc, ix) - ALLOW * np.median([q(f, ix) for f in bf]), sel, 1000)
        out[nm] = (up, float(q(bc, full)), float(np.median([q(f, full) for f in bf])))
    m = np.isin(cand["case"], np.flatnonzero(sel))
    out["max"] = (float(cand["kl"][m].max()), max(float(x["kl"][np.isin(x["case"], np.flatnonzero(sel))].max()) for x in ctrls))
    out["flips"] = int(((~cand["top"]) & (cand["ptop"] >= 0.90) & m).sum())
    out["top1"] = float(cand["top"][m].mean())
    return out


def audit_check(z):
    am = ~np.isnan(z["kl_audit"])
    if not am.any():
        return None
    ex, cp = z["kl_audit"][am].sum(), z["kl"][am].sum()
    return {"rows": int(am.sum()), "exact": float(ex), "compact": float(cp), "ratio": float(ex / cp) if cp > 0 else (1.0 if ex == 0 else float("inf"))}


def nll_gates(cand, ctrls):
    """Independent text (the BF16-lane texts): candidate-minus-R next-token NLL, overall and per domain, vs log(1.001)/log(1.002)
    plus the controls' own upper bound (controls that leave this lane unchanged contribute 0)."""
    if cand is None:
        return []
    man = [c["id"] for c in json.load(open(pathlib.Path(CFG["bf16_dir"]) / "reference" / "manifest.json"))["cases"]]
    ids = [i for i in man if (pathlib.Path(CFG["bf16_dir"]) / "reference" / f"{i}.f32").exists()]
    n = int(cand["case"].max()) + 1
    ex = lambda z: per_case(z, "nll_c", n, np.nanmean) - per_case(z, "nll_r", n, np.nanmean)
    res, ec = [], ex(cand)
    groups = [("overall", np.ones(n, bool), NLL_ALL)] + [(d, np.array([BF16_DOMAIN.get(ids[i], "other") == d for i in range(n)]), NLL_DOM)
                                                          for d in sorted(set(BF16_DOMAIN.values()))]
    for label, sel, margin in groups:
        if sel.sum() == 0:
            continue
        up, _, pt = boot_upper(lambda ix: ec[ix].mean(), sel)
        cu = [boot_upper(lambda ix, e=ex(c): e[ix].mean(), sel)[0] for c in ctrls if c is not None]
        allow = max([0.0] + cu) + margin
        res.append((label, pt, up, allow, up <= allow))
    return res


def bf16_d(cand):
    """d = KL(B||C) - KL(B||R) per position over the COMPLETE expected key set; mean, 95% bounds by case bootstrap."""
    rbf = REF / "R.bf16.npz"
    if cand is None or not rbf.exists():
        return None
    rb = dict(np.load(rbf))
    kr = {(int(c), int(p)): v for c, p, v in zip(rb["b_case"], rb["b_pos"], rb["b_kl"])}
    kc = {(int(c), int(p)): v for c, p, v in zip(cand["b_case"], cand["b_pos"], cand["b_kl"])}
    if set(kr) != set(kc):
        return {"error": f"BF16 lane key sets differ ({len(kr)} vs {len(kc)} positions)"}
    keys = sorted(kr)
    d = np.array([kc[k] - kr[k] for k in keys])
    dc = np.array([k[0] for k in keys])
    n = int(dc.max()) + 1
    dcase = np.array([d[dc == i].mean() for i in range(n)])
    up, lo, pt = boot_upper(lambda ix: dcase[ix].mean(), np.ones(n, bool))
    rcase = np.array([np.mean([kr[k] for k in keys if k[0] == i]) for i in range(n)])
    return {"d": pt, "up": up, "lo": lo, "R_kl": float(rcase.mean()), "R_top1": float(rb["b_top"].mean())}


def fmt(x, d=6):
    return "-" if x is None else f"{x:.{d}f}"


def verdict(cs, cb, kind, rep):
    hold = []
    for z, lane in ((cs, "sol"), (cb, "bf16")):
        if z is None:
            return "HOLD", [f"no {lane}-lane result: both lanes are required"]
        if z["_rec"]["invalid"]:
            return "FAIL", [f"invalid {lane} run: {z['_rec']['invalid']}"]
    r2 = load(CAL, "R2", "sol")
    if r2 is None or not bool(r2["ident"].all()):
        return "FAIL", ["the reference repeat R2 is missing or not byte-identical to R: nothing is reproducible"]
    ident = bool(cs["ident"].all()) and bool(cb["ident"].all())
    rep.append(f"\nByte-identical positions: sol {int(cs['ident'].sum())}/{len(cs['ident'])}, bf16 {int(cb['ident'].sum())}/{len(cb['ident'])}")
    if kind == "exact":
        return ("PASS-EXACT", ["byte-identical on every scored position of both lanes"]) if ident else ("FAIL", ["promised exact, but the logits differ"])
    if ident:
        return "PASS-EXACT", ["byte-identical on every scored position of both lanes"]
    ctrls = [load(CAL, c, "sol") for c in PROMPT_CTRL]
    sel_aff = np.isin(STRATUM, ["medium", "long"]) if kind == "prompt" else np.ones(len(SOL), bool)
    ac = audit_check(cs)
    if ac:
        rep.append(f"\n**Audit rows** ({ac['rows']}): exact KL {fmt(ac['exact'])} vs compact {fmt(ac['compact'])} (exact/compact {ac['ratio']:.3f}; "
                   f"allowed {AUDIT_RATIO:.2f}). Verdict basis: compact + audit, not a full-vocabulary proof.")
        if ac["ratio"] > AUDIT_RATIO:
            hold.append(f"the compact KL misses {100 * (ac['ratio'] - 1):.0f}% of the exact KL on the audit rows: compact estimate unreliable for this change")
    if kind in ("prompt", "decode", "weights") and all(c is not None and not c["_rec"]["invalid"] for c in ctrls):
        for label, sel in (("all 48", np.ones(len(SOL), bool)), ("affected", sel_aff)):
            g = size_gates(cs, ctrls, sel)
            rep.append(f"\n**Production replay, {label} cases** (candidate vs 1.25 x the prompt controls' median)\n")
            rep.append("| Gate | Candidate | Controls (median) | Upper 95% of excess over 1.25x | Pass |\n| --- | --- | --- | --- | --- |")
            for nm in ("mean KL", "KL99", "worst-1% mean", "sampler TV99"):
                up, cv, fv = g[nm]
                rep.append(f"| {nm} | {fmt(cv)} | {fmt(fv)} | {fmt(up)} | {'yes' if up <= 0 else 'NO'} |")
                if up > 0 and kind == "prompt":
                    hold.append(f"{nm} ({label}) beyond 1.25x the prompt controls")
            okmax = g["max"][0] <= ALLOW * g["max"][1]
            rep.append(f"| max KL | {fmt(g['max'][0], 4)} | envelope {fmt(g['max'][1], 4)} | - | {'yes' if okmax else 'NO'} |")
            rep.append(f"| confident flips (R top >= 0.90) | {g['flips']} | - | - | {'yes' if g['flips'] == 0 else 'NO'} |")
            rep.append(f"| top-1 agreement | {g['top1'] * 100:.3f}% | | | |")
            if kind == "prompt" and not okmax:
                hold.append(f"max KL ({label}) past the controls' envelope")
            if kind == "prompt" and g["flips"]:
                hold.append(f"{g['flips']} confident flip(s) ({label})")
    elif kind == "prompt":
        hold.append("uncalibrated: a prompt control (P1-P3) is missing or invalid")
    for label, pt, up, allow, ok in nll_gates(cb, []):
        rep.append(f"- independent text, {label}: excess NLL {fmt(pt)} nats (upper 95% {fmt(up)}, allowed {fmt(allow)}): {'pass' if ok else 'NO'}")
        if not ok:
            hold.append(f"independent-text NLL ({label}) beyond its allowance")
    dg = bf16_d(cb)
    if dg and "error" in dg:
        return "FAIL", [dg["error"]]
    if dg:
        rep.append(f"\n**BF16 yardstick**: IQ3_S today differs from BF16 by mean KL {fmt(dg['R_kl'], 4)} (top-1 {dg['R_top1'] * 100:.2f}%). "
                   f"The change: d = {fmt(dg['d'])} (95% bounds {fmt(dg['lo'])} .. {fmt(dg['up'])}); "
                   + ("not farther from BF16 (upper bound <= 0)" if dg["up"] <= 0 else "moved away from BF16 (lower bound > 0)" if dg["lo"] > 0 else "unresolved (the range crosses 0)"))
    if kind == "decode":
        hold.append("a decode-path change that is not byte-identical: no calibrated decode control exists (uncalibrated); decide with the user")
    if kind == "placement":
        if dg and dg["up"] > 0:
            hold.append("placement: not shown to stay as close to BF16 (zero margin)")
        hold.append("placement changes are not auto-passed before the correctness set and the deployment panel exist (Phase 2)")
    if kind == "weights":
        hold.append("changes weights or precision: always the user's decision")
    if hold:
        return "HOLD", hold
    return "PASS-NUMERICAL", ["within 1.25x the prompt controls on mean, KL99, worst-1% and sampler TV99; no confident flip; independent text within "
                              "its allowance; audit consistent. The task-correctness phase does not exist yet: this is a numerical pass only"]


if a.calcheck:
    lines = ["# Q-STD calibration check", "", "Size gates only (mean KL, KL99, worst-1% mean, sampler TV99): each prompt control judged against the",
             "median of the other two. Placement stress (D/X) is reported, not judged. This checks the 1.25x allowance; it does not",
             "certify the controls on the other gates.", ""]
    ok = True
    sel = np.isin(STRATUM, ["medium", "long"])
    for name in PROMPT_CTRL:
        z = load(CAL, name, "sol")
        others = [load(CAL, c, "sol") for c in PROMPT_CTRL if c != name]
        if z is None or any(o is None for o in others):
            lines.append(f"- {name}: missing")
            ok = False
            continue
        g = size_gates(z, others, sel)
        passed = all(g[k][0] <= 0 for k in ("mean KL", "KL99", "worst-1% mean", "sampler TV99"))
        ok = ok and passed
        lines.append(f"- {name}: " + ", ".join(f"{k} {fmt(g[k][1])} vs {fmt(g[k][2])}" for k in ("mean KL", "KL99", "worst-1% mean", "sampler TV99"))
                     + f", flips {g['flips']}: {'passes' if passed else 'FAILS'}")
    s1 = load(CAL, "S1", "sol")
    lines.append(f"- S1 (window-size sentinel): " + ("missing" if s1 is None else f"byte-identical {int(s1['ident'].sum())}/{len(s1['ident'])}"))
    for name in STRESS:
        z, zb = load(CAL, name, "sol"), load(CAL, name, "bf16")
        if z is not None:
            dg = bf16_d(zb) if zb is not None else None
            lines.append(f"- {name} (placement stress): sol mean KL {fmt(float(z['kl'].mean()))}, KL99 {fmt(float(np.quantile(z['kl'], .99)))}, slots "
                         f"{z['_rec']['facts']['slots']}" + (f"; BF16 d {fmt(dg['d'])} [{fmt(dg['lo'])}, {fmt(dg['up'])}]" if dg and "error" not in dg else ""))
    lines += ["", "The 1.25x allowance is consistent with the prompt reorderings." if ok else
              "A prompt control fails against the others: the allowance is not calibrated (change it or the controls before use)."]
    txt = "\n".join(lines)
    open(CAL / "CALCHECK.md", "w", encoding="utf-8").write(txt)
    print(txt)
    sys.exit(0)

if not a.cand or not a.run:
    sys.exit("give CAND and --run (or --calcheck)")
RUN = resolve(a.run)
cs, cb = load(RUN, a.cand, "sol"), load(RUN, a.cand, "bf16")
rep = [f"# Q-STD report: {a.cand}", "", f"kind {a.kind}; calibration {CAL}; reference {REF}"]
v, why = verdict(cs, cb, a.kind, rep)
rep[3:3] = ["", f"## Verdict: **{v}**", ""] + [f"- {x}" for x in why]
txt = "\n".join(rep)
open(RUN / f"REPORT-{a.cand}.md", "w", encoding="utf-8").write(txt)
print(txt)
