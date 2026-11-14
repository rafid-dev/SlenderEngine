#!/usr/bin/env python3
"""Tunes an installed Qwen3.6-35B-A3B (setup's --family qwen36) with the switches this fork measured, for this PC's card,
RAM and CPU. FAST-SETUP-35B.bat runs it after setup's install.

It reads the config setup wrote (strata-qwen36-ud-q4_k_m.json by default) and writes its own config and launcher beside it
(strata-qwen36-fast.json, run-qwen36-fast.bat). Setup's files are not changed. Nothing is written with --dry-run.

What it sets (docs/fork/QWEN36-12GB.md has each switch and what was measured; measured on an RTX 4070 Ti 12 GB with 32 GB
of DDR5, other PCs are estimates):
  - every STRATA_Q36_* switch of the reference setup (arm T); on a card before Ampere (compute capability under 8.0: GTX 16,
    RTX 20) the tensor-core prompt kernels are left off (FA2, FA2P, FA2Q, GDN_TC need sm_80)
  - --mtp 2 (drafts from the file's own MTP block), --hybrid-rows 256, the context by VRAM, --pool-workers by the CPU
  - a warning when the RAM is short for the experts' mapped file (it then pages: see Known limits in the doc)

  --dry-run          show the plan and stop
  --config PATH      setup's config to start from (default strata-qwen36-ud-q4_k_m.json in the repo folder)
  --out-dir PATH     where to write the config and launcher (default the repo folder)
  --context N        the context instead of the one chosen by VRAM
  --simulate SPEC    plan for another PC, e.g. "gpu=GeForce GTX 1650,cc=7.5,vram=4,ram=16,p=6,e=4,ht=1" (implies --dry-run)
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent
GIB = 1024 ** 3
EXPERTS_GIB = 18.7       # UD-Q4_K_M with its MTP block: the experts' mapped blob file (measured 18.68 GiB)
ENGINE_OWN_GIB = 4.3     # the engine's own host memory at 262K, 50K into a conversation (measured, RTX 4070 Ti)
OS_APPS_GIB = 5.0        # Windows and a few desktop apps (estimate)

# arm T (state/STATE-q36.md, docs/fork/QWEN36-12GB.md): name -> value, grouped as in run-q36.bat
REFERENCE = {
    # cache and context
    "STRATA_Q36_KV": "q8_0", "STRATA_Q36_KV_RAM": "1", "STRATA_Q36_KV_WINDOW": "16384", "STRATA_Q36_KV_GROW": "1",
    # decode
    "STRATA_Q36_HANDOFF": "1", "STRATA_Q36_HO_THREAD": "1", "STRATA_Q36_HO_GRAPH": "1", "STRATA_Q36_HO_DMA": "1",
    "STRATA_Q36_REBAL_ASYNC": "1", "STRATA_Q36_REBAL_LAG": "8", "STRATA_Q36_PROMPT_DECAY": "1", "STRATA_Q36_ATT_Q8G": "1",
    "STRATA_Q36_ATT_TR": "1", "STRATA_Q36_Q8ST": "1", "STRATA_Q36_GEMV_ILP": "1", "STRATA_POOL_FUSED": "1",
    "STRATA_Q36_GPU_TOPK": "1",
    # the MTP step (2026-11-14: the same sampled text; the draft head only proposes)
    "STRATA_Q36_SPEC_TOPK": "1", "STRATA_Q36_COMMIT_GRAPH": "1", "STRATA_Q36_MTP_GRAPH": "1",
    "STRATA_Q36_DRAFT_VOCAB": "131072",
    # prompts
    "STRATA_Q36_LM": "1", "STRATA_Q36_LM_TAIL": "1", "STRATA_Q36_FA2": "1", "STRATA_Q36_FA2P": "1", "STRATA_Q36_FA2Q": "1",
    "STRATA_Q36_FA2_SEG": "32768", "STRATA_Q36_ATT_MFLOATS": "13", "STRATA_Q36_GDN_TC": "1", "STRATA_Q36_GDN_COLS": "16",
    "STRATA_Q36_MOE_OUT2": "1", "STRATA_Q36_SGEMM8": "1", "STRATA_Q36_DENSE_MMQ": "1", "STRATA_Q36_LAZY_F16": "1",
    "STRATA_Q36_SWQ": "1", "STRATA_Q36_QSCATTER": "1",
    # follow-ups
    "STRATA_Q36_CKPT": "1", "STRATA_Q36_CK_EXACT": "1",
}
SM80_ONLY = ("STRATA_Q36_FA2", "STRATA_Q36_FA2P", "STRATA_Q36_FA2Q", "STRATA_Q36_FA2_SEG", "STRATA_Q36_GDN_TC")


def say(text=""):
    print(text, flush=True)


def total_ram_gib():
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                    ("page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong), ("virt", ctypes.c_ulonglong),
                    ("avail_virt", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
    s = Status()
    s.length = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.total / GIB


def cpu_layout():
    """(P-cores, E-cores, logical processors of the P-cores) from Windows' processor information."""
    k = ctypes.windll.kernel32
    size = ctypes.c_ulong(0)
    k.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if not k.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
        n = os.cpu_count() or 4
        return n, 0, n
    cores, p = [], 0
    while p < size.value:
        rel, sz = ctypes.c_uint.from_buffer(buf, p).value, ctypes.c_uint.from_buffer(buf, p + 4).value
        if rel == 0:   # RelationProcessorCore: flags, efficiency class, group mask
            eff = ctypes.c_ubyte.from_buffer(buf, p + 9).value
            mask = ctypes.c_ulonglong.from_buffer(buf, p + 8 + 24).value
            cores.append((eff, bin(mask).count("1")))
        p += sz
    if not cores:
        n = os.cpu_count() or 4
        return n, 0, n
    top = max(e for e, _ in cores)
    pc = [c for c in cores if c[0] == top]
    ec = [c for c in cores if c[0] != top]
    if not ec:   # not a hybrid CPU: every core counts as a P-core
        return len(cores), 0, sum(n for _, n in cores)
    return len(pc), len(ec), sum(n for _, n in pc)


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True).stdout
        name, mib, cc = [x.strip() for x in out.splitlines()[0].rsplit(",", 2)]
        return name, int(mib) / 1024, float(cc)
    except (OSError, ValueError, IndexError):
        return None, 0.0, 0.0


def has_igpu():
    """True when Windows lists a display adapter besides the NVIDIA one (an iGPU can drive the monitor)."""
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_VideoController).Name"],
                             capture_output=True, text=True, timeout=20).stdout
        return any(n.strip() and "nvidia" not in n.lower() and "virtual" not in n.lower() and "basic" not in n.lower()
                   for n in out.splitlines())
    except (OSError, subprocess.SubprocessError):
        return False


def plan(gpu, vram, cc, ram, p, e, p_threads, igpu, context=None):
    env = dict(REFERENCE)
    notes, warnings = [], []
    if cc < 8.0:
        for k in SM80_ONLY:
            env.pop(k, None)
        notes.append(f"compute capability {cc}: the tensor-core prompt kernels (FA2, FA2P, FA2Q, GDN_TC) stay off; prompts "
                     "use the plain attention and delta-rule kernels")
    if context:
        ctx = context
    else:
        ctx = 262144 if vram >= 10 else 131072 if vram >= 6 else 65536
    notes.append(f"context {ctx // 1024}K: the 8-bit cache keeps 16K cells in VRAM and the rest in pinned RAM, growing into "
                 "expert slots as a conversation lengthens")
    # CPU expert workers: measured 19 on 8 P-cores (16 threads) + 12 E-cores (13 was setup's count); the P-cores' threads
    # plus a quarter of the E-cores, else the cores less one (estimate for other CPUs)
    workers = p_threads + e // 4 if e else max(1, p_threads - 1)
    notes.append(f"--pool-workers {workers} (CPU: {p} performance + {e} efficiency cores)")
    need = EXPERTS_GIB + ENGINE_OWN_GIB + OS_APPS_GIB
    if ram < need:
        warnings.append(f"{ram:.0f} GB of RAM: the experts' mapped file ({EXPERTS_GIB} GiB) plus the engine ({ENGINE_OWN_GIB}) "
                        f"and Windows with a few apps ({OS_APPS_GIB}) need about {need:.0f}. It will page and run far slower "
                        "(estimate: a few tokens a second). This fork has no tuned setup for it yet: a fixed RAM budget with "
                        "SSD reads (V-TIER) is the planned fix.")
    elif ram < need + 4:
        warnings.append(f"{ram:.0f} GB of RAM is just enough: close large programs (browsers with many tabs, games, editors "
                        "above ~4 GB) while it runs, or Windows pages")
    if vram < 6:
        warnings.append(f"{vram:.0f} GB of VRAM: the dense layers (~2 GiB) leave few expert slots; decode will be bound by the "
                        "CPU and RAM (estimate)")
    if vram < 8 and not igpu:
        warnings.append("no second display adapter seen: the desktop runs on this card and takes some of its VRAM. If your "
                        "CPU has integrated graphics, plug the monitor into the motherboard to give the engine the whole card.")
    args = {"--max-context": str(ctx), "--mtp": "2", "--pool-workers": str(workers), "--hybrid-rows": "256", "--prefill": "auto"}
    return env, args, notes, warnings


def set_arg(args, name, value):
    if name in args:
        args[args.index(name) + 1] = value
    else:
        args += [name, value]


def launcher(cfg_path, env, title):
    lines = ["@echo off", f"rem {title}: written by tools\\fast_setup_q36.py (FAST-SETUP-35B.bat). Run it again to rewrite.",
             f"title {title}", 'cd /d "%~dp0"']
    lines += [f'set "{k}={v}"' for k, v in env.items()]
    lines += ['set "PYTHONUTF8=1"',
              f'".venv\\Scripts\\python.exe" serve\\server.py --engine strata --config "{cfg_path}" %*']
    return "\r\n".join(lines) + "\r\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--config", default=str(HOME / "strata-qwen36-ud-q4_k_m.json"))
    ap.add_argument("--out-dir", default=str(HOME))
    ap.add_argument("--context", type=int)
    ap.add_argument("--simulate")
    a = ap.parse_args()
    if a.simulate:
        sim = dict(kv.split("=", 1) for kv in a.simulate.split(","))
        gpu, vram, cc = sim.get("gpu", "simulated"), float(sim.get("vram", 12)), float(sim.get("cc", 8.9))
        ram, p, e = float(sim.get("ram", 32)), int(sim.get("p", 8)), int(sim.get("e", 0))
        pt = p * (2 if sim.get("ht", "1") == "1" else 1)
        igpu = sim.get("igpu", "1") == "1"
        a.dry_run = True
    else:
        gpu, vram, cc = gpu_info()
        if not gpu:
            say("  No NVIDIA card found (nvidia-smi). This tuner covers NVIDIA cards only.")
            return 1
        ram = total_ram_gib()
        p, e, pt = cpu_layout()
        igpu = has_igpu()
    say(f"  This PC{' (simulated)' if a.simulate else ''}: {gpu}, {vram:.1f} GB VRAM, compute capability {cc}; "
        f"{ram:.0f} GB RAM; CPU {p} P + {e} E cores; second display adapter: {'yes' if igpu else 'no'}")
    env, args, notes, warnings = plan(gpu, vram, cc, ram, p, e, pt, igpu, a.context)
    for n in notes:
        say("  - " + n)
    for w in warnings:
        say("  ! " + w)
    say(f"  {len(env)} engine switches; engine arguments: " + " ".join(f"{k} {v}" for k, v in args.items()))
    if a.dry_run:
        say("  Dry run: nothing written.")
        return 0
    cfg_in = Path(a.config)
    if not cfg_in.is_file():
        say(f"  Setup's config is not there: {cfg_in}. Run setup first (FAST-SETUP-35B.bat does).")
        return 1
    cfg = json.loads(cfg_in.read_text(encoding="utf-8"))
    cargs = list(cfg.get("args", []))
    for k, v in args.items():
        set_arg(cargs, k, v)
    cfg["args"] = cargs
    out = Path(a.out_dir)
    cfg_path = out / "strata-qwen36-fast.json"
    cfg["log"] = str(out / "strata-qwen36-fast.log")
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    bat = out / "run-qwen36-fast.bat"
    bat.write_text(launcher(cfg_path, env, "Qwen3.6-35B-A3B (tuned)"), encoding="ascii")
    say(f"  Wrote {cfg_path.name} and {bat.name}. Start it with {bat.name}; setup's own launcher is unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
