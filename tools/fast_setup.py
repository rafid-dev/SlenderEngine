#!/usr/bin/env python3
"""Tunes an installed Qwen3.8-Flash-Next IQ3_S for a 32 GB PC on Windows and, if you want, sets up dual SSD streaming.

It reads the config setup wrote, and writes its own config and launcher beside it (strata-fast.json, run-fast.bat).
Setup's files are not changed. Nothing is written with --dry-run. FAST-SETUP.bat runs this.

  --dry-run           show what would happen and stop
  --yes               no questions: the settings, and the best second drive unless --no-mirror
  --ram-gib N         GiB of RAM for the experts, without asking (default on a 32 GB PC: 10)
  --mirror-dir PATH  the folder for the copy of experts.bin (it must be on another physical drive)
  --no-mirror         one drive only
  --verify            compare an existing copy with the original by hash, too
  --config PATH       the config to start from (default strata-iq3_s.json, else strata-iq3_xxs.json)
  --out-dir PATH      where to write the config and launcher (default this folder)
"""
import argparse
import ctypes
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import threading
from pathlib import Path

HOME = Path(__file__).resolve().parent.parent
GIB = 1024 ** 3
MIRROR_SPARE = 5 * GIB
PATCHED_MARK = b"STRATA_COLD_KEPT_READERS"   # only this fork's engine holds this name
SETUP_ARGS = "--family qwen --model IQ3_S --build --context 262144 --kv int8 --vision gpu --no-start"


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


def cpu_cores():
    """Per physical core: (efficiency class, its logical processors), in Windows' order."""
    k = ctypes.windll.kernel32
    size = ctypes.c_ulong(0)
    k.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))
    buf = ctypes.create_string_buffer(size.value)
    if not k.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
        return []
    cores, p = [], 0
    while p < size.value:
        _rel, step = struct.unpack_from("<II", buf.raw, p)
        eff = buf.raw[p + 9]
        mask = struct.unpack_from("<Q", buf.raw, p + 32)[0]
        cores.append((eff, [b for b in range(64) if mask >> b & 1]))
        p += step
    return cores


def cpu_plan(cores):
    """The layout measured on an i7-14700KF on any hybrid CPU: workers on the P-cores' first threads, the host on the first core,
    the readers on E-cores. E-core workers made every expert slower there, and readers on P-core hyperthreads twice as slow.
    A CPU with one kind of core keeps the engine's own placement."""
    top = max((c[0] for c in cores), default=0)
    p_cores = [c[1] for c in cores if c[0] == top]
    e_threads = [t for c in cores if c[0] != top for t in c[1]]
    if not e_threads or not p_cores:
        return {"hybrid": False, "workers": [], "readers": [], "p": len(p_cores), "e": 0}
    workers = [c[0] for c in p_cores if 0 not in c]
    readers = e_threads[6:10] if len(e_threads) >= 10 else e_threads[-4:] if len(e_threads) >= 4 else []
    return {"hybrid": True, "workers": workers, "readers": readers, "p": len(p_cores), "e": len(e_threads)}


def gpu_info():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        name, mib = out.splitlines()[0].rsplit(",", 1)
        return name.strip(), int(mib) / 1024
    except (OSError, ValueError, IndexError):
        return None, 0.0


def volumes():
    """The fixed drives with a letter: free space, physical disk, and what kind of disk it is."""
    script = r"""
$out = foreach ($v in Get-Volume | Where-Object { $_.DriveLetter -and $_.DriveType -eq 'Fixed' }) {
  try {
    $d = (Get-Partition -DriveLetter $v.DriveLetter -ErrorAction Stop | Get-Disk -ErrorAction Stop)
    $p = Get-PhysicalDisk -DeviceNumber $d.Number -ErrorAction Stop
    [pscustomobject]@{ letter = [string]$v.DriveLetter; free = [int64]$v.SizeRemaining; fs = $v.FileSystem; disk = [int]$d.Number; media = [string]$p.MediaType; bus = [string]$p.BusType; name = [string]$p.FriendlyName }
  } catch {}
}
@($out) | ConvertTo-Json -Compress
"""
    r = subprocess.run(["powershell", "-NoProfile", "-Command", script], capture_output=True, text=True)
    try:
        v = json.loads(r.stdout or "[]")
    except ValueError:
        return []
    return [v] if isinstance(v, dict) else v


def mirror_candidates(vols, pack_letter, need):
    """Other physical SSDs with room. A second partition of the same disk would gain nothing, a hard disk or a USB drive would slow reads down."""
    here = next((v for v in vols if v["letter"].upper() == pack_letter.upper()), None)
    if here is None:
        return []
    return [v for v in vols if v["disk"] != here["disk"] and v["media"] != "HDD" and v["bus"] not in ("USB", "SD", "MMC", "File Backed Virtual")
            and v["free"] >= need]


def sha256_of(path, out):
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        while True:
            b = f.read(16 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    out[str(path)] = h.hexdigest()


def same_file(a, b):
    say("  Comparing the two copies by hash (a few minutes for a large file) ...")
    got = {}
    ts = [threading.Thread(target=sha256_of, args=(p, got)) for p in (a, b)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return got.get(str(a)) is not None and got.get(str(a)) == got.get(str(b))


def copy_experts(src, dst):
    """Unbuffered, so the copy does not push everything else out of RAM."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["robocopy", str(src.parent), str(dst.parent), src.name, "/J", "/R:1", "/W:1", "/NJH", "/NDL", "/NC"])
    return r.returncode < 8 and dst.exists() and dst.stat().st_size == src.stat().st_size   # robocopy: below 8 is success


def choose_mirror(a, src, vols):
    """The path to use, or None for one drive. Copies the file if it is not there yet."""
    size = src.stat().st_size
    pack_letter = src.drive[:1]
    if a.no_mirror:
        say("Dual SSD: off (--no-mirror).")
        return None
    if a.mirror_dir:
        dst = Path(a.mirror_dir) / src.name
        here = next((v for v in vols if v["letter"].upper() == pack_letter.upper()), None)
        there = next((v for v in vols if v["letter"].upper() == dst.drive[:1].upper()), None)
        if here is None or there is None or here["disk"] == there["disk"]:
            sys.exit(f"{a.mirror_dir} is not on another physical drive than {src}. It would gain nothing, so nothing was copied.")
    else:
        found = mirror_candidates(vols, pack_letter, size + MIRROR_SPARE)
        if not found:
            say(f"Dual SSD: no other SSD has {(size + MIRROR_SPARE) / GIB:.0f} GB free, so this stays on one drive. The other settings still apply.")
            return None
        say(f"Dual SSD: a copy of experts.bin ({size / GIB:.1f} GB) on a second SSD lets each read come from both drives.")
        say("  Measured on two 990 PRO: +12% text, +39% prompt reading. Reading does not wear an SSD; the copy writes the file once.")
        for i, v in enumerate(found, 1):
            say(f"  {i}. {v['letter']}: {v['name']}, {v['free'] / GIB:.0f} GB free")
        if a.yes:
            pick = max(found, key=lambda v: (v["bus"] == "NVMe", v["free"]))
        else:
            ans = input(f"Which drive? 1-{len(found)}, or Enter for one drive only: ").strip()
            if not ans.isdigit() or not 1 <= int(ans) <= len(found):
                say("Dual SSD: off.")
                return None
            pick = found[int(ans) - 1]
        dst = Path(f"{pick['letter']}:\\StrataMirror") / src.name
    if dst.exists() and dst.stat().st_size == size:
        say(f"Dual SSD: {dst} is already there with the right size.")
        if a.verify and not a.dry_run and not same_file(src, dst):
            sys.exit(f"{dst} differs from {src}. Delete it and run this again.")
        return dst
    if a.dry_run:
        say(f"Dual SSD: would copy {src} to {dst} and compare the hashes.")
        return dst
    say(f"Dual SSD: copying to {dst} ...")
    if not copy_experts(src, dst):
        sys.exit(f"The copy to {dst} failed. Check the drive, delete what is there and run this again.")
    if not same_file(src, dst):
        dst.unlink(missing_ok=True)
        sys.exit("The copy differs from the original, so it was deleted. Check the drive before trying again.")
    say("  The copy matches the original.")
    return dst


def choose_ram(a, ram):
    """GiB of RAM for the experts: --ram-gib, else the person's answer, else the default. Above the safe line it warns, because Windows then pages."""
    measured = 28 <= ram <= 36
    default = 10 if measured else max(4, min(48, round(ram) - 22))
    top = max(1, round(ram) - 8)
    safe = max(default, round(ram) - 18)
    if a.ram_gib is not None:
        pick = a.ram_gib
    elif a.yes:
        pick = default
    else:
        say("How much RAM should hold the model's experts? More is faster, but leaves less for other apps.")
        say("  Measured on 32 GB: 10 GiB gave 48 tok/s with at least 3.3 GiB free; 14 GiB gave 52 tok/s with 0.7 GiB free and more writes to the page file.")
        say("  The engine uses about 8 GiB more on top of this. Other amounts have not been measured on their own.")
        while True:
            ans = input(f"GiB of RAM, 1-{top}, or Enter for {default}: ").strip()
            if not ans:
                pick = default
                break
            try:
                pick = float(ans)
            except ValueError:
                continue
            if 1 <= pick <= top:
                break
    if not 1 <= pick <= top:
        sys.exit(f"--ram-gib must be between 1 and {top} on this PC.")
    if not measured:
        say(f"This fork is tuned for 32 GB. {pick:g} GiB of experts on a {ram:.0f} GB PC has not been measured.")
    if pick > safe:
        say(f"{pick:g} GiB leaves little free RAM: close other large programs, or Windows will page heavily, which writes to your SSD.")
    return pick


def set_arg(args, name, value):
    if name in args:
        args[args.index(name) + 1] = value
    else:
        args += [name, value]


def launcher(cfg_path, env):
    py = HOME / ".venv" / "Scripts" / "python.exe"
    lines = ["@echo off", "rem Written by FAST-SETUP.bat. Chat page and API: http://127.0.0.1:8080", "title Strata (32 GB tuned)"]
    lines += [f'set "{k}={v}"' for k, v in env.items()]
    lines += ['set "PYTHONUTF8=1"', f'cd /d "{HOME}"', f'"{py}" "{HOME / "serve" / "server.py"}" --engine strata --config "{cfg_path}" --port 8080 --open %*', "if errorlevel 1 pause", ""]
    return "\r\n".join(lines)


def main():
    if os.name != "nt":
        sys.exit("This setup is for Windows. On Linux, use ./setup.sh and the settings in FORK.md.")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--ram-gib", type=float)
    ap.add_argument("--mirror-dir")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--out-dir", default=str(HOME))
    a = ap.parse_args()

    source = Path(a.config) if a.config else next((p for p in (HOME / "strata-iq3_s.json", HOME / "strata-iq3_xxs.json") if p.exists()), None)
    if source is None or not source.exists():
        sys.exit(f"No model is installed yet. Run FAST-SETUP.bat, or: START-HERE.bat {SETUP_ARGS}")
    cfg = json.loads(source.read_text(encoding="utf-8"))
    args = cfg.get("args") or []
    if "--pack" not in args:
        sys.exit(f"{source} has no --pack: this setup is for the IQ packs of Qwen3.8-Flash-Next.")

    exe = Path(cfg.get("exe") or "")
    if not exe.exists():
        sys.exit(f"The engine {exe} is missing. Run: START-HERE.bat {SETUP_ARGS}")
    if PATCHED_MARK not in exe.read_bytes():
        sys.exit(f"{exe} is a stock engine without this fork's changes. Build this one with: START-HERE.bat {SETUP_ARGS}")

    src = Path(args[args.index("--pack") + 1]) / "experts.bin"
    if not src.exists():
        sys.exit(f"{src} is missing. The dual SSD option needs a pack with experts.bin.")

    ram = total_ram_gib()
    cores = cpu_cores()
    cpu = cpu_plan(cores)
    gpu, vram = gpu_info()
    say(f"RAM {ram:.0f} GB. GPU {gpu or 'not found'}{f', {vram:.0f} GB' if gpu else ''}. CPU {cpu['p']} {'P-cores' if cpu['hybrid'] else 'cores'}{f', {cpu["e"]} E-core threads' if cpu['hybrid'] else ''}.")
    if gpu is None:
        sys.exit("No NVIDIA card found. These settings were only measured on NVIDIA, with the CUDA engine.")
    budget = choose_ram(a, ram)
    if vram < 11:
        say(f"Measured on 12 GB of VRAM. With {vram:.0f} GB, fewer experts fit on the card, so expect less speed.")
    if not cpu["hybrid"]:
        say("This CPU has one kind of core, so the engine keeps its own placement. Only a hybrid Intel CPU was measured.")
    if "RTX 40" not in gpu and "RTX 4" not in gpu:
        say("The RTX 40 kernel table will not apply to this card: other cards may gain less.")

    vols = volumes()
    mirror = choose_mirror(a, src, vols)

    for k, v in (("--spec", "4"), ("--spec-min-p", "0.5"), ("--resident-budget-gib", f"{budget:g}"), ("--adapt-every", "4"), ("--adapt-swaps", "24"),
                 ("--adapt-async", "1"), ("--kv-resident", "32768"), ("--conversation-cache-mib", "0")):
        set_arg(args, k, v)
    env = {"STRATA_UNBUFFERED_LOAD": "1", "STRATA_RESIDENT_HEADROOM_GIB": "1.5", "STRATA_RAM_ADAPT": "1", "STRATA_COLD_OVERLAP": "1", "STRATA_COLD_KEPT_READERS": "1",
           "STRATA_PARTIAL_PIN": "1", "STRATA_RAM_ADAPT_PAR": "1", "STRATA_RAM_ADAPT_BG": "1",
           # Each SSD read is handed over the moment it lands, not in batches of 16: about +9% after a 50K prompt on the PC this was tuned on
           # (with K/V lending on), no gain on short prompts, about +6% on an RTX 4060 8 GB laptop. It changes only when a read is handed over, not the arithmetic.
           "STRATA_DIRECT_BATCH": "1"}
    # Read-through: +5-13% at 10 GiB on the PC this was tuned on. At 14 GiB there it lost 19%, so it stays off above 10 until that's explained.
    if budget <= 10:
        env["STRATA_RAM_THROUGH"] = "1"
    if cpu["hybrid"]:
        set_arg(args, "--pool-workers", str(len(cpu["workers"])))
        env["STRATA_POOL_CORES"] = ",".join(map(str, cpu["workers"]))
        if cpu["readers"]:
            env["STRATA_COLD_READER_CPUS"] = ",".join(map(str, cpu["readers"]))
    if mirror is not None:
        env["STRATA_EXPERTS_MIRROR"] = str(mirror)
    # DFlash2 drafts instead of the MTP layer when its drafter is on disk (Strata-data\dflash2\*.pack): on the PC this was tuned on, follow-ups
    # +5% and 50K prompt reading +11%, thinking-on short answers level (FORK.md). The server then sets --draft dflash2 --spec 6 --draft-conf 0.8.
    data_dir = Path(args[args.index("--pack") + 1]).parent.parent
    df2_packs = sorted((data_dir / "dflash2").glob("*.pack"))
    if df2_packs:
        cfg["drafting"] = "dflash2"
    out = Path(a.out_dir)
    cfg["args"] = args
    cfg["log"] = str(out / "strata-fast.log")
    cfg["vram_elastic"] = True
    if isinstance(cfg.get("vision"), dict) and cfg["vision"].get("gpu"):
        cfg["vision"].update({"elastic": True, "elastic_mib": 1536, "elastic_idle_s": 45})

    say()
    say("Settings:")
    for k, v in env.items():
        say(f"  {k}={v}")
    say(("  DFlash2 drafter" if df2_packs else "  MTP drafter, " + " ".join(args[args.index("--spec"):args.index("--spec") + 2])) +
        f", {budget:g} GiB of experts in RAM, conversations not saved in RAM")
    if a.dry_run:
        say("\nDry run: nothing was written.")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "strata-fast.json"
    cfg_path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "run-fast.bat").write_text(launcher(cfg_path, env), encoding="utf-8", newline="")
    say(f"\nDone. Start it with {out / 'run-fast.bat'}")
    say("Close your browser and other large programs first: this uses most of a 32 GB PC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
