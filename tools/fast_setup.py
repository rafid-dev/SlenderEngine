#!/usr/bin/env python3
"""Tunes an installed Qwen3.8-Flash-Next IQ3_S for a 32 GB PC on Windows or Linux and, if you want, sets up dual SSD streaming.

It reads the config setup wrote, and writes its own config and launcher beside it (strata-fast.json, and run-fast.bat on
Windows or run-fast.sh on Linux). Setup's files are not changed. Nothing is written with --dry-run. FAST-SETUP.bat (Windows)
and FAST-SETUP.sh (Linux) run this.

  --dry-run           show what would happen and stop
  --yes               no questions: the settings, and the best second drive unless --no-mirror
  --ram-gib N         GiB of RAM for the experts, without asking (default on a 32 GB PC: 10)
  --skip N            the expert skip, without asking: 0 (the default, every expert computed) or 7 (faster, answers change)
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
PATCHED_MARK = b"STRATA_FORK_DEFAULTS"   # only this fork's engine holds this name, from the build that applies its defaults itself
SETUP_ARGS = "--family qwen --model IQ3_S --build --context 262144 --kv int8 --vision gpu --no-start"
WINDOWS = os.name == "nt"
SETUP_CMD = "START-HERE.bat" if WINDOWS else "./setup.sh"
FAST_CMD = "FAST-SETUP.bat" if WINDOWS else "./FAST-SETUP.sh"


def say(text=""):
    print(text, flush=True)


def total_ram_gib():
    if not WINDOWS:
        for line in open("/proc/meminfo", encoding="ascii"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024 / GIB
        return 0.0
    class Status(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                    ("page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong), ("virt", ctypes.c_ulonglong),
                    ("avail_virt", ctypes.c_ulonglong), ("ext", ctypes.c_ulonglong)]
    s = Status()
    s.length = ctypes.sizeof(Status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.total / GIB


def cpu_list(text):
    """A Linux CPU list ("0-15,20,22-23") as numbers."""
    out = []
    for part in text.strip().split(","):
        if "-" in part:
            a, b = part.split("-")
            out += range(int(a), int(b) + 1)
        elif part:
            out.append(int(part))
    return out


def cpu_cores_linux():
    """Per physical core: (1 for a P-core, 0 for an E-core or a CPU with one kind of core, its logical processors). Hybrid Intel CPUs
    name their two kinds in /sys/devices/cpu_core/cpus and /sys/devices/cpu_atom/cpus; the core a thread belongs to is in its topology."""
    try:
        p_set = set(cpu_list(Path("/sys/devices/cpu_core/cpus").read_text()))
        e_set = set(cpu_list(Path("/sys/devices/cpu_atom/cpus").read_text()))
    except (OSError, ValueError):
        p_set = e_set = set()
    seen, cores = set(), []
    for d in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*"), key=lambda p: int(p.name[3:])):
        n = int(d.name[3:])
        if n in seen:
            continue
        try:
            sib = cpu_list((d / "topology" / "thread_siblings_list").read_text())
        except (OSError, ValueError):
            sib = [n]
        seen.update(sib)
        cores.append((1 if (p_set and e_set and n in p_set) else 0, sib))
    return cores


def cpu_cores():
    """Per physical core: (efficiency class, its logical processors), in Windows' order."""
    if not WINDOWS:
        return cpu_cores_linux()
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


def volumes_linux():
    """The mounted filesystems, with free space, the physical disk under each (lsblk's tree: a partition, LVM or LUKS volume counts as
    its top disk) and what kind of disk it is. "letter" holds the mount point, so the rest of this file reads the same on both systems."""
    r = subprocess.run(["lsblk", "-J", "-b", "-o", "NAME,MOUNTPOINT,FSAVAIL,FSTYPE,ROTA,TRAN,MODEL,TYPE"], capture_output=True, text=True)
    try:
        tree = json.loads(r.stdout or "{}").get("blockdevices") or []
    except ValueError:
        return []
    out = []

    def walk(dev, disk):
        mnt = dev.get("mountpoint")
        if mnt and dev.get("fsavail") is not None and dev.get("fstype") not in ("swap", None):
            out.append({"letter": mnt, "free": int(dev["fsavail"]), "fs": dev.get("fstype"), "disk": disk["name"],
                        "media": "HDD" if disk.get("rota") in (True, "1", 1) else "SSD",
                        "bus": {"nvme": "NVMe", "usb": "USB", "sata": "SATA", "mmc": "MMC"}.get((disk.get("tran") or "").lower(), disk.get("tran") or ""),
                        "name": (disk.get("model") or disk["name"]).strip()})
        for c in dev.get("children") or []:
            walk(c, disk)

    for d in tree:
        if d.get("type") in ("disk",):
            walk(d, d)
    return out


def volume_of(path, vols):
    """The volume a path is on: by drive letter on Windows, by the longest mount point that holds it on Linux."""
    if WINDOWS:
        return next((v for v in vols if v["letter"].upper() == Path(path).drive[:1].upper()), None)
    p = str(Path(path).resolve())
    best = None
    for v in vols:
        m = v["letter"].rstrip("/") or "/"
        if (p == m or p.startswith(m.rstrip("/") + "/")) and (best is None or len(m) > len(best["letter"])):
            best = v
    return best


def volumes():
    """The fixed drives with a letter: free space, physical disk, and what kind of disk it is."""
    if not WINDOWS:
        return volumes_linux()
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


def mirror_candidates(vols, src, need):
    """Other physical SSDs with room. A second partition of the same disk would gain nothing, a hard disk or a USB drive would slow reads down.
    On Linux the mount must also be writable by this user (the copy goes in <mount>/StrataMirror)."""
    here = volume_of(src, vols)
    if here is None:
        return []
    return [v for v in vols if v["disk"] != here["disk"] and v["media"] != "HDD" and v["bus"] not in ("USB", "SD", "MMC", "File Backed Virtual")
            and v["free"] >= need and (WINDOWS or os.access(v["letter"], os.W_OK))]


def mirror_dir_for(v):
    return Path(f"{v['letter']}:\\StrataMirror") if WINDOWS else Path(v["letter"]) / "StrataMirror"


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


def file_extents(path):
    """How many pieces the file is stored in (fsutil on Windows, filefrag on Linux), or None if the system won't say."""
    if not WINDOWS:
        try:
            r = subprocess.run(["filefrag", str(path)], capture_output=True, text=True)
        except OSError:
            return None
        try:   # "<path>: 12 extents found"
            return int(r.stdout.rsplit(":", 1)[1].split()[0]) if r.returncode == 0 else None
        except (IndexError, ValueError):
            return None
    r = subprocess.run(["fsutil", "file", "queryextents", str(path)], capture_output=True, text=True)
    n = r.stdout.count("VCN:")
    return n if r.returncode == 0 and n > 0 else None


FRAGMENTED = 1000   # more pieces than this: some reads of the copy become many small disk requests (68,990 measured 3x slower)


def copy_experts_linux(src, dst):
    """The Linux copy, for the same reasons: the whole size reserved first (posix_fallocate), then 64 MiB at a time with O_DIRECT through
    a page-aligned buffer, so the page cache isn't filled. A filesystem that refuses O_DIRECT gets a buffered copy that drops each
    piece from the cache after writing it."""
    import mmap
    chunk, size = 64 << 20, src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.posix_fallocate(fd, 0, size)
    except OSError:
        os.close(fd)
        return False
    os.close(fd)
    n = file_extents(dst)
    if n is not None:
        say(f"  Space reserved in {n} piece{'s' if n != 1 else ''}.")
    direct = getattr(os, "O_DIRECT", 0)
    try:
        s = os.open(src, os.O_RDONLY | direct)
        d = os.open(dst, os.O_WRONLY | direct)
    except OSError:
        direct = 0
        s = os.open(src, os.O_RDONLY)
        d = os.open(dst, os.O_WRONLY)
    buf, done, ok = mmap.mmap(-1, chunk), 0, True
    try:
        while ok and done < size:
            want = min(chunk, (size - done + 4095) // 4096 * 4096) if direct else min(chunk, size - done)
            view = memoryview(buf)[:want]
            got = os.preadv(s, [view], done)
            if got <= 0:
                ok = False
                break
            n_out = (got + 4095) // 4096 * 4096 if direct else got
            ok = os.pwritev(d, [memoryview(buf)[:n_out]], done) == n_out
            if not direct:
                os.posix_fadvise(d, done, n_out, os.POSIX_FADV_DONTNEED)
                os.posix_fadvise(s, done, got, os.POSIX_FADV_DONTNEED)
            done += got
            if done % (10 << 30) < chunk:
                say(f"  {done / GIB:.0f} of {size / GIB:.0f} GB")
    finally:
        os.close(s)
        os.close(d)
        buf.close()
    if ok and size % 4096:   # the last block was written whole: cut the file back to the original's size
        os.truncate(dst, size)
    return ok and done == size and dst.stat().st_size == size


def copy_experts(src, dst):
    """Unbuffered, so the copy does not push everything else out of RAM, and with the whole size reserved first, so the
    copy lands in a few large pieces. A copy written as it goes (robocopy /J) took the free gaps of a well-used drive:
    68,990 pieces, and about 880 experts read 3x slower from it."""
    if not WINDOWS:
        return copy_experts_linux(src, dst)
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    k.CreateFileW.restype = ctypes.c_void_p
    k.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    k.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    k.WriteFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32), ctypes.c_void_p]
    k.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong, ctypes.c_void_p, ctypes.c_uint32]
    k.SetEndOfFile.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    k.VirtualAlloc.restype = ctypes.c_void_p
    k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32]
    k.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32]
    READ, WRITE, SHARE_RW, OPEN_EXISTING, CREATE_ALWAYS, NO_BUFFERING = 0x80000000, 0x40000000, 3, 3, 2, 0x20000000
    invalid, chunk, size = ctypes.c_void_p(-1).value, 64 << 20, src.stat().st_size
    dst.parent.mkdir(parents=True, exist_ok=True)
    h = k.CreateFileW(str(dst), READ | WRITE, 1, None, CREATE_ALWAYS, NO_BUFFERING, None)
    if h in (None, invalid):
        return False
    k.SetFilePointerEx(h, size, None, 0)   # reserve every cluster now, as contiguous as the free space allows
    reserved = bool(k.SetEndOfFile(h))
    k.CloseHandle(h)
    if not reserved:
        return False
    n = file_extents(dst)
    if n is not None:
        say(f"  Space reserved in {n} piece{'s' if n != 1 else ''}.")
    s = k.CreateFileW(str(src), READ, SHARE_RW, None, OPEN_EXISTING, NO_BUFFERING, None)
    d = k.CreateFileW(str(dst), WRITE, 1, None, OPEN_EXISTING, NO_BUFFERING, None)
    buf = k.VirtualAlloc(None, chunk, 0x3000, 0x04)
    ok = s not in (None, invalid) and d not in (None, invalid) and bool(buf)
    got, put, done = ctypes.c_uint32(0), ctypes.c_uint32(0), 0
    while ok and done < size:
        want = min(chunk, (size - done + 4095) // 4096 * 4096)   # unbuffered I/O moves whole sectors; the tail is cut below
        ok = bool(k.ReadFile(s, buf, want, ctypes.byref(got), None)) and got.value > 0
        if ok:
            n_out = (got.value + 4095) // 4096 * 4096
            ok = bool(k.WriteFile(d, buf, n_out, ctypes.byref(put), None)) and put.value == n_out
            done += got.value
            if done % (10 << 30) < chunk:
                say(f"  {done / GIB:.0f} of {size / GIB:.0f} GB")
    for x in (s, d):
        if x not in (None, invalid):
            k.CloseHandle(x)
    if buf:
        k.VirtualFree(buf, 0, 0x8000)
    if ok and size % 4096:   # the last sector was written whole: cut the file back to the original's size
        with open(dst, "r+b") as f:
            f.truncate(size)
    return ok and done == size and dst.stat().st_size == size


def choose_mirror(a, src, vols):
    """The path to use, or None for one drive. Copies the file if it is not there yet."""
    size = src.stat().st_size
    if a.no_mirror:
        say("Dual SSD: off (--no-mirror).")
        return None
    if a.mirror_dir:
        dst = Path(a.mirror_dir) / src.name
        here = volume_of(src, vols)
        there = volume_of(dst.parent if dst.parent.exists() else Path(a.mirror_dir).anchor or "/", vols)
        if here is None or there is None or here["disk"] == there["disk"]:
            sys.exit(f"{a.mirror_dir} is not on another physical drive than {src}. It would gain nothing, so nothing was copied.")
    else:
        found = mirror_candidates(vols, src, size + MIRROR_SPARE)
        if not found:
            say(f"Dual SSD: no other SSD has {(size + MIRROR_SPARE) / GIB:.0f} GB free, so this stays on one drive. The other settings still apply.")
            return None
        say(f"Dual SSD: a copy of experts.bin ({size / GIB:.1f} GB) on a second SSD lets each read come from both drives.")
        say("  Measured on two 990 PRO: +12% text, +39% prompt reading. Reading does not wear an SSD; the copy writes the file once.")
        for i, v in enumerate(found, 1):
            say(f"  {i}. {v['letter']}{':' if WINDOWS else ''} {v['name']}, {v['free'] / GIB:.0f} GB free")
        if a.yes:
            pick = max(found, key=lambda v: (v["bus"] == "NVMe", v["free"]))
        else:
            ans = input(f"Which drive? 1-{len(found)}, or Enter for one drive only: ").strip()
            if not ans.isdigit() or not 1 <= int(ans) <= len(found):
                say("Dual SSD: off.")
                return None
            pick = found[int(ans) - 1]
        dst = mirror_dir_for(pick) / src.name
    if dst.exists() and dst.stat().st_size == size:
        say(f"Dual SSD: {dst} is already there with the right size.")
        n = file_extents(dst)
        if n is not None and n > FRAGMENTED:
            say(f"  It is stored in {n:,} pieces, so some of its reads are slow (measured: about 880 experts 3x slower at 68,990 pieces).")
            say(f"  To fix it: delete {dst} and run {FAST_CMD} again; the new copy reserves its space first.")
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
        say(f"{pick:g} GiB leaves little free RAM: close other large programs, or the system will page heavily, which writes to your SSD.")
    return pick


def choose_skip(a):
    """The route tail skip: 0 (the fork's default, which the engine applies by itself) or what the person picks."""
    if a.skip is not None:
        pick = a.skip
    elif a.yes:
        pick = 0
    else:
        say("Expert skip: should a missed expert that every token ranks low be skipped (neither read nor computed)?")
        say("  No (recommended): every expert is computed, so the answers are the full model's.")
        say("  Yes: about 22% faster text on the PC this was tuned on (45.9 against 35.9 tok/s), but the answers change")
        say("  (KL 0.081 against no skip, the top choice the same at 90.7% of positions).")
        ans = input("Skip experts? y/N: ").strip().lower()
        pick = 7 if ans in ("y", "yes") else 0
    if not 0 <= pick <= 8:
        sys.exit("--skip must be between 0 and 8 (0 is off, 7 is upstream's default).")
    return pick


def set_arg(args, name, value):
    if name in args:
        args[args.index(name) + 1] = value
    else:
        args += [name, value]


def launcher_sh(cfg_path, env):
    py = HOME / ".venv" / "bin" / "python"
    lines = ["#!/usr/bin/env bash", "# Written by FAST-SETUP.sh. Chat page and API: http://127.0.0.1:8080"]
    lines += [f"export {k}={json.dumps(str(v))}" for k, v in env.items()]
    lines += ["export PYTHONUTF8=1", f'cd "{HOME}" || exit 1',
              f'exec "{py}" "{HOME / "serve" / "server.py"}" --engine strata --config "{cfg_path}" --port 8080 --open "$@"', ""]
    return "\n".join(lines)


def launcher(cfg_path, env):
    if not WINDOWS:
        return launcher_sh(cfg_path, env)
    py = HOME / ".venv" / "Scripts" / "python.exe"
    lines = ["@echo off", "rem Written by FAST-SETUP.bat. Chat page and API: http://127.0.0.1:8080", "title SlenderEngine (32 GB tuned)"]
    lines += [f'set "{k}={v}"' for k, v in env.items()]
    lines += ['set "PYTHONUTF8=1"', f'cd /d "{HOME}"', f'"{py}" "{HOME / "serve" / "server.py"}" --engine strata --config "{cfg_path}" --port 8080 --open %*', "if errorlevel 1 pause", ""]
    return "\r\n".join(lines)


def main():
    if not WINDOWS and not sys.platform.startswith("linux"):
        sys.exit("This setup is for Windows and Linux. On a Mac, use ./setup.sh and the settings in FORK.md.")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--ram-gib", type=float)
    ap.add_argument("--skip", type=int)
    ap.add_argument("--mirror-dir")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--out-dir", default=str(HOME))
    a = ap.parse_args()

    source = Path(a.config) if a.config else next((p for p in (HOME / "strata-iq3_s.json", HOME / "strata-iq3_xxs.json") if p.exists()), None)
    if source is None or not source.exists():
        sys.exit(f"No model is installed yet. Run {FAST_CMD}, or: {SETUP_CMD} {SETUP_ARGS}")
    cfg = json.loads(source.read_text(encoding="utf-8"))
    args = cfg.get("args") or []
    if "--pack" not in args:
        sys.exit(f"{source} has no --pack: this setup is for the IQ packs of Qwen3.8-Flash-Next.")

    exe = Path(cfg.get("exe") or "")
    if not exe.exists():
        sys.exit(f"The engine {exe} is missing. Run: {SETUP_CMD} {SETUP_ARGS}")
    if PATCHED_MARK not in exe.read_bytes():
        sys.exit(f"{exe} is a stock engine, or this fork's from before it applied its own defaults. Build this one with: {SETUP_CMD} {SETUP_ARGS}")

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

    skip = choose_skip(a)
    vols = volumes()
    mirror = choose_mirror(a, src, vols)

    for k, v in (("--spec", "4"), ("--spec-min-p", "0.5"), ("--resident-budget-gib", f"{budget:g}"), ("--adapt-every", "4"), ("--adapt-swaps", "24"),
                 ("--adapt-async", "1"), ("--kv-resident", "32768"), ("--conversation-cache-mib", "0")):
        set_arg(args, k, v)
    # The fork's lossless switches (K/V lending, the stager, read-through, one VRAM arena, 16K prompt chunks
    # with a RAM budget ...) and skip 0 are the engine's own defaults now (FORK.md, "Defaults"), so only this PC's settings go here.
    env = {"STRATA_RESIDENT_HEADROOM_GIB": "1.5"}
    if skip != 0:
        env["STRATA_ROUTE_TAIL_SKIP"] = str(skip)
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
    say(f"  Expert skip {skip}" + (" (none: every expert computed)" if skip == 0 else " (answers change)"))
    say("  Plus the fork's defaults, which the engine applies by itself (FORK.md, \"Defaults\"; STRATA_FORK_DEFAULTS=0 turns them off)")
    say(("  DFlash2 drafter" if df2_packs else "  MTP drafter, " + " ".join(args[args.index("--spec"):args.index("--spec") + 2])) +
        f", {budget:g} GiB of experts in RAM, conversations not saved in RAM")
    if a.dry_run:
        say("\nDry run: nothing was written.")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = out / "strata-fast.json"
    cfg_path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    run = out / ("run-fast.bat" if WINDOWS else "run-fast.sh")
    run.write_text(launcher(cfg_path, env), encoding="utf-8", newline="")
    if not WINDOWS:
        run.chmod(0o755)
    say(f"\nDone. Start it with {run}")
    say("Close your browser and other large programs first: this uses most of a 32 GB PC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
