# Linux support: plan

Branch `linux-support`, started 2026-11-09 from `speed-fork` (`88b24a6c`). Goal: a Linux PC with an NVIDIA card gets the same
speed changes a Windows PC gets, with the same answers.

## Progress (2026-11-09)

| Item | State |
|---|---|
| 1. The 28 GB RAM check | Done (`3c78d3e8`). Windows banner unchanged |
| 2. The defaults on Linux | Done (`3c78d3e8`). Windows output byte-identical to `speed-fork`; WSL lists the full set and answers correctly. No memlock check needed: CUDA's pinning isn't bound by `ulimit -l`, and the one `mlock` fallback already says how to raise it |
| 3. The mirror | Done (`087c6355`). WSL: logits byte-identical with and without the mirror (second copy on C:). Speed: needs native Linux |
| 4. The head reload | Done (`dad67791`). WSL: 2,115-2,809 ms with the old fallback, 1,328 ms now (WSL's file access limits it), and `STRATA_DF2_ROOM_VERIFY=1` reports a match |
| 5. FAST-SETUP | Done (`3fdbdce1`). WSL dry run; helpers checked against a two-NVMe / i7-14700KF sample; Windows dry run unchanged |
| 6. Docs | FORK.md and README updated on this branch |

With all of these in, the Windows build's first-token logits are still byte-identical to `speed-fork`'s (same MD5 at 3K / 29K / 50K, same answers).

Still open: speed on a native Linux PC (all items), and a real Linux FAST-SETUP run with two drives.

## Where it stood when planned

On 2026-11-09 the fork was built on Linux (WSL 2, Ubuntu 26.04, CUDA 13.3, gcc 15.2) with no errors. It ran the model twice at
temperature 0 and answered correctly both times:

- **Defaults only:** Linux applies only the first row of the fork's defaults (skip 0 and the two prompt rules).
- **The Windows defaults set by hand:** the unbuffered file tier, the dense arena, read-through, K/V lending and the prompt room
  all ran. The room's head check matched.

WSL reads the Windows drives slowly, so those runs say nothing about speed. Speed needs a native Linux PC (see Testing).

## The work, in order

### 1. The 28 GB RAM check (small)

`apply_fork_defaults` (`src/program/generate.cpp`) decides on `STRATA_PREFILL_AUTO_MAX=16384` with `GlobalMemoryStatusEx`, which
is Windows-only. `strata::platform::total_physical_memory()` (`src/platform/memory.cpp`) already exists on both systems (Linux:
`sysconf`). Use it.

- **Done when:** the Windows build's banner is unchanged, and a Linux PC with 28 GB or more lists `PREFILL_AUTO_MAX=16384`.

### 2. The fork's defaults on Linux with NVIDIA

Today the pack defaults are inside `#if defined(_WIN32) && !defined(STRATA_USE_HIP) && !defined(STRATA_GPU_METAL)`. Widen it to
CUDA builds on Linux too (not HIP, not Metal), switch by switch:

| Switch | Linux code | Plan |
|---|---|---|
| `UNBUFFERED_LOAD`, `DIRECT_BATCH` | O_DIRECT + kernel async reads (upstream) | On |
| `RAM_ADAPT`, `RAM_ADAPT_PAR`, `RAM_ADAPT_BG`, `COLD_OVERLAP`, `COLD_KEPT_READERS` | Shared code | On |
| `KV_LEND`, `DENSE_ARENA`, `STAGER_IN_PLACE`, the stager's 3 settings | Shared code | On |
| `RAM_THROUGH`, `PARTIAL_PIN` | `mlock`, limited by `ulimit -l` | On, plus a start-up check of the limit with a clear note when it is too low |
| `PREFILL_AUTO_MAX` | Item 1 | On at 28 GB or more |
| The prompt room (`DF2_ROOM_*`, `MTP_ROOM_*`), `HC_UPMIX` | Shared code | On |
| The lend cap (`g_fork_pack_defaults`) | Shared code | On |
| `EXPERTS_MIRROR` | Windows only | Item 3 |

- **Done when:** a Linux start with nothing set by hand lists the same switches as Windows, and its first-token logits are
  byte-identical to the same Linux build with those switches set by hand.
- **The memlock check:** read what `ulimit -l` allows and how much the tier needs. If it is too low, say so once at start, with
  the fix (`ulimit -l unlimited`, or `memlock` in `/etc/security/limits.conf`). Find out first whether `cudaHostRegister` needs
  the limit at all, or only `mlock` does.

### 3. The two-SSD mirror on Linux (largest)

On Windows, `open_direct` opens the second copy (`STRATA_EXPERTS_MIRROR`) unbuffered and checks its size, and `read_direct` splits
each request in half: one half from each drive. The Linux `read_direct` (kernel async reads, `io_submit`) has neither part.

- **Open:** a second O_DIRECT descriptor in the Linux branch of `open_direct`, with the same size check and the same log line.
- **Read:** in the Linux branch of `read_direct`, split each merged request at a 4 KiB boundary near its middle and queue the two
  halves on the two descriptors in the same `io_submit`. Keep everything else as it is (the 32 MiB merge limit, the waves of 256).
- **The prompt path's stager** reads through the same function, so it gets the mirror too. Check that it does.
- **Done when:** first-token logits with the mirror are byte-identical to without it (WSL can test this: a second copy already
  exists on C:). Then on native Linux, the mirror is faster, measured the same way as on Windows.

### 4. A faster head reload on Linux (optional)

After a long prompt, the prompt room re-reads the output head (497 MiB) from the model file. Windows reads it unbuffered with 4
threads (`upload_direct` in `src/core/native_head.cpp`), about 140 ms. Linux falls back to `cudaMemcpy` from the file's mapping
(2.8 s in WSL, though WSL's file access is slow too).

- **Port:** an O_DIRECT version of `upload_direct`: 4 threads, aligned 4 MiB reads into pinned buffers, then copies to the card.
- **Done when:** `STRATA_DF2_ROOM_VERIFY=1` still reports a match, and on native Linux the reload is under about 200 ms.

### 5. FAST-SETUP for Linux

`tools/fast_setup.py` stops on anything but Windows. A Linux version needs:

| Part | Windows today | Linux |
|---|---|---|
| RAM | Windows API | `/proc/meminfo` |
| CPU cores | `GetLogicalProcessorInformationEx` | `/sys/devices/cpu_core/cpus` and `/sys/devices/cpu_atom/cpus` on hybrid Intel; otherwise all cores |
| GPU | `nvidia-smi` | The same |
| Drives for the mirror | Volume APIs | `lsblk -J` (physical disk of each mount, so two partitions of one SSD are not offered) |
| The copy for the mirror | Copy and verify by hash | The same, preallocated with `fallocate` |
| Launcher | `run-fast.bat` | `run-fast.sh` |
| Wrapper | `FAST-SETUP.bat` | `FAST-SETUP.sh` (runs `setup.sh` first, then the tuning) |

- **Done when:** a dry run in WSL prints sensible settings, and a native run writes a working `run-fast.sh`.

### 6. Docs

FORK.md's Defaults table gains the Linux rows. The README gets a Linux section (requirements, `FAST-SETUP.sh`, the memlock note).
Linux speed numbers go in only after a native Linux measurement.

## Testing

- **Windows, after every engine change:** the Windows build must stay byte-identical to `speed-fork` (first-token logits, same MD5,
  at 3K / 29K / 50K), since nothing on Windows should change.
- **WSL (now):** builds, starts, and byte-identical checks within the Linux build: defaults on against the same switches by hand,
  mirror on against off, the head check.
- **Native Linux (later):** speed. Dual boot on this PC (same hardware) gives a direct Windows against Linux comparison:
  short, after 50K and follow-up decode, and 50K prompt reading, 3 runs each, alternated, as for every other change.
- **A community Linux tester:** the `tools/strata_bench.py` suite on their PC.

## Out of scope for now

- AMD (HIP) and Mac. The fork's defaults stay off there.
- Linux packages or an installer. Setup stays a source build (`setup.sh`).
