# Strata speed worklog

Changes made to reach higher speeds, measured against the vanilla engine.

- **Vanilla baseline:** Strata engine 0.1.55, repo commit c62d5e3, kept in `strata\engine-before-tg`.
- **Setup:** i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB, two 990 PRO 1 TB, Windows 11. Qwen3.8-Flash-Next IQ3_S, `--kv int8`,
  262K context, `--spec 3` (`--spec 4` since the 2026-11-03 install), 14 GiB expert RAM.
- **Method:** fresh engine per run, interleaved runs, four coding prompts (`tg_verify_bench.py`). "Text" is the mean tok/s of those prompts.

## Current state (2026-11-03, same-day, 5 runs each, mixed order)

| | Vanilla | Previous setup | Installed now | Installed against vanilla |
|---|---|---|---|---|
| Text | 37.6 tok/s (36.7-39.0) | 49.9 tok/s (49.0-51.3) | **57.3 tok/s (55.9-59.1)** | +52% |
| Generating after a 26K file | 28.2 tok/s | 37.1 tok/s | 45.5 tok/s | +61% |
| Follow-up | 32.1 tok/s | 44.4 tok/s | 51.5 tok/s | +60% |
| Image answer | 24.2 tok/s | 36.5 tok/s | 47.0 tok/s | +94% |
| Prompt reading | 989 tok/s | 1,432 tok/s | 1,496 tok/s | +51% |
| Tokens per step | 1.85 | 1.83 | 2.09 | |
| ms per step | 51.5 | 37.7 | 37.5 | |
| Lowest free RAM during the run | 1.9-2.3 GB | 2.0-2.4 GB | 2.0-2.3 GB | |

Text ranges do not overlap. Same model, quant, context, 14 GiB RAM tier and VRAM use (11.93-11.95 GB) in all three. Every text run stays well short of the 70 tok/s target.

Nothing is committed to git. Every change is an uncommitted source edit or a local file.

## Install of the candidate (2026-11-03)

- **What changed:** `strata\engine` now holds the candidate build (sha 4858167c7c3ef11a), `strata-daily.json` has `--pool-workers 7` and `--spec 4`, and `run-strata.bat` sets the seven switches below. The vision programs and `strata\engine-candidate` were not changed.
- **Backups:** `strata\engine-before-candidate`, `strata\strata-daily.before-candidate.json`, `run-strata.before-candidate.bat`. Each file was hash-checked against its original before the install.
- **Check from `run-strata.bat`:** ready in 23 s; 7 workers on the P-cores; 10 GiB of the RAM tier registered for the GPU. Three coding prompts of 400 tokens gave 45.3, 57.8 and 59.9 tok/s with the first after a fresh start, an image question gave 33.3 tok/s and read the name correctly, no damaged answers, and it stopped with no processes left. Single runs of one prompt, so not comparable with the table above.
- **In older sections below,** "installed" and "current" mean the setup of that date.

## Candidate (2026-11-03, installed the same day)

It was built and tested as a separate setup: `run-strata-candidate.bat`, `strata\engine-candidate` and `strata\strata-daily.candidate.json`. They still work, but `run-strata.bat` now does the same with its own log.

| Part | Switch | Patch needed |
|---|---|---|
| Ready experts compute during SSD reads, on kept reader threads | `STRATA_COLD_OVERLAP=1`, `STRATA_COLD_KEPT_READERS=1`, `STRATA_COLD_READER_CPU=22` | Yes, kept readers are new |
| 7 workers on P-cores, plus the host | `--pool-workers 7`, `STRATA_POOL_CORES=2,4,6,8,10,12,14` | No |
| Partial page-lock | `STRATA_PARTIAL_PIN=1` | No |
| Parallel RAM moves during the draft | `STRATA_RAM_ADAPT_PAR=1`, `STRATA_RAM_ADAPT_BG=1` | Yes |
| Deeper drafts | `--spec 4` | No |

- **Files:** `src/core/expert_source.cpp`, `include/strata/core/expert_source.hpp`, `src/program/generate.cpp`. All new switches are off unless set.
- **Session test (30 min, 175 mixed requests, on one engine):** no failed requests, no damaged answers, no steady slowdown: 51.3, 48.8, 50.1, 50.1, 49.0 tok/s per 5 minutes. This mix includes large files and growing conversations, so it is not comparable with the text figure, and the installed engine was not run through it. Lowest free RAM 1.49 GB.
- **Launcher check:** started from the `.bat` file, ready in about 15 s, answered a prompt, stopped cleanly.
- **Answers:** at `--spec 3` with the reproducibility switches, byte-identical to installed and vanilla; `--spec 4` changes the wording on every engine (see the identical-output check). Page-locked PCIe work and RAM moves were not part of that check.
- **Not established:** part-by-part gains are not isolated; they come from 2-3 run batches.
- **Still untested:** large pages, because they need the "Lock pages in memory" right, a security setting for the user to change.

## Kept

### 1. RTX 40 speed switches (2026-11-02)
- **What:** new `sm_89` table with 7 existing exact-kernel switches, which had no table for this GPU family. Off with `STRATA_SM89_PERF=0`.
- **Files:** `src/core/nv_defaults.cpp`, `include/strata/core/nv_defaults.hpp`, `src/core/nv_defaults_test.cpp`.
- **Result:** 38.2 -> 40.3 tok/s (+5.5%), 3 rounds, every run above every reference run.

### 2. No extra copy on unbuffered reads (2026-11-02)
- **What:** on Windows, aligned stage buffers let `read_direct` read a blob straight into its buffer. Off with `STRATA_DIRECT_IN_PLACE=0`.
- **Files:** `src/core/expert_source.cpp`, `include/strata/core/expert_source.hpp`.
- **Result, with change 1:** 30.8 -> 37.4 tok/s (3 rounds, fixed order), then 35.1 -> 41.1 tok/s (2 runs each, reversed order), +17-21%.
- **Not isolated:** the copy fix was never measured apart from change 1, so the share of each is inferred.

### 3. Reads split across two SSDs (2026-11-02)
- **What:** with `STRATA_EXPERTS_MIRROR` set, each read of 256 KiB or more takes half from each drive. The engine only compares file sizes.
- **Files:** `src/core/expert_source.cpp`, `include/strata/core/expert_source.hpp`, `run-strata.bat`, plus the copy at
  `C:\StrataMirror\experts.bin` (46.8 GB, SHA-256 matched the original `ED9D0A87...E67C`).
- **Result, 4 runs each, every run above every reference run:**

| | Single SSD | Dual SSD | Change |
|---|---|---|---|
| Text | 42.2 tok/s | 47.4 tok/s | +12% |
| Prompt reading | 997 tok/s | 1,389 tok/s | +39% |
| Generating after a 26K file | 31.4 tok/s | 37.3 tok/s | +19% |
| Follow-up | 34.5 tok/s | 41.5 tok/s | +20% |
| Image answer | 24.8 tok/s | 31.6 tok/s | +28% |

## Tried and not kept

| Idea | Result | Status |
|---|---|---|
| Read-ahead from the router prediction | 30.5 against 37.4 tok/s with it off, about 18% slower; about 70% of blobs never used | Code present, off by default (`STRATA_DIRECT_AHEAD`) |
| Promote experts to RAM on first read (`STRATA_RAM_ADAPT_ROUTED`, `_UNUSED`) | 41.3 and 40.5 against 40.4 tok/s, within noise; SSD reads about flat | Switches present, defaults unchanged |
| `STRATA_COLD_OVERLAP=1` | 35.0 / 33.4 against 32.0 / 37.9 tok/s, noise | Not used |
| `STRATA_RAM_ADAPT2=1` | 27.0 / 25.9 tok/s, slower | Not used |
| Compressing weights in RAM | 80 MiB sampled: zlib 105%, lzma 99.6% of original size | Ruled out |
| `--pipeline-windows` | Documented as off on a split onto one GPU | Not applicable to one card |
| Clang 23.1.1 for the CPU kernels (`iq_avx2.cpp` only) | 44.6 against 47.0 tok/s, 4 runs each, ranges overlap; CPU time per expert entry 39.1 against 38.7 us | Not installed; test build in `strata\engine-clang` |

## Measured, not yet decided

### Partial page-locking (`STRATA_PARTIAL_PIN=1`, 2026-11-03)
- **What:** an existing opt-in switch that lets the driver page-lock the expert cache, so the GPU runs part of the misses over PCIe. All 14 GiB locked here, about 0.4-0.9 experts per layer went over PCIe, and no extra RAM was used.
- **Result, 3 runs each, Edge closed:** the bigger differences are in prompt reading and image answers, and only prompt reading kept every run above every reference run.

| | Without | With | Change |
|---|---|---|---|
| Text | 48.6 tok/s | 50.3 tok/s | +3.6%, ranges overlap |
| Prompt reading | 1,422 tok/s | 1,501 tok/s | +5.6%, no overlap |
| Follow-up | 44.3 tok/s | 42.0 tok/s | within noise |
| Generating after a 26K file | 37.9 tok/s | 37.6 tok/s | no change |
| Image answer | 33.3 tok/s | 35.1 tok/s | within noise |

- **Cost:** startup 22-24 s against 20 s. Not yet tested in long sessions.

## Draft and cache screening (2026-11-03, 3 runs each, mixed order, reference 6 runs)

| Setting | Text tok/s | Tokens per step | ms per step | VRAM hits |
|---|---|---|---|---|
| Current (`--spec 3 --spec-min-p 0.5`) | 45.4 | 1.85 | 42.8 | 55.5% |
| `--adapt-swaps 96` | 47.1 | 1.91 | 42.0 | 60.1% |
| `--spec 4 --spec-min-p 0.35` | 46.5 | 2.27 | 50.3 | 54.3% |
| `--spec 4` + `STRATA_SPEC_COST=1` | 46.4 | 1.65 | 36.9 | 55.8% |
| `--spec 4` | 46.1 | 2.07 | 46.7 | 54.3% |
| `--spec 4 --spec-min-p 0.65` | 46.1 | 1.85 | 41.5 | 55.1% |
| `--adapt-decay 0.9` | 45.6 | 1.85 | 41.6 | 55.4% |
| `--spec 5` | 45.2 | 2.11 | 49.5 | 54.2% |
| `--adapt-swaps 48` | 43.0 | 1.80 | 43.6 | 58.6% |
| `--spec 3 --spec-min-p 0.3` | 39.9 | 2.02 | 52.6 | 54.7% |
| `--adapt-every 2` | 35.4 | 1.88 | 54.2 | 56.3% |

- Single runs vary by about +/-10%, so the differences under about 5% are not established at 3 runs each.
- Draft settings look flat: more tokens per step cost about the same in step time.
- `--adapt-every 2` starves the RAM tier, whose moves only run while no VRAM round is in flight: adapt wait fell from 1.57 to 0.09 ms and SSD wait rose from 12.2 to 24.9 ms.

## Cache simulator (2026-11-03, `cache_sim.py`, the engine's own rules on a 17-turn trace)
- **Check against the engine:** VRAM share 54.0% against 55.5% measured; going to 96 swaps gains 4.8 points against 4.6 measured. SSD reads are too high, 47 per window against about 33.
- **Experts per window, VRAM / CPU / SSD:** current 359 / 307 / 46.9; 96 swaps 397 / 279 / 47.0; 96 swaps + decay 0.9 400 / 276 / 44.8; 96 swaps + slots moving between layers 404 / 272 / 46.9.
- **Rejected by the simulator:** not counting skipped experts 357 / 305 / 49.7; favouring SSD-only experts x2 358 / 308 / 47.0 and x4 343 / 321 / 47.3.
- **Limits found later:** it counts experts rather than time, treats swaps as free and instant, and uses one trace from the older `--spec 3` setup. Its rejections are not proof.

## Cache miss audit (2026-11-03, `miss_audit.py`, same trace, offline, 5,567 windows)

| Rule | VRAM share | CPU entries per window |
|---|---|---|
| Shipped profile, never adapts | 31.0% | 588 |
| Learned profile, never adapts (trained on first half, scored on second) | 42.2% | 475 |
| Best fixed set for this trace (knows the whole trace) | 49.0% | 422 |
| Engine's rule | 54.2% | 376 |
| Decay 0.95 | 54.6% | 373 |
| Long memory x0.1 | 50.8% | 405 |
| 96 swaps | 59.4% | 335 |
| Knows the next 4 / 16 / 64 windows, 24 swaps | 44.5% / 55.3% / 63.7% | 437 / 354 / 292 |
| Perfect foresight, unlimited free moves | 78.6% | 183 |

- **Missed entries by last use:** last window 30.2%, 2-4 windows ago 20.4%, 5-16 24.4%, 17-64 15.5%, 65-512 8.3%, older 0.6%, never before 0.6%. Most misses are experts the engine had just used.
- **Reading:** the swap budget limits the cache more than the scoring does. Scoring changes are flat, fixed profiles lose to adapting, and even perfect knowledge reaches only 63.7% at the current budget.
- **GPU wait against hits:** in one candidate run, hit rates from 38.5% to 65.5% left GPU-reach wait at 11-12 ms, so extra hits don't look costly on the GPU side.
- **Real swap tests:** with 96 swaps, later requests in one run reached 54-67% hits, but adapt wait rose to about 2-3 ms and text speed was 55.1 against 57.8.
- **Turn starts:** first 20 windows 36.6% against 54.6% later, but only 1.8% of windows here. The trace showed about 5 gaps over 1 s, not the 17 turns it was recorded as, so this split is rough.
- **SSD experts per window (VRAM + RAM together, offline):** shipped profile 176, best fixed set 57, perfect foresight 13.5.

## Cache plan so far (2026-11-03, test engine `strata\engine-q`, nothing installed)

- **New simulator (`sim2.py`):** replays the engine's real swap steps. Against the logs: file reads 78.6 against 78.2 per window on one trace, CPU experts 3-7% high. It predicted real results in the right direction: 96 swaps -3.2% (real -4.7%), 48 swaps -0.7% (real -1.7%), more frequent rounds doubling SSD reads (real SSD wait doubled).
- **Swap rounds take about 3 windows whatever their size:** 110 ms for 24 or 75 experts, so window-by-window steps set the delay, not copy speed.
- **Swap choices are mostly right:** over the next 32 windows, the new expert was used 12-13 times against 3 for the evicted one, and 76% of swaps paid off.
- **Route prediction from tokens, offline:** (previous + token) was known for 30-36% of rows, with 57-63% precision. Too weak for the swap delay of several windows. Not built.
- **Skipping copy-back:** rejected by the simulator, 6-8% worse, because SSD reads rise.
- **RAM moves during rounds, 3 runs each (text, comparing at the same draft size and prompt):**

| Setting | Text tok/s | ms per window at the same draft size | SSD wait | Draft |
|---|---|---|---|---|
| Reference | 57.0 | | | |
| `STRATA_RAM_ADAPT_ANY=1` | 55.4 | -0.10 (+-0.52) | -0.58 | +0.51 |
| Also admit every read (`_ROUTED=1 _UNUSED=2 _PER=16`) | 53.3 | +1.21 (+-0.55) | -0.87 | +2.03 |

  File reads fell 30% with admission, but the extra copying costs more than the reads save. The simulator's +8-9% missed that cost. Not kept.
- **Stop drafting on low joint chance (`STRATA_SPEC_CUM_MIN`), 3 runs each:** reference 52.2, 0.3 53.3, 0.4 53.2, 0.5 50.2 tok/s. Within noise or worse. Not kept. All of this batch ran slower than the batch before it, while the PC was also in use; order was mixed.
- **96 swaps again, this batch:** 49.9 against 52.2 tok/s; hit rate 61.5% against 56.9%, but +1.57 (+-0.90) ms per window at the same draft size, nearly all adapt wait.
- **Each extra draft row costs 11-12 ms; acceptance is 79-83%, 70% and 61% for rows 1-3.**
- **Runs fall into two modes with the same settings:** swap wait 0.04 or about 0.75 ms per window. The slow mode costs 1.37 ms per window overall. Cause not established.
- **Thread placement, 3 runs each:** reference 52.4 tok/s (48.9-55.7). Adapt job thread on an E-core (`STRATA_ADAPT_JOB_CPU=26`) 51.4. Host on logical 1 (`STRATA_HOST_CORE=sibling`) 35.1, with one run at 19.3 and GPU wait of 33 ms: rejected. Both together 53.1 (52.7-53.5), -1.19 (+-0.91) ms per window at the same draft size, which is not beyond noise. Not kept.
- **Job thread on an E-core ended the slow swap mode in this batch:** tick 0.03-0.04 ms in all 6 runs, against 0.04-0.71 for the reference. But the draft got 0.6-0.8 ms slower, so the net gain is about zero.
- **`FAST-SETUP.bat` output:** started from the generated launcher, ready in 26 s, 7 workers on the P-cores, 50.8 tok/s on one 400-token answer, stopped cleanly. A clean-clone install has not been run.

## Candidate screening (2026-11-03, 43 runs, mirrored order, 2 runs per candidate unless noted, reference 5 runs at 49.9 tok/s)

| Candidate | Text tok/s | After 26K file | Follow-up | Image | ms per step | SSD wait | CPU | us per entry |
|---|---|---|---|---|---|---|---|---|
| Reference | 49.9 | 38.0 | 43.1 | 36.2 | 38.2 | 9.6 | 13.3 | 36.4 |
| Read overlap with kept readers (`STRATA_COLD_OVERLAP=1`, `STRATA_COLD_KEPT_READERS=1`) | 53.0 | 47.8 | 49.1 | 43.4 | 33.4 | 4.8 | 13.1 | 37.9 |
| 7 workers on P-cores only | 51.8 | 40.2 | 46.0 | 36.8 | 37.2 | 9.8 | 12.1 | 32.1 |
| 10 workers | 50.4 | 37.5 | 42.1 | 35.5 | 38.0 | 9.6 | 13.2 | 36.3 |
| `STRATA_FUSE_DECODE=1` | 49.8 | 40.5 | 43.0 | 35.1 | 37.1 | 9.2 | 12.8 | 36.8 |
| 96 swaps + decay 0.9 | 49.5 | 36.9 | 45.0 | 35.8 | 39.9 | 11.0 | 13.2 | 37.4 |
| `STRATA_COLD_OVERLAP=1` alone (new threads per layer) | 49.4 | 42.2 | 44.5 | 38.4 | 38.7 | 9.1 | 14.4 | 38.3 |
| 13 workers across P-cores and hyperthreads (`--pool-affinity p-cores`) | 43.2 | 31.3 | 36.9 | 29.6 | 45.4 | 13.5 | 14.5 | 38.2 |

- **Read overlap with kept readers:** the only clear win, and strongest on long sessions. Both of its text runs (51.9, 54.1) sit above all reference runs (48.4-51.6). SSD wait halved, from 9.6 to 4.8 ms.
- **The same overlap on new threads gained little:** creating threads per layer used up most of the benefit, which is why the earlier `STRATA_COLD_OVERLAP=1` test showed nothing.
- **Fewer workers are faster per entry:** 32.1 against 36.4 us with 7 P-core workers plus the host. Adding E-core workers makes each expert slower.
- **No effect on GPU wait (10.8-10.9 ms for all):** `STRATA_HC_UPMIX`, `STRATA_EXPDEC`, `STRATA_GRAPH_PRIO`+`STRATA_QIL`, `STRATA_FUSE_DECODE`, `STRATA_FUSE_WAIT`, `STRATA_GR_V3`. Text 49.0-49.8 tok/s, within noise of the reference.
- **Moved RAM copies into the draft (`STRATA_RAM_ADAPT_BG=1`):** adapt wait 1.49 -> 0.22 ms, draft 1.33 -> 2.39 ms, 49.0 tok/s. The copy takes longer than the draft, so most of the cost only moved.
- **Worse or flat:** `STRATA_DIRECT_POLL=1` 47.9 tok/s (adapt wait 2.75 ms), `STRATA_SPEC_COST=1` at `--spec 3` 47.6, prefetch distance 0 and 4096 at 48.2 and 48.4 (us per entry unchanged), `STRATA_PROC_HIGH=1` 49.2.
- **Second-choice counter:** `STRATA_MTP_TOP2=1` reported 0 hits from 223 rejections, but this is not a measurement. Nothing in the engine calls `record_top2`, so the counter compared against -1. Fixed for the next build.

## Read overlap combinations (2026-11-03, 19 runs, mirrored order, 2 runs each, reference 3 runs at 48.5 tok/s)

| Setup, all with read overlap | Text tok/s | After 26K file | Follow-up | Image | Prompt | ms per step | SSD | CPU | adapt | us per entry |
|---|---|---|---|---|---|---|---|---|---|---|
| 7 P workers + partial page-lock | 55.7 | 46.5 | 48.0 | 43.0 | 1,506 | 33.8 | 7.0 | 11.0 | 1.67 | 32.1 |
| 7 P workers + `--spec 4` | 55.5 | 42.0 | 51.1 | 46.3 | 1,427 | 37.1 | 6.5 | 13.8 | 2.52 | 31.1 |
| 7 P workers | 55.4 | 45.1 | 51.7 | 44.5 | 1,436 | 35.6 | 6.7 | 12.6 | 2.49 | 32.1 |
| 6 P workers | 54.5 | 42.4 | 47.3 | 43.1 | 1,412 | 35.8 | 6.3 | 13.5 | 2.21 | 34.7 |
| 5 P workers | 54.4 | 44.3 | 48.4 | 39.0 | 1,422 | 35.3 | 4.5 | 15.4 | 1.62 | 40.9 |
| 13 workers (P + E) | 53.5 | 41.0 | 48.0 | 43.2 | 1,421 | 36.4 | 5.8 | 14.2 | 2.46 | 37.6 |
| 7 P + 2 E workers | 52.9 | 41.0 | 49.0 | 40.2 | 1,419 | 37.3 | 5.9 | 15.6 | 1.95 | 40.2 |
| 7 P workers + 96 swaps + decay 0.9 | 52.2 | 44.4 | 50.2 | 40.3 | 1,430 | 36.9 | 6.7 | 11.3 | 4.69 | 33.3 |
| Reference (no overlap, 13 workers) | 48.5 | 37.3 | 43.7 | 35.3 | 1,431 | 41.7 | 11.9 | 14.4 | 1.56 | 35.6 |

- **Memory is not the limit at 6-8 threads:** threads x time per entry stays about flat (host included): 6 x 40.9 = 245, 7 x 34.7 = 243, 8 x 32.1 = 257 us. This answers the open question from the Clang test.
- **E-core workers slow everything down:** adding just 2 E workers to 7 P workers raised time per entry from 32.1 to 40.2 us. Why (slow tail tasks, or the ring bus clocking down) is not established.
- **96 swaps:** VRAM hits rise to 60.8%, and CPU time falls, but adapt wait rises to 4.69 ms and takes the gain.

## Parallel RAM moves, reader placement, second choice (2026-11-03, 2 runs each, reference 3 runs at 55.4 tok/s)

All with read overlap and 7 P workers.

| Added | Text tok/s | low-high | After 26K | Follow-up | Image | Prompt | tok/step | ms/step | adapt | draft |
|---|---|---|---|---|---|---|---|---|---|---|
| Page-lock + RAM moves parallel, during draft + `--spec 4` | 59.5 | 58.6-60.5 | 42.0 | 52.0 | 43.5 | 1,506 | 2.05 | 35.4 | 0.05 | 2.56 |
| Nothing (reference) | 55.4 | 54.1-56.2 | 47.0 | 50.2 | 42.3 | 1,423 | 1.88 | 34.8 | 2.39 | 1.35 |
| RAM moves parallel | 55.3 | 55.0-55.6 | 47.2 | 51.3 | 45.5 | 1,441 | 1.82 | 33.5 | 2.20 | 1.31 |
| Page-lock | 55.3 | 53.9-56.7 | 42.7 | 49.0 | 41.2 | 1,490 | 1.91 | 35.5 | 1.96 | 1.37 |
| RAM moves parallel, during draft | 55.2 | 55.1-55.2 | 46.0 | 51.2 | 44.2 | 1,428 | 1.89 | 34.9 | 1.16 | 2.21 |
| Page-lock + RAM moves parallel, during draft | 54.9 | 54.6-55.1 | 45.9 | 52.3 | 43.2 | 1,496 | 1.80 | 33.2 | 0.74 | 2.12 |
| `--pool-tasks 48` | 54.4 | 53.7-55.1 | 45.3 | 47.4 | 45.4 | 1,422 | 1.76 | 33.5 | 2.38 | 1.29 |
| Readers on P-core hyperthreads | 42.1 | 41.3-42.9 | 32.2 | 35.0 | 28.6 | 1,418 | 1.88 | 47.0 | 2.56 | 1.40 |

- **Noise warning:** tokens per step varies from 1.76 to 1.91 between runs of the same settings (sampling at temperature 0.6), which moves text speed as much as the changes do.
- **Readers beside workers are costly:** CPU time per entry doubled to 61.6 us, so readers stay on E-cores.

### Second choice for multi-branch drafts (1 run, 8 requests, 1,141 windows)
- **Measured:** 248 windows had a rejected draft. The draft model's second choice was right in 101 of them: 59 of 159 at depth 0 (37%) and 42 of 89 at depth 1 (47%).
- **Most it could gain:** 101 of 1,141 windows is about 0.09 tokens per step, against 1.9, or +4.7%.
- **Cost:** a branch row costs about the same as any other row, roughly 6-11 ms against a 35 ms step. A branch on every window costs about 20%; even with perfect knowledge of which windows will be rejected, 22% of windows x 8 ms is about 5%.
- **Decision:** not built. It breaks even at best and needs attention mask and K/V changes. This estimate is arithmetic on the measured rate, not a prototype.

## Draft and swap tuning on the best setup (2026-11-03, 2 runs each, reference 3 runs at 57.8 tok/s)

All with read overlap, 7 P workers, parallel RAM moves during the draft and, except where noted, page-locking.

| Setting | Text tok/s | low-high | After 26K | Follow-up | Image | Prompt | tok/step | ms/step |
|---|---|---|---|---|---|---|---|---|
| `--spec 5` | 58.8 | 58.1-59.5 | 46.5 | 52.7 | 42.2 | 1,498 | 2.19 | 38.2 |
| `--spec 4` (reference) | 57.8 | 57.6-58.1 | 47.3 | 51.4 | 46.6 | 1,500 | 2.00 | 35.5 |
| `--spec 4 --adapt-swaps 48` | 56.8 | 56.3-57.4 | 43.5 | 51.3 | 45.0 | 1,501 | 1.99 | 35.7 |
| `--spec 4 --spec-min-p 0.65` | 56.5 | 56.1-57.0 | 46.4 | 50.5 | 43.3 | 1,505 | 1.90 | 34.4 |
| `--spec 4`, no page-lock | 56.0 | 54.9-57.1 | 44.4 | 49.3 | 47.2 | 1,422 | 1.93 | 35.4 |
| `--spec 4 --spec-min-p 0.35` | 55.1 | 55.0-55.3 | 43.1 | 51.1 | 43.6 | 1,502 | 2.19 | 40.7 |
| `--spec 4 --adapt-swaps 96 --adapt-decay 0.9` | 55.1 | 52.3-58.0 | 44.4 | 51.5 | 46.0 | 1,502 | 1.95 | 36.2 |

- **Chosen:** `--spec 4`. `--spec 5` is within noise on text and lower on image answers. `--spec 4 --spec-min-p 0.5` is also the default that setup uses (`docs/DETAILS.md`).
- **Where the wait for cache moves goes (`STRATA_ADAPT_TIMING=1`, 1 run each):** in the 7-worker setup, 2.3-2.5 ms per step, of which RAM copies 1.0-1.8 and evicted-expert bookkeeping 0.3-0.8; the table upload is only 0.03. With 96 swaps, 2.7-5.4 ms. With page-lock, parallel RAM moves during the draft and `--spec 4` it falls to 0.05 ms; the same setup at `--spec 3` still waits 0.6-0.9 ms, and why is not established.

## Identical-output check (2026-11-03, temperature 0, 3 prompts)
- **Cause of the run-to-run difference: the CPU share of short prompts.** Prompt chunks under 1,024 tokens (on by default, `src/prefill/prefill.cpp:227-240`) give the CPU some experts, and how many depends on CUDA timings measured layer by layer. The docs' reproducibility switches don't turn it off. One prompt, 4 repeats per start, 2 starts, installed engine with tail skip off: on, 5 different first-token logits and 5 different answers of 8 (one start was stable, one differed every time); `STRATA_PREFILL_CPU_SHARE=0`, 1 and 1 of 8, including across the restart. The stable start's answer matched the share-off answer byte for byte.
- **Settings for the comparison:** `STRATA_IQ_MT_MIN=1`, `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_PREFILL_CPU_SHARE=0`, `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0`.
- **Same answers on all 3 prompts:** installed twice (control), vanilla, installed with RAM moves switched on, the new build, full candidate at `--spec 3`, and installed at `--spec 2`.
- **`--spec 4` changes the wording:** 2 of 3 prompts differ from `--spec 3`, from character 398 of the first. Vanilla, installed and candidate at `--spec 4` all give the same answers as each other, so this comes from the stock option and not from the patches. Why depth 4 differs while depth 2 doesn't was not established, and answer quality at each depth was not measured. `--spec 0` isn't a valid setting for this pack (it needs at least 2), so there is no no-draft reference.
- **Not covered:** adaptive swaps were off and the logs show 0 exchanged, and `--pcie-frac 0` keeps the GPU from using page-locked RAM. So the page-lock PCIe path and the parallel RAM moves weren't exercised, and PCIe work can change rounding according to the docs. Only 3 short prompts of up to 300 tokens were compared. In normal use the CPU share, tail skip, PCIe work and swaps stay on, so answers can still vary between runs on every engine.
- **Earlier result replaced:** the first version of this check called the result inconclusive because the share was still on, so the control differed from itself.

## Measurement tools added
- `STRATA_ROUTE_TRACE=<path>`: records expert routing to a file (`src/core/expert_source.cpp`). Off unless set.
- Simulator on a 17-turn session: LRU 52.5% VRAM / 40.2% RAM / 7.4% SSD; a perfect cache 74.7% / 22.0% / 3.3%. The engine measures about 55% VRAM and 7% SSD.

## Findings

### GPU profile (2026-11-03, `STRATA_VERIFY_PROFILE=1`, one run, 39 ms step)
- About 15.5 ms is real GPU work, and the GPU sits waiting on the CPU for about 20 ms.
- There is no driver call per layer, because the window is one prebuilt graph and the host watches a flag.
- VRAM experts: 547 in 2.2 ms. CPU experts: about 300 in 14 ms.
- Largest fixed GPU costs: dense matvec 3.2 ms, hyper-connection 3.2 ms, output projection 1.6 ms, head 1.2 ms.
- The PCIe share stays at 0 because the 14 GiB cache is not page-locked (`pinned 0.00 GiB`), although the engine computes `pcie_frac` 0.20-0.25.

### Benchmark noise floor (2026-11-03)
- A first "Clang" run was invalid: the build recompiled the file with MSVC, so `engine-clang` matched the installed engine except for 9 bytes.
- That accidental same-engine test, 3 runs each: 47.5 against 46.7 tok/s (1.7% apart), with single runs from 45.0 to 49.0. Differences this small mean nothing at 3 runs each.

### Clang against MSVC on the CPU kernels (2026-11-03, 4 runs each, mixed order)
- **Build:** only `iq_avx2.cpp` changed compiler, linked by hand with `link_clang.bat`; 183 of 221 sampled pieces of Clang's code were found in the engine.
- **Generated code:** in the 8 matched VNNI kernels, vector stack accesses fell from 50 to 2, and static instructions rose from 1,273 to 1,895.
- **Speed per entry:** 39.1 us with Clang against 38.7 us with MSVC, about 1% apart, so the better register use did not make the kernels faster.
- **Text speed:** 44.6 tok/s (48.0, 42.1, 44.8, 43.6) against 47.0 tok/s (48.8, 45.9, 49.1, 44.1). The Clang runs also did about 6% more CPU entries per step (384 against 361) and had slightly lower VRAM hit rates; whether that is chance or caused by the build was not established.
- **Other measures, Clang against MSVC:** prompt 1,426 against 1,404 tok/s, after a 26K file 35.9 against 35.2, follow-up 43.2 against 43.3, image 31.3 against 28.6 (one MSVC run at 20.4).
- **Reading:** this points away from kernel code generation as the limit and towards memory, but it does not prove that. Lowering the worker count would test it directly.

### CPU memory use (2026-11-03)
- At about 39 us per expert entry and about 2 MiB per expert, the pool moves about 50 GB/s. A dual-channel DDR5-6000 peak is 96 GB/s. Whether memory or code is the limit is not established.

## Local files
- **Engine in use:** `strata\engine\strata.exe`.
- **Backups:** `strata\engine-before-tg` (vanilla) and `strata\engine-before-mirror` (changes 1 and 2).
- **Test builds:** `strata\engine-tg`, `strata\engine-clang` (Clang CPU kernels), `strata\engine-p` (patched test build) and `strata\engine-candidate` (copy of the final `engine-p`).
- **Other:** `strata\engine-vision-gpu` is the GPU picture reader, and `strata\strata-daily.cpu-vision.json` is the config backup from before GPU vision.
