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

This table is the 14 GiB setting. Daily since 2026-11-04: `--resident-budget-gib 10 --conversation-cache-mib 0`, coding answers 48.2 tok/s, see the sections below.

## Comparison video recording (2026-11-03)

Vanilla against installed, 5 runs each, fresh engine per run, mixed order (VFFVFVVFVF), the same four coding prompts as the table above but streamed. `demo\compare\strata-vs-fork.mp4` and `.gif`.

| | Vanilla | Installed | Installed against vanilla |
|---|---|---|---|
| Text | 36.0 tok/s (34.1-38.2) | 55.6 tok/s (53.0-56.5) | +54% |
| Prompt reading | 1,003 tok/s | 1,484 tok/s | +48% |
| Code clip, thinking off, one prompt | 33.1 tok/s (31.8-36.5) | 62.3 tok/s (60.5-64.7) | +88% |
| Generating 150 tokens after the big file | 18.5 tok/s | 34.5 tok/s | +86% |

- **Agreement:** text and prompt reading are close to the table above. No damaged answers.
- **Code clip:** the race on screen shows this, so it looks wider than the +54% on the end card. Why code with thinking off gains more was not established.
- **After the big file:** both engines came in well below the 28.2 and 45.5 in the table above, and the cause wasn't established, so this row was left off the video.
- **Vanilla drift:** its text speed fell from 38.2 to 34.1 across the session while installed stayed at 53.0-56.5. The cause wasn't established.

## SSD writes and paging (2026-11-03)

Windows counters sampled every 5 s on the installed engine, mixed coding session (`soak.py`). Writes are for the whole PC, both SSDs together.

| Session | Disk writes | Paged out | Paged in | Available RAM, lowest / median | Page file, highest |
|---|---|---|---|---|---|
| Strata stopped, 1.9 min | 3.0 GB/h | 0 | 0.08 GB | 20.0 GB lowest | 2.5% |
| Normal, 20.4 min, 105 requests | 4.2 GB/h | 0.3 GB/h | 9.16 GB | 0.28 / 1.13 GB | 2.5% |
| Pressure test, no extra app, 4.8 min | 6.2 GB/h | 4.3 GB/h | 2.02 GB | 0.25 / 1.22 GB | 3.2% |
| With a 4 GB app, 4.8 min | 5.2 GB/h | 2.6 GB/h | 6.28 GB | 0.03 / 0.49 GB | 6.1% |
| With an 8 GB app, 4.5 min | 145.6 GB/h | 138.8 GB/h | 36.20 GB | 0.00 / 0.27 GB | 9.0% |

- **Extra app:** another process rewrote every page of its memory every 5 s, which is harsher than a real app. Its own pages are part of the paging figures; the split between it and Strata was not measured.
- **Strata's working set:** 19.2 GB, then 16.9 GB with the 4 GB app and 10.2 GB with the 8 GB app, so Windows took memory from Strata.
- **Reliability:** no failed or damaged requests in 172 requests across both sessions. Strata's own file writes were about 0.
- **Lifespan against a 600 TBW drive, 8 h a day:** about 49 years at 4.2 GB/h, about 1.4 years if the 8 GB case ran all the time.
- **Reads:** 3,729 GB in the normal session, about 3 GB/s. Reads don't count toward the rating. Drive temperature was not measured.
- **Limit:** the two no-extra-app runs differ by 14 times in paged out (0.3 and 4.3 GB/h), so small figures here are noisy.

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

## `STRATA_PREFILL_CPU_SHARE=auto` (2026-11-03, stopped early)
A friend reported about 60% faster prompt reading with no decode loss on 8 GB VRAM, DDR4 and a lower-end SSD. Tested here against unset on the daily setting, fresh engine per run; stopped at the user's request after 2 unset and 3 auto runs (one auto run unfinished), and the 8 GB app runs never ran. Not enough to decide, and not this PC's case: the engine's log showed the share already on in the unset runs.
- **Prompt reading, first round, unset against auto:** 300 tokens 112-116 against 112-114 tok/s; 2,500 tokens 631-635 against 594-615; 26,000 tokens 1,410-1,428 against 1,388-1,404. No gain, if anything slightly slower.
- **Decode:** 55.2-55.6 against 56.1-56.4 tok/s in the finished runs; the unfinished auto run was at 50.7.
- **Paged out:** 0.6-13.1 against 16.9-24.2 GB/h over windows of about 2 minutes, too short to rely on.
- **Lowest available RAM:** 1.64-1.71 against 1.56-1.63 GiB.

## `--resident-budget-gib` 10 against 14 (2026-11-04)
Daily setting, two SSDs, fresh engine per run. 4 runs at 10 GiB; 1 run at 14 GiB, run just before and not alternated, so the gaps are rough.

| | 14 GiB (1 run) | 10 GiB (4 runs) |
|---|---:|---:|
| Coding answers | 51.5 tok/s | 48.2 tok/s (47.3-49.4) |
| Prompt, 26K tokens | 1,491 tok/s | 1,424 tok/s (1,421-1,427) |
| After that prompt | 42.6 tok/s | 34.6 tok/s (32.5-37.0) |
| Follow-up | 50.5 tok/s | 40.0 tok/s (39.1-41.2) |
| Image | 40.4 tok/s | 29.4 tok/s (27.0-30.7) |
| Lowest available RAM | 0.7 GiB | 3.46 GiB (3.27-3.58) |
| Paged out | 2.5 GB/h | 0.0-0.5 GB/h |
| Damaged answers | 0 | 0 |

- A later rerun at 10 GiB, stopped after 3 finished runs: coding answers 45.6, 47.0 and 46.3 tok/s, prompt 1,411, 1,429 and 1,437, after the prompt 26.4, 33.5 and 33.1, follow-up 41.0, 41.2 and 38.7, image 33.0, 32.3 and 33.1, lowest available RAM 4.46, 4.11 and 3.56 GiB, paged out 1.0, 0 and 0 GB/h.
- The same four coding prompts gave 55.2-56.4 tok/s at 14 GiB in the CPU share test, so the coding loss at 10 GiB is about 6-14%.
- At 10 GiB the whole tier page-locks ("page-locked and mapped", 3 s); at 14 GiB page-locking was refused and the tier took 12 s. The engine was ready in 12 s against 21 s.
- Paging windows were about 1.5 minutes, too short for an SSD-wear figure.
- **Test script flaw:** the "8 GB app" in `pressure_test.py`, `share_test.py` and `budget_test.py` grows with `held[0] += bytearray(...)`, which from the code briefly holds about twice the memory (not measured). The 145.6 GB/h paging figure above came from this script and may be too high. One 10 GiB run with the app was left out of the table.

## What is in the engine's memory (2026-11-04)
Stock model, 10 GiB tier, one short session; `memtools/mem_map.py` walks the address space with `VirtualQueryEx` and `QueryWorkingSetEx`. One run.

| In RAM, GiB | Ready | After an 88K prompt |
|---|---:|---:|
| Expert tier | 10.00 | 10.00 |
| Blocks of 64-256 MiB (checkpoints and parked conversations; this session parked 2) | 0.32 | 1.51 |
| Blocks of 16-64 MiB | 0.09 | 0.73 |
| K/V regions (13 of 266 MiB, 1 of 324 MiB) | 0.00 | 0.35 |
| Under 16 MiB (3,938 to 4,503 regions), not yet identified | 1.61 | 2.95 |
| **Total private** | **12.02** | **15.54** |

- About 14.7 GiB more is committed but not in RAM at the end, including 3.34 GiB of K/V not yet filled and 122 blocks of 64 MiB that hold almost nothing. That costs commit limit, not RAM.
- **This map misses memory the driver pinned:** it doesn't appear in the process working set. The ledger below finds it, and the K/V row above is therefore too low.
- The 1,466 and 2,018 MiB of weights in the log are not held twice in the process's own memory: the blocks that size hold almost nothing. The map can't rule out a copy pinned by the driver.

## Whole-PC memory ledger (2026-11-04)
Stock model, 10 GiB tier, one conversation of three large turns reaching 85K tokens, no parking, no images, browser closed. `memtools/ledger.py`, one run. Free RAM with the engine off: 22.62 GiB of 31.8.

| GiB | Ready | 85K context | After a minute idle |
|---|---:|---:|---:|
| Available (almost all standby) | 5.80 | 2.62 | 2.38 |
| Engine, working set | 12.06 | 16.56 | 16.56 |
| Engine, not in its working set (change in unaccounted RAM against engine off) | about 3.7 | about 3.0 | about 3.0 |
| GPU counter `Shared Usage` | 14.52 | 14.52 | 14.52 |
| GPU counter `Dedicated Usage` | 10.35 | 10.48 | 10.48 |
| Page file | 2.1% | 2.1% | |

- **The K/V looks pinned from the start:** about 3-3.7 GiB is held outside the working set from startup, which fits the logged 3.09 GiB. This is inferred from the accounting, not traced.
- **Growth:** the working set rose 4.5 GiB by 85K tokens, in steps: +0.86 after a short request, +1.14, +0.23, then +2.27 on the last turn. It didn't fall during a minute idle. Most of it isn't identified; checkpoints account for about 0.6.
- **No leak at exit:** available RAM returned to 22.49 GiB after the engine stopped.
- **No paging in this run,** at 2.38 GiB available.
- **Image encoder:** not running without an image, so it costs nothing until one is sent.
- **Correction:** this run did park. The +2.27 GiB on the last turn was a 1,557 MiB snapshot of the same conversation, taken because the prompt differed 308 tokens from its end.

## What the engine's memory growth is (2026-11-04)
`memtools/mem_diff.py` diffs every private allocation turn by turn. One conversation to 114K tokens, one run, no large park.

| Added to the working set | Size | What it is |
|---|---:|---|
| 256 blocks of 2.5 MiB, filled on the first request | 640 MiB | Expert stage buffers, from the code's "pool grows to ~257 buffers" |
| 6 blocks of 112 MiB, then replaced one for one | 672 MiB | Checkpoints |
| 33 blocks of about 21 MiB, on the first long prompt | 650 MiB | Not identified |

The working set was flat from 85K to 114K tokens (14.58 GiB) and peaked at 14.4-14.6 GiB in all four runs below. So growth is bounded: nothing here climbs with context beyond these, apart from parked snapshots, which are capped by `--conversation-cache-mib`.

## `STRATA_PARK_TAIL_MAX` (2026-11-04, off by default, not installed)
Skips parking when a request is the same conversation with at most its last N tokens rewritten. Test engine `engine-memtest`, 2,048 tokens, client rewriting the end of its last answer on every turn, 2 runs each, about 116K tokens.

| | Off | On |
|---|---:|---:|
| Peak working set | 14.62 GiB | 14.52 GiB |
| Lowest available RAM | 2.0 GiB | 2.01 GiB |
| Tokens read | 233,305 | 233,289 |
| Decode | 43.6 tok/s | 43.5 tok/s |
| Damaged answers | 0 | 0 |

- **It works but saved nothing here:** it did skip all 4 parks, but with it off those parks were already refused by the RAM floor (`skip parking`). It can only help when more than the snapshot plus 2,560 MiB is free, and then the park doesn't take RAM below that floor. Not worth installing on this evidence.
- **Bigger finding:** after one short side request, the 116K conversation was read again from the start in both settings: about 116,780 tokens, 77-79 s. The conversation's snapshot no longer fits the cache or the floor, so it's lost.

## Real Claude Code session, saving off (2026-11-04)
Installed engine, 10 GiB tier, `--conversation-cache-mib 0`, Claude Code doing a code review on copies of 5 engine files, 25 minutes, stopped by the time limit. `memtools/real_session.py`, one run.

| | |
|---|---:|
| Paged out | 0.43 GB, 1.0 GB/h, 2 bursts of 20 MB/s or more (target: 0.3 GB/h) |
| Page file | 3.1% to 4.4% |
| Peak engine working set | 14.48 GiB |
| Available RAM | lowest 3.33 GiB, median 4.23 GiB |
| Requests | 22, up to 217K tokens, then compaction |
| Speed | 44.4 tok/s over 56,946 tokens |
| Long-context re-reads | 1, 14 s, after the compaction |

- **Not comparable with the OpenCode session's 4.2 GB/h:** different harness, task and apps open. Saving off may have helped, but this doesn't show how much.
- **No side requests:** the log shows no parking attempts, and `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` was set. So the cost of saving off on a real conversation switch wasn't tested here.
- **When it paged:** all 0.43 GB fell in 3 samples, 3-7 minutes into the session (36, 28 and 16 MB/s), with 3.7-4.3 GiB available and the engine's working set unchanged. Nothing of 5 MB/s or more in the remaining 18 minutes. So in this run it was one push when the load started, not a steady rate.
- **Engine off, 10 minutes, apps open:** paged out 0.00 GB, available RAM at least 18.9 GiB, all disk writes 7.7 GB/h. So the paging comes from the engine's memory pressure, not from Windows on its own.

## `--lookup-chain 3` (2026-11-04, no gain, not adopted)
Installed engine, 10 GiB tier, saving off, fresh engine per run, 3 runs with the flag on. `memtools/lookup_test.py`. No off runs in this test: compared with the 10 GiB section above, so only a clear gap would count.

| | `--lookup-chain 3` (3 runs) | Earlier 10 GiB, off |
|---|---:|---:|
| Coding answers | 48.5 tok/s (47.2-49.7) | 48.2 (47.3-49.4), rerun 45.6-47.0 |
| Prompt, 26K tokens | 1,434 tok/s (1,432-1,436) | 1,424 (1,421-1,427), rerun 1,411-1,437 |
| After that prompt | 35.3 tok/s (34.4-36.5) | 34.6 (32.5-37.0), rerun 26.4-33.5 |
| File rewrite, 700 tokens | 33.1 tok/s (32.6-33.6) | none |
| Damaged answers | 0 | 0 |

- **Chain drafts were rare:** 11-28 of about 1,100-1,200 windows used one, and accepted chain drafts were 0.44%, 0.58% and 1.45% of the output. That is too few to move the speed.
- **Suffix drafting probably already covers this:** it is on by default. In the first run's rewrite it drafted in 101 of 119 windows with 553 of 556 accepted; the chain added 9 drafts in 4 windows. That it leaves the chain little to do is an inference, not tested with suffix drafting off.
- **Cost:** 522-525 MiB of VRAM free at start against 593 in the earlier real session, one slot fewer (1,977 against 1,978), lowest free 358-373 MiB. Different runs and app state, so the gap is rough.
- **Follow-up not compared:** 31.3 tok/s (30.1-32.2) against 40.0, but the old prompt is lost with its script and this answer was short (126 tokens in the first run).
- **Rewrite is slow although almost every draft is right:** the SSD wait is the whole drop. First run, rewrite against the coding request just before it (one request each; all 3 runs gave 32.6-33.6 tok/s and the first two hit rates checked were 33%):

| | Coding (450 tokens) | Rewrite (700 tokens) |
|---|---:|---:|
| Per token | 22.1 ms | 30.1 ms |
| SSD wait per token | 8.9 ms | 18.8 ms |
| Everything else per token | 13.2 ms | 11.3 ms |
| Tokens per window | 1.9 | 5.9 |
| Experts served from VRAM / RAM / SSD | 60 / 29 / 11% | 40 / 22 / 37% |
| Expert blobs from SSD per token | 55 | 122 |
| Read from SSD while generating | 43.6 GB (4.4 GB/s) | about 174 GB (8.3 GB/s), worked out as 84,211 experts at 2.07 MB; the log's 209 GB seems to include the prompt read, since the 300-token answer after the 26K prompt shows 199 GB |
| Experts per layer that needed a read | 3.3 | 14.8 |
| Cache swaps per token | 3.2 | 1.0 |

- **Why the SSD share rises, not established:** swaps run every 4 windows, so with bigger windows the cache learns about 3 times slower per token (the comment at `generate.cpp:3847` describes the same effect). The new text and the 1,657-token prompt before it may also count: the requests after the 26K file had 2.2-2.9 tokens per window and still only 36.6-41.9% hits. Earlier tests rule out the easy setting on coding prompts: `--adapt-every 2` gave 35.4 tok/s and 48 or 96 swaps gave no gain. Neither was tried on a rewrite.

### `--suffix-draft 0` on the same workload (2 runs, against the 3 `--lookup-chain 3` runs above, not alternated)
`memtools/suffix1`. The chain fired too rarely to matter, so those runs stand in for the default. Coding prompts had thinking on; the rewrite and the requests after the file had it off, so thinking and text type are not separated.

| | Suffix drafting on | Off |
|---|---:|---:|
| Coding answers | 48.5 tok/s (47.2-49.7) | 45.4 (44.3-46.5) |
| File rewrite | 33.1 tok/s (32.6-33.6) | 31.6 (30.6-32.5) |
| Rewrite tokens per step | 5.9 | about 3.7 (700 tokens, 511-512 accepted drafts) |
| Rewrite hit rate | 32.9-34.2% | 39.8-42.5% |
| Rewrite experts from SSD | 36-37% | 29-32% |
| Rewrite blobs from SSD per token | 119-122 | 106-117 |
| Rewrite SSD wait per token | 18.0-18.8 ms | 18.3-20.2 ms |
| Rewrite everything else per token | 11.3-11.9 ms | 12.5 ms |
| After the 26K file | 35.3 tok/s (34.4-36.5) | 28.4 (26.5-30.3) |
| Damaged answers | 0 | 0 |

- **Big steps are not the main cause of the slow rewrite.** Smaller steps raised the hit rate, but SSD reads per token barely fell and SSD wait did not fall. Steps only shrank to about 3.7 tokens, not to the coding 1.9, so this does not rule out a part for step size.
- **Reading per expert was not slower in the rewrite:** 0.18 ms per expert against 0.26 on coding (R per layer divided by experts per layer, first run), so the count of reads is the problem, not the drives slowing down.
- Suffix drafting stays on. `--lookup-chain 3` left 522-525 MiB free and 1,977 slots here against 555-584 MiB and 1,980 slots without it, so it does cost a little VRAM.

### Coding prompts with thinking off (2 runs, default flags, against the 3 `--lookup-chain 3` runs above, not alternated)
`memtools/nothink1`, broken down with `memtools/req_breakdown.py`. Only the four coding prompts changed; the rewrite and the requests after the file had thinking off in both.

| | Thinking on | Thinking off |
|---|---:|---:|
| First coding request | 39.7-42.0 tok/s | 32.9-34.2 |
| Its SSD blobs per token | 52-56 | 101-102 |
| Other three coding requests, mean | 51.0 tok/s (45.3-54.9) | 56.8 (50.8-62.1) |
| Their SSD blobs per token | 34-55 | 40-54 |
| Their tokens per step | about 1.9 (not recounted per request) | 3.1-3.4 |
| Their everything-else time per token | 12.9-14.2 ms | 9.8-11.3 ms |
| Rewrite afterwards | 33.1 tok/s, 119-122 blobs per token, hit 33-34% | 35.6, 96-102, hit 42-43% |
| Answer after the 26K file | 35.3 tok/s, hit 40-42% | 26.0, hit 35% |
| Damaged answers | 0 | 0 |

- **Thinking is not why the rewrite is slow:** warmed-up new code with thinking off read about as little from the SSD as with thinking on.
- **Copying is not needed for high reads either:** answers right after the 26K file are not copies and needed 121-129 blobs per token with thinking off. Every request above about 95 was a first request or came after a long prompt. Whether the long prompt itself is the cause is not tested. The follow-up (337 new tokens, 112-126 blobs per token) was also short, and short answers have low hit rates anyway.
- **What ran before matters:** after code, the rewrite is faster and the English answer slower; after thinking text, the reverse. This fits a cache holding the recent kind of text, but it is only these two orderings.
- **Not a free gain:** thinking off changes the answers, so it is not comparable as a speed setting. The 450-token limit may also mean the thinking-on "coding" figures are mostly reasoning text, not code; the answers were not saved, so this is not checked.

### Coding prompts after about 1,600 unrelated tokens (2 runs, thinking off, against the thinking-off runs above, not alternated)
`memtools/longprompt1`. Each coding prompt starts with a different 6,000-character part of `serve/server.py`, so nothing is reused.

| | Short prompt | Long prompt |
|---|---:|---:|
| First coding request | 32.9-34.2 tok/s, 101-102 blobs per token | 33.9-34.0, 89-94 |
| Other three coding requests | 56.8 tok/s (50.8-62.1), 40-54 blobs per token | 53.6 (49.0-58.8), 39-48 |
| Rewrite afterwards | 35.6 tok/s, 96-102 blobs per token | 35.3, 99-101 |
| Answer after the 26K file | 26.0 tok/s, 121-129 blobs per token | 31.3, 93-100 |
| Damaged answers | 0 | 0 |

- **A long prompt is not the cause:** at about 1,600 tokens, warmed-up new code reads no more from the SSD. A 26K prompt was not tested this way.
- **The rewrite stays at 96-122 blobs per token in every test so far,** after warmed-up new code too, so the copied text itself is the best remaining explanation. Why copying needs more experts is not established.
- Everything-else time on those three requests was 11.1-13.2 ms per token against 9.8-11.3, from 2 runs each.

### Speed against context (the real Claude Code session above, per-request logs, one session, tasks change along the way)
| Context | Tok/s | Per token | SSD wait | Everything else |
|---|---:|---:|---:|---:|
| 29-56K, 3 requests | 47.5 | 21.1 ms | 5.8 ms | 15.3 ms |
| 158-217K, 5 requests | 42.6 | 23.5 ms | 7.2 ms | 16.3 ms |

Means weighted by tokens, requests of 2,000 tokens or more. Rows per token stayed at about 1.4.

| Context | Everything else per row | K/V blocks per row | Not in VRAM | K/V from RAM per row |
|---|---:|---:|---:|---:|
| 29K | 11.04 ms | 6,164 | 0.10% | 0.02 MiB |
| 56K | 11.19 ms | 6,165 | 0.91% | 0.23 MiB |
| 107K | 11.10 ms | 6,172 | 2.90% | 0.72 MiB |
| 158K | 12.50 ms | 6,167 | 4.03% | 1.00 MiB |
| 216K | 11.99 ms | 6,164 | 6.68% | 1.66 MiB |

- **The attention work per row does not grow with context:** K/V blocks per row are flat. What grows is the share outside the 32,768 cells per layer held in VRAM.
- **Part of the drop follows K/V reads from RAM (correlation only):** about 1 ms per row above 150K. 1.66 MiB per row at the measured 24.1 GB/s PCIe is about 0.07 ms, so bandwidth alone does not explain it; small-read delay would have to, which is not measured.
- **More than half is SSD wait:** +1.4 ms of the +2.4 ms per token. Hit rates fell from 57-61% to 51-58%, but the agent's task changed too, so context is not established as the cause of that part.
- **Correction:** an earlier reply said SSD wait barely changes with context. By the weighted means it is the larger part.
- `--kv-resident 65536` was not tried. Estimated from the 13.7 KB per token in `docs/DETAILS.md`: about 450 MiB more VRAM, about 230 of 1,978 cache slots, for at most about 1 ms of 23.5 ms per token.

### All K/V in VRAM (no `--kv-resident`, 2026-11-04, rejected, stopped after one run each)
`memtools/kv_test.py`, `memtools/kv1`. Installed engine, 10 GiB tier, saving off, one run each, daily first.

| | Daily (32,768 cells in VRAM) | All K/V in VRAM |
|---|---:|---:|
| Expert cache | 1,978 slots, 3.80 GiB | 367 slots, 0.71 GiB |
| Coding answers | 44.6, 49.5, 55.6, 49.1 tok/s | 31.0, 37.5, 35.2, 35.2 |
| Coding hit rate | 56-67% | 31-35% |
| Coding experts from VRAM / RAM / SSD | 59-68 / 25-30 / 7-11% | 32-38 / 48-49 / 14-19% |
| Everything else per token | 12.4-14.9 ms | 15.9-20.6 ms |
| 204K-token prompt read | 143 s (1,423 tok/s) | not finished after about 15 minutes, stopped |
| Answers at 204K | 36.3-47.3 tok/s | no result |
| VRAM free at start | 544 MiB | 670 MiB |

- The engine warned that all 238 lendable slots would read from the SSD during long prompts. The prompt read may still have been progressing when stopped; the log prints nothing until it ends, so a hang is not ruled out.
- So the long-context question this was meant to answer has no number on this setting. It costs too much before that point.
- Daily at 204K after warm-up gave 47.3 tok/s on two 800-token answers (hit 61.5-64.0%), which is as fast as the session's early requests, in a run with a synthetic prompt. One run, thinking on.

## DFlash2 drafter on a compiled v0.2.10 (2026-11-04, 2 runs each, mixed order, not installed)
`memtools/dflash_test.py`, `memtools/dflash1`. The installed engine 0.1.55 runs DFlash2 only from the command line, not with `--serve`, so v0.2.10 (engine 0.1.56) was compiled from the tag in a separate worktree, `strata-0210`, with the fork's engine change cherry-picked. Same build flags as the installed engine. The fork's `STRATA_SPEC_CUM_MIN` was dropped in the merge. Drafter: the Q4_0 pack with Q4_0 selector codebooks, `qwen3.8-flash-next-dflash2.q4_0-selq4.pack`.

All arms: 10 GiB tier, saving off, 7 workers, vision on, temperature 0.6. MTP arms use `--spec 4 --spec-min-p 0.5`. The DFlash2 arm uses what the v0.2.10 launcher sets: `--spec 6 --draft-conf 0.8`, no MTP layer, windows up to 8 tokens. Mean (low-high), tok/s.

| | Installed + MTP | v0.2.10 + MTP | v0.2.10 + DFlash2 |
|---|---:|---:|---:|
| Coding, thinking on, all 4 | 47.1 (46.6-47.6) | 47.4 (45.7-49.1) | 47.4 (45.0-49.9) |
| Same, without the first request | 49.4 (48.8-50.0) | 49.6 (47.7-51.5) | 49.0 (46.5-51.6) |
| Coding, thinking off | 51.9 (50.2-53.6) | 53.5 (52.1-54.9) | **62.3 (57.4-67.2)** |
| Rewrite a file | 36.0 (35.4-36.7) | 38.0 (36.1-39.8) | 36.2 (34.2-38.2) |
| Right after a 26K-token file | 29.3 (27.7-30.9) | 28.6 (27.0-30.3) | **37.1 (36.3-38.0)** |
| Follow-up on that file | 31.9 (31.5-32.4) | 31.7 (31.1-32.2) | 34.1 (33.3-34.9) |
| Prompt read | 1,417 | 1,415 | 1,392 |
| Expert cache slots | 1,978 | 1,974 | 2,102 |
| VRAM free with everything loaded | 527-577 MiB | 541-561 MiB | 643-644 MiB |
| Accepted of drafted, after the file | 57-70% | 56-62% | 81-94% |
| SSD wait per token, after the file | 15.5-17.7 ms | 15.1-17.2 ms | 9.5-10.0 ms |
| SSD blobs per token, after the file | 104-117 | 106-111 | 66-71 |
| Tokens per step, thinking on | 1.7-2.3 | 1.7-2.4 | 1.6-2.0 |
| Tokens per step, thinking off | 3.1-3.3 | 3.1-3.4 | 3.5-3.7 |

- v0.2.10 alone gives nothing measurable. Its MTP arm is inside the installed arm's range or close to it on everything, and the rewrite difference is within noise at 2 runs.
- DFlash2 ranges do not overlap the MTP ranges on thinking off, after a file and follow-up. Best single answers 68.3 and 66.0 tok/s, thinking off.
- Thinking on, the work the coding prompts mostly do, does not change. DFlash2 makes smaller windows there (1.6-2.0 tokens per step) with fewer wrong drafts (accepted 79-87% of drafted, MTP 56-72%), and the two cancel.
- Most likely reason for the after-file gain: MTP's rejected drafts still pull their experts in, so fewer wrong drafts means fewer SSD reads. The table fits that, but it was not tested on its own.
- Answers: 0 damaged in all 6 runs. Rewrites had no copy errors in any arm; the only changed source line is the last one, cut off by the 700-token limit. Answers were only checked by these rules and by reading the openings of a few; sampled answers differ per run, so there is no identical-output comparison.
- The engine warns on this card: "on a 12 GB NVIDIA card the MTP drafter was faster in most measurements". sm_89 is not in its tested list. On this workload it is not slower anywhere beyond noise.
- Not measured: a long real session. The release notes report about 0 gain in long agent sessions on another machine. Also not measured: paging and RAM use on DFlash2, and answer quality beyond the checks above. The daily config is unchanged.

## Smallest config beside Roblox Studio (2026-11-04, 2 runs, not daily)
`strata/strata-studio.json`, `run-strata-studio.bat`, `memtools/studio_bench.py`, `memtools/studio2`. Installed engine, Studio open with 3 windows. Changes from daily: `--max-context 65536`, `--resident-budget-gib 1`, no `--kv-resident`, `--pool-workers 4`, `--vram-reserve-mib 1024`. Thinking on, fresh engine per run, 1,264 cache slots.

| | Run 1 | Run 2 |
|---|---:|---:|
| Short context, generation, 4 answers | 18.4, 21.3, 19.6, 19.7 tok/s | 18.5, 22.3, 20.2, 18.6 tok/s |
| Short context, prefill (69-84 tokens) | 22-36 tok/s | 21-36 tok/s |
| Long context, prefill, 50,046 tokens | 886 tok/s (56 s) | 884 tok/s (57 s) |
| Long context, generation | 14.4 tok/s (177 tokens) | 15.1 tok/s (270 tokens) |
| Follow-up at long context, generation | 17.3 tok/s | 17.7 tok/s |
| Paged out | 0.1 GB/h | 0.0 GB/h |
| Paged in | 21.3 GB/h | 11.9 GB/h |
| Lowest available RAM | 2.12 GiB | 2.80 GiB |
| Engine working set, peak | 5.08 GiB | 5.09 GiB |
| Lowest free VRAM | 883 MiB | 838 MiB |
| Studio working set, start to end | 12.11 to 11.72 GiB | 11.73 to 9.64 GiB |

- **An earlier mixed run, with a play test starting during it:** 0.6 GB/h paged out, 121.5 GB/h paged in, lowest available RAM 0.24 GiB, 18.4-21.9 tok/s, one image at 10.3 tok/s with 0 MiB of VRAM free at its start (`memtools/studio1`). 0 damaged answers.
- **Windows of about 3.5 minutes:** too short to rely on the paging figures.
- **Studio's working set fell by about 2.5 GiB across the two runs.** Windows may have trimmed it or Studio may have released it; not traced. Whether Studio stuttered wasn't measured.
- **Not compared with daily at 50K:** no daily run has this prompt, and daily doesn't fit beside Studio.

## Each drafter's own time (2026-11-04, 2 runs each, order DFlash2, MTP, MTP, DFlash2)
`memtools/drafter_speed.py`, `memtools/drafter1`. v0.2.10 engine, the minimum Studio memory settings, thinking on, temperature 0, suffix drafting off. MTP `--spec 4 --spec-min-p 0.5`, DFlash2 `--spec 6 --draft-conf 0.8`. Draft time is the engine's `STRATA_DECODE_TIMING` figure per window.

| | DFlash2 | MTP |
|---|---:|---:|
| Short context, draft time | 2.54-2.57 ms | 2.15-2.56 ms |
| Short context, whole window | 81.5-86.0 ms | 104.4-120.2 ms |
| Short context, draft share of the window | 3.0-3.1% | 2.1% |
| 26K context, draft time | 2.73-2.80 ms | 2.66-3.14 ms |
| 26K context, whole window | 99.5-108.9 ms | 148.3-167.5 ms |
| 26K context, draft share of the window | 2.6-2.7% | 1.8-1.9% |
| Short context, tokens per window | 1.61-1.71 | 2.01-2.05 |
| Short context, acceptance | 88.7-92.8% | 73.5-74.9% |
| 26K context, acceptance | 83.6-87.8% | 68.5-73.7% |
| Short context, 4 answers, tok/s | 18.8, 21.0 | 16.8, 19.9 |
| 26K context, tok/s | 16.9, 18.7 | 13.9, 15.2 |

- **Finding:** the two drafters take about the same time, and drafting is about 2-3% of a window. Removing it entirely would gain little.
- **The speed rows aren't a fair drafter comparison:** Studio was open for some runs and closed by the end, not noted when, and cache slots moved from 1,059-1,066 on MTP to 1,375-1,638 on DFlash2. DFlash2's first run took 477 s against 152-183 s for the others; cause not checked.
- **The acceptance gap follows the gate:** DFlash2 offers about 0.7-0.8 drafts per window against about 1.4 for MTP. Reports on other cards at similar draft volume (Strata #5129, #3965) show about equal acceptance or MTP slightly ahead. Not tested here at equal volume.

## `STRATA_HC_Q8=1` and `STRATA_ROUTE_TAIL_SKIP=5` (2026-11-04, 2 runs each, mixed order, not adopted)
`memtools/switch_test.py`, `memtools/switch1`. Installed engine, daily config (10 GiB tier, saving off), Studio closed, thinking on, temperature 0.6, fresh engine per run. Order: normal, HC, tail, tail, normal, HC.

| | Normal | `STRATA_HC_Q8=1` | `STRATA_ROUTE_TAIL_SKIP=5` |
|---|---:|---:|---:|
| Short context, generation | 48.2, 44.6 tok/s | 49.3, 50.5 tok/s | **59.6, 59.7 tok/s** |
| Prefill, 50,046 tokens | 1,337, 1,376 tok/s | **1,501, 1,505 tok/s** | 1,366, 1,407 tok/s |
| Generation at 50K | 37.3, 30.7 tok/s | 42.4, 36.7 tok/s | **46.8, 48.7 tok/s** |
| Follow-up, generation | 37.9, 40.8 tok/s | 39.2, 39.0 tok/s | **48.1, 50.4 tok/s** |
| Tokens per step, short context | 1.92, 2.05 | 1.90, 2.22 | 2.01, 2.08 |
| Cache slots | 1,978 | 2,181 | 1,978 |
| Paged out | 17.67, 0.03 GB/h | 0.0, 0.0 GB/h | 0.0, 0.0 GB/h |
| Damaged answers | 0 | 0 | 0 |

- **Tail skip 5 is the biggest gain on this PC so far, about 28% at short context.** The ranges don't overlap normal's. It raises speed per step, not tokens per step. It zeroes more experts, so it changes answers.
- **HC_Q8:** frees about 390 MiB of VRAM as about 200 more slots; prefill about 10% faster; generation about 8% faster at short context, within noise at 50K and follow-up.
- **Quality not measured.** Neither switch can be judged until `tools/quant_quality.py` shows the perplexity cost. Not installed, not in setup.
- **Normal's first paging figure** covers under 2 minutes straight after other tests, so it shouldn't be read as this arm's rate.
- **`STRATA_ROUTE_TAIL_SKIP_FILE`:** not found in upstream. Added to this fork as an opt-in switch. It skips from the given rank only for experts that would be read from the file. Start-up check only, one 300-token answer each, not a benchmark: off 42.1 tok/s and 29,746 experts skipped, `=5` 44.2 tok/s and 39,552 skipped, so it fires. Speed and quality not measured.

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
