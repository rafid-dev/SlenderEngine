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

## AMD RX 6600: Qwen3.6-35B-A3B on 8 GB VRAM + 32 GB RAM (2026-11-07, branch `amd-rdna2-q36`)

Another PC: RX 6600 8 GB (gfx1032, HIP from the TheRock wheels, ROCm 10.2.0a20260930), 6 cores / 12 threads, 32 GB RAM,
Windows 11. Qwen3.6-35B-A3B UD-Q4_K_M. Server tests with `scripts/strata_bench.py` (the owner's tree): two "how to pass A
level" questions and "lalaland" (600 tokens each), a 24,167-token prompt (300 tokens), a follow-up; thinking on,
temperature 1.0, top-k 20, top-p 0.95; free RAM sampled every second. Engine tests with `--bench-pp` / `--bench-tg`.
llama.cpp on the same PC (Vulkan, its own hybrid expert cache, 262K context, KV in RAM): decode 25-28 tok/s, ~500 tok/s on
the 24K prompt, ~23 tok/s after it, 5+ GB free.

**Build:** `STRATA_HIP_ARCHS=gfx1032`, `STRATA_NO_VISION=1`, `tools\hipuild_windows.bat tests`: 546 targets. 219 of 223
unit tests pass (AMD_HIP.md, gfx1032 on Windows). The antivirus blocked one test binary until `build-hip-win` was excluded.

**Start (engine 0.1.57 as merged):** `--max-context 32768` refused (0.42 GiB free, 1.0 GiB needed). At 8K: 3.78 GiB of
weights in VRAM, 2 of 256 experts per layer, decode 17.0-17.9 tok/s, a 4.4K prompt at 110 tok/s, the engine at 24 GB of
RAM (a private 18.7 GiB copy of the expert blobs) and 485 MB free.

**Steps, each measured on the server test unless it says otherwise:**

| Step | Result |
|---|---|
| `STRATA_Q8_PACKED=0` (env) | 2 -> 19 slots per layer at 8K; decode 18.3 / 22.2 / 18.2; free RAM 2 MB at the low point |
| Expert blob cache (`<model>.q36blobs`, mapped) | written in 51.6 s once; free RAM 3.7 GB (from 2 MB); decode 17.1 / 20.8 / 17.5 |
| `--pool-workers 11` / `8` | 11: decode 3.7-6.2 (the pool starves the GPU's thread); 8: 18.5 / 22.4 / 18.5, kept |
| f16 KV, 32K context | runs (13 slots per layer); decode 18.2 / 21.4 / 17.6; 24K prompt 94 tok/s; 13.8 after it |
| Prompt chunk 2048 / 4096 | 2048: 219 tok/s, 4 slots, decode 15.5-16.9; 4096: did not fit |
| Engine profile, pp4096, chunk 2048 | 7.4 s per chunk, 4.56 s of it inside `stream_kick` on the host (pageable copies, ~3.9 GB/s) |
| Copier thread + pinned ring | pp4096 275 -> 531 tok/s (engine); greedy text identical with `STRATA_Q36_RING=0` |
| Ring: 6 threads / 32 MiB / 64 MiB chunks / 2 DMA streams | 513 / 511 / 511 / 467 tok/s against 515: not kept |
| Sampled speculative decoding | decode 20.8 / 23.4 / 23.3 (from 18.2 / 21.4 / 17.6); 64% of drafts accepted; 24K prompt 135 tok/s |
| Swap budget 80 / 160 per 32 tokens (one budget for all layers) | 23.5 / 26.1 / 23.9 and 23.2 / 25.2 / 23.7: the old budget of 8 reached the first layers only |
| Per-layer rebalance (2 each) + 492 extra slots in staging | 28.6 / 27.5 / 26.3; after the long prompt 16.7; follow-up 28.0 |
| Coalesced K reads in decode attention | after the 24K prompt 18.5 -> 22.5 tok/s (chunk 1024) |
| Token embeddings in pinned host memory | chunk 2048 with 10 main slots (was 4): decode 27.1-28.6, 24K prompt 363, 20.6 after |
| Stage profile, pp4096 chunk 4096 (`STRATA_Q36_PROF=2`) | delta-net layers 3.74 s, attention layers 1.83 s, experts 1.29 s, shared 0.56 s, router 0.11 s of 7.55 s |
| `STRATA_HIP_PROMPT_F16=1` | 532 against 534 tok/s: not the cause |
| `STRATA_DENSE_MMQ=1` | pp4096 533 -> 917 tok/s (engine); server 24K prompt 534 at chunk 4096, 417 at 2048 |
| Shared mixer / MMQ buffers | identical greedy text (placement pinned); chunk 4096 fits beside the slots |
| Chunk-sized buffers after the slots (to hold extra slots) | every prompt kernel ~2.4x slower (pp4096 298 against 718-972): likely in system memory. Allocated before the slots instead |
| Final, chunk 2048, defaults only | decode 26.0 / 26.9 / 26.7, 24K prompt 420, 21.1 after it, follow-up 25.3, lowest free RAM 3.5 GB |
| Final, chunk 4096 | decode 23.6 / 24.8 / 24.4, 24K prompt 539, 18.0 after it |

**Exactness:** greedy text bit-identical on/off for the ring, the shared buffers and the two buffer sets, with the
expert placement pinned (`--expert-slots 4`, `STRATA_Q36_FIXED_SLOTS=1`, `STRATA_Q36_XSLOTS=0`); unpinned the slot counts
differ and the GPU / CPU split changes the rounding. Teacher-forced KL, 3,000 prompt tokens + 512 forced
(`tools/qwen35/q36_config_kl.py`), reference fp32 KV + FP16 GEMMs at chunk 2048: chunk 1024 (the floor) 0.0098 mean,
0.089 p99, 96.3% same top-1; f16 KV 0.0144 / 0.086 / 96.9%; dense MMQ 0.0129 / 0.085 / 96.1%; both 0.0081 / 0.100 /
95.7%. The single changes' means come from one near-tie each (largest KL 4.5 and 2.8 of 513 steps). The sampled
speculative path draws once per emitted token in the same order as without drafts.

**Open:** context beyond 32K (KV in RAM with a VRAM window, or q8 KV); the server shows the first token about 2 s after the
engine has it (15-token prompts: engine 0.4 s, client 2.5 s); chunk 4096 together with chunk 2048's decode slots.

## Read-through before a default: the three checks (agreed 2026-11-05, before the fair comparison below)

All on `strata\build`, Windows 26H2, model card sampling, thinking on, fresh engine per run, read-through off against on.

1. **26H2 re-check, 10 GiB:** `switch_test.py th26 bRRbbR card`. Compare with +5.2% and +6.0% on the old Windows.
2. **14 GiB:** `switch_test.py th14 kKKkkK card`, checkpoints off, paging per request, then one verify run, arm `M`.
3. **Long real session, 10 GiB:** `real_session.py`, 25 minutes, one run each way, on then off. Target 0.3 GB/h paged.

Then the fair comparison below.

**Check 1, 26H2 at 10 GiB (`th26`), 3 runs each, order off, on, on, off, off, on, 450 tokens:**

| | Off | On | Change |
|---|---:|---:|---:|
| Whole run | 42.4 (42.0-42.7) | 47.0 (46.4-47.3) | **+10.9%** |
| Coding answers | 44.6 (43.5-45.1) | 49.8 (49.4-50.1) | **+11.6%** |
| After 50K | 37.1 (35.8-38.9) | 40.9 (38.5-45.5) | +10.1%, overlap |
| Follow-up | 39.6 (37.5-41.2) | 42.5 (40.6-44.5) | +7.1%, overlap |
| 50K prompt | 1,405 | 1,400 | flat |
| Tokens per window | 1.88 | 1.93 | overlap |
| Blobs read | 108,604 | 89,362 | -17.7% |
| Waited on reads | 22.0 s | 16.9 s | -23.5% |
| Lowest free RAM | 0.3-1.95 GiB | 1.57-2.39 GiB | |
| Paged out in the run | 0-0.01 GB | 0.3-0.87 GB | **see below** |

- **Active:** 65,536+ landings in all 3 on runs, tier page-locked on 26H2.
- **Larger than on the old Windows (+5.2%, +6.0%),** mostly because off fell: about 45 to 42.4, on 48.2 to 47.0. Can't be pinned on 26H2: this batch started with 19.7 GiB available, against 22.3-22.8 in the morning. By 13:05 it was back to 22.5 with Firefox still open, so Firefox isn't the cause. The batch ran 20-30 minutes after the reboot, and what held the extra 3 GiB wasn't established. Rerun at settled RAM: `th26b`.
- **Paging is new:** on the old Windows read-through paged 0-0.13 GB/h with 3.6-3.9 GiB free. Here each on run paged 0.3-0.84 GB, nearly all in its first coding answer, and off paged none. In 100 s runs that reads as 11-30 GB/h, but it's one burst. Why read-through bursts when RAM is lower isn't known. `th26b` and the long session decide this one.
- **Task Manager's rows don't add up to its total, and that's expected.** 14 GiB run, 13:08: 30.2 GiB in use, rows about 22.1 GiB (private working sets). Loading the engine took available RAM from 23.28 to 1.62 GiB, so 21.7 GiB, while its row showed 17.6: about 4.1 GiB of it is outside its working set, which fits the logged 3.09 GiB of pinned K/V and the earlier ledger's 3.0-3.7. Kernel pools add 1.2 GiB, and the remaining 2.8 GiB or so fits shared pages (shared working sets count 5.7 GiB, double counted), not separately measured. Before and after the engine, available RAM was the same, so it isn't a leak.

**Check 2, 14 GiB (`th14`), 3 runs each, order off, on, on, off, off, on. Read-through loses clearly:**

| | Off | On | Change |
|---|---:|---:|---:|
| Whole run | 49.5 (49.0-50.0) | 40.3 (39.2-41.5) | **-18.6%** |
| Coding answers | 52.7 (51.3-53.5) | 42.6 (41.8-44.3) | **-19.1%** |
| After 50K | 42.6 (40.6-45.0) | 37.9 (36.4-39.2) | **-11.0%** |
| Follow-up | 45.2 (44.7-45.5) | 35.1 (33.8-36.5) | **-22.3%** |
| 50K prompt | 1,465 | 1,455 | flat |
| Tokens per window | 1.95 | 1.79 | -7.9%, overlap |
| Blobs read | 64,673 | 54,225 | -16.2% |
| Waited on reads | 12.5 s | 24.4 s | **+95%** |
| Lowest free RAM | 0.32-0.91 GiB | 0.32-0.79 GiB | |
| Paged out in the run | 0.03-1.03 GB | 0.63-0.96 GB | same, nearly all at load |

- **Active:** 32,768+ landings in every on run. Paging doesn't separate the arms.
- **Fewer reads, but each takes about 1.8x as long, and very steadily.** Time reading per blob, from the logs' R, per run:

| | Off | On |
|---|---:|---:|
| 10 GiB | 276, 282, 280 µs | 284, 290, 290 µs |
| 14 GiB | 336, 322, 310 µs | 577, 598, 610 µs |

- **Correction:** the 4 GiB outside the registered part isn't pageable, as written earlier. The log says "locked 4096 MiB via working-set minimum + VirtualLock", so Windows can't trim the tier. At 10 GiB the engine gets the tier from the driver (`cudaHostAlloc`, "page-locked and mapped"). At 14 GiB the driver refuses, so it's ordinary memory with 10 GiB registered afterwards: the engine tries 14 and 12, and 10 succeeds.
- **Two explanations, under test with a 2 x 2:** memory pressure (the user's view), or a read landing in ordinary memory instead of driver-allocated memory, a cost on every read.

| | Tier from the driver | Tier in ordinary memory |
|---|---|---|
| RAM to spare | `th26b` | `th10u`, `STRATA_RESIDENT_PIN=0` |
| Low free RAM, `ballast.py` holding 4 GiB | `th10p` | `th10up`, `STRATA_RESIDENT_PIN=0` |

All at 10 GiB, so tier size doesn't change between cells (the user's point). The `th14c` rerun was stopped after one run. `memtools/blob_cost.py` reads the per-blob time.

**Stopped early so the merge could go ahead, with one cell started (`th10u`, one run each):** ordinary memory with RAM to spare, 257 µs per blob off, **555 µs on**. So the slowdown reproduces at 10 GiB with no memory pressure: landing in ordinary memory costs about 300 µs a read, against about 290 total in driver memory. One run each, and the low-RAM cells didn't run, so pressure may still add to it. That fits every run here. The laptop's unpinned tier still gained, by how much per read wasn't compared; its "waited" figure isn't the same measure as R.

**`th26b`, 10 GiB rerun with RAM settled, 3 runs each:** whole run 40.9 (38.4-42.7) to 46.4 (45.2-47.3), **+13.3%**; coding +11.3%, after 50K +17.5%, follow-up +13.1%, all with no overlap. Blobs read -22.9%, waited -27.4%. Per blob 276-296 µs off, 291-297 on. Lowest free RAM 3.4-3.9 GiB. **Paged out 0 in all 3 on runs,** so the `th26` bursts came from that batch's low RAM, not from read-through itself.

- **The RAM sum (user's question, 13:15, engine off between runs):** 22.25 GiB available of 31.76, so 9.5 GiB in use before the engine loads. Firefox 2.36, Claude 1.41, Spotify 1.11 and smaller apps come to about 5.8 GiB of working set (some shared and counted twice), which leaves about 3.7 GiB for Windows, about what the user expected. 22 + 9.5 = 31.5, leaving about 0.3 GiB, which matches the 0.3-0.9 GiB measured. No missing memory.
- **At 10 GiB the whole tier page-locks, and read-through wins.** 10 GiB with read-through (47.0, `th26`) comes close to 14 GiB without it (49.5), for about 4 GiB less RAM. Different batches and RAM conditions, so rough.
- **Default:** read-through can't be on everywhere. Setup should only write it at 10 GiB or below, or the engine should turn it off when the tier isn't fully page-locked. Other tier sizes aren't measured.

**Pushed to `speed-fork` at `450dd436` on 2026-11-05, by the user's OK.** Installed the same day with the user's OK: `strata\engine` is this build (old one in `strata\engine-before-merge`), and `run-strata.bat` sets `STRATA_RAM_THROUGH=1` (old one in `run-strata.before-merge.bat`). The real launcher started, read-through active, tier page-locked, one answer fine. Left: close PR #1 on GitHub, the long sessions, the fair comparison on this engine, the README's same-settings table, raw data, the re-rendered video and GIF, and the clean-clone `FAST-SETUP.bat` test.

**Merge first (the user's call, 2026-11-05):** read-through at 10 GiB has enough evidence: +5.2%, +6.0%, +10.9%, +13.3% over four batches here, +30% and +15% on short prompts on the laptop, byte checks clean on both PCs, identical answers here. Setup turns it on at 10 GiB or below, off above. Long sessions, the fair comparison, README numbers and the video come after the merge. Before it: the `iq_avx512.cpp` fix, then an all-off answer check and unit tests. Local branch `merge-pr1`.

**Finish plan (agreed by the user 2026-11-05):** the three checks, decide the switches, merge on a new branch, all-off check, setup, then the fair comparison on the merged engine so the README describes what ships, then docs, data and video, then ask before pushing. Engine defaults stay off; `FAST-SETUP.bat` and the daily launcher turn the winners on. FORK.md's "`STRATA_DIRECT_AHEAD` about 18% slower" is out of date: K=5 measured +2.7% here and more on the laptop.

## Fair comparison (asked by the user 2026-11-03, never run until 2026-11-05)

The user asked for this on 2026-11-03 after watching the first video: "compare with the same literal config on both base engine and our fork (spec 4, worker 7)", then stock's best, then overall gain from stock's best to fork's best. Agreed the same day, with "assuming someone will critique us" as the bar. It was never written here, its scripts were lost with an old scratchpad, and so the 2026-11-05 README runs below missed it. Those compare each engine at its own settings.

**The three claims it should support:** same settings on both engines (the code's gain), best we tried on stock against the fork's best, and the second SSD alone. Say "best we tried", never "stock's best".

**Design:** fresh engine per run, mixed order, one SSD on both except the two-SSD row. 3 runs each instead of the agreed 5 and 3, by the user's 2026-11-05 call on run count. 10 GiB and checkpoints off on every arm, as setup now installs, instead of 2026-11-03's 14 GiB. Both engines get every switch they share. Fork arms also keep the switches only the fork has, since those are its code.

| Arm | Engine | `--spec` | Workers | SSDs | Answers |
|---|---|---:|---:|---:|---|
| a | Stock | 3 | 13 | 1 | Same settings, test 1 |
| b | Fork | 3 | 13 | 1 | Same settings, test 1 |
| c | Stock | 4 | 7, P-cores | 1 | Same settings, test 2 |
| d | Fork | 4 | 7, P-cores | 1 | Same settings, test 2 |
| e | Fork | 4 | 7, P-cores | 2 | The second SSD alone, and the fork's best |
| f | Stock | 4 | 13 | 1 | Is there a better stock setting |
| g | Stock | 3 | 7, P-cores | 1 | Is there a better stock setting |

**Windows changed before these runs:** the user updated to 26H2 on 2026-11-05, after every result above this section's date. Compare these runs among themselves, not with `readme1`, `readme2` or older numbers. Before running, check the update's background work has settled, and that the driver, "Lock pages in memory", the page file and `C:\StrataMirror` are as before.

**Order agreed by the user 2026-11-05:** read-through-on long session finishes, then this comparison (21 runs in `memtools/fair2`, read-through added to the fork arms because setup ships it at 10 GiB), then the read-through-off long session. The chain was stopped between the two long sessions for this, so they are not back to back. The user dropped the two-terminal race the same day, so no new recording or video; the 2026-11-03 video stays in `demo\compare` but its numbers are out of date.

**Read-through-on long session (`memtools/sess_on`, 2026-11-05, 26H2, merged engine, 10 GiB, Edge and Spotify open, Firefox closed):** 25 minutes, 17 requests, context up to 143,970 tokens, 55,549 tokens generated, 46.0 tok/s, 3 rereads of the long context taking 139 s, no failures. Paged out 1.21 GB (2.9 GB/h, counted from engine start), 3 bursts of at least 20 MB/s, lowest free RAM 2.36 GiB, median 3.11. Over the 0.3 GB/h target. Needs the read-through-off session, same apps open, before blaming read-through. By time (`memtools/paging_view.py`): 828 MB in the first 3 minutes, 344 MB at minutes 12-15, 46 MB in the other 19 minutes. Pages read back in weren't recorded; `real_session.py` records them from the off session on, and `memtools/paging_whole_1434.csv` logs reads, writes and faults machine-wide from fair comparison run 5 on.

**Read-through-off long session (`memtools/sess_off`, 15:24-15:49, same engine and tier, one session each, not back to back; Chrome open and Spotify closed by then, so apps weren't the same as `sess_on`):**

| | On | Off |
|---|---:|---:|
| Overall tok/s | 46.0 | 43.1 |
| Context up to 50K | 46.7 | 46.8 |
| 50-100K | 45.4 | 43.4 |
| 100-150K | 45.7 | 43.1 |
| 150K and above | not reached | 39.2 |
| Largest context | 143,970 | 199,179 |
| Rereads of long context | 3, 139 s | 0 |
| Written to page file | 1.21 GB, 2.9 GB/h, 3 bursts | 0.62 GB, 1.5 GB/h, 1 burst |
| Lowest free RAM, median | 2.36, 3.11 GiB | 1.77, 3.43 GiB |
| Failures | 0 | 0 |

- **The overall figures aren't a fair race:** off reached higher context, where it ran slower. Compare the rows up to 150K: read-through's edge is about 0 at short context and about 5-6% from 50K to 150K, on different text.
- **Paging isn't only read-through.** Off also wrote 573 MB in one burst, at minutes 15-18, more than any one 3-minute block of on's later bursts, and went over the 0.3 GB/h target too. On wrote twice as much in this pair, but one session each with different apps can't settle the cause.
- **Pages read back, off:** 3.13 GB, which includes mapped file reads.

**Results (`memtools/fair2`, 2026-11-05 14:20-15:24, 26H2, merged engine, 21 runs, 0 failed, 0 damaged). Mean (low-high) of 3 runs, tok/s:**

| Arm | Text | After 26K file | Follow-up | Image | Prompt read | ms/step | tok/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| a Stock, spec 3, 13 workers | 23.6 (23.2-24.2) | 17.8 | 17.4 | 14.5 | 931 | 75.4 | 1.7 |
| b Fork, same | **41.4** (40.4-42.5) | 35.6 | 34.6 | 31.8 | 948 | 41.6 | 1.7 |
| c Stock, spec 4, 7 workers | 24.9 (23.2-26.1) | 15.6 | 18.5 | 16.1 | 938 | 78.2 | 1.9 |
| d Fork, same | **41.8** (39.8-42.8) | 33.8 | 32.0 | 30.5 | 932 | 43.7 | 1.8 |
| e Fork, same, two SSDs | **48.9** (48.1-49.8) | 43.3 | 42.2 | 38.6 | 1,410 | 40.4 | 2.0 |
| f Stock, spec 4, 13 workers | 22.3 (21.2-23.3) | 18.3 | 17.8 | 14.0 | 928 | 81.1 | 1.8 |
| g Stock, spec 3, 7 workers | 24.2 (23.5-24.7) | 16.9 | 18.0 | 15.5 | 925 | 73.8 | 1.8 |

- **Same settings, one SSD:** text +75% at a/b, +68% at c/d. After the file +100%, +117%. Prompt reading +2%, -1%, so no prompt gain from the fork's code on one SSD.
- **Second SSD alone, d to e:** text +17%, after the file +28%, follow-up +32%, image +27%, prompt reading +51%.
- **The speed gap is time per step, not tokens per step.** Stock 74-81 ms, fork 40-44 ms, with tokens per step about equal.
- **Not yet a claim: stock's best.** All stock arms here ran 22-25 tok/s, below stock's own 30.0 in `readme2` (10 GiB, its three switches, 13 workers, spec 3, old Windows). These arms also had `STRATA_COLD_OVERLAP` and `STRATA_PARTIAL_PIN`. Stock's overlap makes new threads per layer. So either these two switches hurt stock or 26H2 did; not tested. Stock at only its three switches on 26H2 is needed before any "best stock" number. A critic would call 24.9 a handicapped stock. Don't publish +96% (c to e). **Agreed with the user the same day:** after the skip test, stock with only its three switches, 3 runs each, order `WjjWWj`, `memtools/stock3`: arm W repeats the settings of the 30.0, arm j is spec 4 with 7 P-core workers.
- **Fork arms all include read-through,** so b and d aren't only shared settings: they include every fork-only switch.
- **Disturbance:** lowest free RAM fell to 1.06-2.17 GiB on runs ending 14:42-14:59, both engines, and returned to 3.1-3.5 after. That fits Chrome research during the batch, not confirmed. Order was mixed. The text ranges above are narrow anyway.
- **Paging, written per run:** stock 0.14 GB on average, fork 0.22 GB, bursts in both, and stock's runs last longer. Whole PC from run 5 on: 3.46 GB written in about 50 minutes. Pages Input was 95.66 GB, which also counts mapped file reads, so it isn't all page file.

**Stock with only its three memory switches (`memtools/stock3`, 16:04-16:22, same afternoon, 3 runs each, order `WjjWWj`, 0 failed, 0 damaged):**

| Arm | Text | After 26K file | Follow-up | Image | Prompt read | ms/step | tok/step |
|---|---:|---:|---:|---:|---:|---:|---:|
| W Stock, spec 3, 13 workers | 28.2 (26.9-29.9) | 20.8 | 22.0 | 15.2 | 941 | 64.6 | 1.8 |
| j Stock, spec 4, 7 P-core workers | **30.0** (29.1-31.5) | 22.9 | 23.4 | 15.2 | 945 | 64.9 | 1.9 |

- **So stock's best we tried is 30.0, arm j.** `STRATA_COLD_OVERLAP` and `STRATA_PARTIAL_PIN` cost stock about 17% at both settings (24.9 and 23.6 against 30.0 and 28.2), about 10-13 ms a step. Which of the two wasn't tested. 26H2 isn't the main cause: W matches the old 30.0 within noise.
- **Best against best:** fork one SSD 41.8, +39%; fork two SSDs 48.9, +63%. Not interleaved with these stock runs. The fork's skip-7 arm, 45.9, ended just before them, which would make it +53%.
- **Same-settings gains flatter the fork,** since those shared switches hurt stock. Lead with best against best.

**Also agreed 2026-11-03, still open:** one 20-minute mixed session on each engine, raw runs and scripts in the repo, README correction, `FAST-SETUP.bat` from a clean clone before any post. Answer quality with `--spec 4` isn't measured and the README must say so.

## README speed rows at model card settings (2026-11-05, same morning, not in the README yet)

**Each engine at its own settings, so the gains below mix code, flags and the second SSD: see "Fair comparison" above.**

`memtools/readme_bench.py`, `readme_report.py`, folders `readme1` and `readme2`. Fresh engine per run, mixed order, no sampling sent so the model card applies (temperature 1.0, top_p 0.95, top_k 20), thinking on in every row, Roblox closed. 450 tokens, image 150. Installed engine, no read-through. Vanilla is `engine-before-tg`, one SSD, with the switches earlier baselines used (`STRATA_UNBUFFERED_LOAD=1 STRATA_RESIDENT_HEADROOM_GIB=1.5 STRATA_RAM_ADAPT=1`).

| | Vanilla 14 GiB (3) | Fork 14 GiB (5) | Gain | Vanilla 10 GiB (3) | Fork 10 GiB (4) | Gain |
|---|---:|---:|---:|---:|---:|---:|
| Text | 35.7 (35.0-36.5) | 53.8 (50.4-56.2) | +51% | 30.0 (28.9-30.8) | 46.7 (44.3-48.0) | +56% |
| After a 26K file | 28.4 (25.2-30.4) | 44.5 (42.8-47.6) | +57% | 19.7 (19.4-20.0) | 38.1 (33.9-42.6) | +93% |
| Follow-up | 28.2 (26.7-29.8) | 47.4 (45.3-48.8) | +68% | 23.7 (22.6-24.4) | 40.2 (38.2-42.1) | +70% |
| Image | 22.0 (20.7-24.0) | 37.0 (30.3-44.3) | +68% | 16.5 (16.1-17.1) | 30.3 (25.8-32.4) | +83% |
| Prompt reading, 26,272 tokens | 1,017 | 1,484 | +46% | 970 | 1,432 | +48% |
| Tokens per step | 1.67 | 1.89 | | 1.77 | 1.78 | |
| ms per step | 47.7 | 35.4 | | 59.5 | 38.4 | |
| Lowest free RAM | 0.37-0.59 GiB | 0.08-0.94 GiB | | 3.38-4.12 GiB | 3.59-4.52 GiB | |
| Paged out, about 2 min windows | 1.6-20.4 GB/h | 0.8-36.2 GB/h | | 0-0.08 GB/h | 0 GB/h | |
| Damaged answers | 0 | 0 | | 0 | 0 | |

- **Gains hold:** +51% at 14 GiB against +52% in the table above, despite the different sampling and thinking on in every row. Absolute numbers drop a little: fork 57.3 to 53.8, vanilla 37.6 to 35.7. The image row isn't comparable with the old one: thinking on, so those 150 tokens are mostly thinking.
- **Same RAM is new:** the README says Strata hasn't been measured at 10 GiB. Fork against vanilla at 10 GiB is +56% on text.
- **What a person moving to the setup default sees:** vanilla at 14 GiB against the fork at 10 GiB, +31% on text, with about 3.5 GiB more free RAM and no paging.
- **14 GiB is tight:** lowest free RAM 0.08-0.94 GiB and up to 36 GB/h, well over the target, so 10 GiB remains the setting to recommend.
- **Run count:** cut to 3 vanilla runs per setting by the user. The gap is well outside the ranges on every row; the image and after-file ranges are wide.
- **Discarded first attempt:** vanilla without those three switches ran 26.4 tok/s (25.8-27.2, 3 runs) at 14 GiB, its tier cut to 13.22 GiB by the 4 GiB default headroom, and 21.5 (21.1-21.8, 2 runs) at 10 GiB. Left out of the table as it doesn't match earlier baselines. It may be closer to what stock Strata gives without tuning, which is worth a note if that's ever claimed.
- **Fork runs from both folders:** 2 at 10 GiB and 3 at 14 GiB came from the first attempt, about 30 minutes before the vanilla runs; the rest are interleaved with them.

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

## `STRATA_ROUTE_TAIL_SKIP_FILE=5` (2026-11-04, 2 runs each, order file, normal, normal, file, not adopted)
`memtools/switch_test.py`, `memtools/switch2`. The engine in `strata/build` for both arms, same settings and workload as the section above. Tail skip 5 is repeated from that section: an earlier session on the installed engine, so that column is a rough comparison.

| | Normal | File-only, rank 5 | Tail skip 5, all misses |
|---|---:|---:|---:|
| Short context, generation | 48.8, 46.3 tok/s | **53.3, 53.4 tok/s** | 59.6, 59.7 tok/s |
| Prefill, 50,046 tokens | 1,350, 1,332 tok/s | 1,354, 1,340 tok/s | 1,366, 1,407 tok/s |
| Generation at 50K | 36.0, 32.3 tok/s | **37.8, 45.1 tok/s** | 46.8, 48.7 tok/s |
| Follow-up, generation | 36.9, 37.5 tok/s | **44.4, 46.1 tok/s** | 48.1, 50.4 tok/s |
| Experts skipped in the run | 234K-235K | 264K-275K | 373K-396K |
| Short context, SSD wait per token | 7.3 ms | 4.6 ms | 4.7 ms |
| Short context, everything else per token | 14.0 ms | 14.3 ms | 12.2 ms |
| Paged out | 0.0, 0.02 GB/h | 0.0, 0.0 GB/h | 0.0, 0.0 GB/h |
| Lowest available RAM | 2.29, 2.49 GiB | 2.15, 2.08 GiB | 3.28, 2.96 GiB |
| Damaged answers | 0 | 0 | 0 |

- **Gain against normal:** about +12% at short context, +21% at 50K, +22% on the follow-up. Tail skip 5 gains about +25%, +40% and +32% against the same normal runs. The ranges don't overlap normal's at short context or on the follow-up; at 50K they nearly touch, with only 204 and 323 tokens generated.
- **Efficiency:** about 35K more skips than normal gave about half of what about 150K more gave under tail skip 5. Skip counts are rough because answer lengths differed.
- **Mechanism:** both cut SSD wait by about 37%. Tail skip 5 also saves about 2 ms of CPU-side time, which file-only doesn't, because RAM-tier experts are still computed.
- **Quality not measured.** Fewer skips suggest less change to answers, but that isn't shown. Which experts count as file reads also changes as the tier adapts.

## CPU/GPU expert assignment model (2026-11-04, offline, diagnostic only, no scheduler built)
`memtools/assign_sim.py`, `memtools/layer_cal`. `STRATA_LAYER_TRACE` on the `strata/build` engine, daily config, model card sampling, Studio and Roblox closed, 4 separate engine starts. Fitted on one run, with the PCIe cost also from the `--pcie-frac 1` run.

| Check | Windows | Total decode, model / actual | Error | Period per layer, absolute p50 / p90 / p99 |
|---|---:|---:|---:|---:|
| Training run | 1,331 | 63.5 / 66.5 s | -4.5% | 74 / 390 / 2,323 us |
| **Held-out run** | 1,459 | 64.5 / 65.4 s | **-1.4%** | 70 / 375 / 1,885 us |
| `--pcie-frac 0`, not used in fitting | 1,358 | 64.6 / 67.0 s | -3.7% | 77 / 406 / 2,489 us |
| `--pcie-frac 1`, PCIe cost only | 1,341 | 82.2 / 86.9 s | -5.4% | 80 / 430 / 1,681 us |

- **Fitted costs:** PCIe 147 us per expert, against 87 us for 2 MiB at the probed 24.1 GB/s. CPU 75.5 us per job plus 28 us per entry across 8 threads. Reads 135 us plus 199 us per file expert (held-out p50 error 136 us of a 739 us mean). VRAM expert 6.4 us.
- **Counterfactual:** `--pcie-frac 1` against 0, time per window 1.29x in the model, 1.31x actual.
- **Weak spots:** a small bias toward too fast on every run, and a slow tail the model doesn't explain. Reads are treated as landing together, because the read trace wasn't run.

Conditional results on the held-out run, within this model, with routing, VRAM contents, skips and draft sizes as recorded. Not hardware limits.

| Scenario | Modeled speed | Experts per window over PCIe |
|---|---:|---:|
| Best assignment, RAM tier only | +0.8% | 0.6 |
| Same, PCIe at 83 us | +0.9% | 3.6 |
| Same, PCIe at half the fitted cost | +1.0% | 5.8 |
| Best assignment, SSD experts allowed after reading | +1.2% | 7.8 |
| Current assignment, no read/compute barrier | +0.8% | 0 |
| Best assignment, RAM and SSD, no barrier | +1.2% | 7.8 |
| Free PCIe, no barrier (not achievable) | +15.5% | 150.6 |

- **Decision:** a new CPU/GPU assignment scheduler isn't worth building on this evidence. Its modeled upper bound of about +1% is smaller than the model's own error and this PC's run-to-run noise.
- **Why:** PCIe costs more than a CPU job, and even free PCIe gains little because SSD reads and fixed GPU work stay.
- **Not modeled:** memory contention on the CPU, changes to what's in VRAM, and changes in routing.
- **Speed at model card sampling, normal settings:** short context 43.6, 43.9 tok/s; 50K 30.6, 33.8 tok/s; follow-up 37.3, 39.4 tok/s. `--pcie-frac 1`: 31.1, 24.5, 28.9. `--pcie-frac 0`: 47.9, 35.9, 36.8, one run with tokens per step 2.11 against 1.74-1.93, so not a clear gain.
- `STRATA_DIRECT_BATCH` and `STRATA_READ_TRACE` are in the test engine but haven't been tested.

## SSD critical path and cost-aware cache simulator (2026-11-04, offline plus 3 validation runs, no engine change)

`memtools\ssd_ceiling.py`, `memtools\cost_sim.py`, `memtools\read_probe.cpp`. IQ3_S, 10 GiB tier, 12 GB VRAM, 32 GB RAM, model card sampling, Roblox closed. No extra expert skipping: rank 7+ misses stay skipped as today.

**Which reads stall.** Window 44.8 ms: host side 27.7 (waited on reads 14.2), fixed GPU 10.6, after the last layer 6.6. 50% of layers wait on a read. Layers with 5 or more reads are 11.6% of layers and about half the wait.

**How well the simulator predicts real runs.** Fitted on one run, tested on separate engine starts.

| Run | Reads, simulated against real | VRAM share | Time per window |
|---|---:|---:|---:|
| 10 GiB, held out | -0.7% | 52.9% against 53.1% | -2.1% |
| 12 GiB, held out | -1.4% | 49.0% against 49.2% | -4.9% |
| 8 GiB, held out | -0.3% | 52.4% against 52.6% | -0.9% |

- Per-window read correlation 0.99 on all three.
- **It overstates gains.** 8 to 12 GiB: model +15%, real +10.6%. The real runs removed 36% of reads and about 10% of window time.
- **It misses the older moves test.** `STRATA_RAM_ADAPT_ANY` plus admit-every-read: reads -11% against the -30% measured then, speed -3.0% against -6.5%. That test ran on an older setup (14 GiB, temperature 0.6). `ANY` alone: model about +1%, real -2.8%. So move-heavy policies are worse than the model says.

**The latency one cached expert saves.** One avoided read saves 226 us on average: 376 us if it is alone in its layer, about 200 us otherwise. Under 1% of reading layers get nothing from it. Latency value and use count correlate 0.985, and a fixed tier ranked by latency equals one ranked by hits (52.1 ms a window both). So cost-aware ranking adds nothing here. The 96-swap loss was move cost, not bad choices. A RAM place saves about 10 us a window on average, about 4 us at the 10 GiB margin.

**Read time, measured apart (`read_probe`, 2 rounds, 2.2 MiB blobs, half from each drive).** Today's path on the E-core reader: 1 blob 415 us, 2 blobs 590, 3 blobs 765, 8 blobs 1,620, 16 blobs 3,000. The fitted engine costs give 450, 652, 854, 1,864, 3,480, so the probe backs the model.
- Each extra blob adds about 180 us at any batch size, about 12 GB/s, close to what the two drives can sustain. Mostly transfer, not latency.
- Serial `ReadFile` submission is about 90 us per blob on the E-core and 30 on a P-core, but it overlaps the transfer. It costs a fixed amount per batch, not per blob.
- IoRing with registered buffers: about -60 us a batch on the E-core, about -90 on a P-core sibling (which would take time from a worker). One submitter per drive: about -30 on the E-core, no extra on a P-core. Sleeping against spinning: about +20 us.

**Policies, held-out run, model only, VRAM rule as today:**

| Policy | ms a window | Speed | Reads |
|---|---:|---:|---:|
| Today | 43.9 | | |
| No RAM moves at all | 58.8 | -25.5% | +98% |
| Admit more (moves during rounds, routed 1, unused 2) | 44.7 | -1.8% | -17% |
| Read-through, no copy, per layer: LRU | 39.7 | +10.3% | -19% |
| Read-through: least used | 39.5 | +11.0% | -20% |
| Read-through: most latency saved | 39.9 | +10.0% | -18% |
| Read-through: second use only | 41.0 | +7.0% | -12% |
| Oracle read-through, next 16 / 64 / all windows | 37.0 / 34.1 / 33.6 | +18% / +29% / +31% | -35% / -52% / -54% |

Read-through figures assume today's 1.3 ms a window of RAM copies goes away. Without that refund, least used is +7.4%. Going by the real 8 to 12 GiB slope instead of the model, 20% fewer reads is about +5.6%.

**Ceilings against 41 tok/s whole-run at 1.74 tokens a window, conditional on the model:**

| Case | Speed | tok/s |
|---|---:|---:|
| IoRing on the reader's E-core | +3.3% | 43 |
| Read-through, least used | +7 to +11% | 44-46 |
| Read-through + IoRing | +15% | 47 |
| Oracle read-through, all future windows | +31% | 54 |
| No SSD wait at all, same VRAM | +64% | 68 |
| Every expert in VRAM (impossible on 12 GB) | +99% | 82 |

**Reading:** 70-80 tok/s isn't reachable through the SSD path at 1.74 tokens a window. A real policy gets about a quarter of what perfect foresight gets. Even zero SSD wait stops at 68. The target also needs more tokens per window or a smaller fixed GPU cost. The one lossless SSD candidate left is read-through, predicted +5.6% to +11%, and an earlier simulator's +8-9% was wrong, so it needs a real A/B.

## `STRATA_RAM_THROUGH=1` prototype (2026-11-04, test engine `strata\build`, not installed, not adopted yet)

A missed expert's read lands straight in the RAM place of that layer's least used expert not routed this layer, so admitting it needs no copy. It replaces the `STRATA_RAM_ADAPT` moves. Places are 4 KiB aligned and padded by the sectors around the blob, so a read can't touch a neighbour. That leaves 5,272 experts in 10 GiB instead of 5,302. Victims must have a count under 1.4: a VRAM round's incoming expert had at least 2 when chosen and counts decay by 0.7 a round, so a round in flight is never touched. This assumes `--adapt-decay` of at least 0.7.

**Speed, 10 GiB, model card settings, order normal, through, through, normal, normal, through:**

| | Normal (3 runs) | Read-through (3 runs) | Change |
|---|---:|---:|---:|
| Whole run | 42.8 (42.0-44.2) | 45.0 (44.7-45.5) | +5.2% |
| Short prompts | 46.2 (44.9-47.8) | 47.7 (47.3-48.1) | +3.4% |
| After a 50K prompt | 36.5 (35.7-37.6) | 39.2 (36.5-42.2) | +7.5% |
| Follow-up | 38.5 (37.2-39.2) | 40.7 (39.8-41.7) | +5.8% |
| Prefill | 1,370 | 1,368 | -0.2% |
| Experts read | 106,030 | 88,888 | -16.2% |
| Waited on reads | 21.6 s | 17.7 s | -18.2% |
| Lowest available RAM | 1.24 GiB | 0.82 GiB | one run at 0.32 |
| Paged out | 0.61 GB/h | 0.53 GB/h | |

- Tokens per step were also 6.4% higher in these runs (1.86 against 1.98), which read-through shouldn't change, so part of the gain may be luck of the answers.
- The result is at the bottom of the predicted +5.6% to +11%. The model's +7.4% without a copy refund was high. Reads fell less than the simulated 20%.

**Correctness:**
- **Bytes:** with swaps and PCIe on, the whole tier against the file 4 times, after up to 80,046 landings: 5,272 blobs each, 0 differ.
- **Answers, usual settings at temperature 0:** all 6 differed from normal. That comparison can't decide anything, because the CPU share, tail skip, PCIe and swaps make answers vary between runs.
- **Answers, the identical-output settings (`STRATA_IQ_MT_MIN=1`, `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_PREFILL_CPU_SHARE=0`, `--prompt-cache 0 --pcie-frac 0`, plus `--adapt-swaps 1 --adapt-every 1000000` so usage counts exist with VRAM fixed):** normal matched normal, read-through matched read-through, and read-through matched normal on all 6 prompts, up to 50K context. A third read-through run with the tier checked at each 2,000 landings gave 0 differ.
- **My first version of this check was empty:** `--adapt-swaps 0` leaves no usage counts, so nothing landed. It only showed the padded layout changes nothing.
- **Limit:** with no rounds there's no decay, so victims run out after about 6,000-8,000 landings per run. This also makes read-through worse than RAM moves in this special setup (424,085 experts read against 317,014). Answers weren't compared with swaps or PCIe on, where they vary anyway.

## PR #1 on this PC, and the test branch (2026-11-04)

- **Crash check:** on reading the PR, its file-scope tables in `iq_avx512.cpp` (built at start, in a file compiled with `/arch:AVX512`) looked able to crash a CPU without AVX-512. They didn't: the PR engine with every switch off started and ran 4 short prompts, a 50K prompt and a follow-up on the i7-14700KF, 0 damaged. This shows it for this one build. The compiler could still choose AVX-512 there in another build, so moving the tables behind the CPU check is still worth doing before a merge. Speed from that run doesn't count: a build was running beside it (lowest available RAM 0.03 GiB).
- **`pool_tasks_test` on this CPU:** 792 bitwise comparisons passed, 432 with `STRATA_POOL_FUSED=1`. The part needing AVX-512 was skipped.
- **Read-through's victim floor** now follows `--adapt-decay` (2 x decay, was a fixed 1.4), and read-through stays off on multi-GPU, peer, batch and pipelined paths.
- **Branch `ram-through-test`:** PR #1's 3 commits plus read-through (`cef19ce8`), pushed so the PR's author can test on an RTX 4060 8 GB laptop. PR #1 itself is not merged.

## `STRATA_RAM_THROUGH` on an RTX 4060 laptop (2026-11-04, measured and reported by PR #1's author, not reproduced here)

RTX 4060 Laptop 8 GB, Ryzen 7 8845HS, 32 GB, unequal drives (7.1 and 3.5 GB/s, share 0.25), branch `ram-through-test` at `cef19ce`, unpinned tier. Model card sampling with a fixed seed, 3 runs each, order A B B A A B, without read-ahead, file tail skip or `STRATA_HC_Q8`. A browser stayed open with its processes capped at 256 MB each.

| | 10 GiB off | 10 GiB on | Change | 16 GiB off | 16 GiB on | Change |
|---|---:|---:|---:|---:|---:|---:|
| Short prompts | 16.90 | 22.06 | +30.5% | 23.73 | 24.95 | +5.1% |
| After a 50K prompt | 15.77 | 18.43 | +16.9% | 22.47 | 24.47 | +8.9% |
| Follow-up | 18.73 | 18.63 | -0.5% | 25.93 | 26.20 | +1.0% |
| Prompt | 653 | 646 | -1.1% | 659 | 660 | +0.2% |
| Tokens per step | 1.86 | 1.93 | +3.5% | 1.87 | 1.88 | +0.4% |
| Experts read | 143,888 | 89,087 | -38% | 62,549 | 43,112 | -31% |
| Waited on reads | 43.0 s | 30.0 s | -30% | 17.4 s | 13.4 s | -23% |

- **Correctness:** an unpinned 16 GiB tier against the file, 5 checks up to 100,019 landings, 8,364 blobs each, all 0 differ. No verify run at 10 GiB.
- **Spread:** at 10 GiB every on run beat every off run on short prompts and after the long prompt. At 16 GiB the after-long-prompt ranges overlap, so he reads it as a likely gain of a few percent.
- **Memory:** at 16 GiB, 0.3-0.6 GB stayed available and paging ran 7-63 GB/h in both arms, so memory numbers there are noise. At 10 GiB about 4 GB stayed free.
- **Places:** 5,231 against 5,258 at 10 GiB, 8,364 against 8,406 at 16 GiB (-0.5%).
- **Not run:** with his usual read-ahead, file tail skip or `STRATA_HC_Q8`; a pinned tier; answer comparison; Linux.
- **Baseline, from the follow-up report:** `STRATA_RAM_ADAPT=1`, `_PAR=1` and `_BG=1` were on in both arms, and the off runs' logs show the moves happening. So a weak baseline of that kind doesn't explain the gain. Reads fell 38% there and 16% here at the same 10 GiB, and the short-prompt gain is about five times the +6% here. Off, that PC read 143,888 experts and waited 43 s in a run with one 50K prompt. Not comparable directly, but the 8 GB card has fewer VRAM slots, so more misses to save is the likely part (not tested).

**Follow-up: read-through with read-ahead on in both arms (`STRATA_DIRECT_AHEAD=2 STRATA_LOOKAHEAD_K=5`), same method, 2026-11-05, reported by the author:**

| | 10 GiB off | 10 GiB on | Change | 16 GiB off | 16 GiB on | Change |
|---|---:|---:|---:|---:|---:|---:|
| Short prompts | 19.73 | 22.60 | +14.5% | 25.23 | 26.48 | +4.9%, overlap |
| After a 50K prompt | 18.67 | 20.77 | +11.2%, ranges touch | 24.43 | 26.93 | +10.2%, overlap |
| Follow-up | 20.43 | 23.00 | +12.6% | 31.93 | 28.00 | **-12.3%, no overlap** |
| Prompt | 646 | 639 | -1.1% | 654 | 650 | -0.6% |
| Tokens per step | 1.94 | 1.94 | flat | 1.91 | 1.89 | flat |
| Experts read | 119,171 | 97,972 | -18% | 59,765 | 48,875 | -18% |
| Waited on reads | 25.2 s | 19.0 s | -25% | 11.2 s | 8.9 s | -20% |
| Lowest available | 1.95 GB | 2.42 GB | | 0.62 GB | 0.31 GB | |
| Paged out | 2.0 GB/h | 5.7 GB/h | | 2.0 GB/h | 37.8 GB/h | |

- **Correctness:** 10 GiB tier with read-ahead, 6 checks up to 120,079 landings, 5,231 blobs each, all 0 differ, no crash. No verify at 16 GiB with read-ahead.
- **Read-ahead takes part of the gain:** short prompts on 10 GiB are +30% without it and +15% with it. Across days, with other browser tabs, so rough: read-ahead lifts off from 16.90 to 19.73 and on only from 22.06 to 22.60. That matches this PC, where read-through with read-ahead (+4.9%) did no better than read-through alone (+6.0%).
- **The 16 GiB follow-up looked like a clear loss here, but didn't repeat (see below):** every off run (31.6-32.2) beat every on run (26.8-28.7). All 3 on runs paged 25-59 GB/h with 0.13-0.59 GB available, off 1.4-2.7 GB/h with 0.50-0.81 GB. Paging wasn't timed per request.
- **Not run in this batch:** browser closed (the owner kept it open), pinned tier, with file tail skip or `STRATA_HC_Q8`, more than 3 runs, answer comparison, Linux, power mode not recorded.

**Follow-up 3: 16 GiB, browser closed, paging timed per request, same binary, order A B B A A B, two sets back to back, 2026-11-05, reported by the author:**

| | Read-ahead on: off | on | Change | Read-ahead off: off | on | Change |
|---|---:|---:|---:|---:|---:|---:|
| Short prompts | 24.79 | 26.82 | +8.2%, no overlap | 22.44 | 25.44 | +13.4%, no overlap |
| After a 50K prompt | 22.87 | 25.87 | +13.1%, no overlap | 22.80 | 24.80 | +8.8%, one run overlaps |
| Follow-up | 29.90 | 29.03 | -2.9%, overlap | 26.87 | 26.70 | -0.6%, overlap |
| Prompt | 656 | 657 | flat | 650 | 649 | flat |
| Tokens per step | 1.96 | 1.90 | -3% | 1.91 | 1.89 | -1% |
| Experts read | 67,036 | 50,513 | -25% | 64,318 | 42,128 | -34% |
| Waited on reads | 12.7 s | 8.8 s | -31% | 17.5 s | 12.2 s | -30% |
| Lowest available | 1.33 GB | 1.45 GB | | 1.54 GB | 1.40 GB | |

- **The follow-up loss is mostly gone:** -12% with no overlap became -3% and -0.6%, both overlapping. Read-through's own follow-up barely moved (28.0 to 29.0). The off runs fell, 31.9 to 29.9, so that gap was mostly off running fast that day. Why isn't known.
- **Paging doesn't explain it:** none of the 3 read-ahead on follow-ups paged (0.00 GB, at least 2.2 GB available). The one on follow-up that paged wasn't the slowest, and an off follow-up paged too.
- **The browser mattered for RAM:** with it closed, at least 1.2 GB stayed available in every request, against 0.13-0.81 GB. Paging fell to 0-1.3 GB per run, mostly 0.
- **16 GiB gain is larger than the earlier "few percent":** +8-13% on short and after-long prompts with free RAM. It's now about the size of the 10 GiB read-ahead gain (+11-15%).
- **Read-ahead on top of read-through does add something on that PC, same session, not interleaved:** read-through arm short 25.44 to 26.82 (+5%), after-long 24.80 to 25.87 (+4%), follow-up 26.70 to 29.03 (+9%). The off arm gains more (+10%, flat, +11%). On this PC the pair (+4.9%) was no better than read-through alone (+6.0%), so this differs between the machines.
- **Default so far:** positive or flat on every measure in every batch on both PCs, except the 16 GiB follow-up that didn't repeat. Left: this PC at 14 GiB, a long real session for paging, and `readme2`-style runs on 26H2. Both PCs tested only at model card sampling for speed; the identical-answer check is from this PC alone.
- **Not run:** more than 3 runs per arm, answer comparison, with file tail skip or `STRATA_HC_Q8`, pinned tier, 10 GiB. Not controlled: whether Windows Update, BITS or Defender were working during the runs. 2 s samples can miss short paging bursts.

## Layer-major prompts (2026-11-05, test engine `strata\build`, not installed)

**First attempt was empty.** `STRATA_PREFILL_LAYER_MAJOR=1 STRATA_PREFILL_LM_TOKENS=32768 STRATA_PREFILL_LM_FORCE=1`, 2 runs each: every prompt logged "layer-major off for this prompt: no VRAM for a super-chunk (399-515 MiB free)", so the walk never ran and these are normal reads. Normal and "layer-major" tok/s: 31K 1,296 and 1,318, 50K 1,387 and 1,394, 228K 1,457 and 1,419. This says nothing about layer-major. The report also used `--prefill-head-alias`, which frees the output head's VRAM (521 MB here) during a prompt. The prompt plan here expects about 196 GB read from the drive per 32K-token prompt, which is the re-reading layer-major is meant to cut.

**With the head alias (`--short-read 0 --prefill-head-alias`, `STRATA_HEAD_FAST_LOAD=1`, `STRATA_HEAD_ALIAS_SHORT=64`), one run:** the walk ran on all three prompts, and it's slower:

| Prompt | Normal (2 runs, mean) | Layer-major with alias (1 run) | Change |
|---|---:|---:|---:|
| 31K tokens | 1,296 | 1,070 | -17% |
| 50K tokens | 1,387 | 1,171 | -16% |
| 228K tokens | 1,457 | 1,257 | -14% |

- **Why, as far as the log shows:** with 400-500 MiB of VRAM free, super-chunks fit only 8,192-12,288 tokens, against 32,768 asked for and normal chunks of 6,912, so most of the re-reading stays: 327.5 GB streamed for the 50K prompt. Head restore adds 134-154 ms and rebuilding the workspace 75-91 ms per fresh prompt.
- **Why it helped the report:** its baseline was 320-344 tok/s, on a different path, and its result of 743-847 tok/s is still well under normal here. Not compared directly.
- **Stopped here.** It fails the rule to go on only if it beats about 1,400 tok/s, so no KL, needle test or `STRATA_LM_CPU_ROWS` here. The answer may differ on a setup with more VRAM free.

## Read-through with read-ahead (2026-11-05, test engine `strata\build`, not installed)

**Bytes, both on, one run:** 3 checks up to 60,143 landings, 5,272 blobs each, 0 differ.

**Speed, 10 GiB, model card settings, 3 runs each, order normal, through, both, both, through, normal, normal, both, through:**

| | Normal | Read-through | Both |
|---|---:|---:|---:|
| Whole run | 41.5 (40.9-41.9) | 44.0 (42.2-45.0), +6.0% | 43.5 (41.2-45.0), +4.9% |
| Short prompts | 44.1 (43.9-44.5) | 47.0 (46.7-47.2), +6.7% | 45.8 (43.2-47.8), +3.8% |
| After a 50K prompt | 36.9 (34.3-38.2) | 39.3 (36.1-41.3), +6.5% | 37.8 (35.9-40.8), +2.6% |
| Follow-up | 37.6 (37.2-38.1) | 38.4 (34.4-40.9), +2.0% | 40.7 (38.3-42.2), +8.1% |
| Prefill | 1,396 | 1,389 | 1,399 |
| Tokens per window | 1.82 | 1.85 | 1.93 |
| ms per token | 24.1 | 22.8 | 23.0 |
| Experts read | 105,597 | 87,851, -16.8% | 97,660, -7.5% |
| Waited on reads | 21.1 s | 17.6 s, -16.5% | 17.4 s, -17.3% |
| Paged out | 0.0 GB/h | 1.7 GB/h (one run 5.0, the others about 0) | 0.0 GB/h |

- **Read-through repeated its gain:** +5.2% in the first batch, +6.0% here. Whole-run and short-prompt ranges don't overlap here, and tokens per window nearly match. That's 6 runs each across two batches.
- **Read-ahead adds nothing on top.** Wait is the same and ms per token no better, while it reads more. A read-ahead hit never lands in the tier, so the tier learns less: 97,660 experts read against 87,851, on top of about 89,000 blobs read ahead, 47,000 used. The follow-up's +8.1% goes with 4% more tokens per window, in one of four measures. Not worth the extra SSD work.
- **So:** read-through alone. No answer check for the pair, since it doesn't win.

## Upstream #4040, `--lookup-max-t` (2026-11-05, `strata-0210` test build, not in the fork, not installed)

MTP against MTP, only `--lookup-max-t 0` against 12 changed. 10 GiB, model card sampling, thinking on, 3 runs each, order off, on, on, off, off, on. Engaged: window sizes 1, 2, 3, 4, 5, 6, 12, and some on requests averaged 9.8 tokens per window against at most 5.8 off.

| | Coding off | Coding on | Change | Rewrite off | Rewrite on | Change |
|---|---:|---:|---:|---:|---:|---:|
| tok/s | 44.1 (43.2-45.2) | 43.0 (42.6-43.5) | -2.3% | 38.7 (37.5-39.9) | 40.3 (39.5-40.8) | +4.1% |
| Tokens per window | 1.83 | 1.84 | | 3.57 (3.45-3.66) | 5.10 (4.56-5.54) | +43% |
| ms per window | 41.6 | 42.7 | | 92.3 | 126.7 | +37% |
| ms per token | 22.7 | 23.3 | | 25.9 | 24.8 | -4.0% |
| Waited on reads, ms per token | 7.3 | 7.4 | | 13.2 | 14.1 | +6.6% |
| Experts read per token | 35.3 | 36.1 | | 74.2 (72.3-77.7) | 83.2 (81.8-85.4) | +12% |
| Read time per reading layer | 0.79 ms | 0.80 ms | | 1.51 ms | 1.77 ms | +17% |
| Draft acceptance | 63.2% | 61.7% | | 90.1% | 92.7% | |
| Suffix draft acceptance | 80.3% | 70.7% | | 98.0% | 99.1% | |
| CPU experts per layer-window | 5.2 | 5.4 | | 10.8 | 16.6 | +54% |
| VRAM hits per layer-window | 12.0 | 12.0 | | 16.3 | 20.3 | +25% |

Only tokens per window and expert demand separate cleanly; tok/s ranges overlap in both columns. VRAM free at start 482-519 MiB off, 458-480 on, lowest logged 298-369 off, 272-325 on, slots 1,975 against 1,974. Peak VRAM wasn't sampled. The "damaged" flag on every run, both arms, is the file's own runs of dashes and spaces, not damage.

**Reading:** longer windows work as designed on copy-heavy text, but demand rises with them, and in a rewrite SSD wait is over half of each token, so 43% more tokens per window gives about 4%. No gain on coding, where windows stay at 2.3 tokens. The feature is copy-only, and on this setup SSD wait cancels most of that. Not meaningful enough to port or stack with read-through.

**Answers, MTP against MTP at the identical-output settings (temperature 0, tail skip, CPU share and PCIe off, VRAM fixed), order off, on, on, off:** off matched off and on matched on, on all 6 prompts. Off against on differs on all 6, at the same places in both pairs (characters 299-815). So within this test the difference comes from the switch, not noise, and the PR's "bitwise" doesn't hold on this build at these settings. "`--spec 4` changes the wording" earlier was a similar effect of window shape. Whether either answer is worse wasn't measured, as no KL was run. At these settings rewrites ran 19.4 -> 23.7 tok/s (+22%) with SSD wait up, on text that differs, so it doesn't carry over to daily settings, which gave about +4%.

**Where a token's time goes, from this run:** coding 22.7 ms: 7.3 reads, 15.4 the rest. Rewrite with long windows 24.8 ms: 14.1 reads, 10.7 the rest. So tokens per window lowers the rest, and SSD wait grows back when text is cold.

## Remaining bottlenecks (2026-11-05, from the runs above, not a measurement of its own)

Target 70-80 tok/s is 12.5-14.3 ms a token. Coding today is about 22.7 ms a token at 1.83 tokens a window, 41.6 ms a window (v0.2.10 build, MTP):

| Cost | ms a window | ms a token | Scales with |
|---|---:|---:|---|
| Reads (SSD wait) | about 13 | 7.3 | Experts outside VRAM and RAM |
| CPU expert work | about 14 | 7.7 | Experts outside VRAM |
| GPU-reach wait | about 12.5 | 6.8 | The window, not hit rate |
| Draft | about 2.7 | 1.5 | The window |

- These are rounded and don't sum exactly. SSD wait comes from the cold-read lines, the others from decode timing.
- **Non-SSD cost fits about 13 ms a window plus 8 ms a token**, from two points of different text (coding 15.4 ms a token at 1.83 tokens a window, long-window rewrites 10.7 at 5.1). So 14 ms a token needs SSD wait of about 2 ms and about 3.4 tokens a window. At read-through's 6 ms, no window size reaches it. Rough.
- **Rewrites at 90% acceptance and 3.6 tokens a window aren't cheaper per token than coding at 63% and 1.83:** cold text doubles reads per token.
- **Modeled (`memtools/next_sim.py`, held-out run, real gains have run lower):** read-through about 46 tok/s; with +0.5, +1, +2 GiB of VRAM slots about 47, 48, 50. A swap budget of 96 is about +0.4% even with free swaps, though VRAM share rises to 56.9%. K/V isn't the VRAM to reclaim: 32,768 cells a layer stay in VRAM and most is in pinned RAM.
- **Realistic on coding, same hardware and quantization, no skipping: about 50-55 tok/s by these estimates.** 70-80 isn't in reach of what's been tested.

**Next experiment:** rerun `STRATA_VERIFY_PROFILE=1 STRATA_DECODE_TIMING=1` on today's setup, with no new code, and split the result by window size. Partly done before: "GPU profile" (2026-11-03, one run, older 14 GiB setup) found about 15.5 ms of real GPU work in a 39 ms step (dense matvec 3.2, hyper-connection 3.2, output projection 1.6, head 1.2, VRAM experts 2.2), and "Cache plan so far" measured 11-12 ms per extra draft row for rows 1-3. No single kernel dominates, so expect a few ms at most, not a route to 70-80. New here: today's setup, window sizes above 3, and the cost after the last layer.

## Narrow head-alias loan (2026-11-05, test engine `strata\build`, uncommitted, opt-in `STRATA_HEAD_ALIAS_NARROW=1`, not installed)

`--prefill-head-alias` lent from slot 1 on: 2,230 of 2,231 slots for a 2,048-token chunk that uses 1,413 MiB, and every lent expert is streamed and refilled. The narrow loan borrows only the tail slots the chunk's buffers need, as a loan without the alias does. Same chunk, same alias. `STRATA_LOAN_TRACE=1` counts the loan, `STRATA_LOAN_VERIFY=1` compares every slot marked resident with the file after each refill.

**Bytes before speed (`STRATA_LOAN_VERIFY=1`, arms A-D, 2 refills each):** 8 of 8 checks, 0 differ, up to 2,273 slots a check, kept and lent slots both. That covers bytes, not answers.

**Speed, 3 runs each, mixed order, read-through on in all arms, model card sampling, 64 tokens generated.** Streamed GB is blobs at about 2.05 MB each, or the log's MB for D:

| Arm | Chunk | Slots lent of total | 31K prompt tok/s | Seconds | Streamed GB | 50K prompt tok/s | Seconds | Streamed GB | Refill MiB, ms first / next |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Z normal, alias off | 6,912 | 1,663 of 1,978 | 1,262 (1,225-1,289) | 26.7 | 249 | 1,383 (1,378-1,389) | 37.9 | 398 | 3,269, 1,359 / 940 |
| A alias off, matched | 2,048 | 738 of 2,233 | 513 (509-518) | 62.4 | 737 | 533 (532-535) | 95.6 | 1,173 | 1,443, 691 / 436 |
| B alias, existing loan | 2,048 | 2,230 of 2,231 | 467 (464-471) | 68.3 | 787 | 486 (486-487) | 104.6 | 1,250 | 4,387, 1,569 / 1,294 |
| C alias, narrow | 2,048 | 738 of 2,231 | 512 (507-518) | 62.5 | 736 | 530 (521-534) | 96.2 | 1,172 | 1,442, 723 / 470 |
| D alias, narrow, layer-major | 2,048 | 738 of 2,231 | 1,547 (1,414-1,617) | 22.1 | 107 | 1,469 (1,429-1,519) | 35.7 | 226 | 1,442, 736 / 442 |

- **The narrow loan alone (B to C):** +9.6% and +9.1%, refill bytes -67%, refill 1,569 -> 723 ms and 1,294 -> 470 ms. It lands exactly on A, so the existing loan was the whole cost of the alias at this chunk.
- **A 2,048-token chunk is the real penalty here:** C is 2.6-2.5x slower than Z, streaming about 3x the bytes.
- **Layer-major needed the narrow loan's slots kept and the head's VRAM:** D against Z is +23% on 31K and +6% on 50K, in 4.6 s and 2.2 s less, with -57% and -43% bytes streamed. The earlier 14-17% loss had the old loan and `0 resident`.
- **D is two-mode at 31K:** 2 super-chunks of up to 16,384 (1,608 and 1,617 tok/s, 91.8 GB) or 3 of up to 12,288 (1,414, 136 GB), depending on VRAM free.
- **Slots at decode:** alias arms have 2,231, normal 1,978. A, with its reserve lowered to match, showed about 110 MiB of VRAM free with the head loaded, so alias arms should have about that at decode. Low: the engine warns under 1,024 MiB. Generation here is only 64 tokens, so decode isn't tested.
- **Paging:** D 53-60 GB/h and Z 3-49 GB/h during these 1-2 minute runs. Short windows, so rough, but well over the 0.3 GB/h target. Needs a long-run check.
- **Overlap:** disk scans on D: by another session ran 09:12:30-09:17:31, covering A run 2 and Z run 2 fully, and the end of B run 2 and the start of B run 3. Those numbers look like the others, and so did the reruns with nothing else on the disk (one more each: Z 1,309 and 1,402 tok/s, A 513 and 530, B 456 and 488).

**228K check and more short runs, order Z, D, D, Z, model card sampling, nothing else on the disk:**

| Prompt | Normal (Z) | D | Change | Streamed GB, Z to D |
|---|---:|---:|---:|---:|
| 31K, all runs (6 Z, 5 D) | 1,290 tok/s, 26.1 s | 1,581 tok/s, 21.4 s | -18% time | 249 to 92-136 |
| 50K, all runs (6 Z, 5 D) | 1,394 tok/s, 37.8 s | 1,488 tok/s, 35.3 s | -6.5% time | 398 to 226 |
| 228K (2 each) | 1,468 tok/s (1,451-1,486), 157.3 s | 1,563 (1,561-1,565), 147.8 s | -6% time | 1,674 to 869 |

- **Bytes fall far more than time:** at 228K, -48% bytes for -6% time. Above about 50K, reading the SSD isn't what limits the prompt.
- **Needle:** 6 of 6 found on both, 32K and 128K at depths 10, 50, 90.
- **KL of the first window at these settings, full vocabulary, 2 runs each.** Same top token in every pair:

| Prompt | Z vs Z | D vs D | Z vs D (2 pairs) |
|---|---:|---:|---:|
| 31K, first request | 0.00000 | 0.00000 | 0.00008, 0.00008 |
| 50K | 0.00569 | 0.00998 | 0.05798, 0.16063 |
| 228K | 0.05358 | 0.01513 | 0.02524, 0.06957 |

  The first request is close, but 50K sits about 10x over the noise between runs, with p(top) 0.943 against 0.991-0.999. One token per prompt is thin, and the same-arm numbers show state left by earlier requests matters.

**KL at the identical-output settings (`loan7`, order Z, D, D, Z), where each arm repeats itself exactly:**

| Prompt | Z vs Z | D vs D | Z vs D | p(top), Z / D |
|---|---:|---:|---:|---:|
| 31K, first request | 0.00000 | 0.00000 | 0.00007 | 0.999 / 0.999 |
| 50K | 0.00000 | 0.00000 | 0.02022 | 0.904 / 0.951 |
| 228K | 0.00000 | 0.00000 | 0.01181 | 0.646 / 0.572 |

So the gap is real and comes from the arm, not from run-to-run state. Same top token on all three. Prompt speed at these settings: 31K -17% time, 50K -5%, 228K -2.5% (1,480 against 1,517 tok/s, 156.3 against 152.4 s). Source comments put layer-major's difference down to rounding from expert sum order, but 0.02 at 50K is a lot larger than the 0.00007 at 31K, so that isn't shown yet. Unproven guess: chunk size or `--kv-resident 32768`, since only the prompts past 32,768 tokens show it. 

**Control (`loan8`, one run each, same settings, no layer-major), against normal:**

| Prompt | Chunk 2,048, alias off (A) | Chunk 2,048, narrow alias (C) | D, for comparison |
|---|---:|---:|---:|
| 31K | 0.00025 | 0.00031 | 0.00007 |
| 50K | 0.00227 | 0.00220 | 0.02022 |

- **Chunk size alone moves the logits a little,** and A and C agree, so the alias, the narrow loan and the extra slots add nothing measurable. The narrow loan on its own is clean.
- **Layer-major is about 9x that at 50K,** and smaller than it at 31K. So chunk size doesn't explain the gap. What changes between 31K and 50K isn't known: 2 super-chunks against 4, or crossing `--kv-resident 32768` are the candidates. Not cleared: layer-major stays off until this is explained.

**Decode and paging (`loan6`), `switch_test.py`, read-through against D, 3 runs each, order R, E, E, R, R, E, model card sampling, 450 tokens:**

| | Read-through | D | Change |
|---|---:|---:|---:|
| Coding answers | 50.5 (48.3-51.7) | 52.3 (51.8-53.1) | +3.5% |
| After 50K | 42.4 (40.7-45.1) | 44.1 (43.1-44.8) | +4.1%, overlap |
| Follow-up | 45.2 (42.9-46.6) | 43.9 (41.7-46.2) | -2.8%, overlap |
| Whole run | 48.2 (46.5-49.4) | 49.1 (48.6-49.9) | +1.9%, overlap |
| 50K prompt | 1,398 | 1,508 | +7.8% |
| Tokens per window | 1.80 | 1.87 | overlap |
| Blobs read | 83,114 | 80,098 | -3.6%, overlap |
| Lowest free RAM | 3.76 GiB | 3.79 GiB | |
| Paged out | 0.0-0.13 GB/h | 0.0-0.28 GB/h | under the target |
| Damaged answers | 0 | 0 | |

- **Paging:** the 53-60 GB/h above belongs to the prompt-only runs. Over these whole runs it's gone, 100 s windows, so a long session is still needed.
- **Two costs the prompt table doesn't show:**
  - **VRAM at decode:** 54-58 MiB free once the head is back, against 482 MiB on normal. The 253 extra slots are the likely source of the small decode gain, and the engine warns this is where the driver can start paging GPU memory, with decode falling to a few tok/s. It didn't happen here, with nothing else using the GPU. The image path isn't tested.
  - **Short prompts:** anything over 69 tokens goes the batched way: 78-84 tokens took 2.7-2.8 s against 1.8-2.1 s, and a follow-up of 260 new tokens took 2.9 s against 1.5 s for 117. About 0.4 s of that is the refill. That's paid on every agent turn and may cancel the long-prompt gain over a session. No real session has been run with this.
- **Verdict so far:** the narrow loan is a clean fix for anyone using the alias. D is faster on long prompts, but isn't ready to enable: KL not cleared, and both costs above open. Next, if KL clears: cap the cache so VRAM stays about 500 MiB free at decode, raise `STRATA_HEAD_ALIAS_SHORT` so small turns skip the swap, then a real agent session.

## PR #1's `STRATA_POOL_FUSED` (2026-11-04, PR build `strata-pr1`, not installed)

Daily setup, 10 GiB, model card settings, 3 runs each, order off, on, on, off, off, on. Not the report's conditions (see its method in the queue).

| | Off | On | Change |
|---|---:|---:|---:|
| Whole run | 43.0 (42.3-43.7) | 42.9 (42.6-43.4) | -0.2% |
| Short prompts | 46.6 | 46.0 | -1.1% |
| After a 50K prompt | 35.4 (31.7-39.8) | 37.6 (36.8-38.7) | +6.4% |
| Follow-up | 38.1 | 39.5 | +3.9% |
| ms per window | 43.8 | 44.1 | |
| Tokens per window | 1.88 | 1.89 | |
| CPU per reading layer, ready + just read | 0.266 + 0.139 ms | 0.268 + 0.139 ms | no change |

No gain seen, and no crash or damaged answer. All ranges overlap. Its report got +5.6% on a laptop where CPU experts take about half the window.

**It was active (`STRATA_POOL_PHASE_TIMING=1 STRATA_DECODE_TIMING=1`, one run each, 4 requests):** rows phases per window 137-180 off, 64-84 on, and quantize 0.08-0.14 ms off, 0.00 on. Tasks busy 88% off, 89-94% on. Host tail 1.0-1.8 ms off, 0.5-1.2 ms on. Pool wall about 14.1 ms a window off, 13.2 on, over different answers. So the barrier saving is real but small, which this A/B couldn't resolve.

**Repeat at temperature 0 with pool timing, 3 runs each, order off, on, on, off, off, on:** pool time 14.94 (14.85-15.10) -> 14.32 ms a window (14.03-14.75), -4.1%, with no overlap. Tasks busy 88.5% -> 93.1%. Host tail 1.38 -> 0.83 ms. Whole run 48.3 -> 48.1 tok/s, overlapping. Follow-up 44.3 -> 42.3 tok/s, without overlap, but tokens per window also fell 4.3% in this arm (2.20 -> 2.10), so that's more likely the answers than the switch.

**Verdict for this PC:** the saving is real, about 0.6 ms of a 45 ms window (about 1.4%), and doesn't show in tok/s. Not pushed to `speed-fork` on its own. Harmless when off. Bitwise test passed here, but no whole-engine answer check has been run with it. Temperature 0 didn't make answers repeat: off against off matched on 0 of 6 prompts at these settings.

## Read-ahead at K=5 (2026-11-04, test engine `strata\build`, not installed)

`STRATA_DIRECT_AHEAD=2 STRATA_LOOKAHEAD_K=5`, from PR #1's report. Daily setup otherwise, 10 GiB, model card settings, 3 runs each, order off, on, on, off, off, on.

| | Off | K=5 | Change |
|---|---:|---:|---:|
| Whole run | 43.2 (41.3-44.2) | 44.4 (44.0-44.5) | +2.7% |
| Short prompts | 46.5 | 47.5 | +2.1% |
| After a 50K prompt | 36.5 (31.1-40.2) | 36.6 (32.1-40.5) | +0.5% |
| Follow-up | 38.5 (37.3-40.0) | 40.8 (38.5-42.9) | +6.1% |
| Prefill | 1,392 | 1,395 | |
| ms per window | 44.1 | 42.1 | -4.5% |
| Tokens per window | 1.90 | 1.87 | |
| Read time per reading layer | 0.842 ms | 0.757 ms | -10% |
| Waited on reads | 21.1 s (19.8-22.5) | 18.3 s (17.7-18.8) | -13.2% |
| Lowest available RAM | 3.18 GiB | 3.61 GiB | |
| Paged out | 0.07 GB/h | 0.02 GB/h | |

- **Active:** one run's last count was 79,175 blobs read ahead, 43,282 used (55%), 35,814 never asked for, 0 too late. K=10 used about 28% and ran about 18% slower. So about 45% of what it reads is wasted SSD work.
- **Only the wait doesn't overlap.** Tokens per window were lower in this arm, so ms per window is the fairer measure: about +4.7% at equal tokens per window.
- **Same answers:** normal against K=5 at the identical-output settings, identical on all 6 prompts up to 50K, with read-ahead active (158,768 blobs read). One pair only. Those settings have swaps nearly off, and tail skip and PCIe off.
- **Passes as a candidate.** Combination with read-through is untested.

## Research: lossless or near-lossless speed, no expert skipping (2026-11-05, reading only, nothing run)

The asked rule: no expert skipping or rerouting. Sources were read at abstract or summary level. FluxMoE, DALI and the delayed tree paper couldn't be opened, so only their titles and search snippets are known. Gains below are estimates, and real gains here have run lower than modeled.

| Candidate | Output | Evidence | Estimate here | Next step |
|---|---|---|---|---|
| `STRATA_HC_Q8` | Slightly changed, unmeasured | Measured here, 2 runs each: 2,181 slots, short +8%, prefill +10%, 50K and follow-up within noise | About +3-8% | KL and `tools/quant_quality.py` |
| More VRAM slots | Same | `next_sim.py`: +0.5 GiB about +2%. 593 MiB free at start, 833 MiB used before the engine, engine warns below 1,024 MiB free | About +1-2% | Slot count A/B, fewer GPU-using apps |
| Expert pointers instead of gathering | Same | vLLM RFC: gathering 2.1-3.2 ms a layer against 0.87 ms of compute, pointers 1.39x in the kernel, on Intel GPU | Unknown, this engine not checked | Read the kernels |
| Better draft by training for acceptance (LK losses) | Same text distribution in theory | Abstract only. Own arithmetic: 1.83 to 2.2 tokens a window about +5% | About +5% if reached | Large: training needed |
| IoRing, Windows native NVMe driver | Same | Modeled +3.3-5.1%; driver reported as preview, not checked on 26H2 | About +3% | Prototype; driver is the user's setting |
| Large pages | Same | No source found; 2.1 "points towards memory" only | Unknown | The right is the user's to change |
| Expert split (MoEpic) | Not stated as lossless | Qwen1.5 and Mixtral; speedups against other offloaders | Unknown | `next_sim.py` first, large engine work |
| Delayed tree branching | Same text distribution in theory | Paper unread. Own second-choice data: 37% at depth 0, 47% at depth 1, gross limit +4.7% | Break-even or small | Not now |

**Ruled out:** lossless compression for speed, since published ratios are for floats and no results were found for quantized weights (still cheap to sample after the benchmarks). Edge0's prerouter replaces the routing, and quality fell (79.2 against 83.2 on 35B). Expert substitution. AMX, which this CPU doesn't have. Coalesced batches, since the batch here is 1. Router read-ahead is what we already tested: the vLLM RFC's about -1% on a discrete GPU fits the small gain here.

**Reading:** this doesn't change the estimate of about 50-55 tok/s on coding. Nothing here is a route to 70-80.

**Agreed with the user 2026-11-05:** after the read-through-off session, the daily setup with `STRATA_ROUTE_TAIL_SKIP` at the default 7 against 0, 3 runs each, order `zyyzzy`, `memtools/skip1`, arms y and z in `readme_bench.py`. Also check that skip 0 really fired by the missing banner and skip counter. Then, if the user agrees: the #5166 benchmark here, `STRATA_HC_Q8` KL and quality, and a port of `STRATA_PREFILL_ROUTED_STREAM`.

**Result (`memtools/skip1`, 15:50-16:03, daily setup, two SSDs, read-through on, order `zyyzzy`):**

| | Skip 7 (default) | Skip 0 | Skip 0 against 7 |
|---|---:|---:|---:|
| Text | 45.9 (45.9-46.0) | 35.9 (35.7-36.0) | -21.8% |
| After 26K file | 38.0 | 29.1 | -23.4% |
| Follow-up | 40.2 | 29.5 | -26.6% |
| Image | 31.2 | 24.4 | -21.8% |
| Prompt read | 1,381 | 1,388 | Same |
| ms/step | 42.1 | 55.1 | |
| tok/step | 1.9 | 2.0 | |
| Written per run | 0, 0, 0 GB | 0.37, 0.31, 0 GB | |

Ranges don't overlap on text. It fired: skip 0's log has no banner or skip counter, skip 7's first run skipped 244,615 experts. 0 damaged, 0 failed. Answer quality wasn't compared. The same skip 7 settings ran 48.9 in `fair2` an hour earlier, so about 6% between batches; compare inside a batch. Upstream's source comment gives skip 7's cost as KL about 0.003-0.006; not measured on this setup. **Agreed with the user:** after the stock runs, `memtools/tool_probe.py` (two tools, one enum, temperature 0, 30 identical requests per engine start, order `7007`, `memtools/probe1`), then `tools/quant_quality.py` at 1,978 slots with skip 0 and 7, order `0707`, `memtools/kl_skip`.

**Tool-call probe (`memtools/probe1`, 16:22-16:28, `memtools/tool_probe.py`, order `7007`, temperature 0):** 0 of 60 not clean at skip 7, 0 of 60 at skip 0. All 120 replies gave the same arguments. Cache-hot as in #5225 (468 reused + 5 read). Skip confirmed on and off in the logs. Mean speed 61 against 39 tok/s on these short replies. One schema and one prompt, CUDA, so rarer corruption or real multi-tool sessions aren't ruled out.

**KL, teacher-forced (`memtools/kl_skip`, 16:28-16:48, `tools/quant_quality.py`, 4,995 positions, `STRATA_IQ_MT_MIN=1`, FP16 K/V, no adaptation, no suffix drafts, order `0707`). Reference is skip 0, not Q8_0:**

| Segment | Top-1 agreement | KL | Perplexity, skip 0 to 7 |
|---|---:|---:|---:|
| Code | 96.60% | 0.02732 | 2.031 to 2.033 |
| Other languages | 94.39% | 0.02741 | 2.686 to 2.687 |
| English | 88.99% | 0.04758 | 4.151 to 4.216 |
| Docs | 83.98% | 0.10489 | 10.057 to 10.435 |
| Agent | 89.49% | 0.19632 | 4.699 to 5.109 |
| All | 90.69% | 0.08070 | 4.035 to 4.148 |

- **Repeatable:** skip 0 against skip 0 and skip 7 against skip 7 are both KL 0.00000, 100% top-1. The last run overlapped the other chat's 15-second clone on D: and still matched exactly.
- **Far above upstream's 0.0030-0.0058 and 97.5-98.2%** (`docs/DETAILS.md`: 28 chat prompts, RTX 3060 IQ3_XXS and RTX A4000 Q2_0, against the old behavior). Why isn't established. Differences here: other pack and card, fixed text set, no adaptation, 2,555 slots (`--gpu-experts 1978` was asked for, the log says 2,555, not explained) against the daily 1,978. Skipped 385,467 experts in 4,995 positions.
- **What it doesn't show:** whether answers get worse on real tasks. No task suite was run. It's above the 0.012-0.02 that kept layer-major off, a different KL test.
- **Next, proposed:** `STRATA_ROUTE_TAIL_SKIP` at 8 and 9, and `STRATA_ROUTE_TAIL_SKIP_FILE=7`, for speed and KL.

**Second pass the same day, through Chrome (full paper text and upstream issues). Corrects the table above:**

- **The daily setup already skips experts.** `STRATA_ROUTE_TAIL_SKIP=7` is on by default on both engines (`sess_on/engine.log` line 63). That session skipped 4,948,041 missed experts, against 10,730,049 RAM blobs and 34,244 file blobs, so roughly a third of misses if the counters compare. Every speed in this log and the "no skipping" estimate above include it, except the identical-output checks. Cost of turning it off here isn't measured; upstream advertises +10-20% decode. Upstream #5225 (R9700, engine 0.1.56): tool-call arguments corrupted with copied schema text in about 6% of scripted requests, gone with skip 0 in 32 requests; the link to skipping is the reporter's suspicion. Not seen or checked on this PC.
- **Large pages: crossed off for now.** Upstream #5238, ABAB on an RTX 5080 with 96 GB and 4,161 slots: no gain, 4 KB ahead in 7 of 8 decode metrics, within spread. That PC has more experts in VRAM than this one, so it isn't proof here, but there is no reason left to ask for the right.
- **New lossless, prompts only:** `STRATA_PREFILL_ROUTED_STREAM` (upstream PR #4283, open): stops copying experts no token routes to. RTX 3090: follow-up read -17.9%, 2K read +19.9%, 4K +8.2%, 32K +5.3%; byte-identical in its deterministic lane; agent decode -3.5% on differing texts; flat or slower on 2x RX 6900. `STRATA_SELECT_TWO_PASS` (PR #4119, open): 256K read +5.2%, 32K +0.2%, byte-identical; not run on sm_89 or Windows.
- **Lead to compare:** upstream #5166, RTX 5070 12 GB, IQ3_XXS, 64 GB, Linux, 141.3 tok/s at temperature 0, thinking off, short prompts: 94.7% cache hits, 2,918 slots in 4,887 MiB at 192K, `spec=12 mtp_max=4 lookup=3`, 11 workers. Another card, pack, OS and workload. Its 1 GiB more of expert cache than the 3.8 GiB here is unexplained. Its `mtp-bench-3.py` (PR #4877) can run here for a direct comparison.
- **Figures now read:** LK losses up to 8-10% longer acceptance, which is about +3% here by the 13 ms a window + 8 ms a token fit, not +5%. Delayed tree: about 5% with a neural selector, on other verification methods. DALI: its prefetch added only 9%; its greedy assignment matches what `assign_sim.py` put at about +1% here, and its cache gain comes from workload across a batch, while the batch here is 1. MoEpic: half an expert over PCIe is about 42 us against about 50 us on the CPU pool, serialized on one link, so about break-even by arithmetic. FluxMoE: 32% smaller by coding BF16 exponents, decoded on the GPU at about 400 GB/s on an H20. This pack's experts aren't BF16, so it needs the sample. KV exact-cell window (PR #4068) refuses `--kv-resident`, so it doesn't fit this setup.
- **Route prior (`STRATA_ROUTE_PRIOR`, 0.2.7-0.2.8):** rerouting, excluded, and upstream turned it off for repeated text in long sessions.

## Test queue and what remains (living list, started 2026-11-04)

Update this as tests finish: move results to their own sections and keep only what's still open here. Nothing below is measured unless it says so.

**Queue, in order. Main focus: PR #1, then upstream Strata PRs.**

Part 1, PR #1 and its report:

| # | Test | Status |
|---|---|---|
| 1 | Read-ahead K=5 (`STRATA_DIRECT_AHEAD=2 STRATA_LOOKAHEAD_K=5`) A/B, 6 runs | Done, see its section |
| 2 | Read-ahead answer check, identical-output settings | Done, identical |
| 3 | `STRATA_POOL_FUSED`: proved active, repeated at temperature 0 | Done: real but about 1.4% here, not pushed alone |
| 4 | Layer-major prompts with the head alias | First result 14-17% slower, caused by the old loan. With the narrow loan -18% and -6% time, see "Narrow head-alias loan" |
| 4a | Narrow loan, arms A-D, 228K, needle, decode and paging | Done, see its section |
| 4b | Layer-major KL at identical-output settings (`loan7`), chunk control (`loan8`) | Done: layer-major itself, 0.012-0.020 past 32K, chunk size alone 0.002, same top token |
| 4b2 | Where the layer-major gap starts: a prompt just under and just over 32,768 tokens, and 50K with a larger `--kv-resident` | Next, about 15 minutes |
| 4c | D with the cache capped for about 500 MiB free at decode, and `STRATA_HEAD_ALIAS_SHORT` raised | Only if the gap is explained and accepted. Then a real agent session |
| 5 | `STRATA_LM_CPU_ROWS` | Back open if layer-major clears 4b: changes bits per its comment, so KL first |
| 6 | `--prefill-head-alias`, `STRATA_HEAD_FAST_LOAD`, `STRATA_HEAD_ALIAS_SHORT`, `STRATA_REFILL_PINNED` | Alias measured with the narrow loan. `STRATA_REFILL_PINNED` alone is untested: the narrow loan's refill is already 400-700 ms, so at most that |
| 7 | Read-through with read-ahead K=5 | Done: read-through alone wins, read-ahead adds nothing on top |
| 8 | PR #1 results from its author on the RTX 4060 laptop (branch `ram-through-test`) | Done, see its section, including the 16 GiB rerun with the browser closed. Open from it: read-through at 14 GiB on this PC, with paging timed per request |
| 9 | PR #1 merge plan, below: decide switch by switch | After 1-8 |

Part 2, upstream Strata PRs:

| # | Test | Status |
|---|---|---|
| 10 | #4040: `--lookup-max-t 0` against 12 on `strata-0210` | Done: about +4% on rewrites, none on coding, and answers differ from off. Not ported |
| 11 | #1324 (LRU part of the RAM tier) | Dropped 2026-11-05 by the user: read-through has its own results |
| 12 | #3517 (conversation cache retention) | Dropped 2026-11-05 by the user: saving is off in daily use. Reconsider only if the conversation cache is turned on. If reopened, also read Maya #607: shared system-prompt state kept apart from conversations, reported about 19x sooner first token on DeepSeek. Not checked if the fork already does this at `--conversation-cache-mib 0` |
| 13 | #4283 (routed prompt stream) | Dropped 2026-11-05 by the user: small at long context |

Part 3, Project Maya (github.com/mw00/project-maya, v1.1.53), queued only, none run. Placed after Parts 1 and 2 for now; the order against them is the user's call. MIT licence, so ported code needs its notice and credit. Details as sent by the session reviewing it, not checked here. All figures are from other models and GPUs, and most settings are GLM-specific and may need porting. Only about half of its settings and changelog was read in detail. Developed on Linux. Only its GLM `MAYA_GLM_CPU_PIN` never pins on Windows; its Qwen pool does pin there. Source check by that session in `maya-review.md` (source reading only, not checked here); the statuses below are its suggestions, and moving items is the user's call. SSD ideas stay under the modeled zero-SSD-wait ceiling of about 68 tok/s.

| # | Test | Status |
|---|---|---|
| 14 | Maya's Qwen against the 10 GiB daily setup and vanilla | Suggested low priority: `--import-model` matches by size and sha256, so it won't take IQ3_S, and Maya's Qwen is a different 112.5 GB fine-tune. Its Qwen path lacks several read paths this fork has, so expected slower here (a prediction from source) |
| 15 | `MAYA_GLM_STREAM_FETCH` (GPU copy starts as SSD pieces land). Reported about +22%, V100 | Suggested hold: GLM only. First check if SSD-read experts go to the GPU here at all. `STRATA_STAGE_PIN`'s comment records corrupted output from an asynchronous copy out of a recycled stage buffer |
| 16 | `MAYA_GLM_LEND_PARK_LRU` (eviction ties broken by recency) | Suggested drop: GLM only, and decayed float counts already carry recency |
| 17 | `--prompt-cache-every 0` against the default 16384 on a long prompt (this fork's flag, no port; Maya's `MAYA_GLM_CKPT_EVERY` is GLM only) | Suggested cheap flag test, low expected gain: its reported cost is on a split, and 116,780 tokens here already read at about 1,500 tok/s |
| 18 | `--expert-profile-save`, a learned profile, on unseen prompts (this fork's flag; `--expert-profile` is already used). Reported 40.7 against 35.4 tok/s is GLM on 9 GPUs at a fresh start | Suggested cheap flag test after 17. Helps fresh starts only. Fixed profiles lost to adapting in "Cache miss audit" |
| 19 | `--calibrate`, as a method only: unseen prompts, keep a setting only above 3% | Suggested later: its tuning is GLM, and Qwen's only measured choice is drafter on or off |

Not queued: its read-ahead (mostly flat or worse there), expert skipping and Turbo (trade quality), DFlash2 (already tested here; CC BY-NC-ND), multi-GPU.

**If `STRATA_POOL_FUSED` passes** (gain above noise, same answers, no crashes): one commit on `speed-fork` with only its `pool.cpp`, `pool.hpp` and `pool_tasks_test.cpp` changes, the PR's author as commit author, off by default. Check the normal path and the build first, and push after the user confirms.

**Combinations still untested:**

| Switch | With | Why it matters |
|---|---|---|
| Read-through | `STRATA_POOL_FUSED` | Both change the cold-read layers; gains may not add up |
| Read-through | Read-ahead | Both use the stage buffers, and read-through drops its aliases each layer |
| Read-through | `STRATA_ROUTE_TAIL_SKIP_FILE` | Read-through changes what's in RAM, so it changes what gets skipped |
| Read-through | `STRATA_HC_Q8` | More VRAM slots, fewer reads, so less to gain |
| Read-through | Conversation cache on | Restores and parking near a moving tier |
| Read-through | `--mmproj` image prompts | No image was run with it |
| Read-through | `--adapt-decay` other than 0.7 | The floor follows decay now, but only 0.7 has run |
| `STRATA_POOL_FUSED` | Read-ahead, `STRATA_ROUTE_TAIL_SKIP_FILE`, `STRATA_HC_Q8` | Only on or off alone so far |
| Read-ahead | K other than 5, and tail skip | K=10 is ruled out, K=5 is only the PR author's result |
| `--lookup-max-t` | Read-through, `STRATA_POOL_FUSED` | Not in the fork, needs a port first |
| Layer-major | `STRATA_HC_REQ8` | Don't combine on the fork without upstream #4715's fix: it loses the prompt |
| Layer-major | `STRATA_LM_CPU_ROWS`, CPU share | Its code comment says "other bits", so check KL |

**Read-through before any default:**
- More runs here: tokens per step were 6% higher in the first A/B.
- Tiers other than 10 GiB. The 6-14 GiB gains are model only.
- A long real session. Open: one run at 0.32 GiB lowest available, and 4.15 and 5.61 GB/h paged in the two verify runs (normal read-through runs about 0.5 GB/h).
- The PR author's results on the RTX 4060 laptop, with an unpinned tier and unequal drives (branch `ram-through-test`).

**PR #1 switches not run here:** `STRATA_HEAD_FAST_LOAD`, `STRATA_HEAD_ALIAS_SHORT`, `STRATA_REFILL_PINNED`, `STRATA_LM_CPU_ROWS`, `STRATA_POOL_PHASE_TIMING`. `STRATA_EXPERTS_MIRROR_SHARE` has no use with equal drives, and `STRATA_IQ3_VBMI` can't run on this CPU.

**PR #1 merge plan (later):** move the `iq_avx512.cpp` tables behind the CPU check, check all switches off, HIP, SYCL and Linux builds, and decide how to merge. Also:
- With `STRATA_POOL_FUSED` on, its time all lands in the gate/up counter and the quantize and down counters stay at 0, so any stats built on them mislead.
- The report's working-set cap (`SetProcessWorkingSetSizeEx`, 256 MiB per browser process, held about 35 tok/s with a browser open) isn't in the PR's code. Ask for it if it's wanted: it could help beside other apps here.
- The report's other settings not tried here: `--mtp-q4` (no net gain there), stage ring 128 x 8 threads, `--prefill-head-alias`.

**Measured or modeled, not built or run:**
- IoRing reads: modeled +3.3% on the reader's E-core, +5.1% on a P-core sibling.
- `STRATA_DIRECT_BATCH` 4 and 8 against 16. The read trace's "a finished blob waits" figure isn't usable when split by phase.

**Parked:**
- More tokens per window or less fixed GPU time: the only way to 70-80 tok/s, since zero SSD wait stops at about 68.
- Task benchmark (vision, CTF, systems, webapp, math and physics).
- Conversation cache at 10 GiB, with upstream #3517.
- README speed numbers: measured 2026-11-05, see "README speed rows at model card settings". README text not updated yet: the user's call.

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
