# What this fork changes

This is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata), based on commit `c62d5e3` (engine 0.1.55). It makes Qwen3.8-Flash-Next IQ3_S
generate faster on one PC by changing software only: same model, quant, context and RAM tier size.

**See every changed line:** [compare with upstream](https://github.com/Niko1221/Strata/compare/main...rafid-dev:Strata-32GB:speed-fork).
**See every test, including what failed:** [docs/fork/WORKLOG.md](docs/fork/WORKLOG.md).

## Result

Measured 2026-11-05 on Windows 11 26H2. Model card sampling, thinking on, a fresh engine per run, 3 runs a setting. i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB, two 990 PRO 1 TB. IQ3_S, `--kv int8`, 262K context, 10 GiB tier, checkpoints off. 27 runs, 0 failed. Mean text tok/s (low-high).

| Engine | `--spec`, workers | Switches | SSDs | Text | After 26K file | Follow-up | Image | Prompt read | ms/step |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Upstream | 3, 13 | Three memory switches | 1 | 28.2 (26.9-29.9) | 20.8 | 22.0 | 15.2 | 941 | 64.6 |
| **Upstream, best we tried** | 4, 7 on P-cores | Three memory switches | 1 | **30.0** (29.1-31.5) | 22.9 | 23.4 | 15.2 | 945 | 64.9 |
| Upstream | 3, 13 | Those + read overlap, page locking | 1 | 23.6 (23.2-24.2) | 17.8 | 17.4 | 14.5 | 931 | 75.4 |
| Upstream | 4, 7 on P-cores | Those + read overlap, page locking | 1 | 24.9 (23.2-26.1) | 15.6 | 18.5 | 16.1 | 938 | 78.2 |
| Upstream | 4, 13 | Those + read overlap, page locking | 1 | 22.3 (21.2-23.3) | 18.3 | 17.8 | 14.0 | 928 | 81.1 |
| Upstream | 3, 7 on P-cores | Those + read overlap, page locking | 1 | 24.2 (23.5-24.7) | 16.9 | 18.0 | 15.5 | 925 | 73.8 |
| This fork | 3, 13 | All, one SSD | 1 | 41.4 (40.4-42.5) | 35.6 | 34.6 | 31.8 | 948 | 41.6 |
| This fork | 4, 7 on P-cores | All, one SSD | 1 | 41.8 (39.8-42.8) | 33.8 | 32.0 | 30.5 | 932 | 43.7 |
| **This fork, best** | 4, 7 on P-cores | All | 2 | **48.9** (48.1-49.8) | 43.3 | 42.2 | 38.6 | 1,410 | 40.4 |

- **Best against best:** +39% on one SSD, +63% on two. The upstream best came about an hour after these fork runs and wasn't mixed in with them. The fork with two SSDs ran 45.9 in a later batch, closer to those runs, which would make it +53%.
- **Same settings:** +75% at `--spec 3` with 13 workers, +68% at `--spec 4` with 7. These flatter the fork. Upstream's read overlap creates threads for every layer, and upstream ran faster with overlap and page locking off. Which of the two costs it wasn't tested. The fork arms also have its own switches on.
- **The second SSD alone:** text +17%, prompt reading +51%. On one SSD the fork's code doesn't speed up prompt reading.
- **The gain is time per step, not tokens per step.** Tokens per step were 1.7-2.0 on every row.
- **Expert skip:** both engines ran upstream's default `STRATA_ROUTE_TAIL_SKIP=7`. With it off, the fork's daily setup went from 45.9 to 35.9 tok/s (-22%), 3 runs each, mixed order. `tools/quant_quality.py` against skip 0, 4,995 positions, repeatable: KL 0.081, top-1 agreement 90.7%, perplexity 4.035 to 4.148, and on the agent text KL 0.196, 89.5%, 4.699 to 5.109. Upstream reports KL 0.003-0.006 on other packs and cards; the reason for the gap isn't established. A tool-call test like upstream #5225's was clean at both settings, 60 requests each. No task suite was run.

The three memory switches are `STRATA_UNBUFFERED_LOAD`, `STRATA_RESIDENT_HEADROOM_GIB=1.5` and `STRATA_RAM_ADAPT`. Strata 0.2.10 hasn't been compared this way.

This has only been measured on the PC above. Other cards and CPUs may gain less or lose speed. The placement settings below are for this CPU's layout.

## The changes that matter

| Change | How to use it | On by default in this fork? | Where |
|---|---|---|---|
| **RTX 40 speed switches:** a `sm_89` table for 7 exact kernels the family had no table for | `STRATA_SM89_PERF=0` turns it off | **Yes** | `src/core/nv_defaults.cpp` |
| **No extra copy on unbuffered reads:** a blob is read straight into its aligned buffer | `STRATA_DIRECT_IN_PLACE=0` turns it off | **Yes, on Windows** | `src/core/expert_source.cpp` |
| **Reads split across two SSDs:** with a copy of `experts.bin` on a second drive, each read of 256 KiB or more takes half from each | `STRATA_EXPERTS_MIRROR=<path>` | No | `src/core/expert_source.cpp` |
| **Kept reader threads:** read overlap no longer creates threads for every layer, which had used up its gain | `STRATA_COLD_KEPT_READERS=1`, `STRATA_COLD_READER_CPU=<first cpu>` | No | `src/core/expert_source.cpp` |
| **Parallel RAM moves during the draft** | `STRATA_RAM_ADAPT_PAR=1`, `STRATA_RAM_ADAPT_BG=1` | No | `src/program/generate.cpp`, `src/core/expert_source.cpp` |
| **Read-through RAM tier:** a missed expert's read lands straight in the RAM place of the layer's least used expert, instead of being copied there later. +5-13% at 10 GiB here, -19% at 14 GiB | `STRATA_RAM_THROUGH=1`, `STRATA_RAM_THROUGH_VERIFY=N` checks the tier against the file | No. `FAST-SETUP.bat` writes it at 10 GiB or below | `src/core/expert_source.cpp`, `src/program/generate.cpp` |
| **K/V-tail lending:** the part of the pinned K/V copy that a request can't reach holds extra RAM-tier experts, so no new RAM is used. SSD misses -27%, short decode +11%, answers byte-identical at temperature 0 | `STRATA_KV_LEND=1` | No | `src/core/expert_source.cpp`, `src/program/generate.cpp` |
| **In-place prompt stager:** while a prompt is read, a file-tier blob is read unbuffered straight into its pinned stage buffer, with no copy in between. With fewer, smaller reads it's +8.5% to +13.9% on prompt reading, and the CPU drops from 87% to 10% busy. Answers byte-identical at temperature 0 | `STRATA_STAGER_IN_PLACE=1`, with `STRATA_STAGER_THREADS=3`, `STRATA_STAGER_BATCH=1`, `STRATA_STAGER_RING=128` (tuned for two 990 PROs) | No | `src/core/expert_source.cpp`, `src/prefill/prefill.cpp` |

## Settings used for the result

`FAST-SETUP.bat` works these out for your PC ([README](README.md#easy-setup)). By hand, they are upstream options plus the switches above. Adjust the CPU numbers and the path for your PC.

```bat
set STRATA_UNBUFFERED_LOAD=1
set STRATA_RESIDENT_HEADROOM_GIB=1.5
set STRATA_RAM_ADAPT=1
set STRATA_RAM_THROUGH=1
set STRATA_EXPERTS_MIRROR=C:\StrataMirror\experts.bin
rem P-cores only, the host thread on the eighth
set STRATA_POOL_CORES=2,4,6,8,10,12,14
set STRATA_COLD_OVERLAP=1
set STRATA_COLD_KEPT_READERS=1
set STRATA_COLD_READER_CPU=22
set STRATA_PARTIAL_PIN=1
set STRATA_RAM_ADAPT_PAR=1
set STRATA_RAM_ADAPT_BG=1
rem each SSD read is handed over as it lands, not in batches of 16
set STRATA_DIRECT_BATCH=1
rem not written by FAST-SETUP: skip 0, K/V lending and the in-place prompt stager (tuned for two 990 PROs)
set STRATA_ROUTE_TAIL_SKIP=0
set STRATA_KV_LEND=1
set STRATA_STAGER_IN_PLACE=1
set STRATA_STAGER_THREADS=3
set STRATA_STAGER_BATCH=1
set STRATA_STAGER_RING=128
```

Engine args: `--pool-workers 7 --spec 4 --spec-min-p 0.5 --adapt-swaps 24 --resident-budget-gib 10 --conversation-cache-mib 0 --max-context 262144 --kv int8 --pcie-frac 0`.

The dual SSD copy must be the same file as the pack's `experts.bin`. The engine only compares file sizes, so copy it again if the pack changes.

`FAST-SETUP.bat` no longer writes 14 GiB. It asks how much RAM to use, defaulting to 10 GiB on a 32 GB PC, and writes `--conversation-cache-mib 0`. [README](README.md#how-much-memory-it-uses) has the numbers for that setting and for running beside a large app.

## What setup installs

| Setting | What setup installs |
|---|---|
| Experts cached in RAM | **10 GiB**, or what you choose |
| Saved conversation checkpoints in RAM | **Off** |
| Drafting | `--spec 4` |
| CPU workers | One per P-core, leaving the first for the host (7 here) |
| SSDs | Two if you pick a second drive |
| Read-through | **On at 10 GiB or below** |

## Read-through

| Windows 26H2, 10 GiB, 3 runs each, mixed order | Off | On | Change |
|---|---:|---:|---:|
| Coding answers | 44.2 tok/s | 49.2 tok/s | +11% |
| Generating after a 50K prompt | 34.0 tok/s | 39.9 tok/s | +17% |
| Follow-up question | 36.7 tok/s | 41.5 tok/s | +13% |
| Whole run | 40.9 tok/s | 46.4 tok/s | **+13%** |
| Experts read from the SSD | 108,837 | 83,899 | -23% |
| Written to the page file | 0-0.8 GB/h | 0 GB/h | |

- **Other batches on this PC at 10 GiB:** +5.2%, +6.0% and +10.9% over the whole run. The gain varies, and "off" here is slower than in the table at the top, for reasons not established. The Windows update and other open apps both changed between them.
- **On the RTX 4060 8 GB laptop:** +30% and +15% on short prompts at 10 GiB, without and with read-ahead. At 16 GiB with the browser closed, +8-13% on short and long-context answers, and about flat on follow-ups. Measured and reported by that laptop's owner.
- **Correctness:** the whole cache was compared with the file up to about 80,000 reads on this PC and about 120,000 on the laptop, and nothing differed. With answers made repeatable on this PC at 10 GiB, the answers were the same with it on and off.
- **Not above 10 GiB here:** at 14 GiB it **lost about 19%** (40.3 against 49.5 tok/s). There the driver can't lock the whole cache, and reads landing in ordinary memory took about twice as long. A first test with ordinary memory at 10 GiB, one run each, was just as slow with plenty of RAM free.
- **Not measured:** other amounts of RAM on this PC.

## Two SSDs alone

From an earlier stage of this fork, two 990 PRO, 4 runs each. With everything on, it hasn't been measured separately.

| | One SSD | Two SSDs | Gain |
|---|---:|---:|---:|
| Text generation | 42.2 tok/s | 47.4 tok/s | +12% |
| Reading a prompt | 997 tok/s | 1,389 tok/s | +39% |
| Generating after a 26K file | 31.4 tok/s | 37.3 tok/s | +19% |
| Follow-up question | 34.5 tok/s | 41.5 tok/s | +20% |
| Answering about an image | 24.8 tok/s | 31.6 tok/s | +28% |

Setup refuses a second partition of the same disk, hard disks and USB drives. To turn it off, run `FAST-SETUP.bat --no-mirror` and delete the `StrataMirror` folder.

## Beside a large app

With Roblox Studio open using about 12 GiB, these settings kept writes to the page file at 0.0-0.1 GB/h (2 runs of about 3.5 minutes): `--max-context 65536 --resident-budget-gib 1 --pool-workers 4 --vram-reserve-mib 1024`, with no `--kv-resident`. Speed fell to about 20 tok/s at short context and 15 tok/s at 50K, and context is limited to 64K. Whether Studio stuttered wasn't measured.

## Answers

With `STRATA_IQ_MT_MIN=1`, `STRATA_ROUTE_TAIL_SKIP=0`, `STRATA_PREFILL_CPU_SHARE=0` and `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0`, upstream, this fork
and the tuned setup at `--spec 3` gave byte-identical answers on 3 short prompts. `--spec 4` changes the wording on upstream and this fork alike. Partial page-locking's
PCIe path and the RAM moves did not run in that check. In normal use answers can vary between runs on any engine: `STRATA_PREFILL_CPU_SHARE` splits short prompts by measured timing, and upstream's reproducibility list does not turn it off.

## Also in the code, all off by default

| Switch | What it is | Result |
|---|---|---|
| `STRATA_ROUTE_TRACE`, `STRATA_WINDOW_TRACE`, `STRATA_PROMPT_TRACE`, `STRATA_ADAPT_TIMING` | Measurement only | |
| `STRATA_MTP_TOP2` | Counts how often the draft's second choice was right (the calls were missing) | Measurement only |
| `STRATA_RAM_ADAPT_ANY`, `STRATA_RAM_ADAPT_ROUTED`, `_UNUSED`, `_PER` | More RAM moves | No gain: fewer file reads, but copying costs more |
| `STRATA_SPEC_CUM_MIN` | Stops drafting on a low joint chance | Within noise or worse |
| `STRATA_ADAPT_TAIL_WEIGHT`, `STRATA_ADAPT_SSD_BOOST` | Cache scoring | Not used |
| `STRATA_DIRECT_AHEAD` | Read-ahead from the router's prediction | The first version was about 18% slower. With `STRATA_LOOKAHEAD_K=5`: +2.7% here, no gain on top of read-through, 45% of its reads wasted. More on an RTX 4060 laptop |
| `STRATA_POOL_FUSED`, `STRATA_IQ3_VBMI`, `STRATA_LM_CPU_ROWS`, `STRATA_HEAD_FAST_LOAD`, `STRATA_HEAD_ALIAS_SHORT`, `STRATA_REFILL_PINNED`, `STRATA_EXPERTS_MIRROR_SHARE`, `STRATA_POOL_PHASE_TIMING` | From PR #1, for 8 GB cards, see [its report](docs/fork/LAPTOP-RTX4060.md) | Here `STRATA_POOL_FUSED` gave about 0.5% over a whole run, and `STRATA_IQ3_VBMI` can't run on this CPU. The others aren't measured alone here |
| `STRATA_HEAD_ALIAS_NARROW` | With `--prefill-head-alias`, lends only the slots the chunk needs | About 9% faster than the full loan at the same chunk. With layer-major, long prompts are 6-18% faster, but its answers differ (KL 0.02 at 50K), so that stays off |
| `STRATA_DIRECT_POLL`, `STRATA_PROC_HIGH` | Polling and process priority | No gain |
| `STRATA_PARK_TAIL_MAX` | Skips saving a conversation when only its last tokens were rewritten | Saved nothing: the RAM floor already refused those saves |
| `STRATA_TIER_LOW_MIB`, `_HIGH_MIB`, `_STEP_MIB` | Resizes the RAM tier while running | Not used, no result recorded |
| `STRATA_ROUTE_TAIL_SKIP_FILE=R` | The route tail skip from rank R, only for experts that would be read from the file. Changes answers | 10 GiB, 2 runs: about +12% at short context, +21% at 50K. Quality not measured |

## Limits

- Only the CUDA build on Windows was built and run. The HIP, SYCL and Metal builds were not built.
- `nv_defaults_test` covers the `sm_89` table. Run it after building.
- Tokens per step moves about 8% between runs of the same settings, so a small gain needs many runs. Numbers before 2026-11-05 in the worklog mostly use temperature 0.6.
- With all switches off, this branch gave byte-identical answers to the installed engine on 6 prompts at the settings under Answers, with `--adapt-swaps 1 --adapt-every 1000000`. `tools/test_setup_choices.py` has 2 errors in `test_the_encoder_is_built_native`, in code this branch doesn't change.

The license is upstream's MIT license, unchanged.
