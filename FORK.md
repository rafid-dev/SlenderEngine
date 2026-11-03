# What this fork changes

This is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata), based on commit `c62d5e3` (engine 0.1.55). It makes Qwen3.8-Flash-Next IQ3_S
generate faster on one PC by changing software only: same model, quant, context, RAM and VRAM use.

**See every changed line:** [compare with upstream](https://github.com/Niko1221/Strata/compare/main...YOUR-USERNAME:Strata:speed-fork).
**See every test, including what failed:** [docs/fork/WORKLOG.md](docs/fork/WORKLOG.md).

## Result

Measured 2026-11-03, 5 runs each, mixed order, same day. i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB, two 990 PRO 1 TB, Windows 11. IQ3_S, `--kv int8`,
262K context, 14 GiB of experts in RAM.

| | Upstream | This fork, tuned | Change |
|---|---|---|---|
| Text | 37.6 tok/s (36.7-39.0) | **57.3 tok/s (55.9-59.1)** | +52% |
| Generating after a 26K file | 28.2 tok/s | 45.5 tok/s | +61% |
| Follow-up | 32.1 tok/s | 51.5 tok/s | +60% |
| Image answer | 24.2 tok/s | 47.0 tok/s | +94% |
| Prompt reading | 989 tok/s | 1,496 tok/s | +51% |

This has only been measured on the PC above. Other cards and CPUs may gain less or lose speed. The placement settings below are for this CPU's layout.

## The changes that matter

| Change | How to use it | On by default in this fork? | Where |
|---|---|---|---|
| **RTX 40 speed switches:** a `sm_89` table for 7 exact kernels the family had no table for | `STRATA_SM89_PERF=0` turns it off | **Yes** | `src/core/nv_defaults.cpp` |
| **No extra copy on unbuffered reads:** a blob is read straight into its aligned buffer | `STRATA_DIRECT_IN_PLACE=0` turns it off | **Yes, on Windows** | `src/core/expert_source.cpp` |
| **Reads split across two SSDs:** with a copy of `experts.bin` on a second drive, each read of 256 KiB or more takes half from each | `STRATA_EXPERTS_MIRROR=<path>` | No | `src/core/expert_source.cpp` |
| **Kept reader threads:** read overlap no longer creates threads for every layer, which had used up its gain | `STRATA_COLD_KEPT_READERS=1`, `STRATA_COLD_READER_CPU=<first cpu>` | No | `src/core/expert_source.cpp` |
| **Parallel RAM moves during the draft** | `STRATA_RAM_ADAPT_PAR=1`, `STRATA_RAM_ADAPT_BG=1` | No | `src/program/generate.cpp`, `src/core/expert_source.cpp` |

## Settings used for the result

These are upstream options plus the switches above. Adjust the CPU numbers and the path for your PC.

```bat
set STRATA_RAM_ADAPT=1
set STRATA_EXPERTS_MIRROR=C:\StrataMirror\experts.bin
rem P-cores only, the host thread on the eighth
set STRATA_POOL_CORES=2,4,6,8,10,12,14
set STRATA_COLD_OVERLAP=1
set STRATA_COLD_KEPT_READERS=1
set STRATA_COLD_READER_CPU=22
set STRATA_PARTIAL_PIN=1
set STRATA_RAM_ADAPT_PAR=1
set STRATA_RAM_ADAPT_BG=1
```

Engine args: `--pool-workers 7 --spec 4 --spec-min-p 0.5 --adapt-swaps 24 --resident-budget-gib 14 --max-context 262144 --kv int8`.

The dual SSD copy must be the same file as the pack's `experts.bin`. The engine only compares file sizes, so copy it again if the pack changes.

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
| `STRATA_DIRECT_AHEAD` | Read-ahead from the router's prediction | About 18% slower |
| `STRATA_DIRECT_POLL`, `STRATA_PROC_HIGH` | Polling and process priority | No gain |

## Limits

- Only the CUDA build on Windows was built and run. The HIP, SYCL and Metal builds were not built.
- `nv_defaults_test` covers the `sm_89` table. Run it after building.
- Speed tests use temperature 0.6, so tokens per step moves about 8% between runs of the same settings. A small gain needs many runs.

The license is upstream's MIT license, unchanged.
