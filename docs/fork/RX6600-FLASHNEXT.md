# Qwen3.8-Flash-Next on an 8 GB AMD card with 32 GB of RAM (RX 6600, HIP, Windows), 2026-11-07 to 11-08

AMD Radeon RX 6600 8 GB (gfx1032, on PCIe 3.0 x8: 6.8 GB/s measured), Intel Core i5-10400 (6 cores, AVX2, no AVX-512),
32 GB DDR4-2666, Windows 11. Two NVMe SSDs: an HP EX900 Plus 1 TB (the pack) and a Transcend 110S 512 GB (the mirror).
The engine is this branch built from source for gfx1032 (`tools\hip\build_windows.bat`, `STRATA_HIP_ARCHS=gfx1032`, see
[AMD_HIP.md](../AMD_HIP.md#gfx1032-rx-6600-on-windows)). Model: the ISTA-DASLab GSQ-RCO IQ3_S GGUF, set up with
`setup.py --family qwen --model IQ3_S --gguf-dir ... --kv int8 --kv-streaming on --low-ram on --drafter dflash2`.

All numbers: thinking on, the model card's sampling (temperature 1.0, top-k 20, top-p 0.95), a fresh engine per run.
Decode is the mean of two A-level questions and "lalaland", 600 tokens each. Pages written to the page file were sampled
every 2 s during every run (the limit used here: 1 GB/h).

## Result

| | Setup's config | This page's config | With `STRATA_ROUTE_TAIL_SKIP=7` (changes answers) |
|---|---:|---:|---:|
| Decode, short questions | 5.3 tok/s | **9.5-10.5 tok/s** | 13.5 tok/s |
| 4K-token prompt | 14 tok/s | **127 tok/s** | |
| 24K-token prompt | about 7.5 tok/s (stopped after 443 s) | **142 tok/s** (158 s) | |
| Decode after the 24K prompt | | 9.4 tok/s | |
| Written to the page file | | 0.02 GB/h (whole run) | 0.07 GB/h |
| Lowest free RAM | 64 MB | 7.6 GB | 4.8 GB |

## The config

The run config's `"env"` and `"user_args"` (setup keeps both when it runs again):

```json
"env": {
  "STRATA_UNBUFFERED_LOAD": "1", "STRATA_RAM_ADAPT": "1", "STRATA_COLD_OVERLAP": "1", "STRATA_COLD_KEPT_READERS": "1",
  "STRATA_PARTIAL_PIN": "1", "STRATA_RAM_ADAPT_PAR": "1", "STRATA_RAM_ADAPT_BG": "1", "STRATA_KV_LEND": "1",
  "STRATA_DIRECT_BATCH": "1", "STRATA_STAGER_IN_PLACE": "1", "STRATA_DENSE_ARENA": "1", "STRATA_RAM_THROUGH": "1",
  "STRATA_STAGER_RING": "128", "STRATA_STAGER_THREADS": "3",
  "STRATA_EXPERTS_MIRROR": "C:\\StrataMirror\\experts.bin", "STRATA_EXPERTS_MIRROR_SHARE": "0.57",
  "STRATA_LM_HIP": "1", "STRATA_PREFILL_LAYER_MAJOR": "1", "STRATA_PREFILL_LM_TOKENS": "32768",
  "STRATA_PREFILL_LM_R": "host", "STRATA_DENSE_MMQ": "1"
},
"user_args": ["--resident-budget-gib", "12", "--conversation-cache-mib", "0", "--draft-conf", "0.8",
              "--expert-cache", "60", "--prefill", "512", "--kv-resident", "8192"]
```

What each part does here, biggest first:

| Setting | Setup wrote | Why |
|---|---|---|
| `--expert-cache 60` (77 GPU slots) | auto (357 slots) | **The largest single change.** With the automatic cache the card ran with 270-390 MiB free inside the 7,028 MiB Windows gives this process, and the driver paged GPU memory: the prompt's dense GEMMs ran 10-17x slower (a 4K prompt's GDN input projections 61 s instead of 5.6 s) and decode lost about 20%. On this card the GPU cache holds about 1.6 of the ~15 experts a layer uses, so the slots were worth less than the room |
| `--resident-budget-gib 12` | `--mmap-experts`, no RAM tier | Setup's low-RAM mode mapped all 47 GiB of experts: Windows' file cache took the RAM down to 64 MB free and re-read what it evicted. A 12 GiB tier decodes 11% faster than 10 GiB (94.9 against 106.6 ms a token), 0.01 GB/h to the page file |
| Layer-major prompt reads (`STRATA_LM_HIP=1`, `STRATA_PREFILL_LAYER_MAJOR=1`, `_LM_TOKENS=32768`, `_LM_R=host`) | off (it is opt-in on AMD) | The chunk-major path streams every expert again for every chunk: ~37 GB from the SSD per 256-token chunk. Layer-major streams them once per super-chunk (a 4K prompt is one, a 24K prompt five) |
| `--prefill 512` | auto | A fixed chunk keeps the freed VRAM for the super-chunk (about 100 KB a token); `--prefill auto` (with `STRATA_PREFILL_AUTO_MAX`) grew the chunk into it and turned layer-major off for lack of room. 512 read a 4K prompt at 127 tok/s, 256 at 116 |
| `--kv-resident 8192` | 32768 | ~290 MiB more VRAM for the super-chunk; QSA attention is sparse and the rest of the K/V sits in pinned RAM |
| `STRATA_DENSE_MMQ=1` | off | The prompt's dense projections on MMQ: rocBLAS on RDNA2 has tuned kernels only for FP16-in / FP16-out GEMMs. GDN input projections 90 -> 61 s on the paged card |
| The fork's Windows switches (the first 14 `env` entries) | not set | The engine sets them by itself only on NVIDIA builds (`apply_fork_defaults`: `#if ... !defined(STRATA_USE_HIP)`). Each one took effect here (the engine log names them); they were not measured one by one |
| `STRATA_EXPERTS_MIRROR`, `_SHARE=0.57` | not set | A byte-identical copy of `experts.bin` on the second SSD (SHA-256 checked). Unbuffered random reads, 2 MiB x 4 threads: D: 716 MB/s, C: 1,093; 1 MiB x 8: 1,012 and 1,213. 0.57 is C:'s share of the sum |
| `--conversation-cache-mib 0` | auto | As the fork's 32 GB setup: it parked 237 MB per request |
| DFlash2, `--draft-conf 0.8` | DFlash2, 0.35 | 0.8 is the drafting rule's step for a card with few expert slots |

## Quality

### The prompt-path changes

[`tools/prompt_path_kl.py`](../../tools/prompt_path_kl.py): 4,000 tokens of the quant-quality set read through the prompt
path under test, then 1,000 more teacher-forced through the decode windows, one GPU expert slot in every arm. KL against
the chunk-major path with 512-token chunks:

| Arm | Same top-1 | Mean KL | p99 | Perplexity (reference 4.687) |
|---|---:|---:|---:|---:|
| Chunk-major, 256-token chunks | 100% | 0.00000 | 0 | 4.687 |
| FP32 SGEMM (`STRATA_RDNA2_SGEMM=1`: exact inputs, more precise than the reference) | 94.4% | 0.0623 | 1.25 | 4.693 |
| Layer-major + dense MMQ (this config) | 93.5% | 0.0647 | 0.83 | 4.571 |
| Layer-major | 93.9% | 0.0662 | 1.11 | 4.760 |
| Dense MMQ | 94.2% | 0.0620 | 1.06 | 4.497 |

Against the SGEMM run, the chunk-major path is 0.0673 away and layer-major + MMQ 0.0684. Any valid change of rounding moves
this model by about 0.06 over 4,000 tokens, so the config's prompt path is as close to exact arithmetic as the stock one.

### Expert skip

`tools/quant_quality.py` (5,000 tokens through the decode windows, one GPU expert slot, so a skip fires as often as it
can), against skip 0:

| Arm | Same top-1 | Mean KL | p99 | Perplexity (reference 4.125) |
|---|---:|---:|---:|---:|
| `STRATA_IQ_MT_MIN=1` (another valid grouping of the CPU kernels: the floor) | 94.9% | 0.0296 | 0.53 | 4.049 |
| `STRATA_ROUTE_TAIL_SKIP=7` | 89.8% | 0.0939 | 1.26 | 4.126 |

Skip 7 per segment: code 0.033, docs 0.128, English 0.057, other languages 0.029, the agent transcript 0.224.

## Where a decode step goes

`STRATA_DECODE_TIMING=1`, this config, ~167 ms a step for ~1.7 tokens:

| Part | ms | |
|---|---:|---|
| CPU experts | ~82 | ~12 experts a layer on 5 workers and the host thread, ~131 us each. The pool's counters (CLI run): gate/up 58 ms at ~11 GB/s (compute-bound IQ decode), the IQ4_NL down rows 23.5 ms at ~23 GB/s |
| Waiting for SSD reads | ~44 | ~1.9 experts a layer are read from the file tier |
| GPU (dense layers, cached experts) | ~25 | |
| DFlash2 draft | ~8 | |

They run one after another within a layer, and a wider step costs about as much more as it brings: the tokens of a step
share few experts. So a token costs about 100 ms whatever the drafter does (the MTP layer: 2.06 tokens a step against
1.69, 210 ms against 170).

## Measured, no gain (decode against the same base, one run each)

The first five rows ran with a 10 GiB tier and `--prefill 256` (base 106.6 ms a token), the rest with this config
(base 97-101).

| Change | ms a token | Its base | Note |
|---|---:|---:|---|
| `STRATA_PCIE_FRAC_DEFAULT=auto` (0.19 -> 0.30) | 105.0 | 106.6 | the experts read from the file stay on the CPU |
| `STRATA_POOL_FUSED=1` | 106.0 | 106.6 | |
| `STRATA_DIRECT_AHEAD=2 STRATA_LOOKAHEAD_K=5` | 115.8 | 106.6 | competes for the SSD |
| `--draft-conf 0.65` | 102.5 | 106.6 | |
| 12 GiB tier (adopted) | 94.9 | 106.6 | |
| MTP drafter (`--spec 4 --spec-min-p 0.5`) | 101.7 | 100.9 | 0 MiB of VRAM left |
| `--pool-workers 11` (SMT siblings) / `8` | 146.7 / 100.3 | 96.6 | |
| `STRATA_ROUTE_TAIL_SKIP_FILE=5` / `7` | 98.8 / 97.3 | 96.8 | does not fire on this read path |
| `STRATA_UNBUFFERED_LOAD=0` | 168.9 | 96.6 | 11.3 GB/h to the page file, 36 MB free: the read pages stay in the working set |
| `STRATA_HIP_PROMPT_F16=1` (prompt) | | +5% on a 4K prompt; changes numerics; one crash, under VRAM paging |

## Pending Optimization Tests

**Neither switch below has been benchmarked or verified on the RX 6600.** Both come from reading the code and comparing
this config with the fork's RTX 4070 Ti setup (2026-11-08). That is unconfirmed static analysis, not a measurement.
They are **not** in [the config](#the-config) above and should not be added to it until they are tested.

| Switch | What this config does today | Why it is promising |
|---|---|---|
| `STRATA_CPU_IQ_ALL_EXACT1=1` | Off. The engine turns it on by itself only on NVIDIA sm_86 and sm_89 cards (`src/core/nv_defaults.cpp`), never on a HIP build. A lone token's IQ3_S gate/up rows run on ggml-cpu's single-token dot | Strata's own AVX2 rows for one token, in ggml's float order (`src/kernels/cpu/native_expert.cpp`). They are meant to give the same bits as ggml's AVX2 dot. The i5-10400 meets the gate (ggml built for AVX2, no AVX-512). CPU experts are about 82 ms of the ~167 ms step above, so a faster CPU path counts here. On other PCs: -11% to -60% time per CPU expert, +0.6% (RTX 3060) and +3.3% (RTX A4000) end to end. What it gives here is unknown |
| `STRATA_STAGER_BATCH=1` | Unset. With the pack's file tier the prompt stager takes 8 experts per claim (`src/prefill/prefill.cpp`, `Prefill::init`). With `STRATA_STAGER_IN_PLACE=1` a stager thread reads those 8 one after another and marks none ready until the last one lands | One expert per claim: each one goes to the GPU as soon as it lands. On the RTX 4070 Ti PC, the in-place stager's gain (prompts +8.5% to +13.9%, short prompts about 21% less time, answers byte-identical) was measured with this switch at 1, next to the 3 threads and 128-deep ring this config already uses. The same bytes go to the same places in the same order, so answers should not change. Layer-major reads use the same stager, but how much it helps them is unknown |

Before either goes into the config:

1. **Same answers:** identical-output settings at temperature 0, before and after. The answers must match byte for byte.
   First run the "before" arm twice: without `STRATA_IQ_MT_MIN=1`, temperature 0 was not repeatable on the RTX 4070 Ti
   PC.
2. **Speed, one switch at a time:** `STRATA_CPU_IQ_ALL_EXACT1` on the decode questions above (ms a token);
   `STRATA_STAGER_BATCH` on the 4K and 24K prompts. At least 2 runs per arm, in alternating order (this page sees
   about 5% between runs of the same settings).
3. **Check it took effect:** the log prints `strata ab: GATE=STRATA_CPU_IQ_ALL_EXACT1 active=1` once.
   `STRATA_STAGER_BATCH` prints nothing.

Keep a switch only if the answers are identical and it is not slower.

## Limits

- One PC, one card, one run per arm for most rows; decode moves about 5% between runs of the same settings.
- The 14 Windows switches were applied together; none was measured alone on AMD.
- Layer-major prompts on HIP are opt-in upstream ("only run on NVIDIA CUDA"); here they ran 4K and 24K prompts without
  an error, the KL above is for 4K.
- An exact 4-bit repack of the RAM tier's gate/up rows (IQ3_S, IQ3_XXS and IQ2_S have 8 or fewer magnitudes) was
  estimated from the pool counters at +5-8%, not built.
