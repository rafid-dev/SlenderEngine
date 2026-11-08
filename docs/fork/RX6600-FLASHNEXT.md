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
| Decode, short questions | 5.3 tok/s | **10.4-10.7 tok/s** (9.2-10.5 without `EXACT1`) | 13.5 tok/s (without `EXACT1`) |
| 4K-token prompt | 14 tok/s | **about 185 tok/s** (127 before the prompt work below) | |
| 24K-token prompt | about 7.5 tok/s (stopped after 443 s) | **292-296 tok/s** (76 s; 142-176 before the prompt work below) | |
| Decode after the 24K prompt | | 9.7-10.2 tok/s | |
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
  "STRATA_PREFILL_LM_R": "host", "STRATA_DENSE_MMQ": "1", "STRATA_CPU_IQ_ALL_EXACT1": "1",
  "STRATA_HIP_PROMPT_F16": "1", "STRATA_ROCBLAS_TUNING": "<repo>\tools\hip", "STRATA_DF2_PARK": "8192"
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
| `STRATA_CPU_IQ_ALL_EXACT1=1` | not set | Decode 12% faster with the same output bit for bit, see [below](#two-switches-from-the-rtx-4070-ti-setup-tested-2026-11-08). Like the 14 Windows switches, the engine sets it by itself only on NVIDIA |
| `STRATA_HIP_PROMPT_F16=1`, `STRATA_ROCBLAS_TUNING`, `STRATA_DF2_PARK=8192` | not set | Long prompts 142-176 -> 292-296 tok/s, see [Long prompts](#long-prompts-2026-11-08) |
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

## Two switches from the RTX 4070 Ti setup, tested 2026-11-08

Both were found by comparing this config with the fork's RTX 4070 Ti setup. `STRATA_CPU_IQ_ALL_EXACT1=1` is now in
[the config](#the-config). `STRATA_STAGER_BATCH=1` gave nothing here, and is left out.

| Switch | What it does | Here |
|---|---|---|
| `STRATA_CPU_IQ_ALL_EXACT1=1` | When one token uses an IQ3_S expert on the CPU, Strata's own AVX2 rows run for it instead of ggml-cpu's single-token dot, in ggml's float order (`src/kernels/cpu/native_expert.cpp`). The engine sets it by itself only on NVIDIA sm_86 and sm_89 cards (`src/core/nv_defaults.cpp`) | **Decode 12% faster, the same output bit for bit.** The engine log prints `strata ab: GATE=STRATA_CPU_IQ_ALL_EXACT1 active=1` |
| `STRATA_STAGER_BATCH=1` | The prompt stager claims one expert at a time, not 8 (`src/prefill/prefill.cpp`) | The same output bit for bit, no change in speed |

### Same output

The answers at temperature 0 can't show this on this PC: two runs of the unchanged config already differed about 100
characters into the first answer, as the 4070 Ti PC also found without `STRATA_IQ_MT_MIN=1`. So the check used
[`tools/prompt_path_kl.py`](../../tools/prompt_path_kl.py)'s method instead, which repeats exactly. Each arm read 4,000
tokens through this config's prompt path, then 1,000 more teacher-forced through the decode windows. Each arm's
log-probabilities were byte-identical to the unchanged config's (KL 0, the same top-1 at all 999 positions), and to
the same run from the day before.

### Speed

Two runs per arm, in alternating order, with a fresh engine per run:

| | Base | With the switch |
|---|---:|---:|
| `EXACT1`: decode, the mean of the 3 answers | 9.67, 9.19 tok/s | **10.36, 10.74 tok/s** |
| `EXACT1`: ms per token, the four 600-token A-level answers | 97.2 | **91.2** |
| `STAGER_BATCH`: 4K prompt (3,821 tokens) | 128, 125 tok/s | 121, 129 tok/s |
| `STAGER_BATCH`: 24K prompt (22,376 tokens) | 176, 176 tok/s | 176, 176 tok/s |

The page file got 0.00-0.50 GB/h in every run.

The 24K prompt ran at 176 tok/s here, against 142 tok/s the day before with the same settings. This time the card had
437 MiB free before the prompt, against 231 MiB. So the prompt fit in 2 super-chunks of up to 11,264 tokens instead of
5 of up to 4,608: 96 GB of expert blobs read from the SSD instead of 229. How much VRAM Windows leaves free decides
this, and it changes from day to day.

## Long prompts, 2026-11-08

A 22,376-token prompt (the same text every run), thinking on, a fresh engine per run, two runs or more per row:

| Step (each adds to the row above) | 24K prompt | Quality |
|---|---:|---|
| The config above without these three switches | 142-176 tok/s | |
| `STRATA_HIP_PROMPT_F16=1`: the prompt's 16-bit GEMMs FP16 in and out | 165-169 tok/s | KL 0.0626 against the chunk-major path, where an FP32 SGEMM run is 0.0623 |
| A rocBLAS solution table for gfx1032 (`tools/hip/gfx1032-rocblas-5.8.0.efafc1c1.txt`, made with `tools/hip/tune_rocblas` on this card; `STRATA_ROCBLAS_TUNING`) | 179-181 tok/s | KL 0.0730 against SGEMM (the untuned FP16 route 0.0631, the default path 0.0684). The same FP16 in, FP32 accumulate, other kernels |
| `STRATA_DF2_PARK=8192` (new, below) | **292-296 tok/s** | The drafter's weights come back byte for byte; decode after the prompt 9.7-10.2 tok/s |

What the rocBLAS table changes, on this card: rocBLAS's own pick for the GDN input projection (N 10240, K 2560) runs at
4.1 / 8.8 / 18.0 ms at 256 / 512 / 1,024 tokens, the best of its solutions at 1.0 / 2.0 / 3.9 ms. In the prompt's GPU
timeline the GDN projections went 29.1 -> 14.7 s and the hyper-connection read 12.0 -> 7.2 s.

**`STRATA_DF2_PARK=<tokens>` (engine, opt-in).** Free VRAM decides how many super-chunks a long prompt is read in, and
each one streams every expert again (~10 s each for 24K here). The DFlash2 drafter's pack and draft head (537 MiB) do
nothing while a prompt is read except take the prompt's last 2,047 rows at the end. With the switch, a prompt of at least
that many tokens is read with them in RAM (`Dflash2Drafter::park`, about 0.1 s each way); the prompt's tap rows are held
in RAM and handed to the drafter's usual `context()` once it is back, and the round graphs are captured again if an
address moved. The 24K prompt then fits one super-chunk (984-1,238 MiB free instead of 225-437): 49 GB of expert blobs
streamed instead of 142-272. With the original `--expert-cache 60` and `--kv-resident 8192`, 295 / 295 tok/s; with 20 and
4096, 292-296 and decode 3% slower, so the config keeps 60 and 8192.

**Pinned RAM step-down (engine, default path).** A super-chunk whose residual rows do not fit the pinned RAM now halves
(in whole chunks) instead of turning layer-major off for that prompt. Before, the first parked 24K prompt read at 41
tok/s through the chunk-major path for this reason. A prompt whose pinned buffers fit is read exactly as before.

**`STRATA_PF_W16_LAYER_MIB=<MiB>` (engine, opt-in, not in the config).** With `STRATA_DENSE_MMQ=0` the dense projections
dequantize each weight to FP16 for every 512-token chunk (no FP16 weight cache on HIP). This arena keeps one layer's
FP16 weights, so each is dequantized once per layer. Measured: 367 and 321 tok/s (GDN projections 14.9 -> 6.9 s, GDN out
7.2 -> 2.9 s, attention projections 5.9 -> 2.3 s), KL 0.0752 against SGEMM, but 4 of 7 runs had the driver page GPU
memory (the memory-bound phases 3-5x slower, 136-150 tok/s), at VRAM margins of 96, 320 and 600 MiB. Not safe on 8 GB yet.

Tried, no gain: `STRATA_DENSE_MMQ=0` alone (GDN projections 14.9 -> 17.4 s: each chunk dequantizes again),
`--prefill 256` (120-124 tok/s), `STRATA_STAGER_BATCH=1` (176 / 176 tok/s, the same output), no drafter for decode
(8.80 against 10.13-10.36 tok/s).

**Windows page file.** The RAM tier is sized from free commit (RAM + page file) less 4 GiB. With a system-managed page
file, which is 2 GB after a reboot, the tier came out 7.7-8.4 GiB instead of 12 and decode 8.0 instead of 10.7-10.9
tok/s. A fixed 16-32 GB page file on C: (commit limit 47.9 GiB) gives the full tier. It reserves room on the disk; the
page file writes stayed at 0-0.66 GB/h.

## Limits

- One PC, one card, one run per arm for most rows; decode moves about 5% between runs of the same settings.
- The 14 Windows switches were applied together; none was measured alone on AMD.
- Layer-major prompts on HIP are opt-in upstream ("only run on NVIDIA CUDA"); here they ran 4K and 24K prompts without
  an error, the KL above is for 4K.
- An exact 4-bit repack of the RAM tier's gate/up rows (IQ3_S, IQ3_XXS and IQ2_S have 8 or fewer magnitudes) was
  estimated from the pool counters at +5-8%, not built.
