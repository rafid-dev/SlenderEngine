# Tuning on an 8 GB laptop (RTX 4060 Laptop, Ryzen 7 8845HS, 32 GB), 2026-11-03 to 11-04

Part 2, with the Swift-1.5 model, 150K context and the levers that did not pay: [LAPTOP-RTX4060-SWIFT.md](LAPTOP-RTX4060-SWIFT.md).

Qwen3.8-Flash-Next abliterated IQ3_S pack, 262K context, MTP on (`--spec 4 --spec-min-p 0.5`), expert copy on a second
SSD. No working-set or page locking of the RAM tier (`STRATA_RESIDENT_PIN=0`). Measured with `tools/strata_bench.py`
(2 runs, median) and a decode A/B script (300 tokens x 3 prompts, temperature 0, alternating arms).

The new engine switches used here come with this page, all opt-in with the default path unchanged:
`STRATA_POOL_FUSED`, `STRATA_IQ3_VBMI`, `STRATA_LM_CPU_ROWS`, `STRATA_HEAD_FAST_LOAD`, `STRATA_HEAD_ALIAS_SHORT`,
`STRATA_REFILL_PINNED`, `STRATA_EXPERTS_MIRROR_SHARE` and the diagnostic `STRATA_POOL_PHASE_TIMING`. The file-read tail
skip is the speed-fork's own `STRATA_ROUTE_TAIL_SKIP_FILE`. The dead ends below (pageable PCIe share, MWAITX parking and
the rest) were measured on a local build and are not included.

## Result

| | Start (fork as of 11-03) | Tuned |
|---|---|---|
| chat decode tok/s | 17.4-19.5 | 34.9 |
| agent decode tok/s | 23.8 | 35.5 |
| 4K prompt tok/s | 320-343 | 743 |
| 16K prompt tok/s | 332-344 | 847 |
| agent turn first token | ~10 s | 5.3 s |

The chat decode in the bench spreads 25-40 tok/s per question. Run-to-run decode noise on this laptop is about ±5%,
so every A/B below alternates arms over 2-4 rounds.

## What helped, in order of size

| Change | Effect |
|---|---|
| Layer-major prompt path (`STRATA_PREFILL_LAYER_MAJOR=1`, `_LM_TOKENS=32768`, `_LM_FORCE=1`): each expert blob streamed once per super-chunk | prompts 2.2-2.5x |
| `--prefill-head-alias`, head reload from a pinned 32 MiB buffer, slot refill from 8 pinned buffers | per-prompt fixed cost 420 -> 115 ms (head), 380 -> 115 ms (refill) |
| Stage ring 128 x 8 threads, 25% of reads from the second SSD | prompt streaming at the PCIe limit |
| AVX-512 VBMI kernels for IQ3_XXS / IQ3_S expert rows, also for 1-token experts (bit-exact) | decode +4.5% |
| Read-ahead from the next layer's router on this layer's input, K=5 per token (`STRATA_DIRECT_AHEAD=2`, `STRATA_LOOKAHEAD_K=5`) | decode +6-7%. The default K=10 is -12%: 73% of what it reads is never used |
| CPU experts: each expert's down rows start when its own gate/up rows finish, no per-layer barrier (bit-exact, has a bitwise test) | decode +5.6%, CPU time per expert -13% |
| `STRATA_HC_Q8=1` (lossy, ~+0.75% PPL here) | decode +21% on this card, +~200 cache slots |
| Tail skip for experts read from the file, rank 5 (lossy, +0.75% PPL, KL 0.072) | decode +5-8% |

## Where a decode window goes (~65 ms for 2.4 tokens)

- CPU experts ~31 ms (8 threads, ~57 µs per expert). Gate/up is ~75% of it and compute-bound (IQ3_S 3.8 GB/s per thread);
  the IQ4_NL down kernel is at DRAM bandwidth (54 GB/s at 8 threads).
- GPU reach ~19 ms: ~400 µs of serial dense work per layer (attention/GDN projections at ~230 GB/s, the card's ceiling;
  HC GEMVs ~124 µs; router + GDN ~45 µs).
- MTP draft ~5 ms. Cold reads 5-9 ms with a 17.5 GiB RAM tier.

## Measured, no gain

- **Decode PCIe share without pinning** (helper thread, pageable `cudaMemcpyAsync` of the share from the RAM tier): it
  crosses 1.27 experts per layer-window and cuts CPU experts 10.8 -> 9.4 per window, but the GPU waits on the copies
  (+1.7 ms reach). Tier fixed at 9 GiB: control 25.70 / 25.64 / 26.23, bounce 24.64 / 24.99, bounce at 10% share
  25.63 / 25.68 tok/s.
- `--vram-reserve-mib` 1000 / 1375 / 1700 vs 700: 700 is fastest (32.8 / 30.1 / 31.1 / 29.0). The whole GPU expert cache
  is worth only ~12% of decode here.
- `--mtp-q4 head|all`: draft 7 -> 5.7 ms per window, but acceptance 2.37 -> 2.31 tokens per window, net 0.
- `--spec` 3/5, `--spec-min-p` 0.4/0.65: noise; `--spec-split`: -10%. MTP off: -19 to -22%.
- Pool threads on SMT siblings: decode -5%, prompts -15%. `--pool-tasks` 16/40: flat. MWAITX parking: flat.
- AVX-512 IQ2_S kernels: all slower than scalar. AVX-512 IQ4_NL down: no gain, already at DRAM bandwidth.
- HC_Q8 kernel rewrites (L2 prefetch, float4 sums): bit-identical but ~0.1 ms per window; ceiling ~1-1.5%.
- The AMD-path exact switches (`STRATA_TSUM`, `LFUSE`, `MMVF_ROWS`, ...) on this NVIDIA card: flat to -2.5%.
- `STRATA_EXPDEC=1` + `STRATA_HC_UPMIX=1`: -4%. `--adapt-swaps 48`, `--adapt-decay` 0.5/0.85: flat.

## Conversation cache

Two interleaved conversations (2K and 4K tokens, A B A B, 4 turns each), browser open:

| | RAM tier | RAM available | Tokens read again on the switch-back turns | Time |
|---|---|---|---|---|
| `--conversation-cache-mib 2048` | 15.2 GiB | 0.6-1.6 GiB | 18,944 (every park refused) | 45.0 s |
| `--conversation-cache-mib 0` | 14.9 GiB | ~1.6 GiB | 18,945 | 44.4-46.1 s |
| 2048, 9 GiB tier | 9.0 GiB | ~5.9 GiB | 240 | 18.1 s |

The cache reserves nothing up front. It parks a snapshot (225 MiB for a 4K conversation) only when the snapshot plus
the 2,560 MiB floor is available. With a large tier that never happens, so 2048 and 0 behave the same. With a smaller
tier it cut the switch-back time by ~60%. Off is the right default when other apps need the spare RAM; it costs a full
re-read on every switch when RAM is free.

## Browser open

A browser open at start costs this laptop ~7 GiB of RAM tier (17.5 -> 9-11 GiB) and ~25% decode (~34 -> ~26 tok/s).
Freezing it during the load gets the tier back, but once it is thawed it pages its memory back in and Windows trims
the unpinned tier: decode fell to ~23 tok/s.

What works without locking anything: cap the browser instead of freezing it. A hard working-set maximum
(`SetProcessWorkingSetSizeEx` with `QUOTA_LIMITS_HARDWS_MAX_ENABLE`, 256 MiB per process, no admin needed) set on
every browser process while the engine runs, and removed when it exits. The browser keeps working at 0.3-0.6 GB.
Decode with the browser open and capped, 2 alternating rounds:

| `--resident-budget-gib` | Tier | Decode tok/s |
|---|---|---|
| 20 (auto, capped by free RAM) | 16.5-17.4 GiB | 27.4 / 35.1 |
| 16 | 16.0 GiB | 34.8 / 34.6 |
| 15 | 15.0 GiB | 32.8 / 31.5 |

At the automatic size, available RAM falls to ~0.5 GB and Windows trims the engine's own tier, so the speed swings.
16 GiB leaves room and holds ~35. Measured with the browser idle; heavy browsing during a generation was not tested.
