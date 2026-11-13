# Q-STD: the quality test every speed change goes through

Every speed change in this fork is checked the same way before it is adopted: does it change what the model computes, and if
so, by more than harmless rounding? Q-STD answers that with one verdict per change. The tool is in [`tools/qstd`](../../tools/qstd).

Status (2026-11-13): Phase 1 is built and checked on a few cases. The calibration has not been run yet, so no verdict has been
issued with it. The thresholds below are provisional until that calibration.

## What a verdict means

| Verdict | Meaning |
|---|---|
| **PASS-EXACT** | The logits are byte-identical to the accepted setup at every scored position. Nothing changed but speed |
| **PASS-NUMERICAL** | Not identical, but within the calibrated rounding envelope on every gate. A numerical pass only: the task-correctness tier (Phase 2) does not exist yet |
| **HOLD** | Unclear, uncalibrated, or a decision for the owner (any weight or precision change, any CPU/GPU placement change). Not adopted automatically |
| **FAIL** | A broken run, a broken exactness promise, or a confirmed regression |

A verdict means "passed this regression standard", not "proved that no quality was lost".

## What is compared

The reference (**R**) is the accepted setup exactly as the daily launcher runs it: the engine binary, the launcher's
`STRATA_*` settings (read from the launcher file itself) and the server config's arguments. Every candidate is compared
with R position by position, on the same text, with full-vocabulary log probabilities (FP64, over the valid vocabulary).
Tests use identical-output settings (`STRATA_IQ_MT_MIN=1`, no prompt CPU share, the adaptive tier frozen), so a repeat of R
(**R2**) must be byte-identical; if it is not, nothing is reproducible and the run fails.

Two lanes, both required:

- **Production replay** (`sol` lane): 48 chat cases with thinking (coding, reasoning, instructions/structured output,
  document questions; 32 short, 8 at 12K, 8 at 45-50K tokens). Each prompt is read through the prompt path, then the model's
  own 256-token continuation through verify windows (teacher-forced): 12,288 scored positions. The corpus is local (built
  from a 2026-11-07 collection by the IQ3_S model at temperature 1.0); its long cases put unrelated text before short
  questions, so they test numerical sensitivity to long inputs, not long-document reasoning.
- **BF16 lane**: 13 cases of the published BF16 reference set
  [jyohukuchan/qwen38-flash-next-bf16-logits](https://huggingface.co/datasets/jyohukuchan/qwen38-flash-next-bf16-logits)
  (Qwen Community License 1.0): EN/JA/ZH prose, Rust/Python/SQL code, arithmetic, an agent-style instruction, two
  4,097-token texts and two WikiText-2 rows; ~15,500 positions, every one read through verify windows and compared with both
  R and the full-precision model's own logits (exact). It also serves as the independent (not model-written) text for the
  perplexity check. Fetch the subset with `tools/qstd/fetch_bf16.py`.

## The controls

A change that is not byte-identical is judged against changes known to be harmless:

- **P1-P3, prompt-chunk regroupings** (staging ring 448, ring 384, chunk cap 16,384): the same GPU kernels, the prompt cut
  into other-sized chunks. These are the rounding controls; their spread is the envelope.
- **S1, window size** (`--spec 4`): with placement pinned, the verify windows' size does not change a single bit. It is kept
  as an exactness sentinel, not as a control: a decode-path change that is not byte-identical has no calibrated envelope.
- **D1-D3, X1-X2, placement stress** (expert-cache budget a little below/above the reference's): reported, never used as a
  rounding control. See the CPU/GPU finding below.

Arms other than R run with R's placement pinned. An explicit `--expert-cache N` is a budget in largest-blob units that the
engine packs into more, smaller slots, so the pin passes R's *auto budget* (on the reference PC 1,716, which packs to 2,214
slots), and the slot count is checked after the run.

## Gates (provisional until calibration)

| Change kind | Judged on |
|---|---|
| exact | Byte-identical on both lanes, else FAIL |
| prompt reordering | Mean KL, KL99, the per-case worst-1% mean and sampler TV99, each with the upper 95% bound of the candidate's excess over 1.25x the median of P1-P3 at most 0 (overall and on the medium/long cases); max KL within the controls' envelope; no flip where R gave its top token >= 0.90; independent-text NLL at most +0.1% overall and +0.2% per domain |
| decode, not exact | HOLD: uncalibrated |
| CPU/GPU placement | The BF16 lane with zero margin: the change is "not farther from BF16" only if the upper 95% bound of d = KL(BF16‖C) - KL(BF16‖R) is at most 0. Never auto-passed before the correctness set and the deployment panel exist |
| weights / precision | Always the owner's decision (HOLD), every number reported |

Sampler TV is the total variation between the model card's actual sampler distributions (temperature 0.6, top-k 20, top-p
0.95) of R and the candidate: whether the token that would be drawn changes. Bootstraps resample whole cases.

## Findings so far

- **CPU and GPU experts differ in precision.** VRAM-resident experts take their activations as `q8_1` (one scale per 32
  values; `src/kernels/cuda/iq_kernels.cu`); CPU experts quantize to ggml's `vec_dot_type` for IQ3_S, `Q8_K` (one scale per
  256 values; `src/kernels/cpu/native_expert.cpp`). Moving experts between the tiers is therefore a small precision change,
  not a reordering: 22 fewer slots moved the BF16-lane output by mean KL 0.022 (worst position 4.59) on three small cases,
  against 0.0005 for a prompt-chunk control. The adaptive tier moves experts between the tiers all the time, so in daily use
  outputs already depend on placement history at about this level.
- **Window size is exact** (above).
- **IQ3_S against BF16**: mean KL 0.120, top-1 85.7% on three small cases (preliminary; the published EXL3 3.05 bpw quant
  measures 0.129 on all 21 cases).

## Storage: nothing large, nothing accumulating

The engine's logits (1 MB a position) stream through a named pipe and are scored as they arrive; they never reach the disk.
The reference is kept compact and replaced, never accumulated, when a new setup is adopted (well under 1 GB):

- per position, the top 2,048 tokens' log probabilities plus all other tokens as one tail bucket (log-sum-exp), R's top-1,
  the sampler's distribution and a SHA-256 of the raw row;
- **audit rows**, 8 a case (4 evenly spaced, 4 with the largest tail mass), with the full log-probability vector (FP64 head,
  FP16 rest).

The compact KL is a lower bound of the exact KL (chain rule; the missing part is not bounded by the tail mass), read
~0.2% low on average and up to ~20% low on flat rows in a check with real rows. On the audit rows every candidate gets the
exact KL as well; if the compact value misses more than 10% there, the verdict is HOLD. Reports say "compact + audit".
Byte identity, top-1, flips, sampler TV, the supplied tokens' NLL and everything on the BF16 lane are exact.

## Running it

1. Copy `tools/qstd/qstd.example.json` to a local file and fill in the paths (launcher, server config, the production
   corpus, the BF16 set, a work folder). Pass it with `--config` or `QSTD_CONFIG`.
2. Calibrate once per accepted setup (idle PC, about 2.5-3 h, estimate): `python tools/qstd/calibrate.py cal-<date>`.
3. Check a candidate (about 25-30 min, estimate): write it as `{"name": {"env": {...}, "args": [...], "kind": "prompt"}}`, then
   `python tools/qstd/run.py <run> --lane sol --arms NAME --cand cand.json`, the same with `--lane bf16`, and
   `python tools/qstd/analyze.py NAME --cal cal-<date> --run <run> --kind prompt`. The report is `REPORT-NAME.md` in the run.

## Not yet in place (Phase 2 and 3)

- A 96-problem correctness set (math with verified answers, code with hidden tests, tool calls, long-context retrieval),
  scored on free generation with thinking on.
- A deployment panel for placement changes: a fixed warm-up replayed with the adaptive tier running, then a few sensitive
  probes, so placement policies are judged under realistic placements.
- Corpus v2 (16/16/16 lengths, continuations written in their real context) and a profile for the Qwen3.6-35B-A3B model
  (no BF16 reference exists for it; it is judged against its own accepted build).
