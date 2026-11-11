# 8 GB laptop, part 2: Swift-1.5 IQ3_S, long context and the levers that did not pay (2026-11-05 to 11-11)

Follow-up to [LAPTOP-RTX4060.md](LAPTOP-RTX4060.md). Same PC: RTX 4060 Laptop 8 GB, Ryzen 7 8845HS (Zen 4, AVX-512),
32 GB RAM, Windows 11, laptop in its Quiet power mode. Model: Swift-1.5 Qwen3.8-Flash-Next abliterated IQ3_S (2-file
GGUF) packed with `tools/iq_pack.py`, MTP on (`--spec 4 --spec-min-p 0.7`), `--kv k8v4 --kv-resident 16384`,
`--max-context 262144`, 12 GiB RAM tier, not page-locked (`STRATA_RESIDENT_PIN=0`). The exact default path only:
no tail skip, no HC_Q8, no other lossy switch.

This page is a report. It adds no code. The local-only switches it names are listed at the end; they come in a separate
pull request.

## Result: speed by context length

Fresh prompt each time (random words, nothing reused), then a 256-token answer, through `/v1/chat/completions` with
thinking off and the model card's sampling (temperature 1.0, top-p 0.95, top-k 20). One server, one run per row, 11-11.

| Context | Prompt reading | Time to read it | Answer (256 tokens) | Drafts accepted |
|---|---|---|---|---|
| Short (29 tokens) | - | 1.4 s | 21.1 tok/s | 127 of 174 (73%) |
| 26,138 tokens | 877 tok/s | 29.8 s | 24.5 tok/s | 129 of 186 (69%) |
| 150,026 tokens | 858 tok/s | 174.9 s | 23.3 tok/s | 128 of 184 (70%) |

Prompt reading holds flat from 26K to 150K, and the answer speed does not drop with context.

All speed tables on this page have thinking off. With thinking on (the model's default), answers are longer and the
draft acceptance differs, so these numbers are not comparable with thinking-on runs.

## Run-to-run spread

The same benchmark run earlier the same day, with a fresh server:

| Context | Prompt reading | Answer | Drafts accepted |
|---|---|---|---|
| Short | - | 18.6 tok/s | 129 of 191 (68%) |
| 26,138 tokens | 862 tok/s | 18.9 tok/s | 116 of 155 (75%) |
| 150,026 tokens | 851 tok/s | 19.3 tok/s | 127 of 171 (74%) |

At 26K the later run (first table) accepted fewer drafts (69% vs 75%) and still answered 30% faster (24.5 vs 18.9 tok/s), so
draft acceptance does not explain the gap. Prompt reading moved only 2-3%. On this PC, with RAM nearly full, answer speed
spreads about ±15% between server starts, much more than the ±5% inside one server. Compare decode only within one
server, with alternating arms. For normal text, plan on 19-25 tok/s.

Copy-heavy edit turns are the exception: returning a 120-line file with one change accepts 98-99% of drafts and runs at
26-37 tok/s.

## Levers tested on 11-10

All A/B with alternating arms on one server build, decode = 3 x 300 tokens, edit turns = a code file edited in place
(where `--lookup` drafts from the prompt), quality = `tools/quant_quality.py` teacher-forced at 4,995 positions against
the exact run.

| Lever | Result | Kept? |
|---|---|---|
| `--lookup-max-t 12` | edit turns +12%, output identical | **yes** |
| `STRATA_SPEC_COST` | noise | no |
| `--draft-vocab en` | noise | no |
| `STRATA_HC_UPMIX=1` | noise | no |
| Prefill ring / KV lend for prompts | noise | no |
| `--mtp-hnorm stream` | noise | no |
| Verify deferral to the CPU (local switch, below) | 80% top-1, KL 0.316 vs exact: lossy | no |

Coherence check after the levers ("Make an SVG of cheese on a surfboard", thinking on, no cap): valid SVG, on prompt.

## The n-gram (PLE) table: the IQ3_S file's own vs a Q8_0 table

Swift's IQ3_S GGUF keeps `per_layer_token_embd.weight` in shard 1 (26.8 GiB). Through 11-10 we loaded a Q8_0 table
from another upload instead (`--ple-gguf`, a 54 GB file). Teacher-forced at 4,995 positions against the Q8_0 run:

| `--ple-gguf` | Top-1 agreement | KL | Perplexity |
|---|---|---|---|
| Q8_0 table (reference) | 100% | 0 | 3.999 |
| IQ3_S file, shard 1 | 94.09% | 0.033 | 3.977 |

Decode in the same session, paired: 22.03 (Q8_0) vs 21.93 tok/s (shard 1). So the file's own table costs no speed, saves
the 54 GB download. It is not a rounding-only change: KL 0.033 is about 16 times the 0.002 line, and a little more than
the 8-bit read in [AMD_HIP.md](../AMD_HIP.md) (top-1 95.4%, KL 0.025). Perplexity is no worse. We now run the
shard-1 table.

Two things that tripped us up:

- Pointing `--ple-gguf` at shard 2 fails with "per_layer_token_embd.weight is not in ...00002". The table is in shard 1.
- Shard 2 cannot be deleted even though the pack's `experts.bin` (46.8 GiB) holds every expert from both shards:
  `gguf_split.hpp` refuses to start when any shard of a split is missing, and `NativeDense` reads the dense matrices of
  the later layers (~2 GiB) straight from shard 2. A shard 2 with its `*_exps` tensors replaced by placeholders and its
  split keys kept would likely shrink to about 2 GiB and still pass the split check (suggested in review; not tried yet,
  and the output must be checked against the full shard first).

## Local switches, not in this pull request

Built and measured on this PC, all opt-in with the default path unchanged. They come later as an `[Engine]` pull
request against the current `speed-fork`:

| Switch | What it does | Measured here |
|---|---|---|
| `STRATA_MTP_PARK=N` | a prompt part of at least N tokens is read with the MTP layer's 675 MiB of experts out of VRAM, then reloaded (~0.2 s); the MTP version of `STRATA_DF2_PARK` | 7.4K prompt 600-612 -> 751 tok/s, 14.7K 684-689 -> 820-827, 24.5K 720-723 -> 866-877 (one super-chunk instead of 2-4; 11-08); decode unchanged; byte-identical for the same super-chunk plan |
| `STRATA_KEEP_WARM_S=N` | an idle thread touches the RAM tier every N s so Windows does not trim it | first request after a trim 11 s instead of 22 s |
| `STRATA_DIRECT_READERS=N` | N cold readers for missed experts instead of 4 | decode +2% at N=8 on a 12 GiB tier, flat on 16 GiB |
| `STRATA_IQ512X1=2` | 1-token IQ3_S / IQ2_S experts on the exact 512-bit rows on AMD too (upstream allows them on Intel only) | decode +3%, bit-exact |
| `STRATA_VERIFY_DEFER_CPU` | defers part of the verify step to the CPU | lossy (above), kept off |
| Outside PLE file with a split model | `--ple-gguf` from a 5-shard upload is table-only when the model's own shards hold a table (it was refused for its split keys) | fix |

## Not tested

Other GPUs or CPUs, Performance power mode, a page-locked RAM tier, prompts above 150K, more than one run at 150K.
