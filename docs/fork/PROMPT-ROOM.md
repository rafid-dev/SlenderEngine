# The prompt room: long prompts in fewer passes

Since 2026-11-09 the fork reads long prompts about 10-20% faster on a 12 GB card, with the same answers up to rounding. It does this by
borrowing VRAM that sits unused while a prompt is read. This page explains what it does, how it was found, and how it was checked.

Measured on: i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB, two 990 PRO 1 TB, Windows 11. Qwen3.8-Flash-Next IQ3_S, `--kv int8`,
262K context, 10 GiB RAM tier, `--prefill auto:16384`.

## The short version

| Drafter | 50K prompt before | 50K prompt with the room | Gain |
|---|---:|---:|---:|
| DFlash2, the fork's defaults (ring 96, lend cap 90%) | 1,910 / 1,889 tok/s | 2,145 / 2,064 tok/s | +10% |
| DFlash2, ring 192 and lend cap 95% | 2,045 / 1,963 / 2,010 tok/s | 2,253 / 2,214 / 2,225 tok/s | +11% |
| MTP, the fork's defaults | 1,706 tok/s | 1,880 tok/s | +10% |
| MTP, ring 192 and lend cap 95% | 1,842 / 1,813 tok/s | 2,197 / 2,173 tok/s | +19.5% |

A 29K prompt with DFlash2 went from about 1,960 to about 2,230 tok/s (+14%). Prompts under 4,096 new tokens don't use the room and are
unchanged. Decode speed is unchanged.

It is on by default with the fork's pack defaults (Windows, NVIDIA, `--pack`) when the engine serves (`--serve`):

- `--draft dflash2` turns on `STRATA_DF2_ROOM_MIN=4096`, `STRATA_DF2_ROOM_CHUNK=1`, `STRATA_DF2_ROOM_HEAD=1`.
- `--mtp` turns on `STRATA_MTP_ROOM_MIN=4096`, `STRATA_MTP_ROOM_CHUNK=1`, `STRATA_MTP_ROOM_HEAD=1`. With `--batch` it is left out.

A variable you set yourself wins. To turn the room off, set the `..._ROOM_MIN` variable to a number no prompt reaches, for example
`STRATA_DF2_ROOM_MIN=999999999`. `STRATA_FORK_DEFAULTS=0` turns off every fork default.

## Why prompts read in passes

The model has 24,576 experts (48 layers of 512), about 37 GiB on this quant. Most of them don't fit in VRAM. A prompt is read in
chunks: for each chunk, every expert that isn't already in VRAM is streamed from RAM or the SSD and used for all the chunk's tokens.
So the number of chunks is the number of passes over the experts, and each pass costs about 3 seconds of drive and RAM reading here.

A 50K prompt in chunks of 8,448 tokens takes 6 passes. In chunks of 10,240 it takes 5. Fewer passes means less reading.

The chunk size is limited by VRAM. The chunk's working buffers (activations, attention scratch, the staging ring for streamed experts)
have to fit somewhere. In this fork they live in a "loan": while a prompt is read, the expert cache lends its last slots to the prompt,
and they are refilled after it. The loan's size is fixed when the engine starts, and it decides the largest chunk.

## The idea

While a long prompt is read, three things in VRAM are not used at all:

| What | Size here | Used by |
|---|---:|---|
| The drafter's weights: the DFlash2 pack (or the MTP layer's 512 experts) | 323 MiB (MTP: 675 MiB) | Drafting, during decode |
| The DFlash2 draft head (its own copy of part of the output head) | 81 MiB | Drafting, during decode |
| The output head (`output.weight`) | 497 MiB | Turning the last state into token scores: decode and the last few prompt tokens |

The prompt's chunk passes never read them. So during a long prompt, the chunk's buffers can continue from the loan into these blocks.
That's about 900 MiB more, enough for a chunk of 10,240 tokens at the default 32K K/V window.

Nothing is freed or moved. The blocks stay at the same addresses the whole time. Before anything needs them again, their contents are
written back in place:

- The DFlash2 pack and draft head come back from a copy in RAM, made once at the first long prompt (pageable, about 400 MiB).
- The MTP experts come back from a pinned copy in RAM (that is how the existing `STRATA_MTP_ROOM_MIN` already worked).
- The output head is read again from the model file, straight from the SSD into the same address (about 130-150 ms). No RAM copy is
  kept for it.

Because the addresses never change, the decode's captured CUDA graphs stay valid and nothing has to be rebuilt.

## What happens, step by step

At start:

1. The engine works out the largest chunk whose buffers fit in the loan plus the borrowed blocks, in steps of 256 tokens, with a margin
   of 64 MiB for each block (a buffer can't straddle two blocks). It prints it: `a roomed prompt reads in chunks of up to 10240 tokens`.
2. The prompt path's host-side buffers are sized for that chunk once.

For each request that reads 4,096 new tokens or more:

1. The room opens. The engine waits until the drafter and the head are idle, makes the drafter's RAM copy if it doesn't have one yet,
   and marks the blocks as lent. While lent, the drafter and the output head refuse to run, so a mistake shows as an error, not as wrong
   text.
2. The drafter's per-token inputs for the prompt (its tap rows) go to RAM instead of the drafter, because the drafter's weights are
   lent.
3. The prompt path lays out its buffers for the large chunk: the loan first, then the borrowed blocks, largest first. If the layout
   doesn't fit after all, it tries 256 tokens less, until it fits.
4. The prompt is read in the large chunks.
5. The room closes before the first verify window (the last few prompt tokens are read through windows, which use the output head),
   or when the prompt is done. The prompt path's streams go idle, the weights are written back, the output head is re-read from the
   file, and the held tap rows are handed to the drafter in order.

Any other read is laid out in the loan alone, again if the last layout used the room. A room left open by a cancelled request is closed
at the start of the next one.

## How it was found

The goal was 50K prompts in 5 passes, which needs a chunk of at least 10,010 tokens. Each step was measured before the next one.

1. **Where the time goes.** The prompt path's GPU timeline (`STRATA_TRACE=1`) showed that a large share of a 50K prompt is waiting for
   streamed experts and turning them into GEMM inputs. Both scale with the number of passes.
2. **Ordinary knobs.** A smaller K/V window (`--kv-resident 16384`, then `8192`) and a lend cap of 100% gave chunks up to 9,728: still
   6 passes. Forcing `--prefill 10240` made the engine fall back to a much slower layout (1,335 tok/s). The VRAM wasn't there.
3. **The RX 6600 branch's drafter parking** (`STRATA_DF2_PARK`) frees the drafter's VRAM during a prompt, but by freeing it. In this
   fork the prompt buffers live in the cache's loan, so freed VRAM outside the cache couldn't be used without redesigning the cache.
   Lending the drafter's block in place was simpler.
4. **Room as staging slots** (`STRATA_DF2_ROOM_MIN` alone). The pack became extra slots of the staging ring, so the ring could leave the
   loan and the chunk could grow. It grew by about 512 tokens, still 6 passes, and the ring got shallower: no gain.
5. **Room for the chunk's buffers** (`STRATA_DF2_ROOM_CHUNK=1`). With the K/V window at 8K and a 96-slot ring this reached 10,240 tokens
   and 5 passes. Drive work fell by 25% (host staging 17.6 → 13.3 s), but the shallower ring made the GPU wait longer for copies, and
   the speed stayed the same. So 5 passes only pay with the deeper ring, which needed about 200 MB more.
6. **The heads** (`STRATA_DF2_ROOM_HEAD=1`). The output head and the draft head were the remaining idle VRAM. With them, the chunk
   reached 10,240 at the default 32K K/V window with the 192-slot ring: +11%.
7. **A bug found on the way.** In the first runs, the first verify window ran before the room closed and read the prompt's buffers
   instead of the output head. The engine's own check caught the non-finite scores and retried the request, so no wrong text came out.
   The fix closes the room before the first window. All numbers above are from runs after the fix.
8. **MTP.** The fork already had `STRATA_MTP_ROOM_MIN` (MTP experts as staging slots, no gain). The same two steps were added for it:
   `STRATA_MTP_ROOM_CHUNK` and `STRATA_MTP_ROOM_HEAD`. MTP gains more because its drafter is larger, so its chunk was smaller to start.

## How it was checked

- **The head comes back exactly.** `STRATA_DF2_ROOM_VERIFY=1` takes a checksum of the output head before the first lend and after every
  reload. It matched in every run, with DFlash2 and with MTP.
- **Answers and scores.** First-token KL against the same build without the room, full vocabulary, identical-output settings
  (temperature 0), prompts of 3K, 29K and 50K, two runs of each: 0 at 3K (no room), 0.0022 at 29K, 0.0003 at 50K, the same top token
  every time. Two runs without the room gave exactly 0, and two runs with it gave identical results, so the difference is the chunk
  size's rounding, not noise. That is the same size as the fork's earlier "6 passes instead of 7" change (0.0001-0.0024). The 64-token
  answers were identical at 3K and reworded after about 40 tokens at 29K and 50K, with the same meaning.
- **Speed.** Alternating runs, a fresh engine each time, 2-3 runs per setting; every room run was faster than every run without it.
- **Defaults.** With nothing set by hand: DFlash2 and MTP each turn on their own room, and with `--batch` the MTP room is left out
  and the engine starts normally.

## Limits

- One GPU and `--serve` only. Not with a layer split, `--prefill-head-alias` or (MTP) `--batch`.
- It needs the native output head in VRAM, read from the model file. If the head is elsewhere, `..._ROOM_HEAD` steps aside and only the
  drafter's block is borrowed.
- It costs RAM: about 400 MiB pageable for the DFlash2 copy, 675 MiB pinned for the MTP copy. On a PC that is short of RAM, test it.
- The chunk changes rounding, so long prompts' answers can differ in wording from a build without it (KL above).
- On other cards the gain depends on how close the prompt was to one pass fewer. The engine prints both chunk sizes at start.

## Also on by default since 2026-11-09: the fused hyper-connection read

`STRATA_HC_UPMIX=1` runs the prompt path's hyper-connection up GEMM and mix as one kernel on chunks of 33 tokens or more (NVIDIA sm_80
and newer; older cards keep the two kernels). Here: 50K 2,183 / 2,182 → 2,258 / 2,296 tok/s (+4.3%). Logits and answers were
byte-identical (same file hash at 3K, 29K and 50K), and `STRATA_HC_UPMIX_CHECK=200` found 96 of 96 checked reads exact. Decode
doesn't use it. `STRATA_HC_UPMIX=0` turns it off.

## Code

- `src/core/dflash2.cpp`: `Dflash2Drafter::room_lend` / `room_reclaim` (the pack and the draft head lent in place).
- `src/core/native_head.cpp`: `NativeHead::room_lend` / `room_reclaim` (the output head re-read from the file; `run()` refuses while
  lent), `STRATA_DF2_ROOM_VERIFY`.
- `src/prefill/prefill.cpp`: `Prefill::relayout` with up to 4 borrowed regions after the loan; `set_max_chunk`.
- `src/program/generate.cpp`: the start-up chunk, opening and closing the room per request, and the fork defaults
  (`apply_fork_defaults`).
- Full test log: [WORKLOG.md](WORKLOG.md), "Prompt room".
