# Qwen3.6-35B-A3B on a 12 GB card with 32 GB of RAM

This page is for running [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) on the `src/qwen35` engine path
with a 12 GB NVIDIA card and 32 GB of RAM, at the full 262,144-token context with an 8-bit K/V cache. It lists the
switches that give the best measured speed, what each does and what was measured. The experts don't fit in VRAM on
such a card, so the engine runs in its hybrid mode: some experts in VRAM, the rest computed on the CPU from RAM.

> **Measured on:** RTX 4070 Ti 12 GB, i7-14700KF, 32 GB DDR5-6000, two NVMe SSDs, Windows 11, CUDA 13.4 (sm_89).
> Unsloth's `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (no MTP block), thinking on, the model card's sampling (temperature
> 1.0, top-p 0.95, top-k 20, presence penalty 1.5), server runs. Other cards and PCs will differ.

## The configuration

Engine arguments (the run config's `args`):

```
--native <path>\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --max-context 262144 --prefill auto --mtp 0 --pool-workers 19
```

The config also needs `"arch": "qwen35moe"` and a `tokenizer` folder made by `tools/strata_tokenizer.py --gguf <file>
--out <folder>`. A GGUF without the MTP block loads (the `nextn_predict_layers` key is optional); use `--mtp 0`.

Environment (all off by default; set them before starting the server):

```bat
set STRATA_Q36_KV=q8_0
set STRATA_Q36_KV_RAM=1
set STRATA_Q36_KV_WINDOW=65536
set STRATA_Q36_KV_GROW=1
set STRATA_Q36_HANDOFF=1
set STRATA_Q36_HO_THREAD=1
set STRATA_Q36_HO_GRAPH=1
set STRATA_Q36_REBAL_ASYNC=1
set STRATA_Q36_REBAL_LAG=8
set STRATA_Q36_PROMPT_DECAY=1
set STRATA_Q36_ATT_Q8G=1
set STRATA_Q36_GEMV_ILP=1
set STRATA_Q36_LM=1
set STRATA_Q36_LM_TAIL=1
set STRATA_Q36_FA2=1
set STRATA_Q36_FA2P=1
set STRATA_Q36_FA2_SEG=32768
set STRATA_Q36_ATT_MFLOATS=13
set STRATA_Q36_GDN_TC=1
set STRATA_Q36_GDN_COLS=16
set STRATA_Q36_CKPT=1
```

Optional, for faster long prompts: `STRATA_Q36_CHUNK=4096` with `STRATA_Q36_LM_CHUNKS=2` (50K read +3-6%; the
bigger chunk buffers take expert slots, 57 → 44 a layer).

## Results

Server runs at 262K context with the configuration above, against the first measurements on the same PC (the
engine before this work at 65K, since its f32 cache did not fit 131K; llama.cpp b8abe41a5 with `--fit on`, 131K).

| | Before (65K) | Now (262K) | llama.cpp (131K) |
| --- | ---: | ---: | ---: |
| Short questions, decode | 47.5-48.4 tok/s | 78.7-87.3 | 43-45 |
| 50K-token prompt read | 1,023-1,028 tok/s | 3,924-4,061 (4,178-4,315 with 4096 chunks) | 599 (1,588 with `-ub 2048`) |
| Decode after a 50K prompt | 10.9-11.0 tok/s | ~67-70 | 42 |
| Decode 100K into a conversation (CLI) | - | ~58 | - |
| Follow-up after a 50K prompt | the whole conversation read again (~50 s) | ~1 s | 36 tokens read |
| Page-file writes | 0.6-1.1 GB/h | ~0 (memory pressure aside) | 0.5-4.2 GB/h |
| Ready | 51 s | 24-30 s (the first start writes the blob cache, ~30 s) | 13-16 s |

**Quality.** Every switch either gives the same bits (checked byte for byte with greedy decoding) or stays at
rounding level: first-token KL against the reference at most 0.0007 with the same top-1, where reading the same prompt
in other chunk sizes gives ~0.0005. The 8-bit cache: 0.0007 at 8K, 0.000002 at 30K. The tensor-core prompt attention
0.0001-0.0004, the tensor-core delta-rule scan 0.00002, 4096-token chunks 0.00006. Switches that move experts
between VRAM and the CPU (`REBAL_ASYNC`, `PROMPT_DECAY`, `KV_GROW`) change only where an expert is computed: first-token
KL 0, the same greedy text over 200 steps.

## What each switch does

**Defaults on a card under 12 GiB** (no switch needed): the experts' host copy is a read-only mapped file
(`<model>.q36blobs`, written once) so Windows can drop cold experts without page-file writes; the token embeddings live
in host memory; the packed Q8_0 decode copy is off. `STRATA_Q36_BLOB_CACHE=0`, `STRATA_Q36_EMBD_HOST=0` and
`STRATA_Q8_PACKED=1` turn these back.

**Cache and context**
- `STRATA_Q36_KV=q8_0`: an 8-bit K/V cache (half of f16, a quarter of the f32 default).
- `STRATA_Q36_KV_RAM=1`, `STRATA_Q36_KV_WINDOW=65536`: the first 64K cells in VRAM, the rest in pinned RAM.
- `STRATA_Q36_KV_GROW=1`: as the context passes the window, each new 4,096-cell segment takes expert slots instead of
  RAM, so long conversations keep the cache in VRAM (each layer keeps at least `STRATA_Q36_KV_GROW_MIN`, 8, slots).
  Short chats keep the slots; at 100K the speed matches the cache-all-in-VRAM setup.

**Decode**
- `STRATA_Q36_HANDOFF=1`: a decode step is queued whole; each layer's GPU part publishes its routing to mapped memory
  and waits on a flag for the CPU experts, instead of a host round trip per layer.
- `STRATA_Q36_HO_THREAD=1`: the CPU experts on a host thread of their own.
- `STRATA_Q36_HO_GRAPH=1`: the hybrid decode step as a CUDA graph (one per power of two of context), +11%.
- `STRATA_Q36_REBAL_ASYNC=1`, `STRATA_Q36_REBAL_LAG=8`: expert slot swaps copied in the background, used 8 tokens later.
- `STRATA_Q36_PROMPT_DECAY=1`: after a long prompt the slots follow the answer, not the prompt.
- `STRATA_Q36_ATT_Q8G=1`: decode attention over the 8-bit cache reading each cell once for the 8 query heads that
  share it (+10% after 50K).
- `STRATA_Q36_GEMV_ILP=1`: the router's matrix-vector product with its loads issued ahead (+1%).
- `--pool-workers 19`: CPU expert threads (13 is the setup default; 19 measured best here, 10 next).

**Prompts**
- `STRATA_Q36_LM=1`, `STRATA_Q36_LM_TAIL=1`: layer-major prompts: each layer runs over a super-chunk of chunks before
  the next, so streamed experts cross PCIe once per super-chunk; the last partial chunk joins the last super-chunk.
- `STRATA_Q36_FA2=1`, `STRATA_Q36_FA2P=1`: tensor-core prompt attention (FP16 in, FP32 accumulate), pipelined K/V
  tiles through shared memory.
- `STRATA_Q36_FA2_SEG=32768`, `STRATA_Q36_ATT_MFLOATS=13`: the attention's FP16 copy and score buffers in small
  segments, freeing ~0.9 GiB at 262K for expert slots (8 → 19 a layer).
- `STRATA_Q36_GDN_TC=1`: the delta-rule recurrence on tensor cores (`src/qwen35/q36_gdn_tc.cuh`, from the split
  engine's `gdn_tc_prompt.cuh`), sm_80 and later.
- `STRATA_Q36_CKPT=1`: the recurrent state saved at the end of a prompt, so a follow-up continues from it instead of
  reading the conversation again.

## Limits

- **RAM.** 32 GB is tight: the engine maps the 18 GB expert file and holds 3-4 GB of its own. With other programs open
  (browsers, several desktop apps) runs dropped to under 1 GB free and some paged over 1 GB/h. Close large programs
  for the best speed.
- **VRAM margin.** At 262K with 4096-token chunks and no growing window, ~0.5 GiB stayed free and the driver demoted
  engine memory to system RAM once (heavy paging). The growing window avoids that; keep `STRATA_Q36_KV_GROW=1` with
  big chunks.
- **Full-length prompts.** Prompt reading at 262K is limited by attention (it grows with the square of the length).
- **Builds.** Measured on CUDA only. The HIP build of this code was not compiled here; `STRATA_Q36_GDN_TC` needs
  sm_80+.
- **MTP.** Not measured with this configuration yet (needs Unsloth's MTP GGUF and `--mtp 2`).
