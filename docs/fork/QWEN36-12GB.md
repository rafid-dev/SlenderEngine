# Qwen3.6-35B-A3B on a 12 GB card with 32 GB of RAM

This page is for running [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) on the `src/qwen35` engine path
with a 12 GB NVIDIA card and 32 GB of RAM, at the full 262,144-token context with an 8-bit K/V cache. It lists the
switches that give the best measured speed, what each does and what was measured. The experts don't fit in VRAM on
such a card, so the engine runs in its hybrid mode: some experts in VRAM, the rest computed on the CPU from RAM.

> **Measured on:** RTX 4070 Ti 12 GB, i7-14700KF, 32 GB DDR5-6000, two NVMe SSDs, Windows 11, CUDA 13.4 (sm_89).
> Unsloth's `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (no MTP block), thinking on, the model card's sampling (temperature
> 1.0, top-p 0.95, top-k 20, presence penalty 1.5), server runs, no MTP. Other cards and PCs will differ.

## The configuration

Engine arguments (the run config's `args`):

```
--native <path>\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --max-context 262144 --prefill auto --mtp 0 --pool-workers 19
```

The config also needs `"arch": "qwen35moe"` and a `tokenizer` folder made by `tools/strata_tokenizer.py --gguf <file>
--out <folder>`. A GGUF without the MTP block loads (the `nextn_predict_layers` key is optional); use `--mtp 0`.

Environment (all off by default; set them before starting the server). Every switch in this block gives the same
output bits as the engine without it, or changes only where an expert is computed (see Quality):

```bat
rem cache and context
set STRATA_Q36_KV=q8_0
set STRATA_Q36_KV_RAM=1
set STRATA_Q36_KV_WINDOW=16384
set STRATA_Q36_KV_GROW=1
rem decode
set STRATA_Q36_HANDOFF=1
set STRATA_Q36_HO_THREAD=1
set STRATA_Q36_HO_GRAPH=1
set STRATA_Q36_HO_DMA=1
set STRATA_Q36_REBAL_ASYNC=1
set STRATA_Q36_REBAL_LAG=8
set STRATA_Q36_PROMPT_DECAY=1
set STRATA_Q36_ATT_Q8G=1
set STRATA_Q36_ATT_TR=1
set STRATA_Q36_Q8ST=1
set STRATA_Q36_GEMV_ILP=1
set STRATA_POOL_FUSED=1
rem prompts
set STRATA_Q36_LM=1
set STRATA_Q36_LM_TAIL=1
set STRATA_Q36_FA2=1
set STRATA_Q36_FA2P=1
set STRATA_Q36_FA2Q=1
set STRATA_Q36_FA2_SEG=32768
set STRATA_Q36_ATT_MFLOATS=13
set STRATA_Q36_GDN_TC=1
set STRATA_Q36_GDN_COLS=16
set STRATA_Q36_MOE_OUT2=1
set STRATA_Q36_SGEMM8=1
set STRATA_Q36_CKPT=1
```

Two optional switches trade something for faster long prompts:

- `STRATA_Q36_DENSE_MMQ=1`: 8-bit activations in the prompt's dense projections, as decode already uses. A 50K read
  +7%; the output changes at rounding level (see Quality).
- `STRATA_Q36_CHUNK=4096` with `STRATA_Q36_LM_CHUNKS=2`: a 50K read +5% more, but the bigger chunk buffers take 14
  expert slots a layer and decode drops 4-5%.

## Results

Server runs at 262K context, the PC otherwise quiet, two alternated runs each. "Before" is the engine before this
work at 65K (its f32 cache did not fit 131K); llama.cpp is b8abe41a5 with `--fit on` at 131K.

| | Before (65K) | The configuration above | + `DENSE_MMQ` | + 4096-token chunks | llama.cpp (131K) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Short questions, decode (tok/s) | 47.5-48.4 | **99.0-99.3** | 98.0-98.1 | 94.0-94.6 | 43-45 |
| 50K-token prompt read (tok/s) | 1,023-1,028 | **4,208-4,234** | 4,513-4,531 | 4,748-4,757 | 599 (1,588 with `-ub 2048`) |
| Decode after a 50K prompt (tok/s) | 10.9-11.0 | **82.0-82.9** | 81.3-82.3 | 78.3-79.7 | 42 |
| Decode after the follow-up (tok/s) | 10.7-10.9 | 80.7-81.5 | 82.3-82.7 | 78.1-79.1 | 42 |
| Follow-up after a 50K prompt | the whole conversation read again (~50 s) | ~880 tokens read in ~1 s | same | same | 36 tokens read |
| Page-file writes | 0.6-1.1 GB/h | 0.00 GB/h | 0.00-0.01 | 0.00 | 0.5-4.2 GB/h |

Decode 100K into a conversation measured ~58 tok/s on the CLI before the last decode round (not re-measured since).
The first start writes the expert blob cache (~30 s); later starts are ready in 24-30 s.

**Quality.** The configuration above against the engine without the switches: the same bits where the expert
placement is the same (checked byte for byte with greedy decoding, up to 200 steps, and at 50K), and rounding level
for the few switches that change the order of sums: first-token KL at most 0.0007 with the same top-1, where reading
the same prompt in other chunk sizes gives ~0.0005. The 8-bit cache: 0.0007 at 8K, 0.000002 at 30K. The tensor-core
prompt attention 0.0001-0.0004, the tensor-core delta-rule scan 0.00002. Switches that move experts between VRAM and
the CPU (`REBAL_ASYNC`, `PROMPT_DECAY`, `KV_GROW`) change only where an expert is computed: first-token KL 0, the same
greedy text over 200 steps. `STRATA_Q36_DENSE_MMQ` is a real change of arithmetic (8-bit activations): first-token KL
0.00012, the same top-1 and greedy text over 32 steps, later steps mean 0.0008 and at most 0.0088. 4096-token chunks:
first-token KL 0.00006.

## What each switch does

**Defaults on a card under 12 GiB** (no switch needed): the experts' host copy is a read-only mapped file
(`<model>.q36blobs`, written once) so Windows can drop cold experts without page-file writes; the token embeddings live
in host memory; the packed Q8_0 decode copy is off. `STRATA_Q36_BLOB_CACHE=0`, `STRATA_Q36_EMBD_HOST=0` and
`STRATA_Q8_PACKED=1` turn these back.

**Cache and context**
- `STRATA_Q36_KV=q8_0`: an 8-bit K/V cache (half of f16, a quarter of the f32 default).
- `STRATA_Q36_KV_RAM=1`, `STRATA_Q36_KV_WINDOW=16384`: the first cells of the cache in VRAM, the rest in pinned RAM.
- `STRATA_Q36_KV_GROW=1`: as the context passes the window, each new 4,096-cell segment takes expert slots instead of
  RAM, so long conversations keep the cache in VRAM (each layer keeps at least `STRATA_Q36_KV_GROW_MIN`, 8, slots).
  At 100K the speed matches the cache-all-in-VRAM setup.

**Decode**
- `STRATA_Q36_HANDOFF=1`: a decode step is queued whole; each layer's GPU part publishes its routing to mapped memory
  and waits on a flag for the CPU experts, instead of a host round trip per layer.
- `STRATA_Q36_HO_THREAD=1`: the CPU experts on a host thread of their own.
- `STRATA_Q36_HO_GRAPH=1`: the hybrid decode step as a CUDA graph (one per power of two of context), +11%.
- `STRATA_Q36_HO_DMA=1`: the CPU experts' results reach the GPU by two DMA copies inside the graph instead of the GPU
  reading mapped host memory value by value (the merge 1.29 → 0.06 ms a token).
- `STRATA_Q36_REBAL_ASYNC=1`, `STRATA_Q36_REBAL_LAG=8`: expert slot swaps copied in the background, used 8 tokens later.
- `STRATA_Q36_PROMPT_DECAY=1`: after a long prompt the slots follow the answer, not the prompt.
- `STRATA_Q36_ATT_Q8G=1`: decode attention over the 8-bit cache reading each cell once for the 8 query heads that
  share it, from 16K cells of context (`STRATA_Q36_ATT_Q8G_MIN`, 64 chunks of 256); `STRATA_Q36_ATT_TR=1`: its 8 head
  scores by a transposing butterfly.
- `STRATA_Q36_Q8ST=1`: the dense Q8_0 matrix-vector products read each row as aligned 16-byte words through shared
  memory (375 → ~435 GB/s on the large matrices; `STRATA_Q36_Q8ST_BLOCKS`, default 1024).
- `STRATA_Q36_GEMV_ILP=1`: the router's matrix-vector product with its loads issued ahead.
- `STRATA_POOL_FUSED=1`: the CPU pool runs an expert's gate/up and down rows in one batch, with no barrier between.
- `--pool-workers 19`: CPU expert threads (13 is the setup default; 19 measured best here, 10 next).

**Prompts**
- `STRATA_Q36_LM=1`, `STRATA_Q36_LM_TAIL=1`: layer-major prompts: each layer runs over a super-chunk of chunks before
  the next, so streamed experts cross PCIe once per super-chunk; the last partial chunk joins the last super-chunk.
- `STRATA_Q36_FA2=1`, `STRATA_Q36_FA2P=1`, `STRATA_Q36_FA2Q=1`: tensor-core prompt attention (FP16 in, FP32
  accumulate), K/V tiles pipelined through shared memory, two blocks per SM.
- `STRATA_Q36_FA2_SEG=32768`, `STRATA_Q36_ATT_MFLOATS=13`: the attention's FP16 copy and score buffers in small
  segments, freeing ~0.9 GiB at 262K for expert slots.
- `STRATA_Q36_GDN_TC=1`: the delta-rule recurrence on tensor cores (`src/qwen35/q36_gdn_tc.cuh`, from the split
  engine's `gdn_tc_prompt.cuh`), sm_80 and later. `STRATA_Q36_GDN_COLS=16` sets the fallback kernel's column blocks.
- `STRATA_Q36_MOE_OUT2=1`, `STRATA_Q36_SGEMM8=1`: the expert combine pass and the small FP32 GEMMs (router, delta-net
  alpha / beta, shared-expert gate).
- `STRATA_Q36_CKPT=1`: the recurrent state saved at the end of a prompt, so a follow-up continues from it instead of
  reading the conversation again.

**Diagnostics and measured-but-not-used**
- `STRATA_Q36_PROF=1/2/3`: prompt stage times, exact per-stage times, decode stage times. `STRATA_POOL_PHASE_TIMING=1`:
  the CPU pool's phases in the summary. `STRATA_Q36_EXPERT_HIST=1`: rows per expert in the prompt's expert passes.
- `STRATA_Q36_GGML_MOE=1` with `STRATA_Q36_GGML_MOE_PATH=<dll>` (`src/qwen35/q36_ggml_moe.cpp`): the prompt's expert
  GEMMs through llama.cpp's current kernel. Same quality, no gain under real routing.
- `STRATA_Q36_GDN_NTL=2`: two column tiles a head in the tensor-core scan: 36% slower here.
- `STRATA_Q36_LOOKUP=N`: drafts copied from the context: a loss on this card (each verified row brings CPU experts).

## Limits

- **RAM.** 32 GB is tight: the engine maps the 18 GB expert file and holds 3-4 GB of its own. With other programs open
  (browsers, several desktop apps, a download) runs dropped to under 1 GB free, the expert file paged and speeds fell
  by a third or more. Close large programs for the best speed.
- **Short reads.** A read of about 130 to 2,000 new tokens streams every expert of every layer and takes about 1 s
  whatever its length.
- **VRAM margin.** At 262K with 4096-token chunks and no growing window, ~0.5 GiB stayed free and the driver demoted
  engine memory to system RAM once (heavy paging). The growing window avoids that.
- **Full-length prompts.** Prompt reading at 262K is limited by attention (it grows with the square of the length).
- **Builds.** Measured on CUDA only. The HIP build of this code was not compiled here; `STRATA_Q36_GDN_TC` needs
  sm_80+.
- **MTP.** Not part of these numbers. `--mtp N` runs with the decode graph since the verify-window fix; measurements
  with Unsloth's MTP GGUF are not finished.
