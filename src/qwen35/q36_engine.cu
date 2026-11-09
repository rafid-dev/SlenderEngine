// src/qwen35/q36_engine.cu - strata-q36: the engine path for Qwen3.6-35B-A3B (GGUF architecture qwen35moe),
// NEW-MODELS-PLAN.md T1.  A driver of its own beside Flash-Next's (generate/verify/prefill stay untouched, so the
// Flash-Next identity gate cannot move), built from Strata's kernels:
//
//   * weights straight from the GGUF: dense Q8_0 / K-quant matrices in their blocks (native MMVQ), the 1-D tensors and
//     the F32 projections (ssm_alpha/beta, the router, the shared-expert gate) as f32, the routed experts repacked at
//     load into native expert blobs [gate | up | down] per (layer, expert) for `native_expert_grouped`;
//   * the plain pre-norm residual layer driver: x += mixer(rms(x) * attn_norm); x += moe(rms(x) * post_attention_norm);
//   * GDN: fused conv + L2 and the fused step with the SiLU-gated norm (per token, the state carried);
//   * attention without an indexer: dense, 16 query heads over 2 KV heads, split-K over 256-cell chunks with a
//     log-sum-exp merge, causal for a batch of queries (the prompt) and for one (decode);
//   * MoE: the native router (256 / top-8), the experts grouped per expert on the device (one weight read per expert
//     for the whole batch), the shared expert with its sigmoid gate, the native combine;
//   * the prompt read in chunks of --chunk tokens through the same driver (batched matmuls, ncols <= 8 per launch);
//   * decode one token at a time from a captured graph (positions on the device), or with --no-graph eagerly.
//
// Modes: --serve (serve/server.py's engine protocol: READY / GEN / T / DONE / STOP / PP), --tokens IDS [--n N]
// [--force IDS] [--dump-full-logits F] (tools/qwen35/kl.py's reference interface), --bench-pp N / --bench-tg N.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/family_layout.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/qwen35/engine.hpp"
#include "strata/qwen35/sampling.hpp"
#include "q36_kernels.cuh"
#if defined(STRATA_Q36_CPU_POOL)
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#endif

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

namespace {

namespace K = strata::kernels;
namespace MMQ = strata::prefill::mmq;

#define CK(x)                                                                                                          \
    do {                                                                                                               \
        cudaError_t e_ = (x);                                                                                          \
        if (e_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "strata-q36: %s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    } while (0)

using Clock = std::chrono::steady_clock;
double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

constexpr int kThreads = 256;
constexpr int kChunkCells = 256;   // attention split-K chunk
constexpr int kConvTile = 16;     // tokens per block of the prompt conv (k_conv_seq tile mode)

// ------------------------------------------------------------------------------------------------ small kernels
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }
__device__ __forceinline__ float sigm(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(expf(v)); }

__device__ float block_sum(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float block_max(float v, float* sh) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) sh[threadIdx.x] = fmaxf(sh[threadIdx.x], sh[threadIdx.x + s]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
/// One value of an 8-bit KV row (see kv_mode): blocks of qb values (32: llama.cpp's q8_0; 256: int8, one scale per
/// cell row) as int8 with the f16 scale amax / 127.  Thread i writes value i, the first thread of a block its scale.
/// Every thread of the block (qb 256: of the thread block, sh its kThreads floats) must call it.
__device__ __forceinline__ void q8_store(int8_t* row, __half* scales, int i, float v, int qb, float* sh) {
    float amax = fabsf(v);
    if (qb == 32)
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o, 32));
    else
        amax = block_max(amax, sh);
    const float d = amax / 127.f;
    row[i] = (int8_t) roundf(d != 0.f ? v / d : 0.f);
    if (i % qb == 0) scales[i / qb] = __float2half(d);
}

/// y[r] = rms(x[r]) * w, one block per row of n.
__global__ void k_rms(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
/// x[r] += y[r]; h[r] = rms(x[r]) * w (one block per row of n): the residual add and the next norm in one launch.
/// out[r * ld_out + i] = rms(x[r * ldx + ..]) * w (one block per row of n): a norm into a slice of a wider row.
__global__ void k_rms_ld(const float* x, int ldx, const float* w, float* out, int ld_out, int n, float eps) {
    __shared__ float sh[kThreads];
    const float* xr = x + (size_t) blockIdx.x * ldx;
    float* yr = out + (size_t) blockIdx.x * ld_out;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i] * xr[i];
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * r * w[i];
}
__global__ void k_add_rms(float* x, const float* y, const float* w, float* h, int n, float eps) {
    __shared__ float sh[kThreads];
    float* xr = x + (size_t) blockIdx.x * n;
    const float* yr = y + (size_t) blockIdx.x * n;
    float* hr = h + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = xr[i] + yr[i];
        xr[i] = v;
        s += v * v;
    }
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) hr[i] = xr[i] * r * w[i];
}
/// The MoE block's end for one token row (block per row): x += sum_k w_k part_k (in pick order, FMA after the first
/// product, as native_moe_combine) + sigmoid(sgate) * shared; then, with `wn`, h = rms(x) * wn (the next layer's
/// attention norm).  Replaces combine + gate + add + rms.
__global__ void k_moe_out(float* x, const float* parts, const int32_t* prow, const float* wts, int K_,
                          const float* shared, const float* sgate, int sg_stride, const float* wn, float* h, int n,
                          float eps) {
    __shared__ float sh[kThreads];
    __shared__ const float* pk[16];
    const int t = blockIdx.x;
    float* xr = x + (size_t) t * n;
    if (threadIdx.x < K_)   // pick k's row: t * K + k, or its sorted row (`prow`, the prompt's MMQ order)
        pk[threadIdx.x] = parts + (size_t) (prow ? prow[t * K_ + threadIdx.x] : t * K_ + threadIdx.x) * n;
    __syncthreads();
    const float* wr = wts + (size_t) t * K_;
    const float gs = sigm(sgate[(size_t) t * sg_stride]);
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float acc = wr[0] * pk[0][i];
        for (int k = 1; k < K_; ++k) acc = fmaf(wr[k], pk[k][i], acc);
        acc += gs * shared[(size_t) t * n + i];
        const float v = xr[i] + acc;
        xr[i] = v;
        s += v * v;
    }
    if (wn == nullptr) return;
    s = block_sum(s, sh);
    const float r = rsqrtf(s / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) h[(size_t) t * n + i] = xr[i] * r * wn[i];
}
/// GDN alpha and beta for T <= 8 rows in one launch, their epilogues applied: rows [0, HV) of the output are
/// softplus(alpha . x + dt) * ssm_a, rows [HV, 2 HV) sigmoid(beta . x).  Warp per row (grid (ceil(2 HV / 8), T)).
__global__ void k_gdn_ab(const float* Wa, const float* Wb, const float* x, const float* dt, const float* ssm_a,
                         float* alpha, float* beta, int n_in, int HV) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= 2 * HV) return;
    const bool isb = o >= HV;
    const int r = isb ? o - HV : o;
    const float4* wr = (const float4*) ((isb ? Wb : Wa) + (size_t) r * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane != 0) return;
    if (isb) beta[(size_t) blockIdx.y * HV + r] = sigm(acc);
    else alpha[(size_t) blockIdx.y * HV + r] = softplus(acc + dt[r]) * ssm_a[r];
}
__global__ void k_add(float* x, const float* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}
__global__ void k_swiglu(const float* g, const float* u, float* h, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) h[i] = silu(g[i]) * u[i];
}
/// y[t][o] = W[o] . x[t] for f32 W [n_out][n_in], n_in a multiple of 128: one warp per row, float4 loads, 8 rows per
/// block (grid (ceil(n_out / 8), T)).
__global__ void k_gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int o = blockIdx.x * 8 + warp;
    if (o >= n_out) return;
    const float4* wr = (const float4*) (W + (size_t) o * n_in);
    const float4* xr = (const float4*) (x + (size_t) blockIdx.y * n_in);
    float acc = 0.f;
    for (int i = lane; i < n_in / 4; i += 32) {
        const float4 a = wr[i], b = xr[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (int m = 16; m > 0; m >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, m);
    if (lane == 0) y[(size_t) blockIdx.y * n_out + o] = acc;
}
unsigned rows8(int n) { return (unsigned) ((n + 7) / 8); }
/// GDN gate / beta in place for T rows of HV: alpha -> softplus(alpha + dt) * ssm_a, beta -> sigmoid(beta).
__global__ void k_gdn_gb(float* alpha, const float* dt, const float* ssm_a, float* beta, int HV, int T) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= HV * T) return;
    const int h = i % HV;
    alpha[i] = softplus(alpha[i] + dt[h]) * ssm_a[h];
    beta[i] = sigm(beta[i]);
}
/// sh[t] *= sigmoid(gate[t]) (the shared expert's scalar gate), rows of n.
__global__ void k_gate_rows(float* sh, const float* gate, int n, int T) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) n * T) return;
    sh[i] *= sigm(gate[i / n]);
}

/// The 8-bit cache in segments (STRATA_Q36_KV_RAM, see kv_ram): a layer's cells in segments of `seg` cells, segment s
/// at k[s] / v[s] (device pointers: VRAM for the window's segments, host-mapped memory past it), each laid out as the
/// contiguous cache with seg in place of ctx: [HKV][seg][HD] int8 values, then [HKV][seg][HD / qb] f16 scales.
/// k == nullptr: the contiguous cache.
struct KvTab {
    int8_t* const* k = nullptr;
    int8_t* const* v = nullptr;
    int shift = 0;   // log2(seg)
};
/// cell (kvh, pos)'s values and scales: in the segment table, or in the contiguous cache at base (ctx cells)
__device__ __forceinline__ int8_t* kv_row(int8_t* base, int8_t* const* tab, int shift, int kvh, int pos, int HKV, int ctx,
                                          int HD, int qb, __half** sc) {
    if (tab == nullptr) {
        const size_t cell = (size_t) kvh * ctx + pos;
        *sc = (__half*) (base + (size_t) HKV * ctx * HD) + cell * (HD / qb);
        return base + cell * HD;
    }
    const int seg = 1 << shift;
    int8_t* sb = tab[pos >> shift];
    const size_t cell = (size_t) kvh * seg + (pos & (seg - 1));
    *sc = (__half*) (sb + (size_t) HKV * seg * HD) + cell * (HD / qb);
    return sb + cell * HD;
}

/// Batch rows (several sessions decoded in one window, one token each): row t reads and writes the cache of session
/// slot[t] at cell[t]; its rotary position is cell[t] + delta[t].  Every field null: one session, the rows are its
/// consecutive cells *pos0 + t (the prompt, a verify window, a decode step).
struct RowSel {
    const int32_t* cell = nullptr;   // device [T]
    const int32_t* slot = nullptr;   // device [T]
    const int32_t* delta = nullptr;  // device [T]
    float* const* kc = nullptr;      // device [n_sessions]: this layer's K / V base of each session (f32 or f16
    float* const* vc = nullptr;      // cells, see kv16: the kernels cast to their element type)
};

/// The KV cache's element: f32, or f16 (STRATA_Q36_KV_F16, the default: half the VRAM per cell; the prompt's
/// attention GEMMs took FP16 operands already, so a prompt's arithmetic is unchanged and only decode reads rounded K/V)
__device__ __forceinline__ float kv_ld(float x) { return x; }
__device__ __forceinline__ float kv_ld(__half x) { return __half2float(x); }
__device__ __forceinline__ void kv_st(float& d, float v) { d = v; }
__device__ __forceinline__ void kv_st(__half& d, float v) { d = __float2half(v); }

/// The attention layer's per-token preparation for T rows at positions *pos0 + t: q (from the [q | gate] pairs) and k
/// RMS-normed per head with their weights, NEOX RoPE over the first n_rot dims, k / v into the cache, the gate kept.
/// Rotary position of a cell: the cell itself (text) plus *rdelta (what the pictures before a text run did not use
/// of their cells, see mrope), or, with `mtab` (a request's prompt cells, device int32 [cell][3] = t, h, w), the
/// interleaved M-RoPE position (llama.cpp's IMROPE with sections 11 / 11 / 10: pair p takes stream p % 3).
/// grid (H + HKV, T), block 256 (HD <= 256).
template <class KT>
__global__ void k_attn_prep(const float* qg, const float* kx, const float* vx, const float* qw, const float* kw,
                            float* q, float* gate, KT* kc, KT* vc, int H, int HKV, int HD, int n_rot,
                            const int32_t* pos0, int ctx, float base, float eps, const int32_t* rdelta,
                            const int32_t* mtab, RowSel rs, int qb = 32, KvTab kt = KvTab()) {
    __shared__ float sh[kThreads];
    __shared__ float buf[256];
    const int head = blockIdx.x, t = blockIdx.y;
    const int pos = rs.cell ? rs.cell[t] : *pos0 + t;
    const int dl = rs.delta ? rs.delta[t] : (rdelta ? *rdelta : 0);
    if (rs.kc) {
        kc = (KT*) rs.kc[rs.slot[t]];
        vc = (KT*) rs.vc[rs.slot[t]];
    }
    const bool isq = head < H;
    const float* src = isq ? qg + ((size_t) t * H + head) * 2 * HD : kx + ((size_t) t * HKV + head - H) * HD;
    const int i = threadIdx.x;
    const float v0 = i < HD ? src[i] : 0.f;
    const float s = block_sum(v0 * v0, sh);
    const float r = rsqrtf(s / (float) HD + eps);
    if (i < HD) buf[i] = v0 * r * (isq ? qw[i] : kw[i]);
    __syncthreads();
    if (i >= HD) return;
    float v = buf[i];
    const int half = n_rot / 2;
    if (i < n_rot) {
        const int p = i < half ? i : i - half;
        const int rp = mtab ? __ldg(mtab + (size_t) pos * 3 + p % 3) : pos + dl;
        const float theta = (float) rp * powf(base, -2.0f * (float) p / (float) n_rot);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        v = i < half ? buf[i] * cs - buf[i + half] * sn : buf[i - half] * sn + buf[i] * cs;
    }
    if (isq) {
        q[((size_t) t * H + head) * HD + i] = v;
        gate[((size_t) t * H + head) * HD + i] = qg[((size_t) t * H + head) * 2 * HD + HD + i];
    } else {
        const int kh = head - H;
        const size_t cell = (size_t) kh * ctx + pos;
        if constexpr (std::is_same<KT, int8_t>::value) {
            // q8_0 cells (see kv_mode): the values, then the scales of every cell after a layer's values (or the
            // segment's, see KvTab)
            if (kt.k) {
                __half *ks, *vs;
                int8_t* kr = kv_row(nullptr, kt.k, kt.shift, kh, pos, HKV, ctx, HD, qb, &ks);
                int8_t* vr = kv_row(nullptr, kt.v, kt.shift, kh, pos, HKV, ctx, HD, qb, &vs);
                q8_store(kr, ks, i, v, qb, sh);
                q8_store(vr, vs, i, vx[((size_t) t * HKV + kh) * HD + i], qb, sh);
            } else {
                __half* ks = (__half*) (kc + (size_t) HKV * ctx * HD);
                __half* vs = (__half*) (vc + (size_t) HKV * ctx * HD);
                q8_store(kc + cell * HD, ks + cell * (HD / qb), i, v, qb, sh);
                q8_store(vc + cell * HD, vs + cell * (HD / qb), i, vx[((size_t) t * HKV + kh) * HD + i], qb, sh);
            }
        } else {
            kv_st(kc[cell * HD + i], v);
            kv_st(vc[cell * HD + i], vx[((size_t) t * HKV + kh) * HD + i]);
        }
    }
}

/// Split-K attention, one block per (chunk of 256 cells, query row): query row qi = t * H + h (t from t0) sees the
/// cells [0, *pos0 + t].  Writes the chunk's partial accumulator, max and sum; chunks past the query's cells write
/// nothing (the merge stops before them).  grid (n_chunk, n_rows), block 256, HD == 256.
template <class KT>
__global__ void k_attn_part(const float* q, const KT* kc, const KT* vc, const int32_t* pos0, int t0, int H,
                            int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl, RowSel rs) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x, row = blockIdx.y;
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    if (rs.kc) {
        kc = (const KT*) rs.kc[rs.slot[t]];
        vc = (const KT*) rs.vc[rs.slot[t]];
    }
    const int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    const int p = p0 + j;
    float s = -INFINITY;
#if defined(STRATA_USE_HIP)
    // the scores with coalesced K reads: each 32-lane group takes one cell at a time, every lane 8 of its 256 values
    // (one 16-byte load of an f16 cache row, two of an f32 one), summed across the group; cell c's score lands in
    // thread c's register through the shared buffer. (One thread per cell walking its own row read 512-byte strided
    // addresses across a wavefront.) Same products, summed in a different order.
    {
        const int lane = j & 31, grp = j >> 5, n_grp = kChunkCells / 32;
        const int n = min(kChunkCells, npos - p0);
        float qv[8];
        for (int i = 0; i < 8; ++i) qv[i] = qs[lane * 8 + i];
        for (int c = grp; c < n; c += n_grp) {
            const KT* kp = kc + ((size_t) kvh * ctx + p0 + c) * HD + lane * 8;
            float a = 0.f;
            for (int i = 0; i < 8; ++i) a += qv[i] * kv_ld(kp[i]);
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor(a, o, 32);
            if (lane == 0) sc[c] = a * scale;
        }
        __syncthreads();
        if (j < n) s = sc[j];
        __syncthreads();
    }
#else
    if (p < npos) {
        const KT* kp = kc + ((size_t) kvh * ctx + p) * HD;
        float a = 0.f;
        for (int i = 0; i < HD; ++i) a += qs[i] * kv_ld(kp[i]);
        s = a * scale;
    }
#endif
    const float m = block_max(s, sh);
    const float e = p < npos ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    const int n = min(kChunkCells, npos - p0);
    float acc = 0.f;
    const KT* vp = vc + ((size_t) kvh * ctx + p0) * HD + j;
    for (int k = 0; k < n; ++k) acc += sc[k] * kv_ld(vp[(size_t) k * HD]);
    const size_t o = (size_t) row * n_chunk + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
/// k_attn_part over an 8-bit cache (kv_mode q8_0 / int8: a layer's [HKV][ctx][HD] int8 values, then the f16 scales of
/// their qb-value blocks, [HKV][ctx][HD / qb]).  The scores as in the HIP path above: each 32-lane group takes one cell
/// at a time, every lane 8 of its values (one 8-byte load).  DOT8: the query row's 32-value blocks are int8 too
/// (quantized in registers, scale amax / 127, as llama.cpp's q8_0 x q8_0 dot quantizes its activations) and a lane's 8
/// products are two dp4a (v_dot4_i32_i8 on RDNA2) times the two scales; else the cells are scaled against the float
/// query.  The values: thread j's column, its block's scale folded into the probability.
template <bool DOT8>
__global__ void k_attn_part_q8(const float* q, const int8_t* kc, const int8_t* vc, const int32_t* pos0, int t0, int H,
                               int HKV, int ctx, float scale, int n_chunk, float* pacc, float* pm, float* pl, RowSel rs,
                               int qb, KvTab kt = KvTab(), int c_base = 0, int npos_bias = 0, int slot_base = 0) {
    constexpr int HD = 256;
    __shared__ float qs[HD], sc[kChunkCells], sh[kThreads];
    const int c = blockIdx.x + c_base, row = blockIdx.y;   // c_base: a launch over a range of the chunks (KV_RAM rounds)
    const int t = t0 + row / H, h = row % H, kvh = h / (H / HKV);
    // npos_bias: a compact cache (STRATA_Q36_SPARSE: the selected pages, then the recent cells) where row t sees
    // cells [0, pos + t + 1 + bias); slot_base: its partials after the window's
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1 + npos_bias;
    if (rs.kc) {
        kc = (const int8_t*) rs.kc[rs.slot[t]];
        vc = (const int8_t*) rs.vc[rs.slot[t]];
    }
    int p0 = c * kChunkCells;
    if (p0 >= npos) return;
    const int nb = HD / qb;
    const int pend = npos;   // (npos counts from cell 0; p0 may move into a segment below)
    const int n = min(kChunkCells, pend - p0);
    if (kt.k) {   // the chunk's segment (a chunk never spans two: seg is a multiple of kChunkCells) as a small cache
        const int seg = 1 << kt.shift;
        kc = kt.k[p0 >> kt.shift];
        vc = kt.v[p0 >> kt.shift];
        ctx = seg;
        p0 &= seg - 1;
    }
    const __half* ks = (const __half*) (kc + (size_t) HKV * ctx * HD);
    const __half* vs = (const __half*) (vc + (size_t) HKV * ctx * HD);
    const int j = threadIdx.x;
    qs[j] = q[((size_t) t * H + h) * HD + j];
    __syncthreads();
    float s = -INFINITY;
    {
        const int lane = j & 31, grp = j >> 5, n_grp = kChunkCells / 32;
        const int kb = lane * 8 / qb;   // the lane's block of the cell's scales
        float qv[8];
        for (int i = 0; i < 8; ++i) qv[i] = qs[lane * 8 + i];
        int q0 = 0, q1 = 0;
        float dq = 1.f;
        if (DOT8) {   // the query's 32-value block (4 lanes) as int8
            float amax = 0.f;
            for (int i = 0; i < 8; ++i) amax = fmaxf(amax, fabsf(qv[i]));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1, 32));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2, 32));
            dq = amax / 127.f;
            const float id = dq != 0.f ? 1.f / dq : 0.f;
            for (int i = 0; i < 4; ++i) {
                q0 |= ((int) (uint8_t) (int8_t) roundf(qv[i] * id)) << (8 * i);
                q1 |= ((int) (uint8_t) (int8_t) roundf(qv[4 + i] * id)) << (8 * i);
            }
        }
        for (int cc = grp; cc < n; cc += n_grp) {
            const size_t cell = (size_t) kvh * ctx + p0 + cc;
            const int2 kq = *(const int2*) (kc + cell * HD + lane * 8);
            const float dk = __half2float(ks[cell * nb + kb]);
            float a;
            if (DOT8) {
                a = (float) __dp4a(kq.y, q1, __dp4a(kq.x, q0, 0)) * dq * dk;
            } else {
                a = 0.f;
                for (int i = 0; i < 4; ++i) {
                    a += qv[i] * (float) (int8_t) (kq.x >> (8 * i));
                    a += qv[4 + i] * (float) (int8_t) (kq.y >> (8 * i));
                }
                a *= dk;
            }
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o, 32);
            if (lane == 0) sc[cc] = a * scale;
        }
        __syncthreads();
        if (j < n) s = sc[j];
        __syncthreads();
    }
    const float m = block_max(s, sh);
    const float e = j < n ? expf(s - m) : 0.f;
    sc[j] = e;
    const float l = block_sum(e, sh);
    float acc = 0.f;
    const size_t cell0 = (size_t) kvh * ctx + p0;
    const int8_t* vp = vc + cell0 * HD + j;
    const __half* vsp = vs + cell0 * nb + j / qb;
    for (int k = 0; k < n; ++k) acc += sc[k] * __half2float(vsp[(size_t) k * nb]) * (float) vp[(size_t) k * HD];
    const size_t o = (size_t) row * n_chunk + slot_base + c;
    pacc[o * HD + j] = acc;
    if (j == 0) {
        pm[o] = m;
        pl[o] = l;
    }
}
/// The chunks merged, times sigmoid(gate): grid (n_rows), block HD.
__global__ void k_attn_merge(const float* pacc, const float* pm, const float* pl, const float* gate,
                             const int32_t* pos0, int t0, int H, int n_chunk, float* out, RowSel rs,
                             int win_chunks = -1, int bias = 0, int tail_slot = -1) {
    constexpr int HD = 256;
    const int row = blockIdx.x, t = t0 + row / H, h = row % H, d = threadIdx.x;
    const int npos = (rs.cell ? rs.cell[t] : *pos0 + t) + 1;
    // sparse (win_chunks >= 0): the window's chunks, then the compact cache's for the cells [0, npos + bias)
    const int nc = win_chunks >= 0 ? win_chunks + (npos + bias + kChunkCells - 1) / kChunkCells
                                   : (npos + kChunkCells - 1) / kChunkCells;
    float M = -INFINITY;
    for (int c = 0; c < nc; ++c) M = fmaxf(M, pm[(size_t) row * n_chunk + c]);
    if (tail_slot >= 0) M = fmaxf(M, pm[(size_t) row * n_chunk + tail_slot]);
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nc; ++c) {
        const size_t o = (size_t) row * n_chunk + c;
        const float w = expf(pm[o] - M);
        num += w * pacc[o * HD + d];
        den += w * pl[o];
    }
    if (tail_slot >= 0) {   // the sparse tail estimate (k_sp_tail)
        const size_t o = (size_t) row * n_chunk + tail_slot;
        if (pm[o] > -INFINITY) {
            const float w = expf(pm[o] - M);
            num += w * pacc[o * HD + d];
            den += w * pl[o];
        }
    }
    const size_t gi = ((size_t) t * H + h) * HD + d;
    out[gi] = num / den * sigm(gate[gi]);
}

/// The routed entries (T tokens x K picks, entry e = t * K + k) grouped per expert, on the device: one group per
/// expert used (ascending id), its blob address, its entries in entry order.  One block.
__global__ void k_group(const int32_t* ids, int n_ent, int K_, int n_expert, const uint8_t* blobs, unsigned long long blob,
                        unsigned long long* grp_ptr, int32_t* grp_start, int32_t* n_groups, int32_t* ent_dst,
                        int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) x * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// k_group for the hybrid mode: only the entries whose expert has a VRAM slot (slot_of[x] >= 0) are grouped, each
/// group's blob at blobs + slot * blob; the others are the CPU pool's (their rows come from k_merge_cpu).  One block.
__global__ void k_group_hy(const int32_t* ids, int n_ent, int K_, int n_expert, const int32_t* slot_of,
                           const uint8_t* blobs, unsigned long long blob, unsigned long long* grp_ptr, int32_t* grp_start,
                           int32_t* n_groups, int32_t* ent_dst, int32_t* ent_tok) {
    __shared__ int cnt[512], off[512];
    for (int x = threadIdx.x; x < n_expert; x += blockDim.x) cnt[x] = 0;
    __syncthreads();
    for (int e = threadIdx.x; e < n_ent; e += blockDim.x)
        if (slot_of[ids[e]] >= 0) atomicAdd(&cnt[ids[e]], 1);
    __syncthreads();
    if (threadIdx.x == 0) {
        int g = 0, pos = 0;
        for (int x = 0; x < n_expert; ++x) {
            off[x] = pos;
            if (cnt[x] == 0) continue;
            grp_ptr[g] = (unsigned long long) (blobs + (size_t) slot_of[x] * blob);
            grp_start[g] = pos;
            pos += cnt[x];
            ++g;
        }
        grp_start[g] = pos;
        *n_groups = g;
        for (int e = 0; e < n_ent; ++e) {
            if (slot_of[ids[e]] < 0) continue;
            const int p = off[ids[e]]++;
            ent_dst[p] = e;
            ent_tok[p] = e / K_;
        }
    }
}
/// The CPU pool's expert rows into the entries' rows: parts[e] = cpu[e] where miss[e].  grid (n_ent), block 256.
__global__ void k_merge_cpu(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!miss[e]) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = cpu[(size_t) e * n + i];
}
/// The handoff mode (STRATA_Q36_HANDOFF): one thread waits until the host has set *flag to seq (host-mapped memory,
/// read volatile so every poll reaches it), then the stream goes on.
__global__ void k_ho_wait(const unsigned* flag, unsigned seq) {
    if (threadIdx.x != 0) return;
    while (*(volatile const unsigned*) flag != seq) {
#if defined(STRATA_USE_HIP)
        __builtin_amdgcn_s_sleep(2);
#endif
    }
    __threadfence_system();
}
/// The handoff mode's other direction: the routing's ids and the rows' hidden into host-mapped memory, then *flag =
/// seq after a system fence (the host polls it: on Windows an event queued behind a waiting kernel is not reported
/// until the whole submission ends, so events cannot carry this).  One block.
__global__ void k_ho_publish(const int32_t* ids, int n_ids, const float* h, int n_h, int32_t* dids, float* dh,
                             unsigned* flag, unsigned seq) {
    for (int i = threadIdx.x; i < n_ids; i += blockDim.x) dids[i] = ids[i];
    for (int i = threadIdx.x; i < n_h; i += blockDim.x) dh[i] = h[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        __threadfence_system();
        *(volatile unsigned*) flag = seq;
    }
}
/// k_merge_cpu with the CPU rows and the miss flags read from host-mapped memory (volatile: the host wrote them
/// after the GPU last looked), the same values copied.
__global__ void k_merge_host(float* parts, const float* cpu, const int32_t* miss, int n) {
    const int e = blockIdx.x;
    if (!*(volatile const int32_t*) (miss + e)) return;
    const volatile float* src = cpu + (size_t) e * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) parts[(size_t) e * n + i] = src[i];
}
/// The prompt attention's FP16 operands: cells [0, P) of each KV head's K / V (cache rows ctx apart) into compact
/// [kvh][P][HD] halves.  grid1(HKV * P * HD).
__global__ void k_kv_f16(const float* kc, const float* vc, __half* k16, __half* v16, int ctx, int P, int HD, int HKV) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD;
    const int64_t kvh = i / per, rest = i % per;
    const size_t src = (size_t) kvh * ctx * HD + rest;
    k16[i] = __float2half(kc[src]);
    v16[i] = __float2half(vc[src]);
}
/// k_kv_f16 for an 8-bit cache (see k_attn_part_q8 for its layout): cells [0, P) dequantized.  grid1(HKV * P * HD).
__global__ void k_kv_q8_f16(const int8_t* kc, const int8_t* vc, __half* k16, __half* v16, int ctx, int P, int HD,
                            int HKV, int qb) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD;
    const int64_t kvh = i / per, rest = i % per;
    const size_t src = (size_t) kvh * ctx * HD + rest;
    const size_t sci = src / qb;   // the value's block: cells are HD values, HD a multiple of qb
    const __half* ks = (const __half*) (kc + (size_t) HKV * ctx * HD);
    const __half* vs = (const __half*) (vc + (size_t) HKV * ctx * HD);
    k16[i] = __float2half(__half2float(ks[sci]) * (float) kc[src]);
    v16[i] = __float2half(__half2float(vs[sci]) * (float) vc[src]);
}
/// KV_RAM staging: segments [s0, s0 + n) of a layer's K (blockIdx.y < n) and V into the staging block (K at slot
/// s - s0, V at nwin + s - s0), read from host-mapped memory with 16-byte loads.  grid (64, 2 n), block 256.
__global__ void k_seg_gather(int8_t* const* ktab, int8_t* const* vtab, int s0, int n, int nwin, int seg_u4, int8_t* stage) {
    const int j = blockIdx.y % n;
    const bool v = blockIdx.y >= n;
    const uint4* src = (const uint4*) (v ? vtab : ktab)[s0 + j];
    uint4* dst = (uint4*) (stage + (size_t) ((v ? nwin : 0) + j) * seg_u4 * 16);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < seg_u4; i += gridDim.x * blockDim.x) dst[i] = src[i];
}
// ---- STRATA_Q36_SPARSE: sparse decode attention over the cells past the KV_RAM window (junkinfer's method: this
// model's attention is broad, so a large exact fraction, ranked by an estimate of each page's attention mass)
constexpr int kSpPage = 32;                         // cells a page
constexpr int kSpPagesSeg = 4096 / kSpPage;         // pages a KV_RAM segment
/// Page statistics: the mean and the variance of each of K's 256 dims and the mean of V's over the page's 32 cells
/// (dequantized), as f16 at stab[seg][(page_in_seg * HKV + kvh) * kSpStat + {0: K mean, 256: K var, 512: V mean} + d].
/// Pages [pb, pb + n).  grid (n, HKV), block 256 (one thread a dim).
constexpr int kSpStat = 768;
__global__ void k_sp_stats(KvTab kt, __half* const* stab, int pb, int HKV, int qb) {
    const int page = pb + blockIdx.x, kvh = blockIdx.y, d = threadIdx.x;
    float m = 0.f, m2 = 0.f, mv = 0.f;
    for (int c = 0; c < kSpPage; ++c) {
        __half *sc, *sv;
        const int8_t* r = kv_row(nullptr, kt.k, kt.shift, kvh, page * kSpPage + c, HKV, 0, 256, qb, &sc);
        const int8_t* rv = kv_row(nullptr, kt.v, kt.shift, kvh, page * kSpPage + c, HKV, 0, 256, qb, &sv);
        const float x = __half2float(sc[d / qb]) * (float) r[d];
        const float dl = x - m;
        m += dl / (float) (c + 1);
        m2 += dl * (x - m);
        mv += __half2float(sv[d / qb]) * (float) rv[d];
    }
    __half* o = stab[page / kSpPagesSeg] + ((size_t) (page % kSpPagesSeg) * HKV + kvh) * kSpStat;
    o[d] = __float2half(m);
    o[256 + d] = __float2half(m2 / (float) kSpPage);
    o[512 + d] = __float2half(mv / (float) kSpPage);
}
/// Each page's estimated log attention mass for the step's rows: log n + q.mu / 16 + (q^2).var / 512 (junkinfer's
/// estimate, scale 1/sqrt(256)), the max over the T rows and the query heads of KV head kvh. Pages [pb, pb + n) ->
/// score[kvh][page - pb].  grid (n, HKV), block 256.
__global__ void k_sp_score(const float* q, int T, int H, int HKV, __half* const* stab, int pb, int n, float* score,
                           float* lm) {
    __shared__ float sh[kThreads];
    const int page = pb + blockIdx.x, kvh = blockIdx.y, d = threadIdx.x, G = H / HKV;
    const __half* st = stab[page / kSpPagesSeg] + ((size_t) (page % kSpPagesSeg) * HKV + kvh) * kSpStat;
    const float mu = __half2float(st[d]), var = __half2float(st[256 + d]);
    float best = -INFINITY;
    for (int t = 0; t < T; ++t)
        for (int g = 0; g < G; ++g) {
            const float qd = q[((size_t) t * H + kvh * G + g) * 256 + d];
            const float s = block_sum(qd * mu / 16.f + qd * qd * var / 512.f, sh);
            best = fmaxf(best, s);
            // the tail estimate's log-mass of this row and head (log 32 + s), lm[kvh][page][t * G + g]
            if (d == 0 && lm) lm[((size_t) kvh * n + blockIdx.x) * (T * G) + t * G + g] = s + 3.4657359f;
        }
    if (d == 0) score[(size_t) kvh * n + blockIdx.x] = best;
}
/// STRATA_Q36_SP_PREFILL: a prompt chunk's page scores. The estimate of k_sp_score for R sampled query rows of the chunk
/// (rows t = r / G * stride, head kvh * G + r % G), the max over them; a warp a sampled row, each lane 8 of the 256 dims.
/// grid (n, HKV), block 256.
__global__ void k_sp_score_pf(const float* q, int H, int HKV, int R, int stride, __half* const* stab, int pb, int n,
                              float* score) {
    __shared__ float wbest[8];
    const int page = pb + blockIdx.x, kvh = blockIdx.y, G = H / HKV;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const __half* st = stab[page / kSpPagesSeg] + ((size_t) (page % kSpPagesSeg) * HKV + kvh) * kSpStat;
    float mu[8], var[8];
    for (int i = 0; i < 8; ++i) { mu[i] = __half2float(st[lane * 8 + i]); var[i] = __half2float(st[256 + lane * 8 + i]); }
    float best = -INFINITY;
    for (int r = w; r < R; r += 8) {
        const int t = (r / G) * stride, g = r % G;
        const float* qr = q + ((size_t) t * H + kvh * G + g) * 256 + lane * 8;
        float a = 0.f, b = 0.f;
        for (int i = 0; i < 8; ++i) { const float qd = qr[i]; a += qd * mu[i]; b += qd * qd * var[i]; }
        float v = a / 16.f + b / 512.f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 32);
        best = fmaxf(best, v);
    }
    if (lane == 0) wbest[w] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -INFINITY;
        for (int i = 0; i < 8; ++i) m = fmaxf(m, wbest[i]);
        score[(size_t) kvh * n + blockIdx.x] = m;
    }
}
/// The sparse prompt's compact block in FP16 ([HKV][P][256], P = (S + nr) * 32): slot j < S the selected page sel[kvh][j],
/// then the nr pages from page rp (the local cells), each 32 cells dequantized from the segments.  grid (S + nr, HKV),
/// block 256.
__global__ void k_sp_gather_f16(KvTab kt, const int* sel, int S, int rp, int nr, int HKV, int qb, __half* k16, __half* v16) {
    const int j = blockIdx.x, kvh = blockIdx.y, d = threadIdx.x;
    const int page = j < S ? sel[(size_t) kvh * S + j] : rp + (j - S);
    const int P = (S + nr) * kSpPage;
    for (int c = 0; c < kSpPage; ++c) {
        __half *sk, *sv;
        const int8_t* rk = kv_row(nullptr, kt.k, kt.shift, kvh, page * kSpPage + c, HKV, 0, 256, qb, &sk);
        const int8_t* rv = kv_row(nullptr, kt.v, kt.shift, kvh, page * kSpPage + c, HKV, 0, 256, qb, &sv);
        const size_t o = ((size_t) kvh * P + (size_t) j * kSpPage + c) * 256 + d;
        k16[o] = __float2half(__half2float(sk[d / qb]) * (float) rk[d]);
        v16[o] = __float2half(__half2float(sv[d / qb]) * (float) rv[d]);
    }
}
/// mark[kvh][page - pb] = 1 for the selected pages (sel), 0 for the rest.  grid (HKV), block 256.
__global__ void k_sp_mark(const int* sel, int S, int pb, int n, unsigned char* mark) {
    const int kvh = blockIdx.x;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mark[(size_t) kvh * n + i] = 0;
    __syncthreads();
    for (int j = threadIdx.x; j < S; j += blockDim.x) mark[(size_t) kvh * n + sel[(size_t) kvh * S + j] - pb] = 1;
}
/// The tail estimate (STRATA_Q36_SP_TAIL): each unselected far page as one cell of its estimated mass (lm) and its
/// mean V, as one more partial of row (t, h) at slot `slot`: m = the largest log-mass, l = the sum of exp(lm - m),
/// acc = that sum weighted by the mean Vs.  grid (T * H), block 256.
__global__ void k_sp_tail(const float* lm, const unsigned char* mark, __half* const* stab, int pb, int n, int T, int H,
                          int HKV, int slot, int n_chunk, float* pacc, float* pm, float* pl) {
    __shared__ float sh[kThreads];
    const int row = blockIdx.x, t = row / H, h = row % H, G = H / HKV, kvh = h / G, g = h % G, d = threadIdx.x;
    const int TG = T * G;
    float mx = -INFINITY;
    for (int i = d; i < n; i += blockDim.x)
        if (!mark[(size_t) kvh * n + i]) mx = fmaxf(mx, lm[((size_t) kvh * n + i) * TG + t * G + g]);
    mx = block_max(mx, sh);
    float acc = 0.f, l = 0.f;
    if (mx > -INFINITY)
        for (int i = 0; i < n; ++i) {
            if (mark[(size_t) kvh * n + i]) continue;
            const int page = pb + i;
            const float w = expf(lm[((size_t) kvh * n + i) * TG + t * G + g] - mx);
            const __half* st = stab[page / kSpPagesSeg] + ((size_t) (page % kSpPagesSeg) * HKV + kvh) * kSpStat;
            acc += w * __half2float(st[512 + d]);
            l += w;
        }
    const size_t o = (size_t) row * n_chunk + slot;
    pacc[o * 256 + d] = acc;
    if (d == 0) { pm[o] = mx; pl[o] = l; }
}
/// The top S of score[kvh][0 .. n) per KV head, in page order: a threshold by bisection (count(>= t) >= S, 32
/// steps), then the pages at or above it in page order until S. Writes sel[kvh][0 .. S) (pages, pb added).
/// grid (HKV), block 1024.
__global__ void k_sp_select(const float* score, int n, int S, int pb, int* sel) {
    __shared__ float rlo[1024], rhi[1024];
    __shared__ int cnt[1024];
    const int kvh = blockIdx.x, tid = threadIdx.x;
    const float* sc = score + (size_t) kvh * n;
    float lo = INFINITY, hi = -INFINITY;
    for (int i = tid; i < n; i += 1024) { lo = fminf(lo, sc[i]); hi = fmaxf(hi, sc[i]); }
    rlo[tid] = lo; rhi[tid] = hi;
    __syncthreads();
    for (int o = 512; o > 0; o >>= 1) {
        if (tid < o) { rlo[tid] = fminf(rlo[tid], rlo[tid + o]); rhi[tid] = fmaxf(rhi[tid], rhi[tid + o]); }
        __syncthreads();
    }
    float t_lo = rlo[0], t_hi = rhi[0];
    __syncthreads();
    for (int it = 0; it < 32; ++it) {
        const float mid = 0.5f * (t_lo + t_hi);
        int c = 0;
        for (int i = tid; i < n; i += 1024) c += sc[i] >= mid;
        cnt[tid] = c;
        __syncthreads();
        for (int o = 512; o > 0; o >>= 1) { if (tid < o) cnt[tid] += cnt[tid + o]; __syncthreads(); }
        const int total = cnt[0];
        __syncthreads();
        if (total >= S) t_lo = mid; else t_hi = mid;
    }
    const float thr = t_lo;
    const int per = (n + 1023) / 1024, b = tid * per, e = min(n, b + per);
    int c = 0;
    for (int i = b; i < e; ++i) c += sc[i] >= thr;
    cnt[tid] = c;
    __syncthreads();
    if (tid == 0) {   // exclusive prefix over the threads' ranges
        int run = 0;
        for (int i = 0; i < 1024; ++i) { const int v = cnt[i]; cnt[i] = run; run += v; }
    }
    __syncthreads();
    int w = cnt[tid];
    for (int i = b; i < e && w < S; ++i)
        if (sc[i] >= thr) sel[(size_t) kvh * S + w++] = pb + i;
}
/// The compact cache: per KV head, slots 0 .. S - 1 the selected pages (sel[kvh][j]), then the recent pages from
/// page rp (nr of them), each 32 cells copied from the segments (values, then the scales) into a contiguous cache of
/// cap cells (the layout k_attn_part_q8 reads with ctx = cap).  grid (S + nr, HKV), block 256.
__global__ void k_sp_gather(KvTab kt, const int* sel, int S, int rp, int HKV, int qb, int cap, int8_t* ck, int8_t* cv) {
    const int j = blockIdx.x, kvh = blockIdx.y, tid = threadIdx.x;
    const int page = j < S ? sel[(size_t) kvh * S + j] : rp + (j - S);
    const int nb = 256 / qb;
    for (int kind = 0; kind < 2; ++kind) {
        int8_t* const* tab = kind ? kt.v : kt.k;
        int8_t* dst = kind ? cv : ck;
        __half* sc;
        const int8_t* src = kv_row(nullptr, tab, kt.shift, kvh, page * kSpPage, HKV, 0, 256, qb, &sc);
        // values: 32 cells x 256 B contiguous in the segment
        const uint4* s4 = (const uint4*) src;
        uint4* d4 = (uint4*) (dst + ((size_t) kvh * cap + (size_t) j * kSpPage) * 256);
        for (int i = tid; i < kSpPage * 256 / 16; i += blockDim.x) d4[i] = s4[i];
        // scales: 32 cells x nb halves, contiguous too
        __half* ds = (__half*) (dst + (size_t) HKV * cap * 256) + ((size_t) kvh * cap + (size_t) j * kSpPage) * nb;
        for (int i = tid; i < kSpPage * nb; i += blockDim.x) ds[i] = sc[i];
    }
}
/// k_kv_q8_f16 over a segmented cache (KvTab): cells [pb, pb + P) dequantized into compact [HKV][P][HD] halves.
__global__ void k_kv_q8_f16_tab(KvTab kt, __half* k16, __half* v16, int pb, int P, int HD, int HKV, int qb) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t) HKV * P * HD) return;
    const int64_t per = (int64_t) P * HD;
    const int kvh = (int) (i / per), rest = (int) (i % per);
    const int p = pb + rest / HD, d = rest % HD;
    __half *ks, *vs;
    const int8_t* kr = kv_row(nullptr, kt.k, kt.shift, kvh, p, HKV, 0, HD, qb, &ks);
    const int8_t* vr = kv_row(nullptr, kt.v, kt.shift, kvh, p, HKV, 0, HD, qb, &vs);
    k16[i] = __float2half(__half2float(ks[d / qb]) * (float) kr[d]);
    v16[i] = __float2half(__half2float(vs[d / qb]) * (float) vr[d]);
}
/// The blocked prompt attention (a context past the FP16 operands' room, STRATA_Q36_KV_RAM): one block of cells
/// [b0, b0 + Pb) of row (h, t): S16 = exp(S * scale - m_b) over the cells the row sees (0 past them), and the row's
/// m_b / l_b (max, sum; -inf / 0 when it sees none of the block).  grid (T, H), block 256.
__global__ void k_attn_softmax16_blk(const float* S, __half* S16, int ld, int T, int Pb, int b0, int pos0, float scale,
                                     float* mb, float* lb) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = max(0, min(Pb, pos0 + t + 1 - b0));
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum(sum, sh);
    for (int p = threadIdx.x; p < Pb; p += blockDim.x) out[p] = __float2half(p < n ? expf(row[p] * scale - mx) : 0.f);
    if (threadIdx.x == 0) {
        mb[(size_t) t * gridDim.y + h] = n > 0 ? mx : -INFINITY;
        lb[(size_t) t * gridDim.y + h] = sum;
    }
}
/// The blocked prompt attention's merge: row (t, h)'s block output ob (unnormalized, its m_b / l_b) into the running
/// acc / m / l (the online softmax); `first` starts them.  grid (T * H), block HD.
__global__ void k_attn_blk_merge(float* acc, const float* ob, float* m, float* l, const float* mb, const float* lb,
                                 int HD, bool first) {
    const int r = blockIdx.x, d = threadIdx.x;
    const float mo = first ? -INFINITY : m[r], lo = first ? 0.f : l[r];
    const float mbr = mb[r], lbr = lb[r];
    const float mn = fmaxf(mo, mbr);
    const float a = mo == -INFINITY ? 0.f : expf(mo - mn);
    const float b = mbr == -INFINITY ? 0.f : expf(mbr - mn);
    const size_t i = (size_t) r * HD + d;
    acc[i] = (first ? 0.f : acc[i] * a) + ob[i] * b;
    __syncthreads();
    if (d == 0) {
        m[r] = mn;
        l[r] = lo * a + lbr * b;
    }
}
/// The blocked prompt attention's end: acc / l times sigmoid(gate).  grid (T * H), block HD.
__global__ void k_attn_blk_end(float* acc, const float* l, const float* gate, int HD) {
    const size_t i = (size_t) blockIdx.x * HD + threadIdx.x;
    acc[i] = acc[i] / l[blockIdx.x] * sigm(gate[i]);
}
/// STRATA_Q36_FA=1: the prompt's attention fused (a flash-attention kernel; the GEMM path writes every score to memory
/// and reloads K / V per query head). A block takes kFaQ tokens x the kFaG query heads that share one KV head (64 rows),
/// walks the cells [b0, b0 + Pb) a row can see in tiles of kFaK, scores with FP16 dot products in FP32 (as the GEMMs),
/// an online softmax in FP32, and the values in FP32. q16 [T][H][256] f16; k16 / v16 [HKV][Pb][256] f16 (cells b0 ..).
/// FINAL: out [T][H][256] = O / l * sigmoid(gate); else the block of cells' unnormalized O, its max and sum (ob / mb / lb,
/// row t * H + h) for k_attn_blk_merge.  grid (ceil(T / kFaQ), HKV), block 256.
constexpr int kFaQ = 8, kFaG = 8, kFaK = 16, kFaHD = 256, kFaR = kFaQ * kFaG, kFaPad = 8;
template <bool FINAL>
__global__ void __launch_bounds__(256) k_fa_prompt(const __half* __restrict__ q16, const __half* __restrict__ k16,
                                                   const __half* __restrict__ v16, int T, int H, int Pb, int ld_cells, int b0, int pos0,
                                                   float scale, const float* __restrict__ gate, float* __restrict__ out,
                                                   float* __restrict__ mb, float* __restrict__ lb) {
    constexpr int HD = kFaHD, QS = HD + kFaPad;
    __shared__ __align__(16) __half sQ[kFaR * QS];
    __shared__ __align__(16) __half sK[kFaK * QS];
    __shared__ __align__(16) float sV[kFaK * HD];
    __shared__ float sS[kFaR][kFaK + 1];
    __shared__ float sM[kFaR], sL[kFaR], sA[kFaR];
    const int tid = threadIdx.x, kvh = blockIdx.y, t0 = blockIdx.x * kFaQ;
    // the Q tile: row r = token r / kFaG, head kvh * kFaG + r % kFaG (8 halves a load, 8 loads a thread)
    for (int i = tid; i < kFaR * HD / 8; i += 256) {
        const int r = i / (HD / 8), d = (i % (HD / 8)) * 8, t = t0 + r / kFaG;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (t < T) v = *(const uint4*) (q16 + ((size_t) t * H + kvh * kFaG + r % kFaG) * HD + d);
        *(uint4*) (sQ + r * QS + d) = v;
    }
    if (tid < kFaR) { sM[tid] = -INFINITY; sL[tid] = 0.f; }
    // O: rows ro, ro + 1, columns dg * 32 .. + 32 (64 accumulators)
    const int ro = (tid / 8) * 2, dg = tid % 8;
    float o0[32], o1[32];
    for (int i = 0; i < 32; ++i) o0[i] = o1[i] = 0.f;
    const int t_last = min(T - 1, t0 + kFaQ - 1);
    const int c_end = min(Pb, pos0 + t_last + 1 - b0);   // past it no row of the tile sees a cell
    const __half* kh = k16 + (size_t) kvh * ld_cells * HD;   // (a KV head's cells ld_cells apart: Pb, or ctx in place)
    const __half* vh = v16 + (size_t) kvh * ld_cells * HD;
    // scores: rows sr, sr + 1 x cells sc, sc + 1
    const int sr = (tid / 8) * 2, sc = (tid % 8) * 2;
    for (int c0 = 0; c0 < c_end; c0 += kFaK) {
        __syncthreads();   // (the previous tile's sV / sS reads are done; the Q tile and sM are written)
        for (int i = tid; i < kFaK * HD / 8; i += 256) {
            const int c = i / (HD / 8), d = (i % (HD / 8)) * 8;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (c0 + c < Pb) {
                kv = *(const uint4*) (kh + (size_t) (c0 + c) * HD + d);
                vv = *(const uint4*) (vh + (size_t) (c0 + c) * HD + d);
            }
            *(uint4*) (sK + c * QS + d) = kv;
            const __half* vp = (const __half*) &vv;
            for (int j = 0; j < 8; ++j) sV[c * HD + d + j] = __half2float(vp[j]);
        }
        __syncthreads();
        {   // S = Q K^T * scale, the causal mask
            float a00 = 0.f, a01 = 0.f, a10 = 0.f, a11 = 0.f;
            const __half* q0p = sQ + sr * QS;
            const __half* q1p = q0p + QS;
            const __half* k0p = sK + sc * QS;
            const __half* k1p = k0p + QS;
#pragma unroll 4
            for (int d = 0; d < HD; d += 8) {
                const uint4 qa = *(const uint4*) (q0p + d), qb = *(const uint4*) (q1p + d);
                const uint4 ka = *(const uint4*) (k0p + d), kb = *(const uint4*) (k1p + d);
                const __half2* qa2 = (const __half2*) &qa;
                const __half2* qb2 = (const __half2*) &qb;
                const __half2* ka2 = (const __half2*) &ka;
                const __half2* kb2 = (const __half2*) &kb;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
#if defined(STRATA_USE_HIP)
                    a00 = __builtin_amdgcn_fdot2(qa2[j], ka2[j], a00, false);
                    a01 = __builtin_amdgcn_fdot2(qa2[j], kb2[j], a01, false);
                    a10 = __builtin_amdgcn_fdot2(qb2[j], ka2[j], a10, false);
                    a11 = __builtin_amdgcn_fdot2(qb2[j], kb2[j], a11, false);
#else
                    const float2 fa = __half22float2(qa2[j]), fb = __half22float2(qb2[j]);
                    const float2 ga = __half22float2(ka2[j]), gb = __half22float2(kb2[j]);
                    a00 += fa.x * ga.x + fa.y * ga.y; a01 += fa.x * gb.x + fa.y * gb.y;
                    a10 += fb.x * ga.x + fb.y * ga.y; a11 += fb.x * gb.x + fb.y * gb.y;
#endif
                }
            }
            const int p0r = pos0 + t0 + sr / kFaG, p1r = pos0 + t0 + (sr + 1) / kFaG;   // the rows' cells
            const int ca = b0 + c0 + sc, cb = ca + 1;
            const bool okc_a = c0 + sc < Pb, okc_b = c0 + sc + 1 < Pb;
            sS[sr][sc] = (okc_a && ca <= p0r) ? a00 * scale : -INFINITY;
            sS[sr][sc + 1] = (okc_b && cb <= p0r) ? a01 * scale : -INFINITY;
            sS[sr + 1][sc] = (okc_a && ca <= p1r) ? a10 * scale : -INFINITY;
            sS[sr + 1][sc + 1] = (okc_b && cb <= p1r) ? a11 * scale : -INFINITY;
        }
        __syncthreads();
        {   // the online softmax: 4 threads a row, 4 cells each
            const int r = tid / 4, q = tid % 4;
            float mx = -INFINITY;
            for (int c = q * 4; c < q * 4 + 4; ++c) mx = fmaxf(mx, sS[r][c]);
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1, 4));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2, 4));
            const float mo = sM[r];
            const float mn = fmaxf(mo, mx);
            float sum = 0.f;
            for (int c = q * 4; c < q * 4 + 4; ++c) {
                const float p = mn == -INFINITY ? 0.f : expf(sS[r][c] - mn);
                sS[r][c] = p;
                sum += p;
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1, 4);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2, 4);
            if (q == 0) {
                const float a = (mo == -INFINITY) ? 0.f : expf(mo - mn);
                sA[r] = a;
                sL[r] = sL[r] * a + sum;
                sM[r] = mn;
            }
        }
        __syncthreads();
        {   // O = O * alpha + P V
            const float al0 = sA[ro], al1 = sA[ro + 1];
            for (int i = 0; i < 32; ++i) { o0[i] *= al0; o1[i] *= al1; }
            const int nc = min(kFaK, c_end - c0);
            for (int c = 0; c < nc; ++c) {
                const float p0 = sS[ro][c], p1 = sS[ro + 1][c];
                const float4* vp = (const float4*) (sV + c * HD + dg * 32);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const float4 v = vp[i];
                    o0[i * 4 + 0] += p0 * v.x; o0[i * 4 + 1] += p0 * v.y; o0[i * 4 + 2] += p0 * v.z; o0[i * 4 + 3] += p0 * v.w;
                    o1[i * 4 + 0] += p1 * v.x; o1[i * 4 + 1] += p1 * v.y; o1[i * 4 + 2] += p1 * v.z; o1[i * 4 + 3] += p1 * v.w;
                }
            }
        }
    }
    __syncthreads();
    for (int k = 0; k < 2; ++k) {
        const int r = ro + k, t = t0 + r / kFaG;
        if (t >= T) continue;
        const int h = kvh * kFaG + r % kFaG;
        const size_t base = ((size_t) t * H + h) * HD + dg * 32;
        const float* o = k ? o1 : o0;
        if (FINAL) {
            const float inv = sL[r] > 0.f ? 1.f / sL[r] : 0.f;
            for (int i = 0; i < 32; ++i) out[base + i] = o[i] * inv * sigm(gate[base + i]);
        } else {
            for (int i = 0; i < 32; ++i) out[base + i] = o[i];
            if (dg == 0) { mb[(size_t) t * H + h] = sM[r]; lb[(size_t) t * H + h] = sL[r]; }
        }
    }
}
/// The prompt's causal softmax: row (h, t) of S (ld floats, P cells) -> softmax(S * scale) over the cells
/// [0, pos0 + t], 0 past them, as FP16 probabilities (S16, the same row offsets, ld halves).  grid (T, H), block 256.
__global__ void k_attn_softmax16(const float* S, __half* S16, int ld, int T, int P, int pos0, float scale) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    const float* row = S + ((size_t) h * T + t) * ld;
    __half* out = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, row[p] * scale);
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(row[p] * scale - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x) out[p] = __float2half(p < n ? expf(row[p] * scale - mx) * inv : 0.f);
}
/// STRATA_Q36_ATT_F16: k_attn_softmax16 over FP16 scores already scaled (the GEMM's alpha), in place: row (h, t) of S16.
__global__ void k_attn_softmax16_h(__half* S16, int ld, int T, int P, int pos0) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    __half* row = S16 + ((size_t) h * T + t) * ld;
    const int n = min(P, pos0 + t + 1);
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, __half2float(row[p]));
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(__half2float(row[p]) - mx);
    sum = block_sum(sum, sh);
    const float inv = 1.0f / sum;
    for (int p = threadIdx.x; p < P; p += blockDim.x) row[p] = __float2half(p < n ? expf(__half2float(row[p]) - mx) * inv : 0.f);
}
/// The blocked variant (KV_RAM): the block's softmax over the cells the row sees, normalized within the block (an FP16
/// output stays in range), and m_b / l_b; k_h2f_rows multiplies the block's output by l_b again before the merge.
__global__ void k_attn_softmax16_h_blk(__half* S16, int ld, int T, int Pb, int b0, int pos0, float* mb, float* lb) {
    __shared__ float sh[kThreads];
    const int t = blockIdx.x, h = blockIdx.y;
    __half* row = S16 + ((size_t) h * T + t) * ld;
    const int n = max(0, min(Pb, pos0 + t + 1 - b0));
    float mx = -INFINITY;
    for (int p = threadIdx.x; p < n; p += blockDim.x) mx = fmaxf(mx, __half2float(row[p]));
    mx = block_max(mx, sh);
    float sum = 0.f;
    for (int p = threadIdx.x; p < n; p += blockDim.x) sum += expf(__half2float(row[p]) - mx);
    sum = block_sum(sum, sh);
    const float inv = sum > 0.f ? 1.0f / sum : 0.f;
    for (int p = threadIdx.x; p < Pb; p += blockDim.x) row[p] = __float2half(p < n ? expf(__half2float(row[p]) - mx) * inv : 0.f);
    if (threadIdx.x == 0) {
        mb[(size_t) t * gridDim.y + h] = n > 0 ? mx : -INFINITY;
        lb[(size_t) t * gridDim.y + h] = sum;
    }
}
/// FP16 attention output to FP32: out = o16 * sigmoid(gate) (gated) or out = o16, n values.
__global__ void k_h2f_gate(const __half* o16, const float* gate, float* out, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gate ? __half2float(o16[i]) * sigm(gate[i]) : __half2float(o16[i]);
}
/// out[r][d] = o16[r][d] * scale[r] (rows of HD): a block's normalized FP16 output back to k_attn_blk_merge's form.
__global__ void k_h2f_rows(const __half* o16, const float* scale, float* out, int HD, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(o16[i]) * scale[i / HD];
}
/// att *= sigmoid(gate) (the attention output gate), n values.
__global__ void k_gate_mul(float* att, const float* gate, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) att[i] *= sigm(gate[i]);
}

/// Picture rows over the embeddings of a chunk: x[r] = img[map[r]] for the rows with map[r] >= 0 (the rows of a picture's
/// <|image_pad|> cells).  grid (T), block 256.
__global__ void k_put_rows(float* x, const float* img, const int32_t* map, int n) {
    const int r = blockIdx.x;
    const int m = map[r];
    if (m < 0) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x) x[(size_t) r * n + i] = img[(size_t) m * n + i];
}

/// The prompt's routed entries sorted by expert for MMQ: bounds[x] .. bounds[x + 1] are expert x's rows; sorted row r
/// holds entry dst[r] (= t * K + k, the row of the per-slot outputs) of token src[r].  The order inside an expert is the
/// atomics' (no output depends on it: every row's products are its own).  cnt / cur: n_expert ints of scratch.
__global__ void k_route_count(const int32_t* ids, int n_ent, int32_t* cnt) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e < n_ent) atomicAdd(&cnt[ids[e]], 1);
}
__global__ void k_route_scan(const int32_t* cnt, int n_expert, int32_t* bounds, int32_t* cur, int32_t* max_rows) {
    if (threadIdx.x != 0) return;
    int pos = 0, mx = 0;
    for (int x = 0; x < n_expert; ++x) {
        bounds[x] = pos;
        cur[x] = pos;
        pos += cnt[x];
        mx = max(mx, cnt[x]);
    }
    bounds[n_expert] = pos;
    *max_rows = mx;
}
__global__ void k_route_scatter(const int32_t* ids, int n_ent, int K_, int32_t* cur, int32_t* dst, int32_t* src,
                                int32_t* inv) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n_ent) return;
    const int r = atomicAdd(&cur[ids[e]], 1);
    dst[r] = e;
    src[r] = e / K_;
    inv[e] = r;   // entry e's sorted row: where its down product lands
}

/// The greedy pick of one logits row: out = argmax (lowest id on ties).  Stage 1: `gridDim.x` blocks each write the
/// best of their stride into (pv, pi); stage 2 (one block, n = the stage-1 block count) reads those.
__global__ void k_argmax(const float* x, int n, int32_t* out, float* pv, int32_t* pi, bool stage2) {
    __shared__ float bv[kThreads];
    __shared__ int bi[kThreads];
    // row blockIdx.y: its logits (stage 1, n = the vocabulary) and its kArgBlocks partials
    x += (size_t) blockIdx.y * (stage2 ? 0 : n);
    pv += (size_t) blockIdx.y * 128;
    pi += (size_t) blockIdx.y * 128;
    out += blockIdx.y;
    float v = -INFINITY;
    int id = 0x7fffffff;
    if (stage2) {
        for (int i = threadIdx.x; i < n; i += blockDim.x)
            if (pv[i] > v || (pv[i] == v && pi[i] < id)) { v = pv[i]; id = pi[i]; }
    } else {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
            if (x[i] > v) { v = x[i]; id = i; }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = id;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int) threadIdx.x < s) {
            const float ov = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (stage2) *out = bi[0];
        else { pv[blockIdx.x] = bv[0]; pi[blockIdx.x] = bi[0]; }
    }
}
constexpr int kArgBlocks = 128;   // (the row stride of k_argmax's partials)

unsigned grid1(int64_t n, int b = kThreads) { return (unsigned) ((n + b - 1) / b); }

/// out[t][o] = W[o] . x[t] in FP32 for a prompt chunk's small F32 projections (the router, the GDN alpha / beta,
/// the shared expert's gate: n_out <= 256): a (16 TM) x 64 output tile per block, K in steps of 16 through shared
/// memory, TM x 4 outputs per thread (TM = 4 for the router, 1 for the 32-wide alpha / beta: more blocks).  hipBLAS
/// picked 8 x 8 / 16 x 16 tiles for these shapes (~10% of a prompt).  K % 4 == 0.
template <int TM>
__global__ void __launch_bounds__(256) k_sgemm_nt(const float* __restrict__ W, const float* __restrict__ x,
                                                  float* __restrict__ out, int K, int n_out, int T, int ldo, int kz) {
    constexpr int BM = 16 * TM, BN = 64, BK = 16;
    const int kb = blockIdx.z * kz, ke = min(K, kb + kz);   // split-K: this block's K range, its own partial
    out += (size_t) blockIdx.z * T * ldo;
    __shared__ float As[BK][BM + 4], Bs[BK][BN + 4];
    const int tid = threadIdx.x;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int tm = (tid / 16) * TM, tn = (tid % 16) * 4;
    const int lr = tid / 4, lk = (tid % 4) * 4;   // this thread's loads: row lr of each tile, k lk .. lk + 3
    float acc[TM][4] = {};
    for (int k0 = kb; k0 < ke; k0 += BK) {
        if (lr < BM) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + lr < T && k0 + lk < ke) a = *reinterpret_cast<const float4*>(x + (size_t) (m0 + lr) * K + k0 + lk);
            As[lk][lr] = a.x; As[lk + 1][lr] = a.y; As[lk + 2][lr] = a.z; As[lk + 3][lr] = a.w;
        }
        float4 b = make_float4(0.f, 0.f, 0.f, 0.f);
        if (n0 + lr < n_out && k0 + lk < ke) b = *reinterpret_cast<const float4*>(W + (size_t) (n0 + lr) * K + k0 + lk);
        Bs[lk][lr] = b.x; Bs[lk + 1][lr] = b.y; Bs[lk + 2][lr] = b.z; Bs[lk + 3][lr] = b.w;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ar[TM];
#pragma unroll
            for (int i = 0; i < TM; ++i) ar[i] = As[k][tm + i];
            const float4 bv = *reinterpret_cast<const float4*>(&Bs[k][tn]);
            const float br[4] = {bv.x, bv.y, bv.z, bv.w};
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(ar[i], br[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (m0 + tm + i < T && n0 + tn + j < n_out) out[(size_t) (m0 + tm + i) * ldo + n0 + tn + j] = acc[i][j];
}

/// out[i] = the sum of the split-K partials part[z][i] in order (deterministic), n values each.
__global__ void k_sum_splits(const float* part, float* out, int64_t n, int splits) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float a = 0.f;
    for (int z = 0; z < splits; ++z) a += part[(size_t) z * n + i];
    out[i] = a;
}

/// x f32 -> FP16 bits (the prompt GEMMs' activations).
__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

/// The GDN conv for T tokens in ONE launch (the prompt): one block per 128-channel head, thread = channel, the
/// 3-entry history carried in registers across the tokens and written back at the end; SiLU, then the L2 norm of the
/// q/k heads (block < qk_heads) per token - fused_gdn_conv_l2's arithmetic, token after token.  History [channel][3].
__global__ void __launch_bounds__(128) k_conv_seq(float* hist, const float* qkv, const float* w, float* h, int C,
                                                  int qk_heads, float eps, int T, bool write,
                                                  float* const* hist_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                  const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    __shared__ float part[4];
    if (hist_tab) {   // batch rows, grid y = group: group g's history is session slot_rows[g]'s; its rows are
                      // grow0[g] .. + gcnt[g] - 1 (one row at row g when there are no group arrays)
        const int g = blockIdx.y, r0 = grow0 ? grow0[g] : g;
        hist = hist_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        qkv += (size_t) r0 * C;
        h += (size_t) r0 * C;
    }
    const int c = blockIdx.x * 128 + threadIdx.x;
    float v0 = hist[c * 3], v1 = hist[c * 3 + 1], v2 = hist[c * 3 + 2];
    // TILE mode (a prompt chunk: grid y > 1, no batch tables): the tiles of kConvTile tokens run side by side.  A tile's history
    // is the three inputs before its first token (exactly the values the sequential loop carries); tile 0 reads the old history,
    // and, as the only writer, stores the new one: the last three of (old history, inputs).  The arithmetic per token is the same.
    int t_first = 0, t_end = T;
    const bool tiled = hist_tab == nullptr && gridDim.y > 1;
    if (tiled) {
        t_first = (int) blockIdx.y * kConvTile;
        t_end = min(T, t_first + kConvTile);
        if (t_first == 0 && write) {
            const float o0 = v0, o1 = v1, o2 = v2;
            const float old3[3] = {o0, o1, o2};
            for (int k = 0; k < 3; ++k) {
                const int idx = T - 3 + k;
                hist[c * 3 + k] = idx >= 0 ? qkv[(size_t) idx * C + c] : old3[idx + 3];
            }
        }
        if (t_first > 0) {
            v0 = qkv[(size_t) (t_first - 3) * C + c];
            v1 = qkv[(size_t) (t_first - 2) * C + c];
            v2 = qkv[(size_t) (t_first - 1) * C + c];
        }
        if (t_first >= T) return;
    }
    const float w0 = w[c * 4], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
    const bool qk = (int) blockIdx.x < qk_heads;
    for (int t = t_first; t < t_end; ++t) {
        const float x = qkv[(size_t) t * C + c];
        const float sum = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
        v0 = v1;
        v1 = v2;
        v2 = x;
        float y = sum / (1.0f + __expf(-sum));
        if (qk) {
            float sq = y * y;
            for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
            if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
            __syncthreads();
            const float ss = part[0] + part[1] + part[2] + part[3];
            __syncthreads();
            y *= rsqrtf(ss + eps);
        }
        h[(size_t) t * C + c] = y;
    }
    if (!write || tiled) return;   // a verify window: the history is commit()'s to move; tiles: tile 0 stored it
    hist[c * 3] = v0;
    hist[c * 3 + 1] = v1;
    hist[c * 3 + 2] = v2;
}

/// The gated delta rule + SiLU-gated norm for T tokens in ONE launch (the prompt): fused_gdn_step_norm's kernel
/// (block = one V head, 128 columns x 4 row groups, state [row][head][col]) with its 32 state rows per thread kept in
/// registers across the tokens.  h (T, C) = q | k | v, gate / beta (T, HV), z / y (T, ZV).
__global__ void __launch_bounds__(512) k_gdn_seq(float* state, const float* h, int C, const float* gate,
                                                 const float* beta, const float* z, const float* gamma, float eps,
                                                 float* y, int h_k, int h_v, int T, bool write,
                                                 float* const* state_tab = nullptr, const int32_t* slot_rows = nullptr,
                                                 const int32_t* grow0 = nullptr, const int32_t* gcnt = nullptr) {
    constexpr int S = 128, RG = 4, RPG = 32;
    if (state_tab) {   // batch rows, grid y = group (see k_conv_seq)
        const int g = blockIdx.y, r = grow0 ? grow0[g] : g;
        state = state_tab[slot_rows[g]];
        if (gcnt) T = gcnt[g];
        h += (size_t) r * C;
        gate += (size_t) r * h_v;
        beta += (size_t) r * h_v;
        z += (size_t) r * h_v * S;
        if (y) y += (size_t) r * h_v * S;
    }
    __shared__ float sk[S], sq[S];
    __shared__ float red_kv[RG][S];
    __shared__ float red_o[RG][S];
    __shared__ float wsum[S * RG / 32];
    const int head = blockIdx.x, col = threadIdx.x, rg = threadIdx.y, tid = rg * S + col;
    const int qh = head % h_k;
    const int ZV = h_v * S;
    float s[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
    for (int t = 0; t < T; ++t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { sk[tid] = ht[h_k * S + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
        __syncthreads();
        const float g = __expf(gate[(size_t) t * h_v + head]);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(s[r], sk[rg * RPG + r], kv);
        red_kv[rg][col] = kv;
        __syncthreads();
        const float kv_col = red_kv[0][col] + red_kv[1][col] + red_kv[2][col] + red_kv[3][col];
        const float delta = (ht[2 * h_k * S + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            s[r] = fmaf(g, s[r], sk[rg * RPG + r] * delta);
            o = fmaf(s[r], sq[rg * RPG + r], o);
        }
        red_o[rg][col] = o;
        __syncthreads();
        float oc = 0.0f, sq_part = 0.0f;
        if (rg == 0) {
            oc = (red_o[0][col] + red_o[1][col] + red_o[2][col] + red_o[3][col]) * rsqrtf((float) S);
            sq_part = oc * oc;
        }
        for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += __shfl_xor_sync(0xffffffffu, sq_part, o2);
        if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
        __syncthreads();
        if (rg == 0 && y != nullptr) {   // (the commit runs without outputs)
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            const float scale = rsqrtf(ss / (float) S + eps);
            const float zz = z[(size_t) t * ZV + head * S + col];
            y[(size_t) t * ZV + head * S + col] = oc * scale * gamma[col] * (zz / (1.0f + __expf(-zz)));
        }
        __syncthreads();   // sk / sq / the reductions are rewritten by the next token
    }
    if (!write) return;   // a verify window: the state is commit()'s to write
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
}

/// The prompt's gated delta rule with more parallelism: a state column's update needs only that column (kv, delta and
/// the output o all reduce over the rows), so a V head splits into kGdnColGroups blocks of 128 / kGdnColGroups columns,
/// each with 16 row groups of 8 rows (512 threads, 8 state values per thread in registers).  Writes the raw outputs
/// o (T, HV * 128); k_gdn_norm applies the SiLU-gated RMS norm.  grid (HV, kGdnColGroups), block (32, 16).
constexpr int kGdnColGroups = 4;
__global__ void __launch_bounds__(512) k_gdn_seq_cols(float* state, const float* h, int C, const float* gate,
                                                      const float* beta, float* o_raw, int h_k, int h_v, int T) {
    constexpr int S = 128, NC = S / kGdnColGroups, RG = 16, RPG = S / RG;
    // two barriers per token (was three): sk / sq are double-buffered, and the output sum of token t is done after token t + 1's first
    // barrier, off the chain of the recurrence (red_o is read between that barrier and the next one, written after it)
    __shared__ float sk[2][S], sq[2][S];
    __shared__ float red_kv[RG][NC];
    __shared__ float red_o[RG][NC];
    const int head = blockIdx.x, cl = threadIdx.x, rg = threadIdx.y, tid = rg * NC + cl;
    const int col = blockIdx.y * NC + cl;
    const int qh = head % h_k;
    float sv[RPG];
    float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
    const size_t row_stride = (size_t) h_v * S;
#pragma unroll
    for (int r = 0; r < RPG; ++r) sv[r] = base[r * row_stride];
    // the next token's inputs are loaded while this one is computed (the loads were the head of every iteration)
    float nsk = 0.0f, nsq = 0.0f, nvv = 0.0f, ngate = 0.0f, nbeta = 0.0f;
    auto load = [&](int t) {
        const float* ht = h + (size_t) t * C;
        if (tid < S) { nsk = ht[h_k * S + qh * S + tid]; nsq = ht[qh * S + tid]; }
        nvv = ht[2 * h_k * S + head * S + col];
        ngate = gate[(size_t) t * h_v + head];
        nbeta = beta[(size_t) t * h_v + head];
    };
    if (T > 0) load(0);
    for (int t = 0; t < T; ++t) {
        const int par = t & 1;
        if (tid < S) { sk[par][tid] = nsk; sq[par][tid] = nsq; }
        const float vcur = nvv, gcur = ngate, bcur = nbeta;
        if (t + 1 < T) load(t + 1);
        __syncthreads();
        if (t > 0 && rg == 0) {   // token t - 1's output
            float oc = 0.0f;
#pragma unroll
            for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
            o_raw[(size_t) (t - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
        }
        const float g = __expf(gcur);
        float kv = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) kv = fmaf(sv[r], sk[par][rg * RPG + r], kv);
        red_kv[rg][cl] = kv;
        __syncthreads();
        float kv_col = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) kv_col += red_kv[i][cl];
        const float delta = (vcur - g * kv_col) * bcur;
        float o = 0.0f;
#pragma unroll
        for (int r = 0; r < RPG; ++r) {
            sv[r] = fmaf(g, sv[r], sk[par][rg * RPG + r] * delta);
            o = fmaf(sv[r], sq[par][rg * RPG + r], o);
        }
        red_o[rg][cl] = o;
    }
    __syncthreads();
    if (T > 0 && rg == 0) {   // the last token's output
        float oc = 0.0f;
#pragma unroll
        for (int i = 0; i < RG; ++i) oc += red_o[i][cl];
        o_raw[(size_t) (T - 1) * h_v * S + head * S + col] = oc * rsqrtf((float) S);
    }
#pragma unroll
    for (int r = 0; r < RPG; ++r) base[r * row_stride] = sv[r];
}
/// y = rms(o) * gamma * SiLU(z) per (token, V head): grid (HV, T), block 128.
__global__ void __launch_bounds__(128) k_gdn_norm(const float* o_raw, const float* z, const float* gamma, float eps,
                                                  float* y, int h_v) {
    __shared__ float part[4];
    const size_t i = ((size_t) blockIdx.y * h_v + blockIdx.x) * 128 + threadIdx.x;
    const float oc = o_raw[i];
    float sq = oc * oc;
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffffu, sq, o);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sq;
    __syncthreads();
    const float ss = part[0] + part[1] + part[2] + part[3];
    const float scale = rsqrtf(ss / 128.0f + eps);
    const float zz = z[i];
    y[i] = oc * scale * gamma[threadIdx.x] * (zz / (1.0f + __expf(-zz)));
}

// ------------------------------------------------------------------------------------------------ the model
struct Mat {
    const uint8_t* d = nullptr;   // device, GGUF blocks
    int type = -1;
    int n_in = 0, n_out = 0;
};

struct Layer {
    bool attn = false;
    const float *attn_norm = nullptr, *post_norm = nullptr;
    // GDN
    Mat qkv, zg, ssm_out;
    const float *alpha = nullptr, *beta = nullptr, *dt = nullptr, *ssm_a = nullptr, *conv = nullptr, *ssm_norm = nullptr;
    // attention
    Mat wq, wk, wv, wo;
    const float *q_norm = nullptr, *k_norm = nullptr;
    // MoE
    const float* router = nullptr;   // [n_expert][N] f32
    float* router_ext = nullptr;     // the same with the shared expert's gate row appended: [n_expert + 1][N]
    Mat sh_gate, sh_up, sh_down;
    const float* sh_gate_inp = nullptr;   // [N]
    K::NativeExpertLayout L;
    uint8_t* blobs = nullptr;   // n_expert blobs, `stride` bytes apart (L.bytes rounded up to whole blocks of every
    size_t stride = 0;          // part's type, so MMQ can step over them by blocks), then kMmqTail zero bytes
                                // (hybrid mode: n_slots slots, see place_experts)
    // the GGUF's expert tensors (the mapped file stays open): place_experts builds the blobs from them
    const uint8_t *src_g = nullptr, *src_u = nullptr, *src_d = nullptr;
    size_t src_gsz = 0, src_dsz = 0;
    uint8_t* hblobs = nullptr;  // hybrid mode: every expert's blob in host memory, `stride` apart
    int n_slots = 0;            // hybrid mode: the experts resident in VRAM
};

struct Model {
    strata::core::ModelGeometry g;
    std::shared_ptr<strata::GgufModel> file;   // kept mapped: the experts are placed after the engine's buffers
    std::string path;                          // the GGUF (the expert blob cache is named after it)
    bool hybrid = false;                       // not every expert fits in VRAM: slots + the CPU pool
    // the packed Q8_0 decode copies (gguf weights, packed copy, bytes): given back when that lets every expert fit
    std::vector<std::tuple<const uint8_t*, uint8_t*, size_t>> packed;
    // the GGUF-native MTP block (blk.<n_layers>, loaded with --mtp): eh_proj over [enorm(e) | hnorm(h)], then an
    // attention + MoE layer, shared_head_norm, the main head
    bool has_mtp = false;
    Layer mtp;
    Mat eh_proj;
    const float *mtp_enorm = nullptr, *mtp_hnorm = nullptr, *mtp_shnorm = nullptr;
    int64_t top_k = 8;
    float eps = 1e-6f, rope_base = 1e7f;
    int n_rot = 64;
    std::vector<Layer> layers;
    const float* out_norm = nullptr;
    Mat head;
    const uint8_t* embd = nullptr;
    int embd_type = -1;
    size_t embd_row = 0;
};

constexpr size_t kMmqTail = 4096;
size_t gcd_sz(size_t a, size_t b) { return b ? gcd_sz(b, a % b) : a; }
size_t lcm_sz(size_t a, size_t b) { return a / gcd_sz(a, b) * b; }
size_t block_bytes(int type) {
    int be = 0, bb = 0;
    return strata::block_geometry((uint32_t) type, be, bb) ? (size_t) bb : 1;
}   // zeroed bytes after the last expert (MMQ reads a little past a matrix)

size_t row_bytes(int type, int64_t n) {
    int be = 0, bb = 0;
    if (!strata::block_geometry((uint32_t) type, be, bb)) return 0;
    return (size_t) (n / be) * (size_t) bb;
}

float bf16_to_f32(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// IEEE binary16 -> f32 by hand. __ushort_as_half is a __device__ intrinsic and
// upload_f32 runs on the host, so a CUDA toolkit older than 12.5 refuses the
// call; widen the bits here instead, the way bf16_to_f32 does.
float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    const uint32_t exponent = (h >> 10) & 0x1f;
    const uint32_t mantissa = h & 0x3ff;
    uint32_t u;
    if (exponent == 0) {
        if (mantissa == 0) {
            u = sign;                                               // +-0
        } else {
            uint32_t m = mantissa, shift = 0;                        // subnormal: normalize it
            while ((m & 0x400) == 0) {
                m <<= 1;
                ++shift;
            }
            u = sign | ((uint32_t) (127 - 14 - shift) << 23) | ((m & 0x3ff) << 13);
        }
    } else if (exponent == 0x1f) {
        u = sign | 0x7f800000u | (mantissa << 13);                   // inf / nan
    } else {
        u = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

/// A tensor as f32 on the device (F32 as it is, BF16 / F16 widened).
float* upload_f32(const strata::TensorInfo& t, const uint8_t* src) {
    std::vector<float> h((size_t) t.elements());
    if (t.type == 0) std::memcpy(h.data(), src, h.size() * 4);
    else if (t.type == 30)
        for (size_t i = 0; i < h.size(); ++i) h[i] = bf16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 1)   // F16
        for (size_t i = 0; i < h.size(); ++i)
            h[i] = f16_to_f32((uint16_t) (src[2 * i] | (src[2 * i + 1] << 8)));
    else if (t.type == 8 && h.size() % 32 == 0) {
        // Q8_0 (the Q8_0 file stores ssm_alpha / ssm_beta so): blocks of an fp16 scale and 32 int8, widened here
        for (size_t b = 0; b < h.size() / 32; ++b) {
            const uint8_t* q = src + b * 34;
            const float d = f16_to_f32((uint16_t) (q[0] | (q[1] << 8)));
            for (int j = 0; j < 32; ++j) h[b * 32 + j] = d * (float) (int8_t) q[2 + j];
        }
    } else {
        std::fprintf(stderr, "strata-q36: %s is %s, expected F32, BF16, F16 or Q8_0\n", t.name.c_str(), t.type_name());
        std::exit(1);
    }
    float* d = nullptr;
    CK(cudaMalloc(&d, h.size() * 4));
    CK(cudaMemcpy(d, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    return d;
}

bool load(const std::string& path, Model& m, std::string& err, bool want_mtp) {
    const auto t0 = Clock::now();
    m.file = std::make_shared<strata::GgufModel>(strata::GgufModel::open(path));
    m.path = path;
    const strata::GgufModel& gm = *m.file;
    if (!strata::core::geometry_from_gguf(gm.meta(), m.g, m.top_k, err)) return false;
    if (m.g.family != strata::ModelFamily::Qwen35Moe) {
        err = "strata-q36 runs Qwen3.6-35B-A3B (qwen35moe) files only";
        return false;
    }
    size_t checked = 0;
    if (!strata::core::check_gguf_layout(gm, m.g, err, &checked)) return false;
    const auto& g = m.g;
    if (g.head_dim != 256 || g.ssm_state_size != 128 || g.ssm_d_conv != 4 || g.n_expert > 512 || m.top_k > 15) {
        err = "a geometry this driver's kernels do not take (head 256, state 128, conv 4, <= 512 experts, top-k <= 15)";
        return false;
    }
    const std::string p = "qwen35moe.";
    if (const strata::MetaValue* v = gm.meta().get(p + "attention.layer_norm_rms_epsilon")) m.eps = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.freq_base")) m.rope_base = (float) v->num();
    if (const strata::MetaValue* v = gm.meta().get(p + "rope.dimension_count")) m.n_rot = (int) v->u;
    uint64_t bytes = 0;
    auto find = [&](const std::string& name, size_t& s) -> const strata::TensorInfo* {
        const strata::TensorInfo* t = gm.find(name, &s);
        if (!t) {
            std::fprintf(stderr, "strata-q36: missing %s\n", name.c_str());
            std::exit(1);
        }
        return t;
    };
    auto f32 = [&](const std::string& name) -> const float* {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        bytes += t->elements() * 4;
        return upload_f32(*t, gm.shard(s).tensor_data(*t));
    };
    auto mat = [&](const std::string& name) -> Mat {
        size_t s = 0;
        const strata::TensorInfo* t = find(name, s);
        Mat w;
        w.type = (int) t->type;
        w.n_in = (int) t->shape[0];
        w.n_out = (int) t->shape[1];
        if (!K::native_mmvq_supported(w.type)) {
            std::fprintf(stderr, "strata-q36: %s is %s, which the native MMVQ does not take\n", name.c_str(), t->type_name());
            std::exit(1);
        }
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        uint8_t* d = nullptr;
        CK(cudaMalloc(&d, nb));
        CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
        bytes += nb;
        w.d = d;
        // STRATA_Q8_PACKED (on by default here): the lossless qs / d plane copy the Q8_0 decode kernels read instead,
        // bitwise the same outputs
        if (w.type == 8 && K::native_q8_0_packed_enabled() && K::native_q8_0_packed_eligible(w.n_in, w.n_out)) {
            std::vector<uint8_t> packed(nb);
            K::native_q8_0_pack_host(gm.shard(s).tensor_data(*t), packed.data(), w.n_in, w.n_out);
            uint8_t* pd = nullptr;
            CK(cudaMalloc(&pd, nb));
            CK(cudaMemcpy(pd, packed.data(), nb, cudaMemcpyHostToDevice));
            K::native_q8_0_packed_register(d, pd, w.n_in, w.n_out);
            m.packed.emplace_back(d, pd, (size_t) nb);
            bytes += nb;
        }
        return w;
    };
    auto L = [](int64_t l, const char* s) { return "blk." + std::to_string(l) + "." + s; };
    m.layers.resize((size_t) g.n_layers);
    // the MTP block whenever the file has one, with or without drafts (--mtp): the VRAM it takes is then the same
    // either way, so the hybrid mode's slot tables - and with them the greedy text - do not depend on --mtp
    (void) want_mtp;
    m.has_mtp = g.n_mtp_layers > 0;
    for (int64_t l = 0; l < g.n_layers + (m.has_mtp ? 1 : 0); ++l) {
        const bool is_mtp = l == g.n_layers;
        Layer& y = is_mtp ? m.mtp : m.layers[(size_t) l];
        y.attn = is_mtp || strata::core::is_qsa_layer(g, l);
        y.attn_norm = f32(L(l, "attn_norm.weight"));
        y.post_norm = f32(L(l, "post_attention_norm.weight"));
        if (!y.attn) {
            y.qkv = mat(L(l, "attn_qkv.weight"));
            y.zg = mat(L(l, "attn_gate.weight"));
            y.ssm_out = mat(L(l, "ssm_out.weight"));
            y.alpha = f32(L(l, "ssm_alpha.weight"));
            y.beta = f32(L(l, "ssm_beta.weight"));
            y.dt = f32(L(l, "ssm_dt.bias"));
            y.ssm_a = f32(L(l, "ssm_a"));
            y.conv = f32(L(l, "ssm_conv1d.weight"));
            y.ssm_norm = f32(L(l, "ssm_norm.weight"));
        } else {
            y.wq = mat(L(l, "attn_q.weight"));
            y.wk = mat(L(l, "attn_k.weight"));
            y.wv = mat(L(l, "attn_v.weight"));
            y.wo = mat(L(l, "attn_output.weight"));
            y.q_norm = f32(L(l, "attn_q_norm.weight"));
            y.k_norm = f32(L(l, "attn_k_norm.weight"));
        }
        y.router = f32(L(l, "ffn_gate_inp.weight"));
        y.sh_gate = mat(L(l, "ffn_gate_shexp.weight"));
        y.sh_up = mat(L(l, "ffn_up_shexp.weight"));
        y.sh_down = mat(L(l, "ffn_down_shexp.weight"));
        y.sh_gate_inp = f32(L(l, "ffn_gate_inp_shexp.weight"));
        CK(cudaMalloc(&y.router_ext, (size_t) (g.n_expert + 1) * g.n_embd * 4));
        CK(cudaMemcpy(y.router_ext, y.router, (size_t) g.n_expert * g.n_embd * 4, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(y.router_ext + (size_t) g.n_expert * g.n_embd, y.sh_gate_inp, (size_t) g.n_embd * 4,
                      cudaMemcpyDeviceToDevice));
        bytes += (size_t) (g.n_expert + 1) * g.n_embd * 4;
        // the routed experts: [gate | up | down] per expert, the native expert layout
        size_t sg = 0, su = 0, sd = 0;
        const strata::TensorInfo* tg = find(L(l, "ffn_gate_exps.weight"), sg);
        const strata::TensorInfo* tu = find(L(l, "ffn_up_exps.weight"), su);
        const strata::TensorInfo* td = find(L(l, "ffn_down_exps.weight"), sd);
        if (tg->type != tu->type) {
            err = L(l, "ffn_gate_exps / ffn_up_exps differ in type");
            return false;
        }
        if (!K::native_expert_supported((int) tg->type, (int) td->type, g.n_embd, g.n_ff)) {
            err = L(l, "experts are ") + tg->type_name() + "/" + td->type_name() + ", which the native expert kernels " +
                  "do not take (build with -DSTRATA_Q6K_EXPERTS=ON for Q6_K)";
            return false;
        }
        y.L = K::native_expert_layout((int) tg->type, (int) td->type, g.n_embd, g.n_ff);
        const size_t gsz = (size_t) g.n_ff * y.L.gu_row, dsz = (size_t) g.n_embd * y.L.d_row;
        // a stride in whole blocks of both parts' types (MMQ steps over the experts by blocks), 16-byte aligned
        const size_t al = lcm_sz(lcm_sz(block_bytes(y.L.gu_type), block_bytes(y.L.d_type)), 16);
        y.stride = (y.L.bytes + al - 1) / al * al;
        y.src_g = gm.shard(sg).tensor_data(*tg);
        y.src_u = gm.shard(su).tensor_data(*tu);
        y.src_d = gm.shard(sd).tensor_data(*td);
        y.src_gsz = gsz;
        y.src_dsz = dsz;
    }
    if (m.has_mtp) {
        const int64_t l = g.n_layers;
        m.eh_proj = mat(L(l, "nextn.eh_proj.weight"));
        m.mtp_enorm = f32(L(l, "nextn.enorm.weight"));
        m.mtp_hnorm = f32(L(l, "nextn.hnorm.weight"));
        m.mtp_shnorm = f32(L(l, "nextn.shared_head_norm.weight"));
    }
    m.out_norm = f32("output_norm.weight");
    m.head = mat("output.weight");
    {
        size_t s = 0;
        const strata::TensorInfo* t = find("token_embd.weight", s);
        m.embd_type = (int) t->type;
        m.embd_row = row_bytes(m.embd_type, (int64_t) t->shape[0]);
        const uint64_t nb = strata::tensor_payload_bytes(*t);
        // STRATA_Q36_EMBD_HOST (default: on a card under 12 GiB): the token embedding table (0.5 GiB for UD-Q4_K_M) in
        // pinned host memory, read by the embedding kernel through its device-mapped address (one ~2 KB row per token
        // over PCIe) instead of taking VRAM the expert slots and the prompt chunk can use. Same bytes, same rows.
        bool host_embd = false;
        {
            const char* v = std::getenv("STRATA_Q36_EMBD_HOST");
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            host_embd = v ? v[0] == '1' : tot < ((size_t) 12 << 30);
        }
        uint8_t* d = nullptr;
        if (host_embd) {
            uint8_t* hp = nullptr;
            if (cudaHostAlloc((void**) &hp, nb, cudaHostAllocMapped) == cudaSuccess &&
                cudaHostGetDevicePointer((void**) &d, hp, 0) == cudaSuccess) {
                std::memcpy(hp, gm.shard(s).tensor_data(*t), nb);
                std::fprintf(stderr, "strata-q36: token embeddings in pinned host memory (%.2f GiB of VRAM left to the "
                             "experts)\n", (double) nb / (1 << 30));
            } else {
                (void) cudaGetLastError();
                if (hp) CK(cudaFreeHost(hp));
                d = nullptr;
                host_embd = false;
            }
        }
        if (!host_embd) {
            CK(cudaMalloc(&d, nb));
            CK(cudaMemcpy(d, gm.shard(s).tensor_data(*t), nb, cudaMemcpyHostToDevice));
            bytes += nb;
        }
        m.embd = d;
    }
    std::fprintf(stderr, "strata-q36: %s: %zu tensors (layout checked), %.2f GiB on the GPU in %.1f s (the routed "
                 "experts are placed after the engine's buffers)\n", path.c_str(), checked, (double) bytes / (1 << 30),
                 ms_since(t0) / 1000.0);
    return true;
}

/// Expert e of layer L as its native blob [gate | up | down] (L.L.bytes) at dst (host memory).
void expert_blob(const Layer& L, int64_t e, uint8_t* dst) {
    std::memcpy(dst, L.src_g + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.up_off, L.src_u + (size_t) e * L.src_gsz, L.src_gsz);
    std::memcpy(dst + L.L.down_off, L.src_d + (size_t) e * L.src_dsz, L.src_dsz);
}

// ------------------------------------------------------------------------------------------------ the engine
/// The KV cache's cells: STRATA_Q36_KV = f16 (the default: half f32's VRAM per context token), f32, q8_0 (llama.cpp's
/// block: 32 int8 values and one f16 scale, 0.53 of f16's bytes) or int8 (one f16 scale per 256-value cell row, 0.5 of
/// f16's).  The older switch STRATA_Q36_KV_F16=0 means f32.  The 8-bit caches keep a layer's values, then its scales
/// (see k_attn_part_q8).
enum KvMode { KV_F32, KV_F16, KV_Q8 };
struct KvCfg {
    KvMode mode;
    int qb;   // the 8-bit caches' block: 32 (q8_0) or 256 (int8)
};
const KvCfg& kv_cfg() {
    static const KvCfg c = [] {
        KvCfg r{KV_F16, 32};
        if (const char* v = std::getenv("STRATA_Q36_KV_F16"); v && v[0] == '0') r.mode = KV_F32;
        if (const char* v = std::getenv("STRATA_Q36_KV")) {
            const std::string k(v);
            if (k == "f32") r.mode = KV_F32;
            else if (k == "f16") r.mode = KV_F16;
            else if (k == "q8_0" || k == "q8") r = {KV_Q8, 32};
            else if (k == "int8" || k == "i8") r = {KV_Q8, 256};
            else {
                std::fprintf(stderr, "strata-q36: STRATA_Q36_KV=%s: use f16, f32, q8_0 or int8\n", v);
                std::exit(1);
            }
        }
        return r;
    }();
    return c;
}
bool kv16() { return kv_cfg().mode == KV_F16; }
bool kvq8() { return kv_cfg().mode == KV_Q8; }
int kv_qb() { return kv_cfg().qb; }
/// STRATA_Q36_KV_DOT8 (default 1): an 8-bit cache's decode scores as int8 x int8 dp4a (k_attn_part_q8); =0 scales the
/// cells against the float query.
bool kv_dot8() {
    static const bool on = [] {
        const char* v = std::getenv("STRATA_Q36_KV_DOT8");
        return !(v && v[0] == '0');
    }();
    return on;
}
const char* kv_name() { return kvq8() ? (kv_qb() == 32 ? "q8_0" : "int8") : kv16() ? "f16" : "f32"; }
/// The bytes of n cache elements (whole cells: n a multiple of the 8-bit block)
size_t kv_bytes(size_t n) { return kvq8() ? n + n / (size_t) kv_qb() * 2 : n * (kv16() ? 2 : 4); }
/// STRATA_Q36_KV_RAM=1 (opt-in, an 8-bit cache, one session): the cache in segments of kSeg cells (KvTab). The first
/// `win` cells (STRATA_Q36_KV_WINDOW, default 32768) are in VRAM, the rest in pinned host memory the GPU reads over
/// PCIe, allocated as the context reaches it. The prompt attention's FP16 operands hold `win` cells; a longer context
/// runs it in blocks of that many cells (an online softmax across them). A context up to the window computes exactly
/// what the plain cache computes.
struct KvRam {
    bool on = false;
    int win = 32768;
    static constexpr int kShift = 12, kSeg = 1 << kShift;   // 4096 cells a segment
};
const KvRam& kv_ram() {
    static const KvRam r = [] {
        KvRam k;
        const char* v = std::getenv("STRATA_Q36_KV_RAM");
        k.on = v && v[0] == '1';
        if (const char* w = std::getenv("STRATA_Q36_KV_WINDOW")) k.win = std::max(KvRam::kSeg, std::atoi(w) / KvRam::kSeg * KvRam::kSeg);
        return k;
    }();
    return r;
}
float* dalloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, n * 4));
    CK(cudaMemset(p, 0, n * 4));
    return p;
}
/// n KV cells' elements (see kv_cfg), zeroed; kept as float* and cast to the element type by the kernels
float* kv_alloc(size_t n) {
    float* p = nullptr;
    CK(cudaMalloc(&p, kv_bytes(n)));
    CK(cudaMemset(p, 0, kv_bytes(n)));
    return p;
}
template <class T> T* dalloc_t(size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, n * sizeof(T)));
    CK(cudaMemset(p, 0, n * sizeof(T)));
    return p;
}

/// A picture in a prompt: its first cell, its rows, and the hash the conversation cache compares it by.
struct ImgKey {
    int64_t cell = 0, n = 0;
    uint64_t hash = 0;
    bool operator==(const ImgKey& o) const { return cell == o.cell && n == o.n && hash == o.hash; }
};

/// The prompt's expert stream from a PAGEABLE arena (the file-mapped blob cache, or the malloc'd copy): a
/// cudaMemcpyAsync from pageable memory is done by the calling thread (HIP on Windows measured 4.5 s of a 7.4 s
/// 2048-row chunk inside stream_kick, ~3.9 GB/s, with the GPU's work queued behind it). Here a copier thread takes
/// each layer's copy list: device-resident runs as device-to-device copies, host runs through a ring of pinned chunks
/// (filled by `threads` memcpy threads, sent by DMA on the copy stream, each chunk reused after its copy's event).
/// The engine thread only waits when it needs a layer (stream_take). Same bytes into the same staging buffer.
/// STRATA_Q36_RING=0 keeps the direct copies; STRATA_Q36_RING_MIB (16) and STRATA_Q36_RING_THREADS (3) tune it.
struct ExpertStreamer {
    struct Seg {
        uint8_t* dst;
        const uint8_t* src;
        size_t n;
        bool dev;
    };
    struct Job {
        uint64_t id;
        std::vector<Seg> segs;
        cudaEvent_t wait_ev;   // recorded by the engine before the job: the staging buffer's last reader
        cudaEvent_t done_ev;   // recorded here after the job's copies: the engine's stream waits on it
    };
    cudaStream_t cst = nullptr;
    cudaStream_t cst2 = nullptr;   // a second copy stream (STRATA_Q36_RING_STREAMS=2): ring chunks alternate between them
    cudaEvent_t join_ev = nullptr;
    size_t chunk = 0;
    std::vector<uint8_t*> ring;
    std::vector<cudaEvent_t> ring_ev;
    std::vector<char> ring_busy;
    size_t next = 0;
    std::thread th;
    std::mutex mu;
    std::condition_variable cv, cv_done;
    std::deque<Job> q;
    uint64_t submitted = 0, done = 0;
    bool stop = false;
    // the parallel memcpy (the copier thread takes a share too)
    std::vector<std::thread> helpers;
    std::mutex hm;
    std::condition_variable hcv, hdone;
    uint8_t* h_dst = nullptr;
    const uint8_t* h_src = nullptr;
    size_t h_n = 0, h_part = 0;
    uint64_t h_gen = 0;
    int h_left = 0;

    ExpertStreamer(cudaStream_t s, size_t chunk_bytes, int n_ring, int threads, int streams) : cst(s), chunk(chunk_bytes) {
        if (streams > 1) {
            CK(cudaStreamCreateWithFlags(&cst2, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&join_ev, cudaEventDisableTiming));
        }
        for (int i = 0; i < n_ring; ++i) {
            uint8_t* p = nullptr;
            CK(cudaHostAlloc((void**) &p, chunk, cudaHostAllocDefault));
            ring.push_back(p);
            cudaEvent_t e;
            CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
            ring_ev.push_back(e);
            ring_busy.push_back(0);
        }
        for (int i = 1; i < threads; ++i) helpers.emplace_back([this, i, threads] { helper(i, threads); });
        th = std::thread([this, threads] { run(threads); });
    }
    ~ExpertStreamer() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
        }
        cv.notify_all();
        if (th.joinable()) th.join();
        {
            std::lock_guard<std::mutex> lk(hm);
            h_n = 0;
            ++h_gen;
            h_dst = nullptr;
        }
        hcv.notify_all();
        for (auto& t : helpers) t.join();
    }
    uint64_t submit(std::vector<Seg> segs, cudaEvent_t wait_ev, cudaEvent_t done_ev) {
        std::lock_guard<std::mutex> lk(mu);
        Job j{++submitted, std::move(segs), wait_ev, done_ev};
        q.push_back(std::move(j));
        cv.notify_one();
        return submitted;
    }
    void wait(uint64_t id) {   // the job's copies are queued on cst and done_ev is recorded
        std::unique_lock<std::mutex> lk(mu);
        cv_done.wait(lk, [&] { return done >= id; });
    }

  private:
    void helper(int i, int n) {
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(hm);
            hcv.wait(lk, [&] { return h_gen != seen; });
            seen = h_gen;
            if (!h_dst) return;   // the destructor's wake-up
            uint8_t* d = h_dst;
            const uint8_t* s = h_src;
            const size_t len = h_n, part = h_part;
            lk.unlock();
            const size_t b = (size_t) i * part, e = std::min(len, b + part);
            if (b < e) std::memcpy(d + b, s + b, e - b);
            lk.lock();
            if (--h_left == 0) hdone.notify_one();
        }
    }
    void par_copy(uint8_t* d, const uint8_t* s, size_t n, int threads) {
        if (threads <= 1 || n < ((size_t) 1 << 20)) {
            std::memcpy(d, s, n);
            return;
        }
        const size_t part = (n / (size_t) threads + 4095) & ~(size_t) 4095;
        {
            std::lock_guard<std::mutex> lk(hm);
            h_dst = d; h_src = s; h_n = n; h_part = part;
            h_left = threads - 1;
            ++h_gen;
        }
        hcv.notify_all();
        std::memcpy(d, s, std::min(n, part));   // share 0
        std::unique_lock<std::mutex> lk(hm);
        hdone.wait(lk, [&] { return h_left == 0; });
    }
    void run(int threads) {
        for (;;) {
            Job j;
            {
                std::unique_lock<std::mutex> lk(mu);
                cv.wait(lk, [&] { return stop || !q.empty(); });
                if (stop && q.empty()) return;
                j = std::move(q.front());
                q.pop_front();
            }
            CK(cudaStreamWaitEvent(cst, j.wait_ev, 0));
            if (cst2) CK(cudaStreamWaitEvent(cst2, j.wait_ev, 0));
            int alt = 0;
            for (const Seg& sg : j.segs) {
                if (sg.dev) {
                    CK(cudaMemcpyAsync(sg.dst, sg.src, sg.n, cudaMemcpyDeviceToDevice, cst));
                    continue;
                }
                for (size_t off = 0; off < sg.n; off += chunk) {
                    const size_t n = std::min(chunk, sg.n - off);
                    const size_t r = next;
                    next = (next + 1) % ring.size();
                    if (ring_busy[r]) CK(cudaEventSynchronize(ring_ev[r]));
                    par_copy(ring[r], sg.src + off, n, threads);
                    cudaStream_t cs = (cst2 && (alt++ & 1)) ? cst2 : cst;
                    CK(cudaMemcpyAsync(sg.dst + off, ring[r], n, cudaMemcpyHostToDevice, cs));
                    CK(cudaEventRecord(ring_ev[r], cs));
                    ring_busy[r] = 1;
                }
            }
            if (cst2) {   // the job is done when both streams' copies are
                CK(cudaEventRecord(join_ev, cst2));
                CK(cudaStreamWaitEvent(cst, join_ev, 0));
            }
            CK(cudaEventRecord(j.done_ev, cst));
            {
                std::lock_guard<std::mutex> lk(mu);
                done = j.id;
            }
            cv_done.notify_all();
        }
    }
};

struct Engine {
    const Model* m = nullptr;
    cudaStream_t st = nullptr;
    int ctx = 32768, tmax = 512;
    int64_t pos = 0;   // cells holding a processed token
    // activations [tmax][...]
    float *x, *h, *y, *qkv, *hc, *z, *alpha, *beta, *gdn_y, *qg, *kx, *vx, *q, *gate, *att, *logits_r, *eparts, *moe_out,
        *sg, *su, *shh, *sy, *sgate, *ewts, *logits;
    float* hmain = nullptr;   // [tmax][N] the main model's final normed hidden of each row (the MTP seeds)
    float* hlast = nullptr;   // [N] that of the last processed position (zeros before the first): the current session's
    void *q8, *xq, *escr;
    int32_t *ids, *grp_start, *n_groups, *ent_dst, *ent_tok, *tok, *pos0, *pick;
    int32_t* hpin = nullptr;   // pinned staging: [0] decode token, [1] position, [2..9] picks, [10] max rows, [16..] tokens
    unsigned long long* grp_ptr;
    float *pacc, *pm, *pl, *amv;
    int32_t* ami;
    int n_chunk = 0, att_rows = 0;   // attention partials: att_rows query rows x n_chunk chunks
    std::vector<float*> conv, S, kc, vc;   // per layer; the MTP block's K / V at index n_layers: the CURRENT session's (select)
    // verify windows (T <= kVerify): the GDN inputs of every row per layer, kept for the commit
    static constexpr int kVerify = 8;
    std::vector<float*> v_qkv, v_hc, v_alpha, v_beta;
    // the MTP block (when loaded)
    float *mtp_cat = nullptr, *hmtp = nullptr, *logits_m = nullptr;
    int32_t *tok_m = nullptr, *pos0_m = nullptr, *pick_m = nullptr;
    // graphs: [0] the decode step, [T] a verify window of T rows, mtp1 the MTP draft step
    // (the graphs hold the current session's state pointers as kernel arguments: one set per session)
    struct GSet {
        cudaGraphExec_t g1 = nullptr, gv[kVerify + 1] = {}, gm1 = nullptr;
    };
    std::vector<GSet> gsets;
    GSet& gs() { return gsets[(size_t) cur]; }
    strata::prefill::Gemm gemm;
    cublasHandle_t blas = nullptr;
    float* att_s = nullptr;            // the prompt attention's scores / probabilities, kAttScoreFloats floats
    static constexpr int kSplitK = 16;
    float* sk_part = nullptr;          // the split-K partials of the F32 projections: [kSplitK][tmax][<= 257]
    __half* att_p16 = nullptr;         // the probabilities in FP16 (kAttScoreFloats halves)
    __half *k16 = nullptr, *v16 = nullptr, *q16 = nullptr;   // the prompt attention's FP16 operands
    // 256 MB (16 M floats when the experts' fit is close); STRATA_Q36_ATT_MFLOATS sets it (millions of floats): more
    // query rows per attention GEMM at a long context (64 M: ~200 rows at 20K cells)
    size_t kAttScoreFloats = [] {
        const char* v = std::getenv("STRATA_Q36_ATT_MFLOATS");
        return (size_t) (v ? std::max(4, std::atoi(v)) : 64) << 20;
    }();
    // the prompt's MMQ experts
    bool use_mmq = false;
    std::unique_ptr<MMQ::Context> mctx;
    int32_t *rcnt = nullptr, *rcur = nullptr, *rbounds = nullptr, *rmax = nullptr, *ident = nullptr, *rinv = nullptr;
    void *mq_x = nullptr, *mq_hq = nullptr;
    float *mq_gu = nullptr, *mq_h = nullptr;
    __half* x16 = nullptr;
    bool use_graph = true;
    int mtp_d = 0;   // MTP drafts per speculative step (0: none; greedy requests only)

    // ---- sessions: the conversation states.  One for a single-request engine; --batch N keeps N, each with its own
    // recurrent state and KV cache, so N requests run at once: their rows share one decode window (forward_rows).  The
    // members conv / S / kc / vc / pos / hlast above are the CURRENT session's (select() swaps them).
    struct Sess {
        std::vector<float*> conv, S, kc, vc;
        float* hlast = nullptr;
        int64_t pos = 0;
        int32_t rdelta = 0;                 // rotary offset of the text after the last picture (see k_attn_prep)
        std::vector<int32_t> live;          // the tokens whose state it holds (cell i = live[i])
        std::vector<ImgKey> imgs;           // the pictures among them
        bool active = false;                // a batch slot is decoding in it
        uint64_t used = 0;                  // when it was last used (the LRU pick)
    };
    std::vector<Sess> sess;
    int n_sess = 1, cur = 0;
    uint64_t use_clock = 0;
    // device tables for batch rows: per state index (layer, then the MTP block) the session pointers
    float **d_conv_tab = nullptr, **d_S_tab = nullptr, **d_kc_tab = nullptr, **d_vc_tab = nullptr;
    int32_t *d_rcell = nullptr, *d_rslot = nullptr, *d_rdelta = nullptr;   // [kVerify] the rows' cell / session / rotary offset
    // the GDN layers run a slot's rows in sequence: per group (a slot) its session, its first row and its row count
    int32_t *d_gslot = nullptr, *d_grow0 = nullptr, *d_gcnt = nullptr;
    int rows_G = 0;                                                        // groups of the rows being run
    int32_t* hrow = nullptr;                                                // pinned staging for the rows
    bool rows_mode = false;                                                 // layer(): the rows are sessions' (not consecutive cells)
    bool bmtp = false;                                                      // --batch-mtp: batch windows verify MTP proposals
    int64_t rows_pmax = 0;                                                  // (eager) the highest cell of the rows
    // pictures (--vision): the request's rows replace the embeddings of its <|image_pad|> cells; M-RoPE
    bool vision = false;
    int32_t* d_mtab = nullptr;          // [ctx][3] t, h, w of the current request's prompt cells (vision)
    const int32_t* mtab_now = nullptr;  // set for a forward over prompt cells that need it, else null
    int32_t* d_delta = nullptr;         // device scalar: the current session's rotary offset (decode graphs read it)
    float* d_img = nullptr;             // the request's picture rows [n][N]
    size_t img_cap = 0;                 // floats
    int32_t* d_map = nullptr;           // [tmax] a chunk's row -> picture row (or -1)
    int32_t* h_map = nullptr;           // pinned
    bool map_on = false;                // this forward's embeddings take picture rows
    std::vector<int32_t> img_map;       // the request's cell -> picture row (-1: text)
    int64_t img_prompt_cells = 0;       // the request's prompt length: cells below it use the table

    // ---- the hybrid mode (Model::hybrid): per state index (the layers, then the MTP block) the slot tables; a
    // batch of up to hybrid_t rows runs its VRAM-resident experts on the GPU and the others on the CPU pool, side by
    // side; a bigger prompt chunk streams each layer's experts into a staging buffer for MMQ (two, double-buffered)
    struct Hy {
        std::vector<int> slot_of, exp_in_slot;   // exp_in_slot: by slot id (-1: empty); ids past the main slots are extra
        std::vector<int> slots;    // the slot ids in use: the main ones, then the extra ones while they are active
        std::vector<int> xslots;   // the extra slot ids: VRAM in the prompt's staging buffers (see place_xslots)
        int n_main = 0;
        std::vector<double> score;   // routing counts, halved at each rebalance
        int32_t* slot_of_d = nullptr;
#if defined(STRATA_Q36_CPU_POOL)
        strata::kernels::cpu::NativeFmt fmt;
#endif
    };
    std::vector<Hy> hy;
    bool xs_active = false;   // the extra slots (in the staging buffers) hold experts: no streamed forward since
    int hybrid_t = 32;
    bool main_hybrid = false;   // some main layer is hybrid (else only the MTP block is: the decode graph stays)
    bool mtp_graph = true;      // the MTP draft step may be a graph (not when its layer is hybrid)
    /// state index li's layer runs hybrid (its experts not all in VRAM)
    bool lhy(size_t li) const {
        if (!m->hybrid) return false;
        const Layer& L = li < m->layers.size() ? m->layers[li] : m->mtp;
        return L.n_slots < (int) m->g.n_expert;
    }
#if defined(STRATA_Q36_CPU_POOL)
    std::unique_ptr<strata::kernels::cpu::ExpertPool> pool;
    std::vector<strata::kernels::cpu::ExpertJobMulti> jobs;
#endif
    float *h_h = nullptr, *cpu_out = nullptr, *ecpu = nullptr;
    int32_t *h_ids = nullptr, *h_miss = nullptr, *d_miss = nullptr, *h_cnt = nullptr;
    uint8_t* acts = nullptr;   // per token: its quantized activation (kNativeActBytes)
    cudaEvent_t ev_route = nullptr;
    // STRATA_Q36_HANDOFF=1 (opt-in): a small forward (decode, verify, MTP; <= kHoRows rows) is enqueued whole before
    // the CPU pool runs. Each hybrid layer's GPU work waits in a kernel for that layer's flag in host-mapped memory,
    // then merges the CPU rows straight from it; the host computes the layers' CPU experts in order afterwards and
    // sets each flag. The next layer's launches and the CPU rows' upload leave the critical path. The arithmetic is
    // the same in both modes (the same rows, the same merge), so the output is bitwise the plain mode's.
    static constexpr int kHoRows = 16;
    bool ho = false;
    std::vector<int32_t*> ho_ids;    // per state index: the routing's ids, mapped (written by k_ho_publish)
    std::vector<float*> ho_h;        // per state index: the rows' hidden, mapped
    std::vector<int32_t*> ho_ids_d;  // their device pointers
    std::vector<float*> ho_h_d;
    unsigned *ho_rflag_h = nullptr, *ho_rflag_d = nullptr;   // [state] the routing published (k_ho_publish)
    float *ho_out_h = nullptr, *ho_out_d = nullptr;       // [state][kHoRows * K][N] CPU rows, mapped
    int32_t *ho_miss_h = nullptr, *ho_miss_d = nullptr;   // [state][kHoRows * K] miss flags, mapped
    unsigned *ho_flag_h = nullptr, *ho_flag_d = nullptr;  // [state] the sequence number a layer's GPU part waits for
    struct HoJob { int li, T; unsigned seq; };
    std::vector<HoJob> ho_pending;
    unsigned ho_seq = 0;
    bool cur_graph_mode = false;     // layer()'s graph_mode, for the hybrid halves
    bool ho_active(size_t li, int T, bool graph_mode) const { return ho && !graph_mode && !rows_mode && T <= kHoRows && li < ho_ids.size(); }
    uint8_t* stage[2] = {};
    cudaStream_t cst = nullptr;
    cudaEvent_t ev_copy[2] = {}, ev_used[2] = {};
    std::vector<int> stream_seq;   // the state indices a streamed forward visits, in order
    int stream_pos = 0;
    bool host_pinned = true;                    // the experts' host arena is pinned (else: the streamer's ring)
    std::unique_ptr<ExpertStreamer> streamer;   // pageable arena: the copier thread and its pinned ring
    uint64_t kick_job[2] = {};                  // the streamer job that fills staging buffer b
    long long hy_cpu = 0, hy_gpu = 0, hy_swaps = 0;   // entries computed on the CPU / GPU, slot replacements
    double hy_cpu_ms = 0;
    double hy_wait_ms = 0, fwd_small_ms = 0;   // the hybrid layers' wait for the routing; small forwards' host time
    long long fwd_small_n = 0;
    // decode / verify rows: their routing is counted once the rows are kept (commit), so a verify window's rejected
    // drafts never move the slots; the slots follow the counts at every rebal_every()-th position exactly (a
    // window never steps over one), so plain and speculative decoding see the same slot tables at every position
    // STRATA_Q36_REBAL_EVERY (32) and STRATA_Q36_SWAPS (2): how often the slots follow the counts, and how many
    // evictions per layer one rebalance may make (filling an empty slot evicts nothing and is not limited)
    static int rebal_every() {
        static const int v = [] { const char* e = std::getenv("STRATA_Q36_REBAL_EVERY"); return e ? std::max(1, std::atoi(e)) : 32; }();
        return v;
    }
    static int rebal_swaps() {
        static const int v = [] { const char* e = std::getenv("STRATA_Q36_SWAPS"); return e ? std::max(1, std::atoi(e)) : 2; }();
        return v;
    }
    bool rows_deferred = false;          // set while a decode step or a verify window runs
    std::vector<std::vector<int32_t>> row_ids;   // [main layer][row * top_k + k] of the last such call
    void count_rows(int n) {
        const int Kp = (int) m->top_k;
        for (size_t li = 0; li < row_ids.size() && li < hy.size(); ++li)
            if (!row_ids[li].empty())
                for (int e = 0; e < n * Kp; ++e) hy[li].score[(size_t) row_ids[li][(size_t) e]] += 1.0;
    }
    void after_rows(int n) {   // a decode step / verify window kept n rows: count them, rebalance at the boundary
        if (!m->hybrid) return;
        count_rows(n);
        if (pos % rebal_every() == 0) {
            CK(cudaStreamSynchronize(st));
            rebalance(std::max(rebal_swaps(), decode_swaps_once));
            decode_swaps_once = 0;
        }
    }

    // ---- STRATA_Q36_KV_RAM (see kv_ram): the cache in segments; the window's in VRAM (the session's kc / vc, one
    // block each), the rest pinned host memory mapped for the GPU, allocated as the context reaches it
    bool kvr_on = false;
    int kvr_nseg = 0, kvr_nwin = 0, kvr_upto = 0;   // segments in all / in VRAM / with memory
    size_t kvr_segbytes = 0;
    std::vector<int8_t*> kvr_tab_h;                 // [state][K, V][segment] device pointers
    std::vector<int8_t*> kvr_hseg;                  // [state][K, V][segment] host pointers (host segments only)
    int8_t** d_kvtab = nullptr;
    std::vector<void*> kvr_host;
    // the cells past the window are read in rounds of kvr_nwin segments: DMA copies of the host segments into this VRAM
    // staging block (K segments, then V), and the kernels read them there through d_stab (a GPU reading pinned memory
    // directly ran at ~0.5 GB/s; the copy engine near the link's rate)
    int8_t* kvr_stage = nullptr;
    int8_t** d_stab = nullptr;
    /// the table that maps segments [s0, s0 + kvr_nwin) to the staging block
    KvTab stage_tab(int s0) const {
        KvTab t;
        t.k = d_stab - s0;
        t.v = d_stab + kvr_nwin - s0;
        t.shift = KvRam::kShift;
        return t;
    }
    /// state li's host segments [s0, s1) (s1 - s0 <= kvr_nwin) into the staging block, on the engine's stream
    void stage_copy(size_t li, int s0, int s1) {
        // STRATA_Q36_KVR_GATHER (default on): a GPU kernel reads the host segments through their mapped pointers with
        // 16-byte loads (6.8 GB/s on this PCIe 3.0 x8 link, tools/hip/pcie_gather_bench; the DMA copies ran at ~3-4)
        static const bool gk = [] { const char* v = std::getenv("STRATA_Q36_KVR_GATHER"); return !(v && v[0] == '0'); }();
        if (gk && s1 > s0) {
            k_seg_gather<<<dim3(64, 2 * (s1 - s0)), 256, 0, st>>>(d_kvtab + (li * 2) * (size_t) kvr_nseg,
                d_kvtab + (li * 2 + 1) * (size_t) kvr_nseg, s0, s1 - s0, kvr_nwin, (int) (kvr_segbytes / 16), kvr_stage);
            return;
        }
        for (int s = s0; s < s1; ++s) {
            CK(cudaMemcpyAsync(kvr_stage + (size_t) (s - s0) * kvr_segbytes, kvr_hseg[(li * 2) * kvr_nseg + s], kvr_segbytes,
                               cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(kvr_stage + (size_t) (kvr_nwin + s - s0) * kvr_segbytes,
                               kvr_hseg[(li * 2 + 1) * kvr_nseg + s], kvr_segbytes, cudaMemcpyHostToDevice, st));
        }
    }
    // ---- STRATA_Q36_SPARSE=1 (opt-in, with KV_RAM): sparse decode attention past the window. Per decode step and
    // attention layer: the window's cells and the most recent STRATA_Q36_SP_RECENT (4096) cells exactly, plus the
    // fraction STRATA_Q36_SP_FRAC (0.5) of the pages in between with the highest estimated attention mass (k_sp_score
    // from per-page K statistics), gathered from pinned memory into a compact cache. Prompts stay dense.
    bool sp_on = false, sp_pf = false;
    float sp_frac = 0.5f, sp_pf_frac = 0.5f;
    int sp_recent = 4096, sp_maxsel = 0, sp_cap = 0;
    std::vector<int> sp_built;   // per attention state: the pages below it have statistics
    std::vector<__half*> sp_stats_h;   // [state][segment] page statistics (device), host segments only
    __half** d_sp_stab = nullptr;
    float* sp_score = nullptr;
    float* sp_lm = nullptr;              // per page and row-head log-mass (the tail estimate)
    unsigned char* sp_mark = nullptr;    // the selected pages (the tail estimate)
    int* sp_sel = nullptr;
    int8_t *sp_ck = nullptr, *sp_cv = nullptr;
    struct SpPlan { int pb = 0, n_cand = 0, S = 0, rp = 0, nr = 0, bias = 0; };
    /// the sparse layout for rows that see the cells [0, p_end): false = dense
    bool sp_plan(int64_t p_end, SpPlan& o) const {
        if (!sp_on) return false;
        const int64_t win_cells = (int64_t) kvr_nwin * KvRam::kSeg;
        const int64_t rs_ = std::max<int64_t>(win_cells, (p_end - sp_recent) / kSpPage * kSpPage);
        o.pb = (int) (win_cells / kSpPage);
        o.rp = (int) (rs_ / kSpPage);
        o.n_cand = o.rp - o.pb;
        if (o.n_cand * kSpPage < 8192 || o.rp > sp_built_min) return false;   // too few far cells, or stats missing
        o.S = std::max(1, std::min(sp_maxsel, (int) std::ceil(sp_frac * o.n_cand)));
        o.nr = (int) ((p_end - rs_ + kSpPage - 1) / kSpPage);
        if ((int64_t) (o.S + o.nr) * kSpPage > sp_cap) return false;
        o.bias = (int) ((int64_t) o.S * kSpPage - rs_);
        return true;
    }
    int sp_built_min = 0;   // the least of sp_built (the decode plan needs every state's)
    /// the page statistics of state li's complete pages below the cell `upto` (layer-major prompts write one state's
    /// cells ahead of the others', so each state keeps its own progress)
    void sp_build_state(size_t li, int64_t upto) {
        const int up = (int) (upto / kSpPage), wp = kvr_nwin * kSpPagesSeg;
        int& b = sp_built[li];
        if (up < b) b = std::max(wp, up);   // the position went back (a reused prompt): rebuild from there
        if (up <= b) return;
        k_sp_stats<<<dim3(up - b, (unsigned) m->g.n_head_kv), 256, 0, st>>>(kvtab(li), d_sp_stab + li * (size_t) kvr_nseg, b,
                                                                          (int) m->g.n_head_kv, kv_qb());
        b = up;
    }
    /// every state's statistics below `upto` (a decode step: all states hold the cells below its first position)
    void sp_build(int64_t upto) {
        if (!sp_on) return;
        int mn = INT32_MAX;
        for (size_t li = 0; li < sess[0].kc.size(); ++li)
            if (sess[0].kc[li]) { sp_build_state(li, upto); mn = std::min(mn, sp_built[li]); }
        sp_built_min = mn;
    }
    /// the cells a session's window holds (per K or V of an attention state)
    size_t kv_cells() const { return kvr_on ? (size_t) std::min(ctx, kv_ram().win) : (size_t) ctx; }
    KvTab kvtab(size_t li) const {
        if (!kvr_on) return KvTab();
        KvTab t;
        t.k = d_kvtab + (li * 2) * (size_t) kvr_nseg;
        t.v = d_kvtab + (li * 2 + 1) * (size_t) kvr_nseg;
        t.shift = KvRam::kShift;
        return t;
    }
    void kvr_init() {
        const auto& g = m->g;
        const size_t ns = sess[0].kc.size();
        kvr_nseg = ctx / KvRam::kSeg;
        kvr_nwin = (int) (kv_cells() / KvRam::kSeg);
        kvr_segbytes = kv_bytes((size_t) g.n_head_kv * KvRam::kSeg * g.head_dim);
        kvr_tab_h.assign(ns * 2 * (size_t) kvr_nseg, nullptr);
        for (size_t li = 0; li < ns; ++li) {
            if (!sess[0].kc[li]) continue;
            for (int s = 0; s < kvr_nwin; ++s) {
                kvr_tab_h[(li * 2) * kvr_nseg + s] = (int8_t*) sess[0].kc[li] + (size_t) s * kvr_segbytes;
                kvr_tab_h[(li * 2 + 1) * kvr_nseg + s] = (int8_t*) sess[0].vc[li] + (size_t) s * kvr_segbytes;
            }
        }
        kvr_upto = kvr_nwin;
        kvr_hseg.assign(kvr_tab_h.size(), nullptr);
        CK(cudaMalloc(&d_kvtab, kvr_tab_h.size() * sizeof(int8_t*)));
        CK(cudaMemcpy(d_kvtab, kvr_tab_h.data(), kvr_tab_h.size() * sizeof(int8_t*), cudaMemcpyHostToDevice));
        if (kvr_nseg > kvr_nwin) {   // the staging block (2 x kvr_nwin segments) and its table
            CK(cudaMalloc(&kvr_stage, 2 * (size_t) kvr_nwin * kvr_segbytes));
            std::vector<int8_t*> t(2 * (size_t) kvr_nwin);
            for (size_t i = 0; i < t.size(); ++i) t[i] = kvr_stage + i * kvr_segbytes;
            CK(cudaMalloc(&d_stab, t.size() * sizeof(int8_t*)));
            CK(cudaMemcpy(d_stab, t.data(), t.size() * sizeof(int8_t*), cudaMemcpyHostToDevice));
        }
        if (const char* v = std::getenv("STRATA_Q36_SPARSE"); v && v[0] == '1' && kvr_nseg > kvr_nwin) {
            if (const char* f = std::getenv("STRATA_Q36_SP_FRAC")) sp_frac = std::max(0.05f, std::min(1.0f, (float) std::atof(f)));
            if (const char* r = std::getenv("STRATA_Q36_SP_RECENT")) sp_recent = std::max(kSpPage, std::atoi(r) / kSpPage * kSpPage);
            const int maxp = (kvr_nseg - kvr_nwin) * kSpPagesSeg;
            sp_maxsel = (int) std::ceil(sp_frac * maxp) + 1;
            sp_cap = (sp_maxsel + sp_recent / kSpPage + 2) * kSpPage;
            sp_ck = (int8_t*) kv_alloc((size_t) g.n_head_kv * sp_cap * g.head_dim);
            sp_cv = (int8_t*) kv_alloc((size_t) g.n_head_kv * sp_cap * g.head_dim);
            sp_score = dalloc((size_t) g.n_head_kv * maxp);
            if (const char* tv = std::getenv("STRATA_Q36_SP_TAIL"); !(tv && tv[0] == '0')) {   // the tail estimate (default on)
                sp_lm = dalloc((size_t) g.n_head_kv * maxp * kGemmT * (g.n_head / g.n_head_kv));
                sp_mark = dalloc_t<unsigned char>((size_t) g.n_head_kv * maxp);
            }
            sp_sel = dalloc_t<int>((size_t) g.n_head_kv * sp_maxsel);
            sp_stats_h.assign(ns * (size_t) kvr_nseg, nullptr);
            CK(cudaMalloc(&d_sp_stab, sp_stats_h.size() * sizeof(__half*)));
            CK(cudaMemset(d_sp_stab, 0, sp_stats_h.size() * sizeof(__half*)));
            sp_built.assign(ns, kvr_nwin * kSpPagesSeg);
            sp_built_min = kvr_nwin * kSpPagesSeg;
            sp_on = true;
            if (const char* pv = std::getenv("STRATA_Q36_SP_PREFILL"); pv && pv[0] == '1') {
                sp_pf = true;
                if (const char* f = std::getenv("STRATA_Q36_SP_PF_FRAC")) sp_pf_frac = std::max(0.05f, std::min(1.0f, (float) std::atof(f)));
                std::fprintf(stderr, "strata-q36: STRATA_Q36_SP_PREFILL=1: prompt chunks past the window read the 4096 cells "
                             "before them and %.0f%% of the other far pages (up to the FP16 block)\n", sp_pf_frac * 100.0);
            }
            std::fprintf(stderr, "strata-q36: STRATA_Q36_SPARSE=1: decode past the window reads the recent %d cells and %.0f%% of the "
                         "other far pages (by estimated attention mass); compact cache %.0f MiB\n", sp_recent, sp_frac * 100.0,
                         (double) kv_bytes((size_t) g.n_head_kv * sp_cap * g.head_dim) * 2 / (1 << 20));
        }
        size_t n_attn = 0;
        for (size_t li = 0; li < ns; ++li) n_attn += sess[0].kc[li] ? 1 : 0;
        std::fprintf(stderr, "strata-q36: STRATA_Q36_KV_RAM=1: %d of %d cells in VRAM (%.2f GiB), the rest in pinned RAM as "
                     "the context reaches it (%.2f GiB at the full context)\n", kvr_nwin * KvRam::kSeg, ctx,
                     (double) kvr_nwin * kvr_segbytes * 2 * n_attn / (1 << 30),
                     (double) (kvr_nseg - kvr_nwin) * kvr_segbytes * 2 * n_attn / (1 << 30));
    }
    /// cells [0, end) with memory: the host segments up to end allocated (pinned, mapped), the table updated
    void kvr_ensure(int64_t end) {
        if (!kvr_on) return;
        const int need = (int) std::min<int64_t>(kvr_nseg, (end + KvRam::kSeg - 1) / KvRam::kSeg);
        if (need <= kvr_upto) return;
        const size_t ns = sess[0].kc.size();
        CK(cudaStreamSynchronize(st));   // (the table is read by queued kernels)
        for (int s = kvr_upto; s < need; ++s)
            for (size_t li = 0; li < ns; ++li) {
                if (!sess[0].kc[li]) continue;
                for (int kv = 0; kv < 2; ++kv) {
                    void* hp = nullptr;
                    void* dp = nullptr;
                    if (cudaHostAlloc(&hp, kvr_segbytes, cudaHostAllocMapped) != cudaSuccess ||
                        cudaHostGetDevicePointer(&dp, hp, 0) != cudaSuccess) {
                        std::fprintf(stderr, "strata-q36: STRATA_Q36_KV_RAM: no pinned host memory for cells %d+ (%.0f MiB)\n",
                                     s * KvRam::kSeg, (double) kvr_segbytes / (1 << 20));
                        std::exit(1);
                    }
                    std::memset(hp, 0, kvr_segbytes);
                    kvr_host.push_back(hp);
                    kvr_tab_h[(li * 2 + kv) * kvr_nseg + s] = (int8_t*) dp;
                    kvr_hseg[(li * 2 + kv) * kvr_nseg + s] = (int8_t*) hp;
                }
                if (sp_on && !sp_stats_h[li * kvr_nseg + s])   // the segment's page statistics (VRAM)
                    sp_stats_h[li * kvr_nseg + s] =
                        dalloc_t<__half>((size_t) kSpPagesSeg * m->g.n_head_kv * kSpStat);
            }
        kvr_upto = need;
        CK(cudaMemcpy(d_kvtab, kvr_tab_h.data(), kvr_tab_h.size() * sizeof(int8_t*), cudaMemcpyHostToDevice));
        if (sp_on) CK(cudaMemcpy(d_sp_stab, sp_stats_h.data(), sp_stats_h.size() * sizeof(__half*), cudaMemcpyHostToDevice));
    }

    void alloc_session(Sess& ss) const {
        const auto& g = m->g;
        const size_t C = (size_t) g.ssm_conv_channels;
        for (const Layer& L : m->layers) {
            if (L.attn) {
                ss.conv.push_back(nullptr);
                ss.S.push_back(nullptr);
                ss.kc.push_back(kv_alloc((size_t) g.n_head_kv * kv_cells() * g.head_dim));
                ss.vc.push_back(kv_alloc((size_t) g.n_head_kv * kv_cells() * g.head_dim));
            } else {
                ss.conv.push_back(dalloc(C * (g.ssm_d_conv - 1)));
                ss.S.push_back(dalloc((size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size));
                ss.kc.push_back(nullptr);
                ss.vc.push_back(nullptr);
            }
        }
        if (m->has_mtp) {   // the MTP block's own K / V (an attention layer at index n_layers)
            ss.conv.push_back(nullptr);
            ss.S.push_back(nullptr);
            ss.kc.push_back(kv_alloc((size_t) g.n_head_kv * kv_cells() * g.head_dim));
            ss.vc.push_back(kv_alloc((size_t) g.n_head_kv * kv_cells() * g.head_dim));
        }
        ss.hlast = dalloc((size_t) g.n_embd);
    }

    /// The current session: `i`'s state, cache and position become the engine's (conv / S / kc / vc / pos / hlast).
    void select(int i) {
        if (i == cur) return;
        sess[(size_t) cur].pos = pos;
        cur = i;
        load_cur();
    }
    void load_cur() {
        Sess& ss = sess[(size_t) cur];
        conv = ss.conv; S = ss.S; kc = ss.kc; vc = ss.vc;
        hlast = ss.hlast;
        pos = ss.pos;
        ss.used = ++use_clock;
        set_delta(ss.rdelta);
    }
    /// The rotary offset the decode graphs read (a device scalar), for the current session.
    void set_delta(int32_t d) {
        sess[(size_t) cur].rdelta = d;
        if (!d_delta) return;
        hpin[60] = d;
        CK(cudaMemcpyAsync(d_delta, hpin + 60, 4, cudaMemcpyHostToDevice, st));
        CK(cudaStreamSynchronize(st));   // (the pinned word is reused)
    }

    // ---- the chunk-sized buffers (see init): two sets when the prompt chunk is above kSmallT rows
    static constexpr int kSmallT = 128;
    struct BufSet {
        float *qkv = nullptr, *hc = nullptr, *z = nullptr, *gdn_y = nullptr, *qg = nullptr, *q = nullptr, *gate = nullptr,
              *att = nullptr, *eparts = nullptr, *mq_gu = nullptr, *mq_h = nullptr;
        float* base = nullptr;
        size_t floats = 0;
    };
    BufSet small_set, big_set;
    bool split_bufs = false, big_on = false;
    size_t big_T = 0;
    static size_t al64(size_t n) { return (n + 63) & ~(size_t) 63; }   // 256-byte steps
    size_t mix_floats(size_t T) const {
        const auto& g = m->g;
        const size_t C = (size_t) g.ssm_conv_channels, Kp = (size_t) m->top_k, FF = (size_t) g.n_ff;
        return std::max({al64(T * C) * 2 + al64(T * g.ssm_value_dim) * 2,
                         al64(T * 2 * g.n_head * g.head_dim) + 3 * al64(T * g.n_head * g.head_dim),
                         al64(T * Kp * 2 * FF) + al64(T * Kp * FF)});
    }
    size_t set_floats(size_t T) const { return mix_floats(T) + al64(T * (size_t) m->top_k * m->g.n_embd); }
    void alloc_set(BufSet& b, size_t T, float* base) {
        const auto& g = m->g;
        const size_t C = (size_t) g.ssm_conv_channels, Kp = (size_t) m->top_k, FF = (size_t) g.n_ff;
        b.base = base;
        b.floats = set_floats(T);
        float* p = base;
        b.qkv = p; p += al64(T * C); b.hc = p; p += al64(T * C); b.z = p; p += al64(T * g.ssm_value_dim); b.gdn_y = p;
        p = base;
        b.qg = p; p += al64(T * 2 * g.n_head * g.head_dim); b.q = p; p += al64(T * g.n_head * g.head_dim);
        b.gate = p; p += al64(T * g.n_head * g.head_dim); b.att = p;
        b.mq_gu = base; b.mq_h = base + al64(T * Kp * 2 * FF);
        b.eparts = base + mix_floats(T);
    }
    void use_set(const BufSet& b) {
        qkv = b.qkv; hc = b.hc; z = b.z; gdn_y = b.gdn_y; qg = b.qg; q = b.q; gate = b.gate; att = b.att;
        eparts = b.eparts; mq_gu = b.mq_gu; mq_h = b.mq_h;
    }
    /// the chunk-sized set (allocated by place_experts after the slots, so that it can hold extra slots)
    void alloc_big() {
        if (!split_bufs || big_set.base) return;
        size_t f0 = 0, f1 = 0, tot = 0;
        CK(cudaMemGetInfo(&f0, &tot));
        alloc_set(big_set, big_T, dalloc(set_floats(big_T)));
        CK(cudaMemGetInfo(&f1, &tot));
        std::fprintf(stderr, "strata-q36: prompt-chunk buffers %.2f GiB (VRAM free %.2f -> %.2f GiB)\n",
                     (double) set_floats(big_T) * 4 / (1 << 30), (double) f0 / (1 << 30), (double) f1 / (1 << 30));
    }
    /// before a batch of T rows: the chunk-sized set above kSmallT rows (its extra slots given up first), else the small one
    void ensure_set(int T) {
        if (!split_bufs) return;
        const bool want_big = T > kSmallT;
        if (want_big) {
            drop_xslots();   // every time: a rebalance since the last big batch may have refilled them
            if (!big_on) use_set(big_set);
        } else if (big_on) {
            use_set(small_set);   // no sync: the sets do not overlap (and this can run inside a graph capture)
        }
        big_on = want_big;
    }
    void init(const Model& model, int ctx_, int tmax_, int n_sessions = 1, bool vision_ = false) {
        m = &model;
        ctx = ctx_;
        n_sess = std::max(1, std::min(n_sessions, kVerify));
        vision = vision_;
        tmax = std::max(tmax_, kVerify);
        kvr_on = kv_ram().on;
        if (kvr_on) {
            if (!kvq8()) {
                std::fprintf(stderr, "strata-q36: STRATA_Q36_KV_RAM=1 needs an 8-bit cache: STRATA_Q36_KV=q8_0 or int8\n");
                std::exit(1);
            }
            if (n_sess > 1) {
                std::fprintf(stderr, "strata-q36: STRATA_Q36_KV_RAM=1 runs one session (no --batch / \"parallel\")\n");
                std::exit(1);
            }
            ctx = (ctx + KvRam::kSeg - 1) / KvRam::kSeg * KvRam::kSeg;   // whole segments
        }
        const auto& g = m->g;
        const size_t T = (size_t) tmax, N = (size_t) g.n_embd, Kp = (size_t) m->top_k, FF = (size_t) g.n_ff;
        const size_t C = (size_t) g.ssm_conv_channels;
        CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        x = dalloc(T * N); h = dalloc(T * N); y = dalloc(T * N); hmain = dalloc(T * N);
        // the mixer's buffers share one region: a layer is a delta-net layer (qkv, hc, z, gdn_y) or an attention layer
        // (qg, q, gate, att), each set dead after its output projection into y, and the prompt's MMQ expert buffers
        // (mq_gu, mq_h) live only while the routed experts run, after both (bit-identical outputs; ~0.5 GiB less at a
        // 4096-row chunk). With a prompt chunk above kSmallT rows (STRATA_Q36_SPLIT_BUFS, default 1) there are two such
        // sets: a small one (kSmallT rows) for decode, verify windows and short prompts, and the chunk-sized one, only
        // for bigger batches; place_experts allocates the latter after the slots, so between prompts it holds extra
        // decode slots like the staging buffers (use_set / ensure_set switch, drop_xslots first).
        {
            const char* sv = std::getenv("STRATA_Q36_SPLIT_BUFS");
            split_bufs = !(sv && sv[0] == '0') && T > (size_t) kSmallT;
            big_T = T;
            const size_t sT = split_bufs ? (size_t) kSmallT : T;
            alloc_set(small_set, sT, dalloc(set_floats(sT)));
            use_set(small_set);
            if (!split_bufs) big_set = small_set;
            // the chunk-sized set right away, before the slots: allocated after them (to hold extra slots between prompts)
            // it ran every prompt kernel ~2.4x slower on Windows (pp4096 298 vs 972 tok/s, RX 6600): the late
            // allocation near the budget is likely placed in system memory. STRATA_Q36_BIG_LATE=1 keeps the late one.
            else if (!std::getenv("STRATA_Q36_BIG_LATE")) alloc_big();
        }
        alpha = dalloc(T * g.ssm_v_heads); beta = dalloc(T * g.ssm_v_heads);
        kx = dalloc(T * g.n_head_kv * g.head_dim); vx = dalloc(T * g.n_head_kv * g.head_dim);
        logits_r = dalloc(T * (g.n_expert + 1));
        moe_out = dalloc(T * N);
        sg = dalloc(T * g.n_ff_shexp); su = dalloc(T * g.n_ff_shexp); shh = dalloc(T * g.n_ff_shexp); sy = dalloc(T * N);
        sgate = dalloc(T); ewts = dalloc(T * Kp); logits = dalloc((size_t) kVerify * g.n_vocab);
        const int max_in = (int) std::max<int64_t>({2 * g.n_embd, g.ssm_value_dim, g.n_head * g.head_dim, g.n_ff_shexp});
        CK(cudaMalloc(&q8, K::native_q8_1_bytes(max_in, 1) * T));
        CK(cudaMalloc(&xq, K::native_q8_1_bytes((int) N, 1) * T));
        x16 = dalloc_t<__half>(T * (size_t) max_in);
        if (cublasCreate(&blas) != CUBLAS_STATUS_SUCCESS || cublasSetStream(blas, st) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: cuBLAS handle failed\n");
            std::exit(1);
        }
        att_s = dalloc(kAttScoreFloats);
        sk_part = dalloc((size_t) kSplitK * T * (g.n_expert + 1));
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        if (!kv16()) {   // the f32 / 8-bit cache's FP16 copy for the prompt GEMMs (an f16 cache is read in place)
            const size_t cells = kvr_on ? (size_t) std::min(ctx, kv_ram().win) : (size_t) ctx;   // KV_RAM: one block
            k16 = dalloc_t<__half>((size_t) g.n_head_kv * cells * g.head_dim);
            v16 = dalloc_t<__half>((size_t) g.n_head_kv * cells * g.head_dim);
        }
        q16 = dalloc_t<__half>(T * g.n_head * g.head_dim);
        if (att16_on()) ensure_o16(T * g.n_head * g.head_dim);   // (allocated before the slots)
        if (const char* lv = std::getenv("STRATA_Q36_LM"); lv && lv[0] == '1') {   // layer-major prompts (forward_lm)
            if (const char* c = std::getenv("STRATA_Q36_LM_CHUNKS")) lm_chunks = std::max(2, std::min(32, std::atoi(c)));
            lm_on = true;
            lm_X = dalloc((size_t) lm_chunks * T * N);
            lm_H = dalloc((size_t) lm_chunks * T * N);
            lm_tok = dalloc_t<int32_t>((size_t) lm_chunks * T);
            lm_pos = dalloc_t<int32_t>((size_t) lm_chunks);
            std::fprintf(stderr, "strata-q36: STRATA_Q36_LM=1: prompts layer-major in super-chunks of %d x %zu tokens "
                         "(%.2f GiB of VRAM)\n", lm_chunks, T, (double) lm_chunks * T * N * 8 / (1 << 30));
        }
        use_mmq = MMQ::built() && std::getenv("STRATA_Q36_NO_MMQ") == nullptr;
        auto mmq_ok = [&](const Layer& L) {
            return MMQ::supported(L.L.gu_type) && MMQ::supported(L.L.d_type) && MMQ::fits(L.L.gu_type, 2 * g.n_ff) &&
                   MMQ::fits(L.L.d_type, g.n_embd);
        };
        for (const Layer& L : m->layers) use_mmq = use_mmq && mmq_ok(L);
        if (m->has_mtp) use_mmq = use_mmq && mmq_ok(m->mtp);
        if (use_mmq) {
            const int64_t ne = (int64_t) T * Kp;
            mctx = std::make_unique<MMQ::Context>();
            rcnt = dalloc_t<int32_t>((size_t) g.n_expert);
            rcur = dalloc_t<int32_t>((size_t) g.n_expert);
            rbounds = dalloc_t<int32_t>((size_t) g.n_expert + 1);
            rmax = dalloc_t<int32_t>(1);
            ident = dalloc_t<int32_t>((size_t) ne);
            rinv = dalloc_t<int32_t>((size_t) ne);
            CK(cudaDeviceSynchronize());   // ident's cudaMemset (legacy stream) before the iota on the non-blocking stream
            MMQ::iota(ident, ne, st);
            CK(cudaMalloc(&mq_x, MMQ::q8_bytes(ne, g.n_embd)));
            CK(cudaMalloc(&mq_hq, MMQ::q8_bytes(ne, g.n_ff)));
            // mq_gu / mq_h: in the buffer sets (alloc_set)
        }
        std::fprintf(stderr, "strata-q36: prompt experts on %s\n", use_mmq ? "MMQ (int8)" : "the native grouped kernels");
        {
            std::string gerr;
            int64_t big = 0;
            auto take = [&](const Layer& L) {
                for (const Mat* w : {&L.qkv, &L.zg, &L.ssm_out, &L.wq, &L.wk, &L.wv, &L.wo, &L.sh_gate, &L.sh_up, &L.sh_down})
                    big = std::max<int64_t>(big, (int64_t) w->n_in * w->n_out);
            };
            for (const Layer& L : m->layers) take(L);
            if (m->has_mtp) {
                take(m->mtp);
                big = std::max<int64_t>(big, (int64_t) m->eh_proj.n_in * m->eh_proj.n_out);
            }
            if (!gemm.init(st, big, gerr)) {
                std::fprintf(stderr, "strata-q36: prompt GEMM: %s\n", gerr.c_str());
                std::exit(1);
            }
        }
        CK(cudaMalloc(&escr, K::native_expert_scratch_bytes((int64_t) (T * Kp), (int64_t) FF)));
        ids = dalloc_t<int32_t>(T * Kp);
        grp_ptr = dalloc_t<unsigned long long>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        grp_start = dalloc_t<int32_t>(std::min<size_t>(T * Kp, (size_t) g.n_expert) + 1);
        n_groups = dalloc_t<int32_t>(1);
        ent_dst = dalloc_t<int32_t>(T * Kp);
        ent_tok = dalloc_t<int32_t>(T * Kp);
        tok = dalloc_t<int32_t>(T);
        pos0 = dalloc_t<int32_t>(1);
        pick = dalloc_t<int32_t>(kVerify);
        amv = dalloc((size_t) kArgBlocks * kVerify);
        ami = dalloc_t<int32_t>((size_t) kArgBlocks * kVerify);
        CK(cudaHostAlloc((void**) &hpin, 64 * sizeof(int32_t), cudaHostAllocDefault));
        n_chunk = (ctx + kChunkCells - 1) / kChunkCells;
        att_rows = (int) (g.n_head * std::min<int64_t>((int64_t) T, 64));   // 64 query tokens per attention batch
        // KV_RAM: the split-K path runs only batches of <= kGemmT rows (decode, verify) - their partials at a long
        // context would be ~1 GiB with 64 query tokens
        if (kvr_on) att_rows = (int) (g.n_head * std::min<int64_t>((int64_t) T, kGemmT));
        pacc = dalloc((size_t) att_rows * n_chunk * g.head_dim);
        pm = dalloc((size_t) att_rows * n_chunk);
        pl = dalloc((size_t) att_rows * n_chunk);
        for (const Layer& L : m->layers) {   // the verify windows' GDN inputs (one set, whichever session runs)
            if (L.attn) {
                for (auto* v : {&v_qkv, &v_hc, &v_alpha, &v_beta}) v->push_back(nullptr);
            } else {
                v_qkv.push_back(dalloc((size_t) kVerify * C));
                v_hc.push_back(dalloc((size_t) kVerify * C));
                v_alpha.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
                v_beta.push_back(dalloc((size_t) kVerify * g.ssm_v_heads));
            }
        }
        if (m->has_mtp) {
            mtp_cat = dalloc(T * 2 * N);
            hmtp = dalloc(T * N);
            logits_m = dalloc((size_t) g.n_vocab);
            tok_m = dalloc_t<int32_t>(T);
            pos0_m = dalloc_t<int32_t>(1);
            pick_m = dalloc_t<int32_t>(1);
        }
        sess.resize((size_t) n_sess);
        gsets.resize((size_t) n_sess);
        {   // the sessions' caches: said before they fail (a slot's K / V at the context, fp32, for the attention layers and
            // the MTP block (see kv_cfg), and its recurrent state)
            size_t n_attn = m->has_mtp ? 1 : 0, n_gdn = 0;
            for (const Layer& L : m->layers) (L.attn ? n_attn : n_gdn) += 1;
            const size_t per = kv_bytes((size_t) g.n_head_kv * g.head_dim * ctx) * 2 * n_attn +
                               (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4 * n_gdn;
            size_t fr0 = 0, tot0 = 0;
            CK(cudaMemGetInfo(&fr0, &tot0));
            if (n_sess > 1)
                std::fprintf(stderr, "strata-q36: %d sessions (--batch): %.2f GiB of VRAM each at a context of %d (%.2f GiB in "
                             "all), %.2f GiB free before them\n", n_sess, (double) per / (1 << 30), ctx,
                             (double) per * n_sess / (1 << 30), (double) fr0 / (1 << 30));
            if (n_sess > 1 && per * (size_t) n_sess + ((size_t) 1 << 30) > fr0) {
                std::fprintf(stderr, "strata-q36: not enough VRAM for %d slots at a context of %d (%.2f GiB each, %.2f GiB "
                             "free): use fewer slots (\"parallel\" / --batch) or a shorter --max-context\n", n_sess, ctx,
                             (double) per / (1 << 30), (double) fr0 / (1 << 30));
                std::exit(1);
            }
        }
        for (Sess& ss : sess) alloc_session(ss);
        if (kvr_on) kvr_init();
        {
            size_t n_attn = m->has_mtp ? 1 : 0;
            for (const Layer& L : m->layers) n_attn += L.attn ? 1 : 0;
            std::fprintf(stderr, "strata-q36: KV cache %s%s, %.2f GiB per session at a context of %d\n", kv_name(),
                         kvq8() ? (kv_dot8() ? " (int8 dot scores)" : " (float scores)") : "",
                         (double) kv_bytes((size_t) g.n_head_kv * g.head_dim * ctx) * 2 * n_attn / (1 << 30), ctx);
        }
        if (n_sess > 1) {   // the batch rows' tables: per state index the session pointers, on the device
            const size_t ns = sess[0].conv.size();
            std::vector<float*> hc(ns * (size_t) n_sess), hs(hc), hk(hc), hv(hc);
            for (size_t li = 0; li < ns; ++li)
                for (int s_ = 0; s_ < n_sess; ++s_) {
                    hc[li * n_sess + s_] = sess[(size_t) s_].conv[li];
                    hs[li * n_sess + s_] = sess[(size_t) s_].S[li];
                    hk[li * n_sess + s_] = sess[(size_t) s_].kc[li];
                    hv[li * n_sess + s_] = sess[(size_t) s_].vc[li];
                }
            const std::pair<float***, std::vector<float*>*> tabs[4] = {{&d_conv_tab, &hc}, {&d_S_tab, &hs},
                                                                          {&d_kc_tab, &hk}, {&d_vc_tab, &hv}};
            for (const auto& tb : tabs) {
                CK(cudaMalloc((void**) tb.first, tb.second->size() * sizeof(float*)));
                CK(cudaMemcpy((void*) *tb.first, tb.second->data(), tb.second->size() * sizeof(float*),
                              cudaMemcpyHostToDevice));
            }
            d_rcell = dalloc_t<int32_t>(kVerify);
            d_rslot = dalloc_t<int32_t>(kVerify);
            d_rdelta = dalloc_t<int32_t>(kVerify);
            d_gslot = dalloc_t<int32_t>(kVerify);
            d_grow0 = dalloc_t<int32_t>(kVerify);
            d_gcnt = dalloc_t<int32_t>(kVerify);
            CK(cudaHostAlloc((void**) &hrow, 128 * sizeof(int32_t), cudaHostAllocDefault));
        }
        if (vision) {
            d_mtab = dalloc_t<int32_t>((size_t) ctx * 3);
            d_map = dalloc_t<int32_t>(T);
            CK(cudaHostAlloc((void**) &h_map, T * sizeof(int32_t), cudaHostAllocDefault));
        }
        d_delta = dalloc_t<int32_t>(1);
        cur = 0;
        load_cur();
        CK(cudaDeviceSynchronize());   // every legacy-stream cudaMemset of the buffers above is done
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-q36: context %d cells, prompt chunk %d%s, %.2f GiB of VRAM free\n", ctx, tmax,
                     m->has_mtp ? ", MTP block loaded" : "", (double) fr / (1 << 30));
    }

    /// The recurrent state and the cache to a fresh sequence.
    void reset() {
        const auto& g = m->g;
        for (size_t l = 0; l < conv.size(); ++l) {
            if (conv[l]) CK(cudaMemsetAsync(conv[l], 0, (size_t) g.ssm_conv_channels * (g.ssm_d_conv - 1) * 4, st));
            if (S[l]) CK(cudaMemsetAsync(S[l], 0, (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4, st));
        }
        CK(cudaMemsetAsync(hlast, 0, (size_t) g.n_embd * 4, st));
        CK(cudaStreamSynchronize(st));
        pos = 0;
        sess[(size_t) cur].pos = 0;
        set_delta(0);
    }

    /// y [T][n_out] = W x [T][n_in] on the native MMVQ, 8 columns per launch.
    void mm(const Mat& w, const float* xin, float* out, int T) {
        prep(xin, w.n_in, T, q8);
        apply(w, q8, out, T);
    }
    /// y [T][n_out] = W [n_out][n_in] (f32) x [T][n_in]: a decode batch on the warp-per-row kernel, a prompt chunk on
    /// cuBLAS's FP32 GEMM (the router and the GDN alpha / beta stay in FP32, as llama.cpp keeps them).
    void f32mm(const float* W, const float* xin, float* out, int n_in, int n_out, int T) {
        if (T <= kGemmT) {
            k_gemv_f32<<<dim3(rows8(n_out), T), kThreads, 0, st>>>(W, xin, out, n_in, n_out);
            return;
        }
        static const bool blas_f32 = std::getenv("STRATA_Q36_BLAS_F32") != nullptr;   // the hipBLAS / cuBLAS SGEMM (A/B)
        if (!blas_f32 && n_in % 4 == 0) {
            // K split so that about 128 blocks run (the partials summed in a fixed order: deterministic); TM = 4 rows per
            // thread for the 256-wide router, 1 for the narrow ones (alpha / beta 32, the gate 1)
            const int tm = n_out > 64 ? 4 : 1;
            const int gx = (n_out + 63) / 64, gy = (T + 16 * tm - 1) / (16 * tm);
            const int want = std::max(1, std::min(kSplitK, 128 / std::max(1, gx * gy)));
            const int kz = (n_in / want + 15) / 16 * 16;
            const int splits = (n_in + kz - 1) / kz;
            float* dst = splits > 1 ? sk_part : out;
            if (tm == 4) k_sgemm_nt<4><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            else k_sgemm_nt<1><<<dim3(gx, gy, splits), 256, 0, st>>>(W, xin, dst, n_in, n_out, T, n_out, kz);
            if (splits > 1)
                k_sum_splits<<<grid1((int64_t) T * n_out), kThreads, 0, st>>>(sk_part, out, (int64_t) T * n_out, splits);
            return;
        }
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, W, n_in, xin, n_in, &zero, out, n_out) !=
            CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: cublasSgemm failed\n");
            std::exit(1);
        }
    }

    /// A batch above kGemmT tokens (a prompt chunk) runs its projections as GEMMs (prefill::Gemm::native: the GGUF
    /// blocks dequantized to FP16, the activations FP16); a decode batch on the native MMVQ (q8_1 activations).
    static constexpr int kGemmT = 8;
    void prep(const float* xin, int n, int T, void* q) {
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) n * T), kThreads, 0, st>>>(xin, x16, (int64_t) n * T);
        else quant(xin, n, T, q);
    }
    void apply(const Mat& w, const void* q, float* out, int T) {
        if (T > kGemmT) gemm.native((const uint16_t*) x16, w.type, w.d, out, T, w.n_out, w.n_in);
        else mmq(w, q, out, T);
    }
    /// x [T][n] -> its q8_1 image (T contiguous columns: one vector of T * n), for every matrix that reads x.
    void quant(const float* xin, int n, int T, void* q) { K::native_quantize_q8_1(xin, q, n * T, 1, st); }
    // ---- decode kernels of this engine (q36_kernels.cuh), each the old kernel's bits (STRATA_Q36_VERIFY=1 compares)
    bool fast_k = std::getenv("STRATA_Q36_OLD_KERNELS") == nullptr;
    bool verify_k = std::getenv("STRATA_Q36_VERIFY") != nullptr;
    float* vbuf = nullptr;                       // verify scratch
    long long v_calls = 0, v_bad = 0;
    std::map<std::string, long long> v_by;   // mismatches per check
    void verify_cmp(const char* what, const float* a, const float* b, size_t n, const std::string& tag = "") {
        CK(cudaStreamSynchronize(st));
        std::vector<float> ha(n), hb(n);
        CK(cudaMemcpy(ha.data(), a, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hb.data(), b, n * 4, cudaMemcpyDeviceToHost));
        ++v_calls;
        if (std::memcmp(ha.data(), hb.data(), n * 4) != 0) {
            ++v_bad;
            size_t i = 0;
            while (i < n && std::memcmp(&ha[i], &hb[i], 4) == 0) ++i;
            if (++v_by[what] <= 3)
                std::fprintf(stderr, "strata-q36 VERIFY MISMATCH %s%s: first at %zu of %zu: new %g old %g (bits %08x %08x)\n", what,
                             tag.c_str(), i, n, (double) ha[i], (double) hb[i], *(unsigned*) &ha[i], *(unsigned*) &hb[i]);
        }
    }
    /// A Q8_0 matrix that is not on the packed decode kernel (the shared expert, attention k / v: fewer than 2048
    /// rows) for T <= 8 columns: one warp per row (q36k::k_q8_rows) in place of the 4-warp block of
    /// native_small_mmvq_kernel; the same bits.
    bool q8_small(const Mat& w, const void* q, float* out, int T) {
        if (!fast_k || w.type != 8 || T < 1 || T > 8 || w.n_in % 32 != 0) return false;
        if (K::native_q8_0_packed_eligible(w.n_in, w.n_out) && K::native_q8_0_packed_enabled()) return false;
        const int nb = w.n_in / 32;
        if (nb > 64) return false;
        const void* wp = w.d;
        const half* dpl = nullptr;
        const int NJ = nb <= 32 ? 1 : 2;
#define Q36_Q8(NCV)                                                                                          \
    case NCV:                                                                                                \
        if (NJ == 1) q36k::launch_q8_rows<NCV, 1, false>(wp, dpl, q, out, nb, w.n_out, st);                  \
        else q36k::launch_q8_rows<NCV, 2, false>(wp, dpl, q, out, nb, w.n_out, st);                          \
        break;
        switch (T) { Q36_Q8(1) Q36_Q8(2) Q36_Q8(3) Q36_Q8(4) Q36_Q8(5) Q36_Q8(6) Q36_Q8(7) Q36_Q8(8) }
#undef Q36_Q8
        if (verify_k) {
            if (!vbuf) CK(cudaMalloc((void**) &vbuf, (size_t) 8 * 16384 * 4));
            const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
            for (int c0 = 0; c0 < T; c0 += 8) {
                const int nc = std::min(8, T - c0);
                K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, vbuf + (size_t) c0 * w.n_out, w.n_in, w.n_out,
                               nc, st);
            }
            verify_cmp("q8_small", out, vbuf, (size_t) T * w.n_out,
                       " n_in " + std::to_string(w.n_in) + " n_out " + std::to_string(w.n_out) + " T " + std::to_string(T));
        }
        return true;
    }
    // ---- the q8_1 images produced by the norm kernels themselves (fast_k, T <= kGemmT): xq_ready = the layer's input h is
    // already quantized in xq (k_moe_out_small of the layer before), xq_post = the MoE input h is (k_add_rms_q)
    bool xq_ready = false, xq_post = false;
    void* vq = nullptr;
    void verify_q8(const char* what, const float* hsrc, int T, const std::string& tag) {
        const int N = (int) m->g.n_embd;
        const size_t qb = K::native_q8_1_bytes(N, 1);
        if (!vq) CK(cudaMalloc(&vq, qb * 8));
        K::native_quantize_q8_1(hsrc, vq, N * T, 1, st);
        const long long before = v_bad;
        verify_cmp(what, (const float*) xq, (const float*) vq, qb * (size_t) T / 4, tag);
        if (v_bad != before && std::getenv("STRATA_Q36_VDEBUG") && v_by[what] <= 2) {   // a mismatching block, on the host
            std::vector<uint8_t> a(qb * (size_t) T), b(qb * (size_t) T);
            std::vector<float> hv((size_t) N * T);
            CK(cudaMemcpy(a.data(), xq, a.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), vq, b.size(), cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(hv.data(), hsrc, hv.size() * 4, cudaMemcpyDeviceToHost));
            for (size_t o = 0; o < a.size(); o += 36)
                if (std::memcmp(&a[o], &b[o], 36) != 0) {
                    const size_t blk = o / 36;
                    float v[32], t[32];
                    for (int l = 0; l < 32; ++l) v[l] = t[l] = hv[blk * 32 + (size_t) l];
                    for (int off = 16; off > 0; off >>= 1) {   // the xor butterfly, lane by lane
                        float u[32];
                        for (int l = 0; l < 32; ++l) u[l] = t[l] + t[l ^ off];
                        std::memcpy(t, u, sizeof(u));
                    }
                    unsigned ds_a, ds_b;
                    const __half t0h = __float2half(t[0]);   // host side: __half_as_ushort is __device__ only
                    unsigned short t0bits;
                    std::memcpy(&t0bits, &t0h, sizeof(t0bits));
                    std::memcpy(&ds_a, &a[o], 4);
                    std::memcpy(&ds_b, &b[o], 4);
                    std::fprintf(stderr, "VDEBUG block %zu: ds new %08x old %08x, qs bytes %s, cpu butterfly sum %.9g (half %04x)\n",
                                 blk, ds_a, ds_b, std::memcmp(&a[o + 4], &b[o + 4], 32) == 0 ? "equal" : "DIFFER", (double) t[0],
                                 (unsigned) t0bits);
                    break;
                }
        }
    }
    void prep_in(int T) {
        if (xq_ready && T <= kGemmT) {
            xq_ready = false;
            return;
        }
        xq_ready = false;
        prep(h, (int) m->g.n_embd, T, xq);
    }
    void add_rms_post(const Layer& L, int T) {
        const int N = (int) m->g.n_embd;
        xq_post = false;
        if (fast_k && T <= kGemmT) {
            if (verify_k) {
                if (!vx_) {
                    CK(cudaMalloc((void**) &vx_, (size_t) 8 * 2048 * 4));
                    CK(cudaMalloc((void**) &vh_, (size_t) 8 * 2048 * 4));
                }
                CK(cudaMemcpyAsync(vx_, x, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, st));
                k_add_rms<<<T, kThreads, 0, st>>>(vx_, y, L.post_norm, vh_, N, m->eps);
            }
            q36k::k_add_rms_q<<<T, 256, 0, st>>>(x, y, L.post_norm, h, (q36k::QB*) xq, N, m->eps);
            xq_post = true;
            if (verify_k) {
                verify_cmp("add_rms x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                verify_cmp("add_rms h", h, vh_, (size_t) T * N, " T " + std::to_string(T));
                verify_q8("add_rms q8", h, T, " T " + std::to_string(T));
            }
            return;
        }
        k_add_rms<<<T, kThreads, 0, st>>>(x, y, L.post_norm, h, N, m->eps);
    }
    /// The shared expert's gate + up + SwiGLU + q8_1 image (into q8) for T <= 4 rows in one launch (q36k::k_sh_gu), else false.
    bool shared_gu(const Layer& L, int T) {
        const int FS = (int) m->g.n_ff_shexp;
        const Mat &a = L.sh_gate, &b = L.sh_up;
        if (!fast_k || T < 1 || T > 4 || a.type != 8 || b.type != 8 || a.n_in != b.n_in || a.n_out != FS || b.n_out != FS ||
            a.n_in % 32 != 0 || a.n_in / 32 > 64 || a.n_in / 32 <= 32 || FS % 32 != 0 ||
            (K::native_q8_0_packed_eligible(a.n_in, a.n_out) && K::native_q8_0_packed_enabled()))
            return false;
        const int nb = a.n_in / 32;
        const dim3 grid((unsigned) (FS / 32));
        const bool pkd = false;
        const uint8_t* wa = a.d;
        const uint8_t* wb = b.d;
        const half* da = nullptr;
        const half* db = nullptr;
#define Q36_SH(NCV)                                                                                          \
    case NCV:                                                                                                \
        if (pkd)                                                                                             \
            q36k::k_sh_gu<NCV, 2, true><<<grid, 1024, 0, st>>>(wa, da, wb, db, (const q36k::QB*) xq,        \
                                                               (q36k::QB*) q8, nb, FS);                      \
        else                                                                                                 \
            q36k::k_sh_gu<NCV, 2, false><<<grid, 1024, 0, st>>>(wa, nullptr, wb, nullptr, (const q36k::QB*) xq, \
                                                                (q36k::QB*) q8, nb, FS);                     \
        break;
        switch (T) { Q36_SH(1) Q36_SH(2) Q36_SH(3) Q36_SH(4) }
#undef Q36_SH
        if (verify_k) {
            if (!vbuf) CK(cudaMalloc((void**) &vbuf, (size_t) 8 * 16384 * 4));
            if (!vq) CK(cudaMalloc(&vq, K::native_q8_1_bytes(m->g.n_embd, 1) * 8));
            float* vg = vbuf;
            float* vu = vbuf + (size_t) 8 * FS;
            mmq_ref(a, xq, vg, T);
            mmq_ref(b, xq, vu, T);
            K::native_swiglu_quantize_q8_1(vg, vu, vq, FS, T, st);
            verify_cmp("shared_gu q8", (const float*) q8, (const float*) vq, K::native_q8_1_bytes(FS, 1) * (size_t) T / 4,
                       " T " + std::to_string(T));
        }
        return true;
    }
    /// The old products (native MMVQ), for the checks.
    void mmq_ref(const Mat& w, const void* q, float* out, int T) {
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in, w.n_out, nc, st);
        }
    }
    void group_old(const Layer& L, const int32_t* slot_of, int T, unsigned long long* o_ptr, int32_t* o_start, int32_t* o_ng,
                   int32_t* o_dst, int32_t* o_tok) {
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (slot_of)
            k_group_hy<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, slot_of, L.blobs, L.stride, o_ptr, o_start,
                                               o_ng, o_dst, o_tok);
        else
            k_group<<<1, kThreads, 0, st>>>(ids, n_ent, Kp, (int) m->g.n_expert, L.blobs, L.stride, o_ptr, o_start, o_ng, o_dst,
                                            o_tok);
    }
    /// VERIFY: the old grouping of the current `ids` into scratch, compared with grp_ptr / grp_start / n_groups / ent_dst / ent_tok.
    void group_verify(const Layer& L, const int32_t* slot_of, int T) {
        if (!vgrp) CK(cudaMalloc((void**) &vgrp, 2048));
        unsigned long long* o_ptr = (unsigned long long*) vgrp;
        int32_t* o_start = (int32_t*) (o_ptr + 65);
        int32_t* o_ng = o_start + 66;
        int32_t* o_dst = o_ng + 2;
        int32_t* o_tok = o_dst + 64;
        group_old(L, slot_of, T, o_ptr, o_start, o_ng, o_dst, o_tok);
        CK(cudaStreamSynchronize(st));
        int nn = 0, no = 0;
        CK(cudaMemcpy(&nn, n_groups, 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&no, o_ng, 4, cudaMemcpyDeviceToHost));
        ++v_calls;
        bool bad = nn != no;
        if (!bad) {
            std::vector<unsigned long long> a((size_t) nn + 1), b((size_t) nn + 1);
            std::vector<int32_t> sa((size_t) nn + 1), sb((size_t) nn + 1), da(64), db(64), ta(64), tb(64);
            CK(cudaMemcpy(a.data(), grp_ptr, (size_t) nn * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(b.data(), o_ptr, (size_t) nn * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sa.data(), grp_start, ((size_t) nn + 1) * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(sb.data(), o_start, ((size_t) nn + 1) * 4, cudaMemcpyDeviceToHost));
            const int tot = sa[(size_t) nn];
            CK(cudaMemcpy(da.data(), ent_dst, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(db.data(), o_dst, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(ta.data(), ent_tok, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(tb.data(), o_tok, (size_t) tot * 4, cudaMemcpyDeviceToHost));
            bad = std::memcmp(a.data(), b.data(), (size_t) nn * 8) != 0 ||
                  std::memcmp(sa.data(), sb.data(), ((size_t) nn + 1) * 4) != 0 ||
                  std::memcmp(da.data(), db.data(), (size_t) tot * 4) != 0 ||
                  std::memcmp(ta.data(), tb.data(), (size_t) tot * 4) != 0;
        }
        if (bad) {
            ++v_bad;
            if (v_bad <= 20) std::fprintf(stderr, "strata-q36 VERIFY MISMATCH group (T %d, groups %d vs %d)\n", T, nn, no);
        }
    }
    bool grouped_by_route = false;   // route_group already filled the group arrays of this layer
    /// The router's top-k and the grouping of the picks in one launch (T <= 8, 256 experts, 8 picks); the logits are in logits_r.
    /// `slot_of`: the hybrid mode's slot table (null: every expert resident).  Returns false when the old kernels must run.
    bool route_group(const Layer& L, const int32_t* slot_of, int T, int ld) {
        const int Kp = (int) m->top_k;
        grouped_by_route = false;
        if (!fast_k || T > 8 || m->g.n_expert != 256 || Kp != 8) return false;
        if (slot_of)
            q36k::k_route_group<true><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, slot_of, L.blobs, L.stride, grp_ptr,
                                                                 grp_start, n_groups, ent_dst, ent_tok);
        else
            q36k::k_route_group<false><<<1, dim3(32, 8), 0, st>>>(logits_r, ids, ewts, T, ld, nullptr, L.blobs, L.stride, grp_ptr,
                                                                  grp_start, n_groups, ent_dst, ent_tok);
        grouped_by_route = true;
        if (verify_k) {
            // the old top-k into scratch, the ids / weights compared, then the old grouping of the new ids
            if (!vbuf) CK(cudaMalloc((void**) &vbuf, (size_t) 8 * 16384 * 4));
            int32_t* vid = (int32_t*) (vbuf + 8 * 8192);
            float* vw = vbuf + 8 * 8192 + 64;
            K::native_router_topk(logits_r, vid, vw, 256, Kp, T, st);
            verify_cmp("route ids", (const float*) ids, (const float*) vid, (size_t) T * Kp, " T " + std::to_string(T));
            verify_cmp("route weights", ewts, vw, (size_t) T * Kp, " T " + std::to_string(T));
            group_verify(L, slot_of, T);
        }
        return true;
    }
    /// The routed entries of T rows grouped per expert (k_group; with slot_of the hybrid mode's k_group_hy).
    void group_entries(const Layer& L, const int32_t* slot_of, int T) {
        if (grouped_by_route) {   // route_group did it
            grouped_by_route = false;
            return;
        }
        const int Kp = (int) m->top_k, n_ent = T * Kp;
        if (fast_k && n_ent <= 64) {
            if (slot_of)
                q36k::k_group_small<true><<<1, 64, 0, st>>>(ids, n_ent, Kp, slot_of, L.blobs, L.stride, grp_ptr, grp_start,
                                                            n_groups, ent_dst, ent_tok);
            else
                q36k::k_group_small<false><<<1, 64, 0, st>>>(ids, n_ent, Kp, nullptr, L.blobs, L.stride, grp_ptr, grp_start,
                                                             n_groups, ent_dst, ent_tok);
            if (verify_k) group_verify(L, slot_of, T);
            return;
        }
        group_old(L, slot_of, T, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);
    }
    void* vgrp = nullptr;
    float *vx_ = nullptr, *vh_ = nullptr;
    /// The MoE block's end (k_moe_out) for T rows.
    void moe_end(int T, const int32_t* prow, const float* shared, const float* sgate, const float* wn, float* hout, int sgs = 1) {
        const int N = (int) m->g.n_embd, Kp = (int) m->top_k;
        const bool small = fast_k && T <= 8 && N <= 2048 && N % 256 == 0 && Kp <= 15;
        if (small && verify_k) {
            if (!vx_) {
                CK(cudaMalloc((void**) &vx_, (size_t) 8 * 2048 * 4));
                CK(cudaMalloc((void**) &vh_, (size_t) 8 * 2048 * 4));
            }
            CK(cudaMemcpyAsync(vx_, x, (size_t) T * N * 4, cudaMemcpyDeviceToDevice, st));
            k_moe_out<<<T, kThreads, 0, st>>>(vx_, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, vh_, N, m->eps);
        }
        if (small) {
            const bool mkq = wn != nullptr && hout == h;   // the next layer's input, quantized here
            q36k::k_moe_out_small<<<T, 1024, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, hout, N, m->eps,
                                                      mkq ? (q36k::QB*) xq : nullptr);
            xq_ready = mkq;
            if (verify_k) {
                if (mkq) verify_q8("moe_out q8", hout, T, " T " + std::to_string(T));
                verify_cmp("moe_out x", x, vx_, (size_t) T * N, " T " + std::to_string(T));
                if (wn) verify_cmp("moe_out h", hout, vh_, (size_t) T * N, " T " + std::to_string(T));
            }
            return;
        }
        xq_ready = false;
        k_moe_out<<<T, kThreads, 0, st>>>(x, eparts, prow, ewts, Kp, shared, sgate, sgs, wn, hout, N, m->eps);
    }
    /// y [T][n_out] = W x from x's q8_1 image, 8 columns per launch.
    void mmq(const Mat& w, const void* q, float* out, int T) {
        if (q8_small(w, q, out, T)) return;
        const size_t qb = K::native_q8_1_bytes(w.n_in, 1);
        for (int c0 = 0; c0 < T; c0 += 8) {
            const int nc = std::min(8, T - c0);
            K::native_mmvq(w.type, w.d, (const char*) q + (size_t) c0 * qb, out + (size_t) c0 * w.n_out, w.n_in,
                           w.n_out, nc, st);
        }
    }

    /// STRATA_Q36_KV_RAM: a prompt chunk's attention over a context longer than the FP16 operands hold (kv_cells): the
    /// cells in blocks of kv_cells, each block's scores / exp / values as in the unblocked path (FP16 GEMMs, FP32
    /// accumulation), the blocks' outputs merged with an online softmax, then / l and the gate. q / gate / att as in
    /// layer(); att gets the gated output.
    /// STRATA_Q36_FA=1 (opt-in): the prompt attention on k_fa_prompt (needs 8 query heads a KV head and head_dim 256)
    bool fa_ok() const {
        static const bool on = [] { const char* v = std::getenv("STRATA_Q36_FA"); return v && v[0] == '1'; }();
        return on && m->g.n_head == kFaG * m->g.n_head_kv && m->g.head_dim == kFaHD;
    }
    /// STRATA_Q36_ATT_F16=1 (opt-in): the prompt attention's two GEMMs write FP16 (FP32 compute): rocBLAS's tuned kernel
    /// family on RDNA2 (FP16 in, FP32 out ran at ~3.3 TFLOPS there, FP16 out at ~9-13)
    static bool att16_on() {
        static const bool on = [] { const char* v = std::getenv("STRATA_Q36_ATT_F16"); return v && v[0] == '1'; }();
        return on;
    }
    /// The attention values GEMM (FP16 out): out [nt][H * HD] per query head of KV head kvh = V . P16. (rocBLAS's own
    /// solutions run it 1.2-2.3x faster than its default past 8K cells - tools/hip/tune_attn_rocblas - but some of them
    /// fault on shapes they accept, so the default stays until a checked table exists.)
    void attn_values16(const __half* V, const __half* P16, __half* out, int P, int nt, int ldp, int H, int HD, int G) {
        const float one = 1.0f, zero = 0.0f;
        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nt, P, &one, V, CUDA_R_16F, HD, 0, P16, CUDA_R_16F,
                                       ldp, (long long) nt * ldp, &zero, out, CUDA_R_16F, H * HD, HD, G, CUBLAS_COMPUTE_32F,
                                       CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
            std::fprintf(stderr, "strata-q36: attention values GEMM failed\n");
            std::exit(1);
        }
    }
    __half* o16_buf = nullptr;
    size_t o16_n = 0;
    __half* ensure_o16(size_t n) {
        if (n > o16_n) {
            if (o16_buf) CK(cudaFree(o16_buf));
            o16_buf = dalloc_t<__half>(n);
            o16_n = n;
        }
        return o16_buf;
    }
    float *blk_ob = nullptr, *blk_mb = nullptr, *blk_lb = nullptr, *blk_m = nullptr, *blk_l = nullptr;
    size_t blk_rows = 0;
    void attn_prompt_blocked(size_t li, int T, int64_t p0) {
        const auto& g = m->g;
        const int H = (int) g.n_head, HKV = (int) g.n_head_kv, HD = (int) g.head_dim, G = H / HKV;
        const int P = (int) p0 + T, B = (int) kv_cells();
        if (blk_rows < (size_t) T * H) {
            for (float** p : {&blk_ob, &blk_mb, &blk_lb, &blk_m, &blk_l})
                if (*p) CK(cudaFree(*p));
            blk_rows = (size_t) T * H;
            blk_ob = dalloc(blk_rows * HD);
            blk_mb = dalloc(blk_rows); blk_lb = dalloc(blk_rows);
            blk_m = dalloc(blk_rows); blk_l = dalloc(blk_rows);
        }
        const float one = 1.0f, zero = 0.0f, scale = 1.0f / sqrtf((float) HD);
        k_to_f16<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(q, q16, (int64_t) T * H * HD);
        const KvTab kt = kvtab(li);
        // STRATA_Q36_SP_PREFILL: two blocks - the window [0, W) dense, then a compact block of the selected far pages and
        // the local cells (the 4096 before the chunk and the chunk); else the cells in blocks of B
        const int W = kvr_nwin * KvRam::kSeg;
        int sp_S = 0, sp_rp = 0, sp_nr = 0;
        bool sparse_pf = false;
        if (sp_pf && P > W) {
            sp_build_state(li, p0);
            const int ls = std::max(W, (int) ((p0 - sp_recent) / kSpPage * kSpPage));   // the local cells' start
            const int pb = W / kSpPage, n_cand = ls / kSpPage - pb;
            sp_rp = ls / kSpPage;
            sp_nr = (P - ls + kSpPage - 1) / kSpPage;
            const int cap = B / kSpPage - sp_nr;
            if (n_cand * kSpPage >= 8192 && sp_rp <= sp_built[li] && cap > 0) {
                sp_S = std::max(1, std::min(cap, (int) std::ceil(sp_pf_frac * n_cand)));
                const int R = std::min(16, T) * G, stride = std::max(1, T / 16);
                k_sp_score_pf<<<dim3(n_cand, HKV), 256, 0, st>>>(q, H, HKV, R, stride, d_sp_stab + li * (size_t) kvr_nseg, pb,
                                                                  n_cand, sp_score);
                k_sp_select<<<HKV, 1024, 0, st>>>(sp_score, n_cand, sp_S, pb, sp_sel);
                sparse_pf = true;
            }
        }
        const int n_blocks = sparse_pf ? 2 : (P + B - 1) / B;
        for (int bi = 0; bi < n_blocks; ++bi) {
            int b0 = bi * B;
            int Pb = std::min(B, P - b0);
            if (sparse_pf) {
                b0 = bi == 0 ? 0 : sp_rp * kSpPage - sp_S * kSpPage;   // (the compact block's causal origin)
                Pb = bi == 0 ? W : (sp_S + sp_nr) * kSpPage;
            }
            const int ldp = (Pb + 7) / 8 * 8;
            const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldp))));
            if (sparse_pf && bi == 1) {
                k_sp_gather_f16<<<dim3(sp_S + sp_nr, HKV), 256, 0, st>>>(kt, sp_sel, sp_S, sp_rp, sp_nr, HKV, kv_qb(), k16, v16);
            } else {
                KvTab kb = kt;   // a block past the window: its host segments staged in VRAM first (see stage_copy)
                if (b0 >= kvr_nwin * KvRam::kSeg) {
                    const int s0 = b0 / KvRam::kSeg, s1 = (b0 + Pb + KvRam::kSeg - 1) / KvRam::kSeg;
                    stage_copy(li, s0, s1);
                    kb = stage_tab(s0);
                }
                k_kv_q8_f16_tab<<<grid1((int64_t) HKV * Pb * HD), kThreads, 0, st>>>(kb, k16, v16, b0, Pb, HD, HKV, kv_qb());
            }
            const size_t kv_head = (size_t) Pb * HD;
            if (att16_on()) {   // FP16-out GEMMs: the block normalized in the softmax, l_b put back after the values
                __half* o16b = ensure_o16((size_t) T * H * HD);
                for (int t0 = 0; t0 < T; t0 += tq) {
                    const int nt = std::min(tq, T - t0);
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, Pb, nt, HD, &scale, k16 + (size_t) kvh * kv_head,
                                CUDA_R_16F, HD, 0, q16 + ((size_t) t0 * H + kvh * G) * HD, CUDA_R_16F, H * HD, HD, &zero,
                                att_p16 + (size_t) kvh * G * nt * ldp, CUDA_R_16F, ldp, (long long) nt * ldp, G,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-q36: attention scores GEMM failed\n");
                            std::exit(1);
                        }
                    }
                    k_attn_softmax16_h_blk<<<dim3(nt, H), kThreads, 0, st>>>(att_p16, ldp, nt, Pb, b0, (int) p0 + t0,
                                                                             blk_mb + (size_t) t0 * H, blk_lb + (size_t) t0 * H);
                    for (int kvh = 0; kvh < HKV; ++kvh)
                        attn_values16(v16 + (size_t) kvh * kv_head, att_p16 + (size_t) kvh * G * nt * ldp,
                                      o16b + ((size_t) t0 * H + kvh * G) * HD, Pb, nt, ldp, H, HD, G);
                }
                k_h2f_rows<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(o16b, blk_lb, blk_ob, HD, (int64_t) T * H * HD);
                k_attn_blk_merge<<<T * H, HD, 0, st>>>(att, blk_ob, blk_m, blk_l, blk_mb, blk_lb, HD, bi == 0);
                continue;
            }
            if (fa_ok()) {   // the fused kernel: the block's unnormalized output and its row stats
                k_fa_prompt<false><<<dim3((T + kFaQ - 1) / kFaQ, HKV), 256, 0, st>>>(q16, k16, v16, T, H, Pb, Pb, b0, (int) p0,
                                                                                      scale, nullptr, blk_ob, blk_mb, blk_lb);
                k_attn_blk_merge<<<T * H, HD, 0, st>>>(att, blk_ob, blk_m, blk_l, blk_mb, blk_lb, HD, bi == 0);
                continue;
            }
            for (int t0 = 0; t0 < T; t0 += tq) {
                const int nt = std::min(tq, T - t0);
                for (int kvh = 0; kvh < HKV; ++kvh) {
                    const __half* Kh = k16 + (size_t) kvh * kv_head;
                    const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * HD;
                    float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                    if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, Pb, nt, HD, &one, Kh, CUDA_R_16F, HD, 0, Qh,
                                                   CUDA_R_16F, H * HD, HD, &zero, Sh, CUDA_R_32F, ldp, (long long) nt * ldp, G,
                                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                        std::fprintf(stderr, "strata-q36: attention scores GEMM failed\n");
                        std::exit(1);
                    }
                }
                // the slice's rows (t0 ..) of the block stats: mb / lb at row (t0 + t) * H + h
                k_attn_softmax16_blk<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, Pb, b0, (int) p0 + t0, scale,
                                                                       blk_mb + (size_t) t0 * H, blk_lb + (size_t) t0 * H);
                for (int kvh = 0; kvh < HKV; ++kvh) {
                    const __half* Vh = v16 + (size_t) kvh * kv_head;
                    const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                    float* Oh = blk_ob + ((size_t) t0 * H + kvh * G) * HD;
                    if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nt, Pb, &one, Vh, CUDA_R_16F, HD, 0, Sh,
                                                   CUDA_R_16F, ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F, H * HD, HD, G,
                                                   CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                        std::fprintf(stderr, "strata-q36: attention values GEMM failed\n");
                        std::exit(1);
                    }
                }
            }
            k_attn_blk_merge<<<T * H, HD, 0, st>>>(att, blk_ob, blk_m, blk_l, blk_mb, blk_lb, HD, bi == 0);
        }
        k_attn_blk_end<<<T * H, HD, 0, st>>>(att, blk_l, gate, HD);
    }

    /// One layer for T rows: x (the residual) += mixer(h); then MoE; x += moe; `hout` = rms(x) * wn (the next norm).
    /// `li` indexes the layer's state; positions *dpos .. + T - 1 (device), p0 the same on the host (eager grids);
    /// `verify`: a verify window - the GDN state is read, not written (commit() writes it for the kept rows).
    void layer(const Layer& L, size_t li, int T, int64_t p0, const int32_t* dpos, bool graph_mode, bool verify,
               const float* wn, float* hout) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k, FF = (int) g.n_ff, FS = (int) g.n_ff_shexp;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, D = (int) g.ssm_state_size;
        const int C = (int) g.ssm_conv_channels;
        const int H = (int) g.n_head, HKV = (int) g.n_head_kv, HD = (int) g.head_dim;
        cur_graph_mode = graph_mode;
        pmark("(between layers: the host reaching the next layer)");
        if (!L.attn) {
            float* qkv_b = verify ? v_qkv[li] : qkv;
            float* hc_b = verify ? v_hc[li] : hc;
            float* al_b = verify ? v_alpha[li] : alpha;
            float* be_b = verify ? v_beta[li] : beta;
            prep_in(T);
            apply(L.qkv, xq, qkv_b, T);
            apply(L.zg, xq, z, T);
            if (T <= kGemmT) {
                k_gdn_ab<<<dim3(rows8(2 * HV), T), kThreads, 0, st>>>(L.alpha, L.beta, h, L.dt, L.ssm_a, al_b, be_b, N,
                                                                      HV);
            } else {
                f32mm(L.alpha, h, al_b, N, HV, T);
                f32mm(L.beta, h, be_b, N, HV, T);
                k_gdn_gb<<<grid1((int64_t) HV * T), kThreads, 0, st>>>(al_b, L.dt, L.ssm_a, be_b, HV, T);
            }
            if (rows_mode) {   // batch rows: a slot's rows (one, or a verify window's two) in sequence on its own state
                k_conv_seq<<<dim3(C / 128, rows_G), 128, 0, st>>>(nullptr, qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, 1,
                                                                  !verify, d_conv_tab + li * n_sess, d_gslot, d_grow0, d_gcnt);
                k_gdn_seq<<<dim3(HV, rows_G), dim3(128, 4), 0, st>>>(nullptr, hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps,
                                                                     gdn_y, HK, HV, 1, !verify, d_S_tab + li * n_sess,
                                                                     d_gslot, d_grow0, d_gcnt);
            } else {
                // one token, a window or a chunk: the same kernels (every token in one launch each), so a decode step and a
                // verify window compute bitwise the same rows and the MTP's greedy text is the plain greedy text.  (Strata's
                // fused single-token kernels are not used here: fused_gdn_conv_l2's 4-tap sum contracts into FMAs
                // differently from k_conv_seq's, a 1-ulp difference in h that the Q8_1 activation rounding can turn into a
                // flipped near-tie later; with them the KL is the same, 1.05x / 0.81x the floor, but MTP would not be exact.)
                // a prompt chunk: tiles of kConvTile tokens side by side (tile mode); else one block per head walks the tokens
                const int tiles = (fast_k && T > kGemmT && !verify) ? (T + kConvTile - 1) / kConvTile : 1;
                k_conv_seq<<<dim3(C / 128, tiles), 128, 0, st>>>(conv[li], qkv_b, L.conv, hc_b, C, 2 * HK, m->eps, T, !verify);
                if (T > kGemmT && !verify) {   // a prompt chunk: the column-split delta rule, then the norm
                    k_gdn_seq_cols<<<dim3(HV, kGdnColGroups), dim3(128 / kGdnColGroups, 16), 0, st>>>(S[li], hc_b, C, al_b,
                                                                                                      be_b, gdn_y, HK, HV, T);
                    k_gdn_norm<<<dim3(HV, T), 128, 0, st>>>(gdn_y, z, L.ssm_norm, m->eps, gdn_y, HV);
                } else {   // decode, verify windows and commits: one kernel, so a window's rows are a decode step's rows
                    k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[li], hc_b, C, al_b, be_b, z, L.ssm_norm, m->eps, gdn_y, HK, HV,
                                                           T, !verify);
                }
            }
            mm(L.ssm_out, gdn_y, y, T);
        } else {
            prep_in(T);
            apply(L.wq, xq, qg, T);
            apply(L.wk, xq, kx, T);
            apply(L.wv, xq, vx, T);
            RowSel rs;
            if (rows_mode) {
                rs.cell = d_rcell;
                rs.slot = d_rslot;
                rs.delta = d_rdelta;
                rs.kc = d_kc_tab + li * n_sess;
                rs.vc = d_vc_tab + li * n_sess;
            }
            if (kvq8())
                k_attn_prep<int8_t><<<dim3(H + HKV, T), kThreads, 0, st>>>(qg, kx, vx, L.q_norm, L.k_norm, q, gate,
                    rows_mode ? nullptr : (int8_t*) kc[li], rows_mode ? nullptr : (int8_t*) vc[li], H, HKV, HD, m->n_rot,
                    dpos, ctx, m->rope_base, m->eps, d_delta, mtab_now, rs, kv_qb(), rows_mode ? KvTab() : kvtab(li));
            else if (kv16())
                k_attn_prep<__half><<<dim3(H + HKV, T), kThreads, 0, st>>>(qg, kx, vx, L.q_norm, L.k_norm, q, gate,
                    rows_mode ? nullptr : (__half*) kc[li], rows_mode ? nullptr : (__half*) vc[li], H, HKV, HD, m->n_rot,
                    dpos, ctx, m->rope_base, m->eps, d_delta, mtab_now, rs);
            else
                k_attn_prep<float><<<dim3(H + HKV, T), kThreads, 0, st>>>(qg, kx, vx, L.q_norm, L.k_norm, q, gate,
                    rows_mode ? nullptr : kc[li], rows_mode ? nullptr : vc[li], H, HKV, HD, m->n_rot, dpos, ctx,
                    m->rope_base, m->eps, d_delta, mtab_now, rs);
            if (T > kGemmT) {
                // a prompt chunk: scores = Q K^T and out = P V as FP16 GEMMs (FP32 accumulation, FP32 scores) per KV
                // head (its query heads one strided batch over the shared K / V), the causal softmax in between writing
                // FP16 probabilities; the queries in slices whose scores fit
                const int P = (int) p0 + T;
                if (kvr_on && P > (int) kv_cells()) {   // KV_RAM past the FP16 operands' room: blocks of it
                    attn_prompt_blocked(li, T, p0);
                } else {
                const int ldp = (P + 7) / 8 * 8;
                const int tq = (int) std::max<int64_t>(1, std::min<int64_t>(T, (int64_t) (kAttScoreFloats / ((size_t) H * ldp))));
                const int G = H / HKV;
                const float one = 1.0f, zero = 0.0f;
                // f16 cache: the GEMMs read its cells in place (a KV head's cells are contiguous, ctx rows apart per head)
                const bool in_place = kv16();
                if (kvq8() && kvr_on)
                    k_kv_q8_f16_tab<<<grid1((int64_t) HKV * P * HD), kThreads, 0, st>>>(kvtab(li), k16, v16, 0, P, HD, HKV,
                                                                                         kv_qb());
                else if (kvq8())
                    k_kv_q8_f16<<<grid1((int64_t) HKV * P * HD), kThreads, 0, st>>>((const int8_t*) kc[li],
                        (const int8_t*) vc[li], k16, v16, ctx, P, HD, HKV, kv_qb());
                else if (!in_place)
                    k_kv_f16<<<grid1((int64_t) HKV * P * HD), kThreads, 0, st>>>(kc[li], vc[li], k16, v16, ctx, P, HD, HKV);
                const __half* kbase = in_place ? (const __half*) kc[li] : k16;
                const __half* vbase = in_place ? (const __half*) vc[li] : v16;
                const size_t kv_head = in_place ? (size_t) ctx * HD : (size_t) P * HD;
                k_to_f16<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(q, q16, (int64_t) T * H * HD);
                if (fa_ok()) {   // STRATA_Q36_FA: the fused kernel (the gate in its epilogue)
                    k_fa_prompt<true><<<dim3((T + kFaQ - 1) / kFaQ, HKV), 256, 0, st>>>(q16, kbase, vbase, T, H, P,
                        (int) (kv_head / HD), 0, (int) p0, 1.0f / sqrtf((float) HD), gate, att, nullptr, nullptr);
                } else if (att16_on()) {   // STRATA_Q36_ATT_F16: both GEMMs FP16 out (FP32 compute), the scale in alpha
                    const float sc = 1.0f / sqrtf((float) HD);
                    __half* o16b = ensure_o16((size_t) T * H * HD);
                    for (int t0 = 0; t0 < T; t0 += tq) {
                        const int nt = std::min(tq, T - t0);
                        for (int kvh = 0; kvh < HKV; ++kvh) {
                            if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, HD, &sc, kbase + (size_t) kvh * kv_head,
                                    CUDA_R_16F, HD, 0, q16 + ((size_t) t0 * H + kvh * G) * HD, CUDA_R_16F, H * HD, HD, &zero,
                                    att_p16 + (size_t) kvh * G * nt * ldp, CUDA_R_16F, ldp, (long long) nt * ldp, G,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) != CUBLAS_STATUS_SUCCESS) {
                                std::fprintf(stderr, "strata-q36: attention scores GEMM failed\n");
                                std::exit(1);
                            }
                        }
                        k_attn_softmax16_h<<<dim3(nt, H), kThreads, 0, st>>>(att_p16, ldp, nt, P, (int) p0 + t0);
                        for (int kvh = 0; kvh < HKV; ++kvh)
                            attn_values16(vbase + (size_t) kvh * kv_head, att_p16 + (size_t) kvh * G * nt * ldp,
                                          o16b + ((size_t) t0 * H + kvh * G) * HD, P, nt, ldp, H, HD, G);
                    }
                    k_h2f_gate<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(o16b, gate, att, (int64_t) T * H * HD);
                } else {
                for (int t0 = 0; t0 < T; t0 += tq) {
                    const int nt = std::min(tq, T - t0);
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const __half* Kh = kbase + (size_t) kvh * kv_head;
                        const __half* Qh = q16 + ((size_t) t0 * H + kvh * G) * HD;
                        float* Sh = att_s + (size_t) kvh * G * nt * ldp;
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, P, nt, HD, &one, Kh, CUDA_R_16F, HD,
                                                       0, Qh, CUDA_R_16F, H * HD, HD, &zero, Sh, CUDA_R_32F, ldp,
                                                       (long long) nt * ldp, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                            CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-q36: attention scores GEMM failed\n");
                            std::exit(1);
                        }
                    }
                    k_attn_softmax16<<<dim3(nt, H), kThreads, 0, st>>>(att_s, att_p16, ldp, nt, P, (int) p0 + t0,
                                                                       1.0f / sqrtf((float) HD));
                    for (int kvh = 0; kvh < HKV; ++kvh) {
                        const __half* Vh = vbase + (size_t) kvh * kv_head;
                        const __half* Sh = att_p16 + (size_t) kvh * G * nt * ldp;
                        float* Oh = att + ((size_t) t0 * H + kvh * G) * HD;
                        if (cublasGemmStridedBatchedEx(blas, CUBLAS_OP_N, CUBLAS_OP_N, HD, nt, P, &one, Vh, CUDA_R_16F, HD,
                                                       0, Sh, CUDA_R_16F, ldp, (long long) nt * ldp, &zero, Oh, CUDA_R_32F,
                                                       H * HD, HD, G, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT) !=
                            CUBLAS_STATUS_SUCCESS) {
                            std::fprintf(stderr, "strata-q36: attention values GEMM failed\n");
                            std::exit(1);
                        }
                    }
                }
                k_gate_mul<<<grid1((int64_t) T * H * HD), kThreads, 0, st>>>(att, gate, (int64_t) T * H * HD);
                }   // (the GEMM path)
                }   // (the unblocked prompt attention)
            } else {
                // split-K over 256-cell chunks; the chunk grid covers every cell a row can see (graph mode: the whole
                // context, past-the-end chunks return at once)
                const int rows_per = att_rows / H;
                for (int t0 = 0; t0 < T; t0 += rows_per) {
                    const int nt = std::min(rows_per, T - t0);
                    const int nc = graph_mode ? n_chunk
                                              : (int) std::min<int64_t>(n_chunk, (p0 + t0 + nt + kChunkCells - 1) / kChunkCells);
                    int m_win = -1, m_bias = 0;   // the merge's sparse layout (STRATA_Q36_SPARSE)
                    if (kvq8()) {
                        const int8_t* kq = rows_mode ? nullptr : (const int8_t*) kc[li];
                        const int8_t* vq = rows_mode ? nullptr : (const int8_t*) vc[li];
                        const KvTab kt = rows_mode ? KvTab() : kvtab(li);
                        auto part = [&](const KvTab& tb, int c0, int c1) {   // chunks [c0, c1)
                            if (c1 <= c0) return;
                            if (kv_dot8())
                                k_attn_part_q8<true><<<dim3(c1 - c0, nt * H), kChunkCells, 0, st>>>(q, kq, vq, dpos, t0, H, HKV,
                                    ctx, 1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs, kv_qb(), tb, c0);
                            else
                                k_attn_part_q8<false><<<dim3(c1 - c0, nt * H), kChunkCells, 0, st>>>(q, kq, vq, dpos, t0, H, HKV,
                                    ctx, 1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs, kv_qb(), tb, c0);
                        };
                        const int win_c = kvr_on ? kvr_nwin * (KvRam::kSeg / kChunkCells) : nc;
                        // sparse: the window's cells, the best-scored far pages and the recent cells (see sp_plan)
                        SpPlan sp;
                        if (!rows_mode && nc > win_c && sp_plan(p0 + t0 + nt, sp)) {
                            part(kt, 0, win_c);
                            k_sp_score<<<dim3(sp.n_cand, HKV), 256, 0, st>>>(q + (size_t) t0 * H * HD, nt, H, HKV,
                                d_sp_stab + li * (size_t) kvr_nseg, sp.pb, sp.n_cand, sp_score, sp_lm);
                            k_sp_select<<<HKV, 1024, 0, st>>>(sp_score, sp.n_cand, sp.S, sp.pb, sp_sel);
                            if (sp_lm) {   // the unselected pages as one estimated partial, in the last slot
                                k_sp_mark<<<HKV, 256, 0, st>>>(sp_sel, sp.S, sp.pb, sp.n_cand, sp_mark);
                                k_sp_tail<<<nt * H, 256, 0, st>>>(sp_lm, sp_mark, d_sp_stab + li * (size_t) kvr_nseg, sp.pb,
                                    sp.n_cand, nt, H, HKV, n_chunk - 1, n_chunk, pacc, pm, pl);
                            }
                            k_sp_gather<<<dim3(sp.S + sp.nr, HKV), 256, 0, st>>>(kt, sp_sel, sp.S, sp.rp, HKV, kv_qb(), sp_cap,
                                                                                    sp_ck, sp_cv);
                            const int cc = (p0 + t0 + nt + sp.bias + kChunkCells - 1) / kChunkCells;
                            if (kv_dot8())
                                k_attn_part_q8<true><<<dim3(cc, nt * H), kChunkCells, 0, st>>>(q, sp_ck, sp_cv, dpos, t0, H, HKV,
                                    sp_cap, 1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs, kv_qb(), KvTab(), 0, sp.bias, win_c);
                            else
                                k_attn_part_q8<false><<<dim3(cc, nt * H), kChunkCells, 0, st>>>(q, sp_ck, sp_cv, dpos, t0, H, HKV,
                                    sp_cap, 1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs, kv_qb(), KvTab(), 0, sp.bias, win_c);
                            m_win = win_c;
                            m_bias = sp.bias;
                        } else if (!kvr_on || rows_mode || nc <= win_c) {
                            part(kt, 0, nc);
                        } else {   // KV_RAM past the window: the window's chunks, then the host cells in staged rounds
                            part(kt, 0, win_c);
                            const int cps = KvRam::kSeg / kChunkCells;   // chunks a segment
                            const int s_end = (nc + cps - 1) / cps;
                            for (int s0 = kvr_nwin; s0 < s_end; s0 += kvr_nwin) {
                                const int s1 = std::min(s_end, s0 + kvr_nwin);
                                stage_copy(li, s0, s1);
                                part(stage_tab(s0), s0 * cps, std::min(nc, s1 * cps));
                            }
                        }
                    } else if (kv16())
                        k_attn_part<__half><<<dim3(nc, nt * H), kChunkCells, 0, st>>>(q,
                            rows_mode ? nullptr : (const __half*) kc[li], rows_mode ? nullptr : (const __half*) vc[li], dpos, t0,
                            H, HKV, ctx, 1.0f / sqrtf((float) HD), n_chunk, pacc, pm, pl, rs);
                    else
                        k_attn_part<float><<<dim3(nc, nt * H), kChunkCells, 0, st>>>(q, rows_mode ? nullptr : kc[li],
                            rows_mode ? nullptr : vc[li], dpos, t0, H, HKV, ctx, 1.0f / sqrtf((float) HD), n_chunk, pacc, pm,
                            pl, rs);
                    k_attn_merge<<<nt * H, HD, 0, st>>>(pacc, pm, pl, gate, dpos, t0, H, n_chunk, att, rs, m_win, m_bias,
                                                        m_win >= 0 && sp_lm ? n_chunk - 1 : -1);
                }
            }
            mm(L.wo, att, y, T);
        }
        pmark(L.attn ? "attention layer: projections + attention" : "delta-net layer: projections + recurrence");
        add_rms_post(L, T);
        // routed experts (one token: the router and the shared expert's gate in one product, the gate its last row)
        const bool one = T == 1;
        // a decode batch of up to 8 rows takes the router and the shared expert's gate in one product too (rows of 257) when the fused
        // router follows; the prompt chunks and the checks keep the separate products
        const bool ext = one || (fast_k && std::getenv("STRATA_Q36_NO_EXT") == nullptr && !verify_k && T <= kGemmT && g.n_expert == 256 && Kp == 8);
        if (ext) f32mm(L.router_ext, h, logits_r, N, (int) g.n_expert + 1, T);
        else f32mm(L.router, h, logits_r, N, (int) g.n_expert, T);
        // the hybrid rows path: a batch of up to hybrid_t rows; when only the MTP block is hybrid, its every batch
        // (its prompt pass too: no staging buffers then)
        const bool hy_rows = lhy(li) && (T <= hybrid_t || !main_hybrid);
        const bool mmq_rows = !hy_rows && T > kGemmT && use_mmq;
        if (!(!mmq_rows && route_group(L, hy_rows ? hy[li].slot_of_d : nullptr, T, ext ? (int) g.n_expert + 1 : (int) g.n_expert)))
            K::native_router_topk(logits_r, ids, ewts, (int) g.n_expert, Kp, T, st);
        pmark("norm + router + top-k");
        const int64_t n_ent = (int64_t) T * Kp;
        const int32_t* prow = nullptr;
        if (hy_rows) {
            moe_hybrid_gpu(L, li, T);
        } else if (T > kGemmT && use_mmq) {
            // a prompt chunk: the experts on llama.cpp's MMQ (int8, the weights read once per expert): the entries
            // sorted by expert on the device, gate+up as one [2 FF] product per expert, SwiGLU, down into rows the
            // combine reads through entry -> sorted row
            const int ne = (int) g.n_expert;
            CK(cudaMemsetAsync(rcnt, 0, (size_t) ne * 4, st));
            k_route_count<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, rcnt);
            k_route_scan<<<1, 32, 0, st>>>(rcnt, ne, rbounds, rcur, rmax);
            k_route_scatter<<<grid1(n_ent), kThreads, 0, st>>>(ids, (int) n_ent, Kp, rcur, ent_dst, ent_tok, rinv);
            CK(cudaMemcpyAsync(hpin + 10, rmax, 4, cudaMemcpyDeviceToHost, st));
            if (lhy(li)) CK(cudaMemcpyAsync(h_cnt + li * g.n_expert, rcnt, (size_t) ne * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            if (lhy(li))
                for (int x = 0; x < ne; ++x) hy[li].score[(size_t) x] += h_cnt[li * g.n_expert + x];
            const int64_t maxr = hpin[10];
            pmark("moe: routing sort + host sync");
            // layer-major prompts (STRATA_Q36_LM): a layer's streamed experts are taken by its first chunk of the
            // super-chunk and given back after its last
            const uint8_t* wbase = !lhy(li) ? L.blobs
                                 : !lm_active ? stream_take((int) li)
                                 : lm_first ? (lm_wbase = stream_take((int) li)) : lm_wbase;
            pmark("moe: waiting for the stream");
            MMQ::quantize(h, ent_tok, mq_x, L.L.gu_type, N, N, n_ent, st);
            MMQ::Product gu;
            gu.w = wbase; gu.type = L.L.gu_type; gu.w_rows = 2 * FF; gu.w_cols = N; gu.expert_bytes = L.stride;
            gu.n = ne; gu.xq = mq_x; gu.bounds = rbounds; gu.ids = ident; gu.total_rows = n_ent; gu.max_rows = maxr;
            gu.dst = mq_gu; gu.ld_dst = 2 * FF;
            mctx->run(gu, st);
            pmark("moe: quantize + gate/up MMQ");
            MMQ::swiglu(mq_gu, mq_h, n_ent, FF, false, st);
            MMQ::quantize(mq_h, nullptr, mq_hq, L.L.d_type, FF, FF, n_ent, st);
            MMQ::Product dn;
            dn.w = wbase + L.L.down_off; dn.type = L.L.d_type; dn.w_rows = N; dn.w_cols = FF;
            dn.expert_bytes = L.stride; dn.n = ne; dn.xq = mq_hq; dn.bounds = rbounds; dn.ids = ident;
            dn.total_rows = n_ent; dn.max_rows = maxr; dn.dst = eparts; dn.ld_dst = N;
            mctx->run(dn, st);
            pmark("moe: swiglu + down MMQ");
            if (lhy(li) && (!lm_active || lm_last)) stream_done();
            prow = rinv;
        } else {
            group_entries(L, nullptr, T);
            if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
            K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                     std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
        }
        pmark("routed experts (incl. waiting for the stream)");
        // the shared expert, scaled by sigmoid(w . h)
        if (T > kGemmT) k_to_f16<<<grid1((int64_t) N * T), kThreads, 0, st>>>(h, x16, (int64_t) N * T);
        if (T <= kGemmT) {
            if (!shared_gu(L, T)) {
                apply(L.sh_gate, xq, sg, T);
                apply(L.sh_up, xq, su, T);
                K::native_swiglu_quantize_q8_1(sg, su, q8, FS, T, st);   // SwiGLU and its q8_1 image in one launch
            }
            mmq(L.sh_down, q8, sy, T);
        } else {
            apply(L.sh_gate, xq, sg, T);
            apply(L.sh_up, xq, su, T);
            k_swiglu<<<grid1((int64_t) T * FS), kThreads, 0, st>>>(sg, su, shh, (int64_t) T * FS);
            mm(L.sh_down, shh, sy, T);
        }
        if (!ext) f32mm(L.sh_gate_inp, h, sgate, N, 1, T);
        if (hy_rows) moe_hybrid_cpu(li, T);   // the CPU pool's experts (while the GPU runs the rest), merged
        // the combine, the shared expert's gate, the residual add and the next norm in one launch
        moe_end(T, prow, sy, ext ? logits_r + g.n_expert : sgate, wn, hout, ext ? (int) g.n_expert + 1 : 1);
        pmark("shared expert + combine");
    }

    /// STRATA_Q36_LM=1 (opt-in): layer-major prompts. A prompt's full chunks run in super-chunks of up to
    /// STRATA_Q36_LM_CHUNKS (4) chunks: every layer over all of a super-chunk's chunks before the next layer, so each
    /// layer's experts cross PCIe once per super-chunk instead of once per chunk (the chunk-major order streamed all
    /// 18.7 GiB of experts for every 2048 tokens). Within a layer the chunks run in order, so the recurrent state and the
    /// cache see what the chunk-major order gives them: the same arithmetic. The super-chunk's residual and normed rows
    /// live in lm_X / lm_H. The MTP block's pass over the prompt runs the same way.
    bool lm_on = false, lm_active = false, lm_first = false, lm_last = false;
    int lm_chunks = 4;
    const uint8_t* lm_wbase = nullptr;
    float *lm_X = nullptr, *lm_H = nullptr;
    int32_t *lm_tok = nullptr, *lm_pos = nullptr;
    /// ns rows at p0 (chunks of tmax rows, the last one may be shorter but above hybrid_t), their tokens on the host;
    /// the last row's logits when `head`
    void forward_lm(int ns, int64_t p0, const int32_t* toks, bool head) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, T = tmax;
        const int nsub = (ns + T - 1) / T;
        auto rows = [&](int j) { return std::min(T, ns - j * T); };
        kvr_ensure(p0 + ns);
        ensure_set(T);
        CK(cudaStreamSynchronize(st));   // (the pinned words are reused)
        for (int j = 0; j < nsub; ++j) hpin[16 + j] = (int32_t) (p0 + (int64_t) j * T);
        CK(cudaMemcpyAsync(lm_tok, toks, (size_t) ns * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(lm_pos, hpin + 16, (size_t) nsub * 4, cudaMemcpyHostToDevice, st));
        drop_xslots();
        std::vector<int> seq;
        for (size_t l = 0; l < m->layers.size(); ++l)
            if (lhy(l)) seq.push_back((int) l);
        stream_begin(seq);
        float* const x0 = x;
        float* const h0 = h;
        for (int j = 0; j < nsub; ++j) {
            x = lm_X + (size_t) j * T * N;
            h = lm_H + (size_t) j * T * N;
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, lm_tok + (size_t) j * T, rows(j), N, x, st);
            k_rms<<<rows(j), kThreads, 0, st>>>(x, m->layers[0].attn_norm, h, N, m->eps);
        }
        lm_active = true;
        const auto lm_t0 = Clock::now();
        if (prof_on()) { pmark_on = true; pmark_n = 0; }   // STRATA_Q36_PROF=2: the super-chunk's stages, summed
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const bool last = l + 1 == m->layers.size();
            for (int j = 0; j < nsub; ++j) {
                x = lm_X + (size_t) j * T * N;
                h = lm_H + (size_t) j * T * N;
                lm_first = j == 0;
                lm_last = j == nsub - 1;
                layer(m->layers[l], l, rows(j), p0 + (int64_t) j * T, lm_pos + j, false, false,
                      last ? m->out_norm : m->layers[l + 1].attn_norm, h);
            }
        }
        lm_active = false;
        x = x0;
        h = h0;
        if (head) head_rows(lm_H + (size_t) (ns - 1) * N, 1, logits, pick);
        if (pmark_on) {
            CK(cudaDeviceSynchronize());
            std::map<std::string, float> by;
            for (size_t i = 1; i < pmark_n; ++i) {
                float ms = 0.f;
                CK(cudaEventElapsedTime(&ms, pmarks[i - 1].second, pmarks[i].second));
                by[pmarks[i].first] += ms;
            }
            for (const auto& kv : by) std::fprintf(stderr, "strata-q36 lm prof:   GPU %7.0f ms  %s\n", kv.second, kv.first.c_str());
            std::fprintf(stderr, "strata-q36 lm prof: %d x %d rows at %lld: %.0f ms\n", nsub, T, (long long) p0, ms_since(lm_t0));
            pmark_on = false;
        }
    }
    /// The MTP block over the super-chunk forward_lm just ran (its K / V cells for those positions)
    void forward_mtp_lm(int ns, int64_t p0) {
        const int N = (int) m->g.n_embd, T = tmax;
        const int nsub = (ns + T - 1) / T;
        lm_active = true;
        for (int j = 0; j < nsub; ++j) {
            const float* seeds;
            if (j == 0) {   // h_{q-1} for each row q: hlast, then the super-chunk's rows 0 .. T - 2
                CK(cudaMemcpyAsync(mtp_seed(), hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(mtp_seed() + N, lm_H, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, st));
                seeds = mtp_seed();
            } else {
                seeds = lm_H + ((size_t) j * T - 1) * N;
            }
            const int Tj = std::min(T, ns - j * T);
            CK(cudaMemcpyAsync(tok_m, lm_tok + (size_t) j * T, (size_t) Tj * 4, cudaMemcpyDeviceToDevice, st));
            CK(cudaMemcpyAsync(pos0_m, lm_pos + j, 4, cudaMemcpyDeviceToDevice, st));
            lm_first = j == 0;
            lm_last = j == nsub - 1;
            forward_mtp(Tj, p0 + (int64_t) j * T, seeds, false, false);
        }
        lm_active = false;
    }

    /// The main model over T rows (device `tok`, positions p0 .. p0 + T - 1, also at *pos0): hmain gets every row's
    /// final normed hidden; `heads` rows from the end get logits (rows [T - heads, T): logits row r - (T - heads)) and
    /// their greedy picks (`pick`).  `verify`: a verify window (the GDN state left for commit()).
    void forward(int T, int64_t p0, int heads, bool graph_mode, bool verify) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        ensure_set(T);
        kvr_ensure(p0 + T);   // KV_RAM: memory for the cells this forward writes
        if (T <= kGemmT) sp_build(p0);   // sparse decode: the statistics of the pages completed so far
        const bool streamed = main_hybrid && T > hybrid_t && use_mmq;
        const auto prof_t0 = Clock::now();
        // STRATA_Q36_PROF=3: the small forwards' (decode, verify) GPU timeline split by stage, summed over the run and
        // printed at exit; no host syncs between the stages, so a stage's time includes the GPU idling on the host
        const bool dprof = !streamed && T <= 16 && dprof_on();
        if (dprof) {
            pmark_on = true;
            pmark_n = 0;
        }
        if (streamed) {
            std::vector<int> seq;
            for (size_t l = 0; l < m->layers.size(); ++l)
                if (lhy(l)) seq.push_back((int) l);
            prof_n = 0;
            prof_host_ms = 0.0;
            prof_wait_ms = 0.0;
            drop_xslots();
            pmark_on = prof_on();
            pmark_n = 0;
            stream_begin(seq);
        }
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok, T, N, x, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(x, d_img, d_map, N);   // a picture's cells take its rows
        k_rms<<<T, kThreads, 0, st>>>(x, m->layers[0].attn_norm, h, N, m->eps);   // later layers: from k_moe_out
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const bool last = l + 1 == m->layers.size();
            layer(m->layers[l], l, T, p0, pos0, graph_mode, verify, last ? m->out_norm : m->layers[l + 1].attn_norm,
                  last ? hmain : h);
        }
        if (heads > 0) head_rows(hmain + (size_t) (T - heads) * N, heads, logits, pick);
        ho_run();   // the handoff mode: the CPU experts of the forward just enqueued
        if (dprof) {
            pmark("output norm + head");
            CK(cudaStreamSynchronize(st));
            for (size_t i = 1; i < pmark_n; ++i) {
                float ms = 0.f;
                CK(cudaEventElapsedTime(&ms, pmarks[i - 1].second, pmarks[i].second));
                dprof_by[pmarks[i].first] += ms;
                dprof_host[pmarks[i].first] += std::chrono::duration<double, std::milli>(pmark_t[i] - pmark_t[i - 1]).count();
            }
            ++dprof_n;
            pmark_on = false;
        }
        if (!streamed && T <= 16) {
            fwd_small_ms += ms_since(prof_t0);
            ++fwd_small_n;
        }
        if (streamed && prof_on()) {
            CK(cudaDeviceSynchronize());
            std::map<std::string, float> by;
            std::map<std::string, double> by_host;
            for (size_t i = 1; i < pmark_n; ++i) {
                float ms = 0.f;
                CK(cudaEventElapsedTime(&ms, pmarks[i - 1].second, pmarks[i].second));
                by[pmarks[i].first] += ms;
                by_host[pmarks[i].first] += std::chrono::duration<double, std::milli>(pmark_t[i] - pmark_t[i - 1]).count();
            }
            for (const auto& kv : by) std::fprintf(stderr, "strata-q36 prof:   GPU %7.0f ms  host %7.0f ms  %s\n", kv.second, by_host[kv.first], kv.first.c_str());
            pmark_on = false;
            float copy_ms = 0.f;
            for (size_t i = 0; i < prof_n; ++i) {
                float ms = 0.f;
                CK(cudaEventElapsedTime(&ms, prof_ev[i].first, prof_ev[i].second));
                copy_ms += ms;
            }
            std::fprintf(stderr, "strata-q36 prof: T %d at %lld: forward %.0f ms, expert copies %.0f ms on the GPU, %.0f ms on the host, %.0f ms waited (%zu layers)\n", T,
                         (long long) p0, ms_since(prof_t0), copy_ms, prof_host_ms, prof_wait_ms, prof_n);
        }
    }
    /// logits [n][V] = W_out hn rows, and each row's greedy pick.
    void head_rows(const float* hn, int n, float* lg, int32_t* pk) {
        const auto& g = m->g;
        mm(m->head, hn, lg, n);
        k_argmax<<<dim3(kArgBlocks, n), kThreads, 0, st>>>(lg, (int) g.n_vocab, pk, amv, ami, false);
        k_argmax<<<dim3(1, n), kThreads, 0, st>>>(lg, kArgBlocks, pk, amv, ami, true);
    }

    /// The draft head: the MTP block's pick over the first `draft_n` rows of the output matrix only (the lowest token ids are
    /// the most frequent ones).  A draft only proposes: the verify window decides, so a token outside the subset is a
    /// rejected draft and nothing else - the answer is the plain greedy answer whatever draft_n is.  0 = the whole vocabulary.
    int draft_n = 0;
    void head_draft(const float* hn) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        const Mat& hd = m->head;
        if (draft_n > 0 && draft_n < g.n_vocab && draft_n % 8 == 0 && hd.type != 8) {
            // any other head format (Q6_K in UD-Q4_K_M): its first draft_n rows are contiguous, so the plain row
            // product over a view with fewer rows; the same rows as the full head computes, only fewer of them
            Mat v = hd;
            v.n_out = draft_n;
            mm(v, hn, logits_m, 1);
            k_argmax<<<dim3(kArgBlocks, 1), kThreads, 0, st>>>(logits_m, draft_n, pick_m, amv, ami, false);
            k_argmax<<<dim3(1, 1), kThreads, 0, st>>>(logits_m, kArgBlocks, pick_m, amv, ami, true);
            return;
        }
        if (!fast_k || draft_n <= 0 || draft_n >= g.n_vocab || draft_n % 8 != 0 || hd.type != 8 || hd.n_in != N || N / 32 != 64) {
            head_rows(hn, 1, logits_m, pick_m);
            return;
        }
        quant(hn, N, 1, q8);
        const uint8_t* pk = nullptr;
        for (const auto& p : m->packed)
            if (std::get<0>(p) == hd.d) pk = std::get<1>(p);
        if (pk) {
            const half* dpl = reinterpret_cast<const half*>(pk + (size_t) hd.n_in * hd.n_out);
            q36k::launch_q8_rows<1, 2, true>(pk, dpl, q8, logits_m, 64, draft_n, st);
        } else {
            q36k::launch_q8_rows<1, 2, false>(hd.d, nullptr, q8, logits_m, 64, draft_n, st);
        }
        k_argmax<<<dim3(kArgBlocks, 1), kThreads, 0, st>>>(logits_m, draft_n, pick_m, amv, ami, false);
        k_argmax<<<dim3(1, 1), kThreads, 0, st>>>(logits_m, kArgBlocks, pick_m, amv, ami, true);
    }

    /// The MTP block over T rows: row t reads token tok_m[t] at position p0 + t (also at *pos0_m) and the seed hidden
    /// seeds[t] (the main model's normed hidden of the position before, or the block's own for a chained draft);
    /// hmtp gets each row's normed output.  `draft`: logits of the last row and its pick (`pick_m`).
    void forward_mtp(int T, int64_t p0, const float* seeds, bool draft, bool graph_mode) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        kvr_ensure(p0 + T);
        if (T <= kGemmT) sp_build(p0);
        ensure_set(T);
        if (main_hybrid && lhy(m->layers.size()) && T > hybrid_t && use_mmq && (!lm_active || lm_first)) {
            drop_xslots();   // the stream overwrites the staging buffers that hold extra slots
            stream_begin({(int) m->layers.size()});
        }
        K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
        if (map_on) k_put_rows<<<T, kThreads, 0, st>>>(y, d_img, d_map, N);
        k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);              // [e | ...]
        k_rms_ld<<<T, kThreads, 0, st>>>(seeds, N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);      // [... | h]
        mm(m->eh_proj, mtp_cat, x, T);
        k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
        layer(m->mtp, m->layers.size(), T, p0, pos0_m, graph_mode, false, m->mtp_shnorm, hmtp);
        if (draft) head_draft(hmtp + (size_t) (T - 1) * N);
        ho_run();
    }

    /// After a verify window of T rows: the GDN state takes the first n_keep rows (conv history and recurrence run
    /// again over the rows' kept inputs).
    void commit(int n_keep) {
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        for (size_t l = 0; l < m->layers.size(); ++l) {
            const Layer& L = m->layers[l];
            if (L.attn) continue;
            k_conv_seq<<<C / 128, 128, 0, st>>>(conv[l], v_qkv[l], L.conv, hc, C, 2 * HK, m->eps, n_keep, true);
            k_gdn_seq<<<HV, dim3(128, 4), 0, st>>>(S[l], v_hc[l], C, v_alpha[l], v_beta[l], z, L.ssm_norm, m->eps,
                                                   nullptr, HK, HV, n_keep, true);
        }
    }

    // ---- the hybrid mode
    const Layer& layer_at(int li) const { return li < (int) m->layers.size() ? m->layers[(size_t) li] : m->mtp; }

    /// A streamed forward over the state indices `seq`: the first two layers' experts start copying now.
    void stream_begin(const std::vector<int>& seq) {
        stream_seq = seq;
        stream_pos = 0;
        for (int i = 0; i < 2 && i < (int) seq.size(); ++i) stream_kick(i);
    }
    // STRATA_Q36_PROF=1: each streamed prompt forward prints the GPU time of its expert copies against its wall time
    std::vector<std::pair<cudaEvent_t, cudaEvent_t>> prof_ev;
    size_t prof_n = 0;
    double prof_host_ms = 0.0;   // host time inside stream_kick (a pageable source is copied by the calling thread)
    double prof_wait_ms = 0.0;   // host time in stream_take waiting for the streamer
    static bool prof_on() {
        static const bool on = [] { const char* v = std::getenv("STRATA_Q36_PROF"); return v && v[0] != '3'; }();
        return on;
    }
    static bool dprof_on() {
        static const bool on = [] { const char* v = std::getenv("STRATA_Q36_PROF"); return v && v[0] == '3'; }();
        return on;
    }
    std::map<std::string, double> dprof_by, dprof_host;   // STRATA_Q36_PROF=3: per stage, summed over the small forwards
    long long dprof_n = 0;
    // STRATA_Q36_PROF=1 also splits a streamed prompt forward's GPU time by stage: events between the stages of each
    // layer, summed per stage name at the end of the forward
    std::vector<std::pair<const char*, cudaEvent_t>> pmarks;
    std::vector<Clock::time_point> pmark_t;   // the host's time at each mark
    size_t pmark_n = 0;
    bool pmark_on = false;   // set for the streamed forwards while profiling
    void pmark(const char* name) {
        if (!pmark_on) return;
        if (pmark_n == pmarks.size()) {
            pmarks.emplace_back(name, nullptr);
            CK(cudaEventCreate(&pmarks.back().second));
        }
        pmarks[pmark_n].first = name;
        static const bool sync = [] { const char* v = std::getenv("STRATA_Q36_PROF"); return v && v[0] == '2'; }();
        if (sync) CK(cudaStreamSynchronize(st));   // STRATA_Q36_PROF=2: exact per-stage times (the host waits each time)
        CK(cudaEventRecord(pmarks[pmark_n].second, st));
        if (pmark_t.size() <= pmark_n) pmark_t.resize(pmark_n + 1);
        pmark_t[pmark_n] = Clock::now();
        ++pmark_n;
    }
    void stream_kick(int i) {   // the i-th layer of the sequence into staging buffer i % 2, after its last reader
        const int b = i % 2;
        const Layer& L = layer_at(stream_seq[(size_t) i]);
        CK(cudaStreamWaitEvent(cst, ev_used[b], 0));
        const auto kick_t0 = Clock::now();
        if (prof_on()) {
            if (prof_n == prof_ev.size()) {
                prof_ev.emplace_back();
                CK(cudaEventCreate(&prof_ev.back().first));
                CK(cudaEventCreate(&prof_ev.back().second));
            }
            CK(cudaEventRecord(prof_ev[prof_n].first, cst));
        }
        // the experts with a VRAM slot come from it (device to device, at VRAM speed), only the others over PCIe; runs
        // of consecutive experts from the same source in one copy each.  STRATA_Q36_STREAM_ALL=1: the whole layer from
        // host memory (the first version: PCIe-bound)
        static const bool all = std::getenv("STRATA_Q36_STREAM_ALL") != nullptr;
        const int ne = (int) m->g.n_expert;
        const Hy& H = hy[(size_t) stream_seq[(size_t) i]];
        // fewer than half the experts resident (a 12 GB card): the one whole-layer copy is faster than many small ones
        // (RTX 3060, UD-Q4_K_XL at 37 of 256: pp2048 1970 whole vs 1540 in runs)
        if (streamer) {   // a pageable arena: the same copy list, run by the copier thread through the pinned ring
            std::vector<ExpertStreamer::Seg> segs;
            if (all || 2 * L.n_slots < ne) {
                segs.push_back({stage[b], L.hblobs, L.stride * (size_t) ne, false});
            } else {
                for (int x = 0; x < ne;) {
                    const bool dev = H.slot_of[(size_t) x] >= 0;
                    int y = x + 1;
                    if (dev)
                        while (y < ne && H.slot_of[(size_t) y] == H.slot_of[(size_t) y - 1] + 1) ++y;
                    else
                        while (y < ne && H.slot_of[(size_t) y] < 0) ++y;
                    const size_t n = (size_t) (y - x) * L.stride;
                    segs.push_back({stage[b] + (size_t) x * L.stride,
                                    dev ? L.blobs + (size_t) H.slot_of[(size_t) x] * L.stride : L.hblobs + (size_t) x * L.stride,
                                    n, dev});
                    x = y;
                }
            }
            kick_job[b] = streamer->submit(std::move(segs), ev_used[b], ev_copy[b]);
            if (prof_on()) prof_host_ms += ms_since(kick_t0);
            return;
        }
        if (all || 2 * L.n_slots < ne) {
            CK(cudaMemcpyAsync(stage[b], L.hblobs, L.stride * (size_t) ne, cudaMemcpyHostToDevice, cst));
        } else {
            for (int x = 0; x < ne;) {
                const bool dev = H.slot_of[(size_t) x] >= 0;
                int y = x + 1;
                if (dev)   // consecutive slots holding consecutive experts
                    while (y < ne && H.slot_of[(size_t) y] == H.slot_of[(size_t) y - 1] + 1) ++y;
                else
                    while (y < ne && H.slot_of[(size_t) y] < 0) ++y;
                const size_t n = (size_t) (y - x) * L.stride;
                if (dev)
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.blobs + (size_t) H.slot_of[(size_t) x] * L.stride,
                                       n, cudaMemcpyDeviceToDevice, cst));
                else
                    CK(cudaMemcpyAsync(stage[b] + (size_t) x * L.stride, L.hblobs + (size_t) x * L.stride, n,
                                       cudaMemcpyHostToDevice, cst));
                x = y;
            }
        }
        if (prof_on()) {
            CK(cudaEventRecord(prof_ev[prof_n++].second, cst));
            prof_host_ms += ms_since(kick_t0);
        }
        CK(cudaEventRecord(ev_copy[b], cst));
    }
    const uint8_t* stream_take(int li) {
        if (stream_pos >= (int) stream_seq.size() || stream_seq[(size_t) stream_pos] != li) {
            std::fprintf(stderr, "strata-q36: the expert stream is out of step (layer %d)\n", li);
            std::exit(1);
        }
        const int b = stream_pos % 2;
        if (streamer) {
            const auto w0 = Clock::now();
            streamer->wait(kick_job[b]);   // its copies are queued and ev_copy[b] recorded
            if (prof_on()) prof_wait_ms += ms_since(w0);
        }
        CK(cudaStreamWaitEvent(st, ev_copy[b], 0));
        return stage[b];
    }
    void stream_done() {
        const int b = stream_pos % 2;
        CK(cudaEventRecord(ev_used[b], st));
        if (stream_pos + 2 < (int) stream_seq.size()) stream_kick(stream_pos + 2);
        ++stream_pos;
    }

    /// The GPU half of a hybrid MoE: the routing to the host, the resident experts' entries grouped and run.
    void moe_hybrid_gpu(const Layer& L, size_t li, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int64_t n_ent = (int64_t) T * Kp;
        if (ho_active(li, T, cur_graph_mode)) {   // the routing to mapped memory and its flag (see k_ho_publish)
            k_ho_publish<<<1, 1024, 0, st>>>(ids, (int) n_ent, h, T * N, ho_ids_d[li], ho_h_d[li], ho_rflag_d + li, ho_seq + 1);
        } else {
            CK(cudaMemcpyAsync(h_ids, ids, (size_t) n_ent * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaMemcpyAsync(h_h, h, (size_t) T * N * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaEventRecord(ev_route, st));
        }
        group_entries(L, hy[li].slot_of_d, T);
        if (!xq_post) quant(h, N, T, xq);   // the experts' and the shared expert's input
        K::native_expert_grouped(L.L, grp_ptr, grp_start, n_groups, ent_dst, ent_tok,
                                 std::min<int64_t>(n_ent, g.n_expert), n_ent, xq, escr, eparts, st);
    }
    /// The CPU half: the entries without a slot on the pool (ggml-cpu's arithmetic for the layer's formats), their
    /// rows uploaded and merged into the entries' rows.
    void moe_hybrid_cpu(size_t li, int T) {
#if defined(STRATA_Q36_CPU_POOL)
        namespace C = strata::kernels::cpu;
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        if (ho_active(li, T, cur_graph_mode)) {   // the GPU waits for this layer's flag; the host fills it in ho_run()
            const unsigned seq = ++ho_seq;
            const size_t slot = li * (size_t) kHoRows * Kp;
            k_ho_wait<<<1, 32, 0, st>>>(ho_flag_d + li, seq);
            k_merge_host<<<n_ent, kThreads, 0, st>>>(eparts, ho_out_d + slot * N, ho_miss_d + slot, N);
            ho_pending.push_back({(int) li, T, seq});
            return;
        }
        const auto tw = Clock::now();
        while (cudaEventQuery(ev_route) == cudaErrorNotReady) {
        }
        hy_wait_ms += ms_since(tw);
        const int n_miss = cpu_part(li, T, h_ids, h_h, cpu_out, h_miss);
        if (n_miss == 0) return;
        CK(cudaMemcpyAsync(ecpu, cpu_out, (size_t) n_ent * N * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_miss, h_miss, (size_t) n_ent * 4, cudaMemcpyHostToDevice, st));
        k_merge_cpu<<<n_ent, kThreads, 0, st>>>(eparts, ecpu, d_miss, N);
#else
        (void) li;
        (void) T;
#endif
    }
    /// The handoff mode's host half: each pending layer's CPU experts in order, its rows and miss flags written to
    /// mapped memory, then its flag (the GPU's wait kernel lets the merge run). Called right after a small forward is
    /// enqueued, before anything waits for the stream.
    void ho_run() {
#if defined(STRATA_Q36_CPU_POOL)
        if (ho_pending.empty()) return;
        const int Kp = (int) m->top_k, N = (int) m->g.n_embd;
        (void) cudaStreamQuery(st);   // the queued work to the GPU now (the HIP runtime batches launches)
        for (const HoJob& j : ho_pending) {
            const auto tw = Clock::now();
            while (*(volatile unsigned*) (ho_rflag_h + j.li) != j.seq) {
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            hy_wait_ms += ms_since(tw);
            const size_t slot = (size_t) j.li * kHoRows * Kp;
            cpu_part((size_t) j.li, j.T, ho_ids[(size_t) j.li], ho_h[(size_t) j.li], ho_out_h + slot * N, ho_miss_h + slot);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile unsigned*) (ho_flag_h + j.li) = j.seq;
        }
        ho_pending.clear();
#endif
    }
    /// The entries without a slot of layer li on the pool: ids / hh the routing's copies on the host, the rows into
    /// out (entry e at e * N) and miss[e] set for each; returns how many there were.
    int cpu_part(size_t li, int T, const int32_t* h_ids, const float* h_h, float* cpu_out, int32_t* h_miss) {
#if defined(STRATA_Q36_CPU_POOL)
        namespace C = strata::kernels::cpu;
        const auto& g = m->g;
        const int N = (int) g.n_embd, Kp = (int) m->top_k;
        const int n_ent = T * Kp;
        Hy& H = hy[li];
        const Layer& L = layer_at((int) li);
        const auto t0 = Clock::now();
        // the missed entries per expert (up to MAXT tokens per job)
        jobs.clear();
        std::vector<char> quantized((size_t) T, 0);
        int n_miss = 0;
        std::map<int, size_t> open;   // expert -> its job with room
        const bool defer = rows_deferred && li < m->layers.size();
        if (defer) {
            if (row_ids.size() < m->layers.size()) row_ids.resize(m->layers.size());
            row_ids[li].assign(h_ids, h_ids + n_ent);
        }
        for (int e = 0; e < n_ent; ++e) {
            const int x = h_ids[e];
            if (!defer) H.score[(size_t) x] += 1.0;
            h_miss[e] = H.slot_of[(size_t) x] < 0;
            if (!h_miss[e]) continue;
            ++n_miss;
            const int t = e / Kp;
            if (!quantized[(size_t) t]) {
                C::native_quant_act(H.fmt, h_h + (size_t) t * N, acts + (size_t) t * C::kNativeActBytes);
                quantized[(size_t) t] = 1;
            }
            auto it = open.find(x);
            if (it == open.end() || jobs[it->second].nt == C::MAXT) {
                C::ExpertJobMulti j;
                j.blob = L.hblobs + (size_t) x * L.stride;
                jobs.push_back(j);
                it = open.insert_or_assign(x, jobs.size() - 1).first;
            }
            C::ExpertJobMulti& j = jobs[it->second];
            j.nact[j.nt] = acts + (size_t) t * C::kNativeActBytes;
            j.out[j.nt] = cpu_out + (size_t) e * N;
            ++j.nt;
        }
        hy_cpu += n_miss;
        hy_gpu += n_ent - n_miss;
        if (n_miss == 0) return 0;
        pool->run_split_multi_native(H.fmt, jobs.data(), (int) jobs.size());
        hy_cpu_ms += ms_since(t0);
        return n_miss;
#else
        (void) li; (void) T; (void) h_ids; (void) h_h; (void) cpu_out; (void) h_miss;
        return 0;
#endif
    }

    /// The slots to the experts the routing used most (the counts halved each time): at most max_swaps replacements
    /// over all layers, between forwards (the copies are on the engine's stream).
    /// The slots to the experts the routing used most (the counts halved each time): per layer at most max_swaps
    /// evictions, empty slots filled freely, between forwards (the copies are on the engine's stream). The extra slots
    /// (staging VRAM) become usable again here after a streamed prompt dropped them.
    /// with_extras: the extra slots (staging and chunk-buffer VRAM) are taken back into use here; only the decode-side
    /// rebalances do that, so that a long prompt read in several calls does not refill them between its chunks.
    void rebalance(int max_swaps, bool with_extras = true) {
        static const bool off = std::getenv("STRATA_Q36_FIXED_SLOTS") != nullptr;   // the slots as placed (tests)
        if (!m->hybrid || off || warming) return;
        const int64_t ne = m->g.n_expert;
        if (!xs_active && with_extras) {
            for (Hy& H : hy) H.slots.insert(H.slots.end(), H.xslots.begin(), H.xslots.end());
            xs_active = true;
        }
        for (size_t li = 0; li < hy.size(); ++li) {
            Hy& H = hy[li];
            const Layer& L = layer_at((int) li);
            const int S = (int) H.slots.size();
            if (S <= 0 || S >= ne) {
                for (double& v : H.score) v *= 0.5;
                continue;
            }
            std::vector<int> order((size_t) ne);
            for (int x = 0; x < ne; ++x) order[(size_t) x] = x;
            std::sort(order.begin(), order.end(), [&](int a, int b) { return H.score[(size_t) a] > H.score[(size_t) b]; });
            // the slots to give away: empty ones first, then the residents with the lowest scores
            std::vector<int> victims = H.slots;
            auto sc = [&](int slot) {
                const int y = H.exp_in_slot[(size_t) slot];
                return y < 0 ? -1.0 : H.score[(size_t) y];
            };
            std::stable_sort(victims.begin(), victims.end(), [&](int a, int b) { return sc(a) < sc(b); });
            size_t vi = 0;
            int left = max_swaps;
            bool changed = false;
            for (int i = 0; i < S && vi < victims.size(); ++i) {
                const int x = order[(size_t) i];
                if (H.slot_of[(size_t) x] >= 0) continue;
                const int s_ = victims[vi];
                const int y = H.exp_in_slot[(size_t) s_];
                if (y >= 0) {
                    if (H.score[(size_t) y] >= H.score[(size_t) x] || left <= 0) break;
                    --left;
                    H.slot_of[(size_t) y] = -1;
                }
                ++vi;
                H.slot_of[(size_t) x] = s_;
                H.exp_in_slot[(size_t) s_] = x;
                CK(cudaMemcpyAsync(L.blobs + (size_t) s_ * L.stride, L.hblobs + (size_t) x * L.stride, L.L.bytes,
                                   cudaMemcpyHostToDevice, st));
                ++hy_swaps;
                changed = true;
            }
            if (changed)
                CK(cudaMemcpyAsync(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice, st));
            for (double& v : H.score) v *= 0.5;
        }
        CK(cudaStreamSynchronize(st));   // slot_of's host copy is pageable: the upload is done before it changes
    }
    /// A streamed prompt forward overwrites the staging buffers: the extra slots there give their experts up first
    /// (those experts are read from host memory again until rebalance() refills them).
    void drop_xslots() {
        if (!xs_active) return;
        CK(cudaStreamSynchronize(st));
        const int64_t ne = m->g.n_expert;
        for (Hy& H : hy) {
            if (H.xslots.empty()) continue;
            for (int s_ : H.xslots) {
                const int y = H.exp_in_slot[(size_t) s_];
                if (y >= 0) H.slot_of[(size_t) y] = -1;
                H.exp_in_slot[(size_t) s_] = -1;
            }
            H.slots.resize((size_t) H.n_main);
            CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
        }
        xs_active = false;
    }

    void shrink_att(size_t floats) {
        CK(cudaDeviceSynchronize());
        CK(cudaFree(att_s));
        CK(cudaFree(att_p16));
        kAttScoreFloats = floats;
        att_s = dalloc(kAttScoreFloats);
        att_p16 = dalloc_t<__half>(kAttScoreFloats);
        CK(cudaDeviceSynchronize());
    }
    void reset_graphs() {
        for (GSet& gset : gsets) {
            for (cudaGraphExec_t* g : {&gset.g1, &gset.gm1})
                if (*g) { CK(cudaGraphExecDestroy(*g)); *g = nullptr; }
            for (auto& g : gset.gv)
                if (g) { CK(cudaGraphExecDestroy(g)); g = nullptr; }
        }
        for (auto& kv : row_graphs)
            if (kv.second) CK(cudaGraphExecDestroy(kv.second));
        row_graphs.clear();
    }

    template <class F> void graph_run(cudaGraphExec_t& gx, F body) {
        if (!use_graph) { body(); return; }
        if (!gx) {
            cudaGraph_t graph = nullptr;
            CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
            body();
            CK(cudaStreamEndCapture(st, &graph));
            CK(cudaGraphInstantiate(&gx, graph, 0));
            CK(cudaGraphDestroy(graph));
        }
        CK(cudaGraphLaunch(gx, st));
    }

    // ---- pictures (--vision).  A request with pictures hands the engine its rows (the encoder's, in prompt order), the
    // cell -> row map of its <|image_pad|> cells and the M-RoPE table of its prompt cells; the cells below the prompt
    // length take their rotary positions from the table, the text after it continues at cell + delta (what the pictures
    // did not use of their cells: llama.cpp's mtmd advances the text position by max(nx, ny) per picture).
    bool img_active = false;
    void set_images(const std::vector<float>& rows, const std::vector<int32_t>& cell_row, const std::vector<int32_t>& mrope,
                    int64_t n_prompt, int32_t delta) {
        const size_t need = rows.size();
        if (need > img_cap) {
            CK(cudaDeviceSynchronize());
            if (d_img) CK(cudaFree(d_img));
            CK(cudaMalloc((void**) &d_img, need * 4));
            img_cap = need;
        }
        CK(cudaMemcpy(d_img, rows.data(), need * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(d_mtab, mrope.data(), mrope.size() * 4, cudaMemcpyHostToDevice));
        img_map = cell_row;
        img_prompt_cells = n_prompt;
        img_active = true;
        set_delta(delta);
    }
    void clear_images() {
        img_map.clear();
        img_active = false;
        img_prompt_cells = 0;
        mtab_now = nullptr;
        map_on = false;
        set_delta(0);
    }
    /// A forward over the cells [c0, c0 + T): the picture rows of its cells, and the table when it is a prompt's.
    void chunk_pictures(int64_t c0, int T) {
        mtab_now = nullptr;
        map_on = false;
        if (!img_active) return;
        if (c0 < img_prompt_cells) mtab_now = d_mtab;
        bool any = false;
        for (int j = 0; j < T; ++j) {
            const int64_t c = c0 + j;
            const int32_t v = c < (int64_t) img_map.size() ? img_map[(size_t) c] : -1;
            h_map[j] = v;
            any = any || v >= 0;
        }
        if (any) {
            CK(cudaMemcpyAsync(d_map, h_map, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            map_on = true;
        }
    }

    /// The tokens at the current position: a prompt in chunks (eager), one token from the decode graph.  The last
    /// token's logits stay in `logits` (its greedy pick in `pick`), hlast its normed hidden.  With the MTP block the
    /// prompt also runs through it (its K / V for every position: row q reads h_{q-1} and token q).  on_chunk(done) runs
    /// after every chunk; false ends the read there (a cancelled request): the tokens processed are returned.
    template <class F> int64_t process(const int32_t* toks, int64_t n, F on_chunk_) {
        const int N = (int) m->g.n_embd;
        auto on_chunk = [&](int64_t done) -> bool {   // a callback returning void never cancels
            if constexpr (std::is_void_v<decltype(on_chunk_(done))>) {
                on_chunk_(done);
                return true;
            } else {
                return on_chunk_(done);
            }
        };
        if (n == 1) {
            CK(cudaStreamSynchronize(st));   // the pinned staging is free again
            hpin[0] = toks[0];
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            chunk_pictures(pos, 1);
            rows_deferred = true;
            // a prompt cell of a picture request (its rotary position is in the table, its row a picture's) runs eagerly
            if (mtab_now || map_on) forward(1, pos, 1, false, false);
            else graph_run(gs().g1, [&] { forward(1, pos, 1, use_graph, false); });
            mtab_now = nullptr;
            map_on = false;
            rows_deferred = false;
            CK(cudaMemcpyAsync(hlast, hmain, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += 1;
            after_rows(1);
            on_chunk(1);
            return 1;
        }
        for (int64_t i = 0; i < n;) {
            // layer-major: the full chunks in super-chunks (no pictures; a partial last chunk takes the path below)
            if (lm_on && !img_active && m->hybrid && n - i >= 2 * (int64_t) tmax) {
                // whole chunks; the prompt's last super-chunk also takes a partial last chunk above hybrid_t rows
                const int64_t rem = n - i;
                int ns = (int) std::min<int64_t>((int64_t) lm_chunks * tmax, rem / tmax * tmax);
                const int64_t tail = rem - ns;
                if (tail > hybrid_t && tail < tmax && ns + tail <= (int64_t) lm_chunks * tmax) ns += (int) tail;
                forward_lm(ns, pos, toks + i, i + ns == n);
                if (m->has_mtp && mtp_d > 0) forward_mtp_lm(ns, pos);
                CK(cudaMemcpyAsync(hlast, lm_H + (size_t) (ns - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
                pos += ns;
                i += ns;
                CK(cudaStreamSynchronize(st));
                if (!on_chunk(i)) {
                    if (m->hybrid) { rebalance(hybrid_swaps_after_prompt, false); prompt_decay(n); }
                    return i;
                }
                continue;
            }
            const int T = (int) std::min<int64_t>(tmax, n - i);
            CK(cudaStreamSynchronize(st));
            hpin[1] = (int32_t) pos;
            CK(cudaMemcpyAsync(tok, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            chunk_pictures(pos, T);
            forward(T, pos, i + T == n ? 1 : 0, false, false);
            if (m->has_mtp && mtp_d > 0) {
                // seeds: h_{q-1} for each row q: hlast, then this chunk's rows 0 .. T - 2 (in `y`, free here)
                CK(cudaMemcpyAsync(mtp_seed(), hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
                if (T > 1)
                    CK(cudaMemcpyAsync(mtp_seed() + N, hmain, (size_t) (T - 1) * N * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(tok_m, toks + i, (size_t) T * 4, cudaMemcpyHostToDevice, st));
                CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
                forward_mtp(T, pos, mtp_seed(), false, false);
            }
            CK(cudaMemcpyAsync(hlast, hmain + (size_t) (T - 1) * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            pos += T;
            CK(cudaStreamSynchronize(st));
            mtab_now = nullptr;
            map_on = false;
            if (!on_chunk(i + T)) {
                if (m->hybrid) { rebalance(hybrid_swaps_after_prompt, false); prompt_decay(n); }
                return i + T;
            }
            i += T;
        }
        if (m->hybrid) { rebalance(hybrid_swaps_after_prompt, false); prompt_decay(n); }
        return n;
    }
    /// STRATA_Q36_PROMPT_DECAY=1 (opt-in): after a prompt of more than a few hundred tokens, its routing counts are
    /// scaled down to about one rebalance period's worth, and the first decode rebalance may swap 16 experts a layer:
    /// the slots then follow what the answer uses within ~32 tokens instead of keeping the prompt's experts for hundreds
    /// (after a 24K-token prompt 32% of decode's entries hit a slot, against 52% on a short question). Which experts sit
    /// in VRAM changes only where an expert is computed, not what is computed.
    int decode_swaps_once = 0;
    void prompt_decay(int64_t n) {
        static const bool on = [] { const char* v = std::getenv("STRATA_Q36_PROMPT_DECAY"); return v && v[0] == '1'; }();
        if (!on || n < 256) return;
        const double want = (double) rebal_every() * (double) m->top_k;
        for (Hy& H : hy) {
            double sum = 0.0;
            for (double v : H.score) sum += v;
            if (sum > want)
                for (double& v : H.score) v *= want / sum;
        }
        decode_swaps_once = 16;
    }

    // ---- batch rows: one decode window for the next token of several sessions
    struct Row { int slot; int32_t tok; };
    long long win_count = 0;
    std::map<int, cudaGraphExec_t> row_graphs;   // by (rows, groups, kind): graphs of the batch windows
    /// A slot's rows of one window: `n` consecutive cells from the session's position (1: the next token; 2: also an MTP
    /// proposal the window verifies).
    struct Group { int slot; int n; int32_t tok[2]; };
    int32_t rows_pick[kVerify] = {};
    int rows_G_last = 0;   // groups of the window run last (commit_rows takes their keep counts)

    void run_graph_key(int key, const std::function<void()>& body) {
        if (!use_graph) { body(); return; }
        graph_run(row_graphs[key], body);
    }

    /// One window over the groups' rows.  `verify`: a verify window - the GDN state is read, not written (commit_rows writes
    /// it for the kept rows) and the sessions' positions stay.  The greedy picks end up in rows_pick (one per row), every
    /// row's logits in `logits`.  Without verify the sessions' positions advance by their rows.
    void run_rows(const Group* gr, int G, bool verify) {
        sess[(size_t) cur].pos = pos;
        CK(cudaStreamSynchronize(st));   // the pinned staging is free again
        int64_t pmax = 0;
        int T = 0;
        for (int g = 0; g < G; ++g) {
            const Sess& ss = sess[(size_t) gr[g].slot];
            hrow[40 + g] = gr[g].slot;
            hrow[48 + g] = T;
            hrow[56 + g] = gr[g].n;
            for (int j = 0; j < gr[g].n; ++j) {
                hrow[T] = gr[g].tok[j];
                hrow[8 + T] = (int32_t) (ss.pos + j);
                hrow[16 + T] = gr[g].slot;
                hrow[24 + T] = ss.rdelta;
                ++T;
            }
            pmax = std::max<int64_t>(pmax, ss.pos + gr[g].n - 1);
        }
        CK(cudaMemcpyAsync(tok, hrow, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rcell, hrow + 8, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rslot, hrow + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rdelta, hrow + 24, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gslot, hrow + 40, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_grow0, hrow + 48, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_gcnt, hrow + 56, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = G;
        rows_deferred = true;
        run_graph_key((verify ? 1000 : 0) + G * 10 + (T - G), [&] { forward(T, pmax, T, use_graph, verify); });
        rows_deferred = false;
        rows_mode = false;
        CK(cudaMemcpyAsync(hrow + 32, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int r = 0; r < T; ++r) rows_pick[r] = hrow[32 + r];
        if (!verify)
            for (int g = 0; g < G; ++g) sess[(size_t) gr[g].slot].pos += gr[g].n;
        pos = sess[(size_t) cur].pos;
        rows_G_last = G;
        if (m->hybrid) {
            count_rows(T);
            if (++win_count % rebal_every() == 0) rebalance(rebal_swaps());
        }
    }
    /// One window for the next token of each of the rows' sessions (no proposals): the greedy picks in rows_pick.
    void step_rows(const Row* rows, int nb) {
        Group gr[kVerify];
        for (int r = 0; r < nb; ++r) gr[r] = {rows[r].slot, 1, {rows[r].tok, 0}};
        run_rows(gr, nb, false);
    }
    /// After run_rows(.., verify = true): the GDN state of group g takes its first keep[g] rows (the conv history and the
    /// recurrence run again over the rows' kept inputs, as commit() does) and the session advances by keep[g].
    void commit_rows(const Group* gr, const int* keep, int G) {
        const auto& g = m->g;
        const int HK = (int) g.ssm_k_heads, HV = (int) g.ssm_v_heads, C = (int) g.ssm_conv_channels;
        sess[(size_t) cur].pos = pos;
        for (int i = 0; i < G; ++i) hrow[56 + i] = keep[i];   // (hrow's slot / first-row words are the window's, unchanged)
        CK(cudaMemcpyAsync(d_gcnt, hrow + 56, (size_t) G * 4, cudaMemcpyHostToDevice, st));
        rows_mode = true;
        rows_G = G;
        run_graph_key(2000 + G, [&] {
            for (size_t l = 0; l < m->layers.size(); ++l) {
                const Layer& L = m->layers[l];
                if (L.attn) continue;
                k_conv_seq<<<dim3(C / 128, G), 128, 0, st>>>(nullptr, v_qkv[l], L.conv, hc, C, 2 * HK, m->eps, 1, true,
                                                             d_conv_tab + l * n_sess, d_gslot, d_grow0, d_gcnt);
                k_gdn_seq<<<dim3(HV, G), dim3(128, 4), 0, st>>>(nullptr, v_hc[l], C, v_alpha[l], v_beta[l], z, L.ssm_norm,
                                                                m->eps, nullptr, HK, HV, 1, true, d_S_tab + l * n_sess,
                                                                d_gslot, d_grow0, d_gcnt);
            }
        });
        rows_mode = false;
        CK(cudaStreamSynchronize(st));
        for (int i = 0; i < G; ++i) sess[(size_t) gr[i].slot].pos += keep[i];
        pos = sess[(size_t) cur].pos;
    }

    /// The MTP block over rows of several sessions: row r reads token tok at cell `cell` of session `slot` and the seed
    /// hidden at `seed` (device, n_embd floats); its draft - the block's greedy pick after that row - ends up in
    /// mtp_pick[r], and the block's K / V cells are written.
    struct MtpRow { int slot; int32_t tok; int64_t cell; const float* seed; };
    int32_t mtp_pick[kVerify] = {};
    float* logits_mr = nullptr;
    int32_t* pick_mr = nullptr;
    void mtp_rows(const MtpRow* rows, int T) {
        const auto& g = m->g;
        const int N = (int) g.n_embd;
        if (!logits_mr) {
            logits_mr = dalloc((size_t) kVerify * g.n_vocab);
            pick_mr = dalloc_t<int32_t>(kVerify);
        }
        sess[(size_t) cur].pos = pos;
        CK(cudaStreamSynchronize(st));
        int64_t pmax = 0;
        for (int r = 0; r < T; ++r) {
            hrow[r] = rows[r].tok;
            hrow[8 + r] = (int32_t) rows[r].cell;
            hrow[16 + r] = rows[r].slot;
            hrow[24 + r] = sess[(size_t) rows[r].slot].rdelta;
            pmax = std::max<int64_t>(pmax, rows[r].cell);
            CK(cudaMemcpyAsync(mtp_seed() + (size_t) r * N, rows[r].seed, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        }
        CK(cudaMemcpyAsync(tok_m, hrow, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rcell, hrow + 8, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rslot, hrow + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(d_rdelta, hrow + 24, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        mtab_now = nullptr;
        map_on = false;
        rows_mode = true;
        rows_G = T;
        rows_deferred = true;
        run_graph_key(3000 + T, [&] {
            K::iq_embed_rows(m->embd_type, m->embd, m->embd_row, tok_m, T, N, y, st);
            k_rms_ld<<<T, kThreads, 0, st>>>(y, N, m->mtp_enorm, mtp_cat, 2 * N, N, m->eps);                 // [e | ...]
            k_rms_ld<<<T, kThreads, 0, st>>>(mtp_seed(), N, m->mtp_hnorm, mtp_cat + N, 2 * N, N, m->eps);   // [... | h]
            mm(m->eh_proj, mtp_cat, x, T);
            k_rms<<<T, kThreads, 0, st>>>(x, m->mtp.attn_norm, h, N, m->eps);
            layer(m->mtp, m->layers.size(), T, pmax, pos0_m, use_graph, false, m->mtp_shnorm, hmtp);
            mm(m->head, hmtp, logits_mr, T);
            k_argmax<<<dim3(kArgBlocks, T), kThreads, 0, st>>>(logits_mr, (int) g.n_vocab, pick_mr, amv, ami, false);
            k_argmax<<<dim3(1, T), kThreads, 0, st>>>(logits_mr, kArgBlocks, pick_mr, amv, ami, true);
        });
        rows_deferred = false;
        rows_mode = false;
        CK(cudaMemcpyAsync(hrow + 32, pick_mr, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        for (int r = 0; r < T; ++r) mtp_pick[r] = hrow[32 + r];
    }
    /// Row r's logits of the last step_rows, on the host.
    void host_logits_row(int r, std::vector<float>& out) {
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits + (size_t) r * m->g.n_vocab, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }

    /// Session dst takes session src's state: the recurrent state, the cache's first src.pos cells, the position, the
    /// pictures' offset (for a conversation read in one session that continues in another).
    void copy_session(int src, int dst) {
        if (src == dst) return;
        sess[(size_t) cur].pos = pos;
        const auto& g = m->g;
        const Sess& a = sess[(size_t) src];
        Sess& b = sess[(size_t) dst];
        const size_t C = (size_t) g.ssm_conv_channels;
        // per KV head: its cells' values a row of ctx cells; an 8-bit cache's scales after all values, rows likewise
        const size_t es = kvq8() ? 1 : kv16() ? 2 : 4;
        const size_t row = (size_t) ctx * g.head_dim * es;
        const size_t nb = kvq8() ? (size_t) g.head_dim / kv_qb() : 0;
        const size_t srow = (size_t) ctx * nb * 2, sbase = (size_t) g.n_head_kv * row;
        for (size_t l = 0; l < a.conv.size(); ++l) {
            if (a.conv[l]) {
                CK(cudaMemcpyAsync(b.conv[l], a.conv[l], C * (g.ssm_d_conv - 1) * 4, cudaMemcpyDeviceToDevice, st));
                CK(cudaMemcpyAsync(b.S[l], a.S[l], (size_t) g.ssm_v_heads * g.ssm_state_size * g.ssm_state_size * 4,
                                   cudaMemcpyDeviceToDevice, st));
            }
            if (a.kc[l] && a.pos > 0) {
                for (float* const* pp : {&a.kc[l], &a.vc[l]}) {
                    const char* src = (const char*) *pp;
                    char* dst = (char*) (pp == &a.kc[l] ? b.kc[l] : b.vc[l]);
                    CK(cudaMemcpy2DAsync(dst, row, src, row, (size_t) a.pos * g.head_dim * es, (size_t) g.n_head_kv,
                                         cudaMemcpyDeviceToDevice, st));
                    if (nb)
                        CK(cudaMemcpy2DAsync(dst + sbase, srow, src + sbase, srow, (size_t) a.pos * nb * 2,
                                             (size_t) g.n_head_kv, cudaMemcpyDeviceToDevice, st));
                }
            }
        }
        CK(cudaMemcpyAsync(b.hlast, a.hlast, (size_t) g.n_embd * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaStreamSynchronize(st));
        b.pos = a.pos;
        b.rdelta = a.rdelta;
        b.live = a.live;
        b.imgs = a.imgs;
        if (dst == cur) load_cur();
    }
    int hybrid_swaps_after_prompt = 2048;
    /// the MTP seed rows of a prompt chunk: a buffer the block's own forward does not use (`moe_out`, [tmax][N])
    float* mtp_seed() { return moe_out; }

    /// One speculative step with D MTP drafts (greedy): `x` is the next token (decided, not processed); `fix` the
    /// tokens of the rows the block must redo first (the drafts the last window accepted, seeded by hmain rows
    /// 0 .. fix.size() - 1 of that window).  Drafts d1 .. dD, then the window [x, d1 .. dD] through the main model;
    /// returns the accepted count a and writes the picks of rows 0 .. a (row a's pick is the token after them).
    /// `cancel`: a stop asked for before the state moves keeps only x (the caller then emits no draft), so that a request
    /// stopped mid-answer leaves the state at the tokens it has shown: a conversation that continues from them (the server's
    /// hand-over of a stopped request to a batch slot) reuses it instead of reading them again.
    /// `sample_row` (sampled requests): the token for verify row r is drawn from that row's logits by the request's
    /// sampler instead of taken greedily; the drafts are accepted while the drawn token equals them and the first
    /// draw that differs is the next token (sample-and-match, as llama.cpp's speculative decoding without draft
    /// probabilities: every token comes from the target model's distribution given the tokens before it).
    int spec_step(int x_tok, const std::vector<int32_t>& fix, int D, std::vector<int32_t>& drafts, int32_t* picks,
                  const std::atomic<bool>* cancel = nullptr, const std::function<int(int)>* sample_row = nullptr) {
        const int N = (int) m->g.n_embd;
        if (main_hybrid) D = std::min<int>(D, (int) (rebal_every() - pos % rebal_every()) - 1);
        const int T0 = (int) fix.size() + 1;   // the redo rows and the row of x
        CK(cudaStreamSynchronize(st));
        // rows of the first MTP call: (hmain row i, fix[i]) at positions pos - fix.size() + i, then (hlast, x) at pos
        for (int i = 0; i < (int) fix.size(); ++i) hpin[16 + i] = fix[(size_t) i];
        hpin[16 + fix.size()] = x_tok;
        hpin[1] = (int32_t) (pos - (int64_t) fix.size());
        CK(cudaMemcpyAsync(tok_m, hpin + 16, (size_t) T0 * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        // seeds: hmain rows 0 .. fix - 1 are the h of the positions before the fixed rows; hlast is that before x
        float* seeds = mtp_seed();
        if (!fix.empty())
            CK(cudaMemcpyAsync(seeds, hmain, fix.size() * (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(seeds + fix.size() * (size_t) N, hlast, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        forward_mtp(T0, pos - (int64_t) fix.size(), seeds, D > 0, false);
        drafts.assign((size_t) D, 0);
        if (D > 0) {
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[0] = hpin[2];
        }
        int last_row = T0 - 1;   // the row of the previous MTP call whose output seeds the next draft
        for (int d = 1; d < D; ++d) {   // chained: (the block's own output, the last draft) at the next position
            hpin[0] = drafts[(size_t) d - 1];
            hpin[1] = (int32_t) (pos + d);
            CK(cudaMemcpyAsync(tok_m, hpin, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(pos0_m, hpin + 1, 4, cudaMemcpyHostToDevice, st));
            CK(cudaMemcpyAsync(seeds, hmtp + (size_t) last_row * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
            last_row = 0;
            if (mtp_graph) graph_run(gs().gm1, [&] { forward_mtp(1, pos + d, seeds, true, use_graph); });
            else forward_mtp(1, pos + d, seeds, true, false);
            CK(cudaMemcpyAsync(hpin + 2, pick_m, 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            drafts[(size_t) d] = hpin[2];
        }
        // the verify window
        const int T = D + 1;
        hpin[16] = x_tok;
        for (int d = 0; d < D; ++d) hpin[17 + d] = drafts[(size_t) d];
        hpin[1] = (int32_t) pos;
        CK(cudaMemcpyAsync(tok, hpin + 16, (size_t) T * 4, cudaMemcpyHostToDevice, st));
        CK(cudaMemcpyAsync(pos0, hpin + 1, 4, cudaMemcpyHostToDevice, st));
        rows_deferred = true;
        graph_run(gs().gv[T], [&] { forward(T, pos, T, use_graph, true); });
        rows_deferred = false;
        int a = 0;
        if (sample_row) {
            int t = (*sample_row)(0);
            while (a < D && t == drafts[(size_t) a]) {
                ++a;
                t = (*sample_row)(a);
            }
            if (cancel != nullptr && cancel->load() && a > 0) {
                a = 0;
                t = (*sample_row)(0);   // only x is kept: row 0's draw is the next token (drawn again: the rng moves on)
            }
            for (int i = 0; i < a; ++i) picks[i] = drafts[(size_t) i];
            picks[a] = t;
        } else {
            CK(cudaMemcpyAsync(hpin + 2, pick, (size_t) T * 4, cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            while (a < D && hpin[2 + a] == drafts[(size_t) a]) ++a;
            if (cancel != nullptr && cancel->load()) a = 0;
            for (int i = 0; i <= a; ++i) picks[i] = hpin[2 + i];
        }
        commit(a + 1);
        CK(cudaMemcpyAsync(hlast, hmain + (size_t) a * N, (size_t) N * 4, cudaMemcpyDeviceToDevice, st));
        pos += a + 1;
        after_rows(a + 1);
        return a;
    }

    /// The first prompt after load ran at about half speed (pp4096 rep 0: ~1770 tok/s against ~3290): the GPU's code
    /// objects, the BLAS handles' kernels and the decode graph are made on first use.  A throw-away prompt of one
    /// chunk, a decode step and (with drafts) a speculative step at load do that before the first request; then the
    /// state, the cache position and the hybrid counters are as after load.
    bool warming = false;
    void warmup() {
        const auto t0 = Clock::now();
        warming = true;
        const int32_t filler = 198;   // a newline: any token works
        // #4708: the prompt chunk may equal or exceed the context (--context 2048 with a 2048-token chunk): warm only cells that
        // exist, and start the small-prompt, decode and speculative probes in a fresh sequence instead of appending past the cache.
        std::vector<int32_t> toks((size_t) std::max(1, std::min(tmax, ctx)), filler);
        process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
        reset();
        if (tmax > kGemmT + 1 && ctx > kGemmT + 1) process(toks.data(), kGemmT + 1, [](int64_t) {});   // the smallest GEMM chunk too
        process(&filler, 1, [](int64_t) {});
        if (mtp_d > 0) {
            std::vector<int32_t> fix, drafts;
            int32_t picks[16];
            spec_step(filler, fix, mtp_d, drafts, picks);
        }
        CK(cudaStreamSynchronize(st));
        warming = false;
        for (Hy& H : hy) std::fill(H.score.begin(), H.score.end(), 0.0);
        hy_cpu = hy_gpu = hy_swaps = 0;
        hy_cpu_ms = 0;
        hy_wait_ms = fwd_small_ms = 0;
        fwd_small_n = 0;
        reset();
        std::fprintf(stderr, "strata-q36: warmed up in %.1f s (one prompt chunk, a decode step%s)\n",
                     ms_since(t0) / 1000.0, mtp_d > 0 ? ", a speculative step" : "");
    }

    int greedy() {
        CK(cudaMemcpyAsync(hpin + 2, pick, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        return hpin[2];
    }
    void host_logits(std::vector<float>& out) {
        out.resize((size_t) m->g.n_vocab);
        CK(cudaMemcpyAsync(out.data(), logits, out.size() * 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
    }
};

// ------------------------------------------------------------------------------------------------ expert placement
/// The routed experts after the engine's buffers: all in VRAM when they fit in `vram_gib` (< 0: the free VRAM less a
/// reserve), else the hybrid mode - every expert in host memory, the same number of VRAM slots per layer, a decode
/// batch's other experts on the CPU pool and a prompt chunk's streamed through two staging buffers.
// ------------------------------------------------------------------------------------------------ the blob cache
// The hybrid mode keeps every expert's blob ([gate | up | down], `stride` apart, kMmqTail after each layer) in host
// memory. When the driver will not pin that much (an 8 GB card with 32 GB of RAM: 18.7 GiB for UD-Q4_K_M), the
// fallback was a private malloc: 18.7 GiB of committed memory that only the pagefile can take back, which left
// ~0 MB free on 32 GB. Instead the same bytes are written once to a cache file beside the model and mapped read-only:
// the pages are file-backed, so Windows (or Linux) can move cold experts to the standby list / page cache and read
// them back from the SSD, like llama.cpp's mmap of the GGUF. The engine only ever reads the arena (slot copies, the
// prompt's staging copies, the CPU pool), so the bytes and every output are the same as with the malloc'd copy.
// STRATA_Q36_BLOB_CACHE=<path> picks the file, =0 keeps the old malloc. Default: <model>.q36blobs.
namespace blobcache {
constexpr size_t kHeader = 4096;   // the key, then zeros; the arena starts page-aligned after it

struct Map {
    const uint8_t* base = nullptr;   // the arena (after the header)
    size_t size = 0;
};

static std::string file_key(const std::string& model, size_t host, int64_t ne, const std::vector<std::pair<size_t, size_t>>& layers) {
    std::string k = "strata q36 blob cache v1\nmodel " + model + "\n";
#ifdef _WIN32
    WIN32_FILE_ATTRIBUTE_DATA a{};
    if (GetFileAttributesExA(model.c_str(), GetFileExInfoStandard, &a)) {
        const uint64_t sz = ((uint64_t) a.nFileSizeHigh << 32) | a.nFileSizeLow;
        const uint64_t mt = ((uint64_t) a.ftLastWriteTime.dwHighDateTime << 32) | a.ftLastWriteTime.dwLowDateTime;
        k += "size " + std::to_string(sz) + " mtime " + std::to_string(mt) + "\n";
    }
#else
    struct stat st{};
    if (stat(model.c_str(), &st) == 0) k += "size " + std::to_string((uint64_t) st.st_size) + " mtime " + std::to_string((uint64_t) st.st_mtime) + "\n";
#endif
    k += "arena " + std::to_string(host) + " experts " + std::to_string(ne) + " tail " + std::to_string(kMmqTail) + "\n";
    for (const auto& [stride, bytes] : layers) k += std::to_string(stride) + "/" + std::to_string(bytes) + " ";
    k += "\n";
    return k;
}

static bool header_matches(const std::string& path, const std::string& key, size_t total) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::string h(kHeader, '\0');
    const bool ok = std::fread(h.data(), 1, kHeader, f) == kHeader;
#ifdef _WIN32
    _fseeki64(f, 0, SEEK_END);
    const int64_t sz = _ftelli64(f);
#else
    fseeko(f, 0, SEEK_END);
    const int64_t sz = (int64_t) ftello(f);
#endif
    std::fclose(f);
    return ok && sz == (int64_t) total && h.compare(0, key.size(), key) == 0 && h[key.size()] == '\0';
}

// write the arena the malloc path would have built, in the same order, to path (via a temporary file)
static bool build(const std::string& path, const std::string& key, const std::vector<Layer*>& layers, int64_t ne,
                  std::string& err) {
    const std::string tmp = path + ".part";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) {
        err = "cannot create " + tmp;
        return false;
    }
    std::vector<uint8_t> buf;
    buf.reserve((size_t) 64 << 20);
    auto flush = [&]() -> bool {
        const bool ok = buf.empty() || std::fwrite(buf.data(), 1, buf.size(), f) == buf.size();
        buf.clear();
        return ok;
    };
    std::vector<uint8_t> head(kHeader, 0);
    std::memcpy(head.data(), key.data(), key.size());
    bool ok = std::fwrite(head.data(), 1, kHeader, f) == kHeader;
    for (const Layer* L : layers) {
        std::vector<uint8_t> blob(L->stride, 0);
        for (int64_t e = 0; e < ne && ok; ++e) {
            std::fill(blob.begin(), blob.end(), 0);
            expert_blob(*L, e, blob.data());
            buf.insert(buf.end(), blob.begin(), blob.end());
            if (buf.size() >= ((size_t) 64 << 20)) ok = flush();
        }
        buf.insert(buf.end(), kMmqTail, 0);
    }
    ok = ok && flush();
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) {
        std::remove(tmp.c_str());
        err = "writing " + tmp + " failed (disk full?)";
        return false;
    }
    std::remove(path.c_str());
    if (std::rename(tmp.c_str(), path.c_str()) != 0) {
        err = "cannot rename " + tmp + " to " + path;
        return false;
    }
    return true;
}

static bool map(const std::string& path, size_t total, Map& out, std::string& err) {
#ifdef _WIN32
    HANDLE h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                           FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        err = "cannot open " + path;
        return false;
    }
    HANDLE mh = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(h);   // the mapping keeps the file open
    if (!mh) {
        err = "CreateFileMapping failed for " + path;
        return false;
    }
    const uint8_t* base = (const uint8_t*) MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mh);  // the view keeps the mapping alive for the process's lifetime
    if (!base) {
        err = "MapViewOfFile failed for " + path;
        return false;
    }
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        err = "cannot open " + path;
        return false;
    }
    void* p = mmap(nullptr, total, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        err = "mmap failed for " + path;
        return false;
    }
    const uint8_t* base = (const uint8_t*) p;
#endif
    out.base = base + kHeader;
    out.size = total - kHeader;
    return true;
}

// the arena for `layers` (each L->stride * ne + kMmqTail), mapped from the cache file; false + why to keep the malloc
static bool arena(const std::string& model, const std::vector<Layer*>& layers, int64_t ne, size_t host, Map& out,
                  std::string& why) {
    const char* env = std::getenv("STRATA_Q36_BLOB_CACHE");
    if (env && std::strcmp(env, "0") == 0) {
        why = "STRATA_Q36_BLOB_CACHE=0";
        return false;
    }
    const std::string path = env && *env ? std::string(env) : model + ".q36blobs";
    std::vector<std::pair<size_t, size_t>> geo;
    for (const Layer* L : layers) geo.emplace_back(L->stride, L->L.bytes);
    const std::string key = file_key(model, host, ne, geo);
    if (key.size() >= kHeader) {
        why = "cache key too long";
        return false;
    }
    const size_t total = kHeader + host;
    if (!header_matches(path, key, total)) {
        const auto t0 = Clock::now();
        std::fprintf(stderr, "strata-q36: writing the expert blob cache %s (%.2f GiB, once per model file)...\n",
                     path.c_str(), (double) host / (1 << 30));
        if (!build(path, key, layers, ne, why)) return false;
        std::fprintf(stderr, "strata-q36: expert blob cache written in %.1f s\n", ms_since(t0) / 1000.0);
    }
    if (!map(path, total, out, why)) return false;
    std::fprintf(stderr, "strata-q36: experts mapped from %s (file-backed: cold pages can leave RAM and come back from "
                 "the SSD)\n", path.c_str());
    return true;
}
}  // namespace blobcache

bool place_experts(Model& m, Engine& E, double vram_gib, int threads, int hybrid_t, int slots, std::string& err) {
    const auto t0 = Clock::now();
    const auto& g = m.g;
    const int64_t ne = g.n_expert;
    std::vector<Layer*> all;
    for (Layer& L : m.layers) all.push_back(&L);
    if (m.has_mtp) all.push_back(&m.mtp);
    size_t total = 0, per_slot = 0, max_layer = 0;
    for (Layer* L : all) {
        total += L->stride * (size_t) ne + kMmqTail;
        per_slot += L->stride;
        max_layer = std::max(max_layer, L->stride * (size_t) ne);
    }
    size_t fr = 0, tot = 0;
    CK(cudaMemGetInfo(&fr, &tot));
    const size_t reserve = (size_t) 768 << 20;
    const size_t budget = vram_gib >= 0 ? (size_t) (vram_gib * (double) (1ull << 30))
                                        : (fr > reserve ? fr - reserve : 0);
    // every expert fits once the packed Q8_0 decode copies are given back (a file a few GiB larger than the card's
    // room, UD-Q6_K_XL on 32 GB): drop them - the decode reads the GGUF layout then, bitwise the same outputs, a little
    // slower - rather than run the hybrid mode with its CPU share and streamed prompt layers.  STRATA_Q36_KEEP_PACKED=1
    // keeps them (and the hybrid mode).
    size_t packed_bytes = 0;
    for (const auto& pk : m.packed) packed_bytes += std::get<2>(pk);
    size_t main_need = 0;   // the main layers' experts (the MTP block's can go to the CPU, see below)
    for (Layer& L : m.layers) main_need += L.stride * (size_t) ne + kMmqTail;
    const size_t slack = ((size_t) 512 << 20) + (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // reserve + scores
    const bool mtp_cpu_ok = m.has_mtp && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr;
    if (slots <= 0 && vram_gib < 0 && total > budget && packed_bytes > 0 &&
        (total <= budget + packed_bytes || (mtp_cpu_ok && main_need <= budget + packed_bytes + slack)) &&
        std::getenv("STRATA_Q36_KEEP_PACKED") == nullptr) {
        CK(cudaDeviceSynchronize());
        for (const auto& pk : m.packed) {
            K::native_q8_0_packed_unregister(std::get<0>(pk));
            CK(cudaFree(std::get<1>(pk)));
        }
        m.packed.clear();
        E.reset_graphs();
        CK(cudaMemGetInfo(&fr, &tot));
        std::fprintf(stderr, "strata-q36: the packed Q8_0 decode copies (%.2f GiB) given back so that every routed "
                     "expert fits in VRAM (STRATA_Q36_KEEP_PACKED=1 keeps them and runs the hybrid mode)\n",
                     (double) packed_bytes / (1 << 30));
    }
    size_t budget2 = vram_gib >= 0 ? budget : (fr > reserve ? fr - reserve : 0);
    // the MAIN layers' experts fit, the MTP block's not all (UD-Q6_K_XL on 32 GB): the main layers all in VRAM - the
    // text is the all-in-VRAM text, the decode graph stays - and the MTP block's experts on slots + the CPU pool (its
    // drafts only).  When that needs it: the attention scores buffer at 64 MB instead of 256 MB (more, smaller
    // slices in a long prompt) and the reserve at 256 MiB instead of 768 (said in the log; --expert-vram sets the
    // budget instead).  STRATA_Q36_NO_MTP_CPU=1: the plain hybrid mode for every layer.
    size_t total_main = 0;
    for (Layer& L : m.layers) total_main += L.stride * (size_t) ne + kMmqTail;
    bool mtp_only = false;
    if (slots <= 0 && vram_gib < 0 && m.has_mtp && total > budget2 && std::getenv("STRATA_Q36_NO_MTP_CPU") == nullptr) {
        const size_t att_save = (E.kAttScoreFloats - ((size_t) 16 << 20)) * 6;   // att_s (4 B) + att_p16 (2 B)
        const size_t small_reserve = (size_t) 256 << 20;
        const size_t room = fr + att_save > small_reserve ? fr + att_save - small_reserve : 0;
        if (total_main + ((size_t) 64 << 20) <= budget2) {
            mtp_only = true;
        } else if (total_main + ((size_t) 64 << 20) <= room) {
            E.shrink_att((size_t) 16 << 20);
            CK(cudaMemGetInfo(&fr, &tot));
            budget2 = fr > small_reserve ? fr - small_reserve : 0;
            mtp_only = total_main + ((size_t) 64 << 20) <= budget2;
            std::fprintf(stderr, "strata-q36: the experts nearly fit: the attention scores buffer at 64 MB and the VRAM "
                         "reserve at 256 MiB (instead of 768) so that every main layer's experts fit; a driver or "
                         "another program needing more VRAM later can make it fail - --expert-vram GiB sets the experts' "
                         "budget instead\n");
        }
    }
    if (slots <= 0 && total <= budget2 && total + ((size_t) 256 << 20) <= fr) {
        std::vector<uint8_t> blob;
        for (Layer* L : all) {
            CK(cudaMalloc(&L->blobs, L->stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L->blobs, 0, L->stride * (size_t) ne + kMmqTail));
            blob.assign(L->L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(*L, e, blob.data());
                CK(cudaMemcpy(L->blobs + (size_t) e * L->stride, blob.data(), L->L.bytes, cudaMemcpyHostToDevice));
            }
            L->n_slots = (int) ne;
        }
        std::fprintf(stderr, "strata-q36: every routed expert in VRAM (%.2f GiB) in %.1f s\n", (double) total / (1 << 30),
                     ms_since(t0) / 1000.0);
        return true;
    }
#if !defined(STRATA_Q36_CPU_POOL)
    (void) threads;
    (void) hybrid_t;
    err = "the routed experts (" + std::to_string(total >> 20) + " MiB) do not fit in VRAM, and this build has no CPU "
          "expert pool (STRATA_NATIVE_EXPERTS=OFF)";
    return false;
#else
    namespace C = strata::kernels::cpu;
    m.hybrid = true;
    E.main_hybrid = !mtp_only;
    if (E.main_hybrid) E.use_graph = false;   // the host takes part in every layer
    E.mtp_graph = false;                      // the MTP block's layer is hybrid either way
    E.hybrid_t = std::max(1, std::min(hybrid_t, E.use_mmq ? E.tmax : E.tmax));
    if (!E.use_mmq) E.hybrid_t = E.tmax;   // no streamed prompt path: every batch on slots + the pool
    const size_t staging = E.use_mmq && E.main_hybrid ? 2 * (max_layer + kMmqTail) : 0;
    // the prompt-only VRAM: the staging pair and the chunk-sized buffer set (alloc_big, after the slots)
    const size_t big_bytes = E.split_bufs ? E.set_floats(E.big_T) * 4 + ((size_t) 1 << 20) : 0;
    const size_t prompt_only = staging + big_bytes;
    size_t slot_bytes = budget2 > prompt_only ? budget2 - prompt_only : 0;
    if (slot_bytes + prompt_only + ((size_t) 256 << 20) > fr)
        slot_bytes = fr > prompt_only + ((size_t) 512 << 20) ? fr - prompt_only - ((size_t) 512 << 20) : 0;
    // --expert-slots: that many per layer (tests: the same split with and without the MTP block)
    int S = slots > 0 ? (int) std::min<int64_t>(ne, slots) : (int) std::min<size_t>((size_t) ne, slot_bytes / per_slot);
    // the extra slots (staging) are addressed from a layer's own slots: at least one each when there is staging
    if (S == 0 && staging && slots <= 0 && !mtp_only && fr > per_slot + staging + ((size_t) 256 << 20)) {
        const char* xv = std::getenv("STRATA_Q36_XSLOTS");
        if (!(xv && xv[0] == '0')) S = 1;
    }
    if (mtp_only) {   // the main layers all, the MTP block what is left
        const size_t left = budget2 > total_main ? budget2 - total_main : 0;
        S = (int) std::min<size_t>((size_t) ne, left / m.mtp.stride);
    }
    {   // what the hybrid mode allocates next (the staging pair, the CPU part's outputs, the slots) against the VRAM the
        // sessions and the context left: said with the numbers instead of a cudaMalloc failure (8 slots on a 12 GB card)
        const size_t Tn = (size_t) E.tmax, Nn = (size_t) g.n_embd, Kn = (size_t) m.top_k;
        size_t need = prompt_only + Tn * Kn * Nn * 4 + Tn * Kn * 4;
        for (Layer* L : all) {
            if (mtp_only && L != &m.mtp) need += L->stride * (size_t) ne + kMmqTail;
            else need += (S > 0 ? (size_t) S * L->stride + kMmqTail : 0);
        }
        if (need > fr) {
            err = "not enough VRAM left for the hybrid experts' buffers and slots (" + std::to_string((need >> 20)) +
                  " MiB needed, " + std::to_string(fr >> 20) + " MiB free after the weights, the context and the sessions): use "
                  "fewer slots (\"parallel\" / --batch), a shorter --max-context" +
                  (slots > 0 ? ", or fewer --expert-slots" : "");
            return false;
        }
    }
    // host memory: every expert, pinned where the driver allows it (the prompt's stream and the slot copies)
    size_t host = 0;
    for (Layer* L : all)
        if (!mtp_only || L == &m.mtp) host += L->stride * (size_t) ne + kMmqTail;
    uint8_t* arena = nullptr;
    bool pinned = cudaHostAlloc((void**) &arena, host, cudaHostAllocDefault) == cudaSuccess;
    bool mapped = false;   // the arena is the read-only blob cache file (filled already)
    if (!pinned) {
        (void) cudaGetLastError();
        std::vector<Layer*> hl;
        for (Layer* L : all)
            if (!mtp_only || L == &m.mtp) hl.push_back(L);
        blobcache::Map bm;
        std::string why;
        if (blobcache::arena(m.path, hl, ne, host, bm, why)) {
            arena = const_cast<uint8_t*>(bm.base);   // never written below: `mapped` skips the fill
            mapped = true;
        } else {
            std::fprintf(stderr, "strata-q36: no expert blob cache (%s): a private copy in RAM\n", why.c_str());
            arena = (uint8_t*) std::malloc(host);
        }
        if (!arena) {
            err = "no host memory for the experts (" + std::to_string(host >> 20) + " MiB)";
            return false;
        }
    }
    size_t off = 0;
    E.hy.resize(all.size());
    for (size_t li = 0; li < all.size(); ++li) {
        Layer& L = *all[li];
        if (mtp_only && &L != &m.mtp) {   // a main layer: every expert in VRAM, as in the all-in-VRAM placement
            CK(cudaMalloc(&L.blobs, L.stride * (size_t) ne + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) ne + kMmqTail));
            std::vector<uint8_t> blob(L.L.bytes, 0);
            for (int64_t e = 0; e < ne; ++e) {
                expert_blob(L, e, blob.data());
                CK(cudaMemcpy(L.blobs + (size_t) e * L.stride, blob.data(), L.L.bytes, cudaMemcpyHostToDevice));
            }
            L.n_slots = (int) ne;
            continue;
        }
        L.hblobs = arena + off;
        off += L.stride * (size_t) ne + kMmqTail;
        if (!mapped) {
            std::memset(L.hblobs, 0, L.stride * (size_t) ne + kMmqTail);
            for (int64_t e = 0; e < ne; ++e) expert_blob(L, e, L.hblobs + (size_t) e * L.stride);
        }
        L.n_slots = S;
        if (S > 0) {
            CK(cudaMalloc(&L.blobs, L.stride * (size_t) S + kMmqTail));
            CK(cudaMemset(L.blobs, 0, L.stride * (size_t) S + kMmqTail));
            CK(cudaMemcpy(L.blobs, L.hblobs, L.stride * (size_t) S, cudaMemcpyHostToDevice));
        }
        Engine::Hy& H = E.hy[li];
        H.slot_of.assign((size_t) ne, -1);
        H.exp_in_slot.assign((size_t) S, -1);
        H.n_main = S;
        H.slots.clear();
        for (int s = 0; s < S; ++s) {
            H.slot_of[(size_t) s] = s;
            H.exp_in_slot[(size_t) s] = s;
            H.slots.push_back(s);
        }
        H.score.assign((size_t) ne, 0.0);
        CK(cudaMalloc(&H.slot_of_d, (size_t) ne * 4));
        CK(cudaMemcpy(H.slot_of_d, H.slot_of.data(), (size_t) ne * 4, cudaMemcpyHostToDevice));
        if (!C::native_fmt(L.L.gu_type, L.L.d_type, g.n_embd, g.n_ff, H.fmt, err)) return false;
        if (H.fmt.up_off != L.L.up_off || H.fmt.down_off != L.L.down_off || H.fmt.bytes != L.L.bytes) {
            err = "the CPU and GPU expert layouts differ";
            return false;
        }
    }
    const size_t T = (size_t) E.tmax, N = (size_t) g.n_embd, Kp = (size_t) m.top_k;
    CK(cudaHostAlloc((void**) &E.h_h, T * N * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.cpu_out, T * Kp * N * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_ids, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_miss, T * Kp * 4, cudaHostAllocDefault));
    CK(cudaHostAlloc((void**) &E.h_cnt, all.size() * (size_t) ne * 4, cudaHostAllocDefault));
    CK(cudaMalloc(&E.ecpu, T * Kp * N * 4));
    CK(cudaMalloc(&E.d_miss, T * Kp * 4));
    E.acts = (uint8_t*) std::malloc(T * C::kNativeActBytes + 64);
    CK(cudaEventCreateWithFlags(&E.ev_route, cudaEventDisableTiming));
    if (const char* hv = std::getenv("STRATA_Q36_HANDOFF"); hv && hv[0] == '1') {   // see Engine::ho
        const size_t ns = all.size(), R = (size_t) Engine::kHoRows;
        E.ho_ids.assign(ns, nullptr);
        E.ho_h.assign(ns, nullptr);
        E.ho_ids_d.assign(ns, nullptr);
        E.ho_h_d.assign(ns, nullptr);
        for (size_t i = 0; i < ns; ++i) {
            CK(cudaHostAlloc((void**) &E.ho_ids[i], R * Kp * 4, cudaHostAllocMapped));
            CK(cudaHostGetDevicePointer((void**) &E.ho_ids_d[i], E.ho_ids[i], 0));
            CK(cudaHostAlloc((void**) &E.ho_h[i], R * N * 4, cudaHostAllocMapped));
            CK(cudaHostGetDevicePointer((void**) &E.ho_h_d[i], E.ho_h[i], 0));
        }
        CK(cudaHostAlloc((void**) &E.ho_rflag_h, ns * 4, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void**) &E.ho_rflag_d, E.ho_rflag_h, 0));
        std::memset(E.ho_rflag_h, 0, ns * 4);
        CK(cudaHostAlloc((void**) &E.ho_out_h, ns * R * Kp * N * 4, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void**) &E.ho_out_d, E.ho_out_h, 0));
        CK(cudaHostAlloc((void**) &E.ho_miss_h, ns * R * Kp * 4, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void**) &E.ho_miss_d, E.ho_miss_h, 0));
        CK(cudaHostAlloc((void**) &E.ho_flag_h, ns * 4, cudaHostAllocMapped));
        CK(cudaHostGetDevicePointer((void**) &E.ho_flag_d, E.ho_flag_h, 0));
        std::memset(E.ho_flag_h, 0, ns * 4);
        E.ho = true;
        std::fprintf(stderr, "strata-q36: STRATA_Q36_HANDOFF=1: small forwards enqueued whole, each hybrid layer's GPU part "
                     "waits for its CPU rows in mapped memory (%.1f MiB)\n", (double) (ns * R * Kp * (N + 1) * 4) / (1 << 20));
    }
    E.host_pinned = pinned;
    if (staging) {
        CK(cudaStreamCreateWithFlags(&E.cst, cudaStreamNonBlocking));
        const char* rv = std::getenv("STRATA_Q36_RING");
        if (!pinned && !(rv && rv[0] == '0')) {
            const char* mv = std::getenv("STRATA_Q36_RING_MIB");
            const char* tv = std::getenv("STRATA_Q36_RING_THREADS");
            const size_t mib = mv ? (size_t) std::max(1, std::atoi(mv)) : 16;
            const int thr = tv ? std::max(1, std::atoi(tv)) : 3;
            const char* sv = std::getenv("STRATA_Q36_RING_STREAMS");
            const int nstr = sv ? std::max(1, std::atoi(sv)) : 1;   // two streams measured slower (pp4096 467 vs 515)
            E.streamer = std::make_unique<ExpertStreamer>(E.cst, mib << 20, 6, thr, nstr);
            std::fprintf(stderr, "strata-q36: prompt expert stream through a pinned ring (6 x %zu MiB, %d copy threads)\n",
                         mib, thr);
        }
        for (int b = 0; b < 2; ++b) {
            CK(cudaMalloc(&E.stage[b], max_layer + kMmqTail));
            CK(cudaMemset(E.stage[b], 0, max_layer + kMmqTail));
            CK(cudaEventCreateWithFlags(&E.ev_copy[b], cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&E.ev_used[b], cudaEventDisableTiming));
            CK(cudaEventRecord(E.ev_used[b], E.st));
        }
        // the extra slots: between streamed prompts the two staging buffers (a whole layer's experts each) hold
        // decode experts too. A slot's address is L.blobs + id * stride in every kernel, so each extra slot is put at
        // a stride-aligned offset from its layer's slots (id > 0 needs staging above them); they are dropped before
        // a streamed forward (drop_xslots) and refilled by rebalance(). STRATA_Q36_XSLOTS=0 leaves them out.
        E.alloc_big();   // the chunk-sized buffer set, above the slots: the third region of extra slots
        const char* xv = std::getenv("STRATA_Q36_XSLOTS");
        if (!(xv && xv[0] == '0') && !mtp_only) {
            std::vector<std::pair<uint8_t*, uint8_t*>> regs = {{E.stage[0], E.stage[0] + max_layer + kMmqTail},
                                                              {E.stage[1], E.stage[1] + max_layer + kMmqTail}};
            if (E.split_bufs && E.big_set.base)
                regs.push_back({(uint8_t*) E.big_set.base, (uint8_t*) (E.big_set.base + E.big_set.floats)});
            size_t total = 0;
            for (const auto& rg : regs) total += (size_t) (rg.second - rg.first);
            size_t sum = 0;
            for (Layer* L : all)
                if (L->blobs) sum += L->stride;
            const int nreg = (int) regs.size();
            const int per = sum ? (int) std::max<int64_t>(0, (int64_t) total / (int64_t) sum - nreg) : 0;
            int r = 0, placed = 0, layers_with = 0;
            uint8_t* cur = regs[0].first;
            for (size_t li = 0; li < all.size() && per > 0 && r < nreg; ++li) {
                Layer& L = *all[li];
                Engine::Hy& H = E.hy[li];
                if (!L.blobs) continue;
                int got = 0;
                while (got < per && r < nreg) {
                    uint8_t* end = regs[(size_t) r].second;
                    if (cur <= L.blobs) {   // a region below this layer's slots: no positive id there
                        if (++r < nreg) cur = regs[(size_t) r].first;
                        continue;
                    }
                    uint8_t* a = cur;
                    const size_t rem = (size_t) (a - L.blobs) % L.stride;
                    if (rem) a += L.stride - rem;
                    if (a + L.stride + kMmqTail > end) {   // the next region
                        if (++r < nreg) cur = regs[(size_t) r].first;
                        continue;
                    }
                    const size_t id = (size_t) (a - L.blobs) / L.stride;
                    if (id > (size_t) 0x7fffffff) break;
                    H.xslots.push_back((int) id);
                    if (H.exp_in_slot.size() <= id) H.exp_in_slot.resize(id + 1, -1);
                    cur = a + L.stride;
                    ++got;
                }
                placed += got;
                layers_with += got > 0;
            }
            std::fprintf(stderr, "strata-q36: %d extra decode slots in the staging buffers%s (%d per layer on %d layers), "
                         "dropped while a prompt streams\n", placed, nreg > 2 ? " and the prompt-chunk buffers" : "", per,
                         layers_with);
        }
    }
    E.pool = std::make_unique<C::ExpertPool>(threads);
    CK(cudaDeviceSynchronize());
    if (mtp_only)
        std::fprintf(stderr, "strata-q36: every main layer's routed experts in VRAM (%.2f GiB); the MTP block's: %d of "
                     "%lld in VRAM, the rest on the CPU pool (its drafts only)\n", (double) total_main / (1 << 30), S,
                     (long long) ne);
    else
    std::fprintf(stderr, "strata-q36: hybrid experts: %d of %lld per layer in VRAM (%.2f GiB), all %.2f GiB in %s host "
                 "memory, the rest of a batch of <= %d rows on %d CPU threads, bigger prompt chunks streamed (%.2f GiB "
                 "staging); %.1f s\n", S, (long long) ne, (double) S * per_slot / (1 << 30), (double) host / (1 << 30),
                 pinned ? "pinned" : mapped ? "file-mapped" : "pageable", E.hybrid_t, E.pool->workers() + (E.pool->host_works() ? 1 : 0),
                 (double) staging / (1 << 30), ms_since(t0) / 1000.0);
    return true;
#endif
}

// ------------------------------------------------------------------------------------------------ sampling
using strata::qwen35::Sampling;

int sample(std::vector<float>& lg, const Sampling& s, const std::vector<int32_t>& hist, std::mt19937_64& rng) {
    strata::qwen35::apply_penalties(lg, s, hist);
    if (s.temperature <= 0.f) return (int) (std::max_element(lg.begin(), lg.end()) - lg.begin());
    const int kk = std::max(1, std::min(s.top_k > 0 ? s.top_k : 64, (int) lg.size()));
    std::vector<int> idx(lg.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::partial_sort(idx.begin(), idx.begin() + kk, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
    idx.resize((size_t) kk);
    std::vector<double> p(idx.size());
    const double mx = lg[(size_t) idx[0]];
    double sum = 0;
    for (size_t i = 0; i < idx.size(); ++i) sum += (p[i] = std::exp((lg[(size_t) idx[i]] - mx) / s.temperature));
    size_t keep = idx.size();
    double cum = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
        if (p[i] / sum < s.min_p * (p[0] / sum)) { keep = i; break; }
        cum += p[i] / sum;
        if (cum >= s.top_p) { keep = i + 1; break; }
    }
    keep = std::max<size_t>(1, keep);
    double tot = 0;
    for (size_t i = 0; i < keep; ++i) tot += p[i];
    std::uniform_real_distribution<double> u(0.0, tot);
    double r = u(rng);
    for (size_t i = 0; i < keep; ++i) {
        if ((r -= p[i]) <= 0) return idx[i];
    }
    return idx[keep - 1];
}

std::vector<int32_t> parse_ids(const char* s) {
    std::vector<int32_t> v;
    while (*s) {
        while (*s == ',' || *s == ' ') ++s;
        if (!*s) break;
        char* e = nullptr;
        const long x = std::strtol(s, &e, 10);
        if (e == s) break;
        v.push_back((int32_t) x);
        s = e;
    }
    return v;
}

bool is_eos(int t) { return t == 248044 || t == 248046; }

// ------------------------------------------------------------------------------------------------ pictures
constexpr int32_t kImagePad = 248056;   // <|image_pad|>

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

/// A request's pictures: the encoder's rows in prompt order, each picture's cell run, and the positions.
struct ImgReq {
    std::vector<float> rows;
    std::vector<ImgKey> imgs;
    std::vector<int32_t> cell_row;   // per prompt cell: its row in `rows`, or -1 (text)
    std::vector<int32_t> mrope;      // [cell][3] t, h, w of every prompt cell (llama.cpp's mtmd positions)
    int32_t delta = 0;               // the text after the prompt: rotary position = cell + delta
};

/// Reads a strata-vision file (one or more records: int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats, in prompt
/// order) and lays the pictures over the prompt's <|image_pad|> runs.  Positions as mtmd's: text counts up by one; a
/// picture's cells take (t, h, w) = (p, p + row, p + column) and the text after it continues at p + max(nx, ny).  "" or
/// the error.  `keys`: the server's key for each picture (img_keys=), else the embeddings' hash.
std::string build_images(const std::string& path, const std::vector<int32_t>& ids, int64_t n_embd,
                         const std::vector<uint64_t>& keys, ImgReq& r) {
    struct Img { int64_t n, nx, ny; size_t off; };
    std::vector<Img> imgs;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::string ve;
    while (ve.empty()) {
        int32_t hdr[5];
        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 || (int64_t) hdr[2] * hdr[3] != hdr[1] ||
            hdr[4] != (int32_t) n_embd) {
            ve = "bad embeddings file (expected strata-vision records of width " + std::to_string((long long) n_embd) + ")";
            break;
        }
        const size_t off = r.rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
        r.rows.resize(off + cnt);
        if (std::fread(r.rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
    }
    std::fclose(f);
    if (!ve.empty()) return ve;
    const int64_t n = (int64_t) ids.size();
    r.cell_row.assign((size_t) n, -1);
    r.mrope.assign((size_t) n * 3, 0);
    auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
        r.mrope[(size_t) c * 3] = (int32_t) t;
        r.mrope[(size_t) c * 3 + 1] = (int32_t) h;
        r.mrope[(size_t) c * 3 + 2] = (int32_t) w;
    };
    int64_t p = 0, i = 0;
    size_t k = 0;
    while (i < n) {
        if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
        if (k >= imgs.size()) return "the prompt has more images than the embeddings file";
        const Img& im = imgs[k++];
        uint64_t h;
        {   // what the conversation cache compares: a picture is its grid and its key (or its embeddings)
            const int64_t grid[3] = {im.n, im.nx, im.ny};
            h = fnv1a(grid, sizeof grid);
            if (keys.size() == imgs.size()) h = fnv1a(&keys[k - 1], sizeof(uint64_t), h);
            else h = fnv1a(r.rows.data() + im.off, (size_t) im.n * (size_t) n_embd * sizeof(float), h);
        }
        r.imgs.push_back({i, im.n, h});
        for (int64_t j = 0; j < im.n; ++j)
            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                return "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                       " rows but fewer <|image_pad|> tokens";
        for (int64_t j = 0; j < im.n; ++j) {
            const int64_t y = j / im.nx, x = j % im.nx;
            put(i + j, p, p + y, p + x);
            r.cell_row[(size_t) (i + j)] = (int32_t) ((im.off / (size_t) n_embd) + (size_t) j);
        }
        i += im.n;
        p += std::max(im.nx, im.ny);
    }
    if (k != imgs.size()) return "the embeddings file has more images than the prompt";
    if (n > 0 && ids[(size_t) (n - 1)] == kImagePad) return "the prompt cannot end in an image";
    r.delta = (int32_t) (p - n);
    return "";
}

// ------------------------------------------------------------------------------------------------ serve
/// A batch slot: a session decoding (its next token is `x`, not yet in the state).
struct Slot {
    bool active = false, stop = false;
    int32_t x = 0;
    int64_t produced = 0, max_new = 0;
    Sampling smp;
    std::mt19937_64 rng;
    std::vector<int32_t> hist;
    Clock::time_point t0;
    int32_t draft = 0;            // --batch-mtp: the MTP block's proposal for the token after `x`
    bool draft_ready = false;
    long long offered = 0, accepted = 0;
};

/// The engine protocol (serve/server.py).  One request at a time on the control lines: GEN / GENI (alone: speculative
/// steps with the MTP block), or with --batch N BGEN / BGENI <slot> (the request's prompt is read into that slot's
/// session - the decode windows of the other slots go on between its chunks - then its tokens come as BT lines, one
/// window for every slot decoding, until BDONE).  BSTOP <slot> ends a slot at its next window; STOP the request on the
/// control lines.
struct Serve {
    Engine& E;
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};
    std::vector<Slot> slots;
    int n_active = 0;
    std::vector<float> lg;
    // decode windows of the other slots between two chunks of a prompt read: STRATA_Q36_READ_WINDOWS (default 1; a chunk
    // takes ~0.15 s at 512 tokens, a window ~0.012 s: more windows give the decoding slots more of the time a read takes)
    int read_windows = 1;

    explicit Serve(Engine& e) : E(e), slots((size_t) e.n_sess) {
        if (const char* v = std::getenv("STRATA_Q36_READ_WINDOWS")) read_windows = std::max(0, std::min(64, std::atoi(v)));
    }

    /// BSTOP / BYIELD lines among the queued ones: applied now (a window is a moment away at most).
    void poll_control() {
        std::lock_guard<std::mutex> lk(mu);
        for (auto it = lines.begin(); it != lines.end();) {
            if (it->rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(it->c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
                it = lines.erase(it);
            } else if (it->rfind("BYIELD", 0) == 0) {   // (a read here does not give way: the server then goes on)
                it = lines.erase(it);
            } else {
                ++it;
            }
        }
    }

    /// One decode window with MTP proposals (--batch-mtp): every active slot's next token, and for up to kVerify - S of the
    /// slots (the greedy ones, in turn) also the proposal its MTP block made: the window verifies both at once.  A slot whose
    /// proposal the main model confirms moves two tokens in the window.  The proposals for the next window come right after
    /// it, from the hidden states the window just produced (the first one of a slot from the state its prompt left).
    int mtp_turn = 0;
    void window_mtp() {
        poll_control();
        const int N = (int) E.m->g.n_embd;
        std::vector<int> act;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) act.push_back(b);
        const int S = (int) act.size();
        if (S == 0) return;
        // who verifies a proposal this window: the greedy slots in turn, as many as fit beside one row per slot
        const int room = Engine::kVerify - S;
        std::vector<char> cand((size_t) S, 0);
        int nc = 0;
        for (int k = 0; k < S && nc < room; ++k) {
            const int i = (mtp_turn + k) % S;
            if (!slots[(size_t) act[(size_t) i]].smp.needs_logits()) { cand[(size_t) i] = 1; ++nc; }
        }
        ++mtp_turn;
        // a candidate without a proposal gets its first one from the state its last row left
        std::vector<Engine::MtpRow> first;
        std::vector<int> first_of;
        for (int i = 0; i < S; ++i) {
            Slot& sl = slots[(size_t) act[(size_t) i]];
            if (cand[(size_t) i] && !sl.draft_ready) {
                Engine::Sess& ss = E.sess[(size_t) act[(size_t) i]];
                first.push_back({act[(size_t) i], sl.x, ss.pos, ss.hlast});
                first_of.push_back(i);
            }
        }
        if (!first.empty()) {
            E.mtp_rows(first.data(), (int) first.size());
            for (size_t k = 0; k < first.size(); ++k) {
                Slot& sl = slots[(size_t) act[(size_t) first_of[k]]];
                sl.draft = E.mtp_pick[k];
                sl.draft_ready = true;
            }
        }
        std::vector<Engine::Group> gr((size_t) S);
        std::vector<int> row0((size_t) S);
        int T = 0;
        for (int i = 0; i < S; ++i) {
            const Slot& sl = slots[(size_t) act[(size_t) i]];
            gr[(size_t) i] = {act[(size_t) i], cand[(size_t) i] ? 2 : 1, {sl.x, sl.draft}};
            row0[(size_t) i] = T;
            T += gr[(size_t) i].n;
        }
        E.run_rows(gr.data(), S, true);
        // the picks: emit, decide how many rows each slot keeps
        std::vector<int> keep((size_t) S, 1);
        std::vector<int32_t> last((size_t) S, 0);
        std::vector<char> alive((size_t) S, 0);
        for (int i = 0; i < S; ++i) {
            const int b = act[(size_t) i];
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) b];
            const int n = gr[(size_t) i].n, r0 = row0[(size_t) i];
            int y0 = E.rows_pick[r0];
            if (sl.smp.needs_logits()) {
                E.host_logits_row(r0, lg);
                y0 = sample(lg, sl.smp, sl.hist, sl.rng);
            }
            bool two = n == 2 && y0 == sl.draft && !is_eos(y0) && !sl.stop && sl.produced + 2 <= sl.max_new &&
                       ss.pos + 3 <= E.ctx;   // (ss.pos: the position of the slot's next token: the window did not move it)
            if (n == 2) { ++sl.offered; if (two) ++sl.accepted; }
            const int rows_kept = two ? 2 : 1;
            keep[(size_t) i] = rows_kept;
            const int32_t fed[2] = {sl.x, sl.draft};
            bool done = false;
            int y = y0;
            for (int j = 0; j < rows_kept; ++j) {
                if (j == 1) y = E.rows_pick[r0 + 1];
                ss.live.push_back(fed[j]);
                std::printf("BT %d %d\n", b, y);
                ++sl.produced;
                sl.hist.push_back(y);
                const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                                  : ss.pos + j + 2 > E.ctx ? "length" : nullptr;
                if (fin) {
                    std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                    if (sl.offered > 0)
                        std::fprintf(stderr, "strata batch: slot %d: %lld of %lld MTP proposals accepted\n", b, sl.accepted,
                                     sl.offered);
                    sl.active = false;
                    ss.active = false;
                    --n_active;
                    done = true;
                    break;
                }
            }
            last[(size_t) i] = y;
            alive[(size_t) i] = !done;
            if (!done) sl.x = y;
        }
        E.commit_rows(gr.data(), keep.data(), S);
        // the hidden state after each slot's last kept row (the seed of a later first proposal), and the next proposals
        std::vector<Engine::MtpRow> nx;
        std::vector<int> nx_of, nx_last;
        for (int i = 0; i < S; ++i) {
            const int b = act[(size_t) i];
            Engine::Sess& ss = E.sess[(size_t) b];
            const int r0 = row0[(size_t) i];
            CK(cudaMemcpyAsync(ss.hlast, E.hmain + (size_t) (r0 + keep[(size_t) i] - 1) * N, (size_t) N * 4,
                               cudaMemcpyDeviceToDevice, E.st));
            Slot& sl = slots[(size_t) b];
            sl.draft_ready = false;
            if (!alive[(size_t) i] || !cand[(size_t) i]) continue;
            const int64_t p_new = ss.pos;   // commit_rows moved it: the cell of the token the next window feeds
            if (keep[(size_t) i] == 2) {
                nx.push_back({b, sl.draft, p_new - 1, E.hmain + (size_t) r0 * N});
                nx.push_back({b, last[(size_t) i], p_new, E.hmain + (size_t) (r0 + 1) * N});
                nx_last.push_back((int) nx.size() - 1);
            } else {
                nx.push_back({b, last[(size_t) i], p_new, E.hmain + (size_t) r0 * N});
                nx_last.push_back((int) nx.size() - 1);
            }
            nx_of.push_back(i);
        }
        std::fflush(stdout);
        if (!nx.empty()) {
            // (more rows than the window holds: the proposals come in two calls)
            for (size_t at = 0, k = 0; at < nx.size();) {
                size_t take = std::min<size_t>(nx.size() - at, Engine::kVerify);
                while (take > 0 && at + take < nx.size() && nx[at + take].slot == nx[at + take - 1].slot) --take;   // a slot's rows together
                E.mtp_rows(nx.data() + at, (int) take);
                for (; k < nx_last.size() && (size_t) nx_last[k] < at + take; ++k) {
                    Slot& sl = slots[(size_t) act[(size_t) nx_of[k]]];
                    sl.draft = E.mtp_pick[(size_t) nx_last[k] - at];
                    sl.draft_ready = true;
                }
                at += take;
            }
        }
    }

    /// One decode window over every active slot: one token each.
    void window() {
        poll_control();
        if (E.bmtp && E.mtp_d > 0) {   // proposals need a row to spare and a slot that decodes greedily
            int act = 0, greedy = 0;
            for (const Slot& sl : slots)
                if (sl.active) { ++act; greedy += !sl.smp.needs_logits(); }
            if (greedy > 0 && Engine::kVerify - act > 0) { window_mtp(); return; }
        }
        window_plain();
    }

    /// One token of every active slot, no proposals.
    void window_plain() {
        std::vector<Engine::Row> rows;
        for (int b = 0; b < E.n_sess; ++b)
            if (slots[(size_t) b].active) rows.push_back({b, slots[(size_t) b].x});
        if (rows.empty()) return;
        E.step_rows(rows.data(), (int) rows.size());
        const size_t Nb = (size_t) E.m->g.n_embd * 4;
        for (size_t r = 0; r < rows.size(); ++r) {
            const int b = rows[r].slot;
            Slot& sl = slots[(size_t) b];
            Engine::Sess& ss = E.sess[(size_t) b];
            if (E.bmtp)   // (the hidden state a later first proposal starts from)
                CK(cudaMemcpyAsync(ss.hlast, E.hmain + r * (size_t) E.m->g.n_embd, Nb, cudaMemcpyDeviceToDevice, E.st));
            sl.draft_ready = false;
            ss.live.push_back(sl.x);
            int y;
            if (sl.smp.needs_logits()) {
                E.host_logits_row((int) r, lg);
                y = sample(lg, sl.smp, sl.hist, sl.rng);
            } else {
                y = E.rows_pick[r];
            }
            std::printf("BT %d %d\n", b, y);
            ++sl.produced;
            sl.hist.push_back(y);
            const char* fin = is_eos(y) ? "stop" : sl.stop ? "cancel" : sl.produced >= sl.max_new ? "length"
                              : ss.pos + 1 > E.ctx ? "length" : nullptr;
            if (fin) {
                std::printf("BDONE %d %lld %s %.1f\n", b, (long long) sl.produced, fin, ms_since(sl.t0));
                sl.active = false;
                ss.active = false;
                --n_active;
            } else {
                sl.x = y;
            }
        }
        std::fflush(stdout);
    }

    /// The length of session ss's held tokens when they are a proper start of `ids` with the same pictures, else 0.
    size_t match_len(const Engine::Sess& ss, const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        const size_t L = ss.live.size();
        if (L == 0 || L >= ids.size() || !std::equal(ss.live.begin(), ss.live.end(), ids.begin())) return 0;
        std::vector<ImgKey> in_live;
        for (const ImgKey& k : imgs) {
            if (k.cell + k.n <= (int64_t) L) in_live.push_back(k);
            else if (k.cell < (int64_t) L) return 0;   // the held tokens end inside a picture
        }
        return in_live == ss.imgs ? L : 0;
    }

    /// The session a request alone takes: the one holding the longest start of its prompt, else an empty one, else the
    /// least recently used (never a decoding slot).
    int pick_session(const std::vector<int32_t>& ids, const std::vector<ImgKey>& imgs) const {
        int best = -1, free_empty = -1, lru = -1;
        size_t bl = 0;
        for (int i = 0; i < E.n_sess; ++i) {
            const Engine::Sess& ss = E.sess[(size_t) i];
            if (ss.active) continue;
            const size_t l = match_len(ss, ids, imgs);
            if (l > bl) { bl = l; best = i; }
            if (ss.live.empty() && free_empty < 0) free_empty = i;
            if (lru < 0 || ss.used < E.sess[(size_t) lru].used) lru = i;
        }
        return best >= 0 ? best : free_empty >= 0 ? free_empty : lru;
    }

    void request(const std::string& line) {
        const bool adm = line[0] == 'B';
        const bool geni = line.rfind(adm ? "BGENI " : "GENI ", 0) == 0;
        if (!adm && !geni && line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...> or GENI <max_new> [keys] <file> <id,id,...>\n");
            std::fflush(stdout);
            return;
        }
        char* p = const_cast<char*>(line.c_str()) + (adm ? (geni ? 6 : 5) : (geni ? 5 : 4));
        int slot = -1;
        if (adm) {
            char* e1 = nullptr;
            const long b = std::strtol(p, &e1, 10);
            p = e1;
            slot = (int) b;
            if (e1 == line.c_str() || b < 0 || b >= (long) slots.size() || slots[(size_t) b].active || E.n_sess < 2) {
                std::printf("ERR BGEN: no such free slot (--batch %d) or a bad max_new\n", E.n_sess < 2 ? 0 : E.n_sess);
                std::fflush(stdout);
                return;
            }
        }
        const long long max_new = std::strtoll(p, &p, 10);
        Sampling smp;
        std::vector<uint64_t> img_keys;
        for (;;) {
            while (*p == ' ') ++p;
            const char* st0 = p;
            while (*p && *p != ' ') ++p;
            const std::string tk(st0, (size_t) (p - st0));
            const size_t eq = tk.find('=');
            if (tk.empty() || eq == std::string::npos) { p = const_cast<char*>(st0); break; }
            const std::string key = tk.substr(0, eq);
            const char* v = tk.c_str() + eq + 1;
            if (key == "temperature") smp.temperature = std::strtof(v, nullptr);
            else if (key == "top_p") smp.top_p = std::strtof(v, nullptr);
            else if (key == "top_k") smp.top_k = std::atoi(v);
            else if (key == "min_p") smp.min_p = std::strtof(v, nullptr);
            else if (key == "penalty_last_n") smp.last_n = std::atoi(v);
            else if (key == "penalty_repeat") smp.repeat = std::strtof(v, nullptr);
            else if (key == "penalty_freq") smp.freq = std::strtof(v, nullptr);
            else if (key == "penalty_present") smp.present = std::strtof(v, nullptr);
            else if (key == "seed") smp.seed = std::strtoull(v, nullptr, 10);
            else if (key == "img_keys") {
                const char* q = v;
                img_keys.clear();
                while (*q) {
                    char* qe = nullptr;
                    const unsigned long long x = std::strtoull(q, &qe, 16);
                    if (qe == q) { img_keys.clear(); break; }
                    img_keys.push_back((uint64_t) x);
                    q = *qe == ',' ? qe + 1 : qe;
                    if (*qe != ',' && *qe != '\0') { img_keys.clear(); break; }
                }
            }
        }
        std::string emb;
        if (geni) {
            while (*p == ' ') ++p;
            const char* e = p;
            while (*e && *e != ' ') ++e;
            emb.assign(p, (size_t) (e - p));
            p = const_cast<char*>(e);
        }
        const std::vector<int32_t> ids = parse_ids(p);
        if (max_new < 1 || ids.empty() || (geni && emb.empty())) {
            std::printf("ERR bad request: max_new or the ids\n");
            std::fflush(stdout);
            return;
        }
        if (geni && !E.vision) {
            std::printf("ERR this engine was started without --vision\n");
            std::fflush(stdout);
            return;
        }
        if ((int64_t) ids.size() + max_new > E.ctx) {
            std::printf("ERR prompt (%zu tokens) + max_new (%lld) exceeds the context (%d)\n", ids.size(), max_new, E.ctx);
            std::fflush(stdout);
            return;
        }
        bool bad = false;
        for (int32_t t : ids) bad |= t < 0 || t >= (int32_t) E.m->g.n_vocab;
        if (bad) { std::printf("ERR a token id is outside the vocabulary\n"); std::fflush(stdout); return; }
        ImgReq ir;
        if (geni) {
            const std::string ve = build_images(emb, ids, E.m->g.n_embd, img_keys, ir);
            if (!ve.empty()) { std::printf("ERR %s\n", ve.c_str()); std::fflush(stdout); return; }
        }
        stop.store(false);
        // ---- the session: the held state continues when the request extends it (the recurrent state cannot step back)
        const int s = adm ? slot : pick_session(ids, ir.imgs);
        Engine::Sess& ss = E.sess[(size_t) s];
        size_t reuse = match_len(ss, ids, ir.imgs);
        if (adm) {   // another session holding more of this prompt (the conversation was read alone): copy its state
            int src = -1;
            size_t best = reuse;
            for (int i = 0; i < E.n_sess; ++i) {
                if (i == s || E.sess[(size_t) i].active) continue;
                const size_t l = match_len(E.sess[(size_t) i], ids, ir.imgs);
                if (l > best) { best = l; src = i; }
            }
            if (src >= 0) {
                E.copy_session(src, s);
                reuse = best;
            }
        }
        E.select(s);
        if (reuse == 0) {
            E.reset();
            ss.live.clear();
            ss.imgs.clear();
        }
        if (geni) E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) ids.size(), ir.delta);
        else E.clear_images();
        const auto t0 = Clock::now();
        const int64_t fresh = (int64_t) (ids.size() - reuse);
        bool cancelled = false;
        const int64_t got = E.process(ids.data() + reuse, fresh, [&](int64_t done) -> bool {
            std::printf("PP %lld %lld %.0f %.1f\n", (long long) done, (long long) fresh, ms_since(t0),
                        (double) done / std::max(1e-9, ms_since(t0) / 1000.0));
            std::fflush(stdout);
            if (n_active > 0 && done < fresh) {
                poll_control();
                // the slots decoding go on between a read's chunks (the last chunk's
                for (int w = 0; w < read_windows && n_active > 0; ++w) window();   // logits must stay for the first token)
            } else {
                poll_control();
            }
            if (stop.load() && done < fresh) { cancelled = true; return false; }
            return true;
        });
        ss.live.insert(ss.live.end(), ids.begin() + (long) reuse, ids.begin() + (long) reuse + (long) got);
        ss.imgs.clear();
        bool cut = false;
        for (const ImgKey& k : ir.imgs) {
            if (k.cell + k.n <= (int64_t) ss.live.size()) ss.imgs.push_back(k);
            else if (k.cell < (int64_t) ss.live.size()) cut = true;
        }
        if (cancelled && cut) {   // stopped inside a picture: the state cannot be continued, the next read starts over
            ss.live.clear();
            ss.imgs.clear();
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        const double prompt_ms = ms_since(t0);
        if (cancelled) {
            std::printf("DONE 0 %zu %.1f 0.0 cancel 0 0 %zu 0 0 0 0 0.0 %lld 0\n", ids.size(), prompt_ms, reuse, (long long) got);
            if (adm) std::printf("BADM %d 0\n", slot);
            std::fflush(stdout);
            return;
        }
        std::mt19937_64 rng(smp.seed);
        // #4590: the penalties count the reply's tokens (and the fed-back last prompt token), not the prompt; STRATA_PENALTY_PROMPT=1
        // restores the old window.  `ids` is the request's full prompt whether or not the session was resumed.
        static const bool pen_prompt = strata::kernels::penalty_counts_prompt(std::getenv("STRATA_PENALTY_PROMPT"));
        std::vector<int32_t> hist = strata::qwen35::penalty_history_start(ids, pen_prompt);
        const auto d0 = Clock::now();
        long long produced = 0, offered = 0, accepted = 0;
        const char* finish = "length";
        // greedy requests alone with the MTP block: speculative steps of mtp_d drafts (verified, so the text is the greedy
        // text); sampled ones, and requests beside others, one token at a time
        // sampled requests too (STRATA_Q36_SPEC_SAMPLED, default 1): each verify row's token drawn by the sampler,
        // the drafts kept while they match (see spec_step)
        static const bool spec_sampled = [] {
            const char* v = std::getenv("STRATA_Q36_SPEC_SAMPLED");
            return !(v && v[0] == '0');
        }();
        const bool spec = E.mtp_d > 0 && (!smp.needs_logits() || spec_sampled) && !adm;
        bool have_nx = false;   // nx is a spec step's draw already (not the logits' next draw)
        std::vector<float> lg_row;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = smp.needs_logits() ? -1 : E.greedy();
        if (adm) {
            if (smp.needs_logits()) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            std::printf("T %d\n", nx);
            ++produced;
            hist.push_back(nx);
            const char* fin = is_eos(nx) ? "stop" : stop.load() ? "cancel" : max_new <= 1 ? "length"
                              : E.pos + 1 > E.ctx ? "length" : nullptr;
            const bool cont = fin == nullptr;
            std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                        ms_since(d0), cont ? "length" : fin, reuse, (long long) fresh);
            if (cont) {
                Slot& sl = slots[(size_t) slot];
                sl = Slot{};
                sl.active = true;
                sl.x = nx;
                sl.produced = 1;
                sl.max_new = max_new;
                sl.smp = smp;
                sl.rng = rng;
                sl.hist = hist;
                sl.t0 = Clock::now();
                ss.active = true;
                ++n_active;
            }
            std::printf("BADM %d %d\n", slot, cont ? 1 : 0);
            std::fflush(stdout);
            std::fprintf(stderr, "strata batch: slot %d takes a prompt of %zu tokens = %zu reused + %lld read in %.0f ms (%.1f "
                         "tok/s)%s\n", slot, ids.size(), reuse, (long long) fresh, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, cont ? "" : ", done at once");
            return;
        }
        for (;;) {
            if (smp.needs_logits() && !have_nx) {
                E.host_logits(lg);
                nx = sample(lg, smp, hist, rng);
            }
            have_nx = false;
            auto emit = [&](int t) {   // false: the answer ends here
                std::printf("T %d\n", t);
                std::fflush(stdout);
                ++produced;
                hist.push_back(t);
                if (is_eos(t)) { finish = "stop"; return false; }
                if (produced >= max_new) return false;
                if (stop.load()) { finish = "cancel"; return false; }
                return true;
            };
            if (!emit(nx)) break;
            if (n_active > 0) window();
            if (spec && n_active == 0 && E.pos + E.mtp_d + 1 <= E.ctx) {
                std::function<int(int)> draw;
                if (smp.needs_logits())
                    draw = [&](int r) {   // row r follows x and drafts 0 .. r - 1: they join the penalty history
                        std::vector<int32_t> h2 = hist;
                        for (int i = 0; i < r; ++i) h2.push_back(drafts[(size_t) i]);
                        E.host_logits_row(r, lg_row);
                        return sample(lg_row, smp, h2, rng);
                    };
                const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks, &stop, smp.needs_logits() ? &draw : nullptr);
                offered += E.mtp_d;
                accepted += a;
                // the state now holds x and every accepted draft, whether or not the answer ends inside them (an end of
                // answer, max_new): the held tokens say so, so a later request that continues this conversation matches them
                // (a prompt that has the same tokens next reuses the state, any other reads from the start) and never reads
                // a token twice
                ss.live.push_back(nx);
                for (int i = 0; i < a; ++i) ss.live.push_back(drafts[(size_t) i]);
                fix.assign(drafts.begin(), drafts.begin() + a);
                bool ended = false;
                for (int i = 0; i < a && !ended; ++i)
                    if (!emit(drafts[(size_t) i])) ended = true;
                if (ended) break;
                nx = picks[a];
                have_nx = smp.needs_logits();
                continue;
            }
            if (E.pos + 1 > E.ctx) break;
            const int32_t one = nx;
            E.process(&one, 1, [](int64_t) {});
            ss.live.push_back(nx);
            fix.clear();
            if (!smp.needs_logits()) nx = E.greedy();
        }
        E.sess[(size_t) E.cur].pos = E.pos;
        const double decode_ms = ms_since(d0);
        std::printf("DONE %lld %zu %.1f %.1f %s %lld %lld %zu 0 0 0 0 0.0 %lld 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish, accepted, offered, reuse, (long long) fresh);
        std::fflush(stdout);
        std::fprintf(stderr, "strata serve: session %d: prompt %zu tokens = %zu reused + %lld read in %.0f ms (%.1f tok/s), "
                     "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld (%s)\n", s, ids.size(), reuse,
                     (long long) fresh, prompt_ms, prompt_ms > 0 ? 1000.0 * (double) fresh / prompt_ms : 0.0, produced,
                     decode_ms, decode_ms > 0 ? 1000.0 * (double) produced / decode_ms : 0.0, accepted, offered, finish);
    }

    int run() {
        if (E.n_sess > 1) std::printf("INFO batch_slots=%d slot_cache=1\n", E.n_sess);
        std::printf("READY %d stop\n", E.ctx);
        std::fflush(stdout);
        std::thread reader([&] {
            std::string l;
            while (std::getline(std::cin, l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                if (l == "STOP") { stop.store(true); continue; }
                std::lock_guard<std::mutex> lk(mu);
                lines.push_back(l);
                cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(mu);
            eof = true;
            cv.notify_one();
        });
        reader.detach();
        for (;;) {
            std::string line;
            bool have = false;
            {
                std::unique_lock<std::mutex> lk(mu);
                if (n_active == 0) cv.wait(lk, [&] { return eof || !lines.empty(); });
                if (!lines.empty()) {
                    line = lines.front();
                    lines.pop_front();
                    have = true;
                } else if (n_active == 0) {
                    return 0;
                }
            }
            if (!have) { window(); continue; }
            if (line.empty()) continue;
            if (line.rfind("BSTOP ", 0) == 0) {
                const int b = std::atoi(line.c_str() + 6);
                if (b >= 0 && b < (int) slots.size()) slots[(size_t) b].stop = true;
            } else if (line.rfind("BYIELD", 0) == 0) {
                // (a read that has ended)
            } else if (line.rfind("GEN", 0) == 0 || line.rfind("BGEN", 0) == 0) {
                request(line);
            } else {
                std::printf("ERR expected: GEN <max_new> [key=value ...] <id,id,...>\n");
                std::fflush(stdout);
            }
            if (n_active > 0) window();
        }
    }
};

int serve(Engine& E) {
    Serve sv(E);
    return sv.run();
}

}  // namespace

int strata::qwen35::run(int argc, char** argv) {
    std::string model, tokens, force, dump_full, emb_file;
    int ctx = 32768, chunk = 0, n_gen = 64, bench_pp = 0, bench_tg = 0, mtp_d = 0, cpu_threads = 0, hybrid_rows = 128;
    double expert_vram = -1.0;
    bool warm = true;
    int expert_slots = 0, batch = 0, draft_vocab = -1;
    bool serve_mode = false, graph = true, vision = false, batch_mtp = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "strata-q36: %s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        // strata's names where they mean the same thing (the family switch passes strata's command line on)
        if (a == "--model" || a == "--native") model = next();
        else if (a == "--max-new") n_gen = std::atoi(next().c_str());
        else if (a == "--no-token-graph") graph = false;
        else if (a == "--pool-workers") cpu_threads = std::atoi(next().c_str());
        else if (a == "--gpu") (void) next();   // applied by strata before the switch (CUDA/HIP_VISIBLE_DEVICES)
        else if (a == "--greedy") {}            // --tokens decodes greedily here anyway
        else if (a == "--serve") serve_mode = true;
        else if (a == "--max-context" || a == "--ctx") ctx = std::atoi(next().c_str());
        else if (a == "--chunk" || a == "--prefill") { const std::string v = next(); chunk = v == "auto" ? 0 : std::atoi(v.c_str()); }
        else if (a == "--tokens") {   // ids, or @file with the ids (long prompts past the command line's limit)
            tokens = next();
            if (!tokens.empty() && tokens[0] == '@') {
                std::ifstream tf(tokens.substr(1));
                tokens.assign(std::istreambuf_iterator<char>(tf), std::istreambuf_iterator<char>());
            }
        }
        else if (a == "--n") n_gen = std::atoi(next().c_str());
        else if (a == "--force") force = next();
        else if (a == "--emb") emb_file = next();   // --tokens with pictures: a strata-vision file (the ids carry <|image_pad|> runs)
        else if (a == "--dump-full-logits") dump_full = next();
        else if (a == "--bench-pp") bench_pp = std::atoi(next().c_str());
        else if (a == "--bench-tg") bench_tg = std::atoi(next().c_str());
        else if (a == "--no-graph") graph = false;
        else if (a == "--no-warmup") warm = false;
        else if (a == "--mtp") {
            // a number of drafts; Flash-Next's "--mtp <draft layer dir>" means nothing for this family
            const std::string v = next();
            if (!v.empty() && std::isdigit((unsigned char) v[0])) mtp_d = std::atoi(v.c_str());
            else std::fprintf(stderr, "strata-q36: ignoring --mtp %s (this family's MTP is the GGUF's own block: --mtp N "
                                      "drafts)\n", v.c_str());
        }
        else if (a == "--draft-vocab") draft_vocab = std::atoi(next().c_str());   // the MTP drafts pick among the first N token ids
        else if (a == "--vision") vision = true;                                  // pictures (GENI): the encoder is strata-vision's
        else if (a == "--batch" || a == "--slots") batch = std::atoi(next().c_str());   // sessions decoding together
        else if (a == "--batch-mtp") batch_mtp = true;                                  // ... each verifying an MTP proposal
        else if (a == "--expert-vram") expert_vram = std::atof(next().c_str());   // GiB for the routed experts
        else if (a == "--cpu-threads") cpu_threads = std::atoi(next().c_str());
        else if (a == "--expert-slots") expert_slots = std::atoi(next().c_str());
        else if (a == "--hybrid-rows") hybrid_rows = std::atoi(next().c_str());
        else if (serve_mode && a.rfind("--", 0) == 0) {
            // serve passes the config's args plus a few of its own: one this engine does not have is said and skipped
            // (with its value), not fatal
            const bool val = i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0;
            std::fprintf(stderr, "strata-q36: ignoring %s%s%s (not an option of this engine)\n", a.c_str(),
                         val ? " " : "", val ? argv[i + 1] : "");
            if (val) ++i;
        } else {
            std::fprintf(stderr,
                         "usage: strata-q36 --model GGUF [--max-context 32768] [--chunk 512] [--no-graph]\n"
                         "       (--serve | --tokens IDS [--n 64] [--force IDS] [--dump-full-logits F] | --bench-pp N "
                         "--bench-tg N)\n");
            return 2;
        }
    }
    if (model.empty()) { std::fprintf(stderr, "strata-q36: --model is required\n"); return 2; }
    auto set_default = [](const char* k, const char* v) {   // unless the variable is set already
        if (std::getenv(k) != nullptr) return;
#if !defined(_WIN32)
        setenv(k, v, 0);
#else
        _putenv_s(k, v);
#endif
    };
    set_default("CUDA_MODULE_LOADING", "EAGER");   // every kernel loaded with the context, not in the first prompt
    set_default("STRATA_MMQ_KQ", "1");      // the K-quant MMQ prompt kernels (every expert of this family is a K-quant)
    // the card decides two defaults (each overridden by setting the variable):
    //  - STRATA_Q8_PACKED: the packed Q8_0 decode copy (+2.3% decode, S26) costs ~0.9 GiB for UD-Q4_K_M; under 12 GiB of
    //    VRAM that VRAM holds more expert slots and a bigger prompt chunk instead (RX 6600: 2 -> 19 slots per layer)
    //  - STRATA_DENSE_MMQ: RDNA2 (gfx103x): rocBLAS has tuned kernels only for FP16-in / FP16-out GEMMs, and the dense
    //    prompt projections (FP16 in, FP32 out) ran its generic ones; llama.cpp's MMQ (q8_1 activations, as for the
    //    experts) took pp4096 from 533 to 917 tok/s on an RX 6600 (KL within the chunk-regrouping floor)
    {
        bool small_card = false, rdna2 = false;
        cudaDeviceProp pr{};
        if (cudaGetDeviceProperties(&pr, 0) == cudaSuccess) {
            small_card = pr.totalGlobalMem < ((size_t) 12 << 30);
#if defined(STRATA_USE_HIP)
            rdna2 = std::strncmp(pr.gcnArchName, "gfx103", 6) == 0;
#endif
        } else {
            (void) cudaGetLastError();
        }
        set_default("STRATA_Q8_PACKED", small_card ? "0" : "1");
        if (rdna2) set_default("STRATA_DENSE_MMQ", "1");
        if (small_card || rdna2)
            std::fprintf(stderr, "strata-q36: %s%s: STRATA_Q8_PACKED=%s%s\n", small_card ? "a card under 12 GiB" : "",
                         rdna2 ? (small_card ? ", RDNA2" : "RDNA2") : "", std::getenv("STRATA_Q8_PACKED"),
                         rdna2 ? (std::string(", STRATA_DENSE_MMQ=") + std::getenv("STRATA_DENSE_MMQ")).c_str() : "");
    }
    if (mtp_d < 0 || mtp_d > 7) { std::fprintf(stderr, "strata-q36: --mtp takes 0 .. 7 drafts\n"); return 2; }
    if (chunk < 0 || chunk > 4096 || ctx < 256) { std::fprintf(stderr, "strata-q36: bad --chunk / --max-context\n"); return 2; }
    Model m;
    std::string err;
    try {
        if (!load(model, m, err, mtp_d > 0)) { std::fprintf(stderr, "strata-q36: %s\n", err.c_str()); return 1; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-q36: %s\n", e.what());
        return 1;
    }
    Engine E;
    E.use_graph = graph;
    if (chunk == 0) {
        // auto: the largest chunk (4096 .. 1024) that fits beside every expert in VRAM; 2048 when they will not all fit (a prompt chunk streams every layer's
        // experts over PCIe once, so a bigger chunk pays that once for more tokens)
        size_t fr = 0, tot = 0, experts = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        for (const Layer& L : m.layers) experts += L.stride * (size_t) m.g.n_expert;
        const size_t fr_raw = fr;   // before the packed Q8_0 copies count as given back
        // (the MTP block's experts can go to the CPU, the packed Q8_0 copies can be given back: see place_experts)
        size_t packed = 0;
        for (const auto& pk : m.packed) packed += std::get<2>(pk);
        fr += packed;
        // the engine's buffers at 512 (~1.1 GiB on gfx1201 at 8K, less the scores buffer's shrink) + the small reserve
        const double want = (double) experts + 1.1 * (1ull << 30);
        const bool fits = expert_slots <= 0 && (expert_vram < 0 ? want <= (double) fr
                                                               : (double) experts <= expert_vram * (double) (1ull << 30));
        chunk = fits ? 512 : 2048;
        const double GiB = (double) (1ull << 30);
        if (fits && expert_vram < 0) {
            // a bigger prompt chunk feeds the MMQ expert kernels more rows per expert (R9700, UD-Q4_K_XL, pp4096: 512 3390, 1024
            // 4495, 2048 5353, 4096 5750 tok/s): the largest that fits beside the experts.  Its buffers grow by about 0.5 MiB per
            // token (measured 0.47: 1.66 GiB more at 4096 than at 512), the sessions' K / V caches by 50 KB per cell and session.
            // With several slots a long prompt would hold the other slots' windows for a whole chunk: 1024 at most there.
            // (measured on a 32 GB R9700 at 8192 cells, 512 chunk: 1.77 GiB of buffers + K / V; the free room after init must still hold
            // every expert and the 0.77 GiB reserve, or place_experts gives the packed decode copies back - slower decode - or goes hybrid)
            const int ns = std::max(1, batch);
            const double kv_one = 45056.0, kv_shared = 5120.0;   // per cell: a session's K / V, and the fp16 copies the prompt attention keeps
            const double sess_extra = std::max(0.0, (double) ctx * (kv_one * ns + kv_shared) - 8192.0 * (kv_one + kv_shared));
            const int cap = ns > 1 ? 1024 : 4096;
            for (int c : {4096, 2048, 1024})
                if (c <= cap && (double) experts + 2.75 * GiB + (double) (c - 512) * 0.5 * 1048576.0 + sess_extra <= (double) fr_raw) {   // (no give-back of the packed decode copies for a bigger chunk)
                    chunk = c;
                    break;
                }
        } else if (!fits) {
            // the hybrid mode: 2048 (a chunk streams every layer's experts over PCIe once); a small card, where the chunk's buffers
            // would take the slots' room, falls back to less
            const double fr_gib = (double) fr / GiB;
            chunk = fr_gib >= 7.5 ? 2048 : fr_gib >= 5.5 ? 1024 : 512;
        }
        if (const char* cv = std::getenv("STRATA_Q36_CHUNK")) chunk = std::max(256, std::min(4096, std::atoi(cv)));   // force a chunk (512: the old default)
        if (std::getenv("STRATA_Q36_PLACE_DEBUG"))
            std::fprintf(stderr, "strata-q36: chunk rule: free %.2f GiB (raw %.2f, with %.2f packed), main experts %.2f GiB, chunk %d\n",
                         fr / 1073741824.0, fr_raw / 1073741824.0, packed / 1073741824.0, experts / 1073741824.0, chunk);
    }
    E.init(m, ctx, chunk, batch, vision);
    if (!place_experts(m, E, expert_vram, cpu_threads, hybrid_rows, expert_slots, err)) {
        std::fprintf(stderr, "strata-q36: %s\n", err.c_str());
        return 1;
    }
    struct Report {
        Engine& E;
        ~Report() {
            if (E.verify_k) std::fprintf(stderr, "strata-q36 VERIFY: %lld comparisons, %lld mismatches\n", E.v_calls, E.v_bad);
            if (E.verify_k)
                for (const auto& kv : E.v_by)
                    std::fprintf(stderr, "strata-q36 VERIFY   %s: %lld mismatches\n", kv.first.c_str(), kv.second);
            if (E.dprof_n > 0) {
                std::fprintf(stderr, "strata-q36 dprof: %lld small forwards, GPU timeline per stage (ms total, ms per forward; host ms):\n", E.dprof_n);
                for (const auto& kv : E.dprof_by)
                    std::fprintf(stderr, "strata-q36 dprof: %8.0f %7.2f  host %8.0f  %s\n", kv.second, kv.second / E.dprof_n,
                                 E.dprof_host[kv.first], kv.first.c_str());
            }
            if (!E.m->hybrid) return;
            const long long n = E.hy_cpu + E.hy_gpu;
            std::fprintf(stderr, "strata-q36: hybrid experts: %lld of %lld batch entries on the GPU (%.1f%%), %lld on the "
                         "CPU pool in %.0f ms, %lld slot replacements\n", E.hy_gpu, n, n ? 100.0 * E.hy_gpu / n : 0.0,
                         E.hy_cpu, E.hy_cpu_ms, E.hy_swaps);
            std::fprintf(stderr, "strata-q36: %lld small forwards (decode, verify, MTP) in %.0f ms host time (%.1f ms each), "
                         "of it %.0f ms waiting for the GPU's routing\n", E.fwd_small_n, E.fwd_small_ms,
                         E.fwd_small_n ? E.fwd_small_ms / E.fwd_small_n : 0.0, E.hy_wait_ms);
        }
    } report{E};
    E.reset();
    E.mtp_d = m.has_mtp ? mtp_d : 0;
    if (draft_vocab < 0 && std::getenv("STRATA_Q36_DRAFT_VOCAB") != nullptr) draft_vocab = std::atoi(std::getenv("STRATA_Q36_DRAFT_VOCAB"));
    E.draft_n = std::max(0, draft_vocab);
    E.bmtp = (batch_mtp || (std::getenv("STRATA_BATCH_MTP") != nullptr && std::string(std::getenv("STRATA_BATCH_MTP")) != "0")) && E.n_sess > 1 && E.mtp_d > 0;
    if (batch_mtp && !E.bmtp) std::fprintf(stderr, "strata-q36: --batch-mtp needs --batch N (2 or more) and --mtp N: ignored\n");
    if (warm) E.warmup();
    if (serve_mode) return serve(E);
    if (bench_pp > 0 || bench_tg > 0) {
        // llama-bench's shapes: a prompt of bench_pp tokens from a fresh state, then bench_tg tokens one at a time
        std::vector<int32_t> toks((size_t) std::max(bench_pp, 1));
        std::mt19937 rng(1234);
        for (auto& t : toks) t = (int32_t) (rng() % 200000u);
        for (int rep = 0; rep < 3; ++rep) {
            E.reset();
            const auto t0 = Clock::now();
            E.process(toks.data(), (int64_t) toks.size(), [](int64_t) {});
            CK(cudaStreamSynchronize(E.st));
            const double pp_ms = ms_since(t0);
            const auto t1 = Clock::now();
            int nx = E.greedy();
            for (int i = 0; i < bench_tg; ++i) {
                const int32_t one = nx;
                E.process(&one, 1, [](int64_t) {});
                nx = E.greedy();
            }
            const double tg_ms = ms_since(t1);
            std::printf("bench %d: pp%d %.1f tok/s, tg%d %.1f tok/s\n", rep, (int) toks.size(),
                        (double) toks.size() / (pp_ms / 1000.0), bench_tg,
                        bench_tg > 0 ? (double) bench_tg / (tg_ms / 1000.0) : 0.0);
            std::fflush(stdout);
        }
        return 0;
    }
    const std::vector<int32_t> prompt = parse_ids(tokens.c_str()), forced = parse_ids(force.c_str());
    if (prompt.empty()) { std::fprintf(stderr, "strata-q36: --tokens, --serve or --bench-pp is required\n"); return 2; }
    if (!forced.empty()) n_gen = (int) forced.size() + 1;
    FILE* ff = dump_full.empty() ? nullptr : std::fopen(dump_full.c_str(), "wb");
    std::vector<float> lg;
    std::vector<int32_t> gen;
    if (!emb_file.empty()) {
        if (!vision) { std::fprintf(stderr, "strata-q36: --emb needs --vision\n"); return 2; }
        ImgReq ir;
        const std::string ve = build_images(emb_file, prompt, m.g.n_embd, {}, ir);
        if (!ve.empty()) { std::fprintf(stderr, "strata-q36: %s\n", ve.c_str()); return 2; }
        E.set_images(ir.rows, ir.cell_row, ir.mrope, (int64_t) prompt.size(), ir.delta);
        std::fprintf(stderr, "strata-q36: %zu picture(s), %zu rows, text after them at cell + %d\n", ir.imgs.size(),
                     ir.rows.size() / (size_t) m.g.n_embd, (int) ir.delta);
    }
    const auto t0 = Clock::now();
    E.process(prompt.data(), (int64_t) prompt.size(), [](int64_t) {});
    const double pp_ms = ms_since(t0);
    const auto t1 = Clock::now();
    if (E.mtp_d > 0 && forced.empty() && !ff) {
        long long offered = 0, accepted = 0;
        std::vector<int32_t> fix, drafts;
        int32_t picks[16];
        int nx = E.greedy();
        while ((int) gen.size() < n_gen) {
            gen.push_back(nx);
            if ((int) gen.size() >= n_gen) break;
            const int a = E.spec_step(nx, fix, E.mtp_d, drafts, picks);
            offered += E.mtp_d;
            accepted += a;
            fix.assign(drafts.begin(), drafts.begin() + a);
            for (int i = 0; i < a && (int) gen.size() < n_gen; ++i) gen.push_back(drafts[(size_t) i]);
            nx = picks[a];
        }
        gen.resize((size_t) n_gen);
        std::printf("gen:");
        for (int t : gen) std::printf(" %d", t);
        std::printf("\n");
        std::fprintf(stderr, "strata-q36: prompt %zu tokens in %.1f ms, %zu generated in %.1f ms (%.1f tok/s), MTP %d drafts: "
                     "%lld of %lld accepted (%.0f%%)\n", prompt.size(), pp_ms, gen.size(), ms_since(t1),
                     gen.size() / (ms_since(t1) / 1000.0), E.mtp_d, accepted, offered,
                     offered ? 100.0 * accepted / offered : 0.0);
        return 0;
    }
    for (int i = 0; i < n_gen; ++i) {
        const int a = E.greedy();
        gen.push_back(a);
        if (ff) {
            E.host_logits(lg);
            std::fwrite(lg.data(), 4, lg.size(), ff);
        }
        if (i + 1 == n_gen) break;
        const int32_t nx = i < (int) forced.size() ? forced[(size_t) i] : a;
        E.process(&nx, 1, [](int64_t) {});
    }
    if (ff) std::fclose(ff);
    std::printf("gen:");
    for (int t : gen) std::printf(" %d", t);
    std::printf("\n");
    std::fprintf(stderr, "strata-q36: prompt %zu tokens in %.1f ms (%.1f tok/s), %zu generated in %.1f ms (%.1f tok/s)\n",
                 prompt.size(), pp_ms, prompt.size() / (pp_ms / 1000.0), gen.size(), ms_since(t1),
                 gen.size() / (ms_since(t1) / 1000.0));
    return 0;
}
