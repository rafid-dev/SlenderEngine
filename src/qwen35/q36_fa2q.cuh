// q36_fa2q.cuh - P-1 (Q36-S, STRATA_Q36_FA2Q=1): k_fa2p_prompt in half the shared memory and registers, so two blocks run
// on a SM (Nsight Compute: k_fa2p_prompt held one block of 8 warps a SM by 133 registers and 82.7 KB of shared memory,
// math-pipe throttle its top stall, the tensor pipe idle through each block's softmax).
//  - 4 tokens a block (R = 32 rows: 4 tokens x 8 query heads of a KV head) instead of 8;
//  - one K / V tile buffer: K_j, then V_j (staged by cp.async while the softmax runs), then K_{j+1} (while the next block on
//    the SM computes); shared memory ~41 KB;
//  - warp (rb, ch): O rows rb * 16 .., columns ch * 64 .. (4 accumulator tiles, against 8): ~half the registers;
//  - S: the four 16 x 16 tiles of a 32 x 32 block (warps 0-3), each over the whole head dim in the same k order as
//    k_fa2p_prompt; the softmax by 4 threads a row of 8 cells (threads 0-127) as k_fa2p_prompt; O += P V in the same order.
// Every row's arithmetic is k_fa2p_prompt's: the same bits (checked bitwise in the harness and in the engine).
// grid (ceil(T / 4), HKV), block 256, fa2q::kSmem bytes. Segments (c_lo, c_hi, flags, sml) as k_fa2p_prompt.
#pragma once
#include <cuda_fp16.h>
#include <mma.h>

namespace fa2q {
constexpr int Q = 4, G = 8, R = Q * G, BK = 32, HD = 256, LDH = HD + 8, LDS = BK + 4, LDP = BK + 8;
constexpr size_t kSmem = (size_t) R * LDH * 2 + (size_t) BK * LDH * 2 + (size_t) R * LDS * 4 + (size_t) R * LDP * 2 + 3 * R * 4 +
                         256 * 4;   // + the row-number matrix
__device__ __forceinline__ void cp16(void* smem, const void* gmem, bool valid) {
    const unsigned sa = (unsigned) __cvta_generic_to_shared(smem);
    const int n = valid ? 16 : 0;   // 0: zero-fill
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(gmem), "r"(n));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void wait_all() { asm volatile("cp.async.wait_all;\n" ::); }
}  // namespace fa2q

__global__ void __launch_bounds__(256, 2) k_fa2q_prompt(const __half* __restrict__ q16, const __half* __restrict__ k16,
                                                        const __half* __restrict__ v16, int T, int H, int ld_cells, int pos0,
                                                        float scale, float* __restrict__ out, int c_lo = 0, int c_hi = 0x7fffffff,
                                                        int flags = 3, float* __restrict__ sml = nullptr) {
    using namespace nvcuda;
    using namespace fa2q;
    extern __shared__ __align__(16) unsigned char fa2q_sm[];
    __half* sQ = (__half*) fa2q_sm;
    __half* sKV = sQ + R * LDH;     // K_j, then V_j
    float* sS = (float*) (sKV + BK * LDH);
    __half* sP = (__half*) (sS + R * LDS);
    float* sM = (float*) (sP + R * LDP);
    float* sL = sM + R;
    float* sA = sL + R;
    float* sX = sA + R;             // 256 floats: the row-number matrix
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31, kvh = blockIdx.y, t0 = blockIdx.x * Q;
    const int c_end = min(pos0 + min(T - 1, t0 + Q - 1) + 1, c_hi);
    const __half* kh = k16 + (size_t) kvh * ld_cells * HD;
    const __half* vh = v16 + (size_t) kvh * ld_cells * HD;
    auto stage = [&](const __half* src, int kb) {   // a 32-cell tile into sKV, 4 16-B pieces a thread
#pragma unroll
        for (int m = 0; m < BK * HD / 8 / 256; ++m) {
            const int i = tid + 256 * m, c = i / (HD / 8), d = (i % (HD / 8)) * 8;
            const bool ok = kb + c < c_end;
            cp16(sKV + c * LDH + d, src + (size_t) (ok ? kb + c - c_lo : 0) * HD + d, ok);
        }
        commit();
    };
    for (int i = tid; i < R * HD / 8; i += 256) {   // Q: row r = token t0 + r / G, head kvh * G + r % G
        const int r = i / (HD / 8), d = (i % (HD / 8)) * 8, t = t0 + r / G;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (t < T) v = *(const uint4*) (q16 + ((size_t) t * H + kvh * G + r % G) * HD + d);
        *(uint4*) (sQ + r * LDH + d) = v;
    }
    const bool first = flags & 1, last = flags & 2;
    if (tid < R) {
        const int t = t0 + tid / G;
        const size_t ri = (size_t) t * H + kvh * G + tid % G;
        sM[tid] = !first && t < T ? sml[2 * ri] : -INFINITY;
        sL[tid] = !first && t < T ? sml[2 * ri + 1] : 0.f;
    }
    sX[tid] = (float) (tid / 16);   // the row-number matrix (16 x 16, ld 16)
    __syncthreads();
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> rowf;
    wmma::load_matrix_sync(rowf, sX, 16, wmma::mem_row_major);
    const int rb = w & 1, ch = w >> 1;   // O rows rb * 16 .., columns ch * 64 ..; S (warps 0-3): rows (w & 1) * 16, cells (w >> 1) * 16
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[4];
#pragma unroll
    for (int f = 0; f < 4; ++f) wmma::fill_fragment(o[f], 0.f);
    if (!first) {   // the unnormalized O of the segments before, through a 16 x 16 staging tile a warp in sKV (free until K_0)
        float* stg = (float*) sKV + w * 256;
#pragma unroll
        for (int f = 0; f < 4; ++f) {
            for (int e = lane; e < 256; e += 32) {
                const int r = rb * 16 + e / 16, t = t0 + r / G;
                stg[e] = t < T ? out[((size_t) t * H + kvh * G + r % G) * HD + ch * 64 + f * 16 + e % 16] : 0.f;
            }
            __syncwarp();
            wmma::load_matrix_sync(o[f], stg, 16, wmma::mem_row_major);
            __syncwarp();
        }
        __syncthreads();   // every warp's staging read before K_0 lands there
    }
    if (c_lo < c_end) stage(kh, c_lo);
    const int sr = tid >> 2, sc0 = (tid & 3) * 8, s_tok = t0 + sr / G;   // softmax: threads 0-127 (rows 0-31)
    const int s_lim = s_tok < T ? pos0 + s_tok + 1 : 0;
    wait_all();
    __syncthreads();   // K_0
    for (int kb = c_lo; kb < c_end; kb += BK) {
        if (w < 4) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> s0;
            wmma::fill_fragment(s0, 0.f);
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b0;
            const int srb = w & 1, scb = w >> 1;
#pragma unroll 8
            for (int k0 = 0; k0 < HD; k0 += 16) {
                wmma::load_matrix_sync(a, sQ + srb * 16 * LDH + k0, LDH);
                wmma::load_matrix_sync(b0, sKV + (scb * 16) * LDH + k0, LDH);
                wmma::mma_sync(s0, a, b0, s0);
            }
            wmma::store_matrix_sync(sS + srb * 16 * LDS + scb * 16, s0, LDS, wmma::mem_row_major);
        }
        __syncthreads();   // S complete; every read of K_j done
        stage(vh, kb);     // V_j while the softmax runs
        if (tid < 4 * R) {   // the online softmax step: 4 threads a row, 8 cells each (k_fa2p_prompt's mapping and order)
            float sv[8], mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                sv[i] = kb + sc0 + i < s_lim ? sS[sr * LDS + sc0 + i] * scale : -INFINITY;
                mx = fmaxf(mx, sv[i]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
            const float m_old = sM[sr], m_new = fmaxf(m_old, mx);
            float sum = 0.f;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const float pv = sv[i] == -INFINITY ? 0.f : expf(sv[i] - m_new);
                sP[sr * LDP + sc0 + i] = __float2half(pv);
                sum += pv;
            }
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            if ((tid & 3) == 0) {
                const float al = m_new == -INFINITY ? 1.f : expf(m_old - m_new);
                sA[sr] = al;
                sL[sr] = sL[sr] * al + sum;
                sM[sr] = m_new;
            }
        }
        wait_all();        // V_j (this thread's pieces)
        __syncthreads();   // V_j complete, P and the rescales written
#pragma unroll
        for (int f = 0; f < 4; ++f)
            for (int e = 0; e < o[f].num_elements; ++e) o[f].x[e] *= sA[rb * 16 + (int) rowf.x[e]];
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb;
#pragma unroll
        for (int k0 = 0; k0 < BK; k0 += 16) {
            wmma::load_matrix_sync(pa, sP + rb * 16 * LDP + k0, LDP);
#pragma unroll
            for (int f = 0; f < 4; ++f) {
                wmma::load_matrix_sync(vb, sKV + k0 * LDH + ch * 64 + f * 16, LDH);
                wmma::mma_sync(o[f], pa, vb, o[f]);
            }
        }
        __syncthreads();   // every read of V_j, P and sA done
        if (kb + BK < c_end) {
            stage(kh, kb + BK);   // K_{j+1}
            wait_all();
            __syncthreads();
        }
    }
    __syncthreads();   // (sM / sL final; sQ free)
    float* stg = (float*) sQ + w * 256;   // a 16 x 16 staging tile a warp (8 x 1 KB in sQ's 16.5 KB)
#pragma unroll
    for (int f = 0; f < 4; ++f) {
        wmma::store_matrix_sync(stg, o[f], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = lane; e < 256; e += 32) {
            const int r = rb * 16 + e / 16, t = t0 + r / G;
            if (t < T) out[((size_t) t * H + kvh * G + r % G) * HD + ch * 64 + f * 16 + e % 16] = last ? stg[e] / sL[r] : stg[e];
        }
        __syncwarp();
    }
    if (!last && tid < R) {
        const int t = t0 + tid / G;
        if (t < T) {
            const size_t ri = (size_t) t * H + kvh * G + tid % G;
            sml[2 * ri] = sM[tid];
            sml[2 * ri + 1] = sL[tid];
        }
    }
}
