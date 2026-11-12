// q36_swq.cu - Q36-T TP2: see the kernel's comment. Its own translation unit so it is compiled with --use_fast_math, the
// flags of the two kernels it fuses (MMQ::swiglu in moe_mmq.cu, quantize_mmq_q8_1 in llama.cpp): the same division,
// reciprocal and flush-to-zero, so the same bytes.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// TP2 (Q36-T, STRATA_Q36_SWQ=1), compiled with --use_fast_math (as moe_mmq.cu's swiglu and llama.cpp's quantize): the prompt experts' SwiGLU and the q8_1 MMQ image of its output in one pass, for the
// down product: each value is MMQ::swiglu's expression (g / (1 + exp(-g)) * u, gate half then up half of a row) and each
// block llama.cpp's quantize_mmq_q8_1 (D4: a float scale a 32 values; DS4: a half2 (scale, scale * sum of q) a 32
// values; the sum of the unquantized values, as the build's overlay quantize.cu), its threads and shuffles: the same bytes as swiglu + quantize. A 144-byte block_q8_1_mmq: 16 bytes of
// scales, then 128 int8. grid (rows, ceil(ne0 / 512)), block 128; ne0 = n_ff padded to 512.
template <bool DS4>
__global__ void __launch_bounds__(128) k_swiglu_q8mmq(const float* __restrict__ gu, unsigned char* __restrict__ y, int64_t rows,
                                                      int64_t n_ff, int64_t ne0) {
    const int64_t i0 = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * 4;
    if (i0 >= ne0) return;
    const int64_t r = blockIdx.x;
    float4 xi = make_float4(0.f, 0.f, 0.f, 0.f);
    if (i0 < n_ff) {
        const float* row = gu + r * 2 * n_ff;
        float g = row[i0], u = row[n_ff + i0];
        xi.x = g / (1.0f + __expf(-g)) * u;
        g = row[i0 + 1]; u = row[n_ff + i0 + 1];
        xi.y = g / (1.0f + __expf(-g)) * u;
        g = row[i0 + 2]; u = row[n_ff + i0 + 2];
        xi.z = g / (1.0f + __expf(-g)) * u;
        g = row[i0 + 3]; u = row[n_ff + i0 + 3];
        xi.w = g / (1.0f + __expf(-g)) * u;
    }
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));
#pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));
    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x * d_inv);
    q.y = roundf(xi.y * d_inv);
    q.z = roundf(xi.z * d_inv);
    q.w = roundf(xi.w * d_inv);
    const float d = 1.0f / d_inv;
    float sum = 0.0f;
    if (DS4) {   // the build's quantize_mmq_q8_1 (strata's overlay of llama.cpp): the sum of the unquantized values
        sum = xi.x + xi.y + xi.z + xi.w;
#pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, 32);
    }
    const int64_t k_block = i0 / 128, iqs = i0 % 128;
    unsigned char* b = y + (size_t) (k_block * rows + r) * 144;
    reinterpret_cast<char4*>(b + 16)[iqs / 4] = q;
    if (iqs % 32 == 0) {
        if (DS4) reinterpret_cast<__half2*>(b)[iqs / 32] = __halves2half2(__float2half(d), __float2half(sum));
        else reinterpret_cast<float*>(b)[iqs / 32] = d;
    }
}
void q36_swiglu_q8mmq(const float* gu, void* y, int64_t rows, int64_t n_ff, int lay, cudaStream_t st) {
    const int64_t ne0 = (n_ff + 511) / 512 * 512;
    const dim3 g((unsigned) rows, (unsigned) ((ne0 + 511) / 512));
    if (lay == 2) k_swiglu_q8mmq<true><<<g, 128, 0, st>>>(gu, (unsigned char*) y, rows, n_ff, ne0);
    else k_swiglu_q8mmq<false><<<g, 128, 0, st>>>(gu, (unsigned char*) y, rows, n_ff, ne0);
}
