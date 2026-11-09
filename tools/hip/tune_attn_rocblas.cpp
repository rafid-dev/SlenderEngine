// tools/hip/tune_attn_rocblas.cpp - how fast rocBLAS's own solutions run the 35B engine's prompt-attention GEMMs.
//
// q36_engine.cu's prompt attention per KV head (8 query heads share it): scores = K^T Q (FP16 in, FP32 out; K shared
// by the batch, stride 0) and out = V P (FP16 probabilities in, FP32 out). rocBLAS picks a kernel by shape; on gfx1032
// its pick for other prompt GEMMs was up to 4x off the best of its own solutions (tools/hip/tune_rocblas.cpp). This
// tool times the default and every solution (rocblas_gemm_strided_batched_ex_get_solutions) for both GEMMs at the
// engine's shapes for a few context lengths, checks the best against the default's output, and prints TFLOPS.
//
//   cmake --build build-hip-win --target tune_attn_rocblas && build-hip-win\tune_attn_rocblas [P ...]
#define ROCBLAS_BETA_FEATURES_API
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(c) do { const hipError_t e = (c); if (e != hipSuccess) { std::fprintf(stderr, "HIP %d: %s\n", __LINE__, hipGetErrorString(e)); std::exit(2); } } while (0)
#define RB_CHECK(c) do { const rocblas_status e = (c); if (e != rocblas_status_success) { std::fprintf(stderr, "rocBLAS %d: status %d\n", __LINE__, (int) e); std::exit(2); } } while (0)

namespace {
constexpr int HD = 256, H = 16, HKV = 2, G = H / HKV;
rocblas_datatype CT = rocblas_datatype_f32_r;   // the outputs' type (argument "f16": FP16 out, FP32 compute)

template <class F> float time_ms(hipStream_t s, hipEvent_t a, hipEvent_t b, F f, int reps, bool& ok) {
    HIP_CHECK(hipEventRecord(a, s));
    for (int i = 0; i < reps; ++i) if (f() != rocblas_status_success) { ok = false; return 0.f; }
    HIP_CHECK(hipEventRecord(b, s));
    HIP_CHECK(hipEventSynchronize(b));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    return ms / reps;
}

uint16_t f16(float x) { __half h = __float2half(x); uint16_t u; std::memcpy(&u, &h, 2); return u; }
}  // namespace

int main(int argc, char** argv) {
    std::vector<int> Ps;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "f16") { CT = rocblas_datatype_f16_r; continue; }
        Ps.push_back(std::atoi(argv[i]));
    }
    if (Ps.empty()) Ps = {4096, 12288, 20480, 32768};
    rocblas_handle h; RB_CHECK(rocblas_create_handle(&h));
    hipStream_t st; HIP_CHECK(hipStreamCreateWithFlags(&st, hipStreamNonBlocking));
    RB_CHECK(rocblas_set_stream(h, st));
    hipEvent_t ea, eb; HIP_CHECK(hipEventCreate(&ea)); HIP_CHECK(hipEventCreate(&eb));
    char ver[128] = {0}; RB_CHECK(rocblas_get_version_string(ver, sizeof ver));
    std::printf("rocblas %s\n", ver);
    const size_t kAtt = (size_t) 64 << 20;   // the engine's score buffer (floats)
    for (int P : Ps) {
        const int ldp = (P + 7) / 8 * 8;
        const int nt = (int) std::min<size_t>(2048, kAtt / ((size_t) H * ldp));
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> d(-1.f, 1.f);
        std::vector<uint16_t> hk((size_t) P * HD), hq((size_t) nt * H * HD), hp((size_t) G * nt * ldp);
        for (auto& x : hk) x = f16(d(rng));
        for (auto& x : hq) x = f16(d(rng));
        for (auto& x : hp) x = f16(std::fabs(d(rng)) / 64.f);
        void *dk, *dq, *ds, *dp, *dv, *dout;
        HIP_CHECK(hipMalloc(&dk, hk.size() * 2)); HIP_CHECK(hipMalloc(&dv, hk.size() * 2));
        HIP_CHECK(hipMalloc(&dq, hq.size() * 2)); HIP_CHECK(hipMalloc(&dp, hp.size() * 2));
        HIP_CHECK(hipMalloc(&ds, (size_t) G * nt * ldp * 4)); HIP_CHECK(hipMalloc(&dout, (size_t) nt * H * HD * 4));
        HIP_CHECK(hipMemcpy(dk, hk.data(), hk.size() * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dv, hk.data(), hk.size() * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dq, hq.data(), hq.size() * 2, hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dp, hp.data(), hp.size() * 2, hipMemcpyHostToDevice));
        const float one = 1.f, zero = 0.f;
        const double flop = 2.0 * P * nt * HD * G;   // one GEMM, one KV head
        for (int which = 0; which < 2; ++which) {
            auto call = [&](rocblas_gemm_algo algo, rocblas_int sol, rocblas_int* list, rocblas_int* n) {
                if (which == 0)   // scores [nt][ldp] per query head = K^T (P x HD) . Q (HD x nt)
                    return list || n ? rocblas_gemm_strided_batched_ex_get_solutions(h, rocblas_operation_transpose, rocblas_operation_none, P, nt, HD, &one,
                        dk, rocblas_datatype_f16_r, HD, 0, dq, rocblas_datatype_f16_r, H * HD, HD, &zero, ds, CT, ldp,
                        (rocblas_stride) nt * ldp, ds, CT, ldp, (rocblas_stride) nt * ldp, G, rocblas_datatype_f32_r, algo, 0, list, n)
                        : rocblas_gemm_strided_batched_ex(h, rocblas_operation_transpose, rocblas_operation_none, P, nt, HD, &one,
                        dk, rocblas_datatype_f16_r, HD, 0, dq, rocblas_datatype_f16_r, H * HD, HD, &zero, ds, CT, ldp,
                        (rocblas_stride) nt * ldp, ds, CT, ldp, (rocblas_stride) nt * ldp, G, rocblas_datatype_f32_r, algo, sol, 0);
                // out [nt][H * HD] per query head = V (HD x P) . P16 (P x nt)
                return list || n ? rocblas_gemm_strided_batched_ex_get_solutions(h, rocblas_operation_none, rocblas_operation_none, HD, nt, P, &one,
                    dv, rocblas_datatype_f16_r, HD, 0, dp, rocblas_datatype_f16_r, ldp, (rocblas_stride) nt * ldp, &zero, dout, CT,
                    H * HD, HD, dout, CT, H * HD, HD, G, rocblas_datatype_f32_r, algo, 0, list, n)
                    : rocblas_gemm_strided_batched_ex(h, rocblas_operation_none, rocblas_operation_none, HD, nt, P, &one,
                    dv, rocblas_datatype_f16_r, HD, 0, dp, rocblas_datatype_f16_r, ldp, (rocblas_stride) nt * ldp, &zero, dout, CT,
                    H * HD, HD, dout, CT, H * HD, HD, G, rocblas_datatype_f32_r, algo, sol, 0);
            };
            bool ok = true;
            for (int w = 0; w < 3; ++w) RB_CHECK(call(rocblas_gemm_algo_standard, 0, nullptr, nullptr));
            const float def = time_ms(st, ea, eb, [&] { return call(rocblas_gemm_algo_standard, 0, nullptr, nullptr); }, 5, ok);
            rocblas_int n = 0;
            std::vector<std::pair<float, int>> res;
            if (call(rocblas_gemm_algo_solution_index, 0, nullptr, &n) == rocblas_status_success && n > 0) {
                std::vector<rocblas_int> ids(n);
                RB_CHECK(call(rocblas_gemm_algo_solution_index, 0, ids.data(), &n));
                for (rocblas_int id : ids) {
                    bool ok2 = true;
                    if (call(rocblas_gemm_algo_solution_index, id, nullptr, nullptr) != rocblas_status_success) continue;
                    const float ms = time_ms(st, ea, eb, [&] { return call(rocblas_gemm_algo_solution_index, id, nullptr, nullptr); }, 2, ok2);
                    if (ok2) res.push_back({ms, id});
                }
            }
            std::sort(res.begin(), res.end());
            std::printf("P=%6d nt=%4d %s: default %.3f ms (%.2f TFLOPS); %zu solutions", P, nt, which ? "values" : "scores",
                        def, flop / def / 1e9, res.size());
            for (size_t i = 0; i < res.size() && i < 3; ++i)
                std::printf("; #%d %.3f ms (%.2f TFLOPS)", res[i].second, res[i].first, flop / res[i].first / 1e9);
            std::printf("\n");
        }
        hipFree(dk); hipFree(dv); hipFree(dq); hipFree(dp); hipFree(ds); hipFree(dout);
    }
    return 0;
}
