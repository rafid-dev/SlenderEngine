// tools/hip/tune_attn_rocblas.cpp - a checked rocBLAS solution table for the 35B engine's prompt-attention GEMMs.
//
// q36_engine.cu's prompt attention with STRATA_Q36_ATT_F16=1, per KV head (8 query heads share it), FP16 in, FP16 out,
// FP32 compute, one strided batch over the query heads:
//   scores  S = K^T Q   (M = P cells, N = nt rows, K = 256; K shared by the batch, stride 0; alpha = 1/16)
//   values  O = V P16   (M = 256, N = nt, K = P; V shared by the batch)
// nt = min(2048, score floats / (16 * ldp)), ldp = P rounded up to 8. rocBLAS's default kernel for the values GEMM is
// 1.2-2.3x slower than the best of its own solutions past 8K cells on gfx1032. A runtime tuner that applied solution
// indices measured at one shape to other shapes faulted the GPU (and left the HIP runtime unusable until a reboot):
// rocBLAS runs a solution index as given unless rocblas_gemm_flags_check_solution_index is set, even on a shape that
// kernel cannot handle. Here, and in the engine, every solution call sets that flag (a refused shape returns
// rocblas_status_invalid_value and the engine runs the default), and a solution enters the table only if, at the
// bucket's shape and at ~20 neighbouring (P, nt) shapes, it matches the default kernel's output with the engine's exact
// layout: the operands at the second KV head's offsets, NaN in the score rows' padding and after every input (a kernel
// that reads past its operands turns the output to NaN), a canary around every output (a write outside it is caught).
//
//   cmake --build build-hip-win --target tune_attn_rocblas
//   build-hip-win\tune_attn_rocblas --table-out tools\hip\<arch>-q36-attn-rocblas-<version>.txt
//   options: --p 4096,8192,... (bucket upper ends; default 2048 and 4096 ... 65536 by 4096), --mfloats 64,
//            --min-gain 1.05
// The engine reads the table through STRATA_Q36_ATT_TUNING=<path>; a row "v 12288 -343" applies to the values GEMM
// for P in (the previous row's P, 12288]. Indices are valid for one architecture and one rocBLAS build only.
#define ROCBLAS_BETA_FEATURES_API
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(c) do { const hipError_t e = (c); if (e != hipSuccess) { std::fprintf(stderr, "HIP %d: %s\n", __LINE__, hipGetErrorString(e)); std::exit(2); } } while (0)
#define RB_CHECK(c) do { const rocblas_status e = (c); if (e != rocblas_status_success) { std::fprintf(stderr, "rocBLAS %d: status %d\n", __LINE__, (int) e); std::exit(2); } } while (0)

namespace {
constexpr int HD = 256, H = 16, HKV = 2, G = H / HKV, KVH = 1;   // the operands at KV head 1's offsets
constexpr int PMAX = 65536, NTMAX = 2048;
constexpr uint16_t NAN16 = 0x7E00, CANARY = 0x7BFF;   // a quiet NaN; the largest finite FP16 (never a result here)
constexpr size_t GUARD = (size_t) 8 << 20;            // halves of NaN / canary before and after each operand
constexpr uint32_t CHECK = rocblas_gemm_flags_check_solution_index;
uint32_t g_list_flags = 0;   // --list-flags: the flags rocblas_gemm_strided_batched_ex_get_solutions is called with

uint16_t f16(float x) { __half h = __float2half(x); uint16_t u; std::memcpy(&u, &h, 2); return u; }
float f32(uint16_t u) { __half h; std::memcpy(&h, &u, 2); return __half2float(h); }
int ldp_of(int P) { return (P + 7) / 8 * 8; }
int nt_of(int P, size_t floats) { return (int) std::max<size_t>(1, std::min<size_t>(NTMAX, floats / ((size_t) H * ldp_of(P)))); }

struct Buf {   // GUARD halves, the operand region, GUARD halves
    uint16_t* base = nullptr;
    size_t n = 0;
    explicit Buf(size_t n_) : n(n_) { HIP_CHECK(hipMalloc(&base, (n + 2 * GUARD) * 2)); }
    ~Buf() { (void) hipFree(base); }
    uint16_t* p() const { return base + GUARD; }
    // the guards and the first `used` halves of the operand region
    void put(const std::vector<uint16_t>& h) const { HIP_CHECK(hipMemcpy(base, h.data(), h.size() * 2, hipMemcpyHostToDevice)); }
    void fill(uint16_t v, size_t used) const {
        put(std::vector<uint16_t>(std::min(used, n) + 2 * GUARD, v));
    }
    std::vector<uint16_t> get(size_t used) const {
        std::vector<uint16_t> h(std::min(used, n) + 2 * GUARD);
        HIP_CHECK(hipMemcpy(h.data(), base, h.size() * 2, hipMemcpyDeviceToHost));
        return h;
    }
};

struct Ctx {
    rocblas_handle h;
    hipStream_t st;
    hipEvent_t ea, eb;
    Buf K{(size_t) HKV * PMAX * HD}, V{(size_t) HKV * PMAX * HD}, Q{(size_t) NTMAX * H * HD};
    Buf S{(size_t) 64 << 20}, P16{(size_t) 64 << 20}, O{(size_t) NTMAX * H * HD};
};

// which: 0 scores, 1 values. sol 0 + standard: the default kernel; else the solution index with the check flag.
rocblas_status gemm_raw(Ctx& c, int which, int P, int nt, int32_t sol, uint32_t flags, bool list = false,
                        rocblas_int* ids = nullptr, rocblas_int* n = nullptr) {
    const int ldp = ldp_of(P);
    const float one = 1.f, zero = 0.f, scale = 1.f / 16.f;
    const rocblas_gemm_algo algo = sol || list ? rocblas_gemm_algo_solution_index : rocblas_gemm_algo_standard;
    const auto f = rocblas_datatype_f16_r, f32t = rocblas_datatype_f32_r;
    if (which == 0) {
        const uint16_t* A = c.K.p() + (size_t) KVH * P * HD;
        const uint16_t* B = c.Q.p() + (size_t) KVH * G * HD;
        uint16_t* C = c.S.p() + (size_t) KVH * G * nt * ldp;
        if (list)
            return rocblas_gemm_strided_batched_ex_get_solutions(c.h, rocblas_operation_transpose, rocblas_operation_none, P,
                nt, HD, &scale, A, f, HD, 0, B, f, H * HD, HD, &zero, C, f, ldp, (rocblas_stride) nt * ldp, C, f, ldp,
                (rocblas_stride) nt * ldp, G, f32t, algo, g_list_flags, ids, n);
        return rocblas_gemm_strided_batched_ex(c.h, rocblas_operation_transpose, rocblas_operation_none, P, nt, HD, &scale,
            A, f, HD, 0, B, f, H * HD, HD, &zero, C, f, ldp, (rocblas_stride) nt * ldp, C, f, ldp, (rocblas_stride) nt * ldp,
            G, f32t, algo, sol, flags);
    }
    const uint16_t* A = c.V.p() + (size_t) KVH * P * HD;
    const uint16_t* B = c.P16.p() + (size_t) KVH * G * nt * ldp;
    uint16_t* C = c.O.p() + (size_t) KVH * G * HD;
    if (list)
        return rocblas_gemm_strided_batched_ex_get_solutions(c.h, rocblas_operation_none, rocblas_operation_none, HD, nt, P,
            &one, A, f, HD, 0, B, f, ldp, (rocblas_stride) nt * ldp, &zero, C, f, H * HD, HD, C, f, H * HD, HD, G, f32t,
            algo, g_list_flags, ids, n);
    return rocblas_gemm_strided_batched_ex(c.h, rocblas_operation_none, rocblas_operation_none, HD, nt, P, &one, A, f, HD,
        0, B, f, ldp, (rocblas_stride) nt * ldp, &zero, C, f, H * HD, HD, C, f, H * HD, HD, G, f32t, algo, sol, flags);
}

// a solution: first asked (the check flag: rocBLAS only answers whether that kernel can run this shape, success or
// invalid_value, and runs nothing), then run with no flags
rocblas_status gemm(Ctx& c, int which, int P, int nt, int32_t sol, bool list = false, rocblas_int* ids = nullptr,
                    rocblas_int* n = nullptr) {
    if (list) return gemm_raw(c, which, P, nt, 0, 0, true, ids, n);
    if (sol) {
        const rocblas_status q = gemm_raw(c, which, P, nt, sol, CHECK);
        if (q != rocblas_status_success) return q;
    }
    return gemm_raw(c, which, P, nt, sol, 0);
}

// the inputs for a shape: K, V, Q random in (-1, 1) up to the operand's end, NaN after it; P16 probabilities in the
// P rows of each column, NaN in the padding rows (P .. ldp) and after
// (a table of 4096 random FP16 values indexed by a cheap hash: the host side stays fast at 64M values a shape)
void fill_inputs(Ctx& c, int P, int nt, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<uint16_t> sym(4096), pos(4096);
    for (auto& x : sym) x = f16(d(rng));
    for (auto& x : pos) x = f16(std::fabs(d(rng)) / 64.f);
    uint32_t z = seed | 1u;
    auto next = [&] { z ^= z << 13; z ^= z >> 17; z ^= z << 5; return z & 4095u; };
    const int ldp = ldp_of(P);
    const size_t kv = (size_t) HKV * P * HD, q = (size_t) nt * H * HD, pp = (size_t) HKV * G * nt * ldp;
    for (Buf* b : {&c.K, &c.V}) {
        std::vector<uint16_t> h(kv + 2 * GUARD, NAN16);
        for (size_t i = 0; i < kv; ++i) h[GUARD + i] = sym[next()];
        b->put(h);
    }
    {
        std::vector<uint16_t> h(q + 2 * GUARD, NAN16);
        for (size_t i = 0; i < q; ++i) h[GUARD + i] = sym[next()];
        c.Q.put(h);
    }
    {
        std::vector<uint16_t> h(pp + 2 * GUARD, NAN16);
        for (size_t col = 0; col < (size_t) HKV * G * nt; ++col)
            for (int r = 0; r < P; ++r) h[GUARD + col * ldp + r] = pos[next()];
        c.P16.put(h);
    }
}

// the output buffer after a call (filled with the canary before it)
std::vector<uint16_t> run_out(Ctx& c, int which, int P, int nt, int32_t sol, rocblas_status& status) {
    Buf& out = which ? c.O : c.S;
    const size_t used = which ? (size_t) nt * H * HD : (size_t) HKV * G * nt * ldp_of(P);
    out.fill(CANARY, used);
    status = gemm(c, which, P, nt, sol);
    HIP_CHECK(hipDeviceSynchronize());
    return out.get(used);
}

struct Cmp { bool ok = true; double rel_l2 = 0, max_ulps = 0; size_t stray = 0, nonfinite = 0; };
Cmp compare(const std::vector<uint16_t>& ref, const std::vector<uint16_t>& got) {
    Cmp r;
    long double d2 = 0, r2 = 0;
    double max_abs = 0, max_ref = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] == CANARY) { if (got[i] != CANARY) ++r.stray; continue; }
        const float a = f32(ref[i]), b = f32(got[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) { ++r.nonfinite; continue; }
        const double dd = (double) b - a;
        d2 += (long double) dd * dd; r2 += (long double) a * a;
        max_abs = std::max(max_abs, std::fabs(dd));
        max_ref = std::max(max_ref, (double) std::fabs(a));
    }
    r.rel_l2 = std::sqrt((double) (d2 / std::max(r2, 1e-300L)));
    // the kernels sum in different orders (up to 64K products), then both round to FP16: the error in units of the
    // largest output's FP16 step (as tools/hip/tune_rocblas.cpp), and the relative L2 far below FP16's 2^-11
    const double ulp = std::ldexp(1.0, (int) std::floor(std::log2(std::max(max_ref, 1e-30))) - 10);
    r.max_ulps = max_abs / ulp;
    r.ok = r.stray == 0 && r.nonfinite == 0 && r.rel_l2 <= 1e-4 && r.max_ulps <= 4.0;
    return r;
}

float time_ms(Ctx& c, int which, int P, int nt, int32_t sol, int reps) {
    if (gemm(c, which, P, nt, sol) != rocblas_status_success) return -1.f;
    HIP_CHECK(hipEventRecord(c.ea, c.st));
    for (int i = 0; i < reps; ++i)
        if (gemm(c, which, P, nt, sol) != rocblas_status_success) return -1.f;
    HIP_CHECK(hipEventRecord(c.eb, c.st));
    HIP_CHECK(hipEventSynchronize(c.eb));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, c.ea, c.eb));
    return ms / reps;
}

std::vector<int> csv_ints(const std::string& s) {
    std::vector<int> v;
    size_t b = 0;
    while (b <= s.size()) {
        const size_t e = s.find(',', b);
        v.push_back(std::atoi(s.substr(b, e == std::string::npos ? std::string::npos : e - b).c_str()));
        if (e == std::string::npos) break;
        b = e + 1;
    }
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    std::vector<int> buckets{2048};
    for (int p = 4096; p <= PMAX; p += 4096) buckets.push_back(p);
    size_t floats = (size_t) 64 << 20;
    double min_gain = 1.05;
    std::string out_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&] { if (i + 1 >= argc) { std::fprintf(stderr, "missing value\n"); std::exit(2); } return std::string(argv[++i]); };
        if (a == "--p") buckets = csv_ints(next());
        else if (a == "--mfloats") floats = (size_t) std::max(4, std::atoi(next().c_str())) << 20;
        else if (a == "--min-gain") min_gain = std::atof(next().c_str());
        else if (a == "--table-out") out_path = next();
        else if (a == "--list-flags") g_list_flags = (uint32_t) std::atoi(next().c_str());
        else { std::fprintf(stderr, "usage: %s [--p P1,P2,...] [--mfloats 64] [--min-gain 1.05] [--table-out PATH]\n", argv[0]); return 2; }
    }
    std::sort(buckets.begin(), buckets.end());
    if (buckets.empty() || buckets.front() < 64 || buckets.back() > PMAX) { std::fprintf(stderr, "buckets in [64, %d]\n", PMAX); return 2; }

    Ctx c;
    RB_CHECK(rocblas_create_handle(&c.h));
    HIP_CHECK(hipStreamCreateWithFlags(&c.st, hipStreamNonBlocking));
    RB_CHECK(rocblas_set_stream(c.h, c.st));
    HIP_CHECK(hipEventCreate(&c.ea)); HIP_CHECK(hipEventCreate(&c.eb));
    char ver[128] = {0}; RB_CHECK(rocblas_get_version_string(ver, sizeof ver));
    hipDeviceProp_t prop{}; HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::string arch(prop.gcnArchName); if (const auto k = arch.find(':'); k != std::string::npos) arch.resize(k);
    std::printf("meta arch=%s rocblas=%s buckets=%zu score_floats=%zuM min_gain=%.2f\n", arch.c_str(), ver, buckets.size(),
                floats >> 20, min_gain);

    struct Row { int which, p_hi; int32_t sol; float def_ms, best_ms; int refused, checked; };
    std::vector<Row> rows;
    for (size_t bi = 0; bi < buckets.size(); ++bi) {
        const int P = buckets[bi], p_lo = bi ? buckets[bi - 1] : 0;
        const int nt = nt_of(P, floats);
        for (int which = 0; which < 2; ++which) {
            const char* name = which ? "values" : "scores";
            const double flop = 2.0 * P * nt * HD * G;
            fill_inputs(c, P, nt, 0x5EED + P * 2 + which);
            rocblas_status s0;
            const std::vector<uint16_t> ref = run_out(c, which, P, nt, 0, s0);
            if (s0 != rocblas_status_success) { std::fprintf(stderr, "default %s failed at P=%d\n", name, P); return 2; }
            const float def = time_ms(c, which, P, nt, 0, 5);
            rocblas_int n = 0;
            std::vector<std::pair<float, int32_t>> quick;
            if (gemm(c, which, P, nt, 0, true, nullptr, &n) == rocblas_status_success && n > 0) {
                std::vector<rocblas_int> ids(n);
                RB_CHECK(gemm(c, which, P, nt, 0, true, ids.data(), &n));
                for (int k = 0; k < n; ++k) {
                    const float ms = time_ms(c, which, P, nt, ids[k], 1);
                    if (ms > 0) quick.push_back({ms, ids[k]});
                }
            }
            std::sort(quick.begin(), quick.end());
            std::printf("P=%5d nt=%4d %s: default %.3f ms (%.2f TFLOPS), %d solutions\n", P, nt, name, def, flop / def / 1e9, (int) n);
            // the fastest few, in order: timed again, then checked at the bucket's shape and its neighbours
            Row best{which, P, 0, def, def, 0, 0};
            for (size_t k = 0; k < quick.size() && k < 4; ++k) {
                const int32_t sol = quick[k].second;
                const float ms = time_ms(c, which, P, nt, sol, 5);
                if (ms <= 0 || def / ms < min_gain || ms >= best.best_ms) {
                    std::printf("  #%d %.3f ms (%.2fx): not faster\n", (int) sol, ms, ms > 0 ? def / ms : 0.0);
                    continue;
                }
                std::vector<std::pair<int, int>> shapes;
                for (int pp : {P, p_lo + 1, (p_lo + P) / 2 + 3, P - 37, P - 2048 + 8})
                    if (pp > p_lo && pp <= P)
                        for (int tt : {nt_of(pp, floats), nt_of(pp, (size_t) 16 << 20), nt_of(pp, floats) / 2 + 1, 1, 7})
                            shapes.push_back({pp, tt});
                std::sort(shapes.begin(), shapes.end());
                shapes.erase(std::unique(shapes.begin(), shapes.end()), shapes.end());
                bool ok = true;
                int refused = 0, checked = 0;
                double worst_l2 = 0, worst_ulps = 0;
                for (const auto& [pp, tt] : shapes) {
                    fill_inputs(c, pp, tt, 0xC0DE + pp * 7 + tt * 3 + which);
                    rocblas_status sd, sc;
                    const std::vector<uint16_t> r = run_out(c, which, pp, tt, 0, sd);
                    const std::vector<uint16_t> g = run_out(c, which, pp, tt, sol, sc);
                    if (sd != rocblas_status_success) { ok = false; std::printf("  default failed at P=%d nt=%d\n", pp, tt); break; }
                    if (sc != rocblas_status_success) { ++refused; continue; }   // the engine runs the default there
                    ++checked;
                    const Cmp e = compare(r, g);
                    worst_l2 = std::max(worst_l2, e.rel_l2); worst_ulps = std::max(worst_ulps, e.max_ulps);
                    if (!e.ok) {
                        std::printf("  #%d REJECT at P=%d nt=%d: rel_l2 %.2g, %.1f ulps, %zu stray writes, %zu non-finite\n",
                                    (int) sol, pp, tt, e.rel_l2, e.max_ulps, e.stray, e.nonfinite);
                        ok = false;
                        break;
                    }
                }
                if (ok && checked == 0) ok = false;
                std::printf("  #%d %.3f ms (%.2fx, %.2f TFLOPS): %s, %d shapes matched (rel_l2 <= %.2g, <= %.2f ulps of the largest), %d refused\n",
                            (int) sol, ms, def / ms, flop / ms / 1e9, ok ? "ok" : "rejected", checked, worst_l2, worst_ulps,
                            refused);
                if (ok) best = {which, P, sol, def, ms, refused, checked};
            }
            if (best.sol) std::printf("best %s P<=%d: #%d %.3f -> %.3f ms (%.2fx)\n", name, P, (int) best.sol, def, best.best_ms, def / best.best_ms);
            rows.push_back(best);
        }
    }
    if (!out_path.empty()) {
        std::ofstream f(out_path);
        if (!f) { std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 2; }
        f << "# rocBLAS solution indices for q36_engine.cu's prompt-attention GEMMs (STRATA_Q36_ATT_F16=1); this architecture\n";
        f << "# and rocBLAS build only. rows: kind P_hi solution (s: scores K^T Q, v: values V P16; FP16 in / out, FP32\n";
        f << "# compute); a row applies to P in (the previous row's P_hi, P_hi]; `default`: rocBLAS's own pick.\n";
        f << "# Made by tools/hip/tune_attn_rocblas (each solution checked at ~20 shapes per bucket; called with\n";
        f << "# rocblas_gemm_flags_check_solution_index, so a shape a solution cannot run falls back to the default).\n";
        f << "STRATA_Q36_ATT_TUNING_V1 " << arch << " " << ver << "\n";
        int tuned = 0;
        for (int which = 0; which < 2; ++which)
            for (const Row& r : rows) {
                if (r.which != which) continue;
                f << (which ? "v " : "s ") << r.p_hi << " ";
                if (r.sol) { f << r.sol << "   # " << r.def_ms << " -> " << r.best_ms << " ms\n"; ++tuned; }
                else f << "default   # " << r.def_ms << " ms\n";
            }
        std::printf("wrote %zu rows (%d with a solution) to %s\n", rows.size(), tuned, out_path.c_str());
    }
    return 0;
}
