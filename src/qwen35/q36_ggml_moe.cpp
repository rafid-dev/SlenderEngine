// src/qwen35/q36_ggml_moe.cpp - SP1 (Q36-S): the q36 engine's prompt expert GEMMs on a current llama.cpp's ggml (MUL_MAT_ID,
// SwiGLU, MUL_MAT_ID), as a DLL of its own: the engine's vendored MMQ is an older llama.cpp's and the current one ran
// 1.4-2.6x faster on the engine's shapes (test-backend-ops, QUESTIONS.md "Q36-S results, SP0 + SD0"). Built against that
// tree's ggml (its own CUDA runtime, its own stream), loaded by q36_engine.cu only with STRATA_Q36_GGML_MOE=<this dll>.
// The tensors point at the engine's device buffers (a dummy CUDA buffer marks them as device memory); the caller
// synchronizes its stream before a call, and the call returns after ggml's compute has finished.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

struct Q36MoeArgs {
    int version;               // 1
    int type_gu, type_d;       // ggml types of the gate+up (2 n_ff rows, gate first) and the down weights
    const void* w_gu;          // expert 0's gate+up rows; expert e at w_gu + e * expert_stride
    const void* w_d;           // expert 0's down rows
    size_t gu_row_bytes, d_row_bytes, expert_stride;
    int n_expert, n_embd, n_ff, k_used;
    int64_t n_tok;
    const float* x;            // [n_tok][n_embd]
    const int32_t* ids;        // [n_tok][k_used]
    float* gu_out;             // [n_tok][k_used][2 n_ff] scratch
    float* h_out;              // [n_tok][k_used][n_ff] scratch
    float* out;                // [n_tok][k_used][n_embd]
};

namespace {
std::mutex g_mu;
ggml_backend_t g_be = nullptr;
ggml_backend_buffer_t g_buf = nullptr;   // a 256-B CUDA buffer: the tensors' buffer (their data are the engine's)
int g_dev = -1;

void place(ggml_tensor* t, void* p) {
    t->buffer = g_buf;
    t->data = p;
}
}  // namespace

extern "C" __declspec(dllexport) int q36moe_init(int device) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_be) return 0;
    _putenv("GGML_CUDA_DISABLE_GRAPHS=1");   // the tensors' addresses change every call
    g_be = ggml_backend_cuda_init(device);
    if (!g_be) return 1;
    g_buf = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(device), 256);
    if (!g_buf) return 2;
    g_dev = device;
    return 0;
}

extern "C" __declspec(dllexport) int q36moe_run(const Q36MoeArgs* a) {
    if (!g_be || !a || a->version != 1) return 1;
    std::lock_guard<std::mutex> lk(g_mu);
    ggml_init_params ip = {16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true};
    ggml_context* ctx = ggml_init(ip);
    if (!ctx) return 2;
    const int64_t N = a->n_embd, F = a->n_ff, E = a->n_expert, K = a->k_used, T = a->n_tok;
    ggml_tensor* wgu = ggml_new_tensor_3d(ctx, (ggml_type) a->type_gu, N, 2 * F, E);
    wgu->nb[1] = a->gu_row_bytes ? a->gu_row_bytes : ggml_row_size((ggml_type) a->type_gu, N);
    wgu->nb[2] = a->expert_stride;
    wgu->nb[3] = a->expert_stride * E;
    ggml_tensor* wd = ggml_new_tensor_3d(ctx, (ggml_type) a->type_d, F, N, E);
    wd->nb[1] = a->d_row_bytes ? a->d_row_bytes : ggml_row_size((ggml_type) a->type_d, F);
    wd->nb[2] = a->expert_stride;
    wd->nb[3] = a->expert_stride * E;
    ggml_tensor* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, N, 1, T);
    ggml_tensor* ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, K, T);
    ggml_tensor* gu = ggml_mul_mat_id(ctx, wgu, x, ids);   // [2F, K, T]
    ggml_tensor* h = ggml_swiglu(ctx, gu);                  // [F, K, T]: silu(first half) * second half
    ggml_tensor* y = ggml_mul_mat_id(ctx, wd, h, ids);      // [N, K, T]
    place(wgu, (void*) a->w_gu);
    place(wd, (void*) a->w_d);
    place(x, (void*) a->x);
    place(ids, (void*) a->ids);
    place(gu, a->gu_out);
    place(h, a->h_out);
    place(y, a->out);
    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    const ggml_status st = ggml_backend_graph_compute(g_be, gf);
    ggml_free(ctx);
    return st == GGML_STATUS_SUCCESS ? 0 : 3;
}
