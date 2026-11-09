// tools/hip/pcie_gather_bench.cpp - how fast the GPU reads scattered KV pages out of pinned host memory.
//
// The 35B engine's sparse decode attention (STRATA_Q36_SPARSE) reads a selection of 32-cell KV pages from host memory
// every step; the selection is made on the GPU, so the copy is either a GPU kernel reading mapped memory (a gather) or
// host-issued DMA after a round trip. This times both for random page sets: a gather kernel with 16-byte loads (many
// blocks in flight), DMA of each page, and one contiguous DMA as the link's ceiling.
//
//   cmake --build build-hip-win --target pcie_gather_bench && build-hip-win\pcie_gather_bench [page_bytes] [pages]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

#define CK(c) do { const hipError_t e = (c); if (e != hipSuccess) { std::fprintf(stderr, "HIP %d: %s\n", __LINE__, hipGetErrorString(e)); std::exit(2); } } while (0)

__global__ void gather(const uint4* __restrict__ src, const int* __restrict__ pages, int page_u4, uint4* __restrict__ dst) {
    const int p = blockIdx.x;                       // one block a page
    const uint4* s = src + (size_t) pages[p] * page_u4;
    uint4* d = dst + (size_t) p * page_u4;
    for (int i = threadIdx.x; i < page_u4; i += blockDim.x) d[i] = s[i];
}

int main(int argc, char** argv) {
    const size_t page = argc > 1 ? (size_t) std::atoll(argv[1]) : 8704;   // 32 cells x (256 B + 16 B of scales)
    const int n_sel = argc > 2 ? std::atoi(argv[2]) : 2048;
    const size_t host_bytes = (size_t) 1 << 30;
    const int n_pages = (int) (host_bytes / page);
    void* h = nullptr;
    CK(hipHostMalloc(&h, host_bytes, hipHostMallocMapped));
    std::memset(h, 1, host_bytes);
    void* hd = nullptr;
    CK(hipHostGetDevicePointer(&hd, h, 0));
    void* d = nullptr;
    CK(hipMalloc(&d, (size_t) n_sel * page));
    std::vector<int> all(n_pages);
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), std::mt19937(5));
    std::vector<int> sel(all.begin(), all.begin() + n_sel);
    std::sort(sel.begin(), sel.end());
    int* dsel = nullptr;
    CK(hipMalloc(&dsel, n_sel * sizeof(int)));
    CK(hipMemcpy(dsel, sel.data(), n_sel * sizeof(int), hipMemcpyHostToDevice));
    hipStream_t st; CK(hipStreamCreate(&st));
    hipEvent_t a, b; CK(hipEventCreate(&a)); CK(hipEventCreate(&b));
    const double mb = (double) n_sel * page / 1e6;
    auto report = [&](const char* what, auto f) {
        f(); CK(hipStreamSynchronize(st));
        CK(hipEventRecord(a, st));
        for (int r = 0; r < 5; ++r) f();
        CK(hipEventRecord(b, st)); CK(hipEventSynchronize(b));
        float ms = 0; CK(hipEventElapsedTime(&ms, a, b)); ms /= 5;
        std::printf("%-34s %8.3f ms  %6.2f GB/s\n", what, ms, mb / ms);
    };
    std::printf("%d random pages of %zu B (%.1f MB) from 1 GiB of pinned host memory\n", n_sel, page, mb);
    for (int thr : {64, 256, 1024})
        report(thr == 64 ? "gather kernel, 64 threads/page" : thr == 256 ? "gather kernel, 256 threads/page" : "gather kernel, 1024 threads/page",
               [&] { gather<<<n_sel, thr, 0, st>>>((const uint4*) hd, dsel, (int) (page / 16), (uint4*) d); });
    report("DMA, one copy a page", [&] {
        for (int i = 0; i < n_sel; ++i)
            CK(hipMemcpyAsync((char*) d + (size_t) i * page, (char*) h + (size_t) sel[i] * page, page, hipMemcpyHostToDevice, st));
    });
    report("DMA, one contiguous copy (ceiling)", [&] { CK(hipMemcpyAsync(d, h, (size_t) n_sel * page, hipMemcpyHostToDevice, st)); });
    return 0;
}
