// STRATA_AHEAD_TRACE's ring: no record lost or torn under several producers, drops counted when full, and what a record costs here.
// No GPU and no engine.  cl /std:c++17 /O2 /EHsc /I include bench\micro\ahead_trace_test.cpp /Fe:ahead_trace_test.exe
#include "strata/core/ahead_trace.hpp"

#include <cstdio>
#include <vector>

using namespace strata::core::ahead_trace;

static int fail(const char* what) {
    std::printf("FAIL: %s\n", what);
    return 1;
}

int main() {
    {   // full: push fails and counts, pop gives back the order
        Ring r(8);
        for (int i = 0; i < 8; ++i) if (!r.push(Event{(uint64_t) i, 0, 0, 0, 0, 0, 0})) return fail("push into a free ring");
        if (r.push(Event{99, 0, 0, 0, 0, 0, 0}) || r.dropped() != 1) return fail("a full ring must drop and count");
        Event e;
        for (uint64_t i = 0; i < 8; ++i) if (!r.pop(e) || e.t_ns != i) return fail("order");
        if (r.pop(e)) return fail("pop from an empty ring");
        if (!r.push(Event{7, 0, 0, 0, 0, 0, 0})) return fail("push after draining");
    }
    {   // 4 producers, 1 consumer: everything pushed arrives once and whole
        constexpr int kThreads = 4;
        constexpr uint64_t kEach = 2000000;
        auto ring = std::make_unique<Ring>(1u << 12);
        std::vector<std::thread> th;
        std::vector<uint64_t> pushed(kThreads, 0);
        for (int t = 0; t < kThreads; ++t)
            th.emplace_back([&, t] {
                for (uint64_t i = 0; i < kEach; ++i)
                    if (ring->push(Event{i, ~i, (uint32_t) t, t, (int32_t) (i & 1023), 1, 2})) ++pushed[(size_t) t];
            });
        std::vector<uint64_t> next(kThreads, 0), got(kThreads, 0);
        std::atomic<bool> done{false};
        std::thread joiner([&] { for (auto& x : th) x.join(); done.store(true); });
        Event e;
        bool bad = false;
        for (;;) {
            const bool finished = done.load();
            bool any = false;
            while (ring->pop(e)) {
                any = true;
                const size_t t = (size_t) e.layer;
                // a producer's records stay in order; a drop only skips ahead.  A torn record fails the cross-checks
                if (t >= kThreads || e.a != ~e.t_ns || e.b != (uint32_t) t || e.expert != (int32_t) (e.t_ns & 1023) || e.type != 1 || e.c != 2 || e.t_ns < next[t]) bad = true;
                else { next[t] = e.t_ns + 1; ++got[t]; }
            }
            if (finished && !any) break;
        }
        joiner.join();
        if (bad) return fail("a torn or reordered record");
        uint64_t total = 0, total_got = 0;
        for (int t = 0; t < kThreads; ++t) {
            total += pushed[(size_t) t];
            total_got += got[(size_t) t];
            if (pushed[(size_t) t] != got[(size_t) t]) return fail("records pushed and popped differ");
        }
        std::printf("stress: %llu records through, %llu dropped of %llu tried, none torn\n", (unsigned long long) total_got,
                    (unsigned long long) ring->dropped(), (unsigned long long) (kEach * kThreads));
    }
    {   // the cost of a record: clock plus push, a drained ring of the engine's size
        auto ring = std::make_unique<Ring>(1u << 17);
        const auto run = [&](int producers, double& ns) {
            constexpr int kBurst = 20000, kBursts = 200;
            std::atomic<bool> stop{false};
            std::thread consumer([&] { Event e; while (!stop.load(std::memory_order_relaxed)) while (ring->pop(e)) {} while (ring->pop(e)) {} });
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> th;
            for (int t = 0; t < producers; ++t)
                th.emplace_back([&] {
                    for (int b = 0; b < kBursts; ++b)
                        for (int i = 0; i < kBurst; ++i) {
                            const uint64_t now = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
                            ring->push(Event{now, 1, 2, 3, 4, 5, 6});
                        }
                });
            for (auto& x : th) x.join();
            const double total = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
            stop.store(true);
            consumer.join();
            ns = total / (double) (kBurst * kBursts);
        };
        const uint64_t d0 = ring->dropped();
        double one = 0, four = 0;
        run(1, one);
        run(4, four);
        std::printf("a record costs %.0f ns alone, %.0f ns with 4 threads pushing flat out; ring %.1f MiB; %llu dropped in this flood\n",
                    one, four, (double) (sizeof(std::atomic<uint64_t>) + sizeof(Event)) * (1u << 17) / 1048576.0, (unsigned long long) (ring->dropped() - d0));
    }
    std::printf("PASS\n");
    return 0;
}
