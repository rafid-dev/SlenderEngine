#pragma once
// STRATA_AHEAD_TRACE=<path> (measurement only; off when unset): one 32-byte record per step of a predicted or demanded
// SSD read, for memtools/ahead_trace_report.py.  Nothing the engine decides reads anything from here: the hooks copy
// values out and never write engine state, so routing, cache admission and expert computation can't see it.
//
// A hook is one pointer test when off, and the engine holds no other state for it.  On, it takes a clock reading and a bounded lock-free push; a full ring drops the
// record and counts it (kDropped), never blocks.  One low-priority thread writes the ring out every 250 ms, so a killed
// engine loses at most that much.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace strata::core::ahead_trace {

enum Type : uint16_t {
    kLayerBegin = 1,    // the router has decided: layer, expert = tokens, a = window, c = experts per token
    kSubmit = 2,        // the predictor was given this target layer: layer, expert = tokens, a = window
    kPredSkip = 3,      // the predictor was still busy, so no guess for this target layer: layer, a = window
    kPredict = 4,       // a guess: layer, expert, a = window, b = layers ahead - 1, c = best rank among the tokens
    kAheadDrop = 5,     // a guess that won't be read: layer, expert, c = DropReason
    kNeed = 6,          // routed outside VRAM: layer, expert, a = best rank, b = entries, c = NeedTier
    kClaim = 7,         // the layer asked the file tier for it: layer, expert, a = window, c = ClaimState | kClaimReadAhead
    kReadStart = 8,     // a blob's read is about to be issued: layer, expert, a = batch, b = blob bytes, c = Origin | (in RAM tier ? 16 : 0)
    kReadDone = 9,      // published to the layers: layer, expert, a = batch, c = Origin | (ok ? 16 : 0)
    kBatchBegin = 10,   // read_direct: a = batch, layer = blobs, c = Origin | (sampled ? 16 : 0)
    kBatchIssued = 11,  // every request in flight: a = batch, b = requested KiB, layer = requests, expert = drive I/Os
    kBatchEnd = 12,     // read_direct returns: a = batch, c = ok
    kIoDone = 13,       // sampled batches: one drive I/O finished: a = batch, b = KiB, c = drive, expert = I/O index
    kBlobIoDone = 14,   // sampled batches: every I/O of this blob finished, before its batch published it: layer, expert, a = batch
    kWait = 15,         // the decode thread waited in wait_landed: layer, a = ns, t = end of the wait
    kIssue = 16,        // prefetch_begin on the decode thread: layer, expert = blobs pending, a = ns
    kEndWait = 17,      // prefetch_end on the decode thread: layer, a = ns
    kLayerEnd = 18,     // the CPU's share of the layer is done: layer, expert = blobs pending, a = ns since kLayerBegin's body began
    kEvictUnused = 19,  // a read-ahead blob lost its stage buffer without being asked for: layer, expert
    kDropped = 20,      // the writer: a = records dropped so far
    kMeta = 21,         // once: expert = experts per layer, c = format version
};
enum DropReason : uint16_t { kDropQueueFull = 1, kDropLayerPassed = 2, kDropRamTier = 3, kDropOverride = 4, kDropAlreadyStaged = 5, kDropNoBuffer = 6 };
enum NeedTier : uint16_t { kTierRam = 1, kTierFile = 2, kTierSkipped = 3, kTierPcie = 4 };
enum ClaimState : uint16_t { kClaimNone = 0, kClaimPublished = 1, kClaimBusy = 2, kClaimReadAhead = 4 };
enum Origin : uint8_t { kOriginOther = 0, kOriginDecode = 1, kOriginAhead = 2, kOriginDecodeSync = 3 };

constexpr uint32_t kFormatVersion = 1;

struct Event {
    uint64_t t_ns;
    uint64_t a;
    uint32_t b;
    int32_t layer;
    int32_t expert;
    uint16_t type;
    uint16_t c;
};
static_assert(sizeof(Event) == 32, "the report reads fixed 32-byte records");

// Bounded multi-producer, single-consumer queue (Vyukov's sequence numbers).  `n` must be a power of two.
class Ring {
public:
    explicit Ring(size_t n) : mask_(n - 1), slots_(new Slot[n]) {
        for (size_t i = 0; i < n; ++i) slots_[i].seq.store(i, std::memory_order_relaxed);
    }
    bool push(const Event& e) {
        uint64_t pos = head_.load(std::memory_order_relaxed);
        for (;;) {
            Slot& s = slots_[pos & mask_];
            const uint64_t seq = s.seq.load(std::memory_order_acquire);
            const int64_t diff = (int64_t) seq - (int64_t) pos;
            if (diff == 0) {
                if (head_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    s.ev = e;
                    s.seq.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                dropped_.fetch_add(1, std::memory_order_relaxed);
                return false;
            } else {
                pos = head_.load(std::memory_order_relaxed);
            }
        }
    }
    bool pop(Event& e) {   // one thread only
        Slot& s = slots_[tail_ & mask_];
        if (s.seq.load(std::memory_order_acquire) != tail_ + 1) return false;
        e = s.ev;
        s.seq.store(tail_ + mask_ + 1, std::memory_order_release);
        ++tail_;
        return true;
    }
    uint64_t dropped() const { return dropped_.load(std::memory_order_relaxed); }

private:
    struct Slot {
        std::atomic<uint64_t> seq;
        Event ev;
    };
    const uint64_t mask_;
    std::unique_ptr<Slot[]> slots_;
    alignas(64) std::atomic<uint64_t> head_{0};
    alignas(64) uint64_t tail_ = 0;
    alignas(64) std::atomic<uint64_t> dropped_{0};
};

class Trace {
public:
    // nullptr when STRATA_AHEAD_TRACE is unset.  Never freed: a late hook on another thread can still push while the process
    // exits.  The file is made at the first record, so a short-lived strata.exe with the same environment (a version
    // probe) can't truncate a running engine's trace.
    static Trace* open_from_env() {
        const char* path = std::getenv("STRATA_AHEAD_TRACE");
        if (path == nullptr || *path == 0) return nullptr;
        Trace* t = new Trace(path);
        static Trace* the_trace = t;
        std::atexit([] { the_trace->finish(); });
        t->writer_ = std::thread([t] { t->run(); });
        t->writer_.detach();
        return t;
    }

    uint64_t now_ns() const {
        return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0_).count();
    }
    void emit_at(uint64_t t, Type type, int64_t layer, int64_t expert, uint64_t a = 0, uint32_t b = 0, uint16_t c = 0) {
        ring_.push(Event{t, a, b, (int32_t) layer, (int32_t) expert, (uint16_t) type, c});
    }
    void emit(Type type, int64_t layer, int64_t expert, uint64_t a = 0, uint32_t b = 0, uint16_t c = 0) {
        emit_at(now_ns(), type, layer, expert, a, b, c);
    }

    uint64_t next_batch() { return batch_.fetch_add(1, std::memory_order_relaxed) + 1; }
    uint64_t next_window() { return window_.fetch_add(1, std::memory_order_relaxed) + 1; }
    uint64_t window() const { return window_.load(std::memory_order_relaxed); }

    // STRATA_AHEAD_TRACE_HOL=N (default 8, 0 off): every Nth decode or read-ahead batch times each drive I/O apart.
    bool sample_batch() {
        return hol_every_ > 0 && sampled_.fetch_add(1, std::memory_order_relaxed) % hol_every_ == 0;
    }
    // the writer moves itself off the workers' cores once --aux-cpus is known
    void request_pin(void (*fn)()) { pin_fn_.store(fn, std::memory_order_release); }

private:
    explicit Trace(const char* path) : path_(path), ring_(1u << 17), t0_(std::chrono::steady_clock::now()) {
        const char* v = std::getenv("STRATA_AHEAD_TRACE_HOL");
        hol_every_ = v != nullptr ? (uint64_t) std::strtoull(v, nullptr, 10) : 8;
    }
    bool open_file() {
        if (f_ != nullptr) return true;
        if (open_failed_) return false;
        f_ = std::fopen(path_.c_str(), "wb");
        if (f_ == nullptr) {
            open_failed_ = true;
            std::fprintf(stderr, "strata ahead trace: can't write %s: no trace\n", path_.c_str());
            return false;
        }
        static char file_buffer[1 << 20];
        std::setvbuf(f_, file_buffer, _IOFBF, sizeof file_buffer);
        std::fprintf(stderr, "strata ahead trace: writing %s (measurement only)\n", path_.c_str());
        const char magic[8] = {'S', 'A', 'T', 'R', 'A', 'C', 'E', '1'};
        const uint32_t ver = kFormatVersion, rec = (uint32_t) sizeof(Event);
        const uint64_t wall_ms = (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                                     std::chrono::system_clock::now().time_since_epoch()).count();
        const uint64_t zero = 0;
        std::fwrite(magic, 1, 8, f_);
        std::fwrite(&ver, 4, 1, f_);
        std::fwrite(&rec, 4, 1, f_);
        std::fwrite(&wall_ms, 8, 1, f_);
        std::fwrite(&zero, 8, 1, f_);
        return true;
    }
    void drain() {
        Event e;
        bool any = false;
        while (ring_.pop(e))
            if (open_file()) {
                std::fwrite(&e, sizeof e, 1, f_);
                any = true;
            }
        const uint64_t d = ring_.dropped();
        if (d != said_dropped_ && open_file()) {
            said_dropped_ = d;
            const Event de{now_ns(), d, 0, -1, -1, (uint16_t) kDropped, 0};
            std::fwrite(&de, sizeof de, 1, f_);
            any = true;
        }
        if (any) std::fflush(f_);
    }
    void run() {
#if defined(_WIN32)
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
        for (;;) {
            if (void (*fn)() = pin_fn_.exchange(nullptr, std::memory_order_acq_rel)) fn();
            const bool stop = stop_.load(std::memory_order_acquire);
            drain();
            if (stop) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        done_.store(true, std::memory_order_release);
    }
    void finish() {
        stop_.store(true, std::memory_order_release);
        for (int i = 0; i < 400 && !done_.load(std::memory_order_acquire); ++i)
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    const std::string path_;
    std::FILE* f_ = nullptr;
    bool open_failed_ = false;
    Ring ring_;
    const std::chrono::steady_clock::time_point t0_;
    std::atomic<uint64_t> batch_{0}, window_{0}, sampled_{0};
    uint64_t hol_every_ = 8;
    uint64_t said_dropped_ = 0;
    std::atomic<void (*)()> pin_fn_{nullptr};
    std::atomic<bool> stop_{false}, done_{false};
    std::thread writer_;
};

}  // namespace strata::core::ahead_trace
