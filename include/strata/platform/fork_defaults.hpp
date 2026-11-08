#pragma once
// The speed-fork's defaults (FORK.md, "Defaults").  main() in generate.cpp applies them before anything reads the
// environment: switches the fork measured as lossless or answer-preserving are set when the caller left them unset, so
// a variable set by hand (or by a config's "env") always wins.  STRATA_FORK_DEFAULTS=0 keeps upstream's defaults.
#include <cstdlib>

namespace strata::platform {

/// STRATA_FORK_DEFAULTS unset or anything but 0: on.
inline bool fork_defaults_on() noexcept {
    const char* v = std::getenv("STRATA_FORK_DEFAULTS");
    return v == nullptr || v[0] != '0';
}

/// Set by main() when the measured case applies (Windows, NVIDIA, an IQ pack: --pack): the defaults that live in code
/// rather than in a variable (the prompt path's lend cap, the stager's thread count) read this.
inline bool g_fork_pack_defaults = false;

/// Set by main() when the fork's defaults, not the caller, turned on STRATA_MTP_ROOM_MIN: a room that cannot apply is then left out
/// with a note instead of stopping the start (set by hand, an unusable room still stops it, as upstream's switch does).
inline bool g_fork_mtp_room = false;

}  // namespace strata::platform
