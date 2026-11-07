# Strata for 32 GB

This fork of [Strata](https://github.com/Niko1221/Strata) is tuned for Qwen3.8-Flash-Next IQ3_S with a 262K context on a 32 GB gaming PC. Strata streams a model too big for your GPU between VRAM, RAM and your SSD. The model itself is unchanged.

![A real-time black hole written by this model, running locally on this fork](docs/fork/media/blackhole.gif)

*Written by Qwen3.8-Flash-Next IQ3_S running locally on this fork, through OpenCode, on the PC below. Recorded frame by frame at 4K, so it plays smoother than the page runs live.*

## Updates

### 2026-11-07

- **AMD RX 6600 (8 GB) with 32 GB of RAM: Qwen3.6-35B-A3B now runs well on it.** Built from source with HIP on Windows for gfx1032 (219 of 223 unit tests pass). The 35B engine went from 8K context, 17-18 tok/s decode and 0.5 GB of free RAM to 32K context, 26-27 tok/s decode and 3.5 GB free, with a 24K prompt read at 420 tok/s (540 with `--prefill 4096`), thinking on, temperature 1.0. llama.cpp (Vulkan) on the same PC: 25-28 tok/s and about 500 tok/s. Main changes: the experts' host copy is a mapped file instead of private memory, speculative decoding works for sampled requests too, the KV cache is f16, the prompt's expert stream goes through a pinned ring, and RDNA2 / small-card defaults (`STRATA_DENSE_MMQ=1`, `STRATA_Q8_PACKED=0`). [QWEN36.md](docs/QWEN36.md#an-8-gb-card-with-32-gb-of-ram) explains them, and [the worklog](docs/fork/WORKLOG.md) has every measurement. Branch `amd-rdna2-q36`.

### 2026-11-06

**Method:** the PC under [Speed](#speed), Qwen3.8-Flash-Next IQ3_S at skip 0 (no expert skipped), the model card's sampling. Fresh engine per run, the two settings in mixed order, 3-6 runs each. A change counts when every run of one beats every run of the other. Details are in [FORK.md](FORK.md) and the [worklog](docs/fork/WORKLOG.md).

| Change | Switch | Result | Answers |
|---|---|---|---|
| Skip 0 | `STRATA_ROUTE_TAIL_SKIP=0` | About 22% slower than Strata's default skip 7, but every expert is computed | The full model (skip 7 drops some experts) |
| CPU computes the RAM experts | `--pcie-frac 0` | 7% less time a step, 29.1 → 33.9 tok/s after a 50K prompt | Rounding only |
| K/V lending | `STRATA_KV_LEND=1` | 27% fewer SSD reads, 37.9 → 42.2 tok/s on short answers (+11%) | Byte-identical |
| Each SSD read used as it lands | `STRATA_DIRECT_BATCH=1` | +9% after a 50K prompt, short answers unchanged | Unchanged (same arithmetic) |
| In-place prompt stager | `STRATA_STAGER_IN_PLACE=1` and 3 stager settings | Prompt reading +8.5% at 50K up to +21% on short prompts, CPU 87% → 10% busy | Byte-identical |
| Strata 0.2.11 base with the DFlash2 drafter | `"drafting": "dflash2"` | Follow-ups +5%, 50K prompt reading +11%, thinking-on short answers and 50K level | Same model; sampled text differs per seed |

**Where it stands:** about 45 tok/s on short thinking-on answers, 37 after a 50K prompt, 42 on a follow-up and 52 with thinking off. A 50K prompt reads at about 1,700 tok/s. The speed tables below are still at skip 7.

- Since 2026-11-07 the engine runs these settings by default: skip 0 everywhere, and the table's switches except `--pcie-frac 0` on Windows with an NVIDIA card and an IQ pack. `STRATA_FORK_DEFAULTS=0` gives upstream's defaults. See [FORK.md](FORK.md#defaults).
- `FAST-SETUP.bat` asks how much RAM to use, whether to skip experts (no by default) and which second SSD to use, and picks DFlash2 when its drafter is in `Strata-data\dflash2` (`strata launch --drafting dflash2` downloads it once, 1.2 GB). The stager settings were tuned for two 990 PROs.
- The DFlash2 drafter is licensed CC BY-NC-ND, so it's for non-commercial use.
- The fork's off-by-default switches, including PR #1's for 8 GB cards, are all in the 0.2.11 base. See [FORK.md](FORK.md).

## Speed

The speed and memory tables come from one PC: i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB, two 990 PRO 1 TB, Windows 11 26H2. Your numbers will differ.

This compares each engine at the best settings we tried, with a 10 GiB cache and 3 runs per setting.

| | Strata | This fork, one SSD | This fork, two SSDs |
|---|---:|---:|---:|
| **Text generation** | 30.0 tok/s | **41.8 tok/s** (+39%) | **48.9 tok/s** (+63%) |
| Generating after reading a 26K file | 22.9 tok/s | 33.8 tok/s (+48%) | 43.3 tok/s (+89%) |
| Follow-up question | 23.4 tok/s | 32.0 tok/s (+37%) | 42.2 tok/s (+80%) |
| Answering about an image | 15.2 tok/s | 30.5 tok/s (+101%) | 38.6 tok/s (+154%) |
| Reading a prompt | 945 tok/s | 932 tok/s (-1%) | 1,410 tok/s (+49%) |

Strata's column is the fastest of the 6 setups we tried on it. Its runs came about an hour after the fork's. In a later batch closer to them, the fork with two SSDs ran 45.9 tok/s, which would make that gain +53%.

With the same `--spec`, workers and one SSD, Strata ran 24.9 tok/s against this fork's 41.8 (+68%). That gap flatters the fork, because read overlap and page locking were on for both, and Strata was faster without them.

Both engines use Strata's default expert skip, which leaves out some low-ranked experts that aren't in VRAM. Turning it off took this fork from 45.9 to 35.9 tok/s. The skip also changes answers. On a fixed set of text, the model's top choice stayed the same at 90.7% of positions, from 84% on docs to 97% on code. That shows how far answers move, not whether they get worse.

The fork's columns include read-through, which puts SSD reads straight into the RAM cache. The engine turns it on with any RAM cache. It lost about 19% at 14 GiB on this PC because that size left too little RAM for the driver to lock the cache, not because of the size itself.

Answer quality wasn't measured, and `--spec 4` changes the wording of answers. [FORK.md](FORK.md) explains the changes, settings and measurements. [The worklog](docs/fork/WORKLOG.md) records every test, including the ones that failed.

## How much memory it uses

Setup asks how much RAM to give the model's cache. This sets the cache size, not a limit on total use: about 18 GB total at a 10 GiB cache, or about 22 GB at 14 GiB. VRAM use is about 10.5 GB either way.

| Measured on the PC above | 10 GiB cache (default) | 14 GiB cache |
|---|---:|---:|
| Text generation | 46-49 tok/s | About 50 tok/s |
| Lowest free RAM, 25-minute sessions | 1.8-2.4 GB | Not repeated |
| Written to the page file, 25-minute sessions | 1.5-2.9 GB/h, in short bursts | Not repeated |
| Lowest free RAM, short tests | 1.9-3.6 GB | 0.08-0.94 GB |
| Written to the page file, short tests | Up to about 0.4 GB a run | 0.8-36.2 GB/h |

About 9 GB was already in use before the engine started, including open apps. Closing yours leaves more RAM free. Task Manager's process row shows less than the total because it leaves out about 4 GB of pinned memory.

For running beside a large app, [FORK.md](FORK.md#beside-a-large-app) has settings that ran beside Roblox Studio at about 20 tok/s.

## Is this for you?

| | What you need |
|---|---|
| **RAM** | 32 GB. Close your browser and other large programs for the most speed and the least paging. |
| **GPU** | NVIDIA, ideally 12 GB. The biggest kernel gains are specific to RTX 40 cards. |
| **CPU** | A hybrid Intel CPU gets the measured thread layout. Other CPUs keep Strata's default. |
| **Disk** | About 120 GB for the first download. A second SSD with 52 GB free is optional. |
| **System** | Windows, with the Visual Studio 2022 Build Tools (C++ workload), the CUDA toolkit and CMake. The engine is built from this code. |

## Easy setup

```bat
git clone https://github.com/rafid-dev/Strata-32GB.git
cd Strata-32GB
FAST-SETUP.bat
```

On the first run, setup downloads the model and builds the engine. It checks your PC, asks how much RAM to give the cache, picks the other settings and offers two-SSD reads. The first-run install hasn't been tested from a clean clone yet.

Then start the model:

```bat
run-fast.bat
```

Your chat page opens at http://127.0.0.1:8080. Apps can connect there too, because it speaks the OpenAI and Anthropic APIs. Run `FAST-SETUP.bat` again at any time to change your choices.

| Option | What it does |
|---|---|
| `--dry-run` | Shows what it would do and changes nothing |
| `--yes` | Asks nothing, uses the default RAM and picks a second drive |
| `--ram-gib 12` | Sets the size of the RAM cache without asking |
| `--no-mirror` | Uses one drive only |
| `--mirror-dir E:\StrataMirror` | Puts the SSD copy in a folder you choose |
| `--verify` | Also compares an existing copy by hash. Run it after a pack update, because the engine only compares file sizes |

## Credit

Optional switches for 8 GB cards were contributed by [Gamination](https://github.com/Gamination). See the [RTX 4060 8 GB laptop report](docs/fork/LAPTOP-RTX4060.md). They aren't in the 0.2.11 base yet; they're in the history up to commit `60af6a2a`.

All of the engine is the work of [Niko1221 and the Strata contributors](https://github.com/Niko1221/Strata). For other models, AMD, Mac, the desktop app or multiple GPUs, see the [original project](https://github.com/Niko1221/Strata) and the [docs](docs/DETAILS.md) in this repo.

MIT licensed, the same as Strata. See [LICENSE](LICENSE).
