# Strata for 32 GB

**Run Qwen3.8-Flash-Next IQ3_S with a 262K context on a normal 32 GB gaming PC at about 49 tok/s, using about 18 GB of RAM. Or about 53 tok/s, using about 22 GB.**

![A real-time black hole written by this model, running locally on this fork](docs/fork/media/blackhole.gif)

*Written by Qwen3.8-Flash-Next IQ3_S running locally on this fork, through OpenCode, on the PC below. Recorded frame by frame at 4K, so it plays smoother than the page runs live.*

This is a speed-focused fork of [Strata](https://github.com/Niko1221/Strata), which streams a model too big for your GPU between VRAM, RAM and your SSD. Software only: the model and quant are unchanged.

| At the default memory setting | Strata | This fork |
|---|---:|---:|
| **Text generation** | 30.0 tok/s | **46.7 tok/s** (+56%) |
| Generating after reading a 26K file | 19.7 tok/s | **38.1 tok/s** (+93%) |
| Follow-up question | 23.7 tok/s | **40.2 tok/s** (+70%) |
| Answering about an image | 16.5 tok/s | **30.3 tok/s** (+83%) |
| Reading a prompt | 970 tok/s | **1,432 tok/s** (+48%) |

- **Read-through adds more on top:** it's the newest change and isn't in this table. In separate runs it raised text generation from 44.2 to 49.2 tok/s (+11%).
- **The gain isn't from code alone:** each engine ran at its own settings, and this fork also had a second SSD. Both engines on the same settings haven't been measured yet. [FORK.md](FORK.md) has the details.
- **One PC:** i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB, two 990 PRO 1 TB, Windows 11. Your numbers will differ. There's also a [report from an RTX 4060 8 GB laptop](docs/fork/LAPTOP-RTX4060.md).
- **Answers:** quality wasn't measured, and `--spec 4` changes the wording of answers.

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

On the first run it downloads the model and builds the engine. Then it checks your PC, asks how much RAM to give the model's cache, picks the other settings and offers two-SSD reads. Then start the model:

```bat
run-fast.bat
```

Your chat page opens at http://127.0.0.1:8080. Apps can connect there too, because it speaks the OpenAI and Anthropic APIs. Run `FAST-SETUP.bat` again at any time to change your choices.

| Option | What it does |
|---|---|
| `--dry-run` | Shows what it would do and changes nothing |
| `--yes` | Asks nothing, uses the default RAM and picks the best second drive |
| `--ram-gib 12` | Sets the size of the RAM cache without asking |
| `--no-mirror` | Uses one drive only |
| `--mirror-dir E:\StrataMirror` | Puts the SSD copy in a folder you choose |
| `--verify` | Also compares an existing copy by hash. Run it after a pack update, because the engine only compares file sizes |

Setup writes only `strata-fast.json` and `run-fast.bat`, plus the copy if you ask for one. To undo it, delete those.

## How much memory it uses

Setup's RAM question sets the size of the model's **cache**, not a limit on total use. The engine uses about 8 GB more on top. VRAM use is about 10.5 GB either way.

| Measured on the PC above | 10 GiB cache (default) | 14 GiB cache |
|---|---:|---:|
| **RAM Strata really uses** | **about 18 GB** | **about 22 GB** |
| Text generation | 46.7 tok/s, or about 49 with read-through | 53.8 tok/s |
| Lowest free RAM | 3.6-4.5 GB | 0.08-0.94 GB |
| Written to the page file | 0 GB/h | 0.8-36.2 GB/h |

- **Task Manager shows less than this,** because about 4 GB of it is pinned memory that doesn't appear in the process's row.
- **Those free RAM numbers had about 9 GB in use before the engine started,** open apps included. Closing yours leaves more free.
- **Real use:** Claude Code reviewing code for 25 minutes on the default setting, up to 217K tokens, ran at 44.4 tok/s. That's one run, before read-through.
- **Beside a large app:** [FORK.md](FORK.md) has low-memory settings that ran beside Roblox Studio at about 20 tok/s.

## What's different from Strata

- **RTX 40 kernels:** turns on 7 exact kernels this card family had no settings for.
- **No extra copy on SSD reads:** data goes straight into its buffer.
- **Two-SSD reads:** with a copy of the model's largest file on a second SSD, each large read comes half from each drive. In earlier runs that gave +12% on text and +39% on prompts. It must be a different physical drive, and reading doesn't meaningfully wear an SSD.
- **Read-through:** a read from the SSD lands straight in the RAM cache, so it's there next time. Setup turns it on at a 10 GiB cache or below. It lost about 19% at 14 GiB on this PC, so it stays off above that.
- **Better thread placement:** workers run on P-cores only, and SSD readers stay alive on E-cores.
- **Parallel RAM moves:** the RAM cache updates during drafting, while nothing else needs it.
- **Deeper drafts:** `--spec 4`.
- **Less RAM by default:** a 10 GiB cache, with saved conversation checkpoints off. Switching back to an earlier conversation then has to read it again.
- **Extra switches for 8 GB cards:** off by default, contributed by [Gamination](https://github.com/Gamination).

[FORK.md](FORK.md) lists each change with its switch, files and numbers. [The worklog](docs/fork/WORKLOG.md) has every test, including the ones that failed.

## Known limits

- **One PC:** the numbers above are from Windows with CUDA on one PC. Strata 0.2.10 hasn't been tested.
- **Fresh install:** the first-run download and build hasn't been run from a clean clone yet.
- **Other models:** this is tuned for Qwen3.8-Flash-Next IQ3_S.

## Credit

All of the engine is the work of [Niko1221 and the Strata contributors](https://github.com/Niko1221/Strata). For other models, AMD, Mac, the desktop app or multiple GPUs, see the [original project](https://github.com/Niko1221/Strata) and the [docs](docs/DETAILS.md) in this repo.

MIT licensed, the same as Strata. See [LICENSE](LICENSE).
