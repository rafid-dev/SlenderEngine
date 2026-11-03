# Strata for 32 GB

**Run Qwen3.8-Flash-Next with a 262K context on a normal 32 GB gaming PC, at 57 tok/s.**

This is a speed-focused fork of [Strata](https://github.com/Niko1221/Strata). Strata streams a model too big for your GPU between VRAM, RAM and your SSD. This fork tunes that for one common kind of PC, and adds an option most PCs can already use: streaming the model from **two SSDs at once**.

Software only: the model, quant and memory use are unchanged.

| Measured, 5 runs each, same day | Strata | This fork |
|---|---|---|
| **Text generation** | 37.6 tok/s | **57.3 tok/s** (+52%) |
| Generating after reading a 26K file | 28.2 tok/s | **45.5 tok/s** (+61%) |
| Follow-up question | 32.1 tok/s | **51.5 tok/s** (+60%) |
| Answering about an image | 24.2 tok/s | **47.0 tok/s** (+94%) |
| Reading a prompt | 989 tok/s | **1,496 tok/s** (+51%) |

Measured on an i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB and two 990 PRO 1 TB, on Windows 11, with Qwen3.8-Flash-Next IQ3_S, `--kv int8` and 262K context. This is the only PC it has been measured on, so your numbers will differ.

## Is this for you?

| | What you need |
|---|---|
| **RAM** | 32 GB. This is what it's tuned for. Other sizes work, but those settings haven't been measured. |
| **GPU** | NVIDIA, ideally 12 GB. The biggest kernel gains are specific to RTX 40 cards. |
| **CPU** | A hybrid Intel CPU with P-cores and E-cores gets the measured thread layout. Other CPUs keep Strata's default layout. |
| **Disk** | About 120 GB for the first download. A second SSD with 52 GB free for dual SSD streaming (optional). |
| **System** | Windows. The Visual Studio 2022 Build Tools (C++ workload), the CUDA toolkit and CMake, because the engine is built from this code. |

This uses most of your PC's memory while it runs. In testing, free RAM dropped to about 2 GB, and 1.5 GB at the lowest, so close your browser and other large programs first.

## Easy setup

```bat
git clone https://github.com/YOUR-USERNAME/Strata.git
cd Strata
FAST-SETUP.bat
```

That's the whole install. On the first run it downloads the model and builds the engine. Then it checks your PC, picks the settings and offers dual SSD streaming. When it finishes, start the model with:

```bat
run-fast.bat
```

Your chat page opens at http://127.0.0.1:8080. Apps can connect there too, because it speaks the OpenAI and Anthropic APIs.

Run `FAST-SETUP.bat` again at any time to change your choices.

## Dual SSD streaming

Most of the model lives on your SSD, and each answer keeps reading from it. If you have a second SSD, keep a copy of the model's largest file (`experts.bin`, 47 GB) there too, and every read of 256 KiB or more comes half from each drive.

**To turn it on:** run `FAST-SETUP.bat` and pick the drive when asked. It copies the file, checks the copy against the original by hash, and turns it on.

| Measured with two 990 PRO, 4 runs each | One SSD | Two SSDs | Gain |
|---|---|---|---|
| Text generation | 42.2 tok/s | 47.4 tok/s | +12% |
| Reading a prompt | 997 tok/s | 1,389 tok/s | +39% |
| Generating after a 26K file | 31.4 tok/s | 37.3 tok/s | +19% |
| Follow-up question | 34.5 tok/s | 41.5 tok/s | +20% |
| Answering about an image | 24.8 tok/s | 31.6 tok/s | +28% |

Those runs are from an earlier stage of this fork, before the later speed-ups. The gain with everything on hasn't been measured separately.

**Good to know:**
- **Which drive:** it must be a different physical SSD, not a second partition of the same one. Setup refuses the same disk, and leaves out hard disks and USB drives.
- **SSD wear:** reading doesn't meaningfully wear an SSD. The copy writes 47 GB once.
- **No second SSD?** Press Enter when asked. Every other change still applies.
- **To turn it off:** run `FAST-SETUP.bat --no-mirror`, then delete the `StrataMirror` folder.
- **If the model changes:** the engine only compares file sizes, so run `FAST-SETUP.bat --verify` after a pack update, and delete the copy if it reports a difference.

**Already set up by hand?** Set this before starting the engine:

```bat
set STRATA_EXPERTS_MIRROR=E:\StrataMirror\experts.bin
```

## Setup options

| Command | What it does |
|---|---|
| `FAST-SETUP.bat --dry-run` | Shows what it would do and changes nothing |
| `FAST-SETUP.bat --yes` | Asks nothing and picks the best second drive |
| `FAST-SETUP.bat --no-mirror` | Uses one drive only |
| `FAST-SETUP.bat --mirror-dir E:\StrataMirror` | Puts the copy in a folder you choose |
| `FAST-SETUP.bat --verify` | Also compares an existing copy by hash |

Setup writes only `strata-fast.json` and `run-fast.bat`, plus the copy if you ask for one. To undo it, delete those. The normal Strata config beside them isn't changed.

## What's different from Strata

- **RTX 40 kernels:** turns on 7 exact kernels this card family had no settings for.
- **No extra copy on SSD reads:** data goes straight into its buffer.
- **Two-SSD reads:** described above.
- **Better thread placement:** workers run on P-cores only, and SSD readers stay alive on E-cores, so ready work computes while reads finish.
- **Parallel RAM moves:** the RAM cache updates during drafting, while nothing else needs it.
- **Deeper drafts:** `--spec 4`.

[FORK.md](FORK.md) lists each change with its switch and files. [The worklog](docs/fork/WORKLOG.md) has every test, including the ones that failed.

## Known limits

- **One PC:** only Windows with CUDA on the PC above has been measured.
- **Fresh install:** the first-run download and build path uses Strata's own setup, and hasn't been run from a clean clone yet.
- **Answers:** `--spec 4` changes wording compared with `--spec 3`, on stock Strata too. In a controlled test, the other speed changes left answers byte-identical on 3 short prompts, but partial page-locking's PCIe path and the RAM moves weren't part of that test. Answers can also vary a little between runs on any engine.
- **Other models:** this is tuned for Qwen3.8-Flash-Next IQ3_S.

## Credit

All of the engine is the work of [Niko1221 and the Strata contributors](https://github.com/Niko1221/Strata). For anything not covered here, such as other models, AMD, Mac, the desktop app or multiple GPUs, see the [original project](https://github.com/Niko1221/Strata) and the [docs](docs/DETAILS.md) in this repo.

MIT licensed, the same as Strata. See [LICENSE](LICENSE).
