# Strata for 32 GB

**Run Qwen3.8-Flash-Next IQ3_S with a 262K context on a normal 32 GB gaming PC, at about 48 tok/s with 10 GiB of RAM for the model, or about 55 with 14 GiB.**

![A real-time black hole written by this model, running locally on this fork](docs/fork/media/blackhole.gif)

*Written by Qwen3.8-Flash-Next IQ3_S running locally on this fork, through OpenCode, on the PC below. Recorded frame by frame at 4K, so it plays smoother than the page runs live.*

This is a speed-focused fork of [Strata](https://github.com/Niko1221/Strata). Strata streams a model too big for your GPU between VRAM, RAM and your SSD. This fork tunes that for one common kind of PC, and adds an option most PCs can already use: streaming the model from **two SSDs at once**.

Software only: the model and quant are unchanged.

| Measured at 14 GiB, 5 runs each, same day | Strata | This fork |
|---|---|---|
| **Text generation** | 37.6 tok/s | **57.3 tok/s** (+52%) |
| Generating after reading a 26K file | 28.2 tok/s | **45.5 tok/s** (+61%) |
| Follow-up question | 32.1 tok/s | **51.5 tok/s** (+60%) |
| Answering about an image | 24.2 tok/s | **47.0 tok/s** (+94%) |
| Reading a prompt | 989 tok/s | **1,496 tok/s** (+51%) |

Measured on an i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB and two 990 PRO 1 TB, on Windows 11, with Qwen3.8-Flash-Next IQ3_S, `--kv int8` and 262K context. This is the only PC it has been measured on, so your numbers will differ.

**What this table compares:** Strata 0.2.9 (engine 0.1.55) at the settings it was running here, against this fork at its best settings. So it shows the gain from everything together, not from the code alone. Setup now installs less RAM use than this table, and Strata hasn't been measured at those settings.

| Setting | Strata in the table | This fork in the table | What setup installs |
|---|---|---|---|
| Experts in RAM | 14 GiB | 14 GiB | **10 GiB**, or what you choose |
| Saved conversations in RAM | 2,048 MiB | 2,048 MiB | **Off** |
| Drafting | `--spec 3` | `--spec 4` | `--spec 4` |
| CPU workers | 13 | 7, on P-cores | One per P-core, leaving the first for the host (7 here) |
| SSDs | One | Two | Two if you pick a second drive |
| Context and K/V | 262K, `--kv int8` | 262K, `--kv int8` | 262K, `--kv int8` |

Strata 0.2.10 hasn't been tested. Answer quality wasn't measured, and `--spec 4` changes the wording of answers.

## Is this for you?

| | What you need |
|---|---|
| **RAM** | 32 GB. This is what it's tuned for. Other sizes work, but those settings haven't been measured. |
| **GPU** | NVIDIA, ideally 12 GB. The biggest kernel gains are specific to RTX 40 cards. |
| **CPU** | A hybrid Intel CPU with P-cores and E-cores gets the measured thread layout. Other CPUs keep Strata's default layout. |
| **Disk** | About 120 GB for the first download. A second SSD with 52 GB free for dual SSD streaming (optional). |
| **System** | Windows. The Visual Studio 2022 Build Tools (C++ workload), the CUDA toolkit and CMake, because the engine is built from this code. |

This uses most of your PC's memory while it runs. At the default 10 GiB of experts in RAM, free RAM stayed above about 3.3 GB in testing. At 14 GiB it often dropped to about 1-2 GB, and well under 1 GB at the lowest. Close your browser and other large programs first, because running another large program beside it can make Windows page heavily, which writes to your SSD. [Memory settings](#memory-settings) has the numbers.

## Easy setup

```bat
git clone https://github.com/rafid-dev/Strata-32GB.git
cd Strata-32GB
FAST-SETUP.bat
```

That's the whole install. On the first run it downloads Qwen3.8-Flash-Next IQ3_S and builds the engine. Then it checks your PC, asks how much RAM to give the model, picks the other settings and offers dual SSD streaming. When it finishes, start the model with:

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

## Memory settings

Setup asks how much RAM should hold the model's experts. Press Enter for 10 GiB on a 32 GB PC. The engine uses about 8 GiB on top of that. More RAM is faster, but leaves less for other apps and pages more. Setup also turns off saving conversations in RAM, so switching back to an earlier conversation has to read it again.

| Measured on the PC above | 14 GiB (1 run) | 10 GiB (4 runs) |
|---|---|---|
| Coding answers | 51.5 tok/s | 48.2 tok/s |
| Reading a 26K prompt | 1,491 tok/s | 1,424 tok/s |
| Generating after that prompt | 42.6 tok/s | 34.6 tok/s |
| Follow-up question | 50.5 tok/s | 40.0 tok/s |
| Answering about an image | 40.4 tok/s | 29.4 tok/s |
| Lowest free RAM | 0.7 GiB | 3.46 GiB |
| Written to the page file | 2.5 GB/h | 0.0-0.5 GB/h |

The 14 GiB run came just before the others, so treat those gaps as rough. Paging was measured over about 1.5 minutes.

Run `FAST-SETUP.bat` again to change the amount, or pass `--ram-gib N` to skip the question. Other amounts haven't been measured on their own.

**A real session at 10 GiB, with saving off:** Claude Code reviewing code for 25 minutes, up to 217K tokens, ran at 44.4 tok/s. It wrote 1.0 GB/h to the page file, almost all of it in one burst early on, and free RAM stayed above 3.3 GiB. That's one run, and long sessions run slower than the short tests above.

**Beside a large app:** with Roblox Studio open using about 12 GiB, these settings kept writes to the page file at 0.0-0.1 GB/h (2 runs of about 3.5 minutes): `--max-context 65536 --resident-budget-gib 1 --pool-workers 4 --vram-reserve-mib 1024`, with no `--kv-resident`. Speed fell to about 20 tok/s at short context and 15 tok/s at 50K, and context is limited to 64K. Whether Studio stuttered wasn't measured.

## Setup options

| Command | What it does |
|---|---|
| `FAST-SETUP.bat --dry-run` | Shows what it would do and changes nothing |
| `FAST-SETUP.bat --yes` | Asks nothing, uses the default RAM and picks the best second drive |
| `FAST-SETUP.bat --ram-gib 12` | Sets the GiB of RAM for the experts without asking |
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
