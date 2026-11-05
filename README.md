# Strata for 32 GB

**Run Qwen3.8-Flash-Next IQ3_S with a 262K context on a normal 32 GB gaming PC, at about 49 tok/s with 10 GiB of RAM for the model, or about 53 with 14 GiB.**

![A real-time black hole written by this model, running locally on this fork](docs/fork/media/blackhole.gif)

*Written by Qwen3.8-Flash-Next IQ3_S running locally on this fork, through OpenCode, on the PC below. Recorded frame by frame at 4K, so it plays smoother than the page runs live.*

This is a speed-focused fork of [Strata](https://github.com/Niko1221/Strata). Strata streams a model too big for your GPU between VRAM, RAM and your SSD. This fork tunes that for one common kind of PC, and adds an option most PCs can already use: streaming the model from **two SSDs at once**.

Software only: the model and quant are unchanged.

| Measured 2026-11-05, same morning | Strata, 14 GiB | This fork, 14 GiB | Strata, 10 GiB | This fork, 10 GiB |
|---|---:|---:|---:|---:|
| **Text generation** | 35.7 tok/s | **53.8 tok/s** (+51%) | 30.0 tok/s | **46.7 tok/s** (+56%) |
| Generating after reading a 26K file | 28.4 tok/s | **44.5 tok/s** (+57%) | 19.7 tok/s | **38.1 tok/s** (+93%) |
| Follow-up question | 28.2 tok/s | **47.4 tok/s** (+68%) | 23.7 tok/s | **40.2 tok/s** (+70%) |
| Answering about an image | 22.0 tok/s | **37.0 tok/s** (+68%) | 16.5 tok/s | **30.3 tok/s** (+83%) |
| Reading a prompt | 1,017 tok/s | **1,484 tok/s** (+46%) | 970 tok/s | **1,432 tok/s** (+48%) |
| Runs | 3 | 5 | 3 | 4 |

Measured on an i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB and two 990 PRO 1 TB, on Windows 11 before the 26H2 update, with Qwen3.8-Flash-Next IQ3_S, `--kv int8` and 262K context. Model card sampling, thinking on, a fresh engine per run. All of these tables come from this one PC, so your numbers will differ. There's also a [report from an RTX 4060 8 GB laptop](docs/fork/LAPTOP-RTX4060.md).

**What this table compares:** Strata 0.2.9 (engine 0.1.55) at the settings it was running here, against this fork at its own settings. So it shows the gain from code, settings and the second SSD together, not from the code alone. **Both engines on the same settings haven't been measured yet.** Read-through, described below, isn't in this table.

| Setting | Strata in the table | This fork in the table | What setup installs |
|---|---|---|---|
| Experts in RAM | 14 or 10 GiB | 14 or 10 GiB | **10 GiB**, or what you choose |
| Saved conversation checkpoints in RAM | Up to 2,048 MiB at 14 GiB, off at 10 | Up to 2,048 MiB at 14 GiB, off at 10 | **Off** |
| Drafting | `--spec 3` | `--spec 4` | `--spec 4` |
| CPU workers | 13 | 7, on P-cores | One per P-core, leaving the first for the host (7 here) |
| SSDs | One | Two | Two if you pick a second drive |
| Read-through | Doesn't have it | Off | **On at 10 GiB or below** |
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

This uses most of your PC's memory while it runs. At the default 10 GiB of experts in RAM, free RAM stayed above about 3.4 GB in testing. At 14 GiB it dropped to under 1 GB, as low as 0.08 GB. Close your browser and other large programs first, because running another large program beside it can make Windows page heavily, which writes to your SSD. [Memory settings](#memory-settings) has the numbers.

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

## Read-through

When the model reads an expert from the SSD, this reads it straight into the RAM cache, in place of that layer's least used expert. The next time it's needed, it's already in RAM. Setup turns it on at 10 GiB of experts or below (`STRATA_RAM_THROUGH=1`). The engine leaves it off unless it's set.

| Windows 26H2, 10 GiB, 3 runs each, mixed order | Off | On | Change |
|---|---:|---:|---:|
| Coding answers | 44.2 tok/s | 49.2 tok/s | +11% |
| Generating after a 50K prompt | 34.0 tok/s | 39.9 tok/s | +17% |
| Follow-up question | 36.7 tok/s | 41.5 tok/s | +13% |
| Whole run | 40.9 tok/s | 46.4 tok/s | **+13%** |
| Experts read from the SSD | 108,837 | 83,899 | -23% |
| Written to the page file | 0-0.8 GB/h | 0 GB/h | |

- **Other batches on this PC at 10 GiB:** +5.2%, +6.0% and +10.9% over the whole run. The gain varies, and "off" here is slower than in the table at the top, for reasons not established. The Windows update and other open apps both changed between them.
- **On the RTX 4060 8 GB laptop:** +30% and +15% on short prompts at 10 GiB, without and with read-ahead. At 16 GiB with the browser closed, +8-13% on short and long-context answers, and about flat on follow-ups. Measured and reported by that laptop's owner.
- **Correctness:** the whole cache was compared with the file up to about 80,000 reads on this PC and about 120,000 on the laptop, and nothing differed. With answers made repeatable on this PC at 10 GiB, the answers were the same with it on and off.
- **Not above 10 GiB here:** at 14 GiB it **lost about 19%** (40.3 against 49.5 tok/s). There the driver can't lock the whole cache, and reads landing in ordinary memory took about twice as long. A first test with ordinary memory at 10 GiB, one run each, was just as slow with plenty of RAM free. So setup leaves it off above 10 GiB.
- **Not measured:** a long real session with it on, and other amounts of RAM on this PC.

To turn it off, delete the `STRATA_RAM_THROUGH` line from `run-fast.bat`.

## Memory settings

Setup asks how much RAM should hold the model's experts. Press Enter for 10 GiB on a 32 GB PC. The engine uses about 8 GiB on top of that. More RAM is faster, but leaves less for other apps and pages more. Setup also turns off saved conversation checkpoints (`--conversation-cache-mib 0`) to save more RAM: up to 2 GiB, the limit they had before. Switching back to an earlier conversation then has to read it again. The checkpoints inside the conversation you're using stay on.

| Measured on the PC above, without read-through | 14 GiB (5 runs) | 10 GiB (4 runs) |
|---|---:|---:|
| Coding answers | 53.8 tok/s | 46.7 tok/s |
| Reading a 26K prompt | 1,484 tok/s | 1,432 tok/s |
| Generating after that prompt | 44.5 tok/s | 38.1 tok/s |
| Follow-up question | 47.4 tok/s | 40.2 tok/s |
| Answering about an image | 37.0 tok/s | 30.3 tok/s |
| Lowest free RAM | 0.08-0.94 GiB | 3.59-4.52 GiB |
| Written to the page file | 0.8-36.2 GB/h | 0 GB/h |

14 GiB also keeps checkpoints on in this table. Paging was measured over about 2 minutes, so treat it as rough. About 9 GiB was already in use before the engine started, open apps included, so closing yours leaves more free.

Run `FAST-SETUP.bat` again to change the amount, or pass `--ram-gib N` to skip the question. Other amounts haven't been measured on their own.

**A real session at 10 GiB, with saving off:** Claude Code reviewing code for 25 minutes, up to 217K tokens, ran at 44.4 tok/s. It wrote 1.0 GB/h to the page file, almost all of it in one burst early on, and free RAM stayed above 3.3 GiB. That's one run, before read-through, and long sessions run slower than the short tests above.

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
- **Read-through:** described above.
- **Deeper drafts:** `--spec 4`.
- **Less RAM by default:** setup asks how much RAM to use, starts at 10 GiB of experts, and turns off saved conversation checkpoints.

[FORK.md](FORK.md) lists each change with its switch and files. [The worklog](docs/fork/WORKLOG.md) has every test, including the ones that failed.

## Known limits

- **One PC:** only Windows with CUDA on the PC above has been measured.
- **Fresh install:** the first-run download and build path uses Strata's own setup, and hasn't been run from a clean clone yet.
- **Answers:** `--spec 4` changes wording compared with `--spec 3`, on stock Strata too. In a controlled test, the other speed changes left answers byte-identical on 3 short prompts, but partial page-locking's PCIe path and the RAM moves weren't part of that test. Answers can also vary a little between runs on any engine.
- **Other models:** this is tuned for Qwen3.8-Flash-Next IQ3_S.

## Credit

All of the engine is the work of [Niko1221 and the Strata contributors](https://github.com/Niko1221/Strata). For anything not covered here, such as other models, AMD, Mac, the desktop app or multiple GPUs, see the [original project](https://github.com/Niko1221/Strata) and the [docs](docs/DETAILS.md) in this repo.

MIT licensed, the same as Strata. See [LICENSE](LICENSE).
