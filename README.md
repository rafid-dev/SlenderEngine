# SlenderEngine

**Big models, lean hardware.**

Run **Qwen3.8-Flash-Next IQ3_S locally on a Windows PC with 32 GB of RAM**. SlenderEngine is a speed fork of [Strata](https://github.com/Niko1221/Strata) that tunes how the model uses your GPU, RAM and SSD, with an optional second SSD for faster reads. The model files are unchanged.

**Start here:** [Requirements](#what-you-need) · [Setup](#easy-setup) · [Troubleshooting](#troubleshooting) · [Measurements](docs/fork/PERFORMANCE.md)

## What you need

| Part | Requirement for this setup |
|---|---|
| System | Windows. `FAST-SETUP.bat` is for Windows with an NVIDIA GPU. |
| RAM | 32 GB. Close other large apps before running the model. |
| GPU | NVIDIA; measured on an RTX 4070 Ti with 12 GB of VRAM. Other cards may perform differently. |
| Storage | About 120 GB free on an SSD for model download and preparation, plus room for build tools. A second physical SSD with about 52 GB free is optional. |
| Downloads | Internet access for the first setup. The model download is tens of GB. |

Install [Git for Windows](https://git-scm.com/download/win) and your NVIDIA graphics driver first. Setup can install Python, CMake and Ninja, and offers to install Visual Studio 2022 Build Tools and the CUDA toolkit if they are missing. Installing build tools may ask for Windows administrator permission.

This guide builds the fork from source. For an AMD RX 6600 (8 GB, gfx1032) with 32 GB of RAM on Windows, see [AMD_HIP.md](docs/AMD_HIP.md) for the HIP build and [QWEN36.md](docs/QWEN36.md#an-8-gb-card-with-32-gb-of-ram) for Qwen3.6-35B-A3B on that card. For other AMD cards, Mac, Linux or the upstream desktop app, use the [original Strata guides](https://github.com/Niko1221/Strata).

## Easy setup

### 1. Download this fork

Open a folder on the SSD with enough space. Right-click inside it and choose **Open in Terminal** (PowerShell), then paste:

```powershell
git clone --branch speed-fork --single-branch https://github.com/rafid-dev/SlenderEngine.git
cd SlenderEngine
.\FAST-SETUP.bat
```

Use the **`speed-fork` branch** shown above. Cloning without `--branch speed-fork` gets the repository's `main` branch instead.

### 2. Follow setup's questions

Keep the window open while it downloads the model and builds the engine from this folder. Accept the build-tool installation if offered. Model files normally go in `Strata-data` beside the source folder; check the storage location setup shows before downloading.

When tuning asks, these are good starting choices:

| Question | Choose |
|---|---|
| How much RAM for the cache? | **Press Enter for 10 GiB** on a 32 GB PC. This is the model's RAM cache, not its total memory use. |
| Skip experts? | **Press Enter for No.** Skipping can be faster, but changes answers. |
| Which second SSD? | **Press Enter to use one SSD**, or choose a listed second physical SSD. Setup copies and verifies the model's largest file there. |

Setup creates **`run-fast.bat`** and **`strata-fast.json`**. The first install needs a large download and a local compile; the full path has **not yet been tested from a clean clone**.

### 3. Start chatting

Double-click **`run-fast.bat`**, or run this in the same terminal:

```powershell
.\run-fast.bat
```

Keep its window open. The browser opens at [http://127.0.0.1:8080](http://127.0.0.1:8080); wait for the model to finish loading, then send a message.

To stop, press **Ctrl+C** in the server window. Next time, just run `run-fast.bat` again. To change your RAM or SSD choices, stop the server and rerun `FAST-SETUP.bat`.

## Speed and memory

With expert skipping off, the 2026-11-07 tests measured about **46 tokens/s on short answers** and **39 tokens/s after a 50K-token prompt**, with thinking enabled. These results are from one PC: i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB, two Samsung 990 PRO SSDs, Windows 11, IQ3_S and 262K context. Your speed will differ.

At the default 10 GiB cache, the model used about **18 GB of system RAM in total** and **10.5 GB of VRAM** in the recorded tests. Giving it more cache can leave too little memory for Windows and slow it down. Start at 10 GiB and close other large apps.

See [performance, memory measurements and updates](docs/fork/PERFORMANCE.md) for the test conditions and older comparisons. Answer quality has not been measured with a task suite.

## Troubleshooting

| Problem | What to do |
|---|---|
| `git` is not recognized | Install Git for Windows, then open a new terminal. |
| PowerShell cannot find the batch file | Make sure you are inside `SlenderEngine` and include `.\`, as in `.\FAST-SETUP.bat`. |
| `FAST-SETUP.bat` is missing | Check `git branch --show-current`. This guide needs `speed-fork`; download it with the command above. |
| Python could not be installed | Install [64-bit Python 3.12](https://www.python.org/downloads/windows/), enable **Add python.exe to PATH**, and rerun setup. |
| Build tools or CUDA could not be installed | See the manual installation steps below, then rerun setup. |
| Setup reports a stock or old engine | Rebuild with the command below, then rerun `FAST-SETUP.bat`. |
| The chat page does not open | Keep the server window open, check it for errors, and open [the local page](http://127.0.0.1:8080) yourself. |
| Port 8080 is already in use | Stop the other server, or run `.\run-fast.bat --port 8081` and use [port 8081](http://127.0.0.1:8081). |
| The PC slows down or runs out of memory | Close other large apps and rerun setup with a smaller RAM cache. More cache is not always faster. |

<details>
<summary>Manual build tools and engine rebuild</summary>

If the automatic tool installation fails:

1. Download [Build Tools for Visual Studio 2022](https://aka.ms/vs/17/release/vs_buildtools.exe). In the installer, select **Desktop development with C++**, including the MSVC compiler and Windows SDK.
2. Install **CUDA Toolkit 13.0** from the [NVIDIA archive](https://developer.nvidia.com/cuda-toolkit-archive). The toolkit includes the compiler; a graphics driver alone is not enough. This matches the version setup tries to install.
3. Open a new terminal in the source folder and run:

```powershell
.\START-HERE.bat --family qwen --model IQ3_S --build --context 262144 --kv int8 --vision gpu --no-start
.\FAST-SETUP.bat
```

Setup installs CMake and Ninja if needed. More compiler details: [source installation](docs/SOURCE_INSTALL.md#from-source-windows).

</details>

<details>
<summary>Optional settings, second SSD and app connections</summary>

Run these from PowerShell in the source folder **after the first install**:

| Command | What it does |
|---|---|
| `.\FAST-SETUP.bat --ram-gib 12` | Sets a 12 GiB cache without asking for its size. |
| `.\FAST-SETUP.bat --skip 0` | Keeps expert skipping off. |
| `.\FAST-SETUP.bat --no-mirror` | Turns off second-SSD reads. The existing copy stays on disk. |
| `.\FAST-SETUP.bat --mirror-dir E:\StrataMirror` | Uses a folder on a different physical SSD. Replace `E:` with your drive. |
| `.\FAST-SETUP.bat --verify` | Checks an existing SSD copy by hash. Run this after updating model files; the engine only checks file sizes. |
| `.\FAST-SETUP.bat --dry-run --yes --no-mirror` | Previews tuning without writing files. |
| `.\FAST-SETUP.bat --yes` | Accepts tuning defaults and automatically selects a second drive if available. |

**On a fresh install, `--dry-run` and `--yes` do not apply to the initial download/build stage.** They are passed to the tuning script afterwards.

A second SSD must be a different physical drive, not another partition of the same SSD. A new copy of `experts.bin` is about 47 GB and is verified by hash. You can keep using one SSD without it.

For apps using the local API:

| API | Base URL |
|---|---|
| OpenAI-compatible | `http://127.0.0.1:8080/v1` |
| Anthropic-compatible | `http://127.0.0.1:8080` |

Use your configured API key if enabled. See [API details](docs/DETAILS.md#using-it) for authentication and requests.

If a DFlash2 drafter is already present in the data folder's `dflash2` directory, tuning selects it automatically. That optional drafter has a **CC BY-NC-ND license for non-commercial use**. It is not required for the basic setup.

</details>

<details>
<summary>Example: a black hole demo made with the local model</summary>

![A real-time black hole written by this model, running locally on this fork](docs/fork/media/blackhole.gif)

Written by Qwen3.8-Flash-Next IQ3_S through OpenCode on the PC described above. Recorded frame by frame at 4K, so playback is smoother than the live page.

</details>

## More details

- [FORK.md](FORK.md): engine changes, switches and measurements.
- [Worklog](docs/fork/WORKLOG.md): recorded tests, including failed experiments.
- [The prompt room](docs/fork/PROMPT-ROOM.md): how long prompts read in fewer passes, and how it was found and checked.
- [RTX 4060 8 GB laptop report](docs/fork/LAPTOP-RTX4060.md): optional switches contributed by [Gamination](https://github.com/Gamination).

## Credits

SlenderEngine is [Strata](https://github.com/Niko1221/Strata), by Niko1221 and the Strata contributors, with speed changes for 32 GB PCs on top. The engine, the server, the setup and almost all of the code are theirs.

- **Where it started.** This repository began on 2026-11-03 as a GitHub fork of Niko1221/Strata, from commit `c62d5e3` (engine 0.1.55), under the name Strata-32GB. On 2026-11-06 it moved to upstream `v0.2.11` (engine 0.1.57). It has since left GitHub's fork network and was renamed SlenderEngine; the full git history, including every upstream commit, is kept.
- **Still Strata underneath.** The program is still `strata`, its switches are still `STRATA_*` and its config files are still `strata-*.json`, so Strata's documentation applies. [FORK.md](FORK.md) lists what this fork changes, and [compare with upstream v0.2.11](https://github.com/rafid-dev/SlenderEngine/compare/c3b3f50...speed-fork) shows every changed line.
- **Upstream's credits** for the model, llama.cpp / ggml and the projects Strata learned from are in [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md#credits).
- **Contributors to this fork:** [Gamination](https://github.com/Gamination), the 8 GB card switches and the RTX 4060 laptop measurements.
- **The model:** [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team.

Code is [MIT licensed](LICENSE), keeping Niko1221's copyright alongside this fork's; model weights and optional components have their own licenses. SlenderEngine is not made or endorsed by the Strata project. Problems with this fork belong in [this repository's issues](https://github.com/rafid-dev/SlenderEngine/issues), not upstream's.
