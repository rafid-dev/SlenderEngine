# SlenderEngine

**Big models, lean hardware.**

Run **Qwen3.8-Flash-Next IQ3_S** on a Windows PC with **32 GB of RAM**. Use it with coding tools such as OpenCode and Claude Code, or chat in your browser.

*Measured **46–50 tokens/s** with an RTX 4070 Ti 12 GB, 32GB RAM + x2 990 PRO NVMEs. [Test details](#performance).*

Built on [Strata](https://github.com/Niko1221/Strata), SlenderEngine focuses on speed and memory tuning for 32 GB PCs. It uses your GPU, RAM and SSD together, with an optional second SSD for faster reads. Model files are unchanged, and expert skipping is off by default.

![A black hole demo written by the local model](docs/fork/media/blackhole.gif)

Made with Qwen3.8-Flash-Next IQ3_S through OpenCode, running locally on a 32 GB PC with an RTX 4070 Ti. Recorded frame by frame at 4K; playback is smoother than the live demo.

## What you need

- **Windows and an NVIDIA GPU.** Main testing uses an RTX 4070 Ti 12 GB. There is also a [community RTX 4060 8 GB laptop report](docs/fork/LAPTOP-RTX4060.md), with different settings and a modified model pack.
- **32 GB RAM.** For everyday use alongside other apps, choose an **8–10 GiB expert RAM cache**. This is the cache size, not total memory use.
- **A fast SSD with about 140 GB free**, plus space for build tools. A second physical SSD with about 52 GB free is optional.
- **Internet for installation.** Local chat works offline afterwards; connected tools and online features may still use the internet.

For AMD, Linux, Mac or the upstream desktop app, see [Strata](https://github.com/Niko1221/Strata).

## Setup

Install [Git for Windows](https://git-scm.com/download/win) and your NVIDIA graphics driver first.

Open PowerShell in the folder where you want SlenderEngine, then run:

```powershell
git clone --branch speed-fork --single-branch https://github.com/rafid-dev/SlenderEngine.git
cd SlenderEngine
.\FAST-SETUP.bat
```

Use the **`speed-fork` branch**. Setup downloads the model and builds the engine. It offers to install missing build tools, which may need administrator permission. The full first-install path has not yet been tested from a clean clone.

Check the download location before continuing; model files normally go in `Strata-data` beside the source folder. When setup asks:

- Choose **8–10 GiB** for the RAM cache; **10 GiB is the default** on a 32 GB PC.
- Leave **expert skipping off**. Enabling it can be faster, but changes answers.
- Press Enter to use **one SSD**, or select a second physical SSD. Another partition of the same drive does not count.

Once setup finishes, double-click **`run-fast.bat`**, or run:

```powershell
.\run-fast.bat
```

Keep the window open and wait for the model to load. Chat at **[127.0.0.1:8080](http://127.0.0.1:8080)**.

Press **Ctrl+C** in the server window to stop. Next time, just run `run-fast.bat`. To change RAM or SSD settings, stop the server and rerun `FAST-SETUP.bat`.

## Performance

| Task | Measured speed |
|---|---:|
| Generating answers to short coding prompts | **46–50 tokens/s** |
| Generating an answer after a 50,000-token prompt | **36–41 tokens/s** |

Measured November 8–9, 2026: i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB, two Samsung 990 PRO SSDs, Windows 11, IQ3_S, 262K context and a 10 GiB cache. Three runs per workload, thinking enabled, expert skipping off. Speeds include thinking tokens.

These tests used the **optional DFlash2 drafter**, which setup selects if already present. It has a **CC BY-NC-ND license for non-commercial use**. Other hardware or drafting settings may give different results. Answer quality has not been measured with a task suite.

[Full measurements and comparisons →](docs/fork/PERFORMANCE.md)

## Coding tools

Point your tool at the local server:

| API | Base URL |
|---|---|
| OpenAI-compatible | `http://127.0.0.1:8080/v1` |
| Anthropic-compatible | `http://127.0.0.1:8080` |

Use your configured API key if enabled. See [API details](docs/DETAILS.md#using-it) for requests and authentication.

## Help and details

- [Windows build and installation help](docs/SOURCE_INSTALL.md#from-source-windows)
- [What this fork changes](FORK.md)
- [Privacy and network behavior](docs/SAFETY.md)
- [Measurements and test history](docs/fork/PERFORMANCE.md)
- [Report a problem](https://github.com/rafid-dev/SlenderEngine/issues) — include your hardware and the error message.

## Credits

SlenderEngine builds on **[Strata](https://github.com/Niko1221/Strata), by Niko1221 and the Strata contributors**. The engine, server, setup and almost all of the code are theirs. This fork adds speed and memory tuning, and preserves the upstream git history.

Thanks to [Gamination](https://github.com/Gamination) for the 8 GB GPU switches and laptop measurements, and the [Qwen team](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for the model. See [upstream credits](docs/HOW_IT_WORKS.md#credits) for the projects behind Strata.

Code is [MIT licensed](LICENSE), retaining upstream copyright. Model weights and optional components have separate licenses. SlenderEngine is independently maintained; please report fork-specific problems in [this repository](https://github.com/rafid-dev/SlenderEngine/issues).
