# Performance and updates

Measured on one PC: i7-14700KF, 32 GB DDR5-6000, RTX 4070 Ti 12 GB, two 990 PRO 1 TB, Windows 11 26H2. Qwen3.8-Flash-Next IQ3_S,
`--kv int8`, 262K context, a 10 GiB RAM cache for the experts. These are results on this PC, not a speed guarantee. Your numbers will
differ.

"Short" is the mean speed of six thinking-on coding answers. "After 50K" is the answer to a question about a 50K-token file, and
"follow-up" is the next question in that conversation. Every test uses a fresh engine, the model card's sampling and alternating runs.

## Where it stands (2026-11-08/09, skip 0)

| | Speed | Runs |
|---|---:|---|
| Short answers | 46.5-49.5 tok/s | 3 |
| After a 50K prompt | 35.5-41.4 tok/s | 3 |
| Follow-up | 34.6-41.8 tok/s | 3 |
| Reading a 50K prompt, this PC's launcher | 2,258-2,296 tok/s | 2 |
| Reading a 50K prompt, the fork's defaults only | 2,064-2,145 tok/s | 2 |
| Lowest free RAM during the decode runs | 3.5-4.4 GB | 3 |
| Written to the page file | 0 GB/h | 3 |

- Skip 0: every expert is computed, so answers are the full model's.
- Decode speed moves with how many drafted tokens are accepted (1.6-1.9 a step in these runs), which is why its ranges are wide.
- "This PC's launcher" adds two settings that aren't fork defaults (see 2026-11-08 below): a 192-slot GPU staging ring and a 95% lend
  cap. The fork's defaults use 96 slots and 90%.
- Decode didn't change in the 2026-11-08/09 work; only prompt reading did.

## Updates

### 2026-11-08 and 2026-11-09

| Change | Switch | Result | Answers |
|---|---|---|---|
| Prompt room, default | `STRATA_DF2_ROOM_*` / `STRATA_MTP_ROOM_*`, set by the fork's defaults when serving | A prompt of 4,096+ new tokens also uses VRAM that is idle while it is read (the drafter's weights and the output head) for its buffers, so it reads in larger chunks. With the fork's defaults: DFlash2 50K 1,910 / 1,889 → 2,145 / 2,064 tok/s (+10%), MTP 1,706 → 1,880 (+10%). With this PC's launcher: DFlash2 2,006 → 2,231 (+11%, 3 runs each), 29K +14%, MTP 1,828 → 2,185 (+19.5%). Shorter prompts unchanged. [How it works](PROMPT-ROOM.md) | Rounding only: first-token KL 0.0022 at 29K, 0.0003 at 50K, same top choice. The borrowed weights come back byte-identical |
| Fused hyper-connection read, default | `STRATA_HC_UPMIX=1` | 50K prompt 2,183 / 2,182 → 2,258 / 2,296 tok/s (+4.3%). NVIDIA sm_80 and newer | Byte-identical |
| Parallel refill, this PC's launcher | `STRATA_REFILL_PINNED=4` (from PR #1) | After a 50K prompt, the cache slots it borrowed are refilled in 405-414 ms instead of 1,048-1,059; after short prompts 70 ms instead of 190 | Byte-identical, 0 slots differing |
| Deeper staging ring, this PC's launcher | `STRATA_PREFILL_RING=192`, `STRATA_PREFILL_LEND_PCT=95` | 50K prompt 1,829-1,836 → 1,956-1,976 tok/s (+7.2%) | Byte-identical |
| Mirror copied in one piece | `FAST-SETUP.bat` | Setup now reserves the second SSD's copy before writing it. The old copy here was in 68,990 pieces and read some experts 3 times slower; replacing it gave no measurable prompt gain | Unchanged |
| Stager settings as defaults | Fork defaults | `STRATA_STAGER_THREADS=3`, `STRATA_STAGER_BATCH=1` are now set as variables, so a pack's file tier no longer switches the stager to its SSD profile | Unchanged |

- The deeper ring helps only where the prompt's chunk stays the same size with it. That held on this PC at a 95% lend cap; on other
  cards it hasn't been measured, so it isn't a default.

### 2026-11-07

| Change | Switch | Result | Answers |
|---|---|---|---|
| Prompts in fewer passes | `STRATA_DENSE_ARENA=1`, `--prefill auto:16384`, a prompt lend cap of 90% | A 50K prompt takes 6 passes over the experts instead of 7: 1,741 → 1,879 tok/s (+7.9%). Prompts whose pass count doesn't drop (29K) read about 2.6% slower | Rounding only: first-token KL 0.0001-0.0024, same top choice |

- Since 2026-11-07 the engine applies the fork's settings by itself. Every build: skip 0 and 0.2.11's two answer-changing prompt rules
  off. On Windows with an NVIDIA card and an IQ pack (`--pack`): the lossless switches below, a 90% lend cap, read-through with a RAM
  cache, and the 16K prompt chunk ceiling on PCs with 28 GB of RAM or more. Not defaults: `--pcie-frac 0`, `--spec`, and the DFlash2
  drafter (setup picks it, see below). `STRATA_FORK_DEFAULTS=0` gives upstream's defaults. See [FORK.md](../../FORK.md#defaults).
- `FAST-SETUP.bat` asks how much RAM to use, whether to skip experts (no by default) and which second SSD to use. It picks DFlash2 when
  its drafter is in `Strata-data\dflash2` (`strata launch --drafting dflash2` downloads it once). The stager settings were tuned for two
  990 PROs.

### 2026-11-06

Skip 0, the model card's sampling, 3-6 runs each. A change counts when every run of one beats every run of the other. Details are in
[FORK.md](../../FORK.md) and the [worklog](WORKLOG.md).

| Change | Switch | Result | Answers |
|---|---|---|---|
| Skip 0 | `STRATA_ROUTE_TAIL_SKIP=0` | About 22% slower than Strata's default skip 7 (45.9 → 35.9 tok/s), but every expert is computed | The full model (skip 7 drops some experts) |
| CPU computes the RAM experts | `--pcie-frac 0` | 7% less time a step; 29.1 → 33.9 tok/s after a 50K prompt (one batch) | Rounding only |
| K/V lending | `STRATA_KV_LEND=1` | 27% fewer SSD reads, 37.9 → 42.2 tok/s on short answers (+11%) | Byte-identical |
| Each SSD read used as it lands | `STRATA_DIRECT_BATCH=1` | +9% after a 50K prompt (35.0 → 38.3 tok/s), follow-ups and short answers within noise | Unchanged (same arithmetic) |
| In-place prompt stager | `STRATA_STAGER_IN_PLACE=1` and 3 stager settings | Prompt reading +8.5% at 50K, +9.2% at 29K, +13.9% at 3K; prompts of 150-480 tokens take 21% less time. CPU 87% → 10% busy | Byte-identical |
| Strata 0.2.11 base with the DFlash2 drafter | `"drafting": "dflash2"` | Follow-ups +5%, 50K prompt reading +11%, thinking-on short answers and after 50K level | Same model; sampled text differs per seed |
| Two 0.2.11 prompt rules off | `STRATA_PREFILL_CPU_SHARE_MAX=1024`, `STRATA_PF_SMALL=0` | ~3K prompts read +5.8% (569 → 602 tok/s), 29K and 50K level | The old fork's arithmetic (the CPU share rule moved answers, KL about 0.011) |

- The DFlash2 drafter is licensed CC BY-NC-ND, so it's for non-commercial use.
- The fork's off-by-default switches, including PR #1's for 8 GB cards, are all in the 0.2.11 base. See [FORK.md](../../FORK.md).

## Against upstream Strata (2026-11-05, skip 7)

This comparison is older than everything above and uses Strata's default expert skip (skip 7) on both engines, which leaves out some
low-ranked experts that aren't in VRAM. It compares each engine at the best settings we tried, with a 10 GiB cache and 3 runs per
setting.

| | Strata | This fork, one SSD | This fork, two SSDs |
|---|---:|---:|---:|
| **Text generation** | 30.0 tok/s | **41.8 tok/s** (+39%) | **48.9 tok/s** (+63%) |
| Generating after reading a 26K file | 22.9 tok/s | 33.8 tok/s (+48%) | 43.3 tok/s (+89%) |
| Follow-up question | 23.4 tok/s | 32.0 tok/s (+37%) | 42.2 tok/s (+80%) |
| Answering about an image | 15.2 tok/s | 30.5 tok/s (+101%) | 38.6 tok/s (+154%) |
| Reading a prompt | 945 tok/s | 932 tok/s (-1%) | 1,410 tok/s (+49%) |

- Strata's column is the fastest of the 6 setups we tried on it. Its runs came about an hour after the fork's. In a later batch closer
  to them, the fork with two SSDs ran 45.9 tok/s, which would make that gain +53%.
- With the same `--spec`, workers and one SSD, Strata ran 24.9 tok/s against this fork's 41.8 (+68%). That gap flatters the fork,
  because read overlap and page locking were on for both, and Strata was faster without them.
- Turning the skip off took this fork from 45.9 to 35.9 tok/s. The skip also changes answers: on a fixed set of text, the model's top
  choice stayed the same at 90.7% of positions, from 84% on docs to 97% on code. That shows how far answers move, not whether they get
  worse.
- The fork's columns include read-through, which puts SSD reads straight into the RAM cache. It lost about 19% with a 14 GiB cache on
  this PC because that size left too little RAM for the driver to lock the cache, not because of the size itself.
- Answer quality wasn't measured with a task suite, and `--spec 4` changes the wording of answers. Upstream Strata hasn't been compared
  at skip 0 or with the 2026-11-06 to 11-09 changes.

## How much memory it uses

Setup asks how much RAM to give the model's expert cache. This sets the cache size, not a limit on total use. On the GPU, the engine
sizes its expert cache to fill the card and keeps about 0.7 GB free.

| Measured on the PC above | 10 GiB cache (default) | 14 GiB cache |
|---|---:|---:|
| Lowest free RAM, today's setup (2026-11-08/09, 3 test runs) | 3.5-4.4 GB | Not measured |
| Written to the page file, the same runs | 0 GB/h | Not measured |
| Lowest free RAM, 2026-11-05 short tests | 3.6-4.5 GiB | 0.08-0.94 GiB |
| Written to the page file, 2026-11-05 short tests | 0 GB/h | 0.8-36.2 GB/h |
| Text generation, 2026-11-05 (skip 7) | 44.3-48.0 tok/s | 50.4-56.2 tok/s |

- About 9 GB was already in use before the engine started, including open apps. Closing yours leaves more RAM free.
- Task Manager's process row shows less than the total because it leaves out the engine's pinned memory (about 3-4 GB here, mostly
  the K/V cache).
- 14 GiB was faster but left too little RAM on a 32 GB PC and paged heavily, so 10 GiB is the setting to recommend.
- The prompt room keeps a RAM copy of the drafter's weights after the first long prompt: about 400 MiB for DFlash2, 675 MiB (pinned)
  for MTP.
- For running beside a large app, [FORK.md](../../FORK.md#beside-a-large-app) has settings that ran beside Roblox Studio at about
  20 tok/s.
