# tools/qstd: the Q-STD quality test

The method, the gates and the findings are in [docs/fork/QUALITY.md](../../docs/fork/QUALITY.md). Files:

| File | What |
|---|---|
| `run.py` | Runs arms on a lane (`sol` production replay, `bf16`) and scores the engine's logits live through a named pipe |
| `analyze.py` | One verdict per candidate (`--kind exact/prompt/decode/placement/weights`), or `--calcheck` of the controls |
| `calibrate.py` | The calibration run for an accepted setup |
| `arms.json` | R, R2, the prompt-chunk controls P1-P3, the window sentinel S1, the placement stress panel D/X |
| `qscore.py` | Compact reference, audit rows, sampler TV, BF16 scoring |
| `qcommon.py` | Config, launcher parsing, the engine's serve protocol, cache identity, log facts |
| `pipe.py` | The named-pipe reader (Windows) |
| `fetch_bf16.py` | Fetches the BF16 reference subset and checks every file's SHA-256 |
| `verify_compact.py` | Compact-vs-exact KL on real rows from a logits dump |
| `qstd.example.json` | The config template |
