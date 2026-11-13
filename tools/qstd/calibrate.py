"""Q-STD calibration (docs/fork/QUALITY.md): the accepted setup's reference (R), its repeat (R2), the prompt-chunk controls
(P1-P3), the window-size sentinel (S1) and the placement stress panel (D1-D3, X1-X2) on the production replay; R, R2 and the
placement panel on the BF16 lane (prompt controls do not touch that lane: it reads everything through verify windows); then the
self-check of the controls. Needs an idle PC; about 2.5-3 h on the reference PC (estimate). Redo it after a setup is adopted.

usage: python tools/qstd/calibrate.py CALRUN [--config cfg.json] [--quick]
  --quick  BF16 lane with R, D1 and X1 only (a later recalibration; the first one should be full)
"""
import os
import subprocess
import sys

here = os.path.dirname(os.path.abspath(__file__))
args = [x for x in sys.argv[1:] if x != "--quick"]
if not args:
    sys.exit(__doc__)
cal, extra = args[0], args[1:]
quick = "--quick" in sys.argv
steps = [["--lane", "sol", "--arms", "R,R2,P1,P2,P3,S1,D1,D2,D3,X1,X2"],
         ["--lane", "bf16", "--arms", "R,D1,X1" if quick else "R,R2,D1,D2,D3,X1,X2"]]
os.environ.setdefault("RUNLOG_Q", "QSTD-CAL")
os.environ.setdefault("RUNLOG_REASON", "Q-STD calibration")
for s in steps:
    r = subprocess.run([sys.executable, os.path.join(here, "run.py"), cal, *s, *extra])
    if r.returncode:
        sys.exit(r.returncode)
subprocess.run([sys.executable, os.path.join(here, "analyze.py"), "--cal", cal, "--calcheck", *extra])
