#!/usr/bin/env bash
# One-click setup of this fork on Linux: installs and builds on the first run, then tunes for a 32 GB PC and offers dual SSD
# streaming. The Linux twin of FAST-SETUP.bat.
set -u
export PYTHONUTF8=1
cd "$(dirname "$0")" || exit 1
if [ ! -f strata-iq3_s.json ]; then
  echo
  echo " First run. This will:"
  echo "   - download Qwen3.8-Flash-Next IQ3_S (allow about 140 GB of disk space)"
  echo "   - build the engine from this folder, which needs the CUDA toolkit, CMake and a C++ compiler"
  echo
  read -r -p " Carry on? [y/N] " ans
  case "$ans" in y|Y|yes|YES) ;; *) exit 1 ;; esac
  ./setup.sh --family qwen --model IQ3_S --build --context 262144 --kv int8 --vision gpu --no-start
  if [ ! -f strata-iq3_s.json ]; then
    echo
    echo " Setup did not finish, so nothing was tuned. See the messages above."
    exit 1
  fi
fi
exec .venv/bin/python tools/fast_setup.py "$@"
