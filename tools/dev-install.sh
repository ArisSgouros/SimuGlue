#!/usr/bin/env bash
set -euo pipefail
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,cli]"
python - <<'PY'
import simuglue, sys
print("Installed:", simuglue.__file__)
print("Python   :", sys.executable)
PY

