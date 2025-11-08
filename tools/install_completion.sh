# tools/install_completion.sh
#!/usr/bin/env bash

# ensure argcomplete is installed
python - <<'PY'
import sys, importlib.util
sys.exit(0 if importlib.util.find_spec("argcomplete") else 1)
PY

if [[ $? -ne 0 ]]; then
  echo "argcomplete not installed in this env; install it with: pip install argcomplete"
  exit 1
fi

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" \
         "$CONDA_PREFIX/etc/conda/deactivate.d"

LINE='eval "$(register-python-argcomplete sgl)"'

echo "$LINE" >> "$CONDA_PREFIX/etc/conda/activate.d/argcomplete.sh"
