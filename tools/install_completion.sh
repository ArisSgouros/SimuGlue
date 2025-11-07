# tools/install_completion.sh
#!/usr/bin/env bash
set -euo pipefail

# detect shell rc file
RC="${HOME}/.bashrc"
if [[ -n "${ZSH_VERSION-}" ]]; then RC="${HOME}/.zshrc"; fi

# ensure argcomplete is installed
python - <<'PY'
import sys, importlib.util
sys.exit(0 if importlib.util.find_spec("argcomplete") else 1)
PY

if [[ $? -ne 0 ]]; then
  echo "argcomplete not installed in this env; install it with: pip install argcomplete"
  exit 1
fi

# zsh needs bashcompinit
if [[ -n "${ZSH_VERSION-}" ]]; then
  grep -qF 'bashcompinit' "$RC" || {
    echo 'autoload -U bashcompinit && bashcompinit' >> "$RC"
    echo "Added bashcompinit to $RC"
  }
fi

# idempotently add the eval line for 'sgl'
LINE='eval "$(register-python-argcomplete sgl)"'
if [[ -n "${ZSH_VERSION-}" ]]; then
  LINE='eval "$(register-python-argcomplete --shell zsh sgl)"'
fi

grep -qF "$LINE" "$RC" || {
  echo "$LINE" >> "$RC"
  echo "Added completion hook to $RC"
}

echo "Done. Restart your shell or run: source \"$RC\""

