#!/usr/bin/env bash
# Create (once) the skill's Python environment and print its interpreter path.
# The environment lives in the user cache, keyed by the requirements hash, so
# a changed requirements.txt gets a fresh environment.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
req="$here/requirements.txt"
py="${HEP_PYTHON:-python3}"
"$py" - <<'PY' || { echo "high-end-presentation needs Python 3.12 or newer (set HEP_PYTHON)" >&2; exit 1; }
import sys
sys.exit(0 if sys.version_info >= (3, 12) else 1)
PY
hash="$(shasum -a 256 "$req" | cut -c1-12)"
root="${XDG_CACHE_HOME:-$HOME/.cache}/high-end-presentation"
venv="$root/venv-$hash"
if [ ! -x "$venv/bin/python" ] || [ ! -f "$venv/.complete" ]; then
  rm -rf "$venv"
  mkdir -p "$root"
  "$py" -m venv "$venv" >&2
  "$venv/bin/python" -m pip install --quiet --disable-pip-version-check -r "$req" >&2
  touch "$venv/.complete"
fi
echo "$venv/bin/python"
