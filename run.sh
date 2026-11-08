#!/bin/zsh
# Self-healing runner for eagle_status.py. Uses a private venv (.venv) so
# Homebrew / system Python upgrades can never remove matplotlib again.
# If the venv is missing or broken, it is rebuilt automatically.
DIR="${0:A:h}"
VENV="$DIR/.venv"
PY="$VENV/bin/python"

if ! "$PY" -c "import matplotlib" >/dev/null 2>&1; then
    echo "eagle-status: (re)building Python environment, one moment..."
    BASE=""
    for c in /Library/Frameworks/Python.framework/Versions/Current/bin/python3 \
             /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 \
             /opt/homebrew/bin/python3 /usr/local/bin/python3 /usr/bin/python3; do
        [ -x "$c" ] && "$c" -c "import venv, ensurepip" >/dev/null 2>&1 && { BASE="$c"; break; }
    done
    [ -z "$BASE" ] && { echo "eagle-status: no usable python3 found"; exit 1; }
    rm -rf "$VENV"
    "$BASE" -m venv "$VENV" && "$VENV/bin/pip" install -q --disable-pip-version-check matplotlib \
        || { echo "eagle-status: failed to install matplotlib"; exit 1; }
fi
exec "$PY" "$DIR/eagle_status.py" "$@"
