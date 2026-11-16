#!/usr/bin/env bash
# One command from a fresh clone to your team training on the screen:
#
#   bash start.sh
#
# The first run installs everything into .venv/ in this directory, which takes
# a few minutes and needs the network; every later run goes straight to the
# screen. Nothing is installed outside this directory, and nothing asks for an
# administrator's password.
set -euo pipefail

# Everything runs from the repository root, wherever this was started from:
# the commands import engine/, trainer/ and sdk/ from there.
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=.venv
PYTHON="$VENV/bin/python"

say() {
    printf '%s\n' "$*"
}

# Installed when the environment imports what training needs. A marker file
# could claim an install that a later failure, or a deleted package, undid.
installed() {
    [ -x "$PYTHON" ] && "$PYTHON" -c "import torch, numpy, numba" >/dev/null 2>&1
}

need_python() {
    say "SELFPLAYTV needs Python 3.11 or newer, and $1."
    say "  macOS:            install it from https://www.python.org/downloads/ or with: brew install python"
    say "  Debian, Ubuntu:   ask for: sudo apt install python3 python3-venv"
    say "Then run bash start.sh again."
    exit 1
}

install() {
    command -v python3 >/dev/null 2>&1 || need_python "this computer has no python3"
    python3 -c "import sys; sys.exit(sys.version_info < (3, 11))" \
        || need_python "this computer has $(python3 -c 'import platform; print(platform.python_version())')"
    if [ ! -x "$PYTHON" ]; then
        say "Setting up SELFPLAYTV in $(pwd)/$VENV. This takes a few minutes, once."
        rm -rf "$VENV"
        # Debian and Ubuntu ship venv without pip until python3-venv is installed.
        python3 -m venv "$VENV" || { rm -rf "$VENV"; need_python "the python3-venv package"; }
    else
        say "Finishing the installation in $(pwd)/$VENV."
    fi
    # torch first, from the index that fits the machine: the default wheel on
    # Linux is a large CUDA build that a machine without an NVIDIA card cannot use.
    if [ "$(uname -s)" = "Darwin" ]; then
        "$PYTHON" -m pip install torch
    elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    else
        "$PYTHON" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
    fi
    "$PYTHON" -m pip install -r requirements.txt
    # sdk.doctor is not run on success: its speed check compiles the engine and
    # plays for a while, about fifteen seconds on an idle desktop and minutes
    # on a busy one, and the first training start compiles the same engine
    # anyway. It runs when the install did not take, to say what is missing.
    if ! installed; then
        "$PYTHON" -m sdk.doctor || true
        say "The installation is not complete: see the lines above, then run bash start.sh again."
        exit 1
    fi
}

installed || install
exec "$PYTHON" -m sdk.tui
