#!/usr/bin/env bash
# runpod_start.sh
# Run once after each pod start to set up or reuse the persistent venv.
# The venv lives on /workspace (network volume) and survives pod restarts.
#
# Usage:
#   bash /workspace/nanogpt/runpod_start.sh
#   source /workspace/venv/bin/activate

set -e

VENV=/workspace/venv
REPO=/workspace/nanogpt

echo "=== RunPod startup ==="

if [ ! -d "$VENV" ]; then
    echo "Creating venv at $VENV (first time only)..."
    python3 -m venv "$VENV"
    echo "Installing packages..."
    "$VENV/bin/pip" install --upgrade pip
    "$VENV/bin/pip" install -r "$REPO/requirements.txt" \
        --extra-index-url https://download.pytorch.org/whl/cu128
    echo "Packages installed successfully."
else
    echo "Venv already exists at $VENV — skipping install."
fi

echo ""
echo "Run this to activate:  source /workspace/venv/bin/activate"
