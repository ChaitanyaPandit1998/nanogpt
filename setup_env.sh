#!/usr/bin/env bash
set -e

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing core packages..."
pip install --upgrade pip
pip install -r requirements.txt

# Detect H100 and install flash-attn only if available
if python3 -c "import torch; assert torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name(0)" 2>/dev/null; then
    echo "H100 detected — installing flash-attn (this takes ~10 min to compile)..."
    pip install -r requirements-gpu.txt
else
    echo "No H100 detected — skipping flash-attn (torch SDPA fallback will be used)"
fi

echo ""
echo "Done. Activate with: source venv/bin/activate"
