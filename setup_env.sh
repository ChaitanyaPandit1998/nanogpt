#!/usr/bin/env bash
set -e

echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing packages (PyTorch 2.9.1 + CUDA 12.8)..."
pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

echo ""
echo "Done. Flash Attention 3 will be downloaded automatically on first GPU use."
echo "Activate with: source venv/bin/activate"
