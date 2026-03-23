#!/usr/bin/env bash
# Install project and deps. torch-sparse and torch_scatter require torch at build
# time, so we install them with --no-build-isolation after torch.
# wheel must be installed before them to avoid the legacy setup.py install deprecation.
set -e
cd "$(dirname "$0")/.."
echo "Step 1/4: Installing torch..."
pip install torch
echo "Step 2/4: Installing wheel (avoids legacy setup.py deprecation for torch_scatter)..."
pip install wheel
echo "Step 3/4: Installing torch-sparse and torch_scatter (no build isolation)..."
pip install torch-sparse torch_scatter --no-build-isolation
echo "Step 4/4: Installing project in editable mode..."
pip install -e .
