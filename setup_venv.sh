#!/bin/bash
# Setup script for VidEoMT virtual environment
# Creates a venv with PyTorch 2.7 + Detectron2 + all dependencies
# Target: Python 3.12, CUDA 12.x (cu126 compatible)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_CMD="python3.12"

# 1. Python 3.12 check
if ! command -v "${PYTHON_CMD}" &> /dev/null; then
    echo "Error: ${PYTHON_CMD} is required but not found."
    exit 1
fi

echo "=========================================="
echo "VidEoMT Virtual Environment Setup"
echo "=========================================="

# 2. Create virtual environment
echo ""
echo "[1/6] Creating virtual environment..."
if [ -d "${VENV_DIR}" ]; then
    echo "  Warning: ${VENV_DIR} already exists."
    read -p "  Delete and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "${VENV_DIR}"
        "${PYTHON_CMD}" -m venv "${VENV_DIR}"
        echo "  Virtual environment recreated."
    else
        echo "  Using existing virtual environment."
    fi
else
    "${PYTHON_CMD}" -m venv "${VENV_DIR}"
    echo "  Virtual environment created at ${VENV_DIR}"
fi

# 3. Activate and upgrade pip
echo ""
echo "[2/6] Activating virtual environment..."
source "${VENV_DIR}/bin/activate"
echo "  Activated: $(which python)"

echo ""
echo "[3/6] Upgrading pip..."
pip install --upgrade pip

# 4. Install PyTorch 2.7.0 + torchvision 0.22.0 (cu126)
# cu126 is backward-compatible with CUDA 12.8
echo ""
echo "[4/6] Installing PyTorch 2.7.0 + torchvision 0.22.0 (cu126)..."
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu126

# 5. Install detectron2 from GitHub
echo ""
echo "[5/6] Installing detectron2 and panopticapi..."
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
pip install git+https://github.com/cocodataset/panopticapi.git

# 6. Install requirements.txt
echo ""
echo "[6/6] Installing remaining dependencies..."
pip install -r "${SCRIPT_DIR}/requirements.txt"

# 7. Verify installation
echo ""
echo "Verifying installation..."
python -c "
import sys
print(f'Python: {sys.executable}')

import torch
print(f'PyTorch: {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  CUDA version: {torch.version.cuda}')

import torchvision
print(f'torchvision: {torchvision.__version__}')

import detectron2
print(f'detectron2: {detectron2.__version__}')

import timm
print(f'timm: {timm.__version__}')

import cv2
print(f'OpenCV: {cv2.__version__}')

print()
print('All imports successful!')
"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To run video demo:"
echo "  cd visualization"
echo "  python video_demo.py --config-file ../configs/ytvis19/videomt/vit-large/videomt_online_ViTL.yaml \\"
echo "    --input <frames_dir> --output <output_dir> \\"
echo "    --opts MODEL.WEIGHTS <path_to_weights>"
echo ""
