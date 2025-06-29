#!/bin/bash
set -e

echo "🏁 MojoX Hackathon Quick Setup"
echo "=============================="

# Quick system check
echo "🔍 Checking system..."
python3 --version
pip --version

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support first (most important)
echo "🔥 Installing PyTorch with CUDA 12.8 support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install essential ML packages
echo "🧠 Installing core ML packages..."
pip install ultralytics numpy opencv-python pillow

# Install video processing
echo "🎬 Installing video processing..."
pip install ffmpeg-python

# Install GPU acceleration
echo "⚡ Installing GPU acceleration..."
pip install cupy-cuda12x numba

# Install utilities
echo "🛠️ Installing utilities..."
pip install pyyaml tqdm psutil click rich matplotlib

# Install development tools
echo "🔧 Installing dev tools..."
pip install pytest black mypy

# Optional: Install web demo tools
echo "🌐 Installing web demo tools (optional)..."
pip install streamlit fastapi uvicorn || echo "⚠️ Web tools failed (optional)"

# Test GPU
echo "🧪 Testing GPU setup..."
python3 -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'🔢 GPU count: {torch.cuda.device_count()}')"

# Quick demo
echo "🎯 Testing video processing..."
python3 -c "import cv2, ultralytics; print('✅ Computer vision ready')" || echo "⚠️ CV test failed"

echo ""
echo "🎉 Hackathon setup complete!"
echo ""
echo "🚀 Quick start commands:"
echo "   python src/app.py --demo"
echo "   python src/app.py -i demos/sample.mp4 -o output.mp4"
echo ""
echo "📊 Run benchmarks:"
echo "   python benchmark/benchmark_mojo.py" 