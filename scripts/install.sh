#!/bin/bash
set -e

echo "🚀 MojoX Installation Script"
echo "============================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
echo "📋 Checking system requirements..."

# Check Python
if command_exists python3; then
    echo "✅ Python3 found: $(python3 --version)"
else
    echo "❌ Python3 not found. Please install Python 3.10+"
    exit 1
fi

# Check pip
if command_exists pip3; then
    echo "✅ pip3 found"
else
    echo "❌ pip3 not found. Please install pip"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "pixi.toml" ]; then
    echo "❌ Please run this script from the MojoX project root directory"
    exit 1
fi

echo "🔧 Installing Python dependencies..."

# Install basic requirements
pip3 install --upgrade pip
pip3 install numpy torch torchvision

# Install computer vision libraries
echo "📺 Installing computer vision libraries..."
pip3 install opencv-python pillow

# Install video processing
echo "🎬 Installing video processing libraries..."
pip3 install ffmpeg-python

# Install ML libraries
echo "🧠 Installing ML libraries..."
pip3 install ultralytics

# Install development tools
echo "🛠️ Installing development tools..."
pip3 install pytest black mypy jupyter

# Try to install modular SDK (optional)
echo "🔥 Attempting to install Modular SDK..."
if command_exists curl; then
    curl -fsSL https://get.modular.com | bash || echo "⚠️ Modular SDK installation failed (optional)"
else
    echo "⚠️ curl not found, skipping Modular SDK installation"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p output temp logs demos/output

# Make scripts executable
echo "🔐 Setting permissions..."
chmod +x src/app.py
chmod +x benchmark/benchmark_mojo.py
chmod +x benchmark/benchmark_python.py
chmod +x docker/entrypoint.sh
chmod +x demos/create_sample.py

# Test installation
echo "🧪 Testing installation..."
python3 -c "import numpy, torch; print('✅ Core libraries imported successfully')"

# Check GPU availability
python3 -c "import torch; print(f'✅ GPU available: {torch.cuda.is_available()}')" 2>/dev/null || echo "⚠️ GPU check failed"

echo ""
echo "🎉 Installation complete!"
echo ""
echo "📖 Next steps:"
echo "   1. Run demo: python src/app.py --demo"
echo "   2. Process video: python src/app.py -i video.mp4 -o output.mp4"
echo "   3. Run benchmarks: python benchmark/benchmark_mojo.py"
echo "   4. Check documentation: docs/README.md"
echo ""
echo "🐳 For Docker setup:"
echo "   docker build -t mojox ."
echo "   docker run --gpus all mojox --demo" 