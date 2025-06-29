#!/bin/bash
# MojoX Environment Setup Script

set -e

echo "🚀 Setting up MojoX Mojo environment..."

# Check if Modular CLI is installed
if ! command -v modular &> /dev/null; then
    echo "📦 Installing Modular CLI..."
    
    # Install Modular CLI
    curl -s https://get.modular.com | sh -
    
    # Add to PATH
    export PATH="$HOME/.modular/bin:$PATH"
    echo 'export PATH="$HOME/.modular/bin:$PATH"' >> ~/.bashrc
fi

# Authenticate with Modular (if not already done)
if ! modular auth status &> /dev/null; then
    echo "🔐 Please authenticate with Modular:"
    echo "Run: modular auth"
    echo "Then run this script again."
    exit 1
fi

# Install Mojo
echo "🔥 Installing Mojo..."
modular install mojo || echo "Mojo already installed or installation failed"

# Set up Mojo environment
export MODULAR_HOME="$HOME/.modular"
export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"

# Add to bashrc for persistence
echo 'export MODULAR_HOME="$HOME/.modular"' >> ~/.bashrc
echo 'export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"' >> ~/.bashrc

echo "✅ Mojo environment setup complete!"
echo "🔄 Please restart your shell or run: source ~/.bashrc"

# Test Mojo installation
if command -v mojo &> /dev/null; then
    echo "🎉 Mojo is ready!"
    mojo --version
else
    echo "⚠️ Mojo installation may need manual verification"
fi 