# Multi-stage Docker build for MojoX Video Object Detection
# Base image with CUDA support
FROM nvidia/cuda:12.2-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    ffmpeg \
    libffmpeg-dev \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Mojo and Modular SDK
WORKDIR /tmp
RUN curl -fsSL https://get.modular.com | sh - && \
    modular auth --new-token && \
    modular install mojo

# Set up Mojo environment
ENV MOJO_PATH=/root/.modular/pkg/packages.modular.com_mojo
ENV PATH=$MOJO_PATH/bin:$PATH

# Create application directory
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .
COPY pixi.toml .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install additional GPU-specific packages
RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Copy application code
COPY src/ ./src/
COPY demos/ ./demos/
COPY benchmark/ ./benchmark/
COPY docs/ ./docs/

# Set permissions
RUN chmod +x src/app.py

# Create output directories
RUN mkdir -p /app/output /app/temp /app/logs

# Build stage for Mojo kernels
FROM base as mojo-builder

# Copy Mojo source files
COPY src/kernels/ ./src/kernels/

# Compile Mojo kernels (placeholder - actual compilation would happen here)
# RUN mojo build src/kernels/frame_extractor.mojo -o bin/frame_extractor
# RUN mojo build src/kernels/nms.mojo -o bin/nms

# Production stage
FROM base as production

# Copy compiled kernels from builder stage
# COPY --from=mojo-builder /app/bin/ ./bin/

# Set working directory
WORKDIR /app

# Expose port for potential web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('GPU available:', torch.cuda.is_available())" || exit 1

# Default command
CMD ["python3", "src/app.py", "--demo"]

# Development stage with additional tools
FROM production as development

# Install development dependencies
RUN pip3 install --no-cache-dir \
    jupyter \
    pytest \
    black \
    mypy \
    flake8 \
    ipython

# Install debugging tools
RUN apt-get update && apt-get install -y \
    htop \
    nvtop \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Jupyter configuration
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.port = 8888" >> ~/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> ~/.jupyter/jupyter_notebook_config.py

EXPOSE 8888

# Override command for development
CMD ["bash"] 