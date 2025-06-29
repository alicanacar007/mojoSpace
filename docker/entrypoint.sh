#!/bin/bash
set -e

# MojoX Video Object Detection - Docker Entrypoint Script

echo "Starting MojoX Video Object Detection Pipeline..."

# Function to print usage
show_help() {
    cat << EOF
MojoX Video Object Detection Docker Container

Usage:
    docker run mojox [OPTIONS] [COMMAND]

Options:
    --help          Show this help message
    --demo          Run demo mode
    --benchmark     Run benchmark suite
    --jupyter       Start Jupyter notebook server
    --shell         Start interactive shell

Environment Variables:
    INPUT_VIDEO     Path to input video file
    OUTPUT_VIDEO    Path to output video file
    CONFIG_FILE     Path to configuration file
    TARGET_FPS      Target frames per second
    DEVICE          Device to use (cpu/cuda)
    CONF_THRESHOLD  Confidence threshold for detections

Examples:
    # Run demo mode
    docker run mojox --demo

    # Process video file
    docker run -v /path/to/video:/input mojox python3 src/app.py -i /input/video.mp4 -o /output/result.mp4

    # Start development environment
    docker run -it mojox --shell

    # Run benchmarks
    docker run mojox --benchmark
EOF
}

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Status:"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
        return 0
    else
        echo "Warning: No GPU detected. Running in CPU mode."
        export DEVICE="cpu"
        return 1
    fi
}

# Function to setup environment
setup_environment() {
    echo "Setting up environment..."
    
    # Create necessary directories
    mkdir -p /app/output /app/temp /app/logs
    
    # Set permissions
    chmod -R 755 /app/src
    
    # Check Python environment
    echo "Python version: $(python3 --version)"
    echo "PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
    
    # Check CUDA if available
    if [ "$DEVICE" != "cpu" ]; then
        python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
        if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
            echo "CUDA device count: $(python3 -c 'import torch; print(torch.cuda.device_count())')"
            echo "CUDA device name: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null)"
        fi
    fi
}

# Function to run demo mode
run_demo() {
    echo "Running MojoX demo mode..."
    python3 src/app.py --demo --verbose
}

# Function to run benchmarks
run_benchmark() {
    echo "Running MojoX benchmark suite..."
    if [ -f benchmark/benchmark_mojo.py ]; then
        python3 benchmark/benchmark_mojo.py
    else
        echo "Benchmark script not found. Running basic performance test..."
        python3 src/app.py --demo --verbose
    fi
}

# Function to start Jupyter server
start_jupyter() {
    echo "Starting Jupyter notebook server..."
    cd /app
    jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --no-browser \
        --NotebookApp.token='' --NotebookApp.password=''
}

# Function to start interactive shell
start_shell() {
    echo "Starting interactive shell..."
    exec /bin/bash
}

# Main execution logic
main() {
    # Setup environment
    setup_environment
    
    # Check for GPU
    check_gpu
    
    # Parse command line arguments
    case "${1:-}" in
        --help|-h)
            show_help
            exit 0
            ;;
        --demo)
            run_demo
            ;;
        --benchmark)
            run_benchmark
            ;;
        --jupyter)
            start_jupyter
            ;;
        --shell)
            start_shell
            ;;
        "")
            # No arguments provided, check environment variables
            if [ -n "$INPUT_VIDEO" ]; then
                echo "Processing video: $INPUT_VIDEO"
                ARGS="-i $INPUT_VIDEO"
                
                if [ -n "$OUTPUT_VIDEO" ]; then
                    ARGS="$ARGS -o $OUTPUT_VIDEO"
                fi
                
                if [ -n "$CONFIG_FILE" ]; then
                    ARGS="$ARGS -c $CONFIG_FILE"
                fi
                
                if [ -n "$TARGET_FPS" ]; then
                    ARGS="$ARGS --fps $TARGET_FPS"
                fi
                
                if [ -n "$DEVICE" ]; then
                    ARGS="$ARGS --device $DEVICE"
                fi
                
                if [ -n "$CONF_THRESHOLD" ]; then
                    ARGS="$ARGS --conf-threshold $CONF_THRESHOLD"
                fi
                
                python3 src/app.py $ARGS --verbose
            else
                echo "No command specified. Running demo mode..."
                run_demo
            fi
            ;;
        *)
            # Pass through any other commands
            echo "Executing command: $@"
            exec "$@"
            ;;
    esac
}

# Trap for graceful shutdown
trap 'echo "Shutting down MojoX..."; exit 0' SIGTERM SIGINT

# Run main function
main "$@" 