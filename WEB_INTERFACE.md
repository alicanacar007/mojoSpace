# 🌐 MojoX Web Interface Guide

## 🚀 Getting Started

MojoX can be used through your web browser in two ways:

### Option 1: Streamlit Web App (Recommended)
Interactive web interface with drag-and-drop functionality.

```bash
# Start the web interface
./start_web.sh

# Or manually:
streamlit run web_app.py --server.port 8501
```

**Access**: `http://localhost:8501`

### Option 2: FastAPI REST API
Programmatic API for developers.

```bash
# Start the API server
python api_server.py

# Or with uvicorn:
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**Access**: 
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`

## 🎬 Web Interface Features

### Streamlit App Features:
- **📹 Video Upload**: Drag and drop video files
- **🎥 Demo Videos**: Use pre-loaded sample videos
- **⚙️ Real-time Configuration**: Adjust detection settings
- **📊 Live Processing**: See progress and statistics
- **🎨 Visual Results**: Download processed videos
- **🚀 GPU Monitoring**: Real-time system info

### API Features:
- **REST Endpoints**: Programmatic access
- **Background Processing**: Async video processing
- **Job Tracking**: Monitor processing status
- **File Management**: Upload/download videos
- **OpenAPI Docs**: Interactive documentation

## 📋 Configuration Options

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| Confidence Threshold | 0.0-1.0 | 0.25 | Detection confidence |
| IoU Threshold | 0.0-1.0 | 0.45 | Overlap threshold |
| Device | cuda/cpu | cuda | Processing device |
| Target FPS | 1-60 | 30 | Processing frame rate |
| Use Mojo Kernels | true/false | true | Enable Mojo acceleration |
| Enable NMS | true/false | true | Non-maximum suppression |
| Box Thickness | 1-5 | 2 | Bounding box line width |

## 🎯 Usage Examples

### Upload Video via Web Interface:
1. Go to `http://localhost:8501`
2. Click "📹 Video Upload" tab
3. Drag and drop your video
4. Adjust settings in sidebar
5. Click "🚀 Process Video"
6. View results in "📊 Results" tab

### API Usage:
```bash
# Upload and process video
curl -X POST "http://localhost:8000/upload" \
     -F "file=@your_video.mp4" \
     -F "confidence_threshold=0.25" \
     -F "device=cuda"

# Check job status
curl "http://localhost:8000/jobs/{job_id}"

# Download result
curl "http://localhost:8000/download/{job_id}" -o result.mp4
```

### Demo Videos:
```bash
# List available demos
curl "http://localhost:8000/demos"

# Or use the web interface "🎥 Demo Videos" tab
```

## 🔧 System Requirements

### Minimum:
- Python 3.10+
- 4GB RAM
- 2GB disk space

### Recommended:
- **8x NVIDIA H100 80GB HBM3** (like your setup!)
- NVIDIA GPU (CUDA 12.x)
- 8GB+ GPU memory per GPU
- 16GB+ RAM
- SSD storage

## 🐛 Troubleshooting

### Web Interface Won't Start:
```bash
# Install missing dependencies
pip install streamlit fastapi uvicorn

# Check port availability
netstat -tulpn | grep :8501
```

### GPU Issues:
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Monitor GPU usage
nvidia-smi
```

### Processing Errors:
- Check video format (MP4 recommended)
- Verify file size (<500MB for web upload)
- Ensure sufficient disk space
- Check GPU memory availability

## 📚 Additional Resources

- **Demo Videos**: Available in `demos/` directory
- **Configuration**: Edit `config/default.yaml`
- **Output**: Check `output/` directory
- **Logs**: Available in terminal/console

## 🚀 Performance Tips

1. **Use GPU**: Enable CUDA for best performance
2. **Optimize FPS**: Lower target FPS for faster processing
3. **Batch Processing**: Process multiple videos sequentially
4. **Memory Management**: Close browser tabs when not in use
5. **File Size**: Compress large videos before upload

---

🎉 **Happy Processing!** Your MojoX web interface is ready to detect objects in real-time! 