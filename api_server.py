#!/usr/bin/env python3
"""
MojoX REST API Server
FastAPI-based REST API for video object detection
"""

from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
import sys
import time
import json
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.app import VideoObjectDetectionPipeline
    from src.utils.config import ConfigManager
except ImportError:
    print("⚠️ Warning: MojoX pipeline not available in demo mode")

app = FastAPI(
    title="MojoX API",
    description="Real-Time Video Object Detection API with Mojo Kernels & MAX Graph",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)
upload_dir = Path("uploads")
upload_dir.mkdir(exist_ok=True)

# Mount static files
if Path("output").exists():
    app.mount("/output", StaticFiles(directory="output"), name="output")
if Path("demos").exists():
    app.mount("/demos", StaticFiles(directory="demos"), name="demos")

# Job tracking
processing_jobs = {}

class ProcessingJob:
    def __init__(self, job_id: str, video_path: str, config: Dict[str, Any]):
        self.job_id = job_id
        self.video_path = video_path
        self.config = config
        self.status = "queued"
        self.progress = 0
        self.start_time = time.time()
        self.end_time = None
        self.output_path = None
        self.error = None
        self.stats = {}

@app.get("/", response_class=HTMLResponse)
async def home():
    """Serve the main web interface"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔥 MojoX API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
            .container { max-width: 800px; margin: 0 auto; background: rgba(255,255,255,0.1); padding: 30px; border-radius: 15px; }
            .header { text-align: center; margin-bottom: 40px; }
            .section { margin: 20px 0; padding: 20px; background: rgba(255,255,255,0.1); border-radius: 10px; }
            .code { background: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; font-family: monospace; }
            .endpoint { margin: 10px 0; }
            .method { display: inline-block; padding: 5px 10px; border-radius: 3px; color: white; font-weight: bold; margin-right: 10px; }
            .get { background-color: #61affe; }
            .post { background-color: #49cc90; }
            .button { display: inline-block; padding: 10px 20px; background: linear-gradient(45deg, #FF6B6B, #4ECDC4); color: white; text-decoration: none; border-radius: 25px; margin: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔥 MojoX API</h1>
                <p>Real-Time Video Object Detection with Mojo Kernels & MAX Graph</p>
            </div>
            
            <div class="section">
                <h2>📋 API Endpoints</h2>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <strong>/docs</strong> - Interactive API documentation
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <strong>/health</strong> - Health check
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <strong>/demos</strong> - List demo videos
                </div>
                
                <div class="endpoint">
                    <span class="method post">POST</span>
                    <strong>/upload</strong> - Upload and process video
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <strong>/jobs/{job_id}</strong> - Check job status
                </div>
                
                <div class="endpoint">
                    <span class="method get">GET</span>
                    <strong>/results</strong> - List processed videos
                </div>
            </div>
            
            <div class="section">
                <h2>🚀 Quick Start</h2>
                <p>Upload and process a video:</p>
                <div class="code">
curl -X POST "http://localhost:8000/upload" \\
     -F "file=@your_video.mp4" \\
     -F "confidence_threshold=0.25" \\
     -F "device=cuda"
                </div>
            </div>
            
            <div style="text-align: center; margin-top: 40px;">
                <a href="/docs" class="button">📖 API Documentation</a>
                <a href="/results" class="button">📊 View Results</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        gpu_info = []
        if gpu_available and gpu_count > 0:
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
                gpu_info.append({
                    "gpu_id": i,
                    "name": name,
                    "memory_total_gb": round(memory_total, 1),
                    "memory_allocated_gb": round(memory_allocated, 1),
                    "memory_usage_percent": round((memory_allocated / memory_total) * 100, 1)
                })
        
        return {
            "status": "healthy",
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "gpu_info": gpu_info,
            "timestamp": time.time()
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@app.get("/demos")
async def list_demos():
    """List available demo videos"""
    demos_path = Path("demos")
    if not demos_path.exists():
        return {"demos": []}
    
    demos = []
    for video_file in demos_path.glob("*.mp4"):
        size = video_file.stat().st_size
        demos.append({
            "name": video_file.name,
            "path": f"/demos/{video_file.name}",
            "size_mb": round(size / (1024 * 1024), 2)
        })
    
    return {"demos": demos}

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence_threshold: float = Form(0.25),
    iou_threshold: float = Form(0.45),
    device: str = Form("cuda"),
    target_fps: int = Form(30),
    use_mojo_kernels: bool = Form(True),
    enable_nms: bool = Form(True),
    start_time: float = Form(0.0),
    duration: Optional[float] = Form(None)
):
    """Upload and process video file"""
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_path = upload_dir / f"{job_id}_{file.filename}"
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving file: {str(e)}")
    
    # Create job configuration
    config = {
        "confidence_threshold": confidence_threshold,
        "iou_threshold": iou_threshold,
        "device": device,
        "target_fps": target_fps,
        "use_mojo_kernels": use_mojo_kernels,
        "enable_nms": enable_nms,
        "start_time": start_time,
        "duration": duration
    }
    
    # Create job
    job = ProcessingJob(job_id, str(file_path), config)
    processing_jobs[job_id] = job
    
    # Start background processing
    background_tasks.add_task(process_video_job, job)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Video uploaded successfully, processing started"
    }

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get processing job status"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    response = {
        "job_id": job_id,
        "status": job.status,
        "progress": job.progress,
        "start_time": job.start_time
    }
    
    if job.end_time:
        response["end_time"] = job.end_time
        response["processing_time"] = job.end_time - job.start_time
    
    if job.output_path and Path(job.output_path).exists():
        response["output_url"] = f"/output/{Path(job.output_path).name}"
        response["download_url"] = f"/download/{job_id}"
    
    if job.error:
        response["error"] = job.error
    
    if job.stats:
        response["stats"] = job.stats
    
    return response

@app.get("/download/{job_id}")
async def download_result(job_id: str):
    """Download processed video"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if not job.output_path or not Path(job.output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        job.output_path,
        media_type="video/mp4",
        filename=Path(job.output_path).name
    )

@app.get("/results")
async def list_results():
    """List all processed videos"""
    results = []
    
    for video_file in output_dir.glob("*.mp4"):
        size = video_file.stat().st_size
        mod_time = video_file.stat().st_mtime
        
        results.append({
            "name": video_file.name,
            "url": f"/output/{video_file.name}",
            "size_mb": round(size / (1024 * 1024), 2),
            "created": mod_time
        })
    
    # Sort by creation time (newest first)
    results.sort(key=lambda x: x["created"], reverse=True)
    
    return {"results": results}

@app.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel processing job"""
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[job_id]
    
    if job.status in ["completed", "failed"]:
        return {"message": "Job already finished"}
    
    job.status = "cancelled"
    job.end_time = time.time()
    
    # Clean up files
    if Path(job.video_path).exists():
        Path(job.video_path).unlink()
    
    return {"message": "Job cancelled successfully"}

async def process_video_job(job: ProcessingJob):
    """Background task to process video"""
    try:
        job.status = "processing"
        job.progress = 10
        
        # Load configuration
        config_manager = ConfigManager("config/default.yaml")
        config = config_manager.get_config()
        
        # Update config with job settings
        config.model.conf_threshold = job.config["confidence_threshold"]
        config.model.device = job.config["device"]
        config.frame_extraction.target_fps = job.config["target_fps"]
        config.frame_extraction.use_mojo_kernel = job.config["use_mojo_kernels"]
        config.enable_nms = job.config["enable_nms"]
        
        job.progress = 20
        
        # Initialize pipeline
        pipeline = VideoObjectDetectionPipeline(config)
        job.progress = 30
        
        # Generate output path
        input_filename = Path(job.video_path).stem
        timestamp = int(time.time())
        output_path = output_dir / f"{input_filename}_processed_{timestamp}.mp4"
        job.output_path = str(output_path)
        
        job.progress = 40
        
        # Process video
        stats = pipeline.process_video(
            input_path=job.video_path,
            output_path=str(output_path),
            start_time=job.config["start_time"],
            duration=job.config["duration"]
        )
        
        job.progress = 100
        job.status = "completed"
        job.end_time = time.time()
        job.stats = stats
        
        # Clean up input file
        if Path(job.video_path).exists():
            Path(job.video_path).unlink()
            
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.end_time = time.time()
        
        # Clean up files on error
        if Path(job.video_path).exists():
            Path(job.video_path).unlink()

if __name__ == "__main__":
    import uvicorn
    
    print("🔥 Starting MojoX API Server")
    print("===========================")
    print("📍 API: http://localhost:8000")
    print("📖 Docs: http://localhost:8000/docs")
    print("🛑 Press Ctrl+C to stop")
    
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 