# 🧹 MojoX Image Cleanup Implementation

## Overview

The MojoX project now includes comprehensive image cleanup functionality that automatically removes existing images at the start of every processing task. This ensures clean results and prevents confusion from old processing artifacts.

## 🚀 What's Implemented

### 1. Cleanup Utility Module (`src/utils/cleanup.py`)

A comprehensive cleanup utility with the following functions:

- **`ensure_clean_start(output_dir)`** - Main function called at the start of processing
- **`clean_directory_images(directory, patterns)`** - Clean specific file patterns from a directory
- **`clean_all_result_directories()`** - Clean all known result directories
- **`clean_specific_gpu_results(gpu_id, directory)`** - Clean results for a specific GPU
- **`clean_old_results(directory, days_old)`** - Clean files older than specified days
- **`get_directory_stats(directory)`** - Get statistics about directory contents

### 2. Integration Points

The cleanup functionality has been integrated into all main processing scripts:

#### **`mojo_multi_gpu_run.py`**
- Cleans `results/mojo_images/` directory before processing
- Shows cleanup message when files are removed

#### **`true_multi_gpu_mojo.py`** 
- Cleans `results/true_multi_gpu/` directory before processing
- Shows cleanup message when files are removed

#### **`src/multi_gpu_processor.py`**
- Cleans `results/images/` directory before processing
- Shows cleanup message when files are removed

#### **`web_app.py`**
- Cleans `output/` directory before processing
- Shows Streamlit info message when files are removed

## 🎯 What Gets Cleaned

The cleanup targets these file patterns by default:

- **Image files**: `*.jpg`, `*.jpeg`, `*.png`, `*.bmp`, `*.tiff`, `*.tif`
- **Frame files**: `frame_*.jpg`, `annotated_*.jpg`
- **Result files**: `*_processing_results.json`, `*_results.json`

## 📁 Target Directories

The system automatically cleans these directories:

- `results/images/` - Standard multi-GPU processing results
- `results/mojo_images/` - Mojo-enhanced processing results  
- `results/true_multi_gpu/` - True multi-GPU processing results
- `output/` - Web app processing results

## 🔧 Command Line Usage

The cleanup utility can also be used manually:

```bash
# Clean all result directories
python3 src/utils/cleanup.py --all

# Clean specific directory
python3 src/utils/cleanup.py --dir results/images

# Clean specific GPU results
python3 src/utils/cleanup.py --dir results/images --gpu 0

# Show directory statistics
python3 src/utils/cleanup.py --stats

# Clean files older than 7 days
python3 src/utils/cleanup.py --dir results/images --days 7
```

## 🧪 Testing

A comprehensive test suite is provided in `test_cleanup.py`:

```bash
python3 test_cleanup.py
```

This test demonstrates:
- Creating test images
- Directory statistics before/after cleanup
- Individual directory cleanup
- Pattern-specific cleanup
- Complete cleanup of all directories
- Processing start simulation

## ⚡ Automatic Behavior

**Every time you start a processing task**, the system will:

1. **Check** the target output directory for existing files
2. **Remove** all matching image and result files
3. **Report** how many files were cleaned
4. **Proceed** with clean processing

## 📊 Example Output

When you run any processing script, you'll see:

```
🎬 Processing video: video.mp4
🧹 Cleaned 45 existing files from results/mojo_images
📹 Extracting frames every second (FPS: 30.0)
...
```

## 🛡️ Safety Features

- **Directory validation** - Only cleans if directory exists
- **Pattern matching** - Only removes files matching known patterns
- **Error handling** - Continues processing even if cleanup fails
- **Logging** - All cleanup actions are logged
- **Statistics** - Reports exactly what was cleaned

## 💡 Benefits

1. **Clean Results** - No confusion from old processing artifacts
2. **Disk Space** - Automatic cleanup prevents disk bloat
3. **Consistency** - Every run starts with a clean slate
4. **Transparency** - Shows exactly what was cleaned
5. **Flexibility** - Can be used manually or automatically

## 🔄 Integration Flow

```
Start Processing
       ↓
Create Output Directory
       ↓
🧹 Clean Existing Images  ← NEW!
       ↓
Extract Video Frames
       ↓
Process with GPUs
       ↓
Save New Results
```

## ✅ Verification

To verify the cleanup is working, you can:

1. **Run the test**: `python3 test_cleanup.py`
2. **Check statistics**: `python3 src/utils/cleanup.py --stats`
3. **Process a video**: Any processing script will show cleanup messages
4. **Manual cleanup**: `python3 src/utils/cleanup.py --all`

---

**🎉 Result**: Every processing run now starts with a completely clean directory, ensuring consistent results and preventing old artifacts from interfering with new processing tasks! 