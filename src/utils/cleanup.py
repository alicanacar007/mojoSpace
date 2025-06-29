#!/usr/bin/env python3
"""
Cleanup utilities for MojoX image processing
Handles removal of existing images before starting new processes
"""

import os
import shutil
import glob
from pathlib import Path
from typing import List, Union
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_directory_images(directory: Union[str, Path], file_patterns: List[str] = None) -> int:
    """
    Clean image files from a directory based on patterns
    
    Args:
        directory: Directory path to clean
        file_patterns: List of file patterns to match (default: common image patterns)
    
    Returns:
        Number of files removed
    """
    if file_patterns is None:
        file_patterns = [
            "*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif",
            "frame_*.jpg", "annotated_*.jpg", 
            "*_processing_results.json", "*_results.json"
        ]
    
    directory = Path(directory)
    
    if not directory.exists():
        logger.info(f"Directory {directory} does not exist, skipping cleanup")
        return 0
    
    files_removed = 0
    
    for pattern in file_patterns:
        matching_files = list(directory.glob(pattern))
        for file_path in matching_files:
            try:
                if file_path.is_file():
                    file_path.unlink()
                    files_removed += 1
                    logger.debug(f"Removed: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
    
    logger.info(f"🧹 Cleaned {files_removed} files from {directory}")
    return files_removed

def clean_all_result_directories() -> int:
    """
    Clean all known result directories used by MojoX processing
    
    Returns:
        Total number of files removed
    """
    result_directories = [
        "results/images",
        "results/mojo_images", 
        "results/true_multi_gpu",
        "output"
    ]
    
    total_removed = 0
    logger.info("🧹 Starting cleanup of all result directories...")
    
    for directory in result_directories:
        removed = clean_directory_images(directory)
        total_removed += removed
    
    logger.info(f"🧹 Total cleanup complete: {total_removed} files removed")
    return total_removed

def clean_specific_gpu_results(gpu_id: int, directory: Union[str, Path]) -> int:
    """
    Clean results for a specific GPU
    
    Args:
        gpu_id: GPU ID to clean results for
        directory: Directory containing the results
    
    Returns:
        Number of files removed
    """
    directory = Path(directory)
    
    if not directory.exists():
        return 0
    
    gpu_patterns = [
        f"*_gpu{gpu_id}.jpg",
        f"*_gpu{gpu_id}.png",
        f"frame_*_gpu{gpu_id}.jpg",
        f"annotated_*_gpu{gpu_id}.jpg"
    ]
    
    files_removed = 0
    
    for pattern in gpu_patterns:
        matching_files = list(directory.glob(pattern))
        for file_path in matching_files:
            try:
                if file_path.is_file():
                    file_path.unlink()
                    files_removed += 1
                    logger.debug(f"Removed GPU {gpu_id} file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")
    
    logger.info(f"🧹 Cleaned {files_removed} GPU {gpu_id} files from {directory}")
    return files_removed

def clean_old_results(directory: Union[str, Path], days_old: int = 7) -> int:
    """
    Clean results older than specified days
    
    Args:
        directory: Directory to clean
        days_old: Remove files older than this many days
    
    Returns:
        Number of files removed
    """
    import time
    
    directory = Path(directory)
    
    if not directory.exists():
        return 0
    
    cutoff_time = time.time() - (days_old * 24 * 60 * 60)
    files_removed = 0
    
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            try:
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    files_removed += 1
                    logger.debug(f"Removed old file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove old file {file_path}: {e}")
    
    logger.info(f"🧹 Cleaned {files_removed} old files (>{days_old} days) from {directory}")
    return files_removed

def ensure_clean_start(output_dir: Union[str, Path]) -> int:
    """
    Ensure a clean start by removing existing images from the output directory
    This is the main function to call at the start of processing
    
    Args:
        output_dir: The output directory to clean
    
    Returns:
        Number of files removed
    """
    logger.info(f"🧹 Ensuring clean start for directory: {output_dir}")
    return clean_directory_images(output_dir)

def get_directory_stats(directory: Union[str, Path]) -> dict:
    """
    Get statistics about files in a directory
    
    Args:
        directory: Directory to analyze
    
    Returns:
        Dictionary with file statistics
    """
    directory = Path(directory)
    
    if not directory.exists():
        return {"exists": False, "total_files": 0, "total_size": 0}
    
    stats = {
        "exists": True,
        "total_files": 0,
        "total_size": 0,
        "image_files": 0,
        "json_files": 0
    }
    
    for file_path in directory.rglob("*"):
        if file_path.is_file():
            stats["total_files"] += 1
            stats["total_size"] += file_path.stat().st_size
            
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                stats["image_files"] += 1
            elif file_path.suffix.lower() == '.json':
                stats["json_files"] += 1
    
    return stats

if __name__ == "__main__":
    # Command line interface for cleanup
    import argparse
    
    parser = argparse.ArgumentParser(description="🧹 MojoX Image Cleanup Utility")
    parser.add_argument("--all", action="store_true", help="Clean all result directories")
    parser.add_argument("--dir", type=str, help="Clean specific directory")
    parser.add_argument("--gpu", type=int, help="Clean specific GPU results")
    parser.add_argument("--days", type=int, default=7, help="Clean files older than N days")
    parser.add_argument("--stats", action="store_true", help="Show directory statistics")
    
    args = parser.parse_args()
    
    if args.all:
        clean_all_result_directories()
    elif args.dir:
        if args.gpu is not None:
            clean_specific_gpu_results(args.gpu, args.dir)
        else:
            clean_directory_images(args.dir)
    elif args.stats:
        directories = ["results/images", "results/mojo_images", "results/true_multi_gpu", "output"]
        for directory in directories:
            stats = get_directory_stats(directory)
            if stats["exists"]:
                print(f"\n📊 {directory}:")
                print(f"   Files: {stats['total_files']}")
                print(f"   Images: {stats['image_files']}")
                print(f"   Size: {stats['total_size'] / (1024*1024):.2f} MB")
    else:
        print("Use --all to clean all directories, --dir to clean specific directory, or --stats to show statistics") 