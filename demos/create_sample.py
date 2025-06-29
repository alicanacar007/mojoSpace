#!/usr/bin/env python3
"""
Create sample demo content for MojoX
"""

import cv2
import numpy as np
import os
from pathlib import Path

def create_sample_video(output_path: str, width: int = 640, height: int = 480, fps: int = 30, duration: int = 10):
    """Create a sample video with moving objects"""
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = fps * duration
    
    for frame_idx in range(total_frames):
        # Create background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (20, 20, 20)  # Dark background
        
        # Add moving objects
        time_ratio = frame_idx / total_frames
        
        # Moving car (rectangle)
        car_x = int(width * time_ratio)
        car_y = height // 2
        car_w, car_h = 80, 40
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (0, 255, 0), -1)
        cv2.putText(frame, "CAR", (car_x + 10, car_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Moving person (circle)
        person_x = int(width * (1 - time_ratio))
        person_y = height // 3
        person_r = 25
        cv2.circle(frame, (person_x, person_y), person_r, (255, 0, 0), -1)
        cv2.putText(frame, "PERSON", (person_x - 30, person_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Static bicycle (triangle)
        bike_x = width // 4
        bike_y = height * 3 // 4
        bike_pts = np.array([[bike_x, bike_y], [bike_x + 30, bike_y + 30], [bike_x - 30, bike_y + 30]], np.int32)
        cv2.fillPoly(frame, [bike_pts], (0, 0, 255))
        cv2.putText(frame, "BIKE", (bike_x - 20, bike_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "MojoX Demo Video", (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        writer.write(frame)
    
    writer.release()
    print(f"Sample video created: {output_path}")

def create_sample_image(output_path: str, width: int = 640, height: int = 480):
    """Create a sample image with objects"""
    
    # Create image
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = (30, 30, 30)  # Dark background
    
    # Add objects
    # Car
    cv2.rectangle(image, (100, 200), (200, 250), (0, 255, 0), -1)
    cv2.putText(image, "CAR", (130, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Person
    cv2.circle(image, (400, 150), 30, (255, 0, 0), -1)
    cv2.putText(image, "PERSON", (350, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Bicycle
    bike_pts = np.array([[500, 300], [530, 330], [470, 330]], np.int32)
    cv2.fillPoly(image, [bike_pts], (0, 0, 255))
    cv2.putText(image, "BIKE", (480, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # Title
    cv2.putText(image, "MojoX Demo Image", (width//2 - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imwrite(output_path, image)
    print(f"Sample image created: {output_path}")

def main():
    """Create sample demo content"""
    
    # Create demos directory
    demos_dir = Path(__file__).parent
    demos_dir.mkdir(exist_ok=True)
    
    # Create sample video
    video_path = demos_dir / "sample.mp4"
    create_sample_video(str(video_path))
    
    # Create sample image
    image_path = demos_dir / "sample.jpg"
    create_sample_image(str(image_path))
    
    # Create different resolution samples
    for res_name, (w, h) in [("720p", (1280, 720)), ("1080p", (1920, 1080))]:
        video_path = demos_dir / f"sample_{res_name}.mp4"
        create_sample_video(str(video_path), width=w, height=h, duration=5)
    
    print("All sample content created successfully!")

if __name__ == "__main__":
    main() 