"""
PARKINTEL VIDEO SIMULATION
Creates a video simulation from the Roboflow dataset images
with real-time parking space detection

This script:
1. Loads images from the dataset
2. Processes each with YOLO model
3. Creates an annotated video output
"""

import cv2
import sys
import os
from pathlib import Path
from datetime import datetime
import numpy as np

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.module2_vehicle_detection import VehicleDetector
from modules.module3_occupancy import OccupancyEstimator


def create_video_simulation(
    output_path="data/parking_simulation.mp4",
    max_frames=300,
    fps=10
):
    """
    Create a video simulation from dataset images with YOLO detection.
    """
    print("=" * 70)
    print("PARKINTEL VIDEO SIMULATION CREATOR")
    print("=" * 70)
    
    # Initialize detector
    print("\n[1/4] Loading YOLO model...")
    detector = VehicleDetector(
        model_path="models/best.pt",
        total_slots=50,
        confidence=0.45,
        device="cpu"
    )
    print(f"    Model: {detector.model_name}")
    print(f"    Classes: {detector.PARKING_CLASSES}")
    
    # Initialize occupancy tracker
    print("\n[2/4] Initializing occupancy tracker...")
    occupancy = OccupancyEstimator(total_slots=50, csv_path="data/simulation_log.csv")
    
    # Get dataset images
    print("\n[3/4] Loading dataset images...")
    dataset_dir = project_root / "pfe.v2i.yolov8" / "test" / "images"
    images = sorted(list(dataset_dir.glob("*.jpg")))
    
    if not images:
        dataset_dir = project_root / "pfe.v2i.yolov8" / "train" / "images"
        images = sorted(list(dataset_dir.glob("*.jpg")))
    
    if not images:
        print("[ERROR] No images found in dataset!")
        return None
    
    print(f"    Found {len(images)} images")
    
    # Take subset of images for video
    frame_indices = np.linspace(0, len(images)-1, min(max_frames, len(images)), dtype=int)
    selected_images = [images[i] for i in frame_indices]
    print(f"    Using {len(selected_images)} frames for video")
    
    # Read first image to get dimensions
    first_frame = cv2.imread(str(selected_images[0]))
    height, width = first_frame.shape[:2]
    print(f"    Frame size: {width}x{height}")
    
    # Create video writer - use avc1 codec (H264/AVC) for better browser compatibility
    print(f"\n[4/4] Creating video: {output_path}")
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process each frame
    print("\nProcessing frames...")
    stats = {
        "empty": [],
        "occupied": [],
        "total": [],
        "timestamp": []
    }
    
    for i, img_path in enumerate(selected_images):
        # Read frame
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        # Detect parking spaces
        result = detector.detect(frame)
        
        # Get stats
        empty = result.get('empty_count', 0)
        occupied = result.get('occupied_count', 0)
        total = result.get('count', 0)
        
        stats["empty"].append(empty)
        stats["occupied"].append(occupied)
        stats["total"].append(total)
        stats["timestamp"].append(datetime.now().isoformat())
        
        # Update occupancy
        occupancy.update(result)
        
        # Get annotated frame
        if result['annotated_frame'] is not None:
            output_frame = result['annotated_frame']
        else:
            output_frame = frame
        
        # Add info overlay
        overlay = output_frame.copy()
        
        # Top info bar
        cv2.rectangle(overlay, (0, 0), (width, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output_frame, 0.4, 0, output_frame)
        
        # Title
        cv2.putText(output_frame, "PARKINTEL - AI SMART PARKING SYSTEM",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2)
        
        # Stats
        cv2.putText(output_frame, f"Empty: {empty}",
                    (width-150, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        cv2.putText(output_frame, f"Occupied: {occupied}",
                    (width-150, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2)
        
        # Bottom info bar
        cv2.rectangle(overlay, (0, height-40), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, output_frame, 0.4, 0, output_frame)
        
        # Frame number
        cv2.putText(output_frame, f"Frame: {i+1}/{len(selected_images)}",
                    (10, height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Total spaces
        cv2.putText(output_frame, f"Total Spaces: {total}",
                    (200, height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Occupancy rate
        rate = (occupied / max(total, 1)) * 100
        cv2.putText(output_frame, f"Occupancy: {rate:.1f}%",
                    (400, height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Timestamp
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(output_frame, ts,
                    (width-200, height-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Write frame
        writer.write(output_frame)
        
        # Progress
        if (i+1) % 30 == 0:
            print(f"    Progress: {i+1}/{len(selected_images)} frames")
    
    # Release video writer
    writer.release()
    
    # Print summary
    print("\n" + "=" * 70)
    print("VIDEO SIMULATION COMPLETE!")
    print("=" * 70)
    print(f"\nOutput video: {output_path}")
    print(f"\nStatistics:")
    print(f"  Total frames: {len(stats['empty'])}")
    print(f"  Average empty: {np.mean(stats['empty']):.1f}")
    print(f"  Average occupied: {np.mean(stats['occupied']):.1f}")
    print(f"  Average total spaces: {np.mean(stats['total']):.1f}")
    print(f"  Average occupancy: {np.mean(stats['occupied'])/max(np.mean(stats['total']),1)*100:.1f}%")
    print("\n" + "=" * 70)
    
    return output_path


if __name__ == "__main__":
    # Create video with 100 frames at 10 FPS
    output = create_video_simulation(
        output_path="data/parking_simulation.mp4",
        max_frames=100,
        fps=10
    )
    
    if output:
        print(f"\n✅ Video created successfully: {output}")
        print("You can open this file to view the parking simulation!")
