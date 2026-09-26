"""
Create web-friendly video for Streamlit dashboard
"""
import cv2
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.module2_vehicle_detection import VehicleDetector

def create_web_video(
    output_path="data/web_simulation.mp4",
    max_frames=60,
    fps=15
):
    """Create a compact, web-friendly video for dashboard."""
    print("Creating web-friendly video...")
    
    # Initialize detector
    detector = VehicleDetector(
        model_path="models/best.pt",
        total_slots=50,
        confidence=0.45,
        device="cpu"
    )
    
    # Get dataset images
    dataset_dir = project_root / "pfe.v2i.yolov8" / "test" / "images"
    images = sorted(list(dataset_dir.glob("*.jpg")))
    
    if not images:
        print("No images found!")
        return None
    
    # Take subset
    frame_indices = range(0, min(max_frames, len(images)))
    selected_images = [images[i] for i in frame_indices]
    
    # Read first image
    first_frame = cv2.imread(str(selected_images[0]))
    height, width = first_frame.shape[:2]
    
    # Create video writer with H264 codec
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"Processing {len(selected_images)} frames...")
    
    for i, img_path in enumerate(selected_images):
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        # Detect
        result = detector.detect(frame)
        
        # Get annotated frame
        if result['annotated_frame'] is not None:
            output_frame = result['annotated_frame']
        else:
            output_frame = frame
        
        # Add info overlay
        cv2.rectangle(output_frame, (0, 0), (width, 50), (0, 0, 0), -1)
        cv2.putText(output_frame, "PARKINTEL - YOLOv11 Detection", 
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        
        # Stats
        empty = result.get('empty_count', 0)
        occupied = result.get('occupied_count', 0)
        cv2.putText(output_frame, f"Empty: {empty} | Occupied: {occupied}", 
                    (10, height-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        writer.write(output_frame)
        
        if (i+1) % 20 == 0:
            print(f"  {i+1}/{len(selected_images)}")
    
    writer.release()
    print(f"\nVideo saved: {output_path}")
    
    # Get file size
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"Size: {size_mb:.1f} MB")
    
    return output_path

if __name__ == "__main__":
    create_web_video()
