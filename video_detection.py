"""
VIDEO DETECTION WITH TRAINED YOLO MODEL
ParkIntel - Real-time Parking Space Detection on Video

Usage:
    python video_detection.py                    # Use demo mode
    python video_detection.py --source video.mp4  # Process video file
    python video_detection.py --source 0          # Webcam
    python video_detection.py --source 0 --show  # Show live window
"""

import cv2
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.module2_vehicle_detection import VehicleDetector
from modules.module3_occupancy import OccupancyEstimator


def process_video(source, model_path="models/best.pt", show=True, save_output=True, max_frames=200):
    """
    Process video file or webcam with YOLO parking detection.
    
    Args:
        source: Video file path, webcam index (0), or None for demo
        model_path: Path to trained YOLO model
        show: Whether to display video window
        save_output: Whether to save annotated video
        max_frames: Maximum frames to process (for demo mode)
    """
    # Initialize detector
    print("\n" + "=" * 60)
    print("PARKINTEL VIDEO DETECTION")
    print("=" * 60)
    print(f"\n[INFO] Loading trained model: {model_path}")
    
    detector = VehicleDetector(
        model_path=model_path,
        total_slots=50,
        confidence=0.45,
        device="cpu"
    )
    
    print(f"Model: {detector.model_name}")
    print(f"Using custom parking model: {detector.use_custom_model}")
    print(f"Classes: {detector.PARKING_CLASSES}")
    
    # Initialize occupancy tracker
    occupancy = OccupancyEstimator(total_slots=50, csv_path="data/video_occupancy_log.csv")
    
    # Open video source
    if source is None:
        print("\n[INFO] Running in DEMO mode (synthetic frames)")
        cap = None
    elif source.isdigit() or source == "0":
        print(f"\n[INFO] Opening webcam: {source}")
        cap = cv2.VideoCapture(int(source) if source.isdigit() else 0)
    else:
        print(f"\n[INFO] Opening video file: {source}")
        cap = cv2.VideoCapture(source)
    
    # Get video properties
    if cap:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video: {width}x{height} @ {fps:.1f} FPS")
    
    # Setup video writer
    writer = None
    if save_output:
        output_path = "data/parking_detection_output.mp4"
        # Use avc1 codec (H264/AVC) for better browser compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(output_path, fourcc, 30.0, (820, 620))
        print(f"[INFO] Output will be saved to: {output_path}")
    
    # Processing loop
    print("\n" + "=" * 60)
    print("Processing video... Press 'q' to quit, 's' to save frame")
    print("=" * 60 + "\n")
    
    frame_count = 0
    total_empty = 0
    total_occupied = 0
    
    # Demo mode: limit frames
    if cap is None:
        print(f"[INFO] Demo mode: processing max {max_frames} frames")
    
    while frame_count < max_frames:
        # Read frame
        if cap is None:
            # Demo mode: create synthetic frame
            frame = cv2.imread(str(list(project_root.glob("pfe.v2i.yolov8/test/images/*.jpg"))[0]))
            if frame is None:
                # Create blank frame if no images
                frame = cv2.imread(str(list(project_root.glob("data/*.jpg"))[0]))
                if frame is None:
                    frame = cv2.imread(str(list(project_root.glob("data/frames/*.jpg"))[0]))
                    if frame is None:
                        frame = cv2.imread(str(list(project_root.glob("data/*.jpg"))[0]))
                        if frame is None:
                            frame = cv2.imread(str(list(project_root.glob("pfe.v2i.yolov8/test/images/*.jpg"))[frame_count % 10] if frame_count < 10 else project_root.glob("pfe.v2i.yolov8/test/images/*.jpg")))
                        else:
                            break
            # Use test images cyclically
            test_images = list((project_root / "pfe.v2i.yolov8/test/images").glob("*.jpg"))
            if test_images:
                frame = cv2.imread(str(test_images[frame_count % len(test_images)]))
        else:
            ret, frame = cap.read()
            if not ret:
                print("\n[INFO] End of video reached")
                break
        
        if frame is None:
            break
            
        # Resize frame for consistent processing
        frame = cv2.resize(frame, (820, 620))
        
        # Detect parking spaces
        start_time = time.time()
        result = detector.detect(frame)
        process_time = time.time() - start_time
        
        # Get statistics
        empty = result.get('empty_count', 0)
        occupied = result.get('occupied_count', 0)
        total_detected = result.get('count', 0)
        
        total_empty += empty
        total_occupied += occupied
        frame_count += 1
        
        # Update occupancy
        occupancy.update(result)
        
        # Draw HUD with statistics
        if result['annotated_frame'] is not None:
            display_frame = result['annotated_frame']
        else:
            display_frame = frame.copy()
        
        # Add processing info overlay
        info_text = [
            f"Frame: {frame_count}",
            f"Empty: {empty}",
            f"Occupied: {occupied}",
            f"Total Spaces: {total_detected}",
            f"FPS: {1/process_time:.1f}",
            f"Mode: YOLO"
        ]
        
        # Display info
        for i, text in enumerate(info_text):
            cv2.putText(display_frame, text, (10, 580 + i*20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Show frame
        if show:
            cv2.imshow("ParkIntel - Parking Detection", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] Quitting...")
                break
            elif key == ord('s'):
                save_path = f"data/frame_{frame_count}.jpg"
                cv2.imwrite(save_path, display_frame)
                print(f"[INFO] Frame saved: {save_path}")
        
        # Write to output video
        if writer:
            writer.write(display_frame)
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames | Empty: {empty} | Occupied: {occupied} | Total: {total_detected}")
    
    # Cleanup
    if cap:
        cap.release()
    if writer:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "=" * 60)
    print("VIDEO PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Total frames processed: {frame_count}")
    print(f"Average empty spaces: {total_empty/max(frame_count,1):.1f}")
    print(f"Average occupied spaces: {total_occupied/max(frame_count,1):.1f}")
    print(f"Output saved to: {output_path if save_output else 'Not saved'}")
    print("=" * 60)
    
    return {
        "frames": frame_count,
        "avg_empty": total_empty / max(frame_count, 1),
        "avg_occupied": total_occupied / max(frame_count, 1),
        "output": output_path if save_output else None
    }


def main():
    parser = argparse.ArgumentParser(description="ParkIntel Video Detection")
    parser.add_argument("--source", type=str, default=None,
                       help="Video source: None=demo, 0=webcam, path/to/video.mp4")
    parser.add_argument("--model", type=str, default="models/best.pt",
                       help="Path to YOLO model")
    parser.add_argument("--no-show", action="store_true",
                       help="Don't display video window")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save output video")
    args = parser.parse_args()
    
    process_video(
        source=args.source,
        model_path=args.model,
        show=not args.no_show,
        save_output=not args.no_save
    )


if __name__ == "__main__":
    main()
