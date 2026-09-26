"""
Full Pipeline Test with Trained YOLO Model
Tests the complete detection -> occupancy -> prediction pipeline
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from modules.module2_vehicle_detection import VehicleDetector
from modules.module3_occupancy import OccupancyEstimator
from datetime import datetime
import cv2

print("=" * 70)
print("PARKINTEL FULL PIPELINE TEST WITH TRAINED YOLO MODEL")
print("=" * 70)

# Step 1: Initialize detector with trained model
print("\n[1] Initializing Vehicle Detector with trained model...")
detector = VehicleDetector(
    model_path="models/best.pt",
    total_slots=50,
    confidence=0.45,
    device="cpu"
)
print(f"    Model: {detector.model_name}")
print(f"    Using custom parking model: {detector.use_custom_model}")
print(f"    Classes: {detector.PARKING_CLASSES}")

# Step 2: Initialize occupancy estimator
print("\n[2] Initializing Occupancy Estimator...")
occupancy = OccupancyEstimator(total_slots=50, csv_path="data/test_occupancy_log.csv")

# Step 3: Process test images
print("\n[3] Processing test images from dataset...")
test_dir = Path("pfe.v2i.yolov8/test/images")
test_images = list(test_dir.glob("*.jpg"))[:5]

results = []
for img_path in test_images:
    frame = cv2.imread(str(img_path))
    if frame is None:
        continue
    
    # Detect parking spaces
    detection = detector.detect(frame)
    
    # Update occupancy
    occupancy_record = occupancy.update(detection)
    
    results.append({
        "image": img_path.name,
        "detection": detection,
        "occupancy": occupancy_record
    })
    
    print(f"    {img_path.name}")
    print(f"      -> Detected: {detection['count']} spaces")
    print(f"      -> Empty: {detection.get('empty_count', 0)}")
    print(f"      -> Occupied: {detection.get('occupied_count', 0)}")
    print(f"      -> Occupancy: {occupancy_record['occupancy']}/{occupancy_record['total_slots']}")

# Step 4: Show summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Total images processed: {len(results)}")
print(f"\nOccupancy Records:")
df = occupancy.get_dataframe()
print(df[["timestamp", "occupancy", "available", "occupancy_rate"]].tail(10).to_string())

print(f"\nOccupancy Stats:")
print(f"  Average occupancy: {df['occupancy'].mean():.1f}")
print(f"  Max occupancy: {df['occupancy'].max()}")
print(f"  Min occupancy: {df['occupancy'].min()}")

# Step 5: Save annotated images
print("\n[4] Saving annotated detection results...")
for i, r in enumerate(results[:3]):
    if r['detection']['annotated_frame'] is not None:
        output_path = f"data/pipeline_test_{i+1}.jpg"
        cv2.imwrite(output_path, r['detection']['annotated_frame'])
        print(f"    Saved: {output_path}")

print("\n" + "=" * 70)
print("FULL PIPELINE TEST COMPLETE!")
print("=" * 70)
