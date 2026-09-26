"""
Test the trained YOLO model on actual parking lot images
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from modules.module2_vehicle_detection import VehicleDetector
import cv2

# Initialize detector with trained model
print("=" * 60)
print("Testing Trained YOLO Model on Parking Lot Images")
print("=" * 60)

detector = VehicleDetector(
    model_path="models/best.pt",
    total_slots=50,
    confidence=0.45,
    device="cpu"
)

print(f"\nModel: {detector.model_name}")
print(f"Using custom parking model: {detector.use_custom_model}")
print(f"Classes: {detector.PARKING_CLASSES}")

# Test on actual images from dataset
test_dir = Path("pfe.v2i.yolov8/test/images")
test_images = list(test_dir.glob("*.jpg"))[:3]

print(f"\n{'='*60}")
print(f"Testing on {len(test_images)} images from dataset")
print(f"{'='*60}\n")

for img_path in test_images:
    print(f"Image: {img_path.name}")
    frame = cv2.imread(str(img_path))
    
    if frame is None:
        print(f"  Could not load image")
        continue
    
    result = detector.detect(frame)
    
    print(f"  Mode: {result['mode']}")
    print(f"  Total detections: {result['count']}")
    print(f"  Empty spaces: {result.get('empty_count', 0)}")
    print(f"  Occupied spaces: {result.get('occupied_count', 0)}")
    
    # Show class breakdown
    if result['labels']:
        from collections import Counter
        label_counts = Counter(result['labels'])
        print(f"  Breakdown: {dict(label_counts)}")
    
    # Save annotated result
    if result['annotated_frame'] is not None:
        output_path = f"data/detected_{img_path.name}"
        cv2.imwrite(output_path, result['annotated_frame'])
        print(f"  Saved: {output_path}")
    
    print()

print("=" * 60)
print("Test Complete!")
print("=" * 60)
