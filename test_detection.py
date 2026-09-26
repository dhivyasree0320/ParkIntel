"""
Test the trained YOLO model on a test image
"""
from ultralytics import YOLO
import cv2
import os

# Load the trained model
print("Loading model...")
model = YOLO('models/best.pt')

# Get a test image
test_images = os.listdir('pfe.v2i.yolov8/test/images')
test_image = f'pfe.v2i.yolov8/test/images/{test_images[0]}'

print(f'Testing with: {test_images[0]}')

# Run detection
results = model(test_image, conf=0.45)

# Print results
for r in results:
    print(f'Detected {len(r.boxes)} objects')
    if len(r.boxes) > 0:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            print(f'  Class: {model.names[cls]}, Confidence: {conf:.2f}')

# Save annotated image
annotated = results[0].plot()
cv2.imwrite('data/test_detection.jpg', annotated)
print('Saved: data/test_detection.jpg')
print("Test completed successfully!")
