"""
MODULE 2: VEHICLE DETECTION (YOLOv11)
ParkIntel - AI Smart Parking System

Roboflow Dataset : pfe Object Detection
Author           : bouakkaz144.lotfi@gmail.com

Detects vehicles in parking frames using YOLOv11.
Model loading order:
  1. Your Roboflow-trained model  → models/best.pt
  2. Pretrained YOLOv11 nano      → yolo11n.pt (auto-downloaded)
  3. Demo mode                    → simulates detections without GPU
"""

import cv2
import numpy as np
import time
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("[Detection] ultralytics not installed. pip install ultralytics")


class VehicleDetector:

    # Custom class names from trained Roboflow dataset (pfe)
    # Class 0: empty parking space
    # Class 1: occupied parking space
    PARKING_CLASSES = {0: "empty", 1: "occupied"}
    
    # COCO class IDs for vehicles (used with pretrained model fallback)
    VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    def __init__(self, model_path="models/best.pt", total_slots=50,
                 confidence=0.45, device="cpu"):
        """
        Args:
            model_path  : path to YOLOv11 .pt weights
            total_slots : max parking capacity
            confidence  : detection confidence threshold (0-1)
            device      : "cpu" or "cuda"
        """
        self.total_slots = total_slots
        self.confidence = confidence
        self.device = device
        self.model_path = model_path
        self.model = None
        self.model_name = "demo"
        self.use_custom_model = False  # Track if using trained model
        self._load_model()

    # ------------------------------------------------------------------ #
    #  LOAD MODEL
    # ------------------------------------------------------------------ #
    def _load_model(self):
        if not YOLO_OK:
            print("[Detection] Running in DEMO mode (no ultralytics).")
            return

        # First try our trained model
        try:
            if Path(self.model_path).exists():
                self.model = YOLO(self.model_path)
                self.model.to(self.device)
                self.model_name = str(self.model_path)
                self.use_custom_model = True
                print(f"[Detection] ✅ Trained model loaded: {self.model_path}  device={self.device}")
                print("[Detection] Using custom parking space detection model (empty/occupied)")
                return
        except Exception as err:
            print(f"[Detection] Cannot load trained model {self.model_path}: {err}")

        # Fallback to pretrained YOLO models
        for path in ["yolo11n.pt", "yolov8n.pt"]:
            try:
                self.model = YOLO(path)
                self.model.to(self.device)
                self.model_name = str(path)
                self.use_custom_model = False
                print(f"[Detection] Using pretrained model: {path}  device={self.device}")
                return
            except Exception as err:
                print(f"[Detection] Cannot load {path}: {err}")

        print("[Detection] No model found. Using DEMO mode.")
        self.model = None

    # ------------------------------------------------------------------ #
    #  DETECT PARKING SPACES IN A SINGLE FRAME
    # ------------------------------------------------------------------ #
    def detect(self, frame):
        """
        Run parking space detection on one frame.
        
        For custom trained model: detects empty/occupied parking spaces
        For pretrained model: detects vehicles in the frame

        Args:
            frame : BGR numpy array (H, W, 3)

        Returns dict:
            count          : number of detected items
            boxes          : list of [x1,y1,x2,y2]
            scores         : list of confidence scores
            labels         : list of class name strings
            annotated_frame: frame with drawn boxes and HUD
            timestamp      : ISO format string
            mode           : "yolo" (custom), "vehicle" (pretrained), or "demo"
        """
        if frame is None:
            return self._empty_result()

        if self.model is None:
            return self._demo_detect(frame)

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            verbose=False
        )

        boxes, scores, labels = [], [], []
        empty_count = 0
        occupied_count = 0
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                
                boxes.append([x1, y1, x2, y2])
                scores.append(round(conf, 3))
                
                if self.use_custom_model:
                    # Custom model: empty (0) or occupied (1)
                    label = self.PARKING_CLASSES.get(cls_id, f"class_{cls_id}")
                    labels.append(label)
                    if cls_id == 0:
                        empty_count += 1
                    elif cls_id == 1:
                        occupied_count += 1
                else:
                    # Pretrained model: vehicle detection
                    label = self.VEHICLE_CLASSES.get(cls_id, "vehicle")
                    labels.append(label)

        if self.use_custom_model:
            # For custom model, count is total parking spaces detected
            count = len(boxes)
            annotated = self._draw_parking_boxes(
                frame.copy(), boxes, scores, labels, 
                empty_count, occupied_count
            )
            mode = "yolo"
        else:
            # For pretrained model, count is vehicles
            count = min(len(boxes), self.total_slots)
            annotated = self._draw_boxes(frame.copy(), boxes, scores, labels, count)
            mode = "vehicle"

        return {
            "count": count,
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "empty_count": empty_count,
            "occupied_count": occupied_count,
            "annotated_frame": annotated,
            "timestamp": datetime.now().isoformat(),
            "mode": mode
        }

    # ------------------------------------------------------------------ #
    #  DRAW PARKING BOXES (for custom model)
    # ------------------------------------------------------------------ #
    def _draw_parking_boxes(self, frame, boxes, scores, labels, empty_count, occupied_count):
        """Draw bounding boxes with empty/occupied color coding."""
        h, w = frame.shape[:2]
        
        # Color scheme: green for empty, red for occupied
        colors = {"empty": (0, 255, 100), "occupied": (0, 100, 255)}

        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
            color = colors.get(label, (200, 200, 200))
            
            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label background
            txt = f"{label} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (x1, max(y1-18, 0)), (x1+tw+4, max(y1, 18)), color, -1)
            cv2.putText(frame, txt, (x1+2, max(y1-4, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

        # HUD panel
        available = self.total_slots - occupied_count
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "PARKINTEL  |  YOLOv11 PARKING DETECTION",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 100), 1)
        
        if self.use_custom_model:
            cv2.putText(frame, f"Empty: {empty_count}  Occupied: {occupied_count}  Total: {len(boxes)}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        else:
            cv2.putText(frame, f"Detected: {occupied_count}   Available: {available}/{self.total_slots}",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (w - 75, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        return frame

    # ------------------------------------------------------------------ #
    #  BATCH DETECT (list of image paths)
    # ------------------------------------------------------------------ #
    def detect_batch(self, frame_paths):
        """
        Process a list of image file paths.
        Returns list of detection result dicts.
        """
        results = []
        for i, path in enumerate(frame_paths):
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            res = self.detect(frame)
            res["frame_path"] = str(path)
            res["frame_index"] = i
            results.append(res)
            if i % 30 == 0:
                print(f"[Detection] {i}/{len(frame_paths)} frames  last_count={res['count']}")
        print(f"[Detection] Batch complete. {len(results)} frames processed.")
        return results

    # ------------------------------------------------------------------ #
    #  DRAW BOUNDING BOXES + HUD
    # ------------------------------------------------------------------ #
    def _draw_boxes(self, frame, boxes, scores, labels, count):
        h, w = frame.shape[:2]

        for (x1, y1, x2, y2), score, label in zip(boxes, scores, labels):
            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 255), 2)
            # Label background
            txt = f"{label} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.rectangle(frame, (x1, max(y1-18, 0)), (x1+tw+4, max(y1, 18)), (0, 210, 255), -1)
            cv2.putText(frame, txt, (x1+2, max(y1-4, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

        # HUD panel
        available = self.total_slots - count
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "PARKINTEL  |  YOLOv11 DETECTION",
                    (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 210, 255), 1)
        cv2.putText(frame, f"Detected: {count}   Available: {available}/{self.total_slots}",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, ts, (w - 75, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        return frame

    # ------------------------------------------------------------------ #
    #  DEMO MODE (no model)
    # ------------------------------------------------------------------ #
    def _demo_detect(self, frame):
        """
        Simulates realistic vehicle detection based on time-of-day patterns.
        Used when no YOLOv11 model is available.
        """
        hour = datetime.now().hour
        hourly_base = [8,5,4,4,5,9,14,28,36,40,42,44,46,44,42,40,42,47,48,45,40,34,26,15]
        base = hourly_base[hour % 24]
        np.random.seed(int(time.time() * 3) % 8000)
        count = max(0, min(self.total_slots, base + int(np.random.randint(-4, 5))))

        # Generate plausible bounding boxes matching the demo frame grid
        boxes, scores = [], []
        slot_w, slot_h = 68, 108
        cols_n = 10
        margin_x, margin_y = 30, 40

        for i in range(count):
            r, c = divmod(i, cols_n)
            x1 = margin_x + c * (slot_w + 6) + 7
            y1 = margin_y + r * (slot_h + 6) + 28 + 7  # account for HUD
            x2 = x1 + slot_w - 14
            y2 = y1 + slot_h - 14
            boxes.append([x1, y1, x2, y2])
            scores.append(round(0.70 + np.random.random() * 0.27, 3))

        labels = ["car"] * len(boxes)
        annotated = self._draw_boxes(frame.copy(), boxes, scores, labels, count)

        return {
            "count": count,
            "boxes": boxes,
            "scores": scores,
            "labels": labels,
            "annotated_frame": annotated,
            "timestamp": datetime.now().isoformat(),
            "mode": "demo"
        }

    def _empty_result(self):
        return {
            "count": 0, "boxes": [], "scores": [], "labels": [],
            "annotated_frame": None, "timestamp": datetime.now().isoformat(),
            "mode": "empty"
        }


# ---------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=== Module 2: Vehicle Detection Test ===")
    detector = VehicleDetector(model_path="models/best.pt", total_slots=50)
    print(f"Model: {detector.model_name}")

    # Test with synthetic frame
    frame = np.zeros((620, 820, 3), dtype=np.uint8)
    frame[:] = (35, 35, 35)
    result = detector.detect(frame)

    print(f"Detected: {result['count']} vehicles")
    print(f"Mode    : {result['mode']}")
    print(f"Boxes   : {len(result['boxes'])}")

    if result["annotated_frame"] is not None:
        cv2.imwrite("data/test_detection.jpg", result["annotated_frame"])
        print("Saved → data/test_detection.jpg")
    print("Module 2 OK")
