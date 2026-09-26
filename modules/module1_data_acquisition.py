"""
MODULE 1: DATA ACQUISITION
ParkIntel - AI Smart Parking System

Handles all video input sources:
- Local video file (MP4, AVI, MOV)
- Webcam / USB camera (source=0, 1, 2...)
- RTSP / CCTV stream (source="rtsp://...")
- Demo mode (source=None) - generates synthetic parking frames
"""

import cv2
import numpy as np
import os
import time
from pathlib import Path
from datetime import datetime


class DataAcquisition:

    def __init__(self, source=None, output_dir="data/frames", frame_skip=5, total_slots=50):
        """
        Args:
            source      : None=demo, 0=webcam, "path.mp4"=file, "rtsp://..."=stream
            output_dir  : folder to save extracted frames
            frame_skip  : save every N-th frame
            total_slots : total parking capacity
        """
        self.source = source
        self.output_dir = Path(output_dir)
        self.frame_skip = frame_skip
        self.total_slots = total_slots
        self.cap = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"[DataAcquisition] Source={source}  Slots={total_slots}")

    # ------------------------------------------------------------------ #
    #  OPEN SOURCE
    # ------------------------------------------------------------------ #
    def open(self):
        if self.source is None:
            print("[DataAcquisition] Running in DEMO mode (synthetic frames).")
            return True
        src = int(self.source) if str(self.source).isdigit() else self.source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            print(f"[DataAcquisition] ERROR: Cannot open source '{self.source}'")
            return False
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        print(f"[DataAcquisition] Opened  FPS={fps:.1f}  TotalFrames={total}")
        return True

    # ------------------------------------------------------------------ #
    #  EXTRACT FRAMES FROM VIDEO FILE (batch)
    # ------------------------------------------------------------------ #
    def extract_frames(self, max_frames=300):
        """
        Extract frames from video file or generate demo frames.
        Returns list of saved image paths.
        """
        if self.source is None:
            return self._generate_demo_frames(max_frames)

        if self.cap is None:
            self.open()

        saved = []
        idx = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if idx % self.frame_skip == 0:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = self.output_dir / f"frame_{idx:06d}_{ts}.jpg"
                cv2.imwrite(str(path), frame)
                saved.append(str(path))
                if len(saved) >= max_frames:
                    break
            idx += 1

        print(f"[DataAcquisition] Extracted {len(saved)} frames → {self.output_dir}")
        return saved

    # ------------------------------------------------------------------ #
    #  LIVE FRAME GENERATOR (real-time)
    # ------------------------------------------------------------------ #
    def live_frames(self):
        """
        Generator that yields (frame, timestamp) tuples continuously.
        Use this for real-time detection loop.
        """
        if self.source is None:
            # Demo mode: yield synthetic frames every 0.5s
            while True:
                frame = self._make_demo_frame()
                yield frame, datetime.now()
                time.sleep(0.5)
            return

        if self.cap is None:
            self.open()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("[DataAcquisition] Stream ended.")
                break
            yield frame, datetime.now()

    # ------------------------------------------------------------------ #
    #  GET SINGLE FRAME
    # ------------------------------------------------------------------ #
    def get_frame(self):
        """Returns a single (frame, timestamp) tuple."""
        if self.source is None:
            return self._make_demo_frame(), datetime.now()
        if self.cap is None:
            self.open()
        ret, frame = self.cap.read()
        if ret:
            return frame, datetime.now()
        return None, None

    # ------------------------------------------------------------------ #
    #  DEMO FRAME GENERATOR
    # ------------------------------------------------------------------ #
    def _make_demo_frame(self, width=820, height=620):
        """
        Generates a synthetic parking lot image with random cars.
        Used when no real video source is available.
        """
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (35, 35, 35)  # dark asphalt

        slot_w, slot_h = 68, 108
        cols_n, rows_n = 10, 5
        margin_x, margin_y = 30, 40

        # Realistic hourly occupancy seed
        hour = datetime.now().hour
        hourly_base = [8,5,4,4,5,9,14,28,36,40,42,44,46,44,42,40,42,47,48,45,40,34,26,15]
        base_rate = hourly_base[hour] / 50.0
        np.random.seed(int(time.time()) % 5000)

        car_colors = [
            (30,30,180),(180,30,30),(30,160,30),
            (160,140,20),(30,150,180),(140,30,140),
            (180,100,30),(80,80,200)
        ]

        for r in range(rows_n):
            for c in range(cols_n):
                x1 = margin_x + c * (slot_w + 6)
                y1 = margin_y + r * (slot_h + 6)
                x2, y2 = x1 + slot_w, y1 + slot_h

                # White slot lines
                cv2.rectangle(frame, (x1, y1), (x2, y2), (90, 90, 90), 1)

                # Place car based on base rate + noise
                prob = base_rate + np.random.uniform(-0.15, 0.15)
                if np.random.random() < prob:
                    color = car_colors[np.random.randint(0, len(car_colors))]
                    pad = 7
                    # Car body
                    cv2.rectangle(frame, (x1+pad, y1+pad), (x2-pad, y2-pad), color, -1)
                    # Car roof (darker rectangle in middle)
                    rx1 = x1 + pad + 6
                    ry1 = y1 + pad + 16
                    rx2 = x2 - pad - 6
                    ry2 = y2 - pad - 16
                    darker = tuple(max(0, c-50) for c in color)
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), darker, -1)
                    # Outline
                    cv2.rectangle(frame, (x1+pad, y1+pad), (x2-pad, y2-pad), (210,210,210), 1)

        # HUD overlay
        ts_str = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.rectangle(frame, (0, 0), (width, 28), (0, 0, 0), -1)
        cv2.putText(frame, f"PARKING LOT CCTV  |  {ts_str}",
                    (10, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        cv2.putText(frame, "REC", (width-50, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 220), 1)
        return frame

    def _generate_demo_frames(self, n):
        paths = []
        for i in range(n):
            frame = self._make_demo_frame()
            path = self.output_dir / f"demo_{i:04d}.jpg"
            cv2.imwrite(str(path), frame)
            paths.append(str(path))
        print(f"[DataAcquisition] Generated {n} demo frames → {self.output_dir}")
        return paths

    # ------------------------------------------------------------------ #
    #  VIDEO INFO
    # ------------------------------------------------------------------ #
    def get_info(self):
        if self.cap is None:
            return {"source": "demo", "fps": 0, "total_frames": 0}
        return {
            "source": self.source,
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }

    def release(self):
        if self.cap:
            self.cap.release()
            print("[DataAcquisition] Released video capture.")


# ---------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=== Module 1: Data Acquisition Test ===")
    acq = DataAcquisition(source=None, output_dir="data/frames", total_slots=50)
    frames = acq.extract_frames(max_frames=5)
    print(f"Saved {len(frames)} demo frames")
    for p in frames:
        img = cv2.imread(p)
        print(f"  {p}  shape={img.shape}")
    print("Module 1 OK")
