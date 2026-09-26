"""
MODULE 3: OCCUPANCY ESTIMATION
ParkIntel - AI Smart Parking System

Computes real-time and historical parking occupancy.
Saves structured CSV with engineered features for ML training.

Output CSV columns:
    timestamp, hour, day_of_week, is_weekend,
    occupancy, available, occupancy_rate, total_slots,
    hour_sin, hour_cos, day_sin, day_cos,
    occ_lag1, occ_lag2, occ_rolling3
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from datetime import datetime, timedelta


class OccupancyEstimator:

    def __init__(self, total_slots=50, csv_path="data/occupancy_log.csv"):
        """
        Args:
            total_slots : parking capacity
            csv_path    : path to save/load occupancy log CSV
        """
        self.total_slots = total_slots
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.records = []
        self._load_existing()

    # ------------------------------------------------------------------ #
    #  LOAD EXISTING DATA
    # ------------------------------------------------------------------ #
    def _load_existing(self):
        if self.csv_path.exists():
            try:
                df = pd.read_csv(self.csv_path)
                self.records = df.to_dict("records")
                print(f"[Occupancy] Loaded {len(self.records)} existing records from {self.csv_path}")
            except Exception as e:
                print(f"[Occupancy] Could not load CSV: {e}")
                self.records = []

    # ------------------------------------------------------------------ #
    #  UPDATE WITH ONE DETECTION RESULT
    # ------------------------------------------------------------------ #
    def update(self, detection_result, timestamp=None):
        """
        Process one detection result → compute occupancy → save to CSV.

        Args:
            detection_result : dict from VehicleDetector.detect()
            timestamp        : datetime object (default = now)

        Returns:
            dict with occupancy metrics
            
        Note:
            For custom YOLO model (parking spaces): uses occupied_count
            For vehicle detection: uses count as occupancy
        """
        ts = timestamp if timestamp is not None else datetime.now()
        
        # Get the appropriate count based on detection mode
        mode = detection_result.get("mode", "unknown")
        
        if mode == "yolo" and "occupied_count" in detection_result:
            # Custom parking space model: use occupied_count directly
            count = detection_result.get("occupied_count", 0)
        elif mode == "vehicle":
            # Vehicle detection: use count directly
            count = detection_result.get("count", 0)
        else:
            # Fallback: use count
            count = detection_result.get("count", 0)
        
        # Also track empty spaces if available
        empty_count = detection_result.get("empty_count", 0)
        
        count = max(0, min(self.total_slots, count))
        available = self.total_slots - count
        occ_rate = round(count / self.total_slots, 4)

        record = {
            "timestamp": ts.isoformat(),
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "is_weekend": int(ts.weekday() >= 5),
            "occupancy": count,
            "available": available,
            "occupancy_rate": occ_rate,
            "total_slots": self.total_slots,
            "empty_count": empty_count,
            "detection_mode": mode
        }
        self.records.append(record)
        self._append_csv(record)
        return record

    # ------------------------------------------------------------------ #
    #  BATCH UPDATE FROM DETECTION LIST
    # ------------------------------------------------------------------ #
    def update_batch(self, detection_results):
        """
        Process a list of detection dicts and return full DataFrame.
        """
        for res in detection_results:
            ts_str = res.get("timestamp", datetime.now().isoformat())
            try:
                ts = datetime.fromisoformat(ts_str)
            except Exception:
                ts = datetime.now()
            self.update(res, timestamp=ts)
        print(f"[Occupancy] Batch processed: {len(detection_results)} detections.")
        return self.get_dataframe()

    # ------------------------------------------------------------------ #
    #  GENERATE SYNTHETIC HISTORICAL DATA (30 days)
    # ------------------------------------------------------------------ #
    def generate_historical_data(self, days=30, save=True):
        """
        Creates realistic synthetic occupancy data for ML training.
        Simulates real parking patterns:
          - Low at night (midnight-5am)
          - Morning rush (7-9am)
          - Lunch peak (12-1pm)
          - Evening peak (5-7pm)
          - Weekends 25% less busy
        """
        print(f"[Occupancy] Generating {days} days of historical data...")
        np.random.seed(42)

        # 24-hour base occupancy pattern (fraction of capacity)
        hourly_pattern = np.array([
            0.10, 0.08, 0.06, 0.05, 0.06, 0.12,   # 00-05
            0.20, 0.46, 0.74, 0.82, 0.86, 0.90,   # 06-11
            0.94, 0.88, 0.84, 0.80, 0.84, 0.92,   # 12-17
            0.96, 0.90, 0.82, 0.68, 0.48, 0.26    # 18-23
        ])

        records = []
        base_date = datetime.now() - timedelta(days=days)

        for d in range(days):
            day_dt = base_date + timedelta(days=d)
            is_weekend = int(day_dt.weekday() >= 5)
            weekend_factor = 0.72 if is_weekend else 1.0

            for h in range(24):
                base_rate = hourly_pattern[h] * weekend_factor
                # Add realistic noise
                noise = np.random.normal(0, 0.04)
                occ_rate = float(np.clip(base_rate + noise, 0.0, 1.0))
                count = int(round(occ_rate * self.total_slots))
                available = self.total_slots - count

                # Vary the minute
                minute = int(np.random.randint(0, 60))
                ts = day_dt.replace(hour=h, minute=minute, second=0, microsecond=0)

                records.append({
                    "timestamp": ts.isoformat(),
                    "hour": h,
                    "day_of_week": day_dt.weekday(),
                    "is_weekend": is_weekend,
                    "occupancy": count,
                    "available": available,
                    "occupancy_rate": round(occ_rate, 4),
                    "total_slots": self.total_slots
                })

        df = pd.DataFrame(records)
        if save:
            df.to_csv(self.csv_path, index=False)
            print(f"[Occupancy] Saved {len(df)} records → {self.csv_path}")
        self.records = records
        return df

    # ------------------------------------------------------------------ #
    #  GET DATAFRAME WITH ENGINEERED FEATURES
    # ------------------------------------------------------------------ #
    def get_dataframe(self):
        """
        Returns clean DataFrame with all features needed for ML training.
        """
        if not self.records:
            return pd.DataFrame()

        df = pd.DataFrame(self.records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Cyclic time encodings (prevent hour 0 / hour 23 discontinuity)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_sin"]  = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"]  = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Lag features (previous occupancy values)
        df["occ_lag1"]     = df["occupancy"].shift(1).fillna(df["occupancy"].mean())
        df["occ_lag2"]     = df["occupancy"].shift(2).fillna(df["occupancy"].mean())
        df["occ_rolling3"] = df["occupancy"].rolling(3, min_periods=1).mean()

        return df

    # ------------------------------------------------------------------ #
    #  CURRENT METRICS
    # ------------------------------------------------------------------ #
    def current_metrics(self):
        if not self.records:
            return {
                "occupancy": 0,
                "available": self.total_slots,
                "occupancy_rate": 0.0,
                "total_slots": self.total_slots,
                "timestamp": datetime.now().isoformat()
            }
        last = self.records[-1]
        return {
            "occupancy": last.get("occupancy", 0),
            "available": last.get("available", self.total_slots),
            "occupancy_rate": last.get("occupancy_rate", 0.0),
            "total_slots": self.total_slots,
            "timestamp": last.get("timestamp", "")
        }

    # ------------------------------------------------------------------ #
    #  STATS SUMMARY
    # ------------------------------------------------------------------ #
    def summary(self):
        df = self.get_dataframe()
        if df.empty:
            return {}
        return {
            "total_records": len(df),
            "date_range": f"{df['timestamp'].min()}  to  {df['timestamp'].max()}",
            "avg_occupancy": round(float(df["occupancy"].mean()), 1),
            "max_occupancy": int(df["occupancy"].max()),
            "min_occupancy": int(df["occupancy"].min()),
            "avg_occ_rate":  round(float(df["occupancy_rate"].mean()), 3),
        }

    def _append_csv(self, record):
        """Append single record to CSV (incremental write)."""
        row = pd.DataFrame([record])
        header = not self.csv_path.exists()
        row.to_csv(self.csv_path, mode="a", index=False, header=header)

    def save_all(self):
        df = pd.DataFrame(self.records)
        df.to_csv(self.csv_path, index=False)
        print(f"[Occupancy] Saved {len(df)} records → {self.csv_path}")


# ---------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=== Module 3: Occupancy Estimation Test ===")
    est = OccupancyEstimator(total_slots=50, csv_path="data/occupancy_log.csv")
    df = est.generate_historical_data(days=30, save=True)
    print(f"Shape: {df.shape}")
    df2 = est.get_dataframe()
    print(df2[["timestamp","hour","occupancy","available","occ_lag1"]].tail(10).to_string())
    print("Summary:", est.summary())
    print("Module 3 OK")
