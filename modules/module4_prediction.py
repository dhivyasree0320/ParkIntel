"""
MODULE 4: PREDICTION MODULE (Random Forest)
ParkIntel - AI Smart Parking System

Trains a Random Forest Regressor on historical occupancy data.
Predicts:
  - Available slots for next 1-6 hours
  - Occupancy at any future time point
  - Full 24-hour day forecast

Features used for prediction:
    hour, day_of_week, is_weekend,
    hour_sin, hour_cos, day_sin, day_cos,
    occ_lag1, occ_lag2, occ_rolling3
"""

import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
from datetime import datetime, timedelta

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[Predictor] scikit-learn not installed. pip install scikit-learn")


class ParkingPredictor:

    FEATURES = [
        "hour", "day_of_week", "is_weekend",
        "hour_sin", "hour_cos",
        "day_sin", "day_cos",
        "occ_lag1", "occ_lag2", "occ_rolling3"
    ]
    TARGET = "occupancy"

    def __init__(self, model_path="models/rf_model.pkl", total_slots=50):
        """
        Args:
            model_path  : path to save/load trained model (.pkl)
            total_slots : parking capacity (for clamping predictions)
        """
        self.model_path = Path(model_path)
        self.total_slots = total_slots
        self.model = None
        self.scaler = None
        self.metrics_ = {}
        self.feature_importances_ = {}
        self._try_load()

    # ------------------------------------------------------------------ #
    #  TRAIN MODEL
    # ------------------------------------------------------------------ #
    def train(self, df):
        """
        Train Random Forest on occupancy DataFrame.

        Args:
            df : DataFrame from OccupancyEstimator.get_dataframe()

        Returns:
            dict of performance metrics
        """
        if not SKLEARN_OK:
            print("[Predictor] Cannot train: scikit-learn not available.")
            return {}

        # Drop rows with NaN in required columns
        df_clean = df.dropna(subset=self.FEATURES + [self.TARGET]).copy()
        if len(df_clean) < 50:
            print(f"[Predictor] Not enough data ({len(df_clean)} rows). Need ≥50.")
            return {}

        X = df_clean[self.FEATURES].values
        y = df_clean[self.TARGET].values.astype(float)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, shuffle=True
        )

        # Standardize features
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        # Train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train_s, y_train)

        # Evaluate
        y_pred_train = self.model.predict(X_train_s)
        y_pred_test  = self.model.predict(X_test_s)

        mae  = mean_absolute_error(y_test, y_pred_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))
        r2   = float(r2_score(y_test, y_pred_test))
        acc  = max(0.0, 1.0 - mae / self.total_slots)

        self.metrics_ = {
            "mae":          round(mae, 3),
            "rmse":         round(rmse, 3),
            "r2":           round(r2, 4),
            "accuracy_pct": round(acc * 100, 2),
            "train_size":   len(X_train),
            "test_size":    len(X_test),
            "features":     self.FEATURES
        }

        # Feature importances
        raw_imp = self.model.feature_importances_
        self.feature_importances_ = {
            f: round(float(v), 5)
            for f, v in zip(self.FEATURES, raw_imp)
        }

        print(f"[Predictor] Training complete!")
        print(f"  MAE={mae:.3f}  RMSE={rmse:.3f}  R²={r2:.4f}  Accuracy={acc*100:.1f}%")
        print(f"  Train={len(X_train)}  Test={len(X_test)}")

        self.save()
        return self.metrics_

    # ------------------------------------------------------------------ #
    #  PREDICT NEXT N HOURS
    # ------------------------------------------------------------------ #
    def predict_next_hours(self, current_occupancy, hours=6, start_time=None):
        """
        Predict occupancy for the next `hours` hours.

        Returns:
            list of dicts with timestamp, predicted_occupancy, predicted_available
        """
        if not SKLEARN_OK or self.model is None:
            return self._demo_forecast(current_occupancy, hours, start_time)

        now = start_time or datetime.now()
        predictions = []
        prev1 = float(current_occupancy)
        prev2 = float(current_occupancy)
        rolling = float(current_occupancy)

        for i in range(1, hours + 1):
            ft = now + timedelta(hours=i)
            h   = ft.hour
            dow = ft.weekday()
            is_we = int(dow >= 5)

            feat = np.array([[
                h, dow, is_we,
                np.sin(2 * np.pi * h / 24),
                np.cos(2 * np.pi * h / 24),
                np.sin(2 * np.pi * dow / 7),
                np.cos(2 * np.pi * dow / 7),
                prev1, prev2, rolling
            ]])
            feat_s = self.scaler.transform(feat)
            pred = float(self.model.predict(feat_s)[0])
            pred = max(0, min(self.total_slots, round(pred)))

            predictions.append({
                "hour_offset":           i,
                "timestamp":             ft.strftime("%H:%M"),
                "datetime":              ft.isoformat(),
                "predicted_occupancy":   int(pred),
                "predicted_available":   int(self.total_slots - pred),
                "occupancy_rate":        round(pred / self.total_slots, 3)
            })

            # Update lag features for next iteration
            rolling = (rolling * i + pred) / (i + 1)
            prev2 = prev1
            prev1 = pred

        return predictions

    # ------------------------------------------------------------------ #
    #  PREDICT SINGLE TIME POINT
    # ------------------------------------------------------------------ #
    def predict_one(self, hour, day_of_week, occ_lag1, occ_lag2=None, rolling=None):
        """Predict occupancy at a specific hour/day."""
        if not SKLEARN_OK or self.model is None:
            return self._demo_single(hour)
        if occ_lag2 is None:  occ_lag2 = occ_lag1
        if rolling is None:   rolling  = occ_lag1
        is_we = int(day_of_week >= 5)

        feat = np.array([[
            hour, day_of_week, is_we,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
            occ_lag1, occ_lag2, rolling
        ]])
        feat_s = self.scaler.transform(feat)
        pred = float(self.model.predict(feat_s)[0])
        return max(0, min(self.total_slots, int(round(pred))))

    # ------------------------------------------------------------------ #
    #  FULL 24-HOUR FORECAST
    # ------------------------------------------------------------------ #
    def forecast_day(self, base_occupancy, base_day_of_week=None):
        """Returns predicted occupancy for all 24 hours of a day."""
        base_dow = base_day_of_week if base_day_of_week is not None else datetime.now().weekday()
        results = []
        for h in range(24):
            pred = self.predict_one(h, base_dow, base_occupancy)
            results.append({
                "hour": h,
                "label": f"{h:02d}:00",
                "predicted_occupancy": pred,
                "predicted_available": self.total_slots - pred,
                "occupancy_rate": round(pred / self.total_slots, 3)
            })
        return results

    # ------------------------------------------------------------------ #
    #  SAVE / LOAD
    # ------------------------------------------------------------------ #
    def save(self):
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model":       self.model,
            "scaler":      self.scaler,
            "metrics":     self.metrics_,
            "importances": self.feature_importances_,
            "total_slots": self.total_slots,
            "features":    self.FEATURES
        }
        joblib.dump(payload, self.model_path)
        print(f"[Predictor] Model saved → {self.model_path}")

    def _try_load(self):
        if self.model_path.exists() and SKLEARN_OK:
            try:
                data = joblib.load(self.model_path)
                self.model               = data["model"]
                self.scaler              = data["scaler"]
                self.metrics_            = data.get("metrics", {})
                self.feature_importances_= data.get("importances", {})
                print(f"[Predictor] Loaded saved model ← {self.model_path}")
                print(f"  Accuracy={self.metrics_.get('accuracy_pct','?')}%  "
                      f"R²={self.metrics_.get('r2','?')}")
            except Exception as e:
                print(f"[Predictor] Load failed: {e}")

    # ------------------------------------------------------------------ #
    #  DEMO FALLBACKS (when no model)
    # ------------------------------------------------------------------ #
    _HOUR_PATTERN = [8,5,4,4,5,9,14,28,36,40,42,44,46,44,42,40,42,47,48,45,40,34,26,15]

    def _demo_single(self, hour):
        return self._HOUR_PATTERN[hour % 24]

    def _demo_forecast(self, current_occ, hours, start_time):
        now = start_time or datetime.now()
        results = []
        for i in range(1, hours + 1):
            ft = now + timedelta(hours=i)
            pred = min(self.total_slots, self._HOUR_PATTERN[ft.hour % 24])
            pred = max(0, pred + np.random.randint(-2, 3))
            results.append({
                "hour_offset":         i,
                "timestamp":           ft.strftime("%H:%M"),
                "datetime":            ft.isoformat(),
                "predicted_occupancy": pred,
                "predicted_available": self.total_slots - pred,
                "occupancy_rate":      round(pred / self.total_slots, 3)
            })
        return results


# ---------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from modules.module3_occupancy import OccupancyEstimator

    print("=== Module 4: Prediction Test ===")
    est = OccupancyEstimator(total_slots=50, csv_path="data/occupancy_log.csv")
    df  = est.generate_historical_data(days=30, save=True)

    predictor = ParkingPredictor(model_path="models/rf_model.pkl", total_slots=50)
    df_feat = est.get_dataframe()
    metrics = predictor.train(df_feat)

    print("\nMetrics:", metrics)
    print("\nFeature importances:")
    for k, v in sorted(predictor.feature_importances_.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v:.4f}  {'█' * int(v*60)}")

    print("\n6-Hour Forecast (current occupancy = 35):")
    for p in predictor.predict_next_hours(35, hours=6):
        print(f"  +{p['hour_offset']}h {p['timestamp']}  occ={p['predicted_occupancy']:2d}  "
              f"avail={p['predicted_available']:2d}  rate={p['occupancy_rate']:.2f}")
    print("Module 4 OK")
