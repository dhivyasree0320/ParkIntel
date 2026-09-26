"""
MODULE 5: PEAK HOUR FORECASTING
ParkIntel - AI Smart Parking System

Analyzes historical occupancy data to identify:
  - Peak congestion hours and windows
  - Busiest / quietest days of week
  - Weekday vs weekend patterns
  - Top-5 congested hours
  - Congestion distribution (low / medium / high)
"""

import numpy as np
import pandas as pd
from datetime import datetime


class PeakHourAnalyzer:

    def __init__(self, total_slots=50, peak_threshold=0.75):
        """
        Args:
            total_slots     : parking capacity
            peak_threshold  : occupancy rate above which = peak (0-1)
        """
        self.total_slots = total_slots
        self.peak_threshold = peak_threshold

    # ------------------------------------------------------------------ #
    #  MAIN ANALYSIS
    # ------------------------------------------------------------------ #
    def analyze(self, df):
        """
        Full statistical analysis of occupancy DataFrame.

        Args:
            df : DataFrame from OccupancyEstimator.get_dataframe()

        Returns:
            dict with all peak-hour analysis results
        """
        if df is None or df.empty:
            return self._demo_analysis()

        result = {}

        # ── Hourly averages ──────────────────────────────────────────────
        hourly_avg = df.groupby("hour")["occupancy"].mean()
        hourly_std = df.groupby("hour")["occupancy"].std().fillna(0)
        hourly_max = df.groupby("hour")["occupancy"].max()

        result["hourly_avg"] = {int(h): round(float(v), 2) for h, v in hourly_avg.items()}
        result["hourly_std"] = {int(h): round(float(v), 2) for h, v in hourly_std.items()}
        result["hourly_max"] = {int(h): int(v) for h, v in hourly_max.items()}
        result["hourly_rate"] = {int(h): round(float(v)/self.total_slots, 3)
                                 for h, v in hourly_avg.items()}

        # ── Peak hours ───────────────────────────────────────────────────
        peak_hours = sorted([
            int(h) for h, rate in result["hourly_rate"].items()
            if rate >= self.peak_threshold
        ])
        result["peak_hours"] = peak_hours
        result["peak_hour_max"] = int(hourly_avg.idxmax())
        result["peak_occupancy_max"] = round(float(hourly_avg.max()), 1)
        result["off_peak_hour"] = int(hourly_avg.idxmin())
        result["off_peak_occupancy"] = round(float(hourly_avg.min()), 1)

        if peak_hours:
            result["peak_window_label"] = (
                f"{self._hour_label(min(peak_hours))} – "
                f"{self._hour_label(max(peak_hours) + 1)}"
            )
        else:
            ph = result["peak_hour_max"]
            result["peak_window_label"] = (
                f"{self._hour_label(ph)} – {self._hour_label(ph + 1)}"
            )

        # ── Top-5 peak hours ─────────────────────────────────────────────
        top5 = hourly_avg.nlargest(5)
        result["top5_peak_hours"] = [
            {
                "hour":      int(h),
                "label":     self._hour_label(h),
                "avg_occ":   round(float(v), 1),
                "rate_pct":  round(float(v) / self.total_slots * 100, 1)
            }
            for h, v in top5.items()
        ]

        # ── Day-of-week analysis ─────────────────────────────────────────
        day_names = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_avg = df.groupby("day_of_week")["occupancy"].mean()
        result["daily_avg"] = {
            day_names[int(i)]: round(float(v), 1)
            for i, v in day_avg.items() if int(i) < 7
        }
        result["busiest_day"]  = day_names[int(day_avg.idxmax())] if not day_avg.empty else "Friday"
        result["quietest_day"] = day_names[int(day_avg.idxmin())] if not day_avg.empty else "Sunday"

        # ── Weekday vs weekend ───────────────────────────────────────────
        if "is_weekend" in df.columns:
            wd = df[df["is_weekend"] == 0]["occupancy"]
            we = df[df["is_weekend"] == 1]["occupancy"]
            result["weekday_avg"] = round(float(wd.mean()), 1) if len(wd) > 0 else 0
            result["weekend_avg"] = round(float(we.mean()), 1) if len(we) > 0 else 0
        else:
            result["weekday_avg"] = 0
            result["weekend_avg"] = 0

        # ── Congestion distribution ──────────────────────────────────────
        rate_series = df["occupancy"] / self.total_slots
        result["low_pct"]    = round(float((rate_series < 0.30).mean() * 100), 1)
        result["medium_pct"] = round(float(((rate_series >= 0.30) & (rate_series < 0.70)).mean() * 100), 1)
        result["high_pct"]   = round(float((rate_series >= 0.70).mean() * 100), 1)

        # ── Heatmap (day × hour) ─────────────────────────────────────────
        if "day_of_week" in df.columns:
            pivot = df.pivot_table(
                index="day_of_week", columns="hour",
                values="occupancy_rate", aggfunc="mean"
            ).fillna(0)
            result["heatmap_data"] = pivot.round(3).values.tolist()
            result["heatmap_hours"] = [int(c) for c in pivot.columns]
            result["heatmap_days"]  = [day_names[int(i)] for i in pivot.index if int(i) < 7]

        return result

    # ------------------------------------------------------------------ #
    #  CURRENT MOMENT STRESS LEVEL
    # ------------------------------------------------------------------ #
    def current_stress(self, current_occupancy):
        """Quick stress assessment for the current moment."""
        rate = current_occupancy / self.total_slots
        hour = datetime.now().hour
        hourly_rates = [0.10,0.08,0.06,0.05,0.06,0.12,0.20,0.46,0.74,0.82,
                        0.86,0.90,0.94,0.88,0.84,0.80,0.84,0.92,0.96,0.90,
                        0.82,0.68,0.48,0.26]
        is_peak = hourly_rates[hour] >= self.peak_threshold
        return {
            "occupancy_rate": round(rate, 3),
            "is_peak_hour":   is_peak,
            "hour":           hour,
            "stress_level":   "high" if rate >= 0.70 else "medium" if rate >= 0.30 else "low"
        }

    # ------------------------------------------------------------------ #
    #  HELPERS
    # ------------------------------------------------------------------ #
    def _hour_label(self, h):
        h = int(h) % 24
        suffix = "AM" if h < 12 else "PM"
        disp   = h if h <= 12 else h - 12
        if disp == 0: disp = 12
        return f"{disp} {suffix}"

    def _demo_analysis(self):
        pattern = [8,5,4,4,5,9,14,28,36,40,42,44,46,44,42,40,42,47,48,45,40,34,26,15]
        return {
            "hourly_avg": {h: pattern[h] for h in range(24)},
            "hourly_rate": {h: round(pattern[h]/50, 3) for h in range(24)},
            "peak_hours": [17, 18, 19],
            "peak_hour_max": 18,
            "peak_occupancy_max": 48.0,
            "off_peak_hour": 3,
            "off_peak_occupancy": 4.0,
            "peak_window_label": "5 PM – 8 PM",
            "top5_peak_hours": [
                {"hour":18,"label":"6 PM","avg_occ":48,"rate_pct":96},
                {"hour":17,"label":"5 PM","avg_occ":47,"rate_pct":94},
                {"hour":19,"label":"7 PM","avg_occ":45,"rate_pct":90},
                {"hour":12,"label":"12 PM","avg_occ":46,"rate_pct":92},
                {"hour":11,"label":"11 AM","avg_occ":44,"rate_pct":88}
            ],
            "daily_avg": {"Monday":32,"Tuesday":36,"Wednesday":38,"Thursday":41,
                          "Friday":44,"Saturday":29,"Sunday":20},
            "busiest_day":  "Friday",
            "quietest_day": "Sunday",
            "weekday_avg":  35.2,
            "weekend_avg":  24.8,
            "low_pct":    24.0,
            "medium_pct": 34.0,
            "high_pct":   42.0
        }


# ---------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from modules.module3_occupancy import OccupancyEstimator

    print("=== Module 5: Peak Hour Analyzer Test ===")
    est = OccupancyEstimator(total_slots=50, csv_path="data/occupancy_log.csv")
    df = est.generate_historical_data(days=30, save=False)

    analyzer = PeakHourAnalyzer(total_slots=50)
    result = analyzer.analyze(df)

    print(f"Peak window : {result['peak_window_label']}")
    print(f"Busiest day : {result['busiest_day']}")
    print(f"Quietest day: {result['quietest_day']}")
    print(f"Weekday avg : {result['weekday_avg']}")
    print(f"Weekend avg : {result['weekend_avg']}")
    print(f"Congestion  : Low={result['low_pct']}%  Med={result['medium_pct']}%  High={result['high_pct']}%")
    print("Top 5 peaks:")
    for p in result["top5_peak_hours"]:
        print(f"  {p['label']:8s}  avg={p['avg_occ']:5.1f}  ({p['rate_pct']}%)")
    print("Module 5 OK")
