"""
MODULE 6: PARKING STRESS INDEX (PSI)
ParkIntel - AI Smart Parking System

YOUR UNIQUE INNOVATION.

PSI Formula:
    PSI = (Current Occupancy / Total Capacity) × (Predicted Demand / Total Capacity)

Levels:
    PSI < 0.30   → LOW    (green)  – Comfortable availability
    PSI 0.30-0.69→ MEDIUM (yellow) – Monitor closely
    PSI ≥ 0.70   → HIGH   (red)    – Critical, action required

Time-of-day multiplier is applied to weight peak-hour PSI higher.
"""

import numpy as np
import pandas as pd
from datetime import datetime


class ParkingStressIndex:

    LOW_THRESHOLD  = 0.30
    HIGH_THRESHOLD = 0.70

    COLORS = {
        "low":    "#7cff6b",
        "medium": "#ffcc00",
        "high":   "#ff3d5a"
    }
    EMOJI = {
        "low":    "🟢",
        "medium": "🟡",
        "high":   "🔴"
    }

    def __init__(self, total_slots=50):
        self.total_slots = total_slots
        self.history = []   # rolling PSI history (last 1000 entries)

    # ------------------------------------------------------------------ #
    #  COMPUTE PSI (single reading)
    # ------------------------------------------------------------------ #
    def compute(self, current_occupancy, predicted_demand,
                current_time=None, apply_time_factor=True):
        """
        Compute PSI for a single moment.

        Args:
            current_occupancy : vehicles detected right now
            predicted_demand  : ML-predicted occupancy for next hour
            current_time      : datetime (default = now)
            apply_time_factor : weight PSI higher during peak hours

        Returns dict:
            psi, level, color, emoji, occ_rate, pred_rate,
            time_factor, available, recommendation, timestamp
        """
        ts  = current_time or datetime.now()
        cap = self.total_slots

        occ_rate  = max(0.0, min(1.0, current_occupancy / cap))
        pred_rate = max(0.0, min(1.0, predicted_demand  / cap))

        psi_raw = occ_rate * pred_rate

        # Apply time-of-day multiplier
        time_factor = self._time_multiplier(ts.hour) if apply_time_factor else 1.0
        psi = min(1.0, psi_raw * time_factor)

        level, color = self._classify(psi)
        emoji = self.EMOJI[level]

        entry = {
            "timestamp":          ts.isoformat(),
            "psi":                round(psi, 4),
            "psi_raw":            round(psi_raw, 4),
            "occ_rate":           round(occ_rate, 4),
            "pred_rate":          round(pred_rate, 4),
            "time_factor":        round(time_factor, 3),
            "level":              level,
            "current_occupancy":  current_occupancy,
            "predicted_demand":   predicted_demand
        }
        self.history.append(entry)
        if len(self.history) > 1000:
            self.history = self.history[-500:]

        recommendation = self._build_recommendation(
            psi, level, occ_rate, pred_rate,
            current_occupancy, cap - current_occupancy, ts
        )

        return {
            "psi":                round(psi, 4),
            "psi_raw":            round(psi_raw, 4),
            "psi_pct":            round(psi * 100, 1),
            "level":              level,
            "color":              color,
            "emoji":              emoji,
            "occ_rate":           round(occ_rate, 4),
            "pred_rate":          round(pred_rate, 4),
            "time_factor":        round(time_factor, 3),
            "current_occupancy":  current_occupancy,
            "predicted_demand":   predicted_demand,
            "available":          cap - current_occupancy,
            "total_slots":        cap,
            "timestamp":          ts.isoformat(),
            "recommendation":     recommendation
        }

    # ------------------------------------------------------------------ #
    #  COMPUTE PSI FOR ENTIRE DATAFRAME
    # ------------------------------------------------------------------ #
    def compute_series(self, df, predicted_col="occ_lag1"):
        """
        Compute PSI for each row of an occupancy DataFrame.
        Returns a list of PSI values (same length as df).
        """
        psi_values = []
        for _, row in df.iterrows():
            occ  = int(row.get("occupancy", 0))
            pred = int(row.get(predicted_col, occ))
            ts_val = row.get("timestamp", datetime.now())
            try:
                ts = pd.to_datetime(ts_val)
                ts = ts.to_pydatetime()
            except Exception:
                ts = datetime.now()
            result = self.compute(occ, pred, current_time=ts, apply_time_factor=False)
            psi_values.append(result["psi"])
        return psi_values

    # ------------------------------------------------------------------ #
    #  WHAT-IF SIMULATION
    # ------------------------------------------------------------------ #
    def simulate(self, demand_increase_pct=0, new_slots=0,
                 base_occupancy=None, base_predicted=None):
        """
        Run a what-if scenario.

        Args:
            demand_increase_pct : % increase in parking demand
            new_slots           : additional parking slots to add
            base_occupancy      : current vehicles (default = 64% of capacity)
            base_predicted      : predicted demand (default = 88% of capacity)

        Returns dict with before/after PSI and recommendation.
        """
        base_occ  = base_occupancy  if base_occupancy  is not None else self.total_slots * 0.64
        base_pred = base_predicted  if base_predicted  is not None else self.total_slots * 0.88

        new_cap   = self.total_slots + new_slots
        proj_occ  = min(new_cap, base_occ  * (1 + demand_increase_pct / 100))
        proj_pred = min(new_cap, base_pred * (1 + demand_increase_pct / 100))

        orig_occ_rate  = base_occ  / self.total_slots
        orig_pred_rate = base_pred / self.total_slots
        orig_psi = min(1.0, orig_occ_rate * orig_pred_rate)

        new_occ_rate  = proj_occ  / new_cap
        new_pred_rate = proj_pred / new_cap
        new_psi = min(1.0, new_occ_rate * new_pred_rate)

        orig_level, orig_color = self._classify(orig_psi)
        new_level,  new_color  = self._classify(new_psi)

        return {
            "scenario": {
                "demand_increase_pct": demand_increase_pct,
                "new_slots_added":     new_slots,
                "new_capacity":        int(new_cap)
            },
            "before": {
                "psi":       round(orig_psi, 4),
                "level":     orig_level,
                "color":     orig_color,
                "occupancy": int(base_occ),
                "available": int(self.total_slots - base_occ),
                "capacity":  self.total_slots
            },
            "after": {
                "psi":       round(new_psi, 4),
                "level":     new_level,
                "color":     new_color,
                "occupancy": int(proj_occ),
                "available": int(new_cap - proj_occ),
                "capacity":  int(new_cap)
            },
            "delta_psi": round(new_psi - orig_psi, 4),
            "recommendation": self._sim_recommendation(
                new_psi, new_level, demand_increase_pct, new_slots, int(new_cap)
            )
        }

    # ------------------------------------------------------------------ #
    #  PSI HISTORY
    # ------------------------------------------------------------------ #
    def get_history_df(self):
        """Returns DataFrame of PSI history."""
        return pd.DataFrame(self.history) if self.history else pd.DataFrame()

    def rolling_avg_psi(self, window=5):
        """Rolling average of last `window` PSI readings."""
        if len(self.history) < 2:
            return 0.0
        vals = [h["psi"] for h in self.history[-window:]]
        return round(float(np.mean(vals)), 4)

    # ------------------------------------------------------------------ #
    #  PRIVATE HELPERS
    # ------------------------------------------------------------------ #
    def _classify(self, psi):
        if psi < self.LOW_THRESHOLD:
            return "low", self.COLORS["low"]
        elif psi < self.HIGH_THRESHOLD:
            return "medium", self.COLORS["medium"]
        else:
            return "high", self.COLORS["high"]

    def _time_multiplier(self, hour):
        """Apply weight based on time of day."""
        peak_windows = {
            (7,  9):  1.10,
            (11, 13): 1.08,
            (17, 20): 1.15,
        }
        for (start, end), mult in peak_windows.items():
            if start <= hour < end:
                return mult
        return 1.0

    def _build_recommendation(self, psi, level, occ_rate, pred_rate,
                               occ, avail, ts):
        hour = ts.hour
        is_peak = (8 <= hour < 10) or (12 <= hour < 14) or (17 <= hour < 20)
        peak_note = " (Peak hour active)" if is_peak else ""

        if level == "high":
            return (
                f"⚠ CRITICAL – PSI={psi:.3f}{peak_note}. "
                f"Parking is {occ_rate*100:.0f}% full with demand forecast at "
                f"{pred_rate*100:.0f}%. "
                f"Actions: Activate overflow lot immediately, enable dynamic pricing, "
                f"deploy digital wayfinding signage, alert parking staff."
            )
        elif level == "medium":
            return (
                f"⚡ MODERATE – PSI={psi:.3f}{peak_note}. "
                f"Occupancy at {occ_rate*100:.0f}%  ({avail} free slots). "
                f"Pre-emptively guide vehicles to available zones. Monitor every 5 min."
            )
        else:
            return (
                f"✅ LOW STRESS – PSI={psi:.3f}. "
                f"{avail} of {self.total_slots} slots free ({(1-occ_rate)*100:.0f}%). "
                f"No immediate action required. Normal monitoring."
            )

    def _sim_recommendation(self, psi, level, demand_inc, new_slots, new_cap):
        emoji = self.EMOJI[level]
        if level == "high":
            return (
                f"{emoji} HIGH RISK: Even with {new_slots} extra slots, "
                f"a {demand_inc}% demand surge pushes PSI to {psi:.3f}. "
                f"Consider adding {new_slots + 20}+ slots or implementing "
                f"demand-pricing to redistribute vehicles."
            )
        elif level == "medium":
            return (
                f"{emoji} MANAGEABLE: Total capacity {new_cap} handles "
                f"projected demand at PSI={psi:.3f}. "
                f"Watch peak hours – pre-activate overflow lot during 5–7 PM."
            )
        else:
            return (
                f"{emoji} COMFORTABLE: With {new_cap} slots, PSI={psi:.3f}. "
                f"Expansion plan is sufficient for projected growth."
            )


# ---------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=== Module 6: PSI Test ===")
    psi_calc = ParkingStressIndex(total_slots=50)

    test_cases = [(8, 10), (20, 25), (35, 40), (45, 48), (49, 50)]
    for occ, pred in test_cases:
        result = psi_calc.compute(occ, pred)
        print(f"  Occ={occ:2d}  Pred={pred:2d}  PSI={result['psi']:.3f}  "
              f"[{result['level'].upper():6s}] {result['emoji']}")
        print(f"    {result['recommendation']}\n")

    # Simulation test
    sim = psi_calc.simulate(demand_increase_pct=25, new_slots=10,
                             base_occupancy=32, base_predicted=40)
    print(f"Simulation:")
    print(f"  Before: PSI={sim['before']['psi']} [{sim['before']['level']}]")
    print(f"  After : PSI={sim['after']['psi']}  [{sim['after']['level']}]")
    print(f"  Delta : {sim['delta_psi']:+.4f}")
    print(f"  → {sim['recommendation']}")
    print("Module 6 OK")
