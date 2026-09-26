"""
MODULE 8: SIMULATION ENGINE (What-If Analysis)
ParkIntel - AI Smart Parking System

Runs scenario planning simulations:
  - Demand increase impact
  - Capacity expansion effect
  - Combined strategies
  - Hour-by-hour forecast comparison
  - Multi-scenario comparison table
"""

import numpy as np
import pandas as pd
from datetime import datetime


class SimulationEngine:

    # 24-hour base occupancy pattern
    HOUR_PATTERN = [8,5,4,4,5,9,14,28,36,40,42,44,46,44,42,40,42,47,48,45,40,34,26,15]

    def __init__(self, total_slots=50):
        self.total_slots = total_slots
        self.scenario_history = []

    # ------------------------------------------------------------------ #
    #  RUN A SINGLE SCENARIO
    # ------------------------------------------------------------------ #
    def run(self, demand_change_pct=0, new_slots=0, hour=None,
            base_occupancy=None, base_predicted=None):
        """
        Run one what-if scenario.

        Args:
            demand_change_pct : % change in demand (+/-). E.g. 20 = +20%
            new_slots         : number of additional slots to add
            hour              : hour to simulate (0-23). Default = current hour
            base_occupancy    : current vehicles. Default = hourly pattern
            base_predicted    : predicted demand. Default = base + 5

        Returns:
            dict with before/after metrics, delta, recommendation
        """
        hour = hour if hour is not None else datetime.now().hour
        base_occ  = base_occupancy  if base_occupancy  is not None else self._hour_base(hour)
        base_pred = base_predicted  if base_predicted  is not None else min(50, base_occ + 5)

        new_cap   = self.total_slots + new_slots
        proj_occ  = float(np.clip(base_occ  * (1 + demand_change_pct / 100), 0, new_cap))
        proj_pred = float(np.clip(base_pred * (1 + demand_change_pct / 100), 0, new_cap))

        # PSI calculation
        orig_psi = round((base_occ / self.total_slots) * (base_pred / self.total_slots), 4)
        new_psi  = round((proj_occ / new_cap) * (proj_pred / new_cap), 4)

        orig_level = self._classify(orig_psi)
        new_level  = self._classify(new_psi)

        scenario = {
            "id":        len(self.scenario_history) + 1,
            "timestamp": datetime.now().isoformat(),
            "inputs": {
                "demand_change_pct": demand_change_pct,
                "new_slots":         new_slots,
                "hour":              hour,
                "base_occupancy":    int(base_occ)
            },
            "before": {
                "capacity":   self.total_slots,
                "occupancy":  int(base_occ),
                "available":  int(self.total_slots - base_occ),
                "psi":        orig_psi,
                "level":      orig_level
            },
            "after": {
                "capacity":  int(new_cap),
                "occupancy": int(proj_occ),
                "available": int(new_cap - proj_occ),
                "psi":       new_psi,
                "level":     new_level
            },
            "delta": {
                "psi":       round(new_psi - orig_psi, 4),
                "occupancy": int(proj_occ - base_occ),
                "available": int((new_cap - proj_occ) - (self.total_slots - base_occ)),
                "capacity":  new_slots
            },
            "recommendation": self._rec(new_psi, new_level, demand_change_pct, new_slots, int(new_cap)),
            "alert": new_level == "high" and orig_level != "high"
        }
        self.scenario_history.append(scenario)
        return scenario

    # ------------------------------------------------------------------ #
    #  HOURLY FORECAST (all 24 hours with scenario applied)
    # ------------------------------------------------------------------ #
    def hourly_forecast(self, demand_change_pct=0, new_slots=0):
        """
        Returns DataFrame with before/after projections for all 24 hours.
        Useful for plotting the daily impact of a scenario.
        """
        new_cap = self.total_slots + new_slots
        rows = []
        for h in range(24):
            base_occ  = self._hour_base(h)
            base_pred = min(self.total_slots, base_occ + 4)

            proj_occ  = float(np.clip(base_occ  * (1 + demand_change_pct / 100), 0, new_cap))
            proj_pred = float(np.clip(base_pred * (1 + demand_change_pct / 100), 0, new_cap))

            orig_psi = (base_occ / self.total_slots) * (base_pred / self.total_slots)
            new_psi  = (proj_occ / new_cap) * (proj_pred / new_cap)

            rows.append({
                "hour":           h,
                "label":          f"{h:02d}:00",
                "before_occ":     base_occ,
                "after_occ":      int(proj_occ),
                "before_avail":   self.total_slots - base_occ,
                "after_avail":    int(new_cap - proj_occ),
                "before_psi":     round(orig_psi, 4),
                "after_psi":      round(new_psi, 4),
                "before_level":   self._classify(orig_psi),
                "after_level":    self._classify(new_psi)
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  MULTI-SCENARIO COMPARISON
    # ------------------------------------------------------------------ #
    def compare(self, scenarios):
        """
        Run and compare multiple scenarios.

        Args:
            scenarios : list of dicts, each with keys:
                        demand_change_pct, new_slots, base_occupancy (optional)

        Returns:
            (comparison_df, list_of_results)
        """
        results = []
        for s in scenarios:
            r = self.run(**s)
            results.append(r)

        rows = []
        for r in results:
            rows.append({
                "Scenario":       f"#{r['id']}",
                "Demand Change":  f"{r['inputs']['demand_change_pct']:+d}%",
                "New Slots":      f"+{r['inputs']['new_slots']}",
                "New Capacity":   r["after"]["capacity"],
                "Before PSI":     r["before"]["psi"],
                "After PSI":      r["after"]["psi"],
                "Delta PSI":      f"{r['delta']['psi']:+.4f}",
                "Before Level":   r["before"]["level"].upper(),
                "After Level":    r["after"]["level"].upper(),
                "Available":      r["after"]["available"]
            })
        return pd.DataFrame(rows), results

    # ------------------------------------------------------------------ #
    #  EXPANSION RECOMMENDATION
    # ------------------------------------------------------------------ #
    def recommend_expansion(self, target_psi=0.50, base_occupancy=None,
                             base_predicted=None):
        """
        Find the minimum number of new slots needed to reach target PSI.
        """
        base_occ  = base_occupancy  if base_occupancy  is not None else self.total_slots * 0.80
        base_pred = base_predicted  if base_predicted  is not None else self.total_slots * 0.90

        for extra in range(0, 101, 5):
            new_cap = self.total_slots + extra
            occ_r   = base_occ  / new_cap
            pred_r  = base_pred / new_cap
            psi_val = occ_r * pred_r
            if psi_val <= target_psi:
                return {
                    "slots_needed":    extra,
                    "new_capacity":    int(new_cap),
                    "projected_psi":   round(psi_val, 4),
                    "target_psi":      target_psi,
                    "recommendation":  (
                        f"Add {extra} slots (total {int(new_cap)}) to achieve "
                        f"PSI={psi_val:.3f} ≤ target {target_psi}."
                    )
                }
        return {
            "slots_needed":   100,
            "new_capacity":   self.total_slots + 100,
            "projected_psi":  "still high",
            "recommendation": "Demand is too high — consider demand management strategies."
        }

    # ------------------------------------------------------------------ #
    #  HELPERS
    # ------------------------------------------------------------------ #
    def _hour_base(self, hour):
        return self.HOUR_PATTERN[int(hour) % 24]

    def _classify(self, psi):
        if psi < 0.30:  return "low"
        if psi < 0.70:  return "medium"
        return "high"

    def _rec(self, psi, level, demand_inc, new_slots, new_cap):
        emoji = {"low":"🟢","medium":"🟡","high":"🔴"}.get(level,"⚪")
        if level == "high":
            return (
                f"{emoji} HIGH RISK: {demand_inc}% demand surge with {new_slots} extra slots "
                f"still results in PSI={psi:.3f}. "
                f"Consider adding {new_slots+20}+ slots or implementing demand pricing."
            )
        elif level == "medium":
            return (
                f"{emoji} MANAGEABLE: {new_cap} total slots keeps PSI={psi:.3f}. "
                f"Monitor peak hours. Activate overflow lot preemptively during 5–7 PM."
            )
        else:
            return (
                f"{emoji} COMFORTABLE: {new_cap} slots → PSI={psi:.3f}. "
                f"Expansion plan handles projected demand. No urgent action."
            )


# ---------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=== Module 8: Simulation Engine Test ===")
    sim = SimulationEngine(total_slots=50)

    # Single scenario
    r = sim.run(demand_change_pct=25, new_slots=0, base_occupancy=35, base_predicted=42)
    print(f"Scenario #1: demand +25%, no new slots")
    print(f"  Before: occ={r['before']['occupancy']}  PSI={r['before']['psi']} [{r['before']['level']}]")
    print(f"  After : occ={r['after']['occupancy']}   PSI={r['after']['psi']}  [{r['after']['level']}]")
    print(f"  {r['recommendation']}\n")

    # Multi-scenario comparison
    scenarios = [
        {"demand_change_pct": 0,  "new_slots": 0,  "base_occupancy": 35},
        {"demand_change_pct": 20, "new_slots": 0,  "base_occupancy": 35},
        {"demand_change_pct": 20, "new_slots": 10, "base_occupancy": 35},
        {"demand_change_pct": 30, "new_slots": 20, "base_occupancy": 35},
    ]
    df, _ = sim.compare(scenarios)
    print("Multi-Scenario Comparison:")
    print(df.to_string(index=False))

    # Expansion recommendation
    exp = sim.recommend_expansion(target_psi=0.45, base_occupancy=38, base_predicted=44)
    print(f"\nExpansion Recommendation: {exp['recommendation']}")
    print("Module 8 OK")
