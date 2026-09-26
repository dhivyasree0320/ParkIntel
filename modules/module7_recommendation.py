"""
MODULE 7: EXPLAINABLE RECOMMENDATION ENGINE
ParkIntel - AI Smart Parking System

Generates intelligent, explainable recommendations by combining:
  - Feature importance from the Random Forest model
  - Current PSI level
  - Peak hour status
  - Historical patterns

This makes the system "explainable AI" (XAI).
"""

from datetime import datetime


class RecommendationEngine:

    # Human-readable feature names for explainability
    FEATURE_LABELS = {
        "hour":         "Hour of Day",
        "day_of_week":  "Day of Week",
        "is_weekend":   "Weekend Flag",
        "hour_sin":     "Hour (cyclic)",
        "hour_cos":     "Hour (cyclic)",
        "day_sin":      "Day (cyclic)",
        "day_cos":      "Day (cyclic)",
        "occ_lag1":     "Last-Hour Occupancy",
        "occ_lag2":     "2-Hour Lag",
        "occ_rolling3": "3-Hour Rolling Avg"
    }

    # Consolidated display groups (merge cyclic pairs)
    DISPLAY_GROUPS = {
        "Hour of Day":          ["hour", "hour_sin", "hour_cos"],
        "Day of Week":          ["day_of_week", "day_sin", "day_cos"],
        "Weekend Pattern":      ["is_weekend"],
        "Recent Occupancy":     ["occ_lag1", "occ_lag2"],
        "Rolling Avg (3h)":     ["occ_rolling3"]
    }

    def __init__(self, total_slots=50):
        self.total_slots = total_slots

    # ------------------------------------------------------------------ #
    #  GENERATE FULL RECOMMENDATION
    # ------------------------------------------------------------------ #
    def generate(self, psi_result, peak_analysis, feature_importances,
                 predictor_metrics=None):
        """
        Build complete explainable recommendation.

        Args:
            psi_result         : dict from ParkingStressIndex.compute()
            peak_analysis      : dict from PeakHourAnalyzer.analyze()
            feature_importances: dict {feature_name: importance_score}
            predictor_metrics  : dict from ParkingPredictor.metrics_

        Returns:
            dict with summary, explanation, drivers, actions, confidence
        """
        psi    = psi_result.get("psi", 0.0)
        level  = psi_result.get("level", "low")
        occ    = psi_result.get("current_occupancy", 0)
        avail  = psi_result.get("available", self.total_slots)
        occ_r  = psi_result.get("occ_rate", 0.0)
        pred   = psi_result.get("predicted_demand", 0)
        emoji  = psi_result.get("emoji", "🟢")

        ts_str = psi_result.get("timestamp", datetime.now().isoformat())
        try:
            ts = datetime.fromisoformat(ts_str)
        except Exception:
            ts = datetime.now()
        hour = ts.hour

        # Is it currently a peak hour?
        is_peak = (8 <= hour < 10) or (12 <= hour < 14) or (17 <= hour < 20)
        peak_label = peak_analysis.get("peak_window_label", "5 PM – 7 PM")
        busiest    = peak_analysis.get("busiest_day", "Friday")

        # Top driving factors
        grouped = self._group_importances(feature_importances)
        top_drivers = sorted(grouped.items(), key=lambda x: x[1], reverse=True)[:4]
        top_driver_names = [k for k, _ in top_drivers]

        # Build narrative explanation
        explanation = self._build_explanation(
            psi, level, occ, avail, occ_r, pred,
            is_peak, peak_label, busiest, top_driver_names, ts
        )

        # Action items based on severity
        actions = self._get_actions(level, occ_r, hour, avail, is_peak)

        # Model confidence note
        conf_note = ""
        if predictor_metrics:
            acc = predictor_metrics.get("accuracy_pct", "?")
            r2  = predictor_metrics.get("r2", "?")
            conf_note = f"Model accuracy: {acc}%  |  R²: {r2}"

        return {
            "summary":       f"{emoji} PSI={psi:.3f} [{level.upper()}] — {occ}/{self.total_slots} slots occupied",
            "explanation":   explanation,
            "top_drivers":   top_driver_names,
            "driver_scores": top_drivers,
            "actions":       actions,
            "confidence":    conf_note,
            "level":         level,
            "psi":           psi,
            "is_peak":       is_peak,
            "timestamp":     ts.isoformat()
        }

    # ------------------------------------------------------------------ #
    #  QUICK RECOMMENDATION (no full data required)
    # ------------------------------------------------------------------ #
    def quick_rec(self, psi, level, occupancy, available, hour):
        """Simple one-liner recommendation from basic inputs."""
        is_peak = (8 <= hour < 10) or (12 <= hour < 14) or (17 <= hour < 20)
        if level == "high":
            return (f"⚠ HIGH: PSI={psi:.3f}. Only {available} slots free. "
                    f"{'PEAK HOUR. ' if is_peak else ''}"
                    f"Activate overflow lot & dynamic pricing NOW.")
        elif level == "medium":
            return (f"⚡ MEDIUM: PSI={psi:.3f}. {available} slots available. "
                    f"{'Peak hour approaching. ' if is_peak else ''}"
                    f"Pre-emptive guidance recommended.")
        else:
            return (f"✅ LOW: PSI={psi:.3f}. {available} of {self.total_slots} free. "
                    f"Normal operations.")

    # ------------------------------------------------------------------ #
    #  GROUP FEATURE IMPORTANCES (merge cyclic pairs)
    # ------------------------------------------------------------------ #
    def _group_importances(self, importances):
        if not importances:
            # Default demo importances
            return {
                "Hour of Day":      0.34,
                "Day of Week":      0.28,
                "Recent Occupancy": 0.22,
                "Weekend Pattern":  0.10,
                "Rolling Avg (3h)": 0.06
            }
        grouped = {}
        for label, features in self.DISPLAY_GROUPS.items():
            total = sum(importances.get(f, 0.0) for f in features)
            if total > 0:
                grouped[label] = round(total, 4)
        return grouped

    def _build_explanation(self, psi, level, occ, avail, occ_r, pred,
                            is_peak, peak_label, busiest, top_drivers, ts):
        drivers_str = ", ".join(top_drivers[:3]) if top_drivers else "time of day"
        hour_label  = ts.strftime("%I:%M %p")

        lines = []
        lines.append(
            f"At {hour_label}, the parking lot has {occ}/{self.total_slots} slots occupied "
            f"({occ_r*100:.0f}%) with {avail} spaces remaining."
        )
        lines.append(
            f"The Parking Stress Index (PSI) is {psi:.3f} — a {level.upper()} stress level — "
            f"calculated as occupancy rate × predicted demand rate."
        )
        if level == "high":
            lines.append(
                f"Congestion is CRITICAL because of: {drivers_str}. "
                f"Predicted demand for the next hour is {pred} vehicles."
                + (f" This falls within the peak window ({peak_label})." if is_peak else "")
            )
        elif level == "medium":
            lines.append(
                f"Moderate congestion is driven by: {drivers_str}. "
                + (f"Peak hour ({peak_label}) is a contributing factor. " if is_peak else "")
                + f"Demand expected at {pred} vehicles next hour."
            )
        else:
            lines.append(
                f"Low stress conditions. Primary factors: {drivers_str}. "
                f"{avail} slots available. No congestion risk at this time."
            )
        lines.append(
            f"Historically, {busiest} is the busiest day of the week. "
            f"The AI model uses {len(self.DISPLAY_GROUPS)} feature groups to generate this prediction."
        )
        return "  ".join(lines)

    def _get_actions(self, level, occ_rate, hour, avail, is_peak):
        if level == "high":
            return [
                "🔴 Activate overflow parking lot immediately",
                "💰 Enable dynamic surge pricing to redistribute demand",
                "📍 Deploy digital wayfinding signs to available zones",
                "📢 Push notification to parking app users",
                "👷 Alert parking staff for manual traffic guidance",
                "📊 Escalate alert to city traffic management center"
            ]
        elif level == "medium":
            return [
                "🟡 Pre-emptively guide vehicles to under-utilized zones",
                "📱 Update digital signage with real-time availability",
                "👁 Monitor occupancy every 5 minutes",
                "🅿 Prepare overflow lot for potential activation"
            ]
        else:
            return [
                "🟢 Normal monitoring (check every 15 minutes)",
                "📋 Log data for pattern analysis",
                "🔄 Maintain standard parking guidance"
            ]


# ---------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=== Module 7: Recommendation Engine Test ===")
    engine = RecommendationEngine(total_slots=50)

    psi_res = {
        "psi": 0.82, "level": "high", "emoji": "🔴",
        "current_occupancy": 47, "available": 3,
        "occ_rate": 0.94, "predicted_demand": 48,
        "timestamp": datetime.now().isoformat()
    }
    peak_a = {
        "peak_window_label": "5 PM – 7 PM",
        "busiest_day": "Friday"
    }
    feat_imp = {
        "hour": 0.20, "hour_sin": 0.08, "hour_cos": 0.06,
        "day_of_week": 0.15, "day_sin": 0.07, "day_cos": 0.06,
        "is_weekend": 0.10, "occ_lag1": 0.15, "occ_lag2": 0.07, "occ_rolling3": 0.06
    }
    metrics = {"accuracy_pct": 94.2, "r2": 0.947}

    rec = engine.generate(psi_res, peak_a, feat_imp, metrics)
    print(f"Summary    : {rec['summary']}")
    print(f"Explanation: {rec['explanation']}")
    print(f"Top drivers: {rec['top_drivers']}")
    print(f"Actions:")
    for a in rec["actions"]:
        print(f"   {a}")
    print(f"Confidence : {rec['confidence']}")
    print("Module 7 OK")
