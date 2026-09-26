"""
MAIN PIPELINE - PARKINTEL
ParkIntel - AI Smart Parking System

Connects all 8 modules into a full working pipeline.

Usage:
    python main_pipeline.py --mode setup          # generate data + train model
    python main_pipeline.py --mode demo           # run demo detection loop
    python main_pipeline.py --mode live           # live camera / video file
    python main_pipeline.py --mode report         # print full analysis report
    python main_pipeline.py --mode live --source parking.mp4
    python main_pipeline.py --mode live --source 0   (webcam)
"""

import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime

# Make modules importable from project root
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from modules.module1_data_acquisition  import DataAcquisition
from modules.module2_vehicle_detection import VehicleDetector
from modules.module3_occupancy         import OccupancyEstimator
from modules.module4_prediction        import ParkingPredictor
from modules.module5_peak_hours        import PeakHourAnalyzer
from modules.module6_psi               import ParkingStressIndex
from modules.module7_recommendation    import RecommendationEngine
from modules.module8_simulation        import SimulationEngine


class ParkIntelPipeline:
    """
    Orchestrates the full ParkIntel pipeline.

    Modes:
        setup  → generate 30-day historical data + train Random Forest model
        demo   → simulated live detection loop (no camera required)
        live   → real camera / video file detection loop
        report → full printed analysis report
    """

    def __init__(self, config=None):
        cfg = config or {}
        self.total_slots    = cfg.get("total_slots",    50)
        self.video_source   = cfg.get("video_source",   None)
        self.model_path     = cfg.get("model_path",     "models/best.pt")
        self.rf_path        = cfg.get("rf_path",        "models/rf_model.pkl")
        self.csv_path       = cfg.get("csv_path",       "data/occupancy_log.csv")
        self.confidence     = cfg.get("confidence",     0.45)
        self.device         = cfg.get("device",         "cpu")
        self.show_video     = cfg.get("show_video",     True)

        print()
        print("=" * 50)
        print("   ParkIntel – AI Smart Parking System")
        print("=" * 50)
        print(f"  Total slots : {self.total_slots}")
        print(f"  Video source: {self.video_source or 'demo (synthetic)'}")
        print(f"  YOLO model  : {self.model_path}")
        print(f"  RF model    : {self.rf_path}")
        print("=" * 50)
        print()

        # Initialise all 8 modules
        self.acq        = DataAcquisition(source=self.video_source,
                                          output_dir="data/frames",
                                          total_slots=self.total_slots)
        self.detector   = VehicleDetector(model_path=self.model_path,
                                          total_slots=self.total_slots,
                                          confidence=self.confidence,
                                          device=self.device)
        self.occupancy  = OccupancyEstimator(total_slots=self.total_slots,
                                             csv_path=self.csv_path)
        self.predictor  = ParkingPredictor(model_path=self.rf_path,
                                           total_slots=self.total_slots)
        self.peak       = PeakHourAnalyzer(total_slots=self.total_slots)
        self.psi        = ParkingStressIndex(total_slots=self.total_slots)
        self.rec_engine = RecommendationEngine(total_slots=self.total_slots)
        self.simulator  = SimulationEngine(total_slots=self.total_slots)

        print("All 8 modules initialised.\n")

    # ================================================================== #
    #  SETUP: generate data + train model
    # ================================================================== #
    def setup(self, historical_days=30):
        print("── STEP 1: Generating Historical Occupancy Data ──")
        df = self.occupancy.generate_historical_data(days=historical_days, save=True)
        print(f"  Dataset: {df.shape[0]} rows × {df.shape[1]} cols")
        print()

        print("── STEP 2: Training Random Forest Model ──")
        df_feat = self.occupancy.get_dataframe()
        metrics = self.predictor.train(df_feat)

        if metrics:
            print()
            print("  Training Results:")
            print(f"    Accuracy : {metrics.get('accuracy_pct')}%")
            print(f"    MAE      : {metrics.get('mae')} slots")
            print(f"    RMSE     : {metrics.get('rmse')}")
            print(f"    R²       : {metrics.get('r2')}")
            print(f"    Train    : {metrics.get('train_size')} samples")
            print(f"    Test     : {metrics.get('test_size')} samples")
            print()

            print("  Feature Importances (top 5):")
            fi = self.predictor.feature_importances_
            if fi:
                top5 = sorted(fi.items(), key=lambda x: -x[1])[:5]
                for name, score in top5:
                    bar = "█" * int(score * 60)
                    print(f"    {name:20s}  {score:.4f}  {bar}")
        print()
        return df_feat, metrics

    # ================================================================== #
    #  PROCESS SINGLE FRAME
    # ================================================================== #
    def process_frame(self, frame, timestamp=None):
        """
        Full pipeline for one frame:
          detect → occupancy → predict → PSI → recommendation

        Returns dict with all results.
        """
        ts = timestamp or datetime.now()

        # Module 2: detect
        detection = self.detector.detect(frame)

        # Module 3: occupancy
        occ_record = self.occupancy.update(detection, timestamp=ts)
        current_occ = occ_record["occupancy"]

        # Module 4: predict next 6 hours
        forecasts = self.predictor.predict_next_hours(current_occ, hours=6, start_time=ts)
        pred_next = forecasts[0]["predicted_occupancy"] if forecasts else current_occ

        # Module 5: peak hour stress
        peak_stress = self.peak.current_stress(current_occ)

        # Module 6: PSI
        psi_result = self.psi.compute(current_occ, pred_next, current_time=ts)

        # Module 7: recommendation
        fi = self.predictor.feature_importances_ or {}
        peak_analysis = self.peak._demo_analysis()   # use full analysis if df loaded
        rec = self.rec_engine.generate(psi_result, peak_analysis, fi,
                                        self.predictor.metrics_)

        return {
            "timestamp":      ts.isoformat(),
            "detection":      detection,
            "occupancy":      occ_record,
            "forecasts":      forecasts,
            "peak_stress":    peak_stress,
            "psi":            psi_result,
            "recommendation": rec,
            "frame":          detection.get("annotated_frame")
        }

    # ================================================================== #
    #  DEMO LOOP (no camera)
    # ================================================================== #
    def run_demo(self, iterations=10, delay=1.0, verbose=True):
        print(f"── DEMO LOOP ({iterations} iterations, delay={delay}s) ──")
        print()

        for i in range(1, iterations + 1):
            frame = self.acq._make_demo_frame()
            result = self.process_frame(frame)

            occ   = result["occupancy"]["occupancy"]
            avail = result["occupancy"]["available"]
            psi   = result["psi"]["psi"]
            level = result["psi"]["level"].upper()
            emoji = result["psi"]["emoji"]
            pred  = result["forecasts"][0]["predicted_occupancy"] if result["forecasts"] else "?"

            if verbose:
                print(
                    f"  [{i:02d}/{iterations}] {datetime.now().strftime('%H:%M:%S')}  "
                    f"Detected={occ:2d}  Avail={avail:2d}  "
                    f"Pred+1h={pred:2d}  PSI={psi:.3f}  {emoji} {level}"
                )
                if level == "HIGH":
                    print(f"         ⚠ {result['recommendation']['actions'][0]}")

            time.sleep(delay)

        print()
        print("Demo loop complete.")
        print()

    # ================================================================== #
    #  LIVE DETECTION (camera / video file)
    # ================================================================== #
    def run_live(self, max_frames=None, verbose=True):
        import cv2
        print(f"── LIVE DETECTION ──  source={self.video_source}")
        print("Press 'q' to quit.\n")

        self.acq.open()
        frame_count = 0

        try:
            for frame, ts in self.acq.live_frames():
                frame_count += 1
                result = self.process_frame(frame, timestamp=ts)

                if self.show_video and result["frame"] is not None:
                    cv2.imshow("ParkIntel – Live Detection (q=quit)", result["frame"])
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("Quit by user.")
                        break

                if verbose and frame_count % 10 == 0:
                    occ = result["occupancy"]["occupancy"]
                    psi = result["psi"]["psi"]
                    lvl = result["psi"]["level"].upper()
                    print(f"  Frame {frame_count:4d}  Occ={occ:2d}  PSI={psi:.3f}  [{lvl}]")

                if max_frames and frame_count >= max_frames:
                    break

        except KeyboardInterrupt:
            print("\nStopped by Ctrl+C.")
        finally:
            self.acq.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        print(f"Live detection finished. Processed {frame_count} frames.\n")

    # ================================================================== #
    #  FULL ANALYSIS REPORT
    # ================================================================== #
    def full_report(self):
        print("=" * 55)
        print("         PARKINTEL FULL ANALYSIS REPORT")
        print("=" * 55)

        df = self.occupancy.get_dataframe()
        if df.empty:
            print("No occupancy data found. Run: python main_pipeline.py --mode setup")
            return {}

        # Full peak analysis
        peak_analysis = self.peak.analyze(df)

        # Current state
        current_occ = self.occupancy.current_metrics()["occupancy"]
        forecasts   = self.predictor.predict_next_hours(current_occ, hours=6)
        pred_next   = forecasts[0]["predicted_occupancy"] if forecasts else current_occ
        psi_result  = self.psi.compute(current_occ, pred_next)
        rec         = self.rec_engine.generate(
                          psi_result, peak_analysis,
                          self.predictor.feature_importances_,
                          self.predictor.metrics_
                      )

        # Simulations
        df_scenarios, sim_results = self.simulator.compare([
            {"demand_change_pct": 0,  "new_slots": 0,  "base_occupancy": current_occ},
            {"demand_change_pct": 20, "new_slots": 0,  "base_occupancy": current_occ},
            {"demand_change_pct": 20, "new_slots": 10, "base_occupancy": current_occ},
            {"demand_change_pct": 30, "new_slots": 20, "base_occupancy": current_occ},
        ])

        print()
        print("── CURRENT STATUS ──")
        print(f"  Occupancy  : {current_occ}/{self.total_slots} "
              f"({psi_result['occ_rate']*100:.0f}% full)")
        print(f"  Available  : {psi_result['available']} slots")
        print(f"  PSI        : {psi_result['psi']} [{psi_result['level'].upper()}] "
              f"{psi_result['emoji']}")
        print(f"  Peak window: {peak_analysis.get('peak_window_label','N/A')}")
        print(f"  Busiest day: {peak_analysis.get('busiest_day','N/A')}")
        print()

        print("── 6-HOUR FORECAST ──")
        for f in forecasts:
            pct  = f["occupancy_rate"]
            bar  = "█" * int(pct * 25)
            spc  = "░" * (25 - int(pct * 25))
            lvl  = "HIGH" if pct >= 0.7 else "MED" if pct >= 0.3 else "LOW"
            print(f"  +{f['hour_offset']}h  {f['timestamp']}  "
                  f"occ={f['predicted_occupancy']:2d}  avail={f['predicted_available']:2d}  "
                  f"[{bar}{spc}] {pct*100:5.1f}%  {lvl}")
        print()

        print("── RECOMMENDATION ──")
        print(f"  {rec['summary']}")
        print()
        print(f"  {rec['explanation']}")
        print()
        print("  Actions:")
        for a in rec["actions"]:
            print(f"    {a}")
        print()

        if self.predictor.metrics_:
            m = self.predictor.metrics_
            print("── MODEL PERFORMANCE ──")
            print(f"  Accuracy : {m.get('accuracy_pct')}%")
            print(f"  MAE      : {m.get('mae')} slots")
            print(f"  RMSE     : {m.get('rmse')}")
            print(f"  R²       : {m.get('r2')}")
            print()

        print("── CONGESTION DISTRIBUTION ──")
        print(f"  Low    : {peak_analysis.get('low_pct',0)}%")
        print(f"  Medium : {peak_analysis.get('medium_pct',0)}%")
        print(f"  High   : {peak_analysis.get('high_pct',0)}%")
        print()

        print("── SIMULATION COMPARISON ──")
        print(df_scenarios.to_string(index=False))
        print()
        print("=" * 55)

        return {
            "peak_analysis": peak_analysis,
            "psi":           psi_result,
            "recommendation": rec,
            "forecasts":     forecasts,
            "simulations":   sim_results
        }


# ================================================================== #
#  CLI ENTRY POINT
# ================================================================== #
def main():
    parser = argparse.ArgumentParser(
        description="ParkIntel – AI Smart Parking System",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--mode",
        choices=["setup", "demo", "live", "report", "all"],
        default="all",
        help=(
            "setup  : generate training data + train model\n"
            "demo   : run simulated detection loop\n"
            "live   : real-time camera/video detection\n"
            "report : print full analysis\n"
            "all    : setup + demo + report"
        )
    )
    parser.add_argument("--source", default=None,
                        help="Video source: None=demo, 0=webcam, path.mp4, rtsp://...")
    parser.add_argument("--slots",  type=int, default=50,  help="Total parking slots")
    parser.add_argument("--model",  default="models/best.pt", help="YOLOv11 weights path")
    parser.add_argument("--days",   type=int, default=30,  help="Historical data days")
    parser.add_argument("--iter",   type=int, default=8,   help="Demo loop iterations")
    parser.add_argument("--delay",  type=float, default=0.8, help="Demo loop delay (s)")
    parser.add_argument("--no-video", action="store_true",  help="Disable video window")
    args = parser.parse_args()

    config = {
        "total_slots":  args.slots,
        "video_source": args.source,
        "model_path":   args.model,
        "rf_path":      "models/rf_model.pkl",
        "csv_path":     "data/occupancy_log.csv",
        "show_video":   not args.no_video
    }

    pipeline = ParkIntelPipeline(config)

    if args.mode in ("setup", "all"):
        pipeline.setup(historical_days=args.days)

    if args.mode in ("demo", "all"):
        pipeline.run_demo(iterations=args.iter, delay=args.delay)

    if args.mode in ("report", "all"):
        pipeline.full_report()

    if args.mode == "live":
        pipeline.run_live()


if __name__ == "__main__":
    main()
