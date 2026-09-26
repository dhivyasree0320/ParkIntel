"""
ROBOFLOW MODEL DOWNLOADER
ParkIntel – AI Smart Parking System

Downloads your YOLOv11 model trained on:
  Dataset : pfe Object Detection
  Author  : bouakkaz144.lotfi@gmail.com
  Platform: Roboflow

Usage:
    python download_model.py --key YOUR_API_KEY --workspace YOUR_WORKSPACE_NAME

Get your API key at:
    https://app.roboflow.com/settings/api

OR manually:
    1. Go to https://app.roboflow.com
    2. Open your "pfe" project → Versions
    3. Click "Download" → YOLOv11 → Download ZIP
    4. Unzip and copy best.pt → models/best.pt
"""

import os
import sys
import argparse
import shutil
import glob
from pathlib import Path


def download_roboflow_model(api_key, workspace, project="pfe", version=1):
    """
    Download YOLOv11 model from Roboflow and save to models/best.pt
    """
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[Error] roboflow package not installed.")
        print("  Run: pip install roboflow")
        return False

    try:
        print(f"Connecting to Roboflow...")
        rf = Roboflow(api_key=api_key)

        print(f"Loading workspace: {workspace}")
        ws = rf.workspace(workspace)

        print(f"Loading project: {project} v{version}")
        dataset = ws.project(project).version(version).download("yolov11")

        print(f"Downloaded to: {dataset.location}")

        # Find weights file
        weights = glob.glob(f"{dataset.location}/**/*.pt", recursive=True)
        if not weights:
            # Also check for YOLO folder structure
            weights = glob.glob(f"{dataset.location}/**/weights/*.pt", recursive=True)

        if weights:
            Path("models").mkdir(exist_ok=True)
            dest = Path("models/best.pt")
            shutil.copy(weights[0], dest)
            print(f"Model saved → {dest}  ({dest.stat().st_size/1024/1024:.1f} MB)")
            return True
        else:
            print("[Warning] No .pt file found in downloaded dataset.")
            print(f"  Check: {dataset.location}")
            return False

    except Exception as e:
        print(f"[Error] Download failed: {e}")
        print()
        print("Possible fixes:")
        print("  1. Verify your API key at https://app.roboflow.com/settings/api")
        print("  2. Check your workspace name (lowercase slug, e.g. 'my-workspace')")
        print("  3. Verify the project name ('pfe') and version number")
        return False


def manual_instructions():
    print()
    print("=" * 55)
    print("  MANUAL MODEL DOWNLOAD INSTRUCTIONS")
    print("=" * 55)
    print()
    print("1. Go to: https://app.roboflow.com")
    print("2. Log in with your account")
    print("3. Open project: 'pfe Object Detection'")
    print("4. Click 'Versions' in the left sidebar")
    print("5. Click 'Download' on your trained version")
    print("6. Choose format: YOLOv11")
    print("7. Download the ZIP file")
    print("8. Unzip the file")
    print("9. Copy 'best.pt' (or 'last.pt') to:")
    print("       ParkIntel/models/best.pt")
    print()
    print("The system will auto-detect the model on next run.")
    print("=" * 55)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Download YOLOv11 model from Roboflow for ParkIntel"
    )
    parser.add_argument("--key",       required=False, help="Roboflow API key")
    parser.add_argument("--workspace", required=False, help="Roboflow workspace name")
    parser.add_argument("--project",   default="pfe",  help="Project name (default: pfe)")
    parser.add_argument("--version",   type=int, default=1, help="Dataset version (default: 1)")
    parser.add_argument("--manual",    action="store_true", help="Show manual download instructions")
    args = parser.parse_args()

    if args.manual:
        manual_instructions()
        return

    if not args.key or not args.workspace:
        print("Usage:")
        print("  python download_model.py --key YOUR_KEY --workspace YOUR_WORKSPACE")
        print()
        print("For manual download:")
        print("  python download_model.py --manual")
        return

    success = download_roboflow_model(
        api_key=args.key,
        workspace=args.workspace,
        project=args.project,
        version=args.version
    )

    if success:
        print()
        print("Model download complete!")
        print("Now run: streamlit run dashboard.py")
    else:
        manual_instructions()


if __name__ == "__main__":
    main()
