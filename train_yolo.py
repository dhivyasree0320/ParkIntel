"""
ParkIntel YOLO Training Script
Trains YOLOv8 detection model on parking lot occupancy dataset

Usage:
    python train_yolo.py           # Default training (CPU)
    python train_yolo.py --gpu     # GPU training
    python train_yolo.py --epochs 100 --batch 16  # Custom parameters
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLOv8 detection model for ParkIntel')
    parser.add_argument('--data', type=str, default='pfe.v2i.yolov8/data.yaml',
                        help='Path to data.yaml file')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                        help='YOLO model to use (yolo11n.pt, yolov8n.pt)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size (reduce if GPU memory is limited)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use: cpu, 0, 1, etc.')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Project directory for saving results')
    parser.add_argument('--name', type=str, default='parking_detector',
                        help='Experiment name')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    parser.add_argument('--optimizer', type=str, default='SGD',
                        help='Optimizer: SGD, Adam, AdamW')
    parser.add_argument('--lr0', type=float, default=0.01,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.937,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='Weight decay')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    return parser.parse_args()


def check_dataset(data_yaml):
    """Verify dataset structure exists"""
    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"[ERROR] Data config not found: {data_yaml}")
        return False
    
    # Load and check paths
    import yaml
    with open(data_yaml, 'r') as f:
        data = yaml.safe_load(f)
    
    base_path = data.get('path', str(data_path.parent))
    train_path = Path(base_path) / data['train']
    val_path = Path(base_path) / data['val']
    
    print(f"[INFO] Dataset base path: {base_path}")
    print(f"[INFO] Train path: {train_path}")
    print(f"[INFO] Val path: {val_path}")
    
    if not train_path.exists():
        print(f"[ERROR] Training images not found: {train_path}")
        return False
    
    if not val_path.exists():
        print(f"[ERROR] Validation images not found: {val_path}")
        return False
    
    # Count images
    train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
    val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))
    
    print(f"[INFO] Training images: {len(train_images)}")
    print(f"[INFO] Validation images: {len(val_images)}")
    
    return True


def train(args):
    """Main training function"""
    from ultralytics import YOLO
    
    print("=" * 60)
    print("ParkIntel YOLO Detection Training")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.imgsz}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Check dataset
    if not check_dataset(args.data):
        print("[ERROR] Dataset validation failed!")
        return
    
    # Load model
    print(f"\n[INFO] Loading model: {args.model}")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("[INFO] Trying yolov8n.pt...")
        try:
            model = YOLO('yolov8n.pt')
        except:
            print("[INFO] Downloading yolov8n.pt...")
            model = YOLO('yolov8n.pt')
    
    # Training parameters
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        pretrained=args.pretrained,
        optimizer=args.optimizer,
        lr0=args.lr0,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        workers=args.workers,
        verbose=args.verbose,
        # Detection task
        task='detect',
        # Save best model
        save=True,
        save_period=10,
        # Early stopping
        patience=10,
        # Data augmentation
        augment=True,
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.1,
    )
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    # Copy best model to models directory
    best_model_path = Path(args.project) / args.name / 'weights' / 'best.pt'
    if best_model_path.exists():
        dest_path = project_root / 'models' / 'best.pt'
        import shutil
        shutil.copy(best_model_path, dest_path)
        print(f"[INFO] Model saved to: {dest_path}")
    else:
        # Try last.pt
        last_model_path = Path(args.project) / args.name / 'weights' / 'last.pt'
        if last_model_path.exists():
            dest_path = project_root / 'models' / 'best.pt'
            import shutil
            shutil.copy(last_model_path, dest_path)
            print(f"[INFO] Model saved to: {dest_path}")
        else:
            print("[WARNING] Could not find trained weights!")
    
    return results


if __name__ == "__main__":
    args = parse_args()
    
    # Auto-detect GPU if --gpu flag is passed
    if len(sys.argv) > 1 and '--gpu' in sys.argv:
        args.device = '0'  # Use first GPU
        args.batch = 16   # Can use larger batch with GPU
    
    train(args)
