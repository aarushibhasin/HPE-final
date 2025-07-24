#!/usr/bin/env python3
"""
Quick Start Script for 3D Pose Estimation
Automates the entire training pipeline from data setup to evaluation
"""
import os
import sys
import argparse
import subprocess
import time
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Stdout:", e.stdout)
        if e.stderr:
            print("Stderr:", e.stderr)
        return False

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = ['torch', 'torchvision', 'numpy', 'opencv-python', 'pillow', 'tqdm']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All dependencies found!")
    return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        'data/surreal_3d',
        'data/lsp',
        'logs/prior_3d',
        'logs/3d_pose',
        'checkpoints/prior_3d',
        'checkpoints/3d_pose',
        'results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")

def download_minimal_surreal():
    """Download minimal SURREAL dataset"""
    print("📥 Downloading minimal SURREAL dataset...")
    
    # Use the automated download script
    cmd = f"""python scripts/download_surreal_3d.py \
        --root ./data/surreal_3d \
        --create-annotations"""
    
    if not run_command(cmd, "Downloading SURREAL dataset"):
        print("❌ SURREAL download failed!")
        return False
    
    # Check if data exists
    if not Path("data/surreal_3d/train/run0").exists():
        print("❌ SURREAL data not found after download.")
        return False
    
    print("✅ SURREAL data downloaded successfully!")
    return True

def download_lsp():
    """Download LSP dataset"""
    print("📥 Downloading LSP dataset...")
    
    lsp_url = "http://sam.johnson.io/research/lsp_dataset_original.zip"
    lsp_path = "data/lsp_dataset_original.zip"
    
    if not Path("data/lsp").exists():
        # Download LSP dataset
        cmd = f"wget {lsp_url} -O {lsp_path}"
        if not run_command(cmd, "Downloading LSP dataset"):
            return False
        
        # Extract
        cmd = f"unzip {lsp_path} -d data/"
        if not run_command(cmd, "Extracting LSP dataset"):
            return False
        
        # Clean up
        os.remove(lsp_path)
    
    print("✅ LSP data ready!")
    return True

def train_prior(args):
    """Train 3D prior network"""
    print("🧠 Training 3D prior network...")
    
    cmd = f"""python prior/train_prior_3d.py \
        --data-root ./data/surreal_3d \
        --subset-ratio {args.subset_ratio} \
        --epochs {args.prior_epochs} \
        --batch-size {args.prior_batch_size} \
        --lr {args.prior_lr} \
        --log logs/prior_3d \
        --save-freq 10"""
    
    return run_command(cmd, "Training 3D prior network")

def train_pose_model(args):
    """Train 3D pose estimation model"""
    print("🏋️ Training 3D pose estimation model...")
    
    prior_checkpoint = f"checkpoints/prior_3d/prior_3d_epoch_{args.prior_epochs}.pth.tar"
    
    if not Path(prior_checkpoint).exists():
        print(f"❌ Prior checkpoint not found: {prior_checkpoint}")
        return False
    
    cmd = f"""python train_human_prior_3d.py \
        --source-root ./data/surreal_3d \
        --target-root ./data/lsp \
        --source SURREAL3D \
        --target LSP \
        --target-train LSP_mt \
        --arch pose_resnet50_3d \
        --prior {prior_checkpoint} \
        --subset-ratio {args.subset_ratio} \
        --epochs {args.pose_epochs} \
        --pretrain-epoch {args.pretrain_epoch} \
        --batch-size {args.pose_batch_size} \
        --lr {args.pose_lr} \
        --log logs/3d_pose \
        --save-freq 10"""
    
    return run_command(cmd, "Training 3D pose estimation model")

def evaluate_model(args):
    """Evaluate the trained model"""
    print("📊 Evaluating model...")
    
    checkpoint = f"checkpoints/3d_pose/checkpoint_{args.pose_epochs-1:04d}.pth.tar"
    
    if not Path(checkpoint).exists():
        print(f"❌ Model checkpoint not found: {checkpoint}")
        return False
    
    cmd = f"""python train_human_prior_3d.py \
        --source-root ./data/surreal_3d \
        --target-root ./data/lsp \
        --source SURREAL3D \
        --target LSP \
        --arch pose_resnet50_3d \
        --resume {checkpoint} \
        --phase test"""
    
    return run_command(cmd, "Evaluating model")

def generate_report():
    """Generate training report"""
    print("📋 Generating training report...")
    
    report = f"""
# 3D Pose Estimation Training Report

## Training Summary
- **Start Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}
- **Duration**: [Calculate from logs]
- **Status**: [Success/Failed]

## Model Configuration
- **Architecture**: pose_resnet50_3d
- **Source Domain**: SURREAL3D
- **Target Domain**: LSP
- **Training Strategy**: Source-free domain adaptation

## Results
- **Source Domain Performance**: [From logs]
- **Target Domain Performance**: [From logs]
- **3D Prior Quality**: [From logs]

## Files Generated
- Prior Model: checkpoints/prior_3d/
- Pose Model: checkpoints/3d_pose/
- Logs: logs/
- Results: results/

## Next Steps
1. Analyze training curves
2. Fine-tune hyperparameters if needed
3. Test on new data
4. Deploy model
"""
    
    with open("results/training_report.md", "w") as f:
        f.write(report)
    
    print("✅ Report generated: results/training_report.md")

def main():
    parser = argparse.ArgumentParser(description="Quick Start for 3D Pose Estimation")
    
    # Data arguments
    parser.add_argument('--subset-ratio', type=float, default=0.1, 
                       help='SURREAL dataset subset ratio')
    parser.add_argument('--skip-data-download', action='store_true',
                       help='Skip data download (use existing data)')
    
    # Prior training arguments
    parser.add_argument('--prior-epochs', type=int, default=50,
                       help='Number of prior training epochs')
    parser.add_argument('--prior-batch-size', type=int, default=64,
                       help='Prior training batch size')
    parser.add_argument('--prior-lr', type=float, default=0.001,
                       help='Prior training learning rate')
    
    # Pose training arguments
    parser.add_argument('--pose-epochs', type=int, default=70,
                       help='Number of pose training epochs')
    parser.add_argument('--pretrain-epoch', type=int, default=40,
                       help='Pretrain epochs')
    parser.add_argument('--pose-batch-size', type=int, default=32,
                       help='Pose training batch size')
    parser.add_argument('--pose-lr', type=float, default=0.001,
                       help='Pose training learning rate')
    
    # Pipeline control
    parser.add_argument('--skip-prior', action='store_true',
                       help='Skip prior training')
    parser.add_argument('--skip-pose', action='store_true',
                       help='Skip pose training')
    parser.add_argument('--skip-eval', action='store_true',
                       help='Skip evaluation')
    
    args = parser.parse_args()
    
    print("🚀 Starting 3D Pose Estimation Pipeline")
    print("=" * 60)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return
    
    # Step 2: Setup directories
    setup_directories()
    
    # Step 3: Download data
    if not args.skip_data_download:
        if not download_minimal_surreal():
            return
        if not download_lsp():
            return
    
    # Step 4: Train prior (if not skipped)
    if not args.skip_prior:
        if not train_prior(args):
            print("❌ Prior training failed!")
            return
    
    # Step 5: Train pose model (if not skipped)
    if not args.skip_pose:
        if not train_pose_model(args):
            print("❌ Pose training failed!")
            return
    
    # Step 6: Evaluate (if not skipped)
    if not args.skip_eval:
        if not evaluate_model(args):
            print("❌ Evaluation failed!")
            return
    
    # Step 7: Generate report
    generate_report()
    
    print("\n" + "=" * 60)
    print("🎉 Pipeline completed successfully!")
    print("=" * 60)
    print("\n📁 Check the following directories:")
    print("   - Logs: logs/")
    print("   - Checkpoints: checkpoints/")
    print("   - Results: results/")
    print("\n📋 Next steps:")
    print("   1. Review training_report.md")
    print("   2. Analyze training curves")
    print("   3. Test on your own data")
    print("   4. Fine-tune if needed")

if __name__ == "__main__":
    main() 