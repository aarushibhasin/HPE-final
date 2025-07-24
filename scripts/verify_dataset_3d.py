#!/usr/bin/env python3
"""
Dataset Verification Script for 3D Pose Estimation
Verifies the integrity of downloaded datasets
"""
import os
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def verify_surreal_dataset(root_dir):
    """Verify SURREAL dataset integrity"""
    print("🔍 Verifying SURREAL dataset...")
    
    issues = []
    stats = {
        'total_samples': 0,
        'valid_samples': 0,
        'missing_images': 0,
        'missing_annotations': 0,
        'invalid_annotations': 0
    }
    
    for split in ['train', 'val', 'test']:
        if split == 'train':
            runs = ['run0', 'run1', 'run2']
        else:
            runs = ['run0']
        
        for run in runs:
            annotation_file = Path(root_dir) / split / f'{run}.json'
            
            if not annotation_file.exists():
                print(f"⚠️  Missing annotation file: {annotation_file}")
                continue
            
            try:
                with open(annotation_file, 'r') as f:
                    samples = json.load(f)
                
                print(f"📊 {split}/{run}: {len(samples)} samples")
                stats['total_samples'] += len(samples)
                
                for i, sample in enumerate(samples):
                    # Check image exists
                    image_path = Path(sample['image_path'])
                    if not image_path.exists():
                        issues.append(f"Missing image: {image_path}")
                        stats['missing_images'] += 1
                        continue
                    
                    # Check image is valid
                    try:
                        img = Image.open(image_path)
                        img.verify()
                    except Exception as e:
                        issues.append(f"Invalid image {image_path}: {e}")
                        continue
                    
                    # Check annotations
                    if 'keypoint2d' not in sample or 'keypoint3d' not in sample:
                        issues.append(f"Missing annotations in sample {i}")
                        stats['missing_annotations'] += 1
                        continue
                    
                    # Check annotation format
                    try:
                        joints_2d = np.array(sample['keypoint2d'])
                        joints_3d = np.array(sample['keypoint3d'])
                        
                        if joints_2d.shape != (16, 2) or joints_3d.shape != (16, 3):
                            issues.append(f"Invalid annotation shape in sample {i}")
                            stats['invalid_annotations'] += 1
                            continue
                        
                        stats['valid_samples'] += 1
                        
                    except Exception as e:
                        issues.append(f"Invalid annotation format in sample {i}: {e}")
                        stats['invalid_annotations'] += 1
                        continue
            
            except Exception as e:
                print(f"❌ Error reading {annotation_file}: {e}")
    
    # Print summary
    print(f"\n📊 SURREAL Dataset Summary:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Valid samples: {stats['valid_samples']}")
    print(f"   Missing images: {stats['missing_images']}")
    print(f"   Missing annotations: {stats['missing_annotations']}")
    print(f"   Invalid annotations: {stats['invalid_annotations']}")
    print(f"   Success rate: {stats['valid_samples']/max(stats['total_samples'], 1)*100:.1f}%")
    
    if issues:
        print(f"\n⚠️  Issues found ({len(issues)}):")
        for issue in issues[:10]:  # Show first 10 issues
            print(f"   - {issue}")
        if len(issues) > 10:
            print(f"   ... and {len(issues)-10} more issues")
    
    return len(issues) == 0

def verify_lsp_dataset(root_dir):
    """Verify LSP dataset integrity"""
    print("🔍 Verifying LSP dataset...")
    
    # Check if LSP dataset exists
    if not Path(root_dir).exists():
        print(f"❌ LSP dataset not found at {root_dir}")
        return False
    
    # Check for images and annotations
    images_dir = Path(root_dir) / 'images'
    annotations_file = Path(root_dir) / 'joints.mat'
    
    if not images_dir.exists():
        print(f"❌ LSP images directory not found: {images_dir}")
        return False
    
    if not annotations_file.exists():
        print(f"❌ LSP annotations file not found: {annotations_file}")
        return False
    
    # Count images
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"📊 LSP dataset: {len(image_files)} images found")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Verify 3D pose estimation datasets")
    parser.add_argument('--surreal-root', default='./data/surreal_3d',
                       help='SURREAL dataset root directory')
    parser.add_argument('--lsp-root', default='./data/lsp',
                       help='LSP dataset root directory')
    parser.add_argument('--skip-surreal', action='store_true',
                       help='Skip SURREAL verification')
    parser.add_argument('--skip-lsp', action='store_true',
                       help='Skip LSP verification')
    
    args = parser.parse_args()
    
    print("🔍 Dataset Verification")
    print("=" * 40)
    
    surreal_ok = True
    lsp_ok = True
    
    if not args.skip_surreal:
        surreal_ok = verify_surreal_dataset(args.surreal_root)
    
    if not args.skip_lsp:
        lsp_ok = verify_lsp_dataset(args.lsp_root)
    
    print("\n" + "=" * 40)
    if surreal_ok and lsp_ok:
        print("✅ All datasets verified successfully!")
        print("🚀 Ready to start training!")
    else:
        print("❌ Dataset verification failed!")
        print("Please check the issues above and fix them before training.")
    
    return surreal_ok and lsp_ok

if __name__ == "__main__":
    main() 