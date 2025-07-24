#!/usr/bin/env python3
"""
SURREAL 3D Dataset Download Script
Downloads minimal subset needed for 3D pose estimation
"""
import os
import sys
import argparse
import requests
import json
import zipfile
from pathlib import Path
from tqdm import tqdm
import urllib.request

def download_file(url, filename, description="Downloading"):
    """Download file with progress bar"""
    try:
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc=description) as pbar:
            def progress_hook(count, block_size, total_size):
                pbar.total = total_size
                pbar.update(block_size)
            
            urllib.request.urlretrieve(url, filename, reporthook=progress_hook)
        return True
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        return False

def create_surreal_structure(root_dir):
    """Create SURREAL dataset directory structure"""
    structure = {
        'train': ['run0', 'run1', 'run2'],
        'val': ['run0'],
        'test': ['run0']
    }
    
    for split, runs in structure.items():
        for run in runs:
            run_dir = Path(root_dir) / split / run
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / 'images').mkdir(exist_ok=True)
            print(f"✅ Created: {run_dir}")

def download_surreal_minimal_subset(root_dir, subset_ratio=0.1):
    """Download minimal SURREAL subset for 3D pose estimation"""
    
    # SURREAL base URLs
    base_url = "https://www.di.ens.fr/willow/research/surreal/data/SURREAL/data"
    
    # Minimal subset: 3 sequences from different subjects
    sequences = [
        # Train sequences
        ("train", "run0", "cmu", "c0001", "c0001"),
        ("train", "run1", "cmu", "c0002", "c0002"), 
        ("train", "run2", "cmu", "c0003", "c0003"),
        # Val sequence
        ("val", "run0", "cmu", "c0004", "c0004"),
        # Test sequence
        ("test", "run0", "cmu", "c0005", "c0005"),
    ]
    
    # Frame ranges (subset to reduce download size)
    frame_ranges = {
        'train': range(1, 101),  # 100 frames per sequence
        'val': range(1, 51),     # 50 frames
        'test': range(1, 51),    # 50 frames
    }
    
    total_downloads = 0
    successful_downloads = 0
    
    for split, run, subject, action, subaction in sequences:
        print(f"\n📥 Downloading {split}/{run}/{subject}/{action}/{subaction}")
        
        # Create directory
        seq_dir = Path(root_dir) / split / run / 'images' / f'{subject}_{action}_{subaction}'
        seq_dir.mkdir(parents=True, exist_ok=True)
        
        # Download frames
        frame_range = frame_ranges[split]
        for frame_num in tqdm(frame_range, desc=f"Frames for {split}/{run}"):
            frame_str = f"{frame_num:06d}"
            
            # Image URL
            img_url = f"{base_url}/{run}/{subject}/{split}/{subject}_{action}_{subaction}_{frame_str}.jpg"
            img_path = seq_dir / f"{subject}_{action}_{subaction}_{frame_str}.jpg"
            
            # Pose annotation URL
            pose_url = f"{base_url}/{run}/{subject}/{split}/{subject}_{action}_{subaction}_{frame_str}_poses.json"
            pose_path = seq_dir / f"{subject}_{action}_{subaction}_{frame_str}_poses.json"
            
            total_downloads += 2
            
            # Download image
            if not img_path.exists():
                if download_file(img_url, img_path, f"Image {frame_str}"):
                    successful_downloads += 1
                else:
                    print(f"⚠️  Skipping image {frame_str} (not available)")
            
            # Download pose annotation
            if not pose_path.exists():
                if download_file(pose_url, pose_path, f"Pose {frame_str}"):
                    successful_downloads += 1
                else:
                    print(f"⚠️  Skipping pose {frame_str} (not available)")
    
    print(f"\n📊 Download Summary:")
    print(f"   Total files attempted: {total_downloads}")
    print(f"   Successfully downloaded: {successful_downloads}")
    print(f"   Success rate: {successful_downloads/total_downloads*100:.1f}%")
    
    return successful_downloads > 0

def create_annotation_files(root_dir):
    """Create consolidated annotation files from individual pose files"""
    print("\n📝 Creating annotation files...")
    
    for split in ['train', 'val', 'test']:
        if split == 'train':
            runs = ['run0', 'run1', 'run2']
        else:
            runs = ['run0']
        
        for run in runs:
            run_dir = Path(root_dir) / split / run / 'images'
            if not run_dir.exists():
                continue
            
            # Find all pose files
            pose_files = list(run_dir.rglob("*_poses.json"))
            
            if not pose_files:
                print(f"⚠️  No pose files found in {run_dir}")
                continue
            
            # Consolidate annotations
            consolidated_samples = []
            
            for pose_file in tqdm(pose_files, desc=f"Processing {split}/{run}"):
                try:
                    with open(pose_file, 'r') as f:
                        pose_data = json.load(f)
                    
                    # Extract frame info from filename
                    filename = pose_file.stem  # e.g., "cmu_c0001_c0001_000001_poses"
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        subject, action, subaction, frame_num = parts[:4]
                        image_name = f"{subject}_{action}_{subaction}_{frame_num}.jpg"
                        
                        # Find corresponding image
                        image_path = pose_file.parent / image_name
                        if image_path.exists():
                            # Extract pose information
                            sample = {
                                'name': image_name,
                                'image_path': str(image_path),
                                'keypoint2d': pose_data.get('joints2d', []),
                                'keypoint3d': pose_data.get('joints3d', []),
                                'intrinsic_matrix': pose_data.get('intrinsic_matrix', [])
                            }
                            consolidated_samples.append(sample)
                
                except Exception as e:
                    print(f"⚠️  Error processing {pose_file}: {e}")
            
            # Save consolidated annotation file
            if consolidated_samples:
                annotation_file = Path(root_dir) / split / f'{run}.json'
                with open(annotation_file, 'w') as f:
                    json.dump(consolidated_samples, f, indent=2)
                print(f"✅ Created {annotation_file} with {len(consolidated_samples)} samples")

def create_camera_file(root_dir):
    """Create default camera parameters file"""
    print("\n📷 Creating camera parameters...")
    
    # Default SURREAL camera parameters
    default_camera = {
        "focal_length": [1000, 1000],
        "principal_point": [256, 256],
        "image_size": [512, 512],
        "intrinsic_matrix": [
            [1000, 0, 256],
            [0, 1000, 256],
            [0, 0, 1]
        ]
    }
    
    camera_file = Path(root_dir) / 'camera.json'
    with open(camera_file, 'w') as f:
        json.dump(default_camera, f, indent=2)
    
    print(f"✅ Created {camera_file}")

def main():
    parser = argparse.ArgumentParser(description="Download minimal SURREAL 3D dataset")
    parser.add_argument('--root', default='./data/surreal_3d', 
                       help='Root directory for SURREAL dataset')
    parser.add_argument('--subset-ratio', type=float, default=0.1,
                       help='Subset ratio (not used in this minimal version)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip download, only create structure')
    parser.add_argument('--create-annotations', action='store_true',
                       help='Create consolidated annotation files')
    
    args = parser.parse_args()
    
    print("🚀 SURREAL 3D Dataset Download")
    print("=" * 50)
    
    # Create directory structure
    create_surreal_structure(args.root)
    
    if not args.skip_download:
        # Download minimal subset
        success = download_surreal_minimal_subset(args.root, args.subset_ratio)
        if not success:
            print("❌ Download failed!")
            return
    
    if args.create_annotations:
        # Create annotation files
        create_annotation_files(args.root)
        create_camera_file(args.root)
    
    print("\n✅ SURREAL 3D dataset setup complete!")
    print(f"📁 Dataset location: {args.root}")
    print("\n📋 Next steps:")
    print("   1. Verify data integrity")
    print("   2. Run quick_start_3d.py")
    print("   3. Start training!")

if __name__ == "__main__":
    main() 