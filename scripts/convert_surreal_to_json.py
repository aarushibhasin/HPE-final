#!/usr/bin/env python3
"""
Convert SURREAL .mat files to JSON format for 3D pose estimation
"""
import os
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.io import loadmat
import cv2
from tqdm import tqdm

def extract_frames_from_video(video_path, output_dir, frame_indices=None):
    """Extract frames from SURREAL video file"""
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return []
    
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_indices is None or frame_count in frame_indices:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save frame
            frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            frames.append(str(frame_path))
        
        frame_count += 1
    
    cap.release()
    return frames

def convert_surreal_to_json(surreal_root, output_root, max_frames_per_sequence=50):
    """Convert SURREAL .mat files to JSON format"""
    
    surreal_root = Path(surreal_root)
    output_root = Path(output_root)
    
    # Create output structure
    for split in ['train', 'val', 'test']:
        for run in ['run0', 'run1', 'run2'] if split == 'train' else ['run0']:
            (output_root / split / run / 'images').mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val', 'test']:
        runs = ['run0', 'run1', 'run2'] if split == 'train' else ['run0']
        
        for run in runs:
            print(f"\n📁 Processing {split}/{run}...")
            
            # Find all sequence directories
            split_dir = surreal_root / 'cmu' / split / run
            if not split_dir.exists():
                print(f"⚠️  Directory not found: {split_dir}")
                continue
            
            sequences = [d for d in split_dir.iterdir() if d.is_dir()]
            print(f"Found {len(sequences)} sequences")
            
            all_samples = []
            
            for seq_dir in tqdm(sequences, desc=f"Processing {split}/{run}"):
                seq_name = seq_dir.name
                
                # Find info.mat file
                info_files = list(seq_dir.glob("*_info.mat"))
                if not info_files:
                    print(f"⚠️  No info.mat found in {seq_dir}")
                    continue
                
                info_file = info_files[0]
                
                try:
                    # Load annotation data
                    info_data = loadmat(info_file)
                    
                    # Extract joint data
                    joints2d = info_data['joints2D']  # [2, 24, T]
                    joints3d = info_data['joints3D']  # [3, 24, T]
                    
                    # Get number of frames
                    num_frames = joints2d.shape[2]
                    
                    # Limit frames if needed
                    if max_frames_per_sequence and num_frames > max_frames_per_sequence:
                        frame_indices = np.linspace(0, num_frames-1, max_frames_per_sequence, dtype=int)
                    else:
                        frame_indices = range(num_frames)
                    
                    # Find video file
                    video_files = list(seq_dir.glob("*.mp4"))
                    if not video_files:
                        print(f"⚠️  No video file found in {seq_dir}")
                        continue
                    
                    video_file = video_files[0]
                    
                    # Create output directory for this sequence
                    seq_output_dir = output_root / split / run / 'images' / seq_name
                    seq_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Extract frames from video
                    frame_paths = extract_frames_from_video(video_file, seq_output_dir, frame_indices)
                    
                    if not frame_paths:
                        print(f"⚠️  No frames extracted from {video_file}")
                        continue
                    
                    # Create samples for each frame
                    for i, frame_idx in enumerate(frame_indices):
                        if i >= len(frame_paths):
                            break
                            
                        # Get joint data for this frame
                        joints2d_frame = joints2d[:, :, frame_idx].T  # [24, 2]
                        joints3d_frame = joints3d[:, :, frame_idx].T  # [24, 3]
                        
                        # Map SMPL joints (24) to our format (16 keypoints)
                        # This mapping needs to be adjusted based on your specific needs
                        smpl_to_16 = [7, 4, 1, 2, 5, 8, 0, 11, 8, 10, 16, 15, 14, 11, 12, 13]
                        
                        joints2d_16 = joints2d_frame[smpl_to_16]
                        joints3d_16 = joints3d_frame[smpl_to_16]
                        
                        # Create sample
                        sample = {
                            'name': f"{seq_name}_frame_{frame_idx:06d}.jpg",
                            'image_path': frame_paths[i],
                            'keypoint2d': joints2d_16.tolist(),
                            'keypoint3d': joints3d_16.tolist(),
                            'intrinsic_matrix': [
                                [1000, 0, 160],  # Default camera intrinsics
                                [0, 1000, 120],
                                [0, 0, 1]
                            ]
                        }
                        
                        all_samples.append(sample)
                
                except Exception as e:
                    print(f"❌ Error processing {seq_dir}: {e}")
                    continue
            
            # Save consolidated annotation file
            if all_samples:
                annotation_file = output_root / split / f'{run}.json'
                with open(annotation_file, 'w') as f:
                    json.dump(all_samples, f, indent=2)
                print(f"✅ Created {annotation_file} with {len(all_samples)} samples")
            else:
                print(f"⚠️  No samples created for {split}/{run}")

def main():
    parser = argparse.ArgumentParser(description="Convert SURREAL .mat files to JSON format")
    parser.add_argument('--surreal-root', required=True,
                       help='Path to SURREAL dataset root directory')
    parser.add_argument('--output-root', default='./data/surreal_3d',
                       help='Output directory for converted data')
    parser.add_argument('--max-frames', type=int, default=50,
                       help='Maximum frames per sequence to extract')
    
    args = parser.parse_args()
    
    print("🔄 Converting SURREAL dataset to JSON format...")
    print(f"Input: {args.surreal_root}")
    print(f"Output: {args.output_root}")
    
    # Check if SURREAL root exists
    if not Path(args.surreal_root).exists():
        print(f"❌ SURREAL root directory not found: {args.surreal_root}")
        return
    
    # Convert data
    convert_surreal_to_json(args.surreal_root, args.output_root, args.max_frames)
    
    print("\n✅ Conversion complete!")
    print(f"📁 Converted data available at: {args.output_root}")
    print("\n📋 Next steps:")
    print("   1. Verify the converted data")
    print("   2. Run: python scripts/verify_dataset_3d.py")
    print("   3. Start training: python quick_start_3d.py --skip-data-download")

if __name__ == "__main__":
    main() 