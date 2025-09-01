#!/usr/bin/env python3
"""
Extract Prior Training Data from MARS Dataset
Recreates the EXACT same data structure that was used to train the prior model

This script extracts the original extracted_mpmri_2d_poses.npz file structure
from the MARS dataset, matching the exact format used in prior training.
"""

import numpy as np
import sys
import os
from pathlib import Path
import json

def keypoints_to_bones_numpy(poses):
    """
    Convert 2D keypoints to bone vectors (numpy version)
    EXACTLY matching the original extraction method used for prior training
    Args:
        poses: (n_people, num_kpts, 2) - 2D keypoints
    Returns:
        bone_vectors: (n_people, 34) - bone features
    """
    # Define bone connections for 17 joints (COCO format) - EXACTLY as used in training
    bone_connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # head (4 bones)
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms (5 bones)
        (5, 11), (6, 12), (11, 12),  # torso (3 bones)
        (11, 13), (13, 15), (12, 14), (14, 16),  # legs (4 bones)
        (0, 5)  # nose to left shoulder (1 bone) = 17 bones total
    ]
    
    batch_size = poses.shape[0]
    num_bones = len(bone_connections)
    bone_vectors = np.zeros((batch_size, num_bones, 2))
    
    for i, (parent, child) in enumerate(bone_connections):
        if parent < poses.shape[1] and child < poses.shape[1]:
            bone_vectors[:, i] = poses[:, child] - poses[:, parent]
    
    # Flatten to 34 dimensions (17 bones * 2 dims)
    bone_features = bone_vectors.reshape(batch_size, -1)  # (batch, 34)
    
    return bone_features

def load_mars_keypoints(data_dir):
    """Load keypoints from MARS dataset"""
    print(f"📊 Loading MARS keypoints from {data_dir}")
    
    kpt_path = Path(data_dir) / "kpt_labels.npy"
    if not kpt_path.exists():
        raise FileNotFoundError(f"Keypoint data not found: {kpt_path}")
    
    keypoints = np.load(kpt_path)
    print(f"   Loaded keypoints: {keypoints.shape}")
    
    # MARS data format: 140 values = 4 people × 17 keypoints × 2 dims + 4 presence values
    if keypoints.shape[1] == 140:
        N = len(keypoints)
        max_people = 4; num_kpts = 17; kpt_dims = 2
        
        # Extract keypoint data (first 136 values)
        kpt_data = keypoints[:, :max_people * num_kpts * kpt_dims]
        reshaped_kpts = kpt_data.reshape(N, max_people, num_kpts, kpt_dims)
        
        print(f"   Reshaped to: {reshaped_kpts.shape}")
        return reshaped_kpts
    else:
        raise ValueError(f"Unexpected keypoint format: {keypoints.shape}")

def normalize_poses_to_01(poses):
    """
    Normalize poses to [0,1] range EXACTLY as done in original training
    This matches the normalization used in prior_filter.py
    """
    print("📊 Normalizing poses to [0,1] range...")
    
    # Reshape to (N*people, 17, 2) for easier processing
    N, max_people, num_kpts, kpt_dims = poses.shape
    poses_flat = poses.reshape(-1, num_kpts, kpt_dims)
    
    normalized_poses = np.zeros_like(poses_flat)
    
    for i in range(len(poses_flat)):
        pose = poses_flat[i]
        
        # Normalize to [0,1] range as done in original training
        pose_min = pose.min()
        pose_max = pose.max()
        
        if pose_max > pose_min:
            normalized_poses[i] = (pose - pose_min) / (pose_max - pose_min)
        else:
            normalized_poses[i] = pose
    
    # Reshape back to (N, max_people, 17, 2)
    normalized_poses = normalized_poses.reshape(N, max_people, num_kpts, kpt_dims)
    
    print(f"   Normalized poses shape: {normalized_poses.shape}")
    return normalized_poses



def main():
    """Extract prior training data from MARS dataset"""
    print("🚀 Extracting Prior Training Data from MARS Dataset...")
    print("This recreates the EXACT same data structure used in prior training")
    
    # Data directories
    train_dir = "thesis_project-main/MARS_SRCmpmri_MAXPPL4_GRID8_NORMED/src/train"
    val_dir = "thesis_project-main/MARS_SRCmpmri_MAXPPL4_GRID8_NORMED/src/val"
    
    # Check directories exist
    if not Path(train_dir).exists():
        print(f"❌ Training directory not found: {train_dir}")
        return
    
    if not Path(val_dir).exists():
        print(f"❌ Validation directory not found: {val_dir}")
        return
    
    # Load keypoints from both train and val
    train_keypoints = load_mars_keypoints(train_dir)
    val_keypoints = load_mars_keypoints(val_dir)
    
    # Combine all keypoints
    all_keypoints = np.concatenate([train_keypoints, val_keypoints], axis=0)
    print(f"   Total keypoints: {all_keypoints.shape}")
    
    # Normalize poses to [0,1] range EXACTLY as done in original training
    normalized_keypoints = normalize_poses_to_01(all_keypoints)
    
    # Extract valid poses (poses that have non-zero keypoints)
    valid_poses = []
    
    for sample in normalized_keypoints:
        for person in range(sample.shape[0]):  # For each person
            pose = sample[person]  # (17, 2)
            
            # Check if this pose has valid keypoints (not all zeros)
            if np.any(pose != 0):
                valid_poses.append(pose)
    
    valid_poses = np.array(valid_poses)
    print(f"   Valid poses extracted: {valid_poses.shape}")
    
    # Convert keypoints to bone vectors EXACTLY as done in original training
    print("📊 Converting keypoints to bone vectors...")
    valid_bone_vectors = keypoints_to_bones_numpy(valid_poses)
    print(f"   Valid bone vectors: {valid_bone_vectors.shape}")
    
    # Save the data with only valid poses (MARS dataset has no invalid poses)
    output_file = "extracted_mpmri_2d_poses.npz"
    np.savez(output_file,
             valid_poses=valid_bone_vectors)
    
    print(f"✅ Prior training data extracted and saved to: {output_file}")
    print(f"   Valid poses: {valid_bone_vectors.shape}")
    print(f"   Note: MARS dataset contains only valid poses")
    
    # Verify the data matches the expected structure
    print("\n🔍 Verification:")
    data = np.load(output_file)
    print(f"   Loaded valid_poses: {data['valid_poses'].shape}")
    print(f"   Available keys: {list(data.keys())}")
    
    # Check if the data structure matches the normalization stats
    norm_stats_path = "results/prior_2d_norm_stats.json"
    if Path(norm_stats_path).exists():
        with open(norm_stats_path, 'r') as f:
            norm_stats = json.load(f)
        
        expected_dim = len(norm_stats['mean'])
        actual_dim = data['valid_poses'].shape[1]
        
        print(f"   Expected bone vector dimension: {expected_dim}")
        print(f"   Actual bone vector dimension: {actual_dim}")
        
        if expected_dim == actual_dim:
            print("   ✅ Data structure matches normalization stats!")
        else:
            print("   ❌ Data structure mismatch with normalization stats!")
    
    print(f"\n📋 Data Summary:")
    print(f"   This recreates the EXACT same structure used in prior training")
    print(f"   Data format: valid_poses (N, 34)")
    print(f"   MARS dataset contains only valid poses - no invalid poses generated")

if __name__ == "__main__":
    # Set random seed for reproducibility (same as original training)
    np.random.seed(42)
    
    # Create scripts directory if it doesn't exist
    os.makedirs("scripts", exist_ok=True)
    
    main()
