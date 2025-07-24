#!/usr/bin/env python3
"""
Extract 3D Joint Coordinates from SURREAL _info.mat Files
Converts the downloaded SURREAL data into a format suitable for prior training
"""
import os
import numpy as np
import scipy.io as scio
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import glob

def extract_3d_joints_from_info_mat(info_mat_path):
    """
    Extract 3D joint coordinates from a SURREAL _info.mat file
    
    Args:
        info_mat_path (str): Path to the _info.mat file
        
    Returns:
        dict: Dictionary containing 3D joints and metadata
    """
    try:
        # Load the .mat file
        mat_data = scio.loadmat(info_mat_path)
        
        # Extract 3D joints [3 x 24 x T] where T is number of frames
        joints3d = mat_data['joints3D']  # Shape: (3, 24, T)
        
        # Extract 2D joints [2 x 24 x T] 
        joints2d = mat_data['joints2D']  # Shape: (2, 24, T)
        
        # Extract other useful metadata
        metadata = {
            'sequence': str(mat_data['sequence'][0]) if 'sequence' in mat_data else '',
            'gender': mat_data['gender'].flatten() if 'gender' in mat_data else None,
            'pose': mat_data['pose'] if 'pose' in mat_data else None,  # SMPL pose parameters
            'shape': mat_data['shape'] if 'shape' in mat_data else None,  # SMPL shape parameters
            'camLoc': mat_data['camLoc'].flatten() if 'camLoc' in mat_data else None,
            'camDist': float(mat_data['camDist'][0, 0]) if 'camDist' in mat_data else None,
        }
        
        return {
            'joints3d': joints3d,  # (3, 24, T)
            'joints2d': joints2d,  # (2, 24, T)
            'metadata': metadata
        }
        
    except Exception as e:
        print(f"Error loading {info_mat_path}: {e}")
        return None

def convert_to_prior_format(joints3d, joints2d, metadata, output_format='numpy'):
    """
    Convert extracted data to format suitable for prior training
    
    Args:
        joints3d (np.ndarray): 3D joints [3 x 24 x T]
        joints2d (np.ndarray): 2D joints [2 x 24 x T]
        metadata (dict): Additional metadata
        output_format (str): 'numpy', 'json', or 'csv'
        
    Returns:
        dict: Data in prior training format
    """
    # Transpose to get (T, 24, 3) format for easier processing
    joints3d_t = joints3d.transpose(2, 1, 0)  # (T, 24, 3)
    joints2d_t = joints2d.transpose(2, 1, 0)  # (T, 24, 2)
    
    # SMPL joint names (24 joints)
    joint_names = [
        'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
        'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
        'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
    ]
    
    # Convert to prior training format
    prior_data = {
        'joints3d': joints3d_t,  # (T, 24, 3) - 3D joint coordinates
        'joints2d': joints2d_t,  # (T, 24, 2) - 2D joint coordinates  
        'joint_names': joint_names,
        'num_frames': joints3d_t.shape[0],
        'num_joints': joints3d_t.shape[1],
        'metadata': metadata
    }
    
    return prior_data

def save_prior_data(prior_data, output_path, format='numpy'):
    """
    Save extracted data in the specified format
    
    Args:
        prior_data (dict): Data to save
        output_path (str): Output file path
        format (str): 'numpy', 'json', or 'csv'
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == 'numpy':
        # Save as .npz file (compressed numpy format)
        np.savez_compressed(
            output_path,
            joints3d=prior_data['joints3d'],
            joints2d=prior_data['joints2d'],
            joint_names=prior_data['joint_names'],
            num_frames=prior_data['num_frames'],
            num_joints=prior_data['num_joints'],
            metadata=prior_data['metadata']
        )
        
    elif format == 'json':
        # Save as JSON (convert numpy arrays to lists)
        json_data = {
            'joints3d': prior_data['joints3d'].tolist(),
            'joints2d': prior_data['joints2d'].tolist(),
            'joint_names': prior_data['joint_names'],
            'num_frames': prior_data['num_frames'],
            'num_joints': prior_data['num_joints'],
            'metadata': prior_data['metadata']
        }
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
            
    elif format == 'csv':
        # Save as CSV (flatten 3D joints to 2D table)
        import pandas as pd
        
        # Flatten 3D joints: (T, 24, 3) -> (T*24, 4) with [frame_id, joint_id, x, y, z]
        T, J, D = prior_data['joints3d'].shape
        flattened_data = []
        
        for t in range(T):
            for j in range(J):
                x, y, z = prior_data['joints3d'][t, j]
                flattened_data.append([t, j, x, y, z])
        
        df = pd.DataFrame(flattened_data, columns=['frame_id', 'joint_id', 'x', 'y', 'z'])
        df.to_csv(output_path, index=False)

def process_surreal_dataset(surreal_root, output_dir, output_format='numpy', max_sequences=None):
    """
    Process all SURREAL _info.mat files and extract 3D joint data
    
    Args:
        surreal_root (str): Path to surreal_info/SURREAL/data directory
        output_dir (str): Output directory for extracted data
        output_format (str): 'numpy', 'json', or 'csv'
        max_sequences (int): Maximum number of sequences to process (for testing)
    """
    # Find all _info.mat files
    pattern = os.path.join(surreal_root, "**", "*_info.mat")
    info_mat_files = glob.glob(pattern, recursive=True)
    
    print(f"Found {len(info_mat_files)} _info.mat files")
    
    if max_sequences:
        info_mat_files = info_mat_files[:max_sequences]
        print(f"Processing first {max_sequences} files for testing")
    
    # Process each file
    successful_extractions = 0
    failed_extractions = 0
    
    for info_mat_path in tqdm(info_mat_files, desc="Extracting 3D joints"):
        try:
            # Extract data from .mat file
            extracted_data = extract_3d_joints_from_info_mat(info_mat_path)
            
            if extracted_data is None:
                failed_extractions += 1
                continue
            
            # Convert to prior format
            prior_data = convert_to_prior_format(
                extracted_data['joints3d'],
                extracted_data['joints2d'], 
                extracted_data['metadata'],
                output_format
            )
            
            # Create output path
            relative_path = os.path.relpath(info_mat_path, surreal_root)
            sequence_name = os.path.splitext(os.path.basename(info_mat_path))[0]  # Remove _info.mat
            output_path = os.path.join(output_dir, f"{sequence_name}.{output_format}")
            
            # Save data
            save_prior_data(prior_data, output_path, output_format)
            successful_extractions += 1
            
        except Exception as e:
            print(f"Error processing {info_mat_path}: {e}")
            failed_extractions += 1
    
    print(f"\nExtraction complete!")
    print(f"Successful: {successful_extractions}")
    print(f"Failed: {failed_extractions}")
    print(f"Output directory: {output_dir}")

def create_dataset_summary(output_dir):
    """
    Create a summary of the extracted dataset
    
    Args:
        output_dir (str): Directory containing extracted data
    """
    # Count files and calculate total size
    files = glob.glob(os.path.join(output_dir, "*.*"))
    total_size = sum(os.path.getsize(f) for f in files)
    
    print(f"\nDataset Summary:")
    print(f"Total files: {len(files)}")
    print(f"Total size: {total_size / (1024**3):.2f} GB")
    print(f"Average file size: {total_size / len(files) / (1024**2):.2f} MB")

def main():
    parser = argparse.ArgumentParser(description="Extract 3D joint data from SURREAL _info.mat files")
    parser.add_argument('--surreal-root', default='./surreal_info/SURREAL/data',
                       help='Path to surreal_info/SURREAL/data directory')
    parser.add_argument('--output-dir', default='./extracted_3d_poses',
                       help='Output directory for extracted data')
    parser.add_argument('--format', choices=['numpy', 'json', 'csv'], default='numpy',
                       help='Output format (numpy is most efficient)')
    parser.add_argument('--max-sequences', type=int, default=None,
                       help='Maximum number of sequences to process (for testing)')
    
    args = parser.parse_args()
    
    print("Extracting 3D joint coordinates from SURREAL dataset...")
    print(f"Input: {args.surreal_root}")
    print(f"Output: {args.output_dir}")
    print(f"Format: {args.format}")
    
    # Process the dataset
    process_surreal_dataset(
        args.surreal_root,
        args.output_dir,
        args.format,
        args.max_sequences
    )
    
    # Create summary
    create_dataset_summary(args.output_dir)
    
    print(f"\nExtraction complete! Data ready for prior training.")
    print(f"Next step: Train the prior model using the extracted 3D joint data.")

if __name__ == "__main__":
    main() 