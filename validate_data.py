import numpy as np
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

def validate_3d_pose_data(data_dir="extracted_3d_poses"):
    """Validate the quality of extracted 3D pose data"""
    
    print("Validating 3D pose data...")
    
    # Find all .npz files
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    print(f"Found {len(files)} .npz files")
    
    if len(files) == 0:
        print("No .npz files found!")
        return
    
    # Statistics
    valid_files = 0
    invalid_files = 0
    total_poses = 0
    valid_poses = 0
    
    # Data statistics
    all_joints3d = []
    all_joints2d = []
    
    # Validation results
    validation_results = {
        'nan_detected': 0,
        'inf_detected': 0,
        'extreme_values': 0,
        'all_zeros': 0,
        'missing_data': 0
    }
    
    for file_path in tqdm(files, desc="Validating files"):
        try:
            data = np.load(file_path)
            
            # Check required keys
            required_keys = ['joints3d']
            missing_keys = [key for key in required_keys if key not in data]
            
            if missing_keys:
                validation_results['missing_data'] += 1
                invalid_files += 1
                continue
            
            joints3d = data['joints3d']
            
            # Check for NaN
            if np.isnan(joints3d).any():
                validation_results['nan_detected'] += 1
                invalid_files += 1
                continue
            
            # Check for infinite values
            if np.isinf(joints3d).any():
                validation_results['inf_detected'] += 1
                invalid_files += 1
                continue
            
            # Check for extreme values
            if np.any(np.abs(joints3d) > 1000.0):
                validation_results['extreme_values'] += 1
                invalid_files += 1
                continue
            
            # Check for all zeros
            if np.allclose(joints3d, 0, atol=1e-6):
                validation_results['all_zeros'] += 1
                invalid_files += 1
                continue
            
            # Data is valid
            valid_files += 1
            total_poses += joints3d.shape[0]  # Number of frames
            valid_poses += joints3d.shape[0]
            
            # Store for statistics (take a random frame from each sequence)
            num_frames = joints3d.shape[0]
            random_frame_idx = np.random.randint(0, num_frames)
            all_joints3d.append(joints3d[random_frame_idx])
            
            # Also store 2D joints if available
            if 'joints2d' in data:
                all_joints2d.append(data['joints2d'][random_frame_idx])
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            invalid_files += 1
            continue
    
    # Print validation results
    print("\n=== VALIDATION RESULTS ===")
    print(f"Total files: {len(files)}")
    print(f"Valid files: {valid_files}")
    print(f"Invalid files: {invalid_files}")
    print(f"Success rate: {valid_files/len(files)*100:.2f}%")
    
    print(f"\nTotal poses: {total_poses}")
    print(f"Valid poses: {valid_poses}")
    
    print("\n=== VALIDATION ISSUES ===")
    for issue, count in validation_results.items():
        if count > 0:
            print(f"{issue}: {count} files")
    
    if valid_files == 0:
        print("\nNo valid files found! Cannot proceed with training.")
        return False
    
    # Compute statistics for valid data
    if all_joints3d:
        all_joints3d = np.vstack(all_joints3d)
        
        print("\n=== DATA STATISTICS ===")
        print("3D Joint Positions:")
        print(f"  Shape: {all_joints3d.shape}")
        print(f"  Mean: {np.mean(all_joints3d):.6f}")
        print(f"  Std: {np.std(all_joints3d):.6f}")
        print(f"  Min: {np.min(all_joints3d):.6f}")
        print(f"  Max: {np.max(all_joints3d):.6f}")
        
        if all_joints2d:
            all_joints2d = np.vstack(all_joints2d)
            print("\n2D Joint Positions:")
            print(f"  Shape: {all_joints2d.shape}")
            print(f"  Mean: {np.mean(all_joints2d):.6f}")
            print(f"  Std: {np.std(all_joints2d):.6f}")
            print(f"  Min: {np.min(all_joints2d):.6f}")
            print(f"  Max: {np.max(all_joints2d):.6f}")
        
        # Plot distributions
        plot_data_distributions(all_joints3d, all_joints2d if all_joints2d else None)
    
    return True

def plot_data_distributions(joints3d, joints2d=None):
    """Plot distributions of the data"""
    
    if joints2d is not None:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 3D joint positions distribution
    axes[0, 0].hist(joints3d.flatten(), bins=50, alpha=0.7, color='blue')
    axes[0, 0].set_title('3D Joint Positions Distribution')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 3D joint position ranges
    joint_ranges_3d = np.max(joints3d.reshape(-1, 3), axis=1) - np.min(joints3d.reshape(-1, 3), axis=1)
    axes[0, 1].hist(joint_ranges_3d, bins=50, alpha=0.7, color='red')
    axes[0, 1].set_title('3D Joint Position Ranges')
    axes[0, 1].set_xlabel('Range')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    
    if joints2d is not None:
        # 2D joint positions distribution
        axes[1, 0].hist(joints2d.flatten(), bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('2D Joint Positions Distribution')
        axes[1, 0].set_xlabel('Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 2D joint position ranges
        joint_ranges_2d = np.max(joints2d.reshape(-1, 2), axis=1) - np.min(joints2d.reshape(-1, 2), axis=1)
        axes[1, 1].hist(joint_ranges_2d, bins=50, alpha=0.7, color='orange')
        axes[1, 1].set_title('2D Joint Position Ranges')
        axes[1, 1].set_xlabel('Range')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data_validation_plots.png', dpi=300, bbox_inches='tight')
    print("Data validation plots saved to: data_validation_plots.png")
    plt.show()

if __name__ == "__main__":
    success = validate_3d_pose_data()
    
    if success:
        print("\n✅ Data validation passed! Ready for training.")
    else:
        print("\n❌ Data validation failed! Please check your data.") 