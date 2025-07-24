import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from train_prior_corrected import PoseNDF3D, CorrectedPoseDataset

def create_invalid_pose(valid_pose):
    # Add significant noise and random joint swaps
    noise = np.random.normal(0, 2.0, valid_pose.shape)
    invalid_pose = valid_pose + noise
    if np.random.random() < 0.3:
        idx1, idx2 = np.random.choice(24, 2, replace=False)
        invalid_pose[idx1], invalid_pose[idx2] = invalid_pose[idx2].copy(), invalid_pose[idx1].copy()
    if np.random.random() < 0.2:
        scale = np.random.uniform(5.0, 10.0)
        invalid_pose *= scale
    return invalid_pose

def test_valid_invalid(model_path="checkpoints/prior_3d_final_corrected.pth", data_dir="extracted_3d_poses", num_samples=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    input_dim = 72
    model = PoseNDF3D(input_dim=input_dim).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load dataset
    dataset = CorrectedPoseDataset(data_dir)
    print(f"Loaded {len(dataset.valid_data)} valid poses for testing.")

    for i in range(num_samples):
        idx = random.randint(0, len(dataset.valid_data)-1)
        valid_pose = dataset.valid_data[idx]
        invalid_pose = create_invalid_pose(valid_pose.copy())

        # Normalize using dataset stats
        valid_norm = dataset.normalize_data(valid_pose).flatten().astype(np.float32)
        invalid_norm = dataset.normalize_data(invalid_pose).flatten().astype(np.float32)

        valid_tensor = torch.from_numpy(valid_norm).unsqueeze(0).to(device)
        invalid_tensor = torch.from_numpy(invalid_norm).unsqueeze(0).to(device)

        with torch.no_grad():
            valid_out = model(valid_tensor)
            invalid_out = model(invalid_tensor)

        print(f"Sample {i+1}:")
        print(f"  Valid pose output:   {valid_out.item():.6f}")
        print(f"  Invalid pose output: {invalid_out.item():.6f}")

        # Visualize both
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(valid_pose[:, 0], valid_pose[:, 1], valid_pose[:, 2], c='b', marker='o')
        ax1.set_title(f"Valid Pose {i+1}")
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(invalid_pose[:, 0], invalid_pose[:, 1], invalid_pose[:, 2], c='r', marker='o')
        ax2.set_title(f"Invalid Pose {i+1}")
        plt.show()

if __name__ == "__main__":
    test_valid_invalid() 