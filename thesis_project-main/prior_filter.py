import torch
import numpy as np
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from prior.models_2d import PoseNDF2D

# Paths (edit as needed)
PRIOR_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'prior_2d_final.pth')
NORM_STATS_PATH = os.path.join(os.path.dirname(__file__), '..', 'results', 'prior_2d_norm_stats.json')

# Load prior and normalization stats
prior = PoseNDF2D(input_dim=34)
prior.load_state_dict(torch.load(PRIOR_PATH, map_location='cpu')['model_state_dict'])
prior.eval()
with open(NORM_STATS_PATH, 'r') as f:
    norm_stats = json.load(f)
mean = np.array(norm_stats['mean'])
std = np.array(norm_stats['std'])

def filter_poses_with_prior(pred_kpts, threshold=0.5):
    # pred_kpts: (batch_size, max_poses, num_keypoints, 2)
    batch_size, max_poses, num_keypoints, kpt_dims = pred_kpts.shape
    valid_mask = np.ones((batch_size, max_poses), dtype=bool)
    for b in range(batch_size):
        for p in range(max_poses):
            pose = pred_kpts[b, p].flatten()
            
            # FIXED: Normalize pose to [0,1] range like training data
            pose_min = pose.min()
            pose_max = pose.max()
            if pose_max > pose_min:
                pose_normalized = (pose - pose_min) / (pose_max - pose_min)
            else:
                pose_normalized = pose
            
            # Now normalize using prior stats
            pose_norm = (pose_normalized - mean) / (std + 1e-8)
            pose_tensor = torch.from_numpy(pose_norm).float().unsqueeze(0)
            with torch.no_grad():
                score = prior(pose_tensor).item()
            # FIXED: Valid poses have low scores (close to 0), invalid poses have high scores (close to 1)
            # So we keep poses with scores BELOW threshold (meaning they are valid)
            if score > threshold:
                valid_mask[b, p] = False  # Reject invalid poses (high scores)
    return valid_mask 