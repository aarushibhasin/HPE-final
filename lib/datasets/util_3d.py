"""
3D Data Processing Utilities for Pose Estimation
"""
import numpy as np
import torch
import torch.nn.functional as F
from .util import generate_target, softargmax2d


def generate_3d_target(joints_3d, joints_vis, heatmap_size, sigma, image_size):
    """
    Generate 2D heatmaps + 3D depth targets
    
    Args:
        joints_3d: 3D joint positions (K, 3)
        joints_vis: Joint visibility (K, 1)
        heatmap_size: (W, H) of heatmap
        sigma: Gaussian sigma for heatmap
        image_size: (W, H) of image
    
    Returns:
        target_2d: 2D heatmaps (K, H, W)
        target_3d: 3D depth maps (K, H, W)
        target_weight: Visibility weights (K, 1)
    """
    num_joints = joints_3d.shape[0]
    
    # Generate 2D heatmaps (using existing function)
    target_2d, target_weight = generate_target(
        joints_3d[:, :2], joints_vis, heatmap_size, sigma, image_size
    )
    
    # Generate 3D depth targets
    target_3d = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
    
    # Normalize depth relative to root joint (e.g., hip center)
    root_depth = joints_3d[0, 2]  # Assuming joint 0 is root
    relative_depths = joints_3d[:, 2] - root_depth
    
    for joint_id in range(num_joints):
        if joints_vis[joint_id, 0] > 0.5:
            # Get 2D position
            feat_stride = np.array(image_size) / np.array(heatmap_size)
            mu_x = int(joints_3d[joint_id, 0] / feat_stride[0] + 0.5)
            mu_y = int(joints_3d[joint_id, 1] / feat_stride[1] + 0.5)
            
            # Check bounds
            if 0 <= mu_x < heatmap_size[0] and 0 <= mu_y < heatmap_size[1]:
                # Create depth heatmap (similar to 2D but with depth values)
                tmp_size = sigma * 3
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                
                # Generate gaussian for depth
                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
                
                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
                img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], heatmap_size[1])
                
                # Scale gaussian by relative depth
                depth_value = relative_depths[joint_id]
                target_3d[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]] * depth_value
    
    return target_2d, target_3d, target_weight


def heatmap_to_3d_coords(heatmaps_2d, heatmaps_3d):
    """
    Convert 2D heatmaps and 3D depth maps to 3D coordinates
    
    Args:
        heatmaps_2d: 2D heatmaps (B, K, H, W)
        heatmaps_3d: 3D depth maps (B, K, H, W)
    
    Returns:
        coords_3d: 3D coordinates (B, K, 3)
    """
    B, K, H, W = heatmaps_2d.shape
    
    # Get 2D coordinates from heatmaps
    coords_2d = softargmax2d(heatmaps_2d)  # (B, K, 2)
    
    # Get depth values at 2D coordinates
    coords_2d_norm = coords_2d.clone()
    coords_2d_norm[:, :, 0] = coords_2d_norm[:, :, 0] / (W - 1) * 2 - 1
    coords_2d_norm[:, :, 1] = coords_2d_norm[:, :, 1] / (H - 1) * 2 - 1
    
    # Sample depth values using grid_sample
    coords_2d_norm = coords_2d_norm.view(B, K, 1, 2)
    depths = F.grid_sample(
        heatmaps_3d, coords_2d_norm, mode='bilinear', 
        padding_mode='border', align_corners=True
    ).squeeze(-1).squeeze(-1)  # (B, K)
    
    # Combine 2D coordinates with depth
    coords_3d = torch.cat([coords_2d, depths.unsqueeze(-1)], dim=-1)  # (B, K, 3)
    
    return coords_3d


def normalize_3d_pose(joints_3d, root_joint=0):
    """
    Normalize 3D pose by centering at root joint and scaling
    
    Args:
        joints_3d: 3D joint positions (K, 3)
        root_joint: Index of root joint
    
    Returns:
        normalized_joints: Normalized 3D joints (K, 3)
    """
    # Center at root joint
    centered_joints = joints_3d - joints_3d[root_joint:root_joint+1]
    
    # Scale by distance from root to a reference joint (e.g., shoulder)
    ref_joint = 1  # Right shoulder
    scale_factor = np.linalg.norm(centered_joints[ref_joint])
    if scale_factor > 0:
        normalized_joints = centered_joints / scale_factor
    else:
        normalized_joints = centered_joints
    
    return normalized_joints


def add_3d_noise(joints_3d, rotation_noise=0.1, translation_noise=0.1, joint_noise=0.05):
    """
    Add noise to 3D joints for prior training
    
    Args:
        joints_3d: 3D joint positions (B, K, 3)
        rotation_noise: Rotation noise in radians
        translation_noise: Translation noise
        joint_noise: Per-joint noise
    
    Returns:
        noisy_joints: Noisy 3D joints (B, K, 3)
    """
    B, K, _ = joints_3d.shape
    noisy_joints = joints_3d.copy()
    
    for b in range(B):
        # Add random rotation
        if rotation_noise > 0:
            angle = np.random.uniform(-rotation_noise, rotation_noise)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            noisy_joints[b] = np.dot(noisy_joints[b], rotation_matrix.T)
        
        # Add random translation
        if translation_noise > 0:
            translation = np.random.uniform(-translation_noise, translation_noise, 3)
            noisy_joints[b] += translation
        
        # Add per-joint noise
        if joint_noise > 0:
            joint_noise_vals = np.random.uniform(-joint_noise, joint_noise, (K, 3))
            noisy_joints[b] += joint_noise_vals
    
    return noisy_joints


def compute_3d_accuracy(pred_joints_3d, gt_joints_3d, threshold=0.1):
    """
    Compute 3D pose estimation accuracy (MPJPE)
    
    Args:
        pred_joints_3d: Predicted 3D joints (B, K, 3)
        gt_joints_3d: Ground truth 3D joints (B, K, 3)
        threshold: Distance threshold for accuracy
    
    Returns:
        mpjpe: Mean Per Joint Position Error
        pck_3d: 3D Percentage of Correct Keypoints
    """
    # Compute MPJPE
    distances = np.linalg.norm(pred_joints_3d - gt_joints_3d, axis=-1)  # (B, K)
    mpjpe = np.mean(distances)
    
    # Compute PCK-3D
    correct_joints = distances < threshold
    pck_3d = np.mean(correct_joints)
    
    return mpjpe, pck_3d


def project_3d_to_2d(joints_3d, camera_intrinsics):
    """
    Project 3D joints to 2D using camera intrinsics
    
    Args:
        joints_3d: 3D joint positions (B, K, 3)
        camera_intrinsics: Camera intrinsic matrix (B, 3, 3)
    
    Returns:
        joints_2d: 2D joint positions (B, K, 2)
    """
    B, K, _ = joints_3d.shape
    
    # Add homogeneous coordinate
    joints_3d_homo = np.concatenate([joints_3d, np.ones((B, K, 1))], axis=2)  # (B, K, 4)
    
    # Project to 2D
    joints_2d_homo = np.matmul(joints_3d_homo, camera_intrinsics.transpose(0, 2, 1))  # (B, K, 3)
    joints_2d = joints_2d_homo[:, :, :2] / joints_2d_homo[:, :, 2:3]  # (B, K, 2)
    
    return joints_2d


def backproject_2d_to_3d(joints_2d, depths, camera_intrinsics):
    """
    Backproject 2D joints to 3D using depth and camera intrinsics
    
    Args:
        joints_2d: 2D joint positions (B, K, 2)
        depths: Depth values (B, K)
        camera_intrinsics: Camera intrinsic matrix (B, 3, 3)
    
    Returns:
        joints_3d: 3D joint positions (B, K, 3)
    """
    B, K, _ = joints_2d.shape
    
    # Create homogeneous 2D coordinates
    joints_2d_homo = np.concatenate([joints_2d, np.ones((B, K, 1))], axis=2)  # (B, K, 3)
    
    # Scale by depth
    joints_2d_scaled = joints_2d_homo * depths[:, :, np.newaxis]  # (B, K, 3)
    
    # Backproject to 3D
    camera_intrinsics_inv = np.linalg.inv(camera_intrinsics)  # (B, 3, 3)
    joints_3d = np.matmul(joints_2d_scaled, camera_intrinsics_inv.transpose(0, 2, 1))  # (B, K, 3)
    
    return joints_3d 