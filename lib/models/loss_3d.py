"""
3D Loss Functions for Pose Estimation
Combines 2D heatmap losses with 3D depth losses
"""
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from .loss import JointsMSELoss


class Joints3DLoss(nn.Module):
    """
    Combined 2D + 3D loss for pose estimation
    
    Args:
        alpha_2d (float): Weight for 2D loss
        alpha_3d (float): Weight for 3D loss
        reduction (str): Reduction method for losses
    """
    def __init__(self, alpha_2d=1.0, alpha_3d=1.0, reduction='mean'):
        super(Joints3DLoss, self).__init__()
        self.alpha_2d = alpha_2d
        self.alpha_3d = alpha_3d
        self.reduction = reduction
        
        # 2D heatmap loss
        self.criterion_2d = JointsMSELoss(reduction=reduction)
        
        # 3D depth loss
        if reduction == 'mean':
            self.criterion_3d = nn.MSELoss(reduction='mean')
        else:
            self.criterion_3d = nn.MSELoss(reduction='none')
    
    def forward(self, output_2d, output_3d, target_2d, target_3d, target_weight=None):
        """
        Args:
            output_2d: Predicted 2D heatmaps (B, K, H, W)
            output_3d: Predicted 3D depth maps (B, K, H, W)
            target_2d: Ground truth 2D heatmaps (B, K, H, W)
            target_3d: Ground truth 3D depth maps (B, K, H, W)
            target_weight: Visibility weights (B, K, 1)
        """
        # 2D heatmap loss
        loss_2d = self.criterion_2d(output_2d, target_2d, target_weight)
        
        # 3D depth loss
        if self.reduction == 'mean':
            loss_3d = self.criterion_3d(output_3d, target_3d)
        else:
            # Apply visibility weights to 3D loss
            B, K, H, W = output_3d.shape
            loss_3d = self.criterion_3d(output_3d, target_3d)  # (B, K, H, W)
            if target_weight is not None:
                loss_3d = loss_3d * target_weight.view(B, K, 1, 1)
            loss_3d = loss_3d.mean(dim=(2, 3))  # (B, K)
        
        total_loss = self.alpha_2d * loss_2d + self.alpha_3d * loss_3d
        
        return total_loss, loss_2d, loss_3d


class ConsLoss3D(nn.Module):
    """
    3D consistency loss for teacher-student learning
    Ensures consistency between student and teacher predictions in both 2D and 3D
    """
    def __init__(self, alpha_2d=1.0, alpha_3d=1.0):
        super(ConsLoss3D, self).__init__()
        self.alpha_2d = alpha_2d
        self.alpha_3d = alpha_3d

    def forward(self, stu_out_2d, stu_out_3d, tea_out_2d, tea_out_3d, valid_mask=None, tea_mask=None):
        """
        Args:
            stu_out_2d: Student 2D predictions (B, K, H, W)
            stu_out_3d: Student 3D predictions (B, K, H, W)
            tea_out_2d: Teacher 2D predictions (B, K, H, W)
            tea_out_3d: Teacher 3D predictions (B, K, H, W)
            valid_mask: Valid pixel mask (B, H, W)
            tea_mask: Teacher confidence mask (B, K)
        """
        # 2D consistency loss
        diff_2d = stu_out_2d - tea_out_2d  # (B, K, H, W)
        if tea_mask is not None:
            diff_2d *= tea_mask[:, :, None, None]  # Apply confidence mask
        loss_2d = torch.mean((diff_2d) ** 2, dim=1)  # (B, H, W)
        
        # 3D consistency loss
        diff_3d = stu_out_3d - tea_out_3d  # (B, K, H, W)
        if tea_mask is not None:
            diff_3d *= tea_mask[:, :, None, None]  # Apply confidence mask
        loss_3d = torch.mean((diff_3d) ** 2, dim=1)  # (B, H, W)
        
        # Apply valid mask if provided
        if valid_mask is not None:
            loss_2d = loss_2d[valid_mask]
            loss_3d = loss_3d[valid_mask]
        
        total_loss = self.alpha_2d * loss_2d.mean() + self.alpha_3d * loss_3d.mean()
        
        return total_loss


class DepthConsistencyLoss(nn.Module):
    """
    Depth consistency loss using SURREAL depth maps
    Ensures predicted 3D poses are consistent with depth information
    """
    def __init__(self, alpha=0.1):
        super(DepthConsistencyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred_joints3d, depth_map, camera_intrinsics):
        """
        Args:
            pred_joints3d: Predicted 3D joint positions (B, K, 3)
            depth_map: Ground truth depth map (B, H, W)
            camera_intrinsics: Camera intrinsic matrix (B, 3, 3)
        """
        B, K, _ = pred_joints3d.shape
        
        # Project 3D joints to 2D
        pred_joints2d = self.project_3d_to_2d(pred_joints3d, camera_intrinsics)  # (B, K, 2)
        
        # Sample depth values at predicted 2D locations
        sampled_depths = self.sample_depth_at_joints(depth_map, pred_joints2d)  # (B, K)
        
        # Compare predicted Z with sampled depth
        pred_depths = pred_joints3d[:, :, 2]  # (B, K)
        depth_loss = F.mse_loss(pred_depths, sampled_depths)
        
        return self.alpha * depth_loss

    def project_3d_to_2d(self, joints3d, camera_intrinsics):
        """Project 3D joints to 2D using camera intrinsics"""
        # joints3d: (B, K, 3), camera_intrinsics: (B, 3, 3)
        B, K, _ = joints3d.shape
        
        # Reshape for batch matrix multiplication
        joints3d_homo = torch.cat([joints3d, torch.ones(B, K, 1, device=joints3d.device)], dim=2)  # (B, K, 4)
        
        # Project to 2D
        joints2d_homo = torch.bmm(joints3d_homo, camera_intrinsics.transpose(1, 2))  # (B, K, 3)
        joints2d = joints2d_homo[:, :, :2] / joints2d_homo[:, :, 2:3]  # (B, K, 2)
        
        return joints2d

    def sample_depth_at_joints(self, depth_map, joints2d):
        """Sample depth values at 2D joint locations"""
        B, H, W = depth_map.shape
        K = joints2d.shape[1]
        
        # Normalize coordinates to [-1, 1] for grid_sample
        joints2d_norm = joints2d.clone()
        joints2d_norm[:, :, 0] = joints2d_norm[:, :, 0] / (W - 1) * 2 - 1  # x
        joints2d_norm[:, :, 1] = joints2d_norm[:, :, 1] / (H - 1) * 2 - 1  # y
        
        # Reshape for grid_sample
        joints2d_norm = joints2d_norm.view(B, K, 1, 2)  # (B, K, 1, 2)
        depth_map = depth_map.unsqueeze(1)  # (B, 1, H, W)
        
        # Sample depth values
        sampled_depths = F.grid_sample(depth_map, joints2d_norm, mode='bilinear', 
                                      padding_mode='border', align_corners=True)  # (B, 1, K, 1)
        sampled_depths = sampled_depths.squeeze(1).squeeze(-1)  # (B, K)
        
        return sampled_depths


class GeometricConsistencyLoss(nn.Module):
    """
    Geometric consistency loss for 3D poses
    Ensures bone lengths and joint angles are realistic
    """
    def __init__(self, alpha=0.05):
        super(GeometricConsistencyLoss, self).__init__()
        self.alpha = alpha
        
        # Define bone connections (parent-child pairs)
        self.bone_pairs = [
            [8, 9],   # 0,-1
            [2, 8],   # 1,0
            [3, 8],   # 2,0
            [12, 8],  # 3,0
            [13, 8],  # 4,0
            [1, 2],   # 5,1
            [0, 1],   # 6,5
            [4, 3],   # 7,2
            [5, 4],   # 8,7
            [11, 12], # 9,3
            [10, 11], # 10,9
            [14, 13], # 11,4
            [15, 14]  # 12,11
        ]

    def forward(self, joints3d, target_joints3d=None):
        """
        Args:
            joints3d: Predicted 3D joints (B, K, 3)
            target_joints3d: Target 3D joints (B, K, 3) - optional
        """
        B, K, _ = joints3d.shape
        
        # Compute bone lengths
        bone_lengths = []
        for parent, child in self.bone_pairs:
            bone_vec = joints3d[:, child] - joints3d[:, parent]  # (B, 3)
            bone_length = torch.norm(bone_vec, dim=1)  # (B,)
            bone_lengths.append(bone_length)
        
        bone_lengths = torch.stack(bone_lengths, dim=1)  # (B, num_bones)
        
        if target_joints3d is not None:
            # Compare with target bone lengths
            target_bone_lengths = []
            for parent, child in self.bone_pairs:
                target_bone_vec = target_joints3d[:, child] - target_joints3d[:, parent]
                target_bone_length = torch.norm(target_bone_vec, dim=1)
                target_bone_lengths.append(target_bone_length)
            
            target_bone_lengths = torch.stack(target_bone_lengths, dim=1)
            bone_loss = F.mse_loss(bone_lengths, target_bone_lengths)
        else:
            # Use bone length consistency (bones should have reasonable lengths)
            # This is a simplified version - you could use learned bone length priors
            bone_loss = torch.var(bone_lengths, dim=1).mean()  # Encourage consistent bone lengths
        
        return self.alpha * bone_loss 