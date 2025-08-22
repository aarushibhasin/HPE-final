"""
3D Prior Network for Human Pose Estimation
Extends the 2D prior to work with 3D bone orientations
"""
import torch
import torch.nn as nn


class BoneMLP3D(nn.Module):
    """3D bone feature extractor"""
    def __init__(self, bone_dim=3, bone_feature_dim=9, parent=-1):
        super().__init__()
        if parent == -1:
            in_features = bone_dim
        else:
            in_features = bone_dim + bone_feature_dim
        n_features = bone_dim + bone_feature_dim

        self.net = nn.Sequential(
            nn.Linear(in_features, n_features),
            nn.ReLU(),
            nn.Linear(n_features, bone_feature_dim),
            nn.ReLU()
        )

    def forward(self, bone_feat):
        return self.net(bone_feat)


class StructureEncoder3D(nn.Module):
    """3D structure encoder for human pose"""
    def __init__(self, local_feature_size=9):
        super().__init__()

        self.bone_dim = 3  # 3D bone vectors
        self.input_dim = self.bone_dim  
        self.parent_mapping = [-1, 0, 0, 0, 0, 1, 5, 2, 7, 3, 9, 4, 11]

        self.num_joints = len(self.parent_mapping)
        self.out_dim = self.num_joints * local_feature_size

        self.net = nn.ModuleList([
            BoneMLP3D(self.input_dim, local_feature_size, self.parent_mapping[i]) 
            for i in range(self.num_joints)
        ])

    def get_out_dim(self):
        return self.out_dim

    def forward(self, x):
        """
        Args:
            x: 3D bone orientations (B, num_bones, 3)
        """
        features = [None] * self.num_joints
        for i, mlp in enumerate(self.net):
            parent = self.parent_mapping[i]
            if parent == -1:
                features[i] = mlp(x[:, i, :])
            else:
                inp = torch.cat((x[:, i, :], features[parent]), dim=-1)
                features[i] = mlp(inp)
        features = torch.cat(features, dim=-1) 
        return features


class PoseNDF3D(nn.Module):
    """3D Neural Distance Field for pose validation"""
    def __init__(self, feat_dim=9, hid_layer=[512, 512, 512, 512, 512], weight_norm=True):
        super().__init__()
        self.enc = StructureEncoder3D(feat_dim)
        self.dfnet = DFNet(feat_dim*13, hid_layer, weight_norm)  # 13 joints * 3D features
       
    def train(self, mode=True):
        super().train(mode)

    def forward(self, x):
        """
        Args:
            x: 3D bone orientations (B, num_bones, 3)
        Returns:
            distance: Distance to valid pose manifold (B, 1)
        """
        B = x.shape[0]
        x = self.enc(x)  # Encode 3D bone features
        dist_pred = self.dfnet(x)  # Predict distance to valid pose manifold
        return dist_pred


class DFNet(nn.Module):
    """Distance Field Network (same as 2D version)"""
    def __init__(self, input_size, hid_layer, weight_norm=True):
        super().__init__()
        output_size = 1
        dims = [input_size] + [d_hidden for d_hidden in hid_layer] + [output_size]

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)
        self.actv = nn.ReLU()
        self.out_actv = nn.ReLU()

    def forward(self, p):
        x = p.reshape(len(p), -1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.actv(x)
        x = self.out_actv(x)
        return x


def keypoint_to_orientations_3d(coords):
    """
    Convert 3D keypoints to 3D bone orientations
    
    Args:
        coords: 3D keypoint coordinates (B, K, 3)
    Returns:
        orientations: 3D bone orientations (B, num_bones, 3)
    """
    pairs = [
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
    K = len(pairs) 
    B = coords.shape[0]
    
    # Get 3D orientations
    orientations = torch.zeros((B, K, 3), device=coords.device)
    for i, pair in enumerate(pairs):
        a, b = pair
        vecs = coords[:, b] - coords[:, a]  # 3D bone vectors
        norms = torch.norm(vecs, dim=1, keepdim=True)
        # Avoid division by zero
        norms = torch.clamp(norms, min=1e-8)
        orientations[:, i] = vecs / norms
    
    return orientations


def get_orientations_3d(hmaps_2d, hmaps_3d):
    """
    Extract 3D orientations from 2D heatmaps and 3D depth maps
    
    Args:
        hmaps_2d: 2D heatmaps (B, K, H, W)
        hmaps_3d: 3D depth maps (B, K, H, W)
    Returns:
        orientations: 3D bone orientations (B, num_bones, 3)
    """
    from lib.datasets.util import softargmax2d
    
    # Get 2D coordinates from heatmaps
    coords_2d = softargmax2d(hmaps_2d)  # (B, K, 2)
    
    # Get depth values at 2D coordinates
    B, K, H, W = hmaps_3d.shape
    coords_2d_flat = coords_2d.view(B * K, 2)
    hmaps_3d_flat = hmaps_3d.view(B * K, H * W)
    
    # Sample depth values
    coords_2d_norm = coords_2d_flat.clone()
    coords_2d_norm[:, 0] = coords_2d_norm[:, 0] / (W - 1) * 2 - 1
    coords_2d_norm[:, 1] = coords_2d_norm[:, 1] / (H - 1) * 2 - 1
    
    # Use grid_sample to get depth values
    coords_2d_norm = coords_2d_norm.view(B, K, 1, 2)
    depths = torch.nn.functional.grid_sample(
        hmaps_3d, coords_2d_norm, mode='bilinear', 
        padding_mode='border', align_corners=True
    ).squeeze(-1).squeeze(-1)  # (B, K)
    
    # Combine 2D coordinates with depth
    coords_3d = torch.cat([coords_2d, depths.unsqueeze(-1)], dim=-1)  # (B, K, 3)
    
    # Convert to 3D orientations
    orientations = keypoint_to_orientations_3d(coords_3d)
    
    return orientations 