import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for presence detection."""
    def __init__(self, alpha=1.0, gamma=1.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ImprovedPoseEstimator(nn.Module):
    """Improved pose estimator with residual connections and separate heads"""
    def __init__(
        self,
        max_poses: int = 4,
        num_kpts: int = 17,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.max_poses = max_poses
        self.num_kpts = num_kpts
        self.hidden_dim = hidden_dim
        
        # Input projection layer
        self.input_projection = nn.Linear(512, hidden_dim)
        
        # Residual layers with batch normalization
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])
        
        # Separate specialized heads
        self.keypoint_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_poses * num_kpts * 2)  # 4×17×2 = 136
        )
        
        self.presence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_poses + 1)  # 5 classes (0-4 people)
        )
        
        # Loss functions
        self.kpts_loss_fn = nn.SmoothL1Loss(reduction='none')
        self.focal_loss_fn = FocalLoss(alpha=1.0, gamma=1.0)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input projection
        x = self.input_projection(x)  # (B, 512) → (B, 128)
        residual = x
        
        # Residual layers
        for layer in self.layers:
            x = layer(x) + residual  # Residual connection
            residual = x
        
        # Separate head predictions
        keypoints = self.keypoint_head(x)  # (B, 136)
        presence = self.presence_head(x)    # (B, 5)
        
        # Reshape keypoints to multi-person format
        keypoints = keypoints.view(-1, self.max_poses, self.num_kpts, 2)  # (B, 4, 17, 2)
        
        return keypoints, presence
    
    def compute_hpe_loss(self, prediction, target_kpts, target_presence=None):
        """Compute human pose estimation loss."""
        batch_size = prediction.shape[0]
        
        # Extract keypoints and presence from prediction
        keypoints = prediction[:, :self.max_poses * self.num_kpts * 2].view(batch_size, self.max_poses, self.num_kpts, 2)
        presence_logits = prediction[:, self.max_poses * self.num_kpts * 2:]
        
        # Keypoint loss
        kpt_loss = self._compute_keypoint_loss(keypoints, target_kpts, target_presence)
        
        # Presence loss
        if target_presence is not None:
            # Convert binary presence to one-hot for focal loss
            batch_size = target_presence.shape[0]
            num_people_per_sample = torch.sum(target_presence, dim=1)  # (batch,)
            
            # Create one-hot encoding: [0, 1, 2, 3, 4] people
            one_hot_presence = torch.zeros(batch_size, self.max_poses + 1, device=target_presence.device)
            for i in range(batch_size):
                num_people = int(min(num_people_per_sample[i].item(), self.max_poses))
                one_hot_presence[i, num_people] = 1.0
            
            presence_loss = self.focal_loss_fn(presence_logits, torch.argmax(one_hot_presence, dim=1))
        else:
            presence_loss = torch.tensor(0.0, device=prediction.device)
        
        return kpt_loss, presence_loss
    
    def _compute_keypoint_loss(self, predicted_kpts, target_kpts, target_presence):
        """Compute keypoint loss only for valid poses."""
        if target_presence is None:
            return torch.tensor(0.0, device=predicted_kpts.device)
        
        # Get number of people for each sample
        num_people = torch.sum(target_presence, dim=1)  # Sum across people dimension
        
        total_loss = 0.0
        valid_samples = 0
        
        for i in range(predicted_kpts.shape[0]):
            n_people = int(num_people[i].item())  # Convert to integer for slicing
            if n_people > 0:  # Skip samples with no people
                # Only compute loss for valid poses
                pred_kpts = predicted_kpts[i, :n_people]  # (n_people, num_kpts, 2)
                target_kpts_sample = target_kpts[i, :n_people]  # (n_people, num_kpts, 2)
                
                # Compute loss for each person
                for j in range(n_people):
                    # Check if this person has valid keypoints (not all zeros)
                    if torch.any(target_kpts_sample[j] != 0):
                        loss = self.kpts_loss_fn(pred_kpts[j], target_kpts_sample[j])
                        total_loss += loss.mean()
                
                valid_samples += 1
        
        if valid_samples > 0:
            return total_loss / valid_samples
        else:
            return torch.tensor(0.0, device=predicted_kpts.device) 