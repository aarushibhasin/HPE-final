import torch
import torch.nn as nn
import os


class PriorLoss(nn.Module):
    """Prior loss based on POST paper implementation"""
    def __init__(self, prior_model_path, lambda_p=1e-4):
        super().__init__()
        self.lambda_p = lambda_p
        
        # Import here to avoid circular imports
        from src.priors.prior_2d.models import PoseNDF2D
        
        # Load prior model
        self.prior_model = PoseNDF2D(input_dim=34, hidden_dims=[256, 256, 256, 256])
        if os.path.exists(prior_model_path):
            checkpoint = torch.load(prior_model_path, map_location='cpu')
            # Handle different checkpoint structures
            if 'model_state_dict' in checkpoint:
                self.prior_model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.prior_model.load_state_dict(checkpoint)
            print(f"✅ Loaded prior model from {prior_model_path}")
        else:
            print(f"⚠️ Prior model not found at {prior_model_path}, using untrained model")
        
        self.prior_model.eval()  # Set to evaluation mode
    
    def forward(self, predicted_poses, presence_gt):
        """
        Compute prior loss for valid poses
        Args:
            predicted_poses: (batch, max_poses, num_kpts, 2) - predicted keypoints
            presence_gt: (batch, max_poses) - ground truth presence (binary)
        """
        if self.lambda_p == 0:
            return torch.tensor(0.0, device=predicted_poses.device)
        
        # Get number of people for each sample (presence is binary)
        num_people = torch.sum(presence_gt, dim=1)  # Sum across people dimension
        
        total_prior_loss = 0.0
        valid_samples = 0
        
        for i in range(predicted_poses.shape[0]):
            n_people = int(num_people[i].item())  # Convert to integer for slicing
            if n_people > 0:
                # Get poses for this sample
                poses = predicted_poses[i, :n_people]  # (n_people, num_kpts, 2)
                
                # Check if poses have valid values (not all zeros)
                if torch.any(poses != 0):
                    # Convert to bone vectors (2D keypoints to bone vectors)
                    bone_vectors = self._keypoints_to_bones(poses)  # (n_people, 34)
                    
                    # Compute prior scores
                    prior_scores = self.prior_model(bone_vectors)  # (n_people, 1)
                    
                    # Prior loss: lower scores are better (closer to valid poses)
                    prior_loss = prior_scores.mean()
                    total_prior_loss += prior_loss
                    valid_samples += 1
        
        if valid_samples > 0:
            return self.lambda_p * (total_prior_loss / valid_samples)
        else:
            return torch.tensor(0.0, device=predicted_poses.device)
    
    @staticmethod
    def _keypoints_to_bones(poses):
        """
        Convert 2D keypoints to bone vectors for prior model
        Args:
            poses: (n_people, num_kpts, 2) - 2D keypoints
        Returns:
            bone_vectors: (n_people, 34) - bone features for prior model
        """
        # Define bone connections for 17 joints (COCO format) - need 17 bones for 34 features
        bone_connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # head (4 bones)
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # arms (5 bones)
            (5, 11), (6, 12), (11, 12),  # torso (3 bones)
            (11, 13), (13, 15), (12, 14), (14, 16)  # legs (4 bones)
            # Total: 4 + 5 + 3 + 4 = 16 bones, need 1 more
        ]
        
        # Add one more bone connection to get exactly 17 bones
        bone_connections.append((0, 5))  # nose to left shoulder
        
        batch_size = poses.shape[0]
        num_bones = len(bone_connections)
        bone_vectors = torch.zeros(batch_size, num_bones, 2, device=poses.device)
        
        for i, (parent, child) in enumerate(bone_connections):
            if parent < poses.shape[1] and child < poses.shape[1]:
                bone_vectors[:, i] = poses[:, child] - poses[:, parent]
        
        # Flatten bone vectors to 34 dimensions (17 bones * 2 dims)
        bone_features = bone_vectors.view(batch_size, -1)  # (batch, 34)
        
        return bone_features 