import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseNDF2D(nn.Module):
    """2D Neural Distance Field for pose prior"""
    def __init__(self, input_dim=34, hidden_dims=[256, 256, 256, 256]):
        super(PoseNDF2D, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: (batch_size, input_dim) - Bone vectors
        Returns:
            scores: (batch_size, 1) - Distance field scores (lower = more realistic)
        """
        return self.mlp(x) 