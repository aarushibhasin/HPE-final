import torch
import torch.nn as nn
import torch.nn.functional as F


class mpMarsBackbone(nn.Module):
    """mpMARS backbone for point cloud feature extraction"""
    def __init__(
        self, in_chs=5, out_chs=32, representation_embedding_dim=512, grid_dims=(16, 16)
    ):
        super(mpMarsBackbone, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.grid_width = grid_dims[0]
        self.grid_height = grid_dims[1]
        self.representation_embedding_dim = representation_embedding_dim
        
        # Convolutional layers
        self.conv_one_1 = nn.Sequential(
            nn.Conv2d(self.in_chs, self.out_chs // 2, kernel_size=(3, 3), 
                     stride=(1, 1), padding="same"),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.conv_one_2 = nn.Sequential(
            nn.Conv2d(self.out_chs // 2, self.out_chs, kernel_size=(3, 3), 
                     stride=(1, 1), padding="same"),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Global average pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.final_projection = nn.Linear(self.out_chs, self.representation_embedding_dim)

    def forward(self, x):
        # Convolutional feature extraction
        x = self.conv_one_1(x)
        x = self.conv_one_2(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Final projection
        x = self.final_projection(x)
        return x


class poseidonPcEncoder(nn.Module):
    """Point cloud encoder for MARS data"""
    def __init__(
        self,
        model_architecture: str = "mpMARS",
        representation_embedding_dim: int = 512,
        pretrained: bool = False,
        grid_dims: tuple = (16, 16),
    ):
        super(poseidonPcEncoder, self).__init__()
        
        self.model_architecture = model_architecture
        self.representation_embedding_dim = representation_embedding_dim
        self.pretrained = pretrained
        self.encoder = None
        self.grid_dims = grid_dims
        
        # Initialize the encoder model
        self.initialize_model()

    def initialize_model(self):
        if self.model_architecture == "mpMARS":
            mpmars_backbone = mpMarsBackbone(
                representation_embedding_dim=self.representation_embedding_dim,
                grid_dims=self.grid_dims,
            )
            self.encoder = mpmars_backbone
        else:
            raise ValueError(f"Unsupported model architecture: {self.model_architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the point cloud encoder
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Encoded features of shape (batch_size, representation_embedding_dim)
        """
        if self.encoder is None:
            raise ValueError("Encoder model has not been initialized.")
        return self.encoder(x) 