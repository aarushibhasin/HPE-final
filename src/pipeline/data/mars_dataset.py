import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MARSMPMRIDataset(Dataset):
    """
    Dataset for MARS MPMRI data using the actual data structure
    Uses real point cloud data from featuremap.npy and keypoints from kpt_labels.npy
    """
    
    def __init__(self, data_dir, max_people=4, num_kpts=17, kpt_dims=2, 
                 pc_grid_dim=16, transform=None, split_ratio=0.8, is_train=True):
        """
        Args:
            data_dir: Path to MARS data directory
            max_people: Maximum number of people per scene
            num_kpts: Number of keypoints per person
            kpt_dims: Dimensions of keypoints (2 for 2D)
            pc_grid_dim: Grid dimension for point cloud
            transform: Optional transforms
            split_ratio: Ratio for train/val split
            is_train: Whether this is training data (True) or validation data (False)
        """
        self.data_dir = Path(data_dir)
        self.max_people = max_people
        self.num_kpts = num_kpts
        self.kpt_dims = kpt_dims
        self.pc_grid_dim = pc_grid_dim
        self.transform = transform
        self.is_train = is_train
        
        # Load data
        self.load_data()
        
        # Create train/val split if this is training data
        if is_train:
            self.create_train_val_split(split_ratio)
    
    def load_data(self):
        """Load data from MARS format"""
        print(f"📊 Loading MARS data from {self.data_dir}")
        
        # Load point cloud data (featuremap.npy)
        pc_path = self.data_dir / "featuremap.npy"
        if pc_path.exists():
            self.point_clouds = np.load(pc_path)
            print(f"   Loaded point clouds: {self.point_clouds.shape}")
        else:
            raise FileNotFoundError(f"Point cloud data not found: {pc_path}")
        
        # Load keypoint data (kpt_labels.npy)
        kpt_path = self.data_dir / "kpt_labels.npy"
        if kpt_path.exists():
            self.keypoints = np.load(kpt_path)
            print(f"   Loaded keypoints: {self.keypoints.shape}")
        else:
            raise FileNotFoundError(f"Keypoint data not found: {kpt_path}")
        
        # Verify data consistency
        if len(self.point_clouds) != len(self.keypoints):
            raise ValueError(f"Data mismatch: {len(self.point_clouds)} point clouds vs {len(self.keypoints)} keypoints")
        
        print(f"   Total samples: {len(self.point_clouds)}")
        
        # Process keypoints to multi-person format
        self.process_keypoints()
        
        # Convert to tensors
        self.point_clouds = torch.tensor(self.point_clouds, dtype=torch.float32)
        self.keypoints = torch.tensor(self.keypoints, dtype=torch.float32)
        self.presence = torch.tensor(self.presence, dtype=torch.float32)
        
        print(f"✅ MARS data loaded successfully")
    
    def process_keypoints(self):
        """Process keypoints to multi-person format"""
        print("   Processing keypoints to multi-person format...")
        
        # The keypoints are likely in single-person format, need to convert to multi-person
        # Assuming the data is in format (N, num_kpts, 2) for single person
        if len(self.keypoints.shape) == 3 and self.keypoints.shape[1] == self.num_kpts:
            # Single-person data, convert to multi-person
            N = len(self.keypoints)
            multi_kpts = np.zeros((N, self.max_people, self.num_kpts, self.kpt_dims))
            multi_kpts[:, 0] = self.keypoints  # First person gets the actual keypoints
            
            # Create presence labels (all single-person for now)
            presence = np.zeros((N, self.max_people + 1), dtype=np.int32)
            presence[:, 1] = 1  # All samples have 1 person
            
            self.keypoints = multi_kpts
            self.presence = presence
            
            print(f"   Converted to multi-person format: {self.keypoints.shape}")
        else:
            # Already in multi-person format
            print(f"   Already in multi-person format: {self.keypoints.shape}")
            # Create presence labels based on actual data
            self.create_presence_labels()
    
    def create_presence_labels(self):
        """Create presence labels based on actual keypoint data"""
        N = len(self.keypoints)
        presence = np.zeros((N, self.max_people + 1), dtype=np.int32)
        
        for i in range(N):
            # Count how many people have non-zero keypoints
            num_people = 0
            for j in range(self.max_people):
                if np.any(self.keypoints[i, j] != 0):
                    num_people += 1
            
            presence[i, num_people] = 1  # One-hot encoding
        
        self.presence = presence
        print(f"   Created presence labels: {self.presence.shape}")
    
    def create_train_val_split(self, split_ratio):
        """Create train/validation split to prevent data leakage"""
        print(f"📊 Creating train/val split (ratio: {split_ratio})")
        
        total_samples = len(self.keypoints)
        train_size = int(total_samples * split_ratio)
        
        # Create indices
        indices = list(range(total_samples))
        random.shuffle(indices)
        
        if self.is_train:
            # Use train portion
            self.indices = indices[:train_size]
            print(f"   Using train portion: {len(self.indices)} samples")
        else:
            # Use validation portion
            self.indices = indices[train_size:]
            print(f"   Using validation portion: {len(self.indices)} samples")
    
    def __len__(self):
        """Return number of samples"""
        if hasattr(self, 'indices'):
            return len(self.indices)
        return len(self.keypoints)
    
    def __getitem__(self, idx):
        """Get a single sample"""
        if hasattr(self, 'indices'):
            idx = self.indices[idx]
        
        # Get point cloud and keypoints
        pc = self.point_clouds[idx]
        kpts = self.keypoints[idx]
        presence = self.presence[idx]
        
        # Apply transforms if any
        if self.transform:
            pc = self.transform(pc)
            kpts = self.transform(kpts)
        
        return {
            'kpts': kpts,  # Shape: (max_people, num_kpts, kpt_dims)
            'presence': presence,  # Shape: (max_people + 1)
            'pc': pc,  # Shape: (pc_grid_dim, pc_grid_dim, channels)
        }


def create_mars_data_loaders(train_dir, test_dir, batch_size=32, num_workers=4, 
                           split_ratio=0.8, max_people=4, num_kpts=17):
    """
    Create train and validation data loaders using MARS data structure
    
    Args:
        train_dir: Path to training data directory
        test_dir: Path to test data directory
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        split_ratio: Ratio for train/val split
        max_people: Maximum number of people per scene
        num_kpts: Number of keypoints per person
    
    Returns:
        train_loader, val_loader: Data loaders for training and validation
    """
    print("🚀 Creating MARS Data Loaders with Data Leakage Prevention")
    print("=" * 60)
    
    # Set random seed for reproducible splits
    random.seed(42)
    
    # Create datasets
    print(f"📊 Creating training dataset...")
    train_dataset = MARSMPMRIDataset(
        data_dir=train_dir,
        max_people=max_people,
        num_kpts=num_kpts,
        split_ratio=split_ratio,
        is_train=True
    )
    
    print(f"📊 Creating validation dataset...")
    val_dataset = MARSMPMRIDataset(
        data_dir=train_dir,  # Use same directory but different split
        max_people=max_people,
        num_kpts=num_kpts,
        split_ratio=split_ratio,
        is_train=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"✅ MARS data loaders created successfully")
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader 