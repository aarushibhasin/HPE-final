#!/usr/bin/env python3
"""
Train 2D Prior Model
Based on POST paper implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import sys
from pathlib import Path
import random
import time

# Add src to path for imports
sys.path.append('src')

from priors.prior_2d.models import PoseNDF2D

def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_prior_data(data_path):
    """Load prior training data"""
    print(f"📊 Loading prior data from {data_path}")
    
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return None, None
    
    # Load data
    data = np.load(data_path)
    
    # Extract valid and invalid poses
    valid_poses = data['valid_poses']  # (N, 34)
    invalid_poses = data['invalid_poses']  # (M, 34)
    
    print(f"   Valid poses: {valid_poses.shape}")
    print(f"   Invalid poses: {invalid_poses.shape}")
    
    return valid_poses, invalid_poses

def create_prior_dataset(valid_poses, invalid_poses, valid_ratio=0.7):
    """Create dataset for prior training"""
    print("📊 Creating prior dataset...")
    
    # Combine valid and invalid poses
    all_poses = np.concatenate([valid_poses, invalid_poses], axis=0)
    
    # Create labels: 0 for valid poses, 1 for invalid poses
    valid_labels = np.zeros(len(valid_poses))
    invalid_labels = np.ones(len(invalid_poses))
    all_labels = np.concatenate([valid_labels, invalid_labels])
    
    # Shuffle data
    indices = np.random.permutation(len(all_poses))
    all_poses = all_poses[indices]
    all_labels = all_labels[indices]
    
    # Split into train and validation
    split_idx = int(len(all_poses) * valid_ratio)
    train_poses = all_poses[:split_idx]
    train_labels = all_labels[:split_idx]
    val_poses = all_poses[split_idx:]
    val_labels = all_labels[split_idx:]
    
    print(f"   Train samples: {len(train_poses)}")
    print(f"   Validation samples: {len(val_poses)}")
    
    return train_poses, train_labels, val_poses, val_labels

def train_prior_model():
    """Train the 2D prior model"""
    print("🚀 Starting 2D Prior Model Training...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    # Set random seed
    set_random_seed(42)
    
    # Data path
    data_path = "extracted_mpmri_2d_poses.npz"
    
    # Load data
    valid_poses, invalid_poses = load_prior_data(data_path)
    if valid_poses is None:
        return None
    
    # Create dataset
    train_poses, train_labels, val_poses, val_labels = create_prior_dataset(
        valid_poses, invalid_poses
    )
    
    # Convert to tensors
    train_poses = torch.tensor(train_poses, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.float32)
    val_poses = torch.tensor(val_poses, dtype=torch.float32)
    val_labels = torch.tensor(val_labels, dtype=torch.float32)
    
    # Create data loaders
    train_dataset = TensorDataset(train_poses, train_labels)
    val_dataset = TensorDataset(val_poses, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # Create model
    model = PoseNDF2D(input_dim=34, hidden_dims=[256, 256, 256, 256])
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)
    
    # Training parameters
    num_epochs = 100
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15
    
    print(f"\n🎯 Training Parameters:")
    print(f"   Epochs: {num_epochs}")
    print(f"   Learning rate: {optimizer.param_groups[0]['lr']}")
    print(f"   Batch size: 64")
    print(f"   Patience: {patience}")
    
    print(f"\n🚀 Starting Training Loop...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_losses = []
        
        for batch_idx, (poses, labels) in enumerate(train_loader):
            poses = poses.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(poses).squeeze()
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                      f"Batch {batch_idx:4d}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f}")
        
        # Validation phase
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for poses, labels in val_loader:
                poses = poses.to(device)
                labels = labels.to(device)
                
                outputs = model(poses).squeeze()
                loss = criterion(outputs, labels)
                val_losses.append(loss.item())
        
        # Calculate averages
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'checkpoints/prior_2d_final.pth')
            print(f"💾 Saved best model (epoch {epoch+1})")
        else:
            patience_counter += 1
        
        # Print epoch summary
        epoch_time = time.time() - start_time
        print(f"\n📊 Epoch {epoch+1:3d}/{num_epochs} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {epoch_time:.1f}s")
        print(f"   Best Val Loss: {best_val_loss:.4f} | Patience: {patience_counter}/{patience}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\n✅ Prior training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return model

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    
    # Train the prior model
    model = train_prior_model() 