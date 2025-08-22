#!/usr/bin/env python3
"""
Improved Training Script with Prior Loss Integration
Based on POST paper: Prior-guided Source-free Domain Adaptation for Human Pose Estimation
"""

# Fix OpenMP runtime conflict
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import random
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

# Import our modules
from pipeline.models.encoder_pc import poseidonPcEncoder
from pipeline.models.pose_estimator import ImprovedPoseEstimator
from pipeline.models.prior_loss import PriorLoss
from pipeline.data.mars_dataset import create_mars_data_loaders
from pipeline.utils.metrics import evaluate_model

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

def train_improved_model():
    """Train the improved model with prior loss integration."""
    print("🚀 Starting Improved Training with Prior Loss...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    # Set random seed
    set_random_seed(42)
    
    # Data paths
    train_dir = "thesis_project-main/MARS_SRCmpmri_MAXPPL4_GRID8_NORMED/src/train"
    test_dir = "thesis_project-main/MARS_SRCmpmri_MAXPPL4_GRID8_NORMED/src/test"
    
    # Check if data directories exist
    if not Path(train_dir).exists():
        print(f"❌ Training data directory not found: {train_dir}")
        return None, None
    
    if not Path(test_dir).exists():
        print(f"❌ Test data directory not found: {test_dir}")
        return None, None
    
    # Create data loaders with data leakage prevention
    print("\n📊 Creating Data Loaders...")
    train_loader, val_loader = create_mars_data_loaders(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=64,  # Increased batch size for speed
        num_workers=2,  # Reduced workers for speed
        split_ratio=0.8,
        max_people=4,
        num_kpts=17
    )
    
    # Create models
    print("\n🏗️ Creating Models...")
    encoder = poseidonPcEncoder(
        model_architecture="mpMARS",
        representation_embedding_dim=512,
        pretrained=False,
        grid_dims=(16, 16)  # Changed from (8, 8) to match MARS data
    )
    
    pose_estimator = ImprovedPoseEstimator(
        max_poses=4,
        num_kpts=17,
        hidden_dim=128,  # Reduced complexity
        num_layers=2,    # Reduced layers
        dropout=0.3      # Increased dropout
    )
    
    # Create prior loss
    prior_loss = PriorLoss(
        prior_model_path="checkpoints/prior_2d_final.pth",  # Use 2D prior model
        lambda_p=1e-4  # Use the correct weight from training_params
    )
    
    # Move to device
    encoder.to(device)
    pose_estimator.to(device)
    prior_loss.to(device)
    
    # Training parameters
    training_params = {
        "num_epochs": 50,  # Reduced from 100 to 50
        "learning_rate": 8e-5,  # Slightly increased for faster convergence
        "weight_decay": 1e-4,   # Reduced weight decay
        "presence_loss_weight": 0.1,  # Reduced presence weight
        "prior_loss_weight": 1e-4,    # Increased prior loss weight significantly
        "early_stopping_patience": 15,  # Reduced from 25 to 15
        "scheduler_patience": 10,
        "scheduler_factor": 0.7,
        "gradient_clip": 1.0
    }
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(
        list(encoder.parameters()) + list(pose_estimator.parameters()),
        lr=training_params["learning_rate"],
        weight_decay=training_params["weight_decay"]
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=training_params["scheduler_factor"],
        patience=training_params["scheduler_patience"],
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    training_history = []
    
    print(f"\n🎯 Training Parameters:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    print(f"\n🚀 Starting Training Loop...")
    for epoch in range(training_params["num_epochs"]):
        start_time = time.time()
        
        # Training phase
        encoder.train()
        pose_estimator.train()
        train_losses = []
        train_kpt_losses = []
        train_presence_losses = []
        train_prior_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Move data to device
            pc = batch['pc'].to(device)
            kpts = batch['kpts'].to(device)
            presence = batch['presence'].to(device)
            
            # Fix input shape: (batch, height, width, channels) -> (batch, channels, height, width)
            pc = pc.permute(0, 3, 1, 2)
            
            # Forward pass
            # Encode point clouds
            pc_features = encoder(pc)
            # Get pose predictions
            pred_keypoints, pred_presence = pose_estimator(pc_features)
            
            # Create presence mask for loss computation
            presence_mask = torch.argmax(presence, dim=1)  # (B,)
            mask = torch.zeros(presence.shape[0], 4, device=presence.device)
            for i in range(presence.shape[0]):
                num_people = presence_mask[i].item()
                mask[i, :num_people] = 1
            
            # Compute losses
            kpt_loss, presence_loss = pose_estimator.compute_hpe_loss(
                torch.cat([pred_keypoints.view(pred_keypoints.shape[0], -1), pred_presence], dim=1),
                kpts, 
                mask
            )
            prior_loss_val = prior_loss(pred_keypoints, mask)
            
            # Total loss
            total_loss = (
                kpt_loss + 
                training_params["presence_loss_weight"] * presence_loss +
                prior_loss_val  # PriorLoss already applies lambda_p
            )
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(pose_estimator.parameters()), 
                training_params["gradient_clip"]
            )
            
            optimizer.step()
            
            # Record losses
            train_losses.append(total_loss.item())
            train_kpt_losses.append(kpt_loss.item())
            train_presence_losses.append(presence_loss.item())
            train_prior_losses.append(prior_loss_val.item())
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch+1:3d}/{training_params['num_epochs']} | "
                      f"Batch {batch_idx:4d}/{len(train_loader)} | "
                      f"Loss: {total_loss.item():.4f} | "
                      f"Kpt: {kpt_loss.item():.4f} | "
                      f"Pres: {presence_loss.item():.4f} | "
                      f"Prior: {prior_loss_val.item():.6f}")
        
        # Validation phase
        encoder.eval()
        pose_estimator.eval()
        val_metrics = evaluate_model(encoder, pose_estimator, val_loader, device)
        avg_val_loss = val_metrics.get('total_loss', 0.0)  # We'll compute this properly
        
        # Calculate training averages
        avg_train_loss = np.mean(train_losses)
        avg_train_kpt = np.mean(train_kpt_losses)
        avg_train_pres = np.mean(train_presence_losses)
        avg_train_prior = np.mean(train_prior_losses)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'pose_estimator_state_dict': pose_estimator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'training_params': training_params
            }, 'checkpoints/best_improved_model.pth')
            print(f"💾 Saved best model (epoch {epoch+1})")
        else:
            patience_counter += 1
        
        # Record history
        epoch_time = time.time() - start_time
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_kpt_loss': avg_train_kpt,
            'train_presence_loss': avg_train_pres,
            'train_prior_loss': avg_train_prior,
            'val_loss': avg_val_loss,
            'val_pck_005': val_metrics.get('PCK@0.05', 0.0),
            'val_pck_002': val_metrics.get('PCK@0.02', 0.0),
            'val_mpjpe': val_metrics.get('MPJPE', 0.0),
            'val_presence_accuracy': val_metrics.get('presence_accuracy', 0.0),
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time
        })
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1:3d}/{training_params['num_epochs']} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} (Kpt: {avg_train_kpt:.4f}, Pres: {avg_train_pres:.4f}, Prior: {avg_train_prior:.6f})")
        print(f"   Val Loss: {avg_val_loss:.4f}")
        print(f"   Val PCK@0.05: {val_metrics.get('PCK@0.05', 0.0):.4f}")
        print(f"   Val MPJPE: {val_metrics.get('MPJPE', 0.0):.4f}")
        print(f"   Val Presence Acc: {val_metrics.get('presence_accuracy', 0.0):.4f}")
        print(f"   LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {epoch_time:.1f}s")
        print(f"   Best Val Loss: {best_val_loss:.4f} | Patience: {patience_counter}/{training_params['early_stopping_patience']}")
        
        # Early stopping check
        if patience_counter >= training_params["early_stopping_patience"]:
            print(f"\n⏹️ Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final model and history
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'pose_estimator_state_dict': pose_estimator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_val_loss': avg_val_loss,
        'training_params': training_params
    }, 'checkpoints/final_improved_model.pth')
    
    # Save training history
    history_df = pd.DataFrame(training_history)
    history_df.to_csv('results/improved_training_history.csv', index=False)
    
    print(f"\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Training history saved to: results/improved_training_history.csv")
    
    return encoder, pose_estimator, training_history

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Train the improved model
    encoder, pose_estimator, history = train_improved_model() 