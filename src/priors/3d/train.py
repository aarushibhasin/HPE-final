"""
3D Prior Training Script
Trains the 3D pose prior network using SURREAL dataset
"""
import os
import sys
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

# Add src to path for imports
sys.path.append('src')

from priors.3d.models import PoseNDF3D, keypoint_to_orientations_3d
from lib.datasets.surreal_3d import SURREAL3D
from lib.transforms.keypoint_detection import *
from lib.logger import CompleteLogger


def add_3d_noise(joints_3d, rotation_noise=0.1, translation_noise=0.1, joint_noise=0.05):
    """
    Add 3D noise to joint coordinates for prior training
    
    Args:
        joints_3d: 3D joint coordinates (N, K, 3)
        rotation_noise: Rotation noise magnitude
        translation_noise: Translation noise magnitude
        joint_noise: Per-joint noise magnitude
    Returns:
        noisy_joints: Noisy 3D joint coordinates
    """
    from scipy.spatial.transform import Rotation
    
    N, K, _ = joints_3d.shape
    noisy_joints = joints_3d.copy()
    
    for i in range(N):
        # Add random rotation
        if rotation_noise > 0:
            rot = Rotation.from_rotvec(np.random.uniform(-rotation_noise, rotation_noise, 3))
            noisy_joints[i] = rot.apply(noisy_joints[i])
        
        # Add random translation
        if translation_noise > 0:
            translation = np.random.uniform(-translation_noise, translation_noise, 3)
            noisy_joints[i] += translation
        
        # Add per-joint noise
        if joint_noise > 0:
            joint_noise_vals = np.random.uniform(-joint_noise, joint_noise, (K, 3))
            noisy_joints[i] += joint_noise_vals
    
    return noisy_joints


def train_prior_3d(args):
    """Train 3D pose prior network"""
    
    # Set up logging
    logger = CompleteLogger(args.log, 'train')
    logger.write(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    
    # Set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = SURREAL3D(
        root=args.data_root,
        split='train',
        transforms=transform,
        subset_ratio=args.subset_ratio
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    logger.write(f"Dataset size: {len(dataset)}")
    
    # Create model
    prior = PoseNDF3D().cuda()
    prior = torch.nn.DataParallel(prior)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(prior.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # Training loop
    prior.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, (target_2d, target_3d), target_weight, meta) in enumerate(dataloader):
            # Get 3D joint coordinates from meta
            joints_3d = meta['keypoint3d'].cuda()  # (B, K, 3)
            
            # Convert to 3D orientations
            orientations_3d = keypoint_to_orientations_3d(joints_3d)  # (B, num_bones, 3)
            
            # Add noise for training
            noisy_orientations = add_3d_noise(
                orientations_3d.cpu().numpy(),
                rotation_noise=args.rotation_noise,
                translation_noise=args.translation_noise,
                joint_noise=args.joint_noise
            )
            noisy_orientations = torch.from_numpy(noisy_orientations).cuda()
            
            # Forward pass
            optimizer.zero_grad()
            
            # Valid poses should have low distance scores
            valid_scores = prior(orientations_3d)
            valid_targets = torch.zeros_like(valid_scores)
            
            # Noisy poses should have higher distance scores
            noisy_scores = prior(noisy_orientations)
            noisy_targets = torch.ones_like(noisy_scores) * args.noise_penalty
            
            # Compute loss
            valid_loss = criterion(valid_scores, valid_targets)
            noisy_loss = criterion(noisy_scores, noisy_targets)
            total_loss = valid_loss + args.noise_weight * noisy_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            # Log progress
            if batch_idx % args.log_interval == 0:
                logger.write(f'Epoch: {epoch}, Batch: {batch_idx}, '
                           f'Loss: {total_loss.item():.4f}, '
                           f'Valid Loss: {valid_loss.item():.4f}, '
                           f'Noisy Loss: {noisy_loss.item():.4f}')
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch summary
        avg_loss = epoch_loss / num_batches
        logger.write(f'Epoch {epoch} completed. Average loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': prior.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            torch.save(checkpoint, os.path.join(args.log, f'prior_3d_epoch_{epoch}.pth'))
    
    # Save final model
    torch.save(prior.module.state_dict(), os.path.join(args.log, 'prior_3d_final.pth'))
    logger.write('Training completed!')


def main():
    parser = argparse.ArgumentParser(description='Train 3D Pose Prior')
    
    # Data arguments
    parser.add_argument('--data-root', type=str, required=True, help='Path to SURREAL dataset')
    parser.add_argument('--subset-ratio', type=float, default=1.0, help='Subset ratio for training')
    
    # Model arguments
    parser.add_argument('--image-size', type=int, default=256, help='Input image size')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--lr-step', type=int, default=30, help='LR step size')
    parser.add_argument('--lr-gamma', type=float, default=0.1, help='LR gamma')
    
    # Noise arguments
    parser.add_argument('--rotation-noise', type=float, default=0.1, help='Rotation noise magnitude')
    parser.add_argument('--translation-noise', type=float, default=0.1, help='Translation noise magnitude')
    parser.add_argument('--joint-noise', type=float, default=0.05, help='Per-joint noise magnitude')
    parser.add_argument('--noise-weight', type=float, default=1.0, help='Weight for noisy pose loss')
    parser.add_argument('--noise-penalty', type=float, default=0.5, help='Penalty for noisy poses')
    
    # Logging arguments
    parser.add_argument('--log', type=str, default='logs/prior_3d', help='Log directory')
    parser.add_argument('--log-interval', type=int, default=100, help='Log interval')
    parser.add_argument('--save-interval', type=int, default=10, help='Save interval')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create log directory
    os.makedirs(args.log, exist_ok=True)
    
    # Train model
    train_prior_3d(args)


if __name__ == '__main__':
    main() 