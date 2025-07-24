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

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prior.models_3d import PoseNDF3D, keypoint_to_orientations_3d, add_3d_noise
from lib.datasets.surreal_3d import SURREAL3D
from lib.transforms.keypoint_detection import *
from lib.logger import CompleteLogger


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
            
            # Combine losses
            valid_loss = criterion(valid_scores, valid_targets)
            noisy_loss = criterion(noisy_scores, noisy_targets)
            
            total_loss = valid_loss + args.noise_weight * noisy_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
            
            if batch_idx % args.print_freq == 0:
                logger.write(f'Epoch [{epoch+1}/{args.epochs}] '
                           f'Batch [{batch_idx}/{len(dataloader)}] '
                           f'Loss: {total_loss.item():.6f} '
                           f'Valid: {valid_loss.item():.6f} '
                           f'Noisy: {noisy_loss.item():.6f}')
        
        # Update learning rate
        scheduler.step()
        
        # Log epoch results
        avg_loss = epoch_loss / num_batches
        logger.write(f'Epoch [{epoch+1}/{args.epochs}] Average Loss: {avg_loss:.6f}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model': prior.module.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': avg_loss,
            }
            logger.save_model(checkpoint, is_best=False, 
                            filename=f'prior_3d_epoch_{epoch+1}.pth.tar')
    
    logger.close()


def validate_prior_3d(args):
    """Validate 3D pose prior network"""
    
    # Set up logging
    logger = CompleteLogger(args.log, 'test')
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    dataset = SURREAL3D(
        root=args.data_root,
        split='test',
        transforms=transform,
        subset_ratio=args.subset_ratio
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    # Load model
    prior = PoseNDF3D().cuda()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    prior.load_state_dict(checkpoint['model'])
    prior = torch.nn.DataParallel(prior)
    prior.eval()
    
    # Validation
    valid_scores = []
    noisy_scores = []
    
    with torch.no_grad():
        for batch_idx, (images, (target_2d, target_3d), target_weight, meta) in enumerate(dataloader):
            # Get 3D joint coordinates
            joints_3d = meta['keypoint3d'].cuda()
            
            # Convert to orientations
            orientations_3d = keypoint_to_orientations_3d(joints_3d)
            
            # Add noise
            noisy_orientations = add_3d_noise(
                orientations_3d.cpu().numpy(),
                rotation_noise=args.rotation_noise,
                translation_noise=args.translation_noise,
                joint_noise=args.joint_noise
            )
            noisy_orientations = torch.from_numpy(noisy_orientations).cuda()
            
            # Get scores
            valid_score = prior(orientations_3d)
            noisy_score = prior(noisy_orientations)
            
            valid_scores.append(valid_score.cpu().numpy())
            noisy_scores.append(noisy_score.cpu().numpy())
            
            if batch_idx % args.print_freq == 0:
                logger.write(f'Batch [{batch_idx}/{len(dataloader)}]')
    
    # Compute statistics
    valid_scores = np.concatenate(valid_scores, axis=0)
    noisy_scores = np.concatenate(noisy_scores, axis=0)
    
    logger.write(f'Valid poses - Mean: {valid_scores.mean():.6f}, Std: {valid_scores.std():.6f}')
    logger.write(f'Noisy poses - Mean: {noisy_scores.mean():.6f}, Std: {noisy_scores.std():.6f}')
    
    # Compute separation
    separation = (noisy_scores.mean() - valid_scores.mean()) / (valid_scores.std() + noisy_scores.std())
    logger.write(f'Separation score: {separation:.6f}')
    
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Pose Prior Training')
    
    # Data arguments
    parser.add_argument('--data-root', default='./data/surreal_3d', help='dataset root')
    parser.add_argument('--subset-ratio', type=float, default=0.1, help='subset ratio')
    parser.add_argument('--image-size', type=int, default=256, help='image size')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr-step', type=int, default=20, help='lr step size')
    parser.add_argument('--lr-gamma', type=float, default=0.5, help='lr decay factor')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='weight decay')
    
    # Noise arguments
    parser.add_argument('--rotation-noise', type=float, default=0.1, help='rotation noise')
    parser.add_argument('--translation-noise', type=float, default=0.1, help='translation noise')
    parser.add_argument('--joint-noise', type=float, default=0.05, help='joint noise')
    parser.add_argument('--noise-penalty', type=float, default=1.0, help='noise penalty')
    parser.add_argument('--noise-weight', type=float, default=1.0, help='noise loss weight')
    
    # Other arguments
    parser.add_argument('--log', default='logs/prior_3d', help='log directory')
    parser.add_argument('--checkpoint', default='', help='checkpoint path for validation')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--print-freq', type=int, default=100, help='print frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    parser.add_argument('--phase', default='train', choices=['train', 'test'], help='train or test')
    
    args = parser.parse_args()
    
    if args.phase == 'train':
        train_prior_3d(args)
    else:
        validate_prior_3d(args) 