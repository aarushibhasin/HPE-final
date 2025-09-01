#!/usr/bin/env python3
"""
Evaluation Script for Integrated Pipeline
Evaluates the trained model on validation/test data
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append('src')

# Import our modules
from pipeline.models.encoder_pc import poseidonPcEncoder
from pipeline.models.pose_estimator import ImprovedPoseEstimator
from pipeline.data import create_mars_data_loaders
from pipeline.utils.metrics import evaluate_model

def load_trained_model(checkpoint_path):
    """Load trained model from checkpoint"""
    print(f"📂 Loading model from {checkpoint_path}")
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None, None
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create models
    encoder = poseidonPcEncoder(
        model_architecture="mpMARS",
        representation_embedding_dim=512,
        pretrained=False,
        grid_dims=(16, 16)
    )
    
    pose_estimator = ImprovedPoseEstimator(
        max_poses=4,
        num_kpts=17,
        hidden_dim=128,
        num_layers=2,
        dropout=0.3
    )
    
    # Load encoder
    encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
    print(f"✅ Loaded encoder from {checkpoint_path}")
    
    # Load pose estimator
    pose_estimator.load_state_dict(checkpoint['pose_estimator_state_dict'], strict=False)
    print(f"✅ Loaded pose estimator from {checkpoint_path}")
    
    return encoder, pose_estimator

def evaluate_trained_model(checkpoint_path="checkpoints/best_improved_model.pth"):
    """Evaluate the trained model"""
    print("🚀 Starting Model Evaluation...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    # Load model
    encoder, pose_estimator = load_trained_model(checkpoint_path)
    if encoder is None or pose_estimator is None:
        return None
    
    # Move to device
    encoder.to(device)
    pose_estimator.to(device)
    
    # Set to evaluation mode
    encoder.eval()
    pose_estimator.eval()
    
    # Data paths
    train_dir = "thesis_project-main/MARS_SRCmpmri_MAXPPL4_GRID8_NORMED/src/train"
    test_dir = "thesis_project-main/MARS_SRCmpmri_MAXPPL4_GRID8_NORMED/src/val"  # Use val instead of test
    
    # Check if data directories exist
    if not Path(train_dir).exists():
        print(f"❌ Training data directory not found: {train_dir}")
        return None
    
    if not Path(test_dir).exists():
        print(f"❌ Validation data directory not found: {test_dir}")
        return None
    
    # Create data loaders
    print("\n📊 Creating Data Loaders...")
    train_loader, val_loader = create_mars_data_loaders(
        train_dir=train_dir,
        test_dir=test_dir,
        batch_size=64,
        num_workers=2,
        split_ratio=0.8,
        max_people=4,
        num_kpts=17
    )
    
    # Evaluate on validation set
    print("\n🔍 Evaluating on Validation Set...")
    val_metrics = evaluate_model(encoder, pose_estimator, val_loader, device)
    
    # Print results
    print("\n📊 Validation Results:")
    print(f"   PCK@0.05: {val_metrics['PCK@0.05']:.4f}")
    print(f"   PCK@0.02: {val_metrics['PCK@0.02']:.4f}")
    print(f"   MPJPE: {val_metrics['MPJPE']:.4f}")
    print(f"   Presence Accuracy: {val_metrics['presence_accuracy']:.4f}")
    print(f"   Presence Precision: {val_metrics['presence_precision']:.4f}")
    print(f"   Presence Recall: {val_metrics['presence_recall']:.4f}")
    print(f"   Presence F1-Score: {val_metrics['presence_f1']:.4f}")
    
    # Save results
    results = {
        'checkpoint_path': checkpoint_path,
        'device': str(device),
        'validation_metrics': val_metrics
    }
    
    # Convert numpy types to Python types for JSON serialization
    for key, value in val_metrics.items():
        if isinstance(value, np.floating):
            val_metrics[key] = float(value)
        elif isinstance(value, np.integer):
            val_metrics[key] = int(value)
    
    # Save to file
    results_file = "results/evaluation_results.json"
    os.makedirs("results", exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    return val_metrics

def inference_single_sample(encoder, pose_estimator, point_cloud, device):
    """Inference on single point cloud sample"""
    encoder.eval()
    pose_estimator.eval()
    
    with torch.no_grad():
        # Preprocess input
        pc = point_cloud.unsqueeze(0)  # Add batch dimension
        pc = pc.permute(0, 3, 1, 2)  # (1, 5, 16, 16)
        pc = pc.to(device)
        
        # Forward pass
        features = encoder(pc)
        keypoints, presence = pose_estimator(features)
        
        # Post-process outputs
        keypoints = keypoints.squeeze(0)  # (4, 17, 2)
        presence_probs = torch.softmax(presence, dim=1).squeeze(0)  # (5,)
        
        # Get number of people
        num_people = torch.argmax(presence_probs).item()
        
        return keypoints, presence_probs, num_people

if __name__ == "__main__":
    # Evaluate the model
    metrics = evaluate_trained_model()
    
    if metrics is not None:
        print("\n✅ Evaluation completed successfully!")
    else:
        print("\n❌ Evaluation failed!")
