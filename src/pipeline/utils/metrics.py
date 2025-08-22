import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def calculate_pck(predicted_poses, gt_poses, presence_gt, threshold=0.05):
    """
    Calculate PCK@threshold for keypoint estimation
    
    Args:
        predicted_poses: (B, 4, 17, 2) - Predicted keypoints
        gt_poses: (B, 4, 17, 2) - Ground truth keypoints
        presence_gt: (B, 4) - Ground truth presence mask
        threshold: Distance threshold (fraction of image size)
    Returns:
        pck: Percentage of correct keypoints
    """
    correct = 0
    total = 0
    
    for i in range(predicted_poses.shape[0]):
        n_people = int(torch.sum(presence_gt[i]).item())
        
        for j in range(n_people):
            # Calculate Euclidean distances
            distances = torch.norm(
                predicted_poses[i, j] - gt_poses[i, j], 
                dim=1
            )  # (17,)
            
            # Count correct keypoints
            correct += torch.sum(distances < threshold).item()
            total += 17
    
    return correct / total if total > 0 else 0.0


def calculate_mpjpe(predicted_poses, gt_poses, presence_gt):
    """
    Calculate MPJPE for keypoint estimation
    
    Args:
        predicted_poses: (B, 4, 17, 2) - Predicted keypoints
        gt_poses: (B, 4, 17, 2) - Ground truth keypoints
        presence_gt: (B, 4) - Ground truth presence mask
    Returns:
        mpjpe: Mean Per Joint Position Error
    """
    total_error = 0
    total_joints = 0
    
    for i in range(predicted_poses.shape[0]):
        n_people = int(torch.sum(presence_gt[i]).item())
        
        for j in range(n_people):
            # Calculate Euclidean distances
            errors = torch.norm(
                predicted_poses[i, j] - gt_poses[i, j], 
                dim=1
            )  # (17,)
            
            total_error += torch.sum(errors).item()
            total_joints += 17
    
    return total_error / total_joints if total_joints > 0 else 0.0


def calculate_presence_metrics(predicted_presence, gt_presence):
    """
    Calculate presence detection metrics
    
    Args:
        predicted_presence: (B, 5) - Predicted presence logits
        gt_presence: (B, 5) - Ground truth one-hot presence
    Returns:
        metrics: Dictionary of classification metrics
    """
    # Convert to class predictions
    pred_classes = torch.argmax(predicted_presence, dim=1)  # (B,)
    gt_classes = torch.argmax(gt_presence, dim=1)          # (B,)
    
    # Calculate accuracy
    accuracy = (pred_classes == gt_classes).float().mean().item()
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        gt_classes.cpu().numpy(), 
        pred_classes.cpu().numpy(), 
        average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def evaluate_model(encoder, pose_estimator, val_loader, device):
    """
    Evaluate model on validation set
    
    Args:
        encoder: Point cloud encoder
        pose_estimator: Pose estimator
        val_loader: Validation data loader
        device: Device to run evaluation on
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    encoder.eval()
    pose_estimator.eval()
    
    all_pred_keypoints = []
    all_gt_keypoints = []
    all_pred_presence = []
    all_gt_presence = []
    all_presence_mask = []
    
    with torch.no_grad():
        for batch in val_loader:
            pc = batch['pc'].to(device)
            kpts = batch['kpts'].to(device)
            presence = batch['presence'].to(device)
            
            # Fix input shape: (batch, height, width, channels) -> (batch, channels, height, width)
            pc = pc.permute(0, 3, 1, 2)
            
            # Forward pass
            pc_features = encoder(pc)
            pred_keypoints, pred_presence = pose_estimator(pc_features)
            
            # Store predictions and ground truth
            all_pred_keypoints.append(pred_keypoints.cpu())
            all_gt_keypoints.append(kpts.cpu())
            all_pred_presence.append(pred_presence.cpu())
            all_gt_presence.append(presence.cpu())
            
            # Create presence mask for keypoint evaluation
            presence_mask = torch.argmax(presence, dim=1)  # (B,)
            mask = torch.zeros(presence.shape[0], 4, device=presence.device)
            for i in range(presence.shape[0]):
                num_people = presence_mask[i].item()
                mask[i, :num_people] = 1
            all_presence_mask.append(mask.cpu())
    
    # Concatenate all batches
    pred_keypoints = torch.cat(all_pred_keypoints, dim=0)
    gt_keypoints = torch.cat(all_gt_keypoints, dim=0)
    pred_presence = torch.cat(all_pred_presence, dim=0)
    gt_presence = torch.cat(all_gt_presence, dim=0)
    presence_mask = torch.cat(all_presence_mask, dim=0)
    
    # Calculate metrics
    pck_005 = calculate_pck(pred_keypoints, gt_keypoints, presence_mask, threshold=0.05)
    pck_002 = calculate_pck(pred_keypoints, gt_keypoints, presence_mask, threshold=0.02)
    mpjpe = calculate_mpjpe(pred_keypoints, gt_keypoints, presence_mask)
    presence_metrics = calculate_presence_metrics(pred_presence, gt_presence)
    
    return {
        'PCK@0.05': pck_005,
        'PCK@0.02': pck_002,
        'MPJPE': mpjpe,
        'presence_accuracy': presence_metrics['accuracy'],
        'presence_precision': presence_metrics['precision'],
        'presence_recall': presence_metrics['recall'],
        'presence_f1': presence_metrics['f1_score']
    } 