import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import math
from train_prior_stable import PoseNDF3D, StablePoseDataset, create_dataloaders, train_model_stable

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    pattern = re.compile(r"prior_3d_epoch_(\d+)_stable.pth")
    checkpoints = []
    for fname in os.listdir(checkpoint_dir):
        match = pattern.match(fname)
        if match:
            epoch = int(match.group(1))
            checkpoints.append((epoch, fname))
    if not checkpoints:
        return None, 0
    checkpoints.sort()
    latest_epoch, latest_fname = checkpoints[-1]
    return os.path.join(checkpoint_dir, latest_fname), latest_epoch

def resume_training():
    data_dir = "extracted_3d_poses"
    batch_size = 16
    num_epochs = 50
    max_samples = None
    checkpoint_dir = "checkpoints"

    print("Searching for latest checkpoint...")
    latest_ckpt, last_epoch = find_latest_checkpoint(checkpoint_dir)
    if not latest_ckpt:
        print("No checkpoint found. Please start training from scratch.")
        return
    print(f"Found checkpoint: {latest_ckpt} (epoch {last_epoch})")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Prepare data
    train_loader, val_loader, dataset = create_dataloaders(
        data_dir, batch_size=batch_size, max_samples=max_samples
    )

    # Model
    input_dim = 72
    model = PoseNDF3D(input_dim=input_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Load checkpoint
    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_losses = checkpoint.get('train_losses', [])
    val_losses = checkpoint.get('val_losses', [])
    learning_rates = checkpoint.get('learning_rates', [])

    print(f"Resuming from epoch {last_epoch+1} to {num_epochs}")
    remaining_epochs = num_epochs - last_epoch
    if remaining_epochs <= 0:
        print("Training already completed.")
        return

    # Continue training
    new_train_losses, new_val_losses, new_learning_rates = train_model_stable(
        model, train_loader, val_loader, num_epochs=remaining_epochs, device=device
    )

    # Combine histories
    train_losses += new_train_losses
    val_losses += new_val_losses
    learning_rates += new_learning_rates

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "prior_3d_final_stable.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates
    }, final_model_path)
    print(f"Saved final model: {final_model_path}")

    print("Training resumed and completed!")

if __name__ == "__main__":
    resume_training() 