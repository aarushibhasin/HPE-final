import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import math
from sklearn.preprocessing import StandardScaler
import pickle
import json
from sklearn.metrics import roc_auc_score, accuracy_score


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
DATA_SUMMARY_PATH = os.path.join(RESULTS_DIR, "data_summary.json")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class PoseNDF3D(nn.Module):
    """3D Pose Neural Distance Field with enhanced stability"""
    def __init__(self, input_dim=72, hidden_dims=[256, 256, 256, 256], output_dim=1):
        super(PoseNDF3D, self).__init__()
        
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
        
        # Final layer without activation for distance field
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights properly
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with Xavier/Glorot initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # Add input validation
        if torch.isnan(x).any():
            raise ValueError("NaN detected in input")
        
        output = self.network(x)
        
        # Add output validation
        if torch.isnan(output).any():
            raise ValueError("NaN detected in network output")
        
        return output

class CorrectedPoseDataset(Dataset):
    """Dataset with proper distance field targets"""
    def __init__(self, data_dir, split='train', max_samples=None):
        self.data_dir = data_dir
        self.split = split
        self.max_samples = max_samples
        
        # Find all .npz files
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        
        if max_samples:
            self.files = self.files[:max_samples]
        
        print(f"Found {len(self.files)} files for {split} split")
        
        # Validate and load data
        self.valid_data = []
        self._validate_and_load_data()
        
        print(f"Loaded {len(self.valid_data)} valid samples")
        
        # Compute statistics for normalization
        if self.valid_data:
            self._compute_statistics()
    
    def _validate_and_load_data(self):
        """Validate and load data with comprehensive checks"""
        # Only run validation if summary file does not exist
        if os.path.exists(DATA_SUMMARY_PATH):
            with open(DATA_SUMMARY_PATH, "r") as f:
                summary = json.load(f)
            self.valid_data = [np.array(x) for x in summary["valid_data"]]
            print(f"Loaded {len(self.valid_data)} valid samples from summary.")
            return
        for file_path in tqdm(self.files, desc="Validating data"):
            try:
                data = np.load(file_path)
                
                if 'joints3d' not in data:
                    continue
                
                joints3d = data['joints3d']
                
                # Comprehensive validation
                if self._is_valid_pose(joints3d):
                    # Take a random frame from each sequence
                    num_frames = joints3d.shape[0]
                    random_frame_idx = np.random.randint(0, num_frames)
                    self.valid_data.append(joints3d[random_frame_idx])
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        # After loading, save summary
        with open(DATA_SUMMARY_PATH, "w") as f:
            json.dump({"valid_data": [x.tolist() for x in self.valid_data]}, f)
    
    def _is_valid_pose(self, joints3d):
        """Check if pose data is valid"""
        try:
            # Check for NaN or infinite values
            if np.isnan(joints3d).any() or np.isinf(joints3d).any():
                return False
            
            # Check for extreme values
            if np.any(np.abs(joints3d) > 1000.0):
                return False
            
            # Check for reasonable joint positions (not all zeros)
            if np.allclose(joints3d, 0, atol=1e-6):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _compute_statistics(self):
        """Compute statistics for normalization"""
        all_data = np.vstack(self.valid_data)
        self.mean = np.mean(all_data, axis=0)
        self.std = np.std(all_data, axis=0)
        
        # Handle zero standard deviations
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        
        print(f"Data statistics - Mean: {self.mean.mean():.4f}, Std: {self.std.mean():.4f}")
    
    def normalize_data(self, data):
        """Normalize data using computed statistics"""
        return (data - self.mean) / self.std
    
    def create_invalid_pose(self, valid_pose):
        """Create an invalid pose by adding noise or perturbations"""
        # Method 1: Add significant noise
        noise = np.random.normal(0, 2.0, valid_pose.shape)
        invalid_pose = valid_pose + noise
        
        # Method 2: Randomly swap joint positions
        if np.random.random() < 0.3:
            idx1, idx2 = np.random.choice(24, 2, replace=False)
            invalid_pose[idx1], invalid_pose[idx2] = invalid_pose[idx2].copy(), invalid_pose[idx1].copy()
        
        # Method 3: Scale joints to extreme values
        if np.random.random() < 0.2:
            scale = np.random.uniform(5.0, 10.0)
            invalid_pose *= scale
        
        return invalid_pose
    
    def __len__(self):
        return len(self.valid_data) * 2  # Double the size to include invalid poses
    
    def __getitem__(self, idx):
        # Half valid poses, half invalid poses
        if idx < len(self.valid_data):
            # Valid pose
            joints3d = self.valid_data[idx]
            target = 0.0  # Low distance for valid poses
        else:
            # Invalid pose
            valid_idx = idx - len(self.valid_data)
            valid_joints = self.valid_data[valid_idx]
            joints3d = self.create_invalid_pose(valid_joints)
            target = 1.0  # High distance for invalid poses
        
        # Normalize the data
        normalized_data = self.normalize_data(joints3d)
        
        # Flatten to 1D
        flattened = normalized_data.flatten().astype(np.float32)
        
        return torch.from_numpy(flattened), torch.tensor([target], dtype=torch.float32)

def create_dataloaders(data_dir, batch_size=32, train_split=0.8, max_samples=None):
    """Create train and validation dataloaders"""
    
    # Create full dataset
    full_dataset = CorrectedPoseDataset(data_dir, max_samples=max_samples)
    
    if len(full_dataset) == 0:
        raise ValueError("No valid data found!")
    
    # Split into train and validation
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        drop_last=True
    )
    
    return train_loader, val_loader, full_dataset

def evaluate_metrics(model, val_loader, device):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            y_true.extend(targets.cpu().numpy().flatten())
            y_pred.extend(outputs.cpu().numpy().flatten())
    # AUC and accuracy (threshold 0.5)
    auc = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, np.array(y_pred) < 0.5)
    return auc, acc, y_true, y_pred

def train_model_corrected(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Train model with corrected distance field approach"""
    
    # Conservative hyperparameters
    learning_rate = 1e-4
    weight_decay = 1e-4
    
    # Use AdamW optimizer for better stability
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Use cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training history
    train_losses = []
    val_losses = []
    learning_rates = []
    
    # Gradient clipping value
    max_grad_norm = 1.0
    
    print(f"Starting corrected training with {num_epochs} epochs")
    print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    print(f"Max gradient norm: {max_grad_norm}")
    
    best_val_auc = 0
    best_epoch = 0
    best_model_state = None
    metrics_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        valid_batches = 0
        
        # Training loop
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            try:
                data, targets = data.to(device), targets.to(device)
                
                # Validate input data
                if torch.isnan(data).any() or torch.isinf(data).any():
                    continue
                
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(data)
                
                # Validate outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    continue
                
                # Compute loss
                loss = criterion(outputs, targets)
                
                # Validate loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Check for invalid gradients
                valid_gradients = True
                for param in model.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            valid_gradients = False
                            break
                
                if not valid_gradients:
                    optimizer.zero_grad()
                    continue
                
                optimizer.step()
                
                epoch_train_loss += loss.item()
                valid_batches += 1
                
            except Exception as e:
                continue
        
        # Validation
        model.eval()
        epoch_val_loss = 0.0
        val_valid_batches = 0
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data, targets in val_loader:
                try:
                    data, targets = data.to(device), targets.to(device)
                    
                    if torch.isnan(data).any() or torch.isinf(data).any():
                        continue
                    
                    outputs = model(data)
                    
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                        continue
                    
                    loss = criterion(outputs, targets)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        continue
                    
                    epoch_val_loss += loss.item()
                    val_valid_batches += 1
                    y_true.extend(targets.cpu().numpy().flatten())
                    y_pred.extend(outputs.cpu().numpy().flatten())
                except Exception as e:
                    continue
        
        # Compute average losses
        avg_train_loss = epoch_train_loss / max(valid_batches, 1)
        avg_val_loss = epoch_val_loss / max(val_valid_batches, 1)
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Compute metrics
        if len(set(y_true)) > 1:
            auc = roc_auc_score(y_true, y_pred)
            acc = accuracy_score(y_true, np.array(y_pred) < 0.5)
        else:
            auc = float('nan')
            acc = float('nan')
        metrics_history.append({
            "epoch": epoch+1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "auc": auc,
            "accuracy": acc,
            "learning_rate": current_lr
        })
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f} (valid batches: {valid_batches})")
        print(f"  Val Loss: {avg_val_loss:.6f} (valid batches: {val_valid_batches})")
        print(f"  AUC: {auc:.4f}  Accuracy: {acc:.4f}")
        print(f"  Learning Rate: {current_lr:.2e}")
        
        # Save best model by AUC
        if auc > best_val_auc:
            best_val_auc = auc
            best_epoch = epoch+1
            best_model_state = model.state_dict()
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoints/prior_3d_epoch_{epoch+1}_corrected.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'learning_rates': learning_rates,
                'metrics_history': metrics_history
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
        
        # Early stopping if loss becomes NaN
        if math.isnan(avg_train_loss) or math.isnan(avg_val_loss):
            print("NaN loss detected, stopping training")
            break
    
    # Save final model
    final_model_path = "checkpoints/prior_3d_final_corrected.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'metrics_history': metrics_history
    }, final_model_path)
    print(f"Saved final model: {final_model_path}")
    # Save best model
    if best_model_state is not None:
        best_model_path = os.path.join(RESULTS_DIR, "prior_3d_best_corrected.pth")
        torch.save({'model_state_dict': best_model_state}, best_model_path)
        print(f"Saved best model (by AUC): {best_model_path} (epoch {best_epoch})")
    # Save metrics history
    metrics_path = os.path.join(RESULTS_DIR, "metrics_history.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"Saved metrics history: {metrics_path}")
    # Save validation outputs for report
    val_outputs_path = os.path.join(RESULTS_DIR, "val_outputs.npz")
    np.savez(val_outputs_path, y_true=np.array(y_true), y_pred=np.array(y_pred))
    print(f"Saved validation outputs: {val_outputs_path}")
    return train_losses, val_losses, learning_rates, metrics_history

def plot_training_curves(train_losses, val_losses, learning_rates, save_path=None):
    """Plot training curves with enhanced visualization"""
    
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, "training_curves_corrected.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss (Corrected)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Use log scale only if there are positive values
    if any(loss > 0 for loss in train_losses + val_losses):
        ax1.set_yscale('log')
    
    # Plot learning rate
    ax2.plot(epochs, learning_rates, 'g-', label='Learning Rate', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved to: {save_path}")
    plt.show()

def main():
    """Main training function with corrected setup"""
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    data_dir = "extracted_3d_poses"
    batch_size = 16
    num_epochs = 50  # Start with fewer epochs for testing
    max_samples = None  # Start with smaller dataset for testing
    
    print("CORRECTED TRAINING CONFIGURATION:")
    print(f"  Data directory: {data_dir}")
    print(f"  Batch size: {batch_size}")
    print(f"  Number of epochs: {num_epochs}")
    print(f"  Max samples: {max_samples}")
    print("  Target: 0.0 for valid poses, 1.0 for invalid poses")
    
    try:
        # Create dataloaders
        print("\nCreating dataloaders...")
        train_loader, val_loader, dataset = create_dataloaders(
            data_dir, batch_size=batch_size, max_samples=max_samples
        )
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Create model
        print("\nCreating model...")
        input_dim = 72  # 24 joints * 3 coordinates
        model = PoseNDF3D(input_dim=input_dim).to(device)
        
        # Print model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Train model
        print("\nStarting corrected training...")
        train_losses, val_losses, learning_rates, metrics_history = train_model_corrected(
            model, train_loader, val_loader, num_epochs=num_epochs, device=device
        )
        
        # Plot results
        print("\nPlotting training curves...")
        plot_training_curves(train_losses, val_losses, learning_rates)
        
        # Save config
        config = {
            "data_dir": data_dir,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "max_samples": max_samples,
            "model_params": {
                "input_dim": input_dim,
                "hidden_dims": [256, 256, 256, 256],
                "output_dim": 1
            }
        }
        config_path = os.path.join(RESULTS_DIR, "training_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"Saved training config: {config_path}")
        print("\nCorrected training complete!")
        print("Next step: Test the corrected model")
        print(f"All results saved in: {RESULTS_DIR}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 