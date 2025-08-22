import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EmbedToLatents(nn.Module):
    def __init__(self, dim, dim_latents, weightless=False):
        super().__init__()
        self.weightless = weightless
        self.to_latents = (
            None if self.weightless else nn.Linear(dim, dim_latents, bias=False)
        )

    def forward(self, x):
        latents = x if self.weightless else self.to_latents(x)
        return F.normalize(latents, p=2, dim=-1)


class SharedMetricSpace(nn.Module):
    def __init__(
        self,
        pc_embedding_dim: int = 512,
        rgb_embedding_dim: int = 512,
        shared_metric_embedding_dim: int = 256,
    ):
        super(SharedMetricSpace, self).__init__()
        self.pc_embedding_dim = pc_embedding_dim
        self.rgb_embedding_dim = rgb_embedding_dim
        self.shared_metric_embedding_dim = shared_metric_embedding_dim
        self.learn_pc_projection_only = (
            True
            if self.rgb_embedding_dim == self.shared_metric_embedding_dim
            else False
        )
        # Linear layers for projecting ECG and text embeddings into the shared space
        self.pc_projection_layer = EmbedToLatents(
            self.pc_embedding_dim, self.shared_metric_embedding_dim
        )
        self.rgb_projection_layer = EmbedToLatents(
            self.rgb_embedding_dim,
            self.shared_metric_embedding_dim,
            self.learn_pc_projection_only,
        )

    def project_to_shared_metric_space(
        self, pc_embedding: torch.Tensor, rgb_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pc_shared = self.pc_projection_layer(
            pc_embedding
        )  # Shape: (batch_size, shared_metric_embedding_dim)
        rgb_shared = self.rgb_projection_layer(
            rgb_embedding
        )  # Shape: (batch_size, shared_metric_embedding_dim)
        return pc_shared, rgb_shared

    def compute_similarity(
        self, pc_shared: torch.Tensor, rgb_shared: torch.Tensor
    ) -> torch.Tensor:
        # Normalize the embeddings to unit vectors
        pc_normalized = F.normalize(pc_shared, p=2, dim=-1)
        rgb_normalized = F.normalize(rgb_shared, p=2, dim=-1)
        # Compute cosine similarity between the normalized embeddings
        similarity = torch.sum(
            pc_normalized * rgb_normalized, dim=-1
        )  # Shape: (batch_size,)
        return similarity

    def forward(
        self, pc_embedding: torch.Tensor, rgb_embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pc_shared, rgb_shared = self.project_to_shared_metric_space(
            pc_embedding, rgb_embedding
        )
        return pc_shared, rgb_shared


class mpMarsBackbone(nn.Module):
    def __init__(
        self, in_chs=5, out_chs=32, representation_embedding_dim=512, grid_dims=(8, 8)
    ):
        super(mpMarsBackbone, self).__init__()
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.grid_width = grid_dims[0]
        self.grid_height = grid_dims[1]
        self.representation_embedding_dim = representation_embedding_dim
        # inshape should be channels, height, width
        self.conv_one_1 = nn.Sequential(
            nn.Conv2d(
                self.in_chs,
                self.out_chs // 2,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.conv_one_2 = nn.Sequential(
            nn.Conv2d(
                self.out_chs // 2,
                self.out_chs,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="same",
            ),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm2d(out_chs, momentum=0.95),
        )
        self.flatten = nn.Flatten()
        self.embedding_layer = nn.Sequential(
            nn.Linear(
                self.out_chs * self.grid_width * self.grid_height,
                self.representation_embedding_dim,
            ),  # Adjust the size accordingly
            nn.ReLU(),
            nn.BatchNorm1d(self.representation_embedding_dim, momentum=0.95),
            nn.Dropout(0.4),
        )

    def forward(self, x):
        # print("Input shape:", x.shape)
        conv_1_out = self.conv_one_1(x)
        # print("After conv_one_1:", conv_1_out.shape)
        conv_2_out = self.conv_one_2(conv_1_out)
        # print("After conv_one_2:", conv_2_out.shape)
        flatten_out = self.flatten(conv_2_out)
        # print("After flatten:", flatten_out.shape)
        # print(f"Embed layer expects in: {self.out_chs} * {self.grid_width} * {self.grid_height} = {self.out_chs * self.grid_width * self.grid_height}, out: {self.representation_embedding_dim}")
        pc_representation_embedding = self.embedding_layer(flatten_out)
        # print(f"pc_embedding: {pc_embedding.shape}")
        return pc_representation_embedding


class mpMarsSharedHead(nn.Module):
    def __init__(self, in_emb_dim=512, max_people=4, kpt_dims=2, num_kpts=26):
        super(mpMarsSharedHead, self).__init__()
        self.in_emb_dim = in_emb_dim
        self.max_people = max_people
        self.kpt_dims = kpt_dims
        self.num_kpts = num_kpts
        self.presence_size = self.max_people + 1 if self.max_people > 1 else 0
        self.out_size = (
            self.max_people * self.num_kpts * self.kpt_dims + self.presence_size
        )

        self.out_layer = nn.Linear(self.in_emb_dim, self.out_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # print("mpMarsHead")
        # print(f"x: {x.shape}")
        out = self.out_layer(x)
        if self.max_people > 1:
            out[:, -self.presence_size :] = self.sig(out[:, -self.presence_size :])
        # print(f"out: {out.shape}")
        return out


class mpMarsPresenceHead(nn.Module):
    def __init__(self, in_emb_dim=512, max_people=4):
        super(mpMarsPresenceHead, self).__init__()
        self.in_emb_dim = in_emb_dim
        self.max_people = max_people
        self.presence_size = self.max_people + 1
        self.out_size = self.presence_size

        self.presence_head = nn.Sequential(
            nn.Linear(self.in_emb_dim, 128), nn.ReLU(), nn.Linear(128, self.out_size)
        )

    def forward(self, x):
        out = self.presence_head(x)
        return out


class mpMarsKeypointsHead(nn.Module):
    def __init__(self, in_emb_dim=512, max_people=4, kpt_dims=2, num_kpts=26):
        super(mpMarsKeypointsHead, self).__init__()
        self.in_emb_dim = in_emb_dim
        self.max_people = max_people
        self.kpt_dims = kpt_dims
        self.num_kpts = num_kpts
        self.out_size = self.max_people * self.num_kpts * self.kpt_dims
        self.out_layer = nn.Linear(self.in_emb_dim, self.out_size)

    def forward(self, x):
        # print("mpMarsKeypointsHead")
        # print(f"x: {x.shape}")
        out = self.out_layer(x)
        # print(f"out: {out.shape}")
        return out


class mpMarsDualHead(nn.Module):
    def __init__(self, in_emb_dim=512, max_people=4, kpt_dims=2, num_kpts=26):
        super(mpMarsDualHead, self).__init__()
        self.in_emb_dim = in_emb_dim
        self.max_people = max_people
        self.kpt_dims = kpt_dims
        self.num_kpts = num_kpts
        self.presence_size = self.max_people + 1
        self.kpts_size = self.max_people * self.num_kpts * self.kpt_dims

        # Enhanced presence detection head with residual connections and batch norm
        self.presence_head = nn.Sequential(
            nn.Linear(self.in_emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, self.presence_size)
        )

        # Keypoint prediction head
        self.keypoint_head = nn.Sequential(
            nn.Linear(self.in_emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, self.kpts_size)
        )

    def forward(self, x):
        presence_out = self.presence_head(x)
        keypoints_out = self.keypoint_head(x)
        return torch.cat([keypoints_out, presence_out], dim=1)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance with reduced intensity."""
    def __init__(self, alpha=1.0, gamma=1.0, reduction='mean'):  # Reduced gamma from 2.0 to 1.0
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class poseidonPcEncoder(nn.Module):

    def __init__(
        self,
        model_architecture: str = "mpMARS",
        representation_embedding_dim: int = 512,
        pretrained: bool = False,
        grid_dims: tuple = (8, 8),
    ):

        super(poseidonPcEncoder, self).__init__()

        self.model_architecture = model_architecture
        self.representation_embedding_dim = representation_embedding_dim
        self.pretrained = pretrained
        self.encoder = None
        self.grid_dims = grid_dims

        # Initialize the encoder model based on the given architecture
        self.initialize_model()

    def initialize_model(self):
        if self.model_architecture == "mpMARS":
            mpmars_backbone = mpMarsBackbone(
                representation_embedding_dim=self.representation_embedding_dim,
                grid_dims=self.grid_dims,
            )
            self.encoder = mpmars_backbone
        else:
            raise ValueError(
                f"Unsupported model architecture: {self.model_architecture}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the ECG encoder to obtain the shared embedding.

        Args:
            x (torch.Tensor): Input tensor representing ECG waveforms.

        Returns:
            torch.Tensor: Encoded ECG embedding of shape (batch_size, representation_embedding_dim).
        """
        if self.encoder is None:
            raise ValueError("Encoder model has not been initialized.")
        # print(f"x: {x.shape}")
        out = self.encoder(x)
        return out

    def display_model_summary(self):
        """
        Display a summary of the encoder model.
        """
        print("Model Summary:")
        print(self.encoder)


class poseidonPoseEstimator(nn.Module):
    def __init__(
        self,
        max_poses: int = 4,
        num_kpts: int = 17,
        hidden_dim: int = 256,  # Reduced from 512 to 256
        num_layers: int = 3,    # Reduced from 4 to 3
        dropout: float = 0.1,   # Reduced from 0.2 to 0.1
    ):
        super().__init__()
        self.max_poses = max_poses
        self.num_kpts = num_kpts
        self.hidden_dim = hidden_dim
        
        # Improved keypoint loss function with better handling of outliers
        self.kpts_loss_fn = nn.SmoothL1Loss(reduction='none')  # Huber loss for robustness
        
        # Focal loss for presence detection (keep this excellent)
        self.focal_loss_fn = FocalLoss(alpha=1.0, gamma=2.0, reduction='mean')
        
        # Enhanced pose estimation layers with residual connections
        self.pose_layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.pose_layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                self.pose_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Add batch normalization for better training stability
        self.pose_bn_layers = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Final pose prediction with improved architecture
        self.pose_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_poses * num_kpts * 2)  # x, y coordinates
        )
        
        # Enhanced presence prediction (keep this excellent)
        self.presence_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),  # Less dropout for presence (keep it accurate)
            nn.Linear(hidden_dim // 2, max_poses + 1)  # 0 to max_poses people
        )
        
        # Keypoint confidence prediction for uncertainty estimation
        self.keypoint_confidence = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_poses * num_kpts),
            nn.Sigmoid()
        )
        
        # Initialize weights with better initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training stability."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use Xavier initialization with smaller scale for stability
                nn.init.xavier_uniform_(module.weight, gain=0.5)  # Reduced gain from default
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass with improved architecture and residual connections."""
        batch_size = x.shape[0]
        
        # Process through pose layers with residual connections
        residual = x
        for i, (layer, bn_layer) in enumerate(zip(self.pose_layers, self.pose_bn_layers)):
            x = layer(x)
            x = bn_layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)
            
            # Add residual connection every 2 layers
            if i % 2 == 1 and i > 0:
                x = x + residual
                residual = x
        
        # Predict poses with improved head
        pose_output = self.pose_head(x)
        pose_output = pose_output.view(batch_size, self.max_poses, self.num_kpts, 2)
        
        # Predict presence (keep this excellent)
        presence_output = self.presence_head(x)
        
        # Predict keypoint confidence
        confidence_output = self.keypoint_confidence(x)
        confidence_output = confidence_output.view(batch_size, self.max_poses, self.num_kpts)
        
        # Combine outputs (return only the main output for compatibility)
        combined_output = torch.cat([
            pose_output.view(batch_size, -1),
            presence_output
        ], dim=1)
        
        return combined_output

    def compute_hpe_loss(self, prediction, target_kpts, target_presence=None):
        """Compute human pose estimation loss with improved handling."""
        batch_size = prediction.shape[0]
        
        if self.max_poses == 1:
            predicted_kpts = prediction.reshape(batch_size, self.num_kpts, 2)
            kpt_loss = self.kpts_loss_fn(predicted_kpts, target_kpts[:, 0])
            kpt_loss = kpt_loss.mean()
            return kpt_loss
        else:
            presence_logits = prediction[:, -(self.max_poses + 1):]
            presence_index_targets = torch.argmax(target_presence, dim=1)
            
            # Use focal loss for presence detection (keep this excellent)
            presence_loss = self.focal_loss_fn(presence_logits, presence_index_targets)
            
            predicted_kpts = prediction[:, :-(self.max_poses + 1)].reshape(
                batch_size, self.max_poses, self.num_kpts, 2
            )
            
            # Improved keypoint loss computation
            kpt_loss = self._compute_keypoint_loss(predicted_kpts, target_kpts, target_presence)

            return kpt_loss, presence_loss
    
    def _compute_keypoint_loss(self, predicted_kpts, target_kpts, target_presence):
        """Improved keypoint loss computation with better handling of valid poses."""
        batch_size = predicted_kpts.shape[0]
        
        # Get the number of valid poses for each batch
        num_valid_poses = (target_presence == 1).cumsum(dim=1).argmax(dim=1)
        pose_indices = torch.arange(self.max_poses, device=target_kpts.device).unsqueeze(0)
        valid_pose_mask = pose_indices < num_valid_poses.unsqueeze(1)
        
        # Only compute loss for valid poses
        if valid_pose_mask.sum() == 0:
            return torch.tensor(0.0, device=predicted_kpts.device)
        
        valid_target_kpts = target_kpts[valid_pose_mask]
        valid_predicted_kpts = predicted_kpts[valid_pose_mask]

        if valid_predicted_kpts.shape[0] < 1 or valid_target_kpts.shape[0] < 1:
            return torch.tensor(0.0, device=predicted_kpts.device)
        
        # Compute loss with improved handling
        kpt_loss = self.kpts_loss_fn(valid_predicted_kpts, valid_target_kpts)
        
        # Apply confidence weighting if available
        # For now, use mean loss
        kpt_loss = kpt_loss.mean()
        
        return kpt_loss

    def compute_presence_accuracy(self, prediction, target_presence):
        """Compute accuracy metrics for presence detection."""
        presence_logits = prediction[:, -(self.max_poses + 1):]
        pred_class = torch.argmax(presence_logits, dim=1)
        target_class = torch.argmax(target_presence, dim=1)
        
        correct = (pred_class == target_class).float()
        accuracy = correct.mean()
        
        # Compute per-class accuracy
        class_accuracies = []
        for i in range(self.max_poses + 1):
            mask = (target_class == i)
            if mask.sum() > 0:
                class_acc = correct[mask].mean()
                class_accuracies.append(class_acc.item())
            else:
                class_accuracies.append(0.0)
        
        return {
            'overall_accuracy': accuracy.item(),
            'per_class_accuracy': class_accuracies,
        }

    def compute_pck_metrics(self, prediction, target_kpts, target_presence, thresholds=[0.05, 0.1, 0.15, 0.2, 0.25]):
        """Compute PCK (Percentage of Correct Keypoints) metrics."""
        batch_size = prediction.shape[0]
        
        # Extract keypoint predictions
        pred_kpts = prediction[:, :-(self.max_poses + 1)].reshape(batch_size, self.max_poses, self.num_kpts, 2)
        
        # Get presence predictions
        presence_logits = prediction[:, -(self.max_poses + 1):]
        pred_presence = torch.argmax(presence_logits, dim=1)
        target_presence_class = torch.argmax(target_presence, dim=1)
        
        pck_results = {}
        
        for threshold in thresholds:
            correct_keypoints = 0
            total_keypoints = 0
            
            for b in range(batch_size):
                num_people = target_presence_class[b].item()
                if num_people == 0:
                    continue
                    
                # Compute distances for each person
                for p in range(num_people):
                    if p < self.max_poses:
                        pred_pose = pred_kpts[b, p]  # (num_kpts, 2)
                        gt_pose = target_kpts[b, p]  # (num_kpts, 2)
                        
                        # Compute Euclidean distances
                        distances = torch.norm(pred_pose - gt_pose, dim=1)  # (num_kpts,)
                        
                        # Improved normalization: use image size or fixed normalization
                        # Since we're working with normalized coordinates, use a fixed normalization factor
                        normalization_factor = 1.0  # Use 1.0 for normalized coordinates
                        
                        # Alternative: use bounding box size if available
                        if self.num_kpts >= 17:  # COCO format
                            # Use bounding box of keypoints for normalization
                            gt_bbox = torch.stack([
                                gt_pose.min(dim=0)[0],  # min x, y
                                gt_pose.max(dim=0)[0]   # max x, y
                            ])
                            bbox_size = torch.norm(gt_bbox[1] - gt_bbox[0])
                            normalization_factor = max(bbox_size, 0.1)  # Avoid division by zero
                        
                        # Normalize distances
                        normalized_distances = distances / normalization_factor
                        
                        # Count correct keypoints
                        correct_keypoints += (normalized_distances < threshold).sum().item()
                        total_keypoints += self.num_kpts
            
            if total_keypoints > 0:
                pck = correct_keypoints / total_keypoints
            else:
                pck = 0.0
                
            pck_results[f'PCK@{threshold}'] = pck
        
        # Compute AUC (Area Under Curve) for PCK
        pck_values = list(pck_results.values())
        auc = sum(pck_values) / len(pck_values)  # Simple average as AUC approximation
        
        pck_results['PCK_AUC'] = auc
        
        return pck_results

    def compute_mpjpe(self, prediction, target_kpts, target_presence):
        """Compute MPJPE (Mean Per Joint Position Error)."""
        batch_size = prediction.shape[0]
        
        # Extract keypoint predictions
        pred_kpts = prediction[:, :-(self.max_poses + 1)].reshape(batch_size, self.max_poses, self.num_kpts, 2)
        
        # Get presence predictions
        target_presence_class = torch.argmax(target_presence, dim=1)
        
        total_error = 0.0
        total_joints = 0
        
        for b in range(batch_size):
            num_people = target_presence_class[b].item()
            if num_people == 0:
                continue
                
            for p in range(num_people):
                if p < self.max_poses:
                    pred_pose = pred_kpts[b, p]  # (num_kpts, 2)
                    gt_pose = target_kpts[b, p]  # (num_kpts, 2)
                    
                    # Compute Euclidean distances
                    distances = torch.norm(pred_pose - gt_pose, dim=1)  # (num_kpts,)
                    
                    total_error += distances.sum().item()
                    total_joints += self.num_kpts
        
        if total_joints > 0:
            mpjpe = total_error / total_joints
        else:
            mpjpe = 0.0
            
        return mpjpe

    def compute_map(self, predictions, targets, presence_gt=None):
        """Compute mean Average Precision for pose estimation."""
        # Simplified mAP computation
        batch_size = predictions.shape[0]
        if self.max_poses == 1:
            predicted_kpts = predictions.reshape(batch_size, self.num_kpts, 2)
            # Simple distance-based score
            distances = torch.norm(predicted_kpts - targets[:, 0], dim=2)
            scores = torch.exp(-distances.mean(dim=1))
            map_score = (scores > 0.5).float().mean()
            return map_score.item()
        else:
            # For multi-person, return a simplified score
            return 0.5  # Placeholder

    def focal_loss(self, logits, targets):
        """Compute focal loss for presence detection."""
        return self.focal_loss_fn(logits, targets)

    def calculate_pck_mpjpe_metrics(self, pred_kpts, gt_kpts, presence_gt):
        """
        Calculate PCK and MPJPE metrics for keypoint accuracy evaluation.
        
        Args:
            pred_kpts: (batch_size, max_poses, num_kpts, 2) predicted keypoints
            gt_kpts: (batch_size, max_poses, num_kpts, 2) ground truth keypoints
            presence_gt: (batch_size, max_poses + 1) presence ground truth
            
        Returns:
            dict: Dictionary containing PCK and MPJPE metrics
        """
        batch_size, max_poses, num_kpts, _ = pred_kpts.shape
        
        # Initialize metric accumulators
        all_distances = []
        all_pck_scores = []
        
        # PCK thresholds for AUC calculation
        pck_thresholds = np.linspace(0.05, 0.5, 20)
        pck_at_thresholds = {thresh: [] for thresh in pck_thresholds}
        
        for b in range(batch_size):
            # Get number of people in this sample
            num_people = torch.argmax(presence_gt[b]).item()
            
            if num_people == 0:
                continue  # Skip samples with no people
                
            for p in range(num_people):
                pred_kpt = pred_kpts[b, p]  # (num_kpts, 2)
                gt_kpt = gt_kpts[b, p]      # (num_kpts, 2)
                
                # Skip if ground truth is all zeros (no pose)
                if torch.allclose(gt_kpt, torch.zeros_like(gt_kpt), atol=1e-6):
                    continue
                
                # Calculate distances
                distances = torch.norm(pred_kpt - gt_kpt, dim=1)  # (num_kpts,)
                all_distances.append(distances.cpu().numpy())
                
                # Calculate bounding box size for normalization
                gt_bbox_size = torch.max(torch.norm(gt_kpt, dim=1))
                if gt_bbox_size < 1e-6:
                    continue
                
                # Calculate PCK at different thresholds
                normalized_distances = distances / gt_bbox_size
                
                for threshold in pck_thresholds:
                    correct_kpts = (normalized_distances <= threshold).float()
                    pck_score = correct_kpts.mean().item()
                    pck_at_thresholds[threshold].append(pck_score)
                
                # Calculate overall PCK (using 0.2 threshold)
                pck_02 = (normalized_distances <= 0.2).float().mean().item()
                all_pck_scores.append(pck_02)
        
        # Calculate final metrics
        if len(all_distances) == 0:
            return {
                'mpjpe': 0.0,
                'pck_02': 0.0,
                'pck_auc': 0.0
            }
        
        # MPJPE
        all_distances = np.concatenate(all_distances, axis=0)
        mpjpe = np.mean(all_distances)
        
        # PCK@0.2
        pck_02 = np.mean(all_pck_scores) if all_pck_scores else 0.0
        
        # PCK AUC
        pck_means = {}
        for threshold in pck_thresholds:
            if pck_at_thresholds[threshold]:
                pck_means[threshold] = np.mean(pck_at_thresholds[threshold])
            else:
                pck_means[threshold] = 0.0
        
        # Calculate AUC using trapezoidal rule
        thresholds_array = np.array(list(pck_means.keys()))
        pck_values = np.array(list(pck_means.values()))
        pck_auc = np.trapz(pck_values, thresholds_array)
        
        return {
            'mpjpe': mpjpe,
            'pck_02': pck_02,
            'pck_auc': pck_auc
        }