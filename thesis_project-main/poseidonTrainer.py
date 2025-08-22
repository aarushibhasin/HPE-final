import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import math
from pathlib import Path

import importlib
import gradNorm as gn
from prior_filter import filter_poses_with_prior

importlib.reload(gn)
__all__ = [
    "poseidonPoseTrainer",
    "poseidonBaseTrainer",
    "poseidonContrastivePoseTrainer",
]


class poseidonBaseTrainer:

    def __init__(self, model_wrapper, training_params):
        self.epoch = 0
        self.batch_idx = 0
        self.model_wrapper = model_wrapper
        self.max_people = self.model_wrapper.pose_estimator.max_poses
        self.pc_grid_dim = self.model_wrapper.pc_encoder.grid_dims[0]
        self.training_params = training_params
        self.max_poses = self.model_wrapper.pose_estimator.max_poses
        self.num_kpts = self.model_wrapper.pose_estimator.num_kpts
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        torch.cuda.empty_cache()
        self.model_wrapper.to_device(self.device)
        self.optimizer = self.initialize_optimizers()
        self.scheduler = self.initialize_schedulers()
        self.loss_log = []

    def initialize_optimizers(self):
        optimizers = {}
        for module_name, module in self.model_wrapper.get_trainable_modules().items():
            params = module["params"]
            print(f"{module_name} params: {params}")
            optimizers[module_name] = optim.Adam(
                module["model"].parameters(), lr=params["learning_rate"]
            )  # , weight_decay=params["weight_decay"])
        return optimizers

    def initialize_schedulers(self):
        schedulers = {}
        for module_name, module in self.model_wrapper.get_trainable_modules().items():
            params = module["params"]
            schedulers[module_name] = lr_scheduler.StepLR(
                self.optimizer[module_name],
                step_size=params["scheduler_step_size"],
                gamma=params["scheduler_gamma"],
            )
        return schedulers

    def setup_train(self):
        self.best_eval_metric = self.default_best_eval_metric
        self.train_df = pd.DataFrame(columns=self.results_column_names)
        self.val_df = pd.DataFrame(columns=self.results_column_names)

    def train_one_epoch(self, dataloader: DataLoader):
        raise NotImplementedError("Subclasses should implement train_one_epoch")

    def validate_one_epoch(self, dataloader: DataLoader):
        raise NotImplementedError("Subclasses should implement validate_one_epoch")

    def best_model_criterion(self):
        raise NotImplementedError("Subclasses should implement model_criterion")

    def print_epoch_results(self):
        raise NotImplementedError("Subclasses should implement log_epoch")

    def plot_per_epoch_loss(self):
        raise NotImplementedError("Subclasses should implement plot_per_epoch_loss")

    def run_callbacks(self, eptr_results, epvl_results):
        self.train_df = pd.concat([self.train_df, eptr_results], ignore_index=True)
        self.plot_per_epoch_loss(self.train_df, "tr")
        self.val_df = pd.concat([self.val_df, epvl_results], ignore_index=True)
        self.plot_per_epoch_loss(self.val_df, "vl")
        self.train_df.to_csv("results\\train_results.csv")
        self.val_df.to_csv("results\\val_results.csv")
        self.save_checkpoint()
        # Save model if validation loss improves
        if self.best_model_criterion():
            self.save_checkpoint("results\\best_model.pth")

    def train(self, train_dataloader, val_dataloader):
        self.setup_train()
        for ep in range(self.training_params["num_epochs"]):
            self.epoch = ep + 1
            print(f"Starting epoch {self.epoch}/{self.training_params['num_epochs']}")
            eptr_results_df = self.train_one_epoch_nogradnorm(train_dataloader)
            epvl_results_df = self.validate_one_epoch(val_dataloader)

            self.run_callbacks(eptr_results_df, epvl_results_df)
            torch.cuda.empty_cache()

    def save_checkpoint(self, file_path: str = "results\\last_model.pth"):
        checkpoint = self.model_wrapper.get_checkpoint()
        checkpoint["optimizer_state_dict"] = {
            k: v.state_dict() for k, v in self.optimizer.items()
        }
        checkpoint["scheduler_state_dict"] = {
            k: v.state_dict() for k, v in self.scheduler.items()
        }
        checkpoint["epoch"] = self.epoch
        torch.save(checkpoint, file_path)
        print(f"Model checkpoint saved to {file_path}")

    def load_checkpoint(self, file_path: str = "best_model.pth"):
        checkpoint = torch.load(file_path)
        self.model_wrapper.load_checkpoint(checkpoint)
        for k, i in checkpoint["optimizer_state_dict"].items():
            self.optimizer[k].load_state_dict(checkpoint["optimizer_state_dict"][k])
        for k, i in checkpoint["scheduler_state_dict"].items():
            self.scheduler[k].load_state_dict(checkpoint["scheduler_state_dict"][k])

    def plot_one_prediction(
        self,
        split,
        predicted_kpts,
        target_kpts,
        predicted_presence_probs=None,
        target_presence=None,
    ):
        fig, ax = plt.subplots(figsize=(8, 8))

        if self.max_people == 1:
            predicted_kpts = predicted_kpts.reshape(-1, self.num_kpts, 2)
            print(f"predicted_kpts.shape: {predicted_kpts.shape}")
            x_coords, y_coords = predicted_kpts[0, :, 0], predicted_kpts[0, :, 1]
            ax.scatter(
                x_coords, -y_coords, label=f"Predicted Pose 1", marker="o", c="blue"
            )

            x_coords, y_coords = target_kpts[0, 0, :, 0], target_kpts[0, 0, :, 1]
            ax.scatter(
                x_coords, -y_coords, label=f"Target Pose 1", marker="x", c="green"
            )
        else:
            # print(f"target_presence: {target_presence}")
            # print(f"predicted_presence_probs: {predicted_presence_probs}")
            # num_valid_poses = (target_presence == 1).cumsum(dim=1).argmax(dim=1)  # Shape: (batch_size,)
            num_valid_poses = target_presence.argmax(dim=1)  # Shape: (batch_size,)
            sample_index = (
                np.where(num_valid_poses > 1)[0][0]
                if (num_valid_poses > 1).any()
                else 0
            )
            # print(f"num_valid_poses: {num_valid_poses}")
            pose_indices = torch.arange(
                self.max_poses, device=target_kpts.device
            ).unsqueeze(
                0
            )  # Shape: (1, max_poses)
            # print(f"pose_indices: {pose_indices}")
            valid_pose_mask = pose_indices < num_valid_poses.unsqueeze(
                1
            )  # Shape: (batch_size, max_poses)
            # print(f"valid_pose_mask: {valid_pose_mask}")
            filtered_target_kpts = target_kpts[sample_index][
                valid_pose_mask[sample_index]
            ]
            # print(f"filtered_target_kpts.shape: {filtered_target_kpts.shape}")
            # print(f"predicted_kpts: {predicted_kpts.shape}")
            predicted_kpts = predicted_kpts.reshape(
                -1, self.max_poses, self.num_kpts, 2
            )
            # print(f"predicted_kpts: {predicted_kpts.shape}")
            num_valid_pred_poses = predicted_presence_probs.argmax(
                dim=1
            )  # Shape: (batch_size,)
            # num_valid_pred_poses = (predicted_presence_probs > 0.5).cumsum(dim=1).argmax(dim=1)  # Shape: (batch_size,)
            # print(f"num_valid_pred_poses: {num_valid_pred_poses}")
            valid_pose_mask = pose_indices < num_valid_pred_poses.unsqueeze(
                1
            )  # Shape: (batch_size, max_poses)
            filtered_predicted_kpts = predicted_kpts[sample_index][
                valid_pose_mask[sample_index]
            ]
            # print(f"valid_pose_mask: {valid_pose_mask}")
            # print(f"filtered_predicted_kpts: {filtered_predicted_kpts}")
            # predicted_people_indices = np.where(predicted_presence_probs >= 0.1)[0]
            # filtered_predicted_kpts = predicted_kpts[predicted_people_indices]

            # print(f"sample_index: {sample_index}")
            if filtered_predicted_kpts.shape[0] != 0:
                filtered_predicted_kpts = filtered_predicted_kpts.reshape(
                    -1, self.num_kpts, 2
                ).numpy()
                # print(f"filtered_predicted_kpts.shape: {filtered_predicted_kpts.shape}")
                # Plot predicted poses
                for i, pose in enumerate(filtered_predicted_kpts):
                    x_coords, y_coords = pose[:, 0], pose[:, 1]
                    ax.scatter(
                        x_coords,
                        -y_coords,
                        label=f"Predicted Pose {i+1}",
                        marker="o",
                        c="blue",
                    )
                    # ax.plot(x_coords, y_coords, linestyle='--', alpha=0.6)

            if filtered_target_kpts.shape[0] != 0:
                filtered_target_kpts = filtered_target_kpts.reshape(
                    -1, self.num_kpts, 2
                ).numpy()
                # print(f"filtered_target_kpts.shape: {filtered_target_kpts.shape}")
                # Plot target poses
                for j, pose in enumerate(filtered_target_kpts):
                    x_coords, y_coords = pose[:, 0], pose[:, 1]
                    ax.scatter(
                        x_coords,
                        -y_coords,
                        label=f"Target Pose {j+1}",
                        marker="x",
                        c="green",
                    )
                    # ax.plot(x_coords, y_coords, linestyle='-', alpha=0.6)

        # Set plot details
        ax.set_title("Predicted vs Target Keypoints")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.legend()
        ax.grid(True)
        fig_save_folder = Path("results") / split / "preds"
        fig_save_folder.mkdir(exist_ok=True, parents=True)
        fig_save_path = fig_save_folder / f"{self.epoch}_{self.batch_idx}.png"
        fig.savefig(fig_save_path)
        plt.close()


class poseidonContrastivePoseTrainer(poseidonBaseTrainer):

    results_column_names = [
        "epoch",
        "avg_contrastive_loss",
        "avg_keypoint_loss",
        "avg_presence_loss",
        "avg_total_loss",
    ]
    default_best_eval_metric = float("inf")

    def __init__(self, model_wrapper, training_params):
        super().__init__(model_wrapper, training_params)

    def train_one_epoch(self, dataloader: DataLoader):
        self.model_wrapper.activate_training_mode()
        total_contrastive_loss = 0.0
        total_kpt_loss = 0.0
        total_presence_loss = 0.0
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch_idx, batch_data in enumerate(dataloader):
            self.batch_idx = batch_idx
            kpt_gt = batch_data["kpts"].to(self.device)
            presence_gt = batch_data["presence"].to(self.device)
            pc = batch_data["pc"].to(self.device)
            pc = pc.reshape(-1, 5, 8, 8)
            rgb = batch_data["rgb"].to(self.device)

            pc_rep_embeddings = self.model_wrapper.pc_encoder(pc)
            rgb_rep_embeddings = self.model_wrapper.rgb_encoder(rgb)
            # Get shared embeddings using the shared embedding space
            pc_shared_metric, rgb_shared_metric = (
                self.model_wrapper.shared_metric_space(
                    pc_rep_embeddings, rgb_rep_embeddings
                )
            )
            contrastive_loss = self.model_wrapper.contrastive_loss(
                pc_shared_metric, rgb_shared_metric
            )
            pred_pose = self.model_wrapper.pose_estimator(pc_rep_embeddings)
            kpt_loss, presence_loss = (
                self.model_wrapper.pose_estimator.compute_hpe_loss(
                    pred_pose, kpt_gt, presence_gt
                )
            )

            # Combine losses for back prop
            total_loss_value = (
                self.training_params["contrastive_loss_weight"] * contrastive_loss
                + self.training_params["kpt_loss_weight"] * kpt_loss
                + self.training_params["presence_loss_weight"] * presence_loss
            )

            # Backpropagate and optimize each component
            for _, opt in self.optimizer.items():
                opt.zero_grad()

            total_loss_value.backward()

            for _, opt in self.optimizer.items():
                opt.step()

            total_loss += total_loss_value.detach().item()
            total_contrastive_loss += contrastive_loss.detach().item()
            total_kpt_loss += kpt_loss.detach().item()
            total_presence_loss += presence_loss.detach().item()

            print(
                f"Train Epoch [{self.epoch}/{self.training_params['num_epochs']}], Batch [{batch_idx + 1}/{num_batches}], avg_total_loss: {total_loss / num_batches}, avg_contrastive_loss: {total_contrastive_loss / num_batches}, avg_keypoint_loss: {total_kpt_loss / num_batches}, avg_presence_loss: {total_presence_loss / num_batches}"
            )
            if batch_idx % 10 == 0:
                predicted_presence = pred_pose[:, -(self.max_poses + 1) :]
                predicted_presence_log_probs = F.softmax(predicted_presence, dim=1)
                print(f"predicted_presence: {predicted_presence}")
                predicted_kpts = pred_pose[:, : -(self.max_poses + 1)]
                self.plot_one_prediction(
                    predicted_kpts.detach().cpu(),
                    predicted_presence_log_probs.detach().cpu(),
                    kpt_gt.detach().cpu(),
                    presence_gt.detach().cpu(),
                )

        avg_total_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_kpt_loss = total_kpt_loss / num_batches
        avg_presence_loss = total_presence_loss / num_batches

        loss_df = pd.DataFrame(
            {
                "epoch": self.epoch,
                "avg_total_loss": [avg_total_loss],
                "avg_contrastive_loss": [avg_contrastive_loss],
                "avg_keypoint_loss": [avg_kpt_loss],
                "avg_presence_loss": [avg_presence_loss],
            }
        )
        return loss_df

    def validate_one_epoch(self, dataloader: DataLoader):
        self.model_wrapper.activate_eval_mode()

        total_contrastive_loss = 0.0
        total_kpt_loss = 0.0
        total_presence_loss = 0.0
        total_loss = 0.0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                kpt_gt = batch_data["kpts"].to(self.device)
                presence_gt = batch_data["presence"].to(self.device)
                pc = batch_data["pc"].to(self.device)
                rgb = batch_data["rgb"].to(self.device)
                pc = pc.reshape(-1, 5, 8, 8)

                pc_rep_embeddings = self.model_wrapper.pc_encoder(pc)
                rgb_rep_embeddings = self.model_wrapper.rgb_encoder(rgb)
                # Get shared embeddings using the shared embedding space
                pc_shared_metric, rgb_shared_metric = (
                    self.model_wrapper.shared_metric_space(
                        pc_rep_embeddings, rgb_rep_embeddings
                    )
                )
                contrastive_loss = self.model_wrapper.contrastive_loss(
                    pc_shared_metric, rgb_shared_metric
                )
                pred_pose = self.model_wrapper.pose_estimator(pc_rep_embeddings)
                kpt_loss, presence_loss = (
                    self.model_wrapper.pose_estimator.compute_hpe_loss(
                        pred_pose, kpt_gt, presence_gt
                    )
                )

                # Combine losses for back prop
                total_loss_value = (
                    self.training_params["contrastive_loss_weight"] * contrastive_loss
                    + self.training_params["kpt_loss_weight"] * kpt_loss
                    + self.training_params["presence_loss_weight"] * presence_loss
                )

                total_loss += total_loss_value.detach().item()
                total_contrastive_loss += contrastive_loss.detach().item()
                total_kpt_loss += kpt_loss.detach().item()
                total_presence_loss += presence_loss.detach().item()

                print(
                    f"Eval Epoch [{self.epoch}/{self.training_params['num_epochs']}], Batch [{batch_idx + 1}/{num_batches}], avg_total_loss: {total_loss / num_batches}, avg_contrastive_loss: {total_contrastive_loss / num_batches}, avg_keypoint_loss: {total_kpt_loss / num_batches}, avg_presence_loss: {total_presence_loss / num_batches}"
                )

        avg_total_loss = total_loss / num_batches
        avg_contrastive_loss = total_contrastive_loss / num_batches
        avg_kpt_loss = total_kpt_loss / num_batches
        avg_presence_loss = total_presence_loss / num_batches

        loss_df = pd.DataFrame(
            {
                "epoch": self.epoch,
                "avg_total_loss": [avg_total_loss],
                "avg_contrastive_loss": [avg_contrastive_loss],
                "avg_keypoint_loss": [avg_kpt_loss],
                "avg_presence_loss": [avg_presence_loss],
            }
        )
        return loss_df

    def best_model_criterion(self):
        result = self.train_df["avg_total_loss"][0] < self.best_eval_metric
        if result:
            self.best_eval_metric = self.train_df["avg_total_loss"][0]
            print(
                f"Saving the best model with validation Loss: {self.train_df['avg_total_loss'][0]}"
            )
        return result

    def print_epoch_results(self):
        print(
            f"Tr Epoch [{self.epoch}], avg_contrastive_loss: {self.train_df['avg_contrastive_loss'][0]}, avg_keypoint_loss: {self.train_df['avg_keypoint_loss'][0]}, avg_presence_loss: {self.train_df['avg_presence_loss'][0]}, avg_total_loss: {self.train_df['avg_total_loss'][0]}"
        )
        print(
            f"Vl Epoch [{self.epoch}], avg_contrastive_loss: {self.val_df['avg_contrastive_loss'][0]}, avg_keypoint_loss: {self.val_df['avg_keypoint_loss'][0]}, avg_presence_loss: {self.val_df['avg_presence_loss'][0]}, avg_total_loss: {self.val_df['avg_total_loss'][0]}"
        )


class poseidonPoseTrainer(poseidonBaseTrainer):

    results_column_names = [
        "epoch",
        "avg_keypoint_loss",
        "avg_presence_loss",
        "avg_total_loss",
        "presence_accuracy",
        "presence_per_class_accuracy",
        "mean_avg_precision",
        "pck_auc",
        "mpjpe",
        "pck_0.2"
    ]
    default_best_eval_metric = float("inf")

    def __init__(self, model_wrapper, training_params):
        super().__init__(model_wrapper, training_params)
        self.max_people = self.model_wrapper.pose_estimator.max_poses
        self.pc_grid_dim = self.model_wrapper.pc_encoder.grid_dims[0]

    def best_model_criterion(self):
        if len(self.val_df) == 0:
            return False
        current_val_loss = self.val_df.iloc[-1]["avg_total_loss"]
        if current_val_loss < self.best_eval_metric:
            self.best_eval_metric = current_val_loss
            return True
        return False

    def print_epoch_results(self):
        if len(self.train_df) > 0:
            tr_results = self.train_df.iloc[-1]
            tr_string = f"Train Epoch {self.epoch}: "
            tr_string += f"avg_keypoint_loss: {tr_results['avg_keypoint_loss']:.4f}, "
            tr_string += f"avg_presence_loss: {tr_results['avg_presence_loss']:.4f}, "
            tr_string += f"avg_total_loss: {tr_results['avg_total_loss']:.4f}"
            print(tr_string)

        if len(self.val_df) > 0:
            vl_results = self.val_df.iloc[-1]
            vl_string = f"Val Epoch {self.epoch}: "
            vl_string += f"avg_keypoint_loss: {vl_results['avg_keypoint_loss']:.4f}, "
            vl_string += f"avg_presence_loss: {vl_results['avg_presence_loss']:.4f}, "
            vl_string += f"avg_total_loss: {vl_results['avg_total_loss']:.4f}"
            print(vl_string)

    def plot_per_epoch_loss(self, df, split):
        epochs = df["epoch"]
        result_names = [["avg_keypoint_loss"]]
        if self.max_people > 1:
            result_names[0].append("avg_presence_loss")
            result_names[0].append("avg_total_loss")
            if self.model_wrapper.grad_norm is not None:
                result_names.append(
                    ["gradnorm_kpt_loss_weight", "gradnorm_presence_loss_weight"]
                )

        for result_plot in result_names:
            fig, ax = plt.subplots(figsize=(8, 8))
            for metric in result_plot:
                ax.plot(epochs, df[metric], label=metric)
            plot_name = "+".join(result_plot)
            ax.legend()
            ax.set_xlabel("Epochs")
            fig_save_folder = Path("results") / split / "loss_figures"
            fig_save_folder.mkdir(exist_ok=True, parents=True)
            fig_save_path = fig_save_folder / f"{plot_name}.png"
            fig.savefig(fig_save_path)
            plt.close()

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

    def train_one_epoch_nogradnorm(self, dataloader):
        self.model_wrapper.activate_training_mode()
        total_loss = 0.0
        total_kpt_loss = 0.0
        total_presence_loss = 0.0
        total_presence_accuracy = 0.0
        total_per_class_accuracy = [0.0] * (self.max_poses + 1)
        total_map = 0.0
        total_pck_auc = 0.0
        total_mpjpe = 0.0
        total_pck_02 = 0.0
        num_batches = len(dataloader)
        running_kpt_losses = []  # Track running statistics for anomaly detection

        for batch_idx, batch_data in enumerate(dataloader):
            self.batch_idx = batch_idx
            pc = batch_data["pc"].to(self.device)
            kpt_gt = batch_data["kpts"].to(self.device)
            presence_gt = batch_data["presence"].to(self.device)

            # Zero gradients
            for optimizer in self.optimizer.values():
                optimizer.zero_grad()

            # Forward pass
            pc = pc.reshape(-1, 5, self.pc_grid_dim, self.pc_grid_dim)
            pc_embeddings = self.model_wrapper.pc_encoder(pc)
            pred_pose = self.model_wrapper.pose_estimator(pc_embeddings)

            if self.max_people > 1:
                # Extract predicted keypoints
                predicted_kpts = pred_pose[:, :-(self.max_poses + 1)].reshape(
                    -1, self.max_poses, self.num_kpts, 2
                ).detach().cpu().numpy()

                # Filter with prior
                valid_mask = filter_poses_with_prior(predicted_kpts, threshold=0.5)  # Fixed threshold
                
                # Monitor filtering rate
                filter_rate = 1 - (valid_mask.sum() / valid_mask.size)
                if batch_idx % 10 == 0:  # Print every 10 batches
                    print(f"Batch {batch_idx}: Filter rate = {filter_rate:.2%} ({valid_mask.sum()}/{valid_mask.size} poses kept)")

                # Convert masks to torch and filter predictions and ground truth
                valid_mask_torch = torch.from_numpy(valid_mask).to(kpt_gt.device)
                filtered_predicted_kpts = torch.from_numpy(predicted_kpts[valid_mask]).to(kpt_gt.device)
                filtered_kpt_gt = kpt_gt.cpu().numpy()[valid_mask]
                filtered_kpt_gt = torch.from_numpy(filtered_kpt_gt).to(kpt_gt.device)

                # Compute keypoint loss only on valid poses
                if filtered_predicted_kpts.numel() > 0:
                    kpt_loss = self.model_wrapper.pose_estimator.kpts_loss_fn(filtered_predicted_kpts, filtered_kpt_gt)
                    if kpt_loss.dim() > 0:
                        kpt_loss = kpt_loss.mean()
                else:
                    kpt_loss = torch.tensor(0.0, device=kpt_gt.device)
                # Always compute presence loss on the original batch
                presence_loss = self.model_wrapper.pose_estimator.focal_loss(
                    pred_pose[:, -(self.max_poses + 1):], torch.argmax(presence_gt, dim=1)
                )
                if presence_loss.dim() > 0:
                    presence_loss = presence_loss.mean()
                
                # Calculate PCK and MPJPE metrics
                predicted_kpts_tensor = torch.from_numpy(predicted_kpts).to(kpt_gt.device)
                metrics = self.calculate_pck_mpjpe_metrics(predicted_kpts_tensor, kpt_gt, presence_gt)
                total_pck_auc += metrics['pck_auc']
                total_mpjpe += metrics['mpjpe']
                total_pck_02 += metrics['pck_02']
                
            else:
                kpt_loss, presence_loss = self.model_wrapper.pose_estimator.compute_hpe_loss(
                    pred_pose, kpt_gt, presence_gt
                )
                if kpt_loss.dim() > 0:
                    kpt_loss = kpt_loss.mean()
                if presence_loss.dim() > 0:
                    presence_loss = presence_loss.mean()

            # Track running statistics
            running_kpt_losses.append(kpt_loss.item())
            current_mean = np.mean(running_kpt_losses)
            current_std = np.std(running_kpt_losses) if len(running_kpt_losses) > 1 else 0

            # Detect training anomalies
            if len(running_kpt_losses) > 1 and kpt_loss.item() > current_mean + 2 * current_std:
                print(f"\nWarning: Anomalous training batch detected (ID: {batch_idx})")
                print(f"Current loss: {kpt_loss.item():.4f}")
                print(f"Mean loss: {current_mean:.4f}")
                print(f"Std loss: {current_std:.4f}")

            if self.model_wrapper.grad_norm is None:
                weighted_loss = (
                    self.training_params["kpt_loss_weight"] * kpt_loss
                    + self.training_params["presence_loss_weight"]
                    * presence_loss
                )
            else:
                loss = torch.stack([kpt_loss, presence_loss], axis=0)
                weighted_loss = self.model_wrapper.grad_norm.weights @ loss

            # Backward pass
            weighted_loss.backward()

            # Gradient clipping with stronger clipping
            for name, param in self.model_wrapper.pose_estimator.named_parameters():
                if param.grad is not None:
                    torch.nn.utils.clip_grad_norm_(param, 0.1)  # Stronger clipping

            # Optimize
            for optimizer in self.optimizer.values():
                optimizer.step()

            # Calculate presence accuracy
            if self.max_people > 1:
                presence_pred = torch.softmax(pred_pose[:, -(self.max_poses + 1):], dim=1)
                presence_pred_idx = torch.argmax(presence_pred, dim=1)
                presence_gt_idx = torch.argmax(presence_gt, dim=1)
                presence_accuracy = (presence_pred_idx == presence_gt_idx).float().mean().item()
                total_presence_accuracy += presence_accuracy

                # Per-class accuracy
                for i in range(self.max_poses + 1):
                    mask = presence_gt_idx == i
                    if mask.sum() > 0:
                        class_acc = (presence_pred_idx[mask] == presence_gt_idx[mask]).float().mean().item()
                        total_per_class_accuracy[i] += class_acc

            total_loss += weighted_loss.detach().item()
            total_kpt_loss += kpt_loss.detach().item()
            if self.max_people > 1:
                total_presence_loss += presence_loss.detach().item()

            if batch_idx % 5 == 0:  # Print every 5 batches
                print(f"Train Batch [{batch_idx}/{num_batches}] - "
                      f"Current Loss: {kpt_loss.item():.4f}, "
                      f"Running Mean: {current_mean:.4f}, "
                      f"Running Std: {current_std:.4f}")

        # Calculate averages
        avg_loss = total_loss / num_batches
        avg_kpt_loss = total_kpt_loss / num_batches
        avg_presence_loss = total_presence_loss / num_batches if self.max_people > 1 else 0.0
        avg_presence_accuracy = total_presence_accuracy / num_batches if self.max_people > 1 else 0.0
        avg_pck_auc = total_pck_auc / num_batches if self.max_people > 1 else 0.0
        avg_mpjpe = total_mpjpe / num_batches if self.max_people > 1 else 0.0
        avg_pck_02 = total_pck_02 / num_batches if self.max_people > 1 else 0.0

        # Per-class accuracy
        avg_per_class_accuracy = [acc / num_batches for acc in total_per_class_accuracy]

        # Return DataFrame instead of dictionary
        results_dict = {
            "epoch": self.epoch,
            "avg_keypoint_loss": avg_kpt_loss,
            "avg_presence_loss": avg_presence_loss,
            "avg_total_loss": avg_loss,
            "presence_accuracy": avg_presence_accuracy,
            "presence_per_class_accuracy": [avg_per_class_accuracy],  # Wrap in list for DataFrame
            "mean_avg_precision": total_map / num_batches,
            "pck_auc": avg_pck_auc,
            "mpjpe": avg_mpjpe,
            "pck_0.2": avg_pck_02
        }

        return pd.DataFrame([results_dict])

    def test_num_people_preds(self, dataloader: DataLoader):
        self.model_wrapper.activate_eval_mode()
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                pc = batch_data["pc"].to(self.device)
                kpt_gt = batch_data["kpts"].to(self.device)
                presence_gt = batch_data["presence"].to(self.device)
                pc = pc.reshape(-1, 5, 8, 8)

                pc_embeddings = self.model_wrapper.pc_encoder(pc)
                pred_pose = self.model_wrapper.pose_estimator(pc_embeddings)

                predicted_presence = pred_pose[:, -(self.max_poses + 1) :]
                predicted_presence = F.softmax(predicted_presence, dim=1)
                predicted_kpts = pred_pose[:, : -(self.max_poses + 1)]
                pred_pr = predicted_presence.detach().cpu()
                gt_pr = presence_gt.detach().cpu()
                for i in range(pred_pr.shape[0]):
                    # if True or np.argmax(gt_pr[i]) != 1:
                    print(f"{i}")
                    print(f"pr: {np.argmax(pred_pr[i])}")
                    print(f"pr: {pred_pr[i]}")
                    print(f"gt: {np.argmax(gt_pr[i])}")

    def validate_one_epoch(self, dataloader: DataLoader):
        self.model_wrapper.activate_eval_mode()
        total_kpt_loss = 0.0
        total_presence_loss = 0.0
        total_weighted_loss = 0.0
        total_map = 0.0  # Track validation mAP
        total_pck_auc = 0.0
        total_mpjpe = 0.0
        total_pck_02 = 0.0
        total_presence_accuracy = 0.0
        num_batches = len(dataloader)
        running_val_losses = []  # Track validation statistics

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                self.batch_idx = batch_idx
                pc = batch_data["pc"].to(self.device)
                pc = pc.reshape(-1, 5, self.pc_grid_dim, self.pc_grid_dim)
                kpt_gt = batch_data["kpts"].to(self.device)
                presence_gt = batch_data["presence"].to(self.device)

                pc_embeddings = self.model_wrapper.pc_encoder(pc)
                pred_pose = self.model_wrapper.pose_estimator(pc_embeddings)
                
                if self.max_people == 1:
                    kpt_loss, presence_loss = self.model_wrapper.pose_estimator.compute_hpe_loss(
                        pred_pose, kpt_gt, presence_gt
                    )
                    if kpt_loss.dim() > 0:
                        kpt_loss = kpt_loss.mean()
                    if presence_loss.dim() > 0:
                        presence_loss = presence_loss.mean()
                else:
                    # Extract predicted keypoints
                    predicted_kpts = pred_pose[:, :-(self.max_poses + 1)].reshape(
                        -1, self.max_poses, self.num_kpts, 2
                    ).cpu().numpy()

                    # Filter with prior
                    valid_mask = filter_poses_with_prior(predicted_kpts, threshold=0.5)  # Fixed threshold
                    
                    # Monitor filtering rate
                    filter_rate = 1 - (valid_mask.sum() / valid_mask.size)
                    if batch_idx % 10 == 0:  # Print every 10 batches
                        print(f"Val Batch {batch_idx}: Filter rate = {filter_rate:.2%} ({valid_mask.sum()}/{valid_mask.size} poses kept)")

                    # Convert masks to torch and filter predictions and ground truth
                    valid_mask_torch = torch.from_numpy(valid_mask).to(kpt_gt.device)
                    filtered_predicted_kpts = torch.from_numpy(predicted_kpts[valid_mask]).to(kpt_gt.device)
                    filtered_kpt_gt = kpt_gt.cpu().numpy()[valid_mask]
                    filtered_kpt_gt = torch.from_numpy(filtered_kpt_gt).to(kpt_gt.device)

                    # Compute loss only on valid poses
                    if filtered_predicted_kpts.numel() > 0:
                        kpt_loss = self.model_wrapper.pose_estimator.kpts_loss_fn(filtered_predicted_kpts, filtered_kpt_gt)
                        if kpt_loss.dim() > 0:
                            kpt_loss = kpt_loss.mean()
                    else:
                        kpt_loss = torch.tensor(0.0, device=kpt_gt.device)
                    # Always compute presence loss on the original batch
                    presence_loss = self.model_wrapper.pose_estimator.focal_loss(
                        pred_pose[:, -(self.max_poses + 1):], torch.argmax(presence_gt, dim=1)
                    )
                    if presence_loss.dim() > 0:
                        presence_loss = presence_loss.mean()

                    # Calculate PCK and MPJPE metrics
                    predicted_kpts_tensor = torch.from_numpy(predicted_kpts).to(kpt_gt.device)
                    metrics = self.calculate_pck_mpjpe_metrics(predicted_kpts_tensor, kpt_gt, presence_gt)
                    total_pck_auc += metrics['pck_auc']
                    total_mpjpe += metrics['mpjpe']
                    total_pck_02 += metrics['pck_02']
                    
                    # Calculate presence accuracy
                    presence_pred = torch.softmax(pred_pose[:, -(self.max_poses + 1):], dim=1)
                    presence_pred_idx = torch.argmax(presence_pred, dim=1)
                    presence_gt_idx = torch.argmax(presence_gt, dim=1)
                    presence_accuracy = (presence_pred_idx == presence_gt_idx).float().mean().item()
                    total_presence_accuracy += presence_accuracy

                    # Track running statistics
                    running_val_losses.append(kpt_loss.item())
                    current_mean = np.mean(running_val_losses)
                    current_std = np.std(running_val_losses) if len(running_val_losses) > 1 else 0

                    # Detect validation anomalies
                    if len(running_val_losses) > 1 and kpt_loss.item() > current_mean + 2 * current_std:
                        print(f"\nWarning: Anomalous validation batch detected (ID: {batch_idx})")
                        print(f"Current loss: {kpt_loss.item():.4f}")
                        print(f"Mean loss: {current_mean:.4f}")
                        print(f"Std loss: {current_std:.4f}")
                        
                        # Additional diagnostics for anomalous batches
                        print("Batch Statistics:")
                        print(f"Number of keypoints: {kpt_gt.shape}")
                        print(f"Presence distribution: {presence_gt.sum(dim=0)}")
                        
                    if self.model_wrapper.grad_norm is None:
                        weighted_loss = (
                            self.training_params["kpt_loss_weight"] * kpt_loss
                            + self.training_params["presence_loss_weight"]
                            * presence_loss
                        )
                    else:
                        loss = torch.stack([kpt_loss, presence_loss], axis=0)
                        weighted_loss = self.model_wrapper.grad_norm.weights @ loss

                total_kpt_loss += kpt_loss.detach().item()
                if self.max_people > 1:
                    total_presence_loss += presence_loss.detach().item()
                    total_weighted_loss += weighted_loss.detach().item()

                if batch_idx % 5 == 0:  # Print every 5 batches
                    print(f"Validate Batch [{batch_idx}/{num_batches}] - "
                          f"Current Loss: {kpt_loss.item():.4f}, "
                          f"Running Mean: {current_mean:.4f}, "
                          f"Running Std: {current_std:.4f}")

        # Calculate averages
        avg_kpt_loss = total_kpt_loss / num_batches
        avg_presence_loss = total_presence_loss / num_batches if self.max_people > 1 else 0.0
        avg_weighted_loss = total_weighted_loss / num_batches if self.max_people > 1 else avg_kpt_loss
        avg_pck_auc = total_pck_auc / num_batches if self.max_people > 1 else 0.0
        avg_mpjpe = total_mpjpe / num_batches if self.max_people > 1 else 0.0
        avg_pck_02 = total_pck_02 / num_batches if self.max_people > 1 else 0.0
        avg_presence_accuracy = total_presence_accuracy / num_batches if self.max_people > 1 else 0.0

        # Return DataFrame instead of dictionary
        results_dict = {
            "epoch": self.epoch,
            "avg_keypoint_loss": avg_kpt_loss,
            "avg_presence_loss": avg_presence_loss,
            "avg_total_loss": avg_weighted_loss,
            "presence_accuracy": avg_presence_accuracy,
            "presence_per_class_accuracy": [[0.0] * (self.max_poses + 1)],  # Not calculated for validation
            "mean_avg_precision": total_map / num_batches,
            "pck_auc": avg_pck_auc,
            "mpjpe": avg_mpjpe,
            "pck_0.2": avg_pck_02
        }

        return pd.DataFrame([results_dict])