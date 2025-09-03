#!/usr/bin/env python3
"""
Quick sanity check: valid vs corrupted poses should separate under PoseNDF2D.

Usage:
  python scripts/validate_prior_separation.py
"""

import numpy as np
import torch
from pathlib import Path

from src.pipeline.models.prior_loss import PriorLoss
from src.priors.prior_2d.models import PoseNDF2D


def load_one_valid_pose() -> torch.Tensor:
    # Load a single valid pose from MPMRI val set
    kpt_path = Path("thesis_project-main/MARS_SRCmpmri_MAXPPL4_GRID8_NORMED/src/val/kpt_labels.npy")
    data = np.load(kpt_path)
    N = len(data)
    max_people, num_kpts, kpt_dims = 4, 17, 2
    # Extract first 136 values (keypoints), reshape to (N, 2, 4, 17), then transpose to (N, 4, 17, 2)
    kpts = data[:, :max_people * num_kpts * kpt_dims].reshape(N, kpt_dims, max_people, num_kpts)
    kpts = np.transpose(kpts, (0, 2, 3, 1))
    # pick a non-empty pose
    for i in range(N):
        for p in range(max_people):
            pose = kpts[i, p]
            if np.any(pose != 0):
                return torch.tensor(pose, dtype=torch.float32).unsqueeze(0)
    raise RuntimeError("No valid pose found")


def corrupt_pose(pose: torch.Tensor) -> torch.Tensor:
    # Simple joint swap corruption
    idx = torch.arange(17)
    perm = idx.roll(1)
    corrupted = pose.clone()
    corrupted[:, idx] = pose[:, perm]
    return corrupted


def main():
    pose = load_one_valid_pose()  # (1, 17, 2)
    fake = corrupt_pose(pose)

    real_bones = PriorLoss._keypoints_to_bones(pose)  # (1, 34)
    fake_bones = PriorLoss._keypoints_to_bones(fake)  # (1, 34)

    model = PoseNDF2D(input_dim=34, hidden_dims=[256, 256, 256, 256])
    ckpt = torch.load("checkpoints/prior_2d_final.pth", map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        r = model(real_bones).item()
        f = model(fake_bones).item()

    print(f"Real pose score: {r:.6f}")
    print(f"Fake  pose score: {f:.6f}")
    if r < f:
        print("✅ Lower=better confirmed: real < fake")
    else:
        print("❌ Unexpected: real >= fake. Check data pipeline.")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()


