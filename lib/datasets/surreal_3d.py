"""
3D SURREAL Dataset for Pose Estimation
Extends the original SURREAL dataset to handle 3D annotations
"""
import os
import json
import random
import numpy as np
from PIL import Image, ImageFile
import torch
from .keypoint_dataset import Body16KeypointDataset
from ..transforms.keypoint_detection import *
from .util import *
from .util_3d import generate_3d_target, normalize_3d_pose
from ._util import download as download_data, check_exits

ImageFile.LOAD_TRUNCATED_IMAGES = True


class SURREAL3D(Body16KeypointDataset):
    """3D SURREAL Dataset for pose estimation
    
    Args:
        root (str): Root directory of dataset
        split (str): Dataset split ('train', 'test', 'val')
        download (bool): Whether to download dataset
        image_size (tuple): Image size (W, H)
        heatmap_size (tuple): Heatmap size (W, H)
        sigma (int): Gaussian sigma for heatmaps
        subset_ratio (float): Fraction of data to use (for smaller datasets)
    """
    def __init__(self, root, split='train', task='all', download=True, subset_ratio=1.0, **kwargs):
        assert split in ['train', 'test', 'val']
        self.split = split
        self.subset_ratio = subset_ratio

        if download:
            # Download minimal subset
            self._download_minimal_subset(root)
        else:
            check_exits(root, "train/run0")

        # Load samples
        all_samples = []
        if self.split == 'train':
            for part in [0, 1, 2]:
                annotation_file = os.path.join(root, split, f'run{part}.json')
                if os.path.exists(annotation_file):
                    print(f"Loading {annotation_file}")
                    with open(annotation_file) as f:
                        samples = json.load(f)
                        for sample in samples:
                            sample["image_path"] = os.path.join(root, self.split, f'run{part}', sample['name'])
                        all_samples.extend(samples)
        else:
            annotation_file = os.path.join(root, split, 'run0.json')
            if os.path.exists(annotation_file):
                print(f"Loading {annotation_file}")
                with open(annotation_file) as f:
                    samples = json.load(f)
                    for sample in samples:
                        sample["image_path"] = os.path.join(root, self.split, 'run0', sample['name'])
                    all_samples.extend(samples)

        # Apply subset ratio
        if self.subset_ratio < 1.0:
            random.seed(42)
            subset_size = int(len(all_samples) * self.subset_ratio)
            all_samples = random.sample(all_samples, subset_size)
            print(f"Using {len(all_samples)} samples ({self.subset_ratio*100:.1f}% of total)")

        # Map SMPL joints to our 16 keypoint format
        self.joints_index = (7, 4, 1, 2, 5, 8, 0, 11, 8, 10, 16, 15, 14, 11, 12, 13)

        super(SURREAL3D, self).__init__(root, all_samples, **kwargs)

    def _download_minimal_subset(self, root):
        """Download minimal subset of SURREAL dataset"""
        print("Downloading minimal SURREAL subset...")
        
        # Create directories
        os.makedirs(os.path.join(root, "train"), exist_ok=True)
        os.makedirs(os.path.join(root, "val"), exist_ok=True)
        os.makedirs(os.path.join(root, "test"), exist_ok=True)
        
        # Download minimal data (you'll need to implement this based on SURREAL structure)
        # This is a placeholder - you'll need to adapt based on actual SURREAL download links
        print("Please download SURREAL data manually and place in the following structure:")
        print(f"{root}/")
        print("├── train/")
        print("│   ├── run0/")
        print("│   │   ├── images/")
        print("│   │   ├── joints2d.json")
        print("│   │   ├── joints3d.json")
        print("│   │   └── camera.json")
        print("│   ├── run1/")
        print("│   └── run2/")
        print("├── val/")
        print("└── test/")

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample['name']
        image_path = sample['image_path']
        
        # Load image
        image = Image.open(image_path)
        
        # Load 3D annotations
        keypoint3d_camera = np.array(sample['keypoint3d'])[self.joints_index, :]  # NUM_KEYPOINTS x 3
        keypoint2d = np.array(sample['keypoint2d'])[self.joints_index, :]  # NUM_KEYPOINTS x 2
        intrinsic_matrix = np.array(sample['intrinsic_matrix'])
        Zc = keypoint3d_camera[:, 2]

        # Apply transforms
        image, data = self.transforms(
            image, 
            keypoint2d=keypoint2d, 
            keypoint3d=keypoint3d_camera,
            intrinsic_matrix=intrinsic_matrix
        )
        
        keypoint2d = data['keypoint2d']
        keypoint3d_camera = data['keypoint3d']
        intrinsic_matrix = data['intrinsic_matrix']

        # Normalize 2D pose
        visible = np.array([1.] * 16, dtype=np.float32)
        visible = visible[:, np.newaxis]

        # Generate 2D and 3D targets
        target_2d, target_3d, target_weight = generate_3d_target(
            keypoint3d_camera, visible, self.heatmap_size, self.sigma, self.image_size
        )
        
        target_2d = torch.from_numpy(target_2d)
        target_3d = torch.from_numpy(target_3d)
        target_weight = torch.from_numpy(target_weight)

        # Normalize 3D pose
        keypoint3d_n = normalize_3d_pose(keypoint3d_camera)

        meta = {
            'image': image_name,
            'keypoint2d': keypoint2d,  # (NUM_KEYPOINTS x 2)
            'keypoint3d': keypoint3d_n,  # (NUM_KEYPOINTS x 3)
            'camera_intrinsics': intrinsic_matrix,
            'target_2d': target_2d,
            'target_3d': target_3d,
        }
        
        return image, (target_2d, target_3d), target_weight, meta

    def __len__(self):
        return len(self.samples)


class SURREAL3D_mt(Body16KeypointDataset):
    """3D SURREAL Dataset with mean teacher augmentation"""
    def __init__(self, root, split='train', task='all', download=True, k=1, 
                 transforms_base=None, transforms_stu=None, transforms_tea=None, 
                 subset_ratio=1.0, **kwargs):
        assert split in ['train', 'test', 'val']
        self.split = split
        self.subset_ratio = subset_ratio
        self.transforms_base = transforms_base
        self.transforms_stu = transforms_stu
        self.transforms_tea = transforms_tea
        self.k = k

        if download:
            self._download_minimal_subset(root)
        else:
            check_exits(root, "train/run0")

        # Load samples (same as SURREAL3D)
        all_samples = []
        if self.split == 'train':
            for part in [0, 1, 2]:
                annotation_file = os.path.join(root, split, f'run{part}.json')
                if os.path.exists(annotation_file):
                    print(f"Loading {annotation_file}")
                    with open(annotation_file) as f:
                        samples = json.load(f)
                        for sample in samples:
                            sample["image_path"] = os.path.join(root, self.split, f'run{part}', sample['name'])
                        all_samples.extend(samples)
        else:
            annotation_file = os.path.join(root, split, 'run0.json')
            if os.path.exists(annotation_file):
                print(f"Loading {annotation_file}")
                with open(annotation_file) as f:
                    samples = json.load(f)
                    for sample in samples:
                        sample["image_path"] = os.path.join(root, self.split, 'run0', sample['name'])
                    all_samples.extend(samples)

        # Apply subset ratio
        if self.subset_ratio < 1.0:
            random.seed(42)
            subset_size = int(len(all_samples) * self.subset_ratio)
            all_samples = random.sample(all_samples, subset_size)
            print(f"Using {len(all_samples)} samples ({self.subset_ratio*100:.1f}% of total)")

        self.joints_index = (7, 4, 1, 2, 5, 8, 0, 11, 8, 10, 16, 15, 14, 11, 12, 13)

        super(SURREAL3D_mt, self).__init__(root, all_samples, **kwargs)

    def _download_minimal_subset(self, root):
        """Download minimal subset (same as SURREAL3D)"""
        print("Downloading minimal SURREAL subset...")
        # Same implementation as SURREAL3D

    def __getitem__(self, index):
        sample = self.samples[index]
        image_name = sample['name']
        image_path = sample['image_path']
        
        # Load image and annotations
        image = Image.open(image_path)
        keypoint3d_camera = np.array(sample['keypoint3d'])[self.joints_index, :]
        keypoint2d = np.array(sample['keypoint2d'])[self.joints_index, :]
        intrinsic_matrix = np.array(sample['intrinsic_matrix'])
        Zc = keypoint3d_camera[:, 2]

        # Apply base transform
        image, data = self.transforms_base(
            image, 
            keypoint2d=keypoint2d, 
            keypoint3d=keypoint3d_camera,
            intrinsic_matrix=intrinsic_matrix
        )
        
        keypoint2d = data['keypoint2d']
        keypoint3d_camera = data['keypoint3d']
        intrinsic_matrix = data['intrinsic_matrix']

        # Student transform
        image_stu, data_stu = self.transforms_stu(
            image, 
            keypoint2d=keypoint2d, 
            keypoint3d=keypoint3d_camera,
            intrinsic_matrix=intrinsic_matrix
        )
        
        keypoint2d_stu = data_stu['keypoint2d']
        keypoint3d_stu = data_stu['keypoint3d']
        intrinsic_matrix_stu = data_stu['intrinsic_matrix']
        aug_param_stu = data_stu['aug_param']

        # Generate targets for student
        visible = np.array([1.] * 16, dtype=np.float32)[:, np.newaxis]
        target_stu_2d, target_stu_3d, target_weight_stu = generate_3d_target(
            keypoint3d_stu, visible, self.heatmap_size, self.sigma, self.image_size
        )
        
        target_stu_2d = torch.from_numpy(target_stu_2d)
        target_stu_3d = torch.from_numpy(target_stu_3d)
        target_weight_stu = torch.from_numpy(target_weight_stu)

        # Original targets
        target_ori_2d, target_ori_3d, target_weight_ori = generate_3d_target(
            keypoint3d_camera, visible, self.heatmap_size, self.sigma, self.image_size
        )
        
        target_ori_2d = torch.from_numpy(target_ori_2d)
        target_ori_3d = torch.from_numpy(target_ori_3d)
        target_weight_ori = torch.from_numpy(target_weight_ori)

        # Normalize 3D poses
        keypoint3d_n_stu = normalize_3d_pose(keypoint3d_stu)
        keypoint3d_n_ori = normalize_3d_pose(keypoint3d_camera)

        meta_stu = {
            'image': image_name,
            'keypoint2d_ori': keypoint2d,
            'keypoint3d_ori': keypoint3d_n_ori,
            'target_ori_2d': target_ori_2d,
            'target_ori_3d': target_ori_3d,
            'target_weight_ori': target_weight_ori,
            'keypoint2d_stu': keypoint2d_stu,
            'keypoint3d_stu': keypoint3d_n_stu,
            'aug_param_stu': aug_param_stu,
        }

        # Teacher transforms
        images_tea, targets_tea_2d, targets_tea_3d, target_weights_tea, metas_tea = [], [], [], [], []
        for _ in range(self.k):
            image_tea, data_tea = self.transforms_tea(
                image, 
                keypoint2d=keypoint2d, 
                keypoint3d=keypoint3d_camera,
                intrinsic_matrix=intrinsic_matrix
            )
            
            keypoint2d_tea = data_tea['keypoint2d']
            keypoint3d_tea = data_tea['keypoint3d']
            intrinsic_matrix_tea = data_tea['intrinsic_matrix']
            aug_param_tea = data_tea['aug_param']

            # Generate targets for teacher
            target_tea_2d, target_tea_3d, target_weight_tea = generate_3d_target(
                keypoint3d_tea, visible, self.heatmap_size, self.sigma, self.image_size
            )
            
            target_tea_2d = torch.from_numpy(target_tea_2d)
            target_tea_3d = torch.from_numpy(target_tea_3d)
            target_weight_tea = torch.from_numpy(target_weight_tea)

            # Normalize 3D pose
            keypoint3d_n_tea = normalize_3d_pose(keypoint3d_tea)

            meta_tea = {
                'image': image_name,
                'keypoint2d_tea': keypoint2d_tea,
                'keypoint3d_tea': keypoint3d_n_tea,
                'aug_param_tea': aug_param_tea,
            }
            
            images_tea.append(image_tea)
            targets_tea_2d.append(target_tea_2d)
            targets_tea_3d.append(target_tea_3d)
            target_weights_tea.append(target_weight_tea)
            metas_tea.append(meta_tea)

        return (image_stu, (target_stu_2d, target_stu_3d), target_weight_stu, meta_stu, 
                images_tea, targets_tea_2d, targets_tea_3d, target_weights_tea, metas_tea) 