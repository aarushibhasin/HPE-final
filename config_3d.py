"""
Configuration file for 3D Pose Estimation Training
"""
import os

# Dataset paths
DATA_ROOT = './data'
SURREAL_3D_ROOT = os.path.join(DATA_ROOT, 'surreal_3d')
LSP_ROOT = os.path.join(DATA_ROOT, 'lsp')

# Model configuration
MODEL_CONFIG = {
    'arch': 'pose_resnet50_3d',
    'num_keypoints': 16,
    'pretrained_backbone': True,
    'deconv_with_bias': False,
    'finetune': False,
}

# Training configuration
TRAIN_CONFIG = {
    # General training
    'epochs': 70,
    'pretrain_epoch': 40,
    'batch_size': 32,
    'test_batch': 32,
    'lr': 0.001,
    'lr_step': [50, 60],
    'lr_factor': 0.1,
    'SGD': False,
    'teacher_alpha': 0.999,
    
    # Loss weights
    'alpha_2d': 1.0,
    'alpha_3d': 1.0,
    'lambda_c': 1.0,
    'lambda_b': 1e-3,
    'lambda_p': 1e-6,
    'step_p': 47,
    
    # Data augmentation
    'image_size': 256,
    'heatmap_size': 64,
    'resize_scale': [0.5, 1.0],
    'rotation_stu': 60,
    'rotation_tea': 60,
    'shear_stu': [-30, 30],
    'shear_tea': [-30, 30],
    'translate_stu': [0.05, 0.05],
    'translate_tea': [0.05, 0.05],
    'scale_stu': [0.6, 1.3],
    'scale_tea': [0.6, 1.3],
    'color_stu': 0.25,
    'color_tea': 0.25,
    'blur_stu': 0,
    'blur_tea': 0,
    
    # Training settings
    'k': 1,
    'mask_ratio': 0.5,
    'sigma': 2,
    'fix_head': False,
    'fix_upsample': False,
    'source_free': True,
    
    # Other
    'seed': 0,
    'workers': 4,
    'iters_per_epoch': 1000,
    'print_freq': 100,
    'eval_freq': 10,
    'save_freq': 10,
}

# Prior training configuration
PRIOR_CONFIG = {
    'epochs': 50,
    'batch_size': 64,
    'lr': 0.001,
    'lr_step': 20,
    'lr_gamma': 0.5,
    'weight_decay': 1e-4,
    
    # Noise parameters
    'rotation_noise': 0.1,
    'translation_noise': 0.1,
    'joint_noise': 0.05,
    'noise_penalty': 1.0,
    'noise_weight': 1.0,
    
    # Other
    'seed': 42,
    'workers': 4,
    'print_freq': 100,
    'save_freq': 10,
}

# Dataset configuration
DATASET_CONFIG = {
    'subset_ratio': 0.1,  # Use 10% of SURREAL data
    'image_size': 256,
    'heatmap_size': 64,
    'sigma': 2,
}

# Logging configuration
LOG_CONFIG = {
    'train_log': 'logs/3d_pose_train',
    'prior_log': 'logs/prior_3d',
    'checkpoint_dir': 'checkpoints/3d_pose',
}

# SURREAL dataset structure (for reference)
SURREAL_STRUCTURE = {
    'train': {
        'run0': ['images', 'joints2d.json', 'joints3d.json', 'camera.json'],
        'run1': ['images', 'joints2d.json', 'joints3d.json', 'camera.json'],
        'run2': ['images', 'joints2d.json', 'joints3d.json', 'camera.json'],
    },
    'val': {
        'run0': ['images', 'joints2d.json', 'joints3d.json', 'camera.json'],
    },
    'test': {
        'run0': ['images', 'joints2d.json', 'joints3d.json', 'camera.json'],
    }
}

# Joint mapping (SMPL to 16 keypoints)
JOINT_MAPPING = {
    'smpl_to_16': [7, 4, 1, 2, 5, 8, 0, 11, 8, 10, 16, 15, 14, 11, 12, 13],
    'joint_names': [
        'right_ankle', 'right_knee', 'right_hip', 'left_hip', 'left_knee', 'left_ankle',
        'right_wrist', 'right_elbow', 'right_shoulder', 'left_shoulder', 'left_elbow', 'left_wrist',
        'neck', 'nose', 'head_top', 'pelvis'
    ]
}

# Bone connections for 3D prior
BONE_CONNECTIONS = [
    [8, 9],   # pelvis to neck
    [2, 8],   # right_hip to pelvis
    [3, 8],   # left_hip to pelvis
    [12, 8],  # right_shoulder to neck
    [13, 8],  # left_shoulder to neck
    [1, 2],   # right_knee to right_hip
    [0, 1],   # right_ankle to right_knee
    [4, 3],   # left_knee to left_hip
    [5, 4],   # left_ankle to left_knee
    [11, 12], # right_elbow to right_shoulder
    [10, 11], # right_wrist to right_elbow
    [14, 13], # left_elbow to left_shoulder
    [15, 14]  # left_wrist to left_elbow
]

# Camera parameters (default SURREAL camera)
DEFAULT_CAMERA = {
    'focal_length': [1000, 1000],
    'principal_point': [256, 256],
    'image_size': [512, 512]
}

# 3D coordinate system
COORDINATE_SYSTEM = {
    'origin': 'pelvis',  # Root joint
    'up_axis': 'Y',      # Y-axis is up
    'forward_axis': 'Z', # Z-axis is forward
    'right_axis': 'X',   # X-axis is right
}

# Evaluation metrics
EVALUATION_METRICS = {
    '2d_metrics': ['PCK', 'AUC'],
    '3d_metrics': ['MPJPE', 'PCK_3D', 'AUC_3D'],
    'pck_thresholds': [0.1, 0.2, 0.3, 0.4, 0.5],
    'pck_3d_thresholds': [0.05, 0.1, 0.15, 0.2, 0.25],
} 