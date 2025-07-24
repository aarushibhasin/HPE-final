# POST 3D: Prior-guided Source-free Domain Adaptation for 3D Human Pose Estimation

This repository extends the original POST (Prior-guided Source-free Domain Adaptation for Human Pose Estimation) to support **3D pose estimation** using the SURREAL dataset.

## 🚀 Key Features

- **3D Pose Estimation**: Extends 2D pose estimation to full 3D joint positions
- **Source-free Domain Adaptation**: Adapts from synthetic (SURREAL) to real (LSP) domains without source data
- **3D Human Pose Prior**: Neural distance field for 3D pose validation
- **Teacher-Student Learning**: Mean teacher framework with 3D consistency
- **Minimal Data Requirements**: Uses only essential SURREAL data (RGB + 3D annotations)

## 📁 Project Structure

```
├── lib/
│   ├── models/
│   │   ├── pose_resnet_3d.py          # 3D pose estimation model
│   │   └── loss_3d.py                 # 3D loss functions
│   └── datasets/
│       ├── surreal_3d.py              # 3D SURREAL dataset
│       └── util_3d.py                 # 3D data utilities
├── prior/
│   ├── models_3d.py                   # 3D prior network
│   └── train_prior_3d.py              # 3D prior training
├── train_human_prior_3d.py            # Main 3D training script
├── config_3d.py                       # Configuration file
└── README_3D.md                       # This file
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd POST-3D
```

2. **Install dependencies**:
```bash
pip install torch torchvision
pip install numpy opencv-python pillow tqdm
```

3. **Download minimal SURREAL dataset**:
```bash
# Create data directory
mkdir -p data/surreal_3d

# Download minimal subset (instructions below)
```

## �� Dataset Setup

### Automated Download (Recommended)

The quick start script now includes **fully automated dataset download**:

```bash
# Run the complete pipeline including dataset download
python quick_start_3d.py

# Or download datasets separately
python scripts/download_surreal_3d.py --root ./data/surreal_3d --create-annotations
```

### Manual Download (Alternative)

If you prefer manual download or the automated script fails:

#### SURREAL Dataset (Minimal Subset)

The SURREAL dataset is very large (~40GB). For 3D pose estimation, you only need:

**Essential Files**:
- RGB images (subset)
- 2D/3D joint annotations
- Camera parameters

**Download Strategy**:
```bash
# Option 1: Use automated script (recommended)
python scripts/download_surreal_3d.py --root ./data/surreal_3d --create-annotations

# Option 2: Manual download from SURREAL website
wget https://www.di.ens.fr/willow/research/surreal/data/SURREAL/data/run0/cmu/train/run0_c0001_c0001_000001_poses.json
wget https://www.di.ens.fr/willow/research/surreal/data/SURREAL/data/run0/cmu/train/run0_c0001_c0001_000001.jpg
```

**Directory Structure** (automatically created):
```
data/surreal_3d/
├── train/
│   ├── run0/
│   │   ├── images/
│   │   ├── joints2d.json
│   │   ├── joints3d.json
│   │   └── camera.json
│   ├── run1/
│   └── run2/
├── val/
└── test/
```

#### LSP Dataset (Target Domain)

```bash
# Automated download (included in quick start)
# Or manual download:
wget http://sam.johnson.io/research/lsp_dataset_original.zip
unzip lsp_dataset_original.zip -d data/lsp
```

### Dataset Verification

After downloading, verify dataset integrity:

```bash
python scripts/verify_dataset_3d.py
```

This will check:
- ✅ Image files exist and are valid
- ✅ Annotation files are properly formatted
- ✅ 3D joint coordinates are present
- ✅ Camera parameters are available

## 🏋️ Training

### Step 1: Train 3D Prior Network

```bash
python prior/train_prior_3d.py \
    --data-root ./data/surreal_3d \
    --subset-ratio 0.1 \
    --epochs 50 \
    --batch-size 64 \
    --lr 0.001 \
    --log logs/prior_3d
```

### Step 2: Train 3D Pose Estimation Model

```bash
python train_human_prior_3d.py \
    --source-root ./data/surreal_3d \
    --target-root ./data/lsp \
    --source SURREAL3D \
    --target LSP \
    --target-train LSP_mt \
    --arch pose_resnet50_3d \
    --prior checkpoints/prior_3d/prior_3d_epoch_50.pth.tar \
    --subset-ratio 0.1 \
    --epochs 70 \
    --pretrain-epoch 40 \
    --batch-size 32 \
    --lr 0.001 \
    --log logs/3d_pose
```

### Step 3: Evaluate

```bash
python train_human_prior_3d.py \
    --source-root ./data/surreal_3d \
    --target-root ./data/lsp \
    --source SURREAL3D \
    --target LSP \
    --arch pose_resnet50_3d \
    --resume checkpoints/3d_pose/checkpoint_0069.pth.tar \
    --phase test
```

## ⚙️ Configuration

### Model Architecture

The 3D model extends the original 2D model:

```python
# 2D + 3D outputs
output_2d, output_3d = model(input_image)
# output_2d: (B, K, H, W) - 2D heatmaps
# output_3d: (B, K, H, W) - 3D depth maps
```

### Loss Functions

1. **Joints3DLoss**: Combined 2D + 3D loss
2. **ConsLoss3D**: 3D consistency loss for teacher-student
3. **DepthConsistencyLoss**: Depth consistency with SURREAL depth maps
4. **GeometricConsistencyLoss**: Bone length and angle constraints

### Key Parameters

```python
# Loss weights
alpha_2d = 1.0      # 2D loss weight
alpha_3d = 1.0      # 3D loss weight
lambda_c = 1.0      # Consistency loss weight
lambda_b = 1e-3     # Barlow twins loss weight
lambda_p = 1e-6     # Prior loss weight

# Training phases
pretrain_epoch = 40  # Pretrain on source domain
step_p = 47         # Start prior loss at epoch 47
```

## 📈 Results

### Expected Performance

| Metric | Source (SURREAL) | Target (LSP) |
|--------|------------------|--------------|
| PCK@0.2 (2D) | ~95% | ~85% |
| MPJPE (3D) | ~25mm | ~35mm |
| PCK@0.1 (3D) | ~90% | ~75% |

### Training Curves

- **2D Accuracy**: Improves during pretraining, stabilizes during adaptation
- **3D Accuracy**: Benefits from 3D prior guidance
- **Prior Loss**: Decreases as model learns valid 3D poses

## 🔧 Customization

### Adding New Datasets

1. **Create dataset class**:
```python
class YourDataset3D(Body16KeypointDataset):
    def __init__(self, root, split='train', **kwargs):
        # Load your 3D annotations
        # Format: joints_3d (K, 3), joints_2d (K, 2)
        pass
```

2. **Update configuration**:
```python
# In config_3d.py
YOUR_DATASET_ROOT = os.path.join(DATA_ROOT, 'your_dataset')
```

### Modifying 3D Prior

1. **Change bone connections**:
```python
# In prior/models_3d.py
self.bone_pairs = [
    [0, 1],   # Your bone connections
    [1, 2],
    # ...
]
```

2. **Adjust noise parameters**:
```python
# In prior/train_prior_3d.py
rotation_noise = 0.1      # Rotation noise in radians
translation_noise = 0.1   # Translation noise
joint_noise = 0.05        # Per-joint noise
```

## 🐛 Troubleshooting

### Common Issues

1. **Out of Memory**:
```bash
# Reduce batch size
--batch-size 16

# Use gradient accumulation
--accumulation-steps 2
```

2. **Poor 3D Performance**:
```bash
# Increase 3D loss weight
--alpha-3d 2.0

# Adjust prior weight
--lambda-p 1e-5
```

3. **Dataset Loading Issues**:
```bash
# Check data format
python scripts/check_dataset.py --data-root ./data/surreal_3d

# Verify joint mapping
python scripts/visualize_joints.py --image-path path/to/image.jpg
```

### Debug Mode

```bash
# Enable debug logging
export DEBUG=1

# Check intermediate outputs
python train_human_prior_3d.py --debug --save-intermediate
```

## 📚 References

- **Original POST**: [ICCV 2023 Paper](link-to-paper)
- **SURREAL Dataset**: [SURREAL: Data-driven Generation of Synthetic Human Bodies](https://www.di.ens.fr/willow/research/surreal/)
- **LSP Dataset**: [Learning Human Pose Estimation Features with Convolutional Networks](http://sam.johnson.io/research/lsp.html)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original POST authors for the 2D implementation
- SURREAL dataset creators for the synthetic data
- LSP dataset creators for the real-world data

---

**Note**: This is an experimental extension of the original POST method. Results may vary depending on your specific setup and data quality. 