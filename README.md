# HPE Integrated Model

A comprehensive Human Pose Estimation (HPE) pipeline that integrates 2D and 3D prior models with a multi-person pose estimation system. This project implements the POST (Prior-guided Source-free Domain Adaptation for Human Pose Estimation) approach with enhancements for multi-person scenarios.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Installation
```bash
# Clone the repository
git clone https://github.com/aarushibhasin/HPE-integrated-model.git
cd HPE-integrated-model

# Install dependencies
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install scikit-learn opencv-python
```

### Dataset Setup
1. Place your MARS MPMRI dataset in the `thesis_project-main/MARS_SRCmpmri_MAXPPL4_GRID8_NORMED/` directory
2. The dataset should have the following structure:
   ```
   thesis_project-main/MARS_SRCmpmri_MAXPPL4_GRID8_NORMED/
   ├── src/
   │   ├── train/
   │   │   ├── featuremap.npy
   │   │   └── kpt_labels.npy
   │   └── test/
   │       ├── featuremap.npy
   │       └── kpt_labels.npy
   ```

### Training
```bash
# Train the integrated model with prior loss
python src/pipeline/train.py
```

### Evaluation
```bash
# Evaluate the trained model
python src/pipeline/eval.py
```

## 📁 Project Structure

```
HPE-integrated-model/
├── src/
│   ├── pipeline/
│   │   ├── models/
│   │   │   ├── encoder_pc.py          # Point cloud encoder
│   │   │   ├── pose_estimator.py      # Multi-person pose estimator
│   │   │   └── prior_loss.py          # Prior loss integration
│   │   ├── data/
│   │   │   └── mars_dataset.py        # MARS dataset loader
│   │   ├── utils/
│   │   │   └── metrics.py             # Evaluation metrics
│   │   ├── train.py                   # Main training script
│   │   └── eval.py                    # Evaluation script
│   └── priors/
│       ├── prior_2d/                  # 2D prior models
│       │   ├── models.py
│       │   └── train.py
│       └── prior_3d/                  # 3D prior models
│           ├── models.py
│           └── train.py
├── checkpoints/                       # Model checkpoints
├── results/                           # Training results and plots
├── docs/                              # Documentation
└── thesis_project-main/               # Original core modules
```

## 🔧 Model Architecture

### Integrated Pipeline
- **Point Cloud Encoder**: mpMARS backbone for feature extraction
- **Pose Estimator**: Improved multi-person pose estimator with residual connections
- **Prior Loss**: Integration of 2D pose prior as a regularization term

### Prior Models
- **2D Prior (PoseNDF2D)**: Neural Distance Field for 2D pose validation
- **3D Prior (PoseNDF3D)**: Extended to 3D bone orientations

## 📊 Results

The model achieves:
- **PCK@0.05**: ~0.008 (presence detection: ~95.8%)
- **MPJPE**: ~0.8-1.0
- **Training**: Stable convergence with early stopping at ~21 epochs

## 📚 Documentation

- [Final Integrated Model Implementation](docs/FINAL_INTEGRATED_MODEL_IMPLEMENTATION.md)
- [Training Report](docs/training_report.md)
- [3D Prior Model Development](docs/3D_PRIOR_MODEL_DEVELOPMENT.md)

## 🔧 Recent Fixes

### Package Naming Issues
- **Fixed**: Renamed `src/priors/2d/` to `src/priors/prior_2d/` (Python packages cannot start with digits)
- **Fixed**: Renamed `src/priors/3d/` to `src/priors/prior_3d/`
- **Updated**: All import statements throughout the codebase

### Prior Model Checkpoints
- **Fixed**: Missing `prior_2d_final.pth` checkpoint (copied from `results/` to `checkpoints/`)
- **Fixed**: Corrupted `prior_3d_final.pth` with NaN weights (replaced with working checkpoint)
- **Fixed**: Model architecture mismatch (updated models to match checkpoint structure with BatchNorm layers)

### Model Architecture Updates
- **2D Prior**: Added BatchNorm1d layers to match checkpoint structure
- **3D Prior**: Simplified architecture to match checkpoint (72-dimensional input)
- **Verified**: All prior models load correctly without NaN weights

## 🚨 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure you're using the updated package names (`prior_2d`, `prior_3d`)
2. **Checkpoint Loading**: Verify checkpoints exist in `checkpoints/` directory
3. **Data Loading**: Ensure MARS dataset is in the correct location
4. **OpenMP Conflicts**: Set `KMP_DUPLICATE_LIB_OK=TRUE` environment variable

### Environment Setup
```bash
# For Windows PowerShell
$env:KMP_DUPLICATE_LIB_OK="TRUE"

# For Linux/Mac
export KMP_DUPLICATE_LIB_OK=TRUE
```

### Quick Verification Commands

**Test Prior Models:**
```bash
python -c "
import sys; sys.path.append('src')
from src.priors.prior_2d.models import PoseNDF2D
from src.priors.prior_3d.models import PoseNDF3D
import torch

# Test 2D prior
model_2d = PoseNDF2D()
checkpoint_2d = torch.load('checkpoints/prior_2d_final.pth', map_location='cpu')
model_2d.load_state_dict(checkpoint_2d['model_state_dict'])
print('✅ 2D prior loaded successfully')

# Test 3D prior
model_3d = PoseNDF3D()
checkpoint_3d = torch.load('checkpoints/prior_3d_final.pth', map_location='cpu')
model_3d.load_state_dict(checkpoint_3d['model_state_dict'])
print('✅ 3D prior loaded successfully')
"
```

**Test Main Model Checkpoints:**
```bash
python -c "
import sys; sys.path.append('src')
from src.pipeline.models.encoder_pc import poseidonPcEncoder
from src.pipeline.models.pose_estimator import ImprovedPoseEstimator
import torch

# Create models
encoder = poseidonPcEncoder(model_architecture='mpMARS', representation_embedding_dim=512, pretrained=False, grid_dims=(16, 16))
pose_estimator = ImprovedPoseEstimator(max_poses=4, num_kpts=17, hidden_dim=128, num_layers=2, dropout=0.3)

# Test loading best model
checkpoint = torch.load('checkpoints/best_improved_model.pth', map_location='cpu')
encoder.load_state_dict(checkpoint['encoder_state_dict'], strict=False)
pose_estimator.load_state_dict(checkpoint['pose_estimator_state_dict'], strict=False)
print('✅ Main model checkpoints loaded successfully')
"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Based on the POST paper: "Prior-guided Source-free Domain Adaptation for Human Pose Estimation"
- Original MARS dataset and poseidon framework
- PyTorch and the open-source community

