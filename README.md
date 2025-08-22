# POST: Prior-guided Source-free Domain Adaptation for Human Pose Estimation

This repository contains the implementation of a multi-person 2D pose estimation system with prior-guided regularization, based on the POST paper. The system operates on point cloud representations of MRI scans and outputs both presence detection (0-4 people) and precise 2D keypoint coordinates for each detected person.

## 📁 Repository Structure

```
├── src/
│   ├── pipeline/           # Main integrated 2D pipeline
│   │   ├── train.py        # Training script for integrated model
│   │   ├── eval.py         # Evaluation script
│   │   ├── data/           # Data loading modules
│   │   ├── models/         # Model architectures
│   │   └── utils/          # Utility functions and metrics
│   ├── priors/             # Prior model implementations
│   │   ├── 2d/            # 2D pose prior models
│   │   └── 3d/            # 3D pose prior models
│   └── thesis_project-main/ # Original thesis project files
├── checkpoints/            # Model checkpoints
├── results/               # Training results and evaluations
├── docs/                  # Documentation and reports
└── README.md
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)
- Required packages: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `scipy`

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd POST
```

2. Install dependencies:
```bash
pip install torch torchvision numpy pandas scikit-learn scipy
```

### Dataset Setup

#### MARS Dataset (2D Pipeline)
The integrated pipeline expects MARS dataset in the following structure:
```
thesis_project-main/MARS_SRCmpmri_MAXPPL4_GRID8_NORMED/src/
├── train/
│   ├── featuremap.npy     # Point cloud data (N, 16, 16, 5)
│   └── kpt_labels.npy     # Keypoint annotations (N, 140)
└── test/
    ├── featuremap.npy
    └── kpt_labels.npy
```

#### SURREAL Dataset (3D Prior)
For 3D prior training, download the SURREAL dataset and set the path in the training script.

## 🏋️ Training

### 1. Train 2D Prior Model
```bash
cd src/priors/2d
python train.py
```

### 2. Train 3D Prior Model (Optional)
```bash
cd src/priors/3d
python train.py --data-root /path/to/surreal/dataset
```

### 3. Train Integrated Pipeline
```bash
cd src/pipeline
python train.py
```

## 📊 Evaluation

### Evaluate Integrated Model
```bash
cd src/pipeline
python eval.py
```

This will evaluate the model and save results to `results/evaluation_results.json`.

## 📈 Results

The final integrated model achieves:
- **PCK@0.05**: 96.54% (excellent keypoint localization)
- **PCK@0.02**: 76.34% (good precise localization)
- **MPJPE**: 0.1632 (low positioning error)
- **Presence Accuracy**: 32.25% (needs improvement)

## 🔧 Model Architecture

### Integrated Pipeline
- **PoseidonPC Encoder**: mpMARS backbone for point cloud feature extraction
- **Improved Pose Estimator**: Residual MLP with separate heads for keypoints and presence
- **Prior Loss**: Neural distance field for anatomical constraint enforcement

### Key Features
- **Residual Connections**: Preserve gradient flow and feature information
- **Batch Normalization**: Stabilize training and accelerate convergence
- **Separate Heads**: Task-specific optimization for keypoints and presence
- **Prior-Guided Training**: Effective integration of anatomical constraints

## 📚 Documentation

- `docs/FINAL_INTEGRATED_MODEL_IMPLEMENTATION.md`: Technical implementation details
- `docs/FINAL_INTEGRATED_MODEL_DETAILED_IMPLEMENTATION.md`: Comprehensive technical writeup
- `docs/3D_PRIOR_MODEL_DEVELOPMENT.md`: 3D prior model development details
- `docs/FINAL_MODEL_REPORT.md`: Final model performance report

## 🎯 Key Contributions

1. **Residual Pose Estimator**: Novel architecture with improved gradient flow
2. **Prior Loss Integration**: Seamless incorporation of anatomical constraints
3. **Multi-Person Handling**: Efficient processing of variable person counts
4. **Medical Domain Adaptation**: Specialized for MRI point cloud data
5. **Comprehensive Evaluation Framework**: Multiple metrics for thorough assessment

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size in training scripts
2. **Data Loading Errors**: Ensure MARS dataset structure matches expected format
3. **Prior Model Not Found**: Train 2D prior model before running integrated pipeline

### Performance Tips

- Use GPU acceleration for faster training
- Adjust batch size based on available memory
- Monitor training logs for convergence issues

## 📄 License

This project is for research purposes. Please cite the original POST paper if you use this implementation.

## 🤝 Contributing

This is a research implementation. For questions or issues, please refer to the documentation or create an issue.

## 📞 Contact

For questions about this implementation, please refer to the documentation in the `docs/` folder.

