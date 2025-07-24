@echo off
echo ========================================
echo SURREAL 3D Prior Training Pipeline
echo ========================================

echo.
echo Step 1: Extracting 3D joint data from SURREAL _info.mat files...
python extract_surreal_3d_poses.py --surreal-root ./surreal_info/SURREAL/data --output-dir ./extracted_3d_poses --format numpy

echo.
echo Step 2: Training 3D prior model on extracted data...
python train_prior_on_3d_poses.py --data-dir ./extracted_3d_poses --epochs 50 --batch-size 64 --lr 0.001

echo.
echo ========================================
echo Prior Training Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Check the trained model in checkpoints/prior_3d_final.pth
echo 2. Integrate with your MPMARs POST estimation model
echo 3. Use the prior to guide 3D pose estimation
echo.
pause 