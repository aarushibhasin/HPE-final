@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
python train_human_prior_3d.py --source-root ./data/surreal_3d --target-root ./data/lsp --source SURREAL3D --target LSP --target-train LSP_mt --arch pose_resnet50_3d --subset-ratio 0.1 --epochs 70 --pretrain-epoch 40 --batch-size 32 --lr 0.001 --log logs/3d_pose
pause 