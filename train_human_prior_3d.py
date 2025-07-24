"""
3D Pose Estimation Training Script
Extends the original POST training to handle 3D pose estimation
"""
import random
import time
import warnings
import sys
import argparse
import shutil
import os
import shutil
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToPILImage
import torch.nn.functional as F
import torchvision.transforms.functional as tF
import lib.models as models
from lib.models.loss_3d import Joints3DLoss, ConsLoss3D
import lib.datasets as datasets
import lib.transforms.keypoint_detection as T
from lib.transforms import Denormalize
from lib.data import ForeverDataIterator
from lib.meter import AverageMeter, ProgressMeter, AverageMeterDict, AverageMeterList
from lib.keypoint_detection import accuracy
from lib.logger import CompleteLogger
from utils import *
from prior.models_3d import PoseNDF3D, get_orientations_3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
recover_min = torch.tensor([-2.1179, -2.0357, -1.8044]).to(device)
recover_max = torch.tensor([2.2489, 2.4285, 2.64]).to(device)

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def main(args: argparse.Namespace):
    logger = CompleteLogger(args.log + '_' + args.arch, args.phase)

    logger.write(' '.join(f'{k}={v}' for k, v in vars(args).items()))
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    cudnn.benchmark = True

    # Data loading code
    normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    src_train_transform = T.Compose([
        T.RandomResizedCrop(size=args.image_size, scale=args.resize_scale),
        T.RandomAffineRotation(args.rotation_stu, args.shear_stu, args.translate_stu, args.scale_stu),
        T.ColorJitter(brightness=args.color_stu, contrast=args.color_stu, saturation=args.color_stu),
        T.GaussianBlur(high=args.blur_stu),
        T.ToTensor(),
        normalize
    ])
   
    base_transform = T.Compose([
        T.RandomResizedCrop(size=args.image_size, scale=args.resize_scale),
    ])
    tgt_train_transform_stu = T.Compose([
        T.RandomAffineRotation(args.rotation_stu, args.shear_stu, args.translate_stu, args.scale_stu),
        T.ColorJitter(brightness=args.color_stu, contrast=args.color_stu, saturation=args.color_stu),
        T.GaussianBlur(high=args.blur_stu),
        T.ToTensor(),
        normalize
    ])
    tgt_train_transform_tea = T.Compose([
        T.RandomAffineRotation(args.rotation_tea, args.shear_tea, args.translate_tea, args.scale_tea),
        T.ColorJitter(brightness=args.color_tea, contrast=args.color_tea, saturation=args.color_tea),
        T.GaussianBlur(high=args.blur_tea),
        T.ToTensor(),
        normalize
    ])
    val_transform = T.Compose([
        T.Resize(args.image_size),
        T.ToTensor(),
        normalize
    ])
    image_size = (args.image_size, args.image_size)
    heatmap_size = (args.heatmap_size, args.heatmap_size)
    
    # Load datasets
    source_dataset = datasets.__dict__[args.source]
    train_source_dataset = source_dataset(
        root=args.source_root, 
        transforms=src_train_transform,
        image_size=image_size, 
        heatmap_size=heatmap_size,
        subset_ratio=args.subset_ratio
    )
    train_source_loader = DataLoader(
        train_source_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=True
    )
    
    val_source_dataset = source_dataset(
        root=args.source_root, 
        split='test', 
        transforms=val_transform,
        image_size=image_size, 
        heatmap_size=heatmap_size,
        subset_ratio=args.subset_ratio
    )
    val_source_loader = DataLoader(
        val_source_dataset, 
        batch_size=args.test_batch, 
        shuffle=False, 
        pin_memory=True
    )

    target_dataset = datasets.__dict__[args.target_train]
    train_target_dataset = target_dataset(
        root=args.target_root, 
        transforms_base=base_transform,
        transforms_stu=tgt_train_transform_stu, 
        transforms_tea=tgt_train_transform_tea, 
        k=args.k, 
        image_size=image_size, 
        heatmap_size=heatmap_size
    )
    train_target_loader = DataLoader(
        train_target_dataset, 
        batch_size=args.batch_size,
        shuffle=True, 
        num_workers=args.workers, 
        pin_memory=True, 
        drop_last=True
    )
    
    target_dataset = datasets.__dict__[args.target]
    if args.target_val_root is None:
        val_target_dataset = target_dataset(
            root=args.target_root, 
            split='test', 
            transforms=val_transform,
            image_size=image_size, 
            heatmap_size=heatmap_size
        )
    else:
        val_target_dataset = target_dataset(
            root=args.target_val_root, 
            split='test', 
            transforms=val_transform,
            image_size=image_size, 
            heatmap_size=heatmap_size
        )
    val_target_loader = DataLoader(
        val_target_dataset, 
        batch_size=args.test_batch, 
        shuffle=False, 
        pin_memory=True
    )

    logger.write("Source train: {}".format(len(train_source_loader)))
    logger.write("Target train: {}".format(len(train_target_loader)))
    logger.write("Source test: {}".format(len(val_source_loader)))
    logger.write("Target test: {}".format(len(val_target_loader)))

    train_source_iter = ForeverDataIterator(train_source_loader)
    train_target_iter = ForeverDataIterator(train_target_loader)

    # Create 3D models
    student = models.__dict__[args.arch](num_keypoints=train_source_dataset.num_keypoints).cuda()
    teacher = models.__dict__[args.arch](num_keypoints=train_source_dataset.num_keypoints).cuda()
    prior = PoseNDF3D().cuda()
    
    # 3D loss functions
    criterion = Joints3DLoss(alpha_2d=args.alpha_2d, alpha_3d=args.alpha_3d)
    con_criterion = ConsLoss3D(alpha_2d=args.alpha_2d, alpha_3d=args.alpha_3d)

    if args.fix_head:
        for p in student.head_2d.parameters():
            p.requires_grad = False
        for p in student.head_3d.parameters():
            p.requires_grad = False
    if args.fix_upsample:
        for p in student.upsampling.parameters():
            p.requires_grad = False

    if args.SGD:
        stu_optimizer = SGD(student.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
    else:
        stu_optimizer = Adam(student.parameters(), lr=args.lr)

    tea_optimizer = OldWeightEMA(teacher, student, alpha=args.teacher_alpha)

    lr_scheduler = MultiStepLR(stu_optimizer, args.lr_step, args.lr_factor)

    student = torch.nn.DataParallel(student).cuda()
    teacher = torch.nn.DataParallel(teacher).cuda()

    # Load 3D prior
    if args.prior:
        prior_dict = torch.load(args.prior, map_location='cpu')['model']
        prior.load_state_dict(prior_dict)  
    prior = torch.nn.DataParallel(prior).cuda()
    for p in prior.parameters():
        p.requires_grad = False

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        student.load_state_dict(checkpoint['student'])
        teacher.load_state_dict(checkpoint['teacher'])
        stu_optimizer.load_state_dict(checkpoint['stu_optimizer'])
        tea_optimizer.load_state_dict(checkpoint['tea_optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    elif args.pretrain:
        pretrained_dict = torch.load(args.pretrain, map_location='cpu')['student']
        model_dict = student.state_dict()
        # remove keys from pretrained dict that doesn't appear in model dict
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        student.load_state_dict(pretrained_dict, strict=False)
        teacher.load_state_dict(pretrained_dict, strict=False)

    # define visualization function
    tensor_to_image = Compose([
        Denormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToPILImage()
    ])

    def visualize(image, keypoint2d, name):
        """
        Args:
            image (tensor): image in shape 3 x H x W
            keypoint2d (tensor): keypoints in shape K x 2
            name: name of the saving image
        """
        train_source_dataset.visualize(tensor_to_image(image),
                                       keypoint2d, logger.get_image_path("{}.jpg".format(name)))

    if args.phase == 'test':
        # evaluate on validation set
        source_val_acc = validate_3d(val_source_loader, teacher, criterion, None, args)
        target_val_acc = validate_3d(val_target_loader, teacher, criterion, visualize, args)

        logger.write("Source: {:4.3f} Target: {:4.3f}".format(source_val_acc['all'], target_val_acc['all']))
        return

    # training
    for epoch in range(start_epoch, args.epochs):
        if epoch < args.pretrain_epoch:
            pretrain_3d(train_source_iter, student, criterion, stu_optimizer, epoch, visualize, args)
        else:
            train_sf_3d(train_target_iter, student, teacher, prior, con_criterion,
                       stu_optimizer, tea_optimizer, epoch, args)

        lr_scheduler.step()

        # evaluate on validation set
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            source_val_acc = validate_3d(val_source_loader, teacher, criterion, None, args)
            target_val_acc = validate_3d(val_target_loader, teacher, criterion, visualize, args)

            logger.write("Source: {:4.3f} Target: {:4.3f}".format(source_val_acc['all'], target_val_acc['all']))

        # save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            logger.save_model({
                'epoch': epoch,
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'stu_optimizer': stu_optimizer.state_dict(),
                'tea_optimizer': tea_optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))

    logger.close()


def pretrain_3d(train_source_iter, student, criterion, stu_optimizer, epoch: int, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_all = AverageMeter('Loss (all)', ":.4e")
    losses_2d = AverageMeter('Loss (2d)', ":.4e")
    losses_3d = AverageMeter('Loss (3d)', ":.4e")

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_all, losses_2d, losses_3d],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    student.train()

    end = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for i in range(args.iters_per_epoch):
        stu_optimizer.zero_grad()
        x_s, (target_2d, target_3d), target_weight, meta_s = next(train_source_iter)

        x_s = x_s.to(device)
        target_2d = target_2d.to(device)
        target_3d = target_3d.to(device)
        target_weight = target_weight.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        with torch.cuda.amp.autocast():
            # Forward pass
            output_2d, output_3d = student(x_s)
            
            # Compute 3D loss
            loss_all, loss_2d, loss_3d = criterion(output_2d, output_3d, target_2d, target_3d, target_weight)

        scaler.scale(loss_all).backward()
        scaler.step(stu_optimizer)
        scaler.update()

        # record loss
        losses_all.update(loss_all, x_s.size(0))
        losses_2d.update(loss_2d, x_s.size(0))
        losses_3d.update(loss_3d, x_s.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def train_sf_3d(train_target_iter, student, teacher, prior, con_criterion,
               stu_optimizer, tea_optimizer, epoch: int, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':4.2f')
    data_time = AverageMeter('Data', ':3.1f')
    losses_all = AverageMeter('Loss (all)', ":.4e")
    losses_c = AverageMeter('Loss (c)', ":.4e")
    losses_b = AverageMeter('Loss (b)', ":.4e")
    losses_p = AverageMeter('Loss (p)', ":.4e")

    progress = ProgressMeter(
        args.iters_per_epoch,
        [batch_time, data_time, losses_all, losses_c, losses_b, losses_p],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    student.train()
    teacher.train()

    end = time.time()
    scaler = torch.cuda.amp.GradScaler()

    for i in range(args.iters_per_epoch):
        stu_optimizer.zero_grad()
        x_t_stu, (target_stu_2d, target_stu_3d), target_weight_stu, meta_t_stu, x_t_teas, targets_tea_2d, targets_tea_3d, target_weights_tea, meta_t_tea = next(train_target_iter)

        x_t_stu = x_t_stu.to(device)
        x_t_teas = [x_t_tea.to(device) for x_t_tea in x_t_teas]
        x_t_teas_ori = [x_t_tea.clone() for x_t_tea in x_t_teas]

        # measure data loading time
        data_time.update(time.time() - end)

        ratio = args.image_size / args.heatmap_size
        ratio_f = args.image_size / 8

        with torch.no_grad():
            # Teacher predictions (2D + 3D)
            y_t = [teacher(x_t_tea, intermediate=True) for x_t_tea in x_t_teas]
            y_t_teas_2d = [y_t[j][0][0] for j in range(len(y_t))]  # 2D heatmaps
            y_t_teas_3d = [y_t[j][0][1] for j in range(len(y_t))]  # 3D depth maps
            f_t_teas = [y_t[j][1] for j in range(len(y_t))]  # features
            
            # Reconstruct teacher predictions (similar to 2D version)
            y_t_tea_recon_2d = torch.zeros_like(y_t_teas_2d[0]).cuda()
            y_t_tea_recon_3d = torch.zeros_like(y_t_teas_3d[0]).cuda()
            f_t_tea_recon = torch.zeros_like(f_t_teas[0]).cuda()
            tea_mask = torch.zeros(y_t_teas_2d[0].shape[:2]).cuda()
            
            for ind in range(x_t_teas[0].size(0)):
                recons_2d = torch.zeros(args.k, *y_t_teas_2d[0].size()[1:])
                recons_3d = torch.zeros(args.k, *y_t_teas_3d[0].size()[1:])
                recons_f = torch.zeros(args.k, *f_t_teas[0].size()[1:])
                
                for _k in range(args.k):
                    angle, [trans_x, trans_y], [shear_x, shear_y], scale = meta_t_tea[_k]['aug_param_tea']
                    _angle, _trans_x, _trans_y, _shear_x, _shear_y, _scale = angle[ind].item(), trans_x[ind].item(), trans_y[ind].item(), shear_x[ind].item(), shear_y[ind].item(), scale[ind].item() 
                    
                    # Reconstruct 2D
                    temp_2d = tF.affine(y_t_teas_2d[_k][ind], 0., translate=[_trans_x/ratio, _trans_y/ratio], shear=[0., 0.], scale=1.)
                    temp_2d = tF.affine(temp_2d, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                    temp_2d = tF.affine(temp_2d, 0., translate=[0, 0], shear=[_shear_x, _shear_y], scale=1.)
                    recons_2d[_k] = temp_2d
                    
                    # Reconstruct 3D
                    temp_3d = tF.affine(y_t_teas_3d[_k][ind], 0., translate=[_trans_x/ratio, _trans_y/ratio], shear=[0., 0.], scale=1.)
                    temp_3d = tF.affine(temp_3d, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                    temp_3d = tF.affine(temp_3d, 0., translate=[0, 0], shear=[_shear_x, _shear_y], scale=1.)
                    recons_3d[_k] = temp_3d
                    
                    # Reconstruct features
                    temp_f = tF.affine(f_t_teas[_k][ind], 0., translate=[_trans_x/ratio_f, _trans_y/ratio_f], shear=[0., 0.], scale=1.)
                    temp_f = tF.affine(temp_f, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                    temp_f = tF.affine(temp_f, 0., translate=[0, 0], shear=[_shear_x, _shear_y], scale=1.)
                    recons_f[_k] = temp_f

                y_t_tea_recon_2d[ind] = torch.mean(recons_2d, dim=0)
                y_t_tea_recon_3d[ind] = torch.mean(recons_3d, dim=0)
                f_t_tea_recon[ind] = torch.mean(recons_f, dim=0)
                tea_mask[ind] = 1.

        with torch.cuda.amp.autocast():
            # Student predictions (2D + 3D)
            y_t_stu, f_t_stu = student(x_t_stu, intermediate=True)
            y_t_stu_2d, y_t_stu_3d = y_t_stu

            # 3D prior evaluation
            ori_stu_3d = get_orientations_3d(y_t_stu_2d, y_t_stu_3d)
            prior_score_stu = prior(ori_stu_3d)
            loss_p = prior_score_stu.mean()

            # Reconstruct student predictions
            y_t_stu_recon_2d = torch.zeros_like(y_t_stu_2d).cuda()
            y_t_stu_recon_3d = torch.zeros_like(y_t_stu_3d).cuda()
            f_t_stu_recon = torch.zeros_like(f_t_stu).cuda()
            
            for ind in range(x_t_stu.size(0)):
                angle, [trans_x, trans_y], [shear_x, shear_y], scale = meta_t_stu['aug_param_stu']
                _angle, _trans_x, _trans_y, _shear_x, _shear_y, _scale = angle[ind].item(), trans_x[ind].item(), trans_y[ind].item(), shear_x[ind].item(), shear_y[ind].item(), scale[ind].item() 
                
                # Reconstruct 2D
                temp_2d = tF.affine(y_t_stu_2d[ind], 0., translate=[_trans_x/ratio, _trans_y/ratio], shear=[0., 0.], scale=1.)
                temp_2d = tF.affine(temp_2d, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                y_t_stu_recon_2d[ind] = tF.affine(temp_2d, 0., translate=[0., 0.], shear=[_shear_x, _shear_y], scale=1.)
                
                # Reconstruct 3D
                temp_3d = tF.affine(y_t_stu_3d[ind], 0., translate=[_trans_x/ratio, _trans_y/ratio], shear=[0., 0.], scale=1.)
                temp_3d = tF.affine(temp_3d, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                y_t_stu_recon_3d[ind] = tF.affine(temp_3d, 0., translate=[0., 0.], shear=[_shear_x, _shear_y], scale=1.)
                
                # Reconstruct features
                temp_f = tF.affine(f_t_stu[ind], 0., translate=[_trans_x/ratio_f, _trans_y/ratio_f], shear=[0., 0.], scale=1.)
                temp_f = tF.affine(temp_f, _angle, translate=[0., 0.], shear=[0., 0.], scale=_scale)
                f_t_stu_recon[ind] = tF.affine(temp_f, 0., translate=[0., 0.], shear=[_shear_x, _shear_y], scale=1.)

            # Teacher mask
            activates = y_t_tea_recon_2d.amax(dim=(2,3))
            y_t_tea_recon_2d = rectify(y_t_tea_recon_2d, sigma=args.sigma)
            mask_thresh = torch.kthvalue(activates.view(-1), int(args.mask_ratio * activates.numel()))[0].item()
            tea_mask = tea_mask * (activates > mask_thresh)

            # Barlow twins loss
            f1 = F.adaptive_avg_pool2d(f_t_stu_recon, output_size=1)
            f2 = F.adaptive_avg_pool2d(f_t_tea_recon, output_size=1)
            z_a_norm = (f1 - f1.mean(0)) / f1.std(0)
            z_b_norm = (f2 - f2.mean(0)) / f2.std(0)
            z_a_norm = z_a_norm.squeeze()
            z_b_norm = z_b_norm.squeeze()
            
            # cross-correlation matrix
            c = (z_a_norm.T @ z_b_norm) / f1.shape[0]
            on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
            off_diag = off_diagonal(c).pow_(2).sum()
            loss_b = 1e-3 * (on_diag + 5e-3 * off_diag)
            
            # 3D consistency loss
            loss_c = con_criterion(y_t_stu_recon_2d, y_t_stu_recon_3d, y_t_tea_recon_2d, y_t_tea_recon_3d, tea_mask=tea_mask)

        # Total loss
        if epoch < args.step_p:
             loss_all = args.lambda_c * loss_c + 0.5*args.lambda_b * loss_b + 0.*args.lambda_p * loss_p
        else:
             loss_all = args.lambda_c * loss_c + args.lambda_b * loss_b + args.lambda_p * loss_p

        scaler.scale(loss_all).backward()
        scaler.step(stu_optimizer)
        tea_optimizer.step()
        scaler.update()

        # record loss
        losses_all.update(loss_all, x_t_stu.size(0))
        losses_c.update(loss_c, x_t_stu.size(0))
        losses_b.update(loss_b, x_t_stu.size(0))
        losses_p.update(loss_p, x_t_stu.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate_3d(val_loader, model, criterion, visualize, args: argparse.Namespace):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.2e')
    acc_2d = AverageMeterList(list(range(val_loader.dataset.num_keypoints)), ":3.2f", ignore_val=-1)
    acc_3d = AverageMeterList(list(range(val_loader.dataset.num_keypoints)), ":3.2f", ignore_val=-1)
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses], 
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (x, (target_2d, target_3d), target_weight, meta) in enumerate(val_loader):
            x = x.to(device)
            target_2d = target_2d.to(device)
            target_3d = target_3d.to(device)
            target_weight = target_weight.to(device)

            # compute output
            output_2d, output_3d = model(x)
            loss_all, loss_2d, loss_3d = criterion(output_2d, output_3d, target_2d, target_3d, target_weight)

            # measure accuracy and record loss
            losses.update(loss_all.item(), x.size(0))
            
            # 2D accuracy
            acc_per_points_2d, avg_acc_2d, cnt_2d, pred_2d = accuracy(output_2d.cpu().numpy(), target_2d.cpu().numpy())
            acc_2d.update(acc_per_points_2d, x.size(0))
            
            # 3D accuracy (simplified - you can implement more sophisticated 3D metrics)
            # For now, we'll use the same accuracy function on 3D depth maps
            acc_per_points_3d, avg_acc_3d, cnt_3d, pred_3d = accuracy(output_3d.cpu().numpy(), target_3d.cpu().numpy())
            acc_3d.update(acc_per_points_3d, x.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    return {'all': acc_2d.avg[0], '2d': acc_2d.avg[0], '3d': acc_3d.avg[0]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D Pose Estimation Training')
    
    # Dataset arguments
    parser.add_argument('--source-root', default='./data/surreal_3d', help='source domain dataset root')
    parser.add_argument('--target-root', default='./data/lsp', help='target domain dataset root')
    parser.add_argument('--target-val-root', default=None, help='target domain validation dataset root')
    parser.add_argument('--source', default='SURREAL3D', help='source domain dataset')
    parser.add_argument('--target', default='LSP', help='target domain dataset')
    parser.add_argument('--target-train', default='LSP_mt', help='target domain training dataset')
    parser.add_argument('--subset-ratio', type=float, default=0.1, help='subset ratio for SURREAL dataset')
    
    # Model arguments
    parser.add_argument('--arch', default='pose_resnet50_3d', help='model architecture')
    parser.add_argument('--pretrain', default='', help='pretrained model path')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=70, help='number of epochs')
    parser.add_argument('--pretrain-epoch', type=int, default=40, help='pretrain epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--test-batch', type=int, default=32, help='test batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr-step', type=int, nargs='+', default=[50, 60], help='lr step')
    parser.add_argument('--lr-factor', type=float, default=0.1, help='lr factor')
    parser.add_argument('--SGD', action='store_true', help='use SGD optimizer')
    parser.add_argument('--teacher-alpha', type=float, default=0.999, help='teacher EMA alpha')
    
    # Loss weights
    parser.add_argument('--alpha-2d', type=float, default=1.0, help='2D loss weight')
    parser.add_argument('--alpha-3d', type=float, default=1.0, help='3D loss weight')
    parser.add_argument('--lambda-c', type=float, default=1.0, help='consistency loss weight')
    parser.add_argument('--lambda-b', type=float, default=1e-3, help='barlow twins loss weight')
    parser.add_argument('--lambda-p', type=float, default=1e-6, help='prior loss weight')
    parser.add_argument('--step-p', type=int, default=47, help='epoch to start prior loss')
    
    # Data augmentation
    parser.add_argument('--image-size', type=int, default=256, help='image size')
    parser.add_argument('--heatmap-size', type=int, default=64, help='heatmap size')
    parser.add_argument('--resize-scale', type=float, nargs='+', default=[0.5, 1.0], help='resize scale')
    parser.add_argument('--rotation-stu', type=int, default=60, help='student rotation')
    parser.add_argument('--rotation-tea', type=int, default=60, help='teacher rotation')
    parser.add_argument('--shear-stu', type=int, nargs='+', default=[-30, 30], help='student shear')
    parser.add_argument('--shear-tea', type=int, nargs='+', default=[-30, 30], help='teacher shear')
    parser.add_argument('--translate-stu', type=float, nargs='+', default=[0.05, 0.05], help='student translation')
    parser.add_argument('--translate-tea', type=float, nargs='+', default=[0.05, 0.05], help='teacher translation')
    parser.add_argument('--scale-stu', type=float, nargs='+', default=[0.6, 1.3], help='student scale')
    parser.add_argument('--scale-tea', type=float, nargs='+', default=[0.6, 1.3], help='teacher scale')
    parser.add_argument('--color-stu', type=float, default=0.25, help='student color jitter')
    parser.add_argument('--color-tea', type=float, default=0.25, help='teacher color jitter')
    parser.add_argument('--blur-stu', type=float, default=0, help='student blur')
    parser.add_argument('--blur-tea', type=float, default=0, help='teacher blur')
    
    # Training settings
    parser.add_argument('--k', type=int, default=1, help='number of teacher views')
    parser.add_argument('--mask-ratio', type=float, default=0.5, help='mask ratio')
    parser.add_argument('--sigma', type=float, default=2, help='gaussian sigma')
    parser.add_argument('--fix-head', action='store_true', help='fix head parameters')
    parser.add_argument('--fix-upsample', action='store_true', help='fix upsampling parameters')
    parser.add_argument('--source-free', action='store_true', help='source-free adaptation')
    
    # Prior
    parser.add_argument('--prior', default='', help='prior model path')
    
    # Other
    parser.add_argument('--log', default='logs/3d_pose', help='log directory')
    parser.add_argument('--phase', default='train', help='train or test')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--iters-per-epoch', type=int, default=1000, help='iterations per epoch')
    parser.add_argument('--print-freq', type=int, default=100, help='print frequency')
    parser.add_argument('--eval-freq', type=int, default=10, help='evaluation frequency')
    parser.add_argument('--save-freq', type=int, default=10, help='save frequency')
    
    args = parser.parse_args()
    main(args) 