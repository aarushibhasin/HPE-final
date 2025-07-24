"""
3D Pose Estimation Model for POST
Extends the existing 2D model to output both 2D heatmaps and 3D depth information
"""
import torch.nn as nn
from .resnet import _resnet
from .resnet import Bottleneck as Bottleneck_default


class Upsampling3D(nn.Sequential):
    """
    3-layers deconvolution for 3D pose estimation
    """
    def __init__(self, in_channel=2048, hidden_dims=(256, 256, 256), kernel_sizes=(4, 4, 4), bias=False):
        assert len(hidden_dims) == len(kernel_sizes), \
            'ERROR: len(hidden_dims) is different len(kernel_sizes)'

        layers = []
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            if kernel_size == 4:
                padding = 1
                output_padding = 0
            elif kernel_size == 3:
                padding = 1
                output_padding = 1
            elif kernel_size == 2:
                padding = 0
                output_padding = 0
            else:
                raise NotImplementedError("kernel_size is {}".format(kernel_size))

            layers.append(
                nn.ConvTranspose2d(
                    in_channels=in_channel,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=bias))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_channel = hidden_dim

        super(Upsampling3D, self).__init__(*layers)

        # init following Simple Baseline
        for name, m in self.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                if bias:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class PoseResNet3D(nn.Module):
    """
    3D Pose Estimation Model based on Simple Baseline
    Outputs both 2D heatmaps and 3D depth information

    Args:
        backbone (torch.nn.Module): Backbone to extract 2-d features from data
        upsampling (torch.nn.Module): Layer to upsample image feature to heatmap size
        feature_dim (int): The dimension of the features from upsampling layer.
        num_keypoints (int): Number of keypoints
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: False
    """
    def __init__(self, backbone, upsampling, feature_dim, num_keypoints, finetune=False):
        super(PoseResNet3D, self).__init__()
        self.backbone = backbone
        self.upsampling = upsampling
        
        # 2D heatmap head
        self.head_2d = nn.Conv2d(in_channels=feature_dim, out_channels=num_keypoints, 
                                kernel_size=1, stride=1, padding=0)
        
        # 3D depth head (relative depth for each keypoint)
        self.head_3d = nn.Conv2d(in_channels=feature_dim, out_channels=num_keypoints, 
                                kernel_size=1, stride=1, padding=0)
        
        self.finetune = finetune
        
        # Initialize heads
        for m in self.head_2d.modules():
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)
        for m in self.head_3d.modules():
            nn.init.normal_(m.weight, std=0.001)
            nn.init.constant_(m.bias, 0)

    def forward(self, x, intermediate=False):
        x_b = self.backbone(x)
        x_u = self.upsampling(x_b)
        
        # 2D heatmaps
        y_2d = self.head_2d(x_u)
        
        # 3D depth (relative depth offsets)
        y_3d = self.head_3d(x_u)
        
        if intermediate:
            return (y_2d, y_3d), x_b
        return y_2d, y_3d

    def get_parameters(self, lr=1.):
        return [
            {'params': self.backbone.parameters(), 'lr': 0.1 * lr if self.finetune else lr},
            {'params': self.upsampling.parameters(), 'lr': lr},
            {'params': self.head_2d.parameters(), 'lr': lr},
            {'params': self.head_3d.parameters(), 'lr': lr},
        ]


def _pose_resnet_3d(arch, num_keypoints, block, layers, pretrained_backbone, deconv_with_bias, finetune=False, progress=True, **kwargs):
    backbone = _resnet(arch, block, layers, pretrained_backbone, progress, **kwargs)
    upsampling = Upsampling3D(backbone.out_features, bias=deconv_with_bias)
    model = PoseResNet3D(backbone, upsampling, 256, num_keypoints, finetune)
    return model


def pose_resnet50_3d(num_keypoints, pretrained_backbone=True, deconv_with_bias=False, finetune=False, progress=True, **kwargs):
    """Constructs a 3D Simple Baseline model with a ResNet-50 backbone.

    Args:
        num_keypoints (int): number of keypoints
        pretrained_backbone (bool, optional): If True, returns a model pre-trained on ImageNet. Default: True.
        deconv_with_bias (bool, optional): Whether use bias in the deconvolution layer. Default: False
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: False
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default: True
    """
    return _pose_resnet_3d('resnet50', num_keypoints, Bottleneck_default, [3, 4, 6, 3], 
                          pretrained_backbone, deconv_with_bias, finetune, progress, **kwargs)


def pose_resnet101_3d(num_keypoints, pretrained_backbone=True, deconv_with_bias=False, finetune=False, progress=True, **kwargs):
    """Constructs a 3D Simple Baseline model with a ResNet-101 backbone.

    Args:
        num_keypoints (int): number of keypoints
        pretrained_backbone (bool, optional): If True, returns a model pre-trained on ImageNet. Default: True.
        deconv_with_bias (bool, optional): Whether use bias in the deconvolution layer. Default: False
        finetune (bool, optional): Whether use 10x smaller learning rate in the backbone. Default: False
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default: True
    """
    return _pose_resnet_3d('resnet101', num_keypoints, Bottleneck_default, [3, 4, 23, 3], 
                          pretrained_backbone, deconv_with_bias, finetune, progress, **kwargs) 