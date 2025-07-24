"""
Modified based on torchvision.models.resnet.
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""

import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import BasicBlock, Bottleneck
import copy

# Mapping from arch string to torchvision weights class
RESNET_WEIGHTS = {
    'resnet18':  lambda: __import__('torchvision.models', fromlist=['ResNet18_Weights']).ResNet18_Weights.DEFAULT,
    'resnet34': lambda: __import__('torchvision.models', fromlist=['ResNet34_Weights']).ResNet34_Weights.DEFAULT,
    'resnet50': lambda: __import__('torchvision.models', fromlist=['ResNet50_Weights']).ResNet50_Weights.DEFAULT,
    'resnet101': lambda: __import__('torchvision.models', fromlist=['ResNet101_Weights']).ResNet101_Weights.DEFAULT,
    'resnet152': lambda: __import__('torchvision.models', fromlist=['ResNet152_Weights']).ResNet152_Weights.DEFAULT,
    'resnext50_32x4d': lambda: __import__('torchvision.models', fromlist=['ResNeXt50_32X4D_Weights']).ResNeXt50_32X4D_Weights.DEFAULT,
    'resnext101_32x8d': lambda: __import__('torchvision.models', fromlist=['ResNeXt101_32X8D_Weights']).ResNeXt101_32X8D_Weights.DEFAULT,
    'wide_resnet50_2': lambda: __import__('torchvision.models', fromlist=['Wide_ResNet50_2_Weights']).Wide_ResNet50_2_Weights.DEFAULT,
    'wide_resnet101_2': lambda: __import__('torchvision.models', fromlist=['Wide_ResNet101_2_Weights']).Wide_ResNet101_2_Weights.DEFAULT,
}

def load_pretrained_model(model, arch, progress=True):
    """Load pretrained weights for a model using torchvision >=0.13 weights API"""
    try:
        weights = RESNET_WEIGHTS[arch]()
        model.load_state_dict(weights.get_state_dict())
    except Exception as e:
        print(f"Warning: Could not load pretrained weights for {arch}: {e}")
    return model

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

class ResNet(models.ResNet):
    """ResNets without fully connected layer"""

    def __init__(self, *args, **kwargs):
        super(ResNet, self).__init__(*args, **kwargs)
        self._out_features = self.fc.in_features

    def forward(self, x):
        """"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = x.view(-1, self._out_features)
        return x

    @property
    def out_features(self) -> int:
        """The dimension of output features"""
        return self._out_features

    def copy_head(self) -> nn.Module:
        """Copy the origin fully connected layer"""
        return copy.deepcopy(self.fc)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        model = load_pretrained_model(model, arch, progress)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)

def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)

def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)

def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)

def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)

def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)

def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
