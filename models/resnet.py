'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''

# %% import 

# if __name__ == "__main__":
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torch.autograd import Variable
from src.models.gradcam import GradCAM, blur_with_mask
import numpy as np

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, name=''):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.name = name
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet10(name, args):
    return ResNet(BasicBlock, [1, 1, 1, 1], num_classes=args.num_classes,
                  name=name)


def ResNet12(name, args):
    return ResNet(BasicBlock, [2, 1, 1, 1], num_classes=args.num_classes,
                  name=name)


def ResNet14(name, args):
    return ResNet(BasicBlock, [2, 2, 1, 1], num_classes=args.num_classes,
                  name=name)


def ResNet16(name, args):
    return ResNet(BasicBlock, [2, 2, 2, 1], num_classes=args.num_classes,
                  name=name)


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes
                  )


def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)


def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


class ResNetEncoder(nn.Module):

    def __init__(self, base_model, out_dim, loss=None, include_mlp = True, entropy=False):
        super(ResNetEncoder, self).__init__()

        print("_" * 20)
        print("ResNetEncoder")
        print("_" * 20)
        print(f"base_model: {base_model}")
        print(f"out_dim: {out_dim}")
        print(f"loss: {loss}")
        print(f"include_mlp: {include_mlp}")
        print(f"entropy: {entropy}")
        print("_" * 20)

        # Check if the base model is valid
        if base_model not in ["resnet18", "resnet34", "resnet50"]:
            raise ValueError(f"Invalid base model: {base_model}. Choose from 'resnet18', 'resnet34', or 'resnet50'.")
        if not isinstance(out_dim, int) or out_dim <= 0:
            raise ValueError(f"out_dim should be a positive integer. Got {out_dim} instead.")
        # Check if loss is None and include_mlp is True


        self.resnet_dict = {"resnet18": ResNet18(num_classes=out_dim),
                            "resnet34": ResNet34(num_classes=out_dim),
                            "resnet50": ResNet50(num_classes=out_dim)} # Dont use ResNet50
        self.backbone = self._get_basemodel(base_model)
        self.loss = loss
        self.entropy = entropy
        dim_mlp = self.backbone.fc.in_features # 512
    
        if include_mlp:
            print(f"MLP Included: From {dim_mlp} dim to {out_dim} dim")
            # add mlp projection head
            if self.loss == "symmetrized":
                self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                 nn.BatchNorm1d(dim_mlp),
                                                 nn.ReLU(inplace=True), self.backbone.fc)
            else:
                self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        else:
            self.backbone.fc = nn.Identity() # no head used

        print("Backbone:", self.backbone)
        print("FC: Present inside backbone.")

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise Exception(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet34 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)


class ResNetGradCamEncoder(ResNetEncoder):
    def __init__(self, base_model, out_dim, loss=None, include_mlp = True, entropy=False):
        super(ResNetGradCamEncoder, self).__init__(base_model, out_dim, loss, include_mlp, entropy)
        self.gradcam = GradCAM(self.backbone, self.backbone.layer4[-1])

    def forward(self, x):

        cams = []
        for xi in x:
            xi = xi.unsqueeze(0)
            cam, _ = self.gradcam.generate_cam(xi)
            cam = cam.squeeze().detach().cpu().numpy()
            cams.append(cam)
        cam = np.stack(cams, axis=0)
        self.gradcam.remove_hooks()

        # Convert to numpy array 
        x_np = x.cpu().numpy()

        blurred_x = []
        for x_np_i, cam_i in zip(x_np, cam):
            blurred_x_i = blur_with_mask(x_np_i, cam_i)
            blurred_x.append(blurred_x_i)
        blurred_x = np.stack(blurred_x, axis=0)
        blurred_x = torch.from_numpy(blurred_x).to(x.device)

        blurred_x = blurred_x.permute(0, 3, 1, 2)
        return self.backbone(blurred_x)


class ResNetTorchEncoder(nn.Module):

    def __init__(self, base_model, out_dim, loss=None, include_mlp = True, entropy=False):
        super(ResNetTorchEncoder, self).__init__()

        print("_" * 20)
        print("ResNetTorchEncoder")
        print("_" * 20)
        print(f"base_model: {base_model}")
        print(f"out_dim: {out_dim}")
        print(f"loss: {loss}")
        print(f"include_mlp: {include_mlp}")
        print(f"entropy: {entropy}")
        print("_" * 20)
        # Check if the base model is valid
        if base_model not in ["resnet18", "resnet34", "resnet50"]:
            raise ValueError(f"Invalid base model: {base_model}. Choose from 'resnet18', 'resnet34', or 'resnet50'.")
        if not isinstance(out_dim, int) or out_dim <= 0:
            raise ValueError(f"out_dim should be a positive integer. Got {out_dim} instead.")
        # Check if loss is None and include_mlp is True


        self.resnet_dict = {"resnet18": models.resnet18(weights=None, num_classes=out_dim),
                            "resnet34": models.resnet34(weights=None, num_classes=out_dim),
                            "resnet50": models.resnet50(weights=None, num_classes=out_dim)} # Dont use ResNet50
        self.backbone = self._get_basemodel(base_model)
        self.loss = loss
        self.entropy = entropy
        dim_mlp = self.backbone.fc.in_features # 512
    
        if include_mlp:
            print(f"MLP Included: From {dim_mlp} dim to {out_dim} dim")
            # add mlp projection head
            if self.loss == "symmetrized":
                self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
                                                 nn.BatchNorm1d(dim_mlp),
                                                 nn.ReLU(inplace=True), self.backbone.fc)
            else:
                self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)
        else:
            self.backbone.fc = nn.Identity() # no head used

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise Exception(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18, resnet34 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
    
# %% test
if __name__ == "__main__":

    resnetGradCamEncoder = ResNetGradCamEncoder("resnet18", 10, include_mlp=True)
    print(resnetGradCamEncoder)