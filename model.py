import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import vgg11, mobilenet_v2, mobilenet_v3_large
from torchvision.models.resnet import resnet34
from typing import Optional


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet34().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(nn.Linear(512, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, feature_dim, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class Classifier(nn.Module):
    def __init__(self, num_class, pretrained_path:Optional[str]=None):
        super(Classifier, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.fc = nn.Linear(512, num_class, bias=True)
        if pretrained_path is not None:
            self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


class TwoLayerClassifier(nn.Module):
    def __init__(self, num_class, pretrained_path):
        super(TwoLayerNet, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.cls = nn.Sequential(
            nn.Linear(512, 512, bias=True),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_class, bias=True)
        )
        self.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.cls(feature)
        return out


def create_mobilenet_v2(num_classes: int=10):
    # Reference https://yul.hatenablog.com/entry/2018/08/27/162423
    config = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 1],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]
    model_instance = mobilenet_v2(num_classes=num_classes, inverted_residual_setting=config)
    model_instance.features[0] = torchvision.ops.Conv2dNormActivation(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    return model_instance

def StudentModel(num_classes: int=10, model: str='mobilenet_v2'):
    if model == 'mobilenet_v3':
        return mobilenet_v3_large(num_classes=num_classes)
    elif model == 'mobilenet_v2':
        return create_mobilenet_v2(num_classes)
    elif model == 'vgg':
        return vgg11(num_classes=num_classes)
    else:
        return create_mobilenet_v2(num_classes)
