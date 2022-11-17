import torch
import torchvision
import torch.nn as nn

def resnet50():
    model = torchvision.models.resnet50()
    model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3,bias=False)
    model.fc = nn.Linear(2048, 12)
    return model

def resnet152():
    model = torchvision.models.resnet152()
    model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3,bias=False)
    model.fc = nn.Linear(2048, 12)
    return model