import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights

# https://debuggercafe.com/advanced-facial-keypoint-detection-with-pytorch/
# class ResNet50(nn.Module):
#     def __init__(self, pretrained=False):
#         super().__init__()
#         if pretrained == True:
#             self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
#             for param in self.model.parameters():
#                 param.requires_grad = False
#             print('Freezing intermediate layer parameters...')
#         else:
#             self.model = resnet50()
#             for param in self.model.parameters():
#                 param.requires_grad = True
#             print('Training intermediate layer parameters...')

#         # first layer
#         self.conv0 = nn.Conv2d(12, 3, kernel_size=7, stride=2, padding=3, dilation=1, groups=1, bias=True)

#         # output layer
#         self.fc = nn.Linear(2048, 12)
#     def forward(self, x):
#         # get the batch size only, ignore (c, h, w)
#         batch, _, _, _ = x.shape
#         x = self.conv0(x)
#         x = self.model.features(x)
#         x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
#         x = self.fc(x)
#         return x

def resnet50():
    model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    # change input shape
    # https://discuss.pytorch.org/t/transfer-learning-usage-with-different-input-size/20744/13
    # first_conv_layer = nn.Conv2d(12, 3, kernel_size=1, stride=1, bias=True)
    # layers = list(model.children())[:-1]

    # model = nn.Sequential(first_conv_layer, *layers)
    # model.fc = nn.Linear(2048, 12)

    model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3,bias=False)
    model.fc = nn.Linear(2048, 12)

    return model

def resnet152():
    model = torchvision.models.resnet152()
    model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3,bias=False)
    model.fc = nn.Linear(2048, 12)
    return model