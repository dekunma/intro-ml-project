from datasets.robot_hand import RobotHandDataset
from datasets.depth import RobotHandDepthDataset
from torch import nn
import models

def get_dataset(model_name, split, dataroot, mode):
    if model_name == 'depth_net':
        dataset = RobotHandDepthDataset(split, dataroot, mode=mode)
    else:
        dataset = RobotHandDataset(split, dataroot, mode=mode)
    
    return dataset

def get_model(model_name):
    name2model = {
        'resnet50': models.resnet50,
        'resnet152': models.resnet152,
        'depth_net': models.depth_net.DepthNet,
        'pose_resnet152': models.poseresnet152,
        'hrnet': models.get_hrnet,
        'resnest269': models.resnest.resnest269,
        'convnext': models.convnext.convnext_xlarge,
    }

    if model_name not in name2model:
        raise ValueError('Model {} is not supported.'.format(model_name))
    model = name2model[model_name]()
    return model

def get_loss_fn(loss_fn_name):
    name2lossfn = {
        'mse': nn.MSELoss(),
        'smoothl1': nn.SmoothL1Loss(),
    }

    if loss_fn_name not in name2lossfn:
        raise ValueError('Loss function {} is not supported.'.format(loss_fn_name))

    return name2lossfn[loss_fn_name]
