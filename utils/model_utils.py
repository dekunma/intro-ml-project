import models

name2model = {
    'resnet50': models.resnet50,
    'resnet152': models.resnet152,
    'depth_net': models.depth_net.DepthNet,
    'pose_resnet152': models.poseresnet152,
    'hrnet': models.get_hrnet,
    'resnest269': models.resnest.resnest269,
}

def get_model(model_name):
    if model_name not in name2model:
        raise ValueError('Model {} is not supported.'.format(model_name))
    model = name2model[model_name]()
    return model