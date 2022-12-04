import models

name2model = {
    'resnet50': models.resnet50,
    'resnet152': models.resnet152,
    'depth_net': models.depth_net.DepthNet,
    'pose_resnet152': models.poseresnet152,
    'hrnet': models.get_hrnet,
}