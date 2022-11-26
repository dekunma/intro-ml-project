from datasets.robot_hand import RobotHandDataset
from datasets.depth import RobotHandDepthDataset

def get_dataset(model_name, split, dataroot, mode):
    if model_name == 'depth_net':
        dataset = RobotHandDepthDataset(split, dataroot, mode=mode)
    else:
        dataset = RobotHandDataset(split, dataroot, mode=mode)
    
    return dataset