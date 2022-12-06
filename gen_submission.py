import pandas as pd
import torch
from tqdm import tqdm
import argparse
import os

from utils.model_utils import name2model
from utils.dataset_utils import get_dataset

def main(model_name, epoch, dataroot, model_path):
    dataset = get_dataset(model_name, 'test', dataroot=dataroot, mode='full')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    model = name2model[model_name]()
    model.load_state_dict(torch.load(os.path.join('logs', model_path, f'{model_name}_{epoch}.pth'))['model_state_dict'])
    
    log_base_dir = os.path.join('submissions', model_path)
    os.makedirs(log_base_dir, exist_ok=True)
    outfile = os.path.join(log_base_dir, f'{model_path}_epoch_{epoch}.csv')
    outfile = open(outfile, 'w')

    titles = ['ID', 'FINGER_POS_1', 'FINGER_POS_2', 'FINGER_POS_3', 'FINGER_POS_4', 'FINGER_POS_5', 'FINGER_POS_6',
            'FINGER_POS_7', 'FINGER_POS_8', 'FINGER_POS_9', 'FINGER_POS_10', 'FINGER_POS_11', 'FINGER_POS_12']
    preds = []

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for image_input, label in tqdm(data_loader):
            image_input = image_input.to(device)

            output = model(image_input.to(device)) / 1000
            preds.append(output[0].cpu().detach().numpy())


    df = pd.concat([pd.DataFrame(dataset.field_ids), pd.DataFrame.from_records(preds)], axis = 1, names = titles)
    df.columns = titles
    df.to_csv(outfile, index = False)
    print("Written to csv file {}".format(outfile))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default=None,
        type=str,
        help="",
    )
    parser.add_argument(
        "--logpath",
        default='logs',
        type=str,
        help="path to store the log file",
    )
    parser.add_argument(
        "--dataroot",
        default='/scratch/dm4524/data/robot_hand',
        type=str,
        help="path to the dataset root",
    )
    parser.add_argument(
        "--epoch",
        default=30,
        type=int,
        help="number of epochs",
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        help="path to model dir in logs dir",
    )
    args = parser.parse_args()
    model_name = args.model_name
    logpath = args.logpath
    dataroot = args.dataroot
    epoch = args.epoch
    model_path = args.model_path

    main(model_name=model_name, epoch=epoch, dataroot=dataroot, model_path=model_path)