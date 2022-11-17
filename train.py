import os
import torch
import torch.nn as nn
from tqdm import tqdm
from datasets.robot_hand import RobotHandDataset
import argparse

from utils.model_utils import name2model

def main(model_name, dataroot, num_epochs=10):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = name2model[model_name]()
    model.to(device)

    dataset = RobotHandDataset(split='train', dataroot=dataroot, mode='head')
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    dataset_eval = RobotHandDataset(split='train', dataroot=dataroot, mode='tail')
    data_loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=4)
        
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    len_dataloader = len(data_loader)

    for epoch in range(num_epochs):
        model.train()
        i = 0    
        epoch_loss = 0
        tqdm_bar = tqdm(data_loader, desc="Epoch {}/{}".format(epoch+1, num_epochs))
        for imgs, labels in tqdm_bar:
            i += 1
            imgs = imgs.to(device)
            labels = labels.to(device) * 1000
            output = model(imgs)
            loss = nn.MSELoss()(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            tqdm_bar.set_description(f'Iteration: {i}/{len_dataloader}, Loss: {loss}')
            epoch_loss += loss
        
        epoch_loss /= len_dataloader
        
        # get eval set loss
        with torch.no_grad():
            model.eval()
            eval_loss = 0
            print('Evaluating...')
            for imgs, labels in tqdm(data_loader_eval):
                imgs = imgs.to(device)
                labels = labels.to(device) * 1000
                output = model(imgs)
                loss = nn.MSELoss()(output, labels)
                eval_loss += loss
            
            eval_loss /= len(data_loader_eval)
        
        scheduler.step()
        log_text = f'Epoch: {epoch+1}, Loss: {epoch_loss}, Eval Loss: {eval_loss}'
        print(log_text)

        log_path = f'logs/{model_name}'
        os.makedirs(log_path, exist_ok=True)
        log_file = open(f'{log_path}/log.txt', 'w')
        log_file.write(log_text + '\n')

        torch.save(model.state_dict(), os.path.join(log_path, f'{model_name}_{epoch+1}.pth'))


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
        default='scratch/dm4524/data/robot_hand',
        type=str,
        help="path to the dataset root",
    )
    parser.add_argument(
        "--epoch",
        default=30,
        type=int,
        help="number of epochs",
    )
    args = parser.parse_args()
    model_name = args.model_name
    logpath = args.logpath
    dataroot = args.dataroot
    num_epochs = args.epoch

    main(model_name=model_name, num_epochs=num_epochs, dataroot=dataroot)
