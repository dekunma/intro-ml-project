import os
import torch
import torch.nn as nn
from tqdm import tqdm

import argparse

from utils.model_utils import name2model
from utils.dataset_utils import get_dataset

def main(model_name, dataroot, num_epochs=10, mode='head', resume=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = name2model[model_name]()
    model.to(device)

    batch_size = 32

    dataset = get_dataset(model_name, 'train', dataroot, mode)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    dataset_eval = get_dataset(model_name, 'train', dataroot, 'tail')
    data_loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=batch_size)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    loss_function = nn.MSELoss()

    log_path = f'logs/{model_name}'
    os.makedirs(log_path, exist_ok=True)
    log_file = open(f'{log_path}/log.txt', 'w')

    if resume is not None:
        print(f'Loading checkpoint from epoch {resume}...')
        checkpoint = os.path.join(log_path, f'{model_name}_{resume}.pth')
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f'Resumed from epoch {resume}.')
    
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
            loss = loss_function(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            tqdm_bar.set_description(f'Iteration: {i}/{len_dataloader}, Loss: {loss}')
            epoch_loss += loss
        
        epoch_loss /= len_dataloader
        
        # get eval set loss
        eval_loss = -1
        if mode != "full":
            with torch.no_grad():
                model.eval()
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

        log_file.write(log_text + '\n')

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs or epoch == 0:
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(save_dict, os.path.join(log_path, f'{model_name}_{epoch+1}.pth'))


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
        "--mode",
        default="head",
        type=str,
        help="[head / full]",
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=int,
        help="Resume from which epoch",
    )
    args = parser.parse_args()
    model_name = args.model_name
    logpath = args.logpath
    dataroot = args.dataroot
    num_epochs = args.epoch
    mode = args.mode
    resume = args.resume

    main(model_name=model_name, num_epochs=num_epochs, dataroot=dataroot, mode=mode, resume=resume)
