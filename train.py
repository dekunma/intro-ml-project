import os
import torch
import torch.nn as nn
from tqdm import tqdm

import argparse

from utils.model_utils import name2model
from utils.dataset_utils import get_dataset
import yaml

def main(model_name, dataroot, num_epochs=10, mode='head', resume=None, config=None):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # default config
    current_config = {
        'model_name': model_name,
        'experiment_name': model_name,
        'batch_size': 32,
        'lr': 1e-3,
        'lr_factor': 0.5,
        'lr_step': [30, 60, 90],
        'resume': resume,
        'num_epochs': num_epochs,
    }

    # load config
    if config is not None:
        config_file = yaml.load(open(config, 'r'), Loader=yaml.FullLoader)
        current_config.update(config_file)
        print('Using config file: {}'.format(config))
        print(current_config)
    
    model = name2model[current_config['model_name']]()
    model.to(device)

    dataset = get_dataset(current_config['model_name'], 'train', dataroot, mode)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=current_config['batch_size'], shuffle=True, num_workers=2)

    dataset_eval = get_dataset(model_name, 'train', dataroot, 'tail')
    data_loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=current_config['batch_size'], shuffle=False, num_workers=2)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=current_config['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, current_config['lr_step'], current_config['lr_factor'])
    loss_function = nn.MSELoss()

    log_path = f'logs/{current_config["experiment_name"]}'
    os.makedirs(log_path, exist_ok=True)
    log_file = open(f'{log_path}/log.txt', 'w')

    if current_config['resume'] is not None:
        print(f"Loading checkpoint from epoch {current_config['resume']}...")
        checkpoint = os.path.join(log_path, f"{current_config['experiment_name']}_{current_config['resume']}.pth")
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        print(f"Resumed from epoch {current_config['resume']}.")
    
    len_dataloader = len(data_loader)

    for epoch in range(current_config['num_epochs']):
        model.train()
        i = 0    
        epoch_loss = 0
        tqdm_bar = tqdm(data_loader, desc="Epoch {}/{}".format(epoch+1, current_config['num_epochs']))
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

        if (epoch + 1) % 5 == 0 or (epoch + 1) == current_config['num_epochs'] or epoch == 0:
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            }
            torch.save(save_dict, os.path.join(log_path, f"{current_config['model_name']}_{epoch+1}.pth"))


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
    parser.add_argument(
        "--config",
        default=None,
        type=str,
        help="Load config from a file",
    )
    args = parser.parse_args()
    model_name = args.model_name
    logpath = args.logpath
    dataroot = args.dataroot
    num_epochs = args.epoch
    mode = args.mode
    resume = args.resume
    config = args.config

    main(model_name=model_name, num_epochs=num_epochs, dataroot=dataroot, mode=mode, resume=resume, config=config)
