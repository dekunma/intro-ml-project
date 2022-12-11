import torch
import torch.nn as nn
from ray.tune.search.bayesopt import BayesOptSearch
from ray import tune

from utils import get_model, get_dataset, get_loss_fn

def train(ray_config):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # default config
    current_config = {
        'model_name': 'resnest269',
        'experiment_name': 'resnest269_ray',
        'loss_fn': 'mse',
    }

    model = get_model(current_config['model_name'])
    model.to(device)

    dataroot = '/scratch/dm4524/data/robot_hand'
    mode = 'head'

    dataset = get_dataset(current_config['model_name'], 'train', dataroot, mode)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=int(ray_config['batch_size']), shuffle=True, num_workers=2)

    dataset_eval = get_dataset(current_config['model_name'], 'train', dataroot, 'tail')
    data_loader_eval = torch.utils.data.DataLoader(dataset_eval, batch_size=int(ray_config['batch_size']), shuffle=False, num_workers=2)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=ray_config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, int(ray_config['lr_step']), ray_config['lr_factor'])
    loss_function = get_loss_fn(current_config['loss_fn'])
    
    len_dataloader = len(data_loader)

    while True:
        model.train()
        i = 0    
        epoch_loss = 0
        for imgs, labels in data_loader:
            i += 1
            imgs = imgs.to(device)
            labels = labels.to(device) * 1000
            output = model(imgs)
            loss = loss_function(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            epoch_loss += loss
        
        epoch_loss /= len_dataloader
        
        # get eval set loss
        eval_loss = -1
        if mode != "full":
            with torch.no_grad():
                model.eval()
                for imgs, labels in data_loader_eval:
                    imgs = imgs.to(device)
                    labels = labels.to(device) * 1000
                    output = model(imgs)
                    loss = nn.MSELoss()(output, labels)
                    eval_loss += loss
                
                eval_loss /= len(data_loader_eval)
        
        tune.report(loss=eval_loss)

        scheduler.step()


bayesopt = BayesOptSearch()

bayes_config = {
    "lr": tune.uniform(0.001,0.1),
    "batch_size": tune.uniform(16, 128),
    "lr_step": tune.uniform(10, 40),
    "lr_factor": tune.uniform(0.2, 0.8)
}

bayes_analysis = tune.run(
    train,
    metric="loss",
    mode="min",
    name="exp",
    stop={
        "training_iteration": 40
    },
    resources_per_trial={
        "gpu": 1,
        "cpu": 2
    },
    num_samples=30,
    config=bayes_config,
    verbose=2,
    search_alg=bayesopt)
