import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
import json
import logging
from datetime import datetime
from . import LOGS_DIR, MODELS_DIR

def get_experiment_name(config):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    optim_name = config['optim']['optimizer']['name']
    return f"{optim_name}_{timestamp}"

def get_optimizer(config, parameters):
    from .optimizers import NewOptimizer, AggMo
    import torch.optim as optim
    
    optim_config = config['optim']['optimizer']
    name = optim_config['name']
    
    if name == 'new_optimizer':
        return NewOptimizer(
            parameters,
            lr=optim_config['lr'],
            beta1=optim_config['beta1'],
            beta2=optim_config['beta2'],
            epsilon=optim_config['epsilon'],
            betas_aggmo=optim_config['betas_aggmo'],
            weight_decay=optim_config['weight_decay']
        )
    elif name == 'aggmo':
        return AggMo(
            parameters,
            lr=optim_config['lr'],
            betas=optim_config['betas'],
            weight_decay=optim_config['weight_decay']
        )
    elif name == 'adam':
        return optim.Adam(
            parameters,
            lr=optim_config['lr'],
            betas=optim_config['betas'],
            eps=optim_config['eps'],
            weight_decay=optim_config['weight_decay']
        )
    else:
        raise ValueError(f'Unknown optimizer: {name}')

def get_scheduler(config, optimizer):
    schedule_config = config['optim']['lr_schedule']
    return MultiStepLR(
        optimizer,
        milestones=schedule_config['milestones'],
        gamma=schedule_config['lr_decay'],
        last_epoch=schedule_config['last_epoch']
    )

def train_model(model, train_loader, val_loader, config):
    experiment_name = get_experiment_name(config)
    log_path = os.path.join(LOGS_DIR, experiment_name)
    model_path = os.path.join(MODELS_DIR, experiment_name)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = get_optimizer(config, model.parameters())
    scheduler = get_scheduler(config, optimizer)
    
    criterion_config = config['optim']['criterion']
    criterion = nn.MSELoss() if criterion_config['tag'] == 'mse' else nn.CrossEntropyLoss()
    
    best_metric = float('inf') if criterion_config['minmax'] == 'min' else float('-inf')
    patience = config['optim'].get('patience', 20)
    patience_counter = 0
    
    logging.basicConfig(
        filename=os.path.join(log_path, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s'
    )
    
    for epoch in range(config['optim']['epochs']):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_metric = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                if criterion_config['tag'] == 'acc':
                    pred = output.argmax(dim=1, keepdim=True)
                    val_metric += pred.eq(target.view_as(pred)).sum().item()
        
        val_loss /= len(val_loader)
        if criterion_config['tag'] == 'acc':
            val_metric = 100. * val_metric / len(val_loader.dataset)
        else:
            val_metric = val_loss
            
        scheduler.step()
        
        logging.info(f'Epoch: {epoch}')
        logging.info(f'Train Loss: {train_loss:.6f}')
        logging.info(f'Val Loss: {val_loss:.6f}')
        if criterion_config['tag'] == 'acc':
            logging.info(f'Val Accuracy: {val_metric:.2f}%')
        
        metric = val_metric
        if criterion_config['minmax'] == 'min':
            metric = -metric
            
        if (criterion_config['minmax'] == 'min' and -metric < best_metric) or \
           (criterion_config['minmax'] == 'max' and metric > best_metric):
            best_metric = metric if criterion_config['minmax'] == 'max' else -metric
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metric': best_metric,
                'config': config
            }, os.path.join(model_path, 'best_model.pt'))
            logging.info(f'Saved new best model with metric: {best_metric:.6f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered after {epoch} epochs')
                break
    
    return best_metric
