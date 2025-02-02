import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import math
import os
import json
from typing import Dict, Any, Optional, Tuple

def setup_logging(log_dir: str = 'logs'):
    """Setup logging directory and create log file."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'logs.txt')
    return log_file

def log_metrics(metrics: Dict[str, float], epoch: int, split: str, log_file: str):
    """Log metrics to file."""
    log_str = f"Epoch {epoch} - {split} metrics: {json.dumps(metrics)}\n"
    with open(log_file, 'a') as f:
        f.write(log_str)

def compute_metrics(model_type: str, output: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute metrics based on model type."""
    if model_type == 'autoencoder':
        mse = F.mse_loss(output, target).item()
        return {'mse': mse}
    elif model_type == 'cnn':
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = target.size(0)
        accuracy = correct / total
        return {'accuracy': accuracy}
    elif model_type == 'lstm':
        loss = F.cross_entropy(output.view(-1, output.size(-1)), target)
        perplexity = math.exp(loss.item())
        return {'perplexity': perplexity, 'loss': loss.item()}
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    model_type: str,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_metrics = {'loss': 0.0}
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        if model_type == 'lstm':
            data, target = batch
            hidden = model.init_hidden(train_loader.batch_size, device)
        else:
            data, target = batch
            data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        
        if model_type == 'lstm':
            output, hidden = model(data, hidden)
            loss = F.cross_entropy(output.view(-1, output.size(-1)), target)
        elif model_type == 'autoencoder':
            output = model(data)
            loss = F.mse_loss(output, data)
        else:  # CNN
            output = model(data)
            loss = F.cross_entropy(output, target)
        
        loss.backward()
        optimizer.step()
        
        metrics = compute_metrics(model_type, output.detach(), target)
        for k, v in metrics.items():
            total_metrics[k] = total_metrics.get(k, 0.0) + v
        total_metrics['loss'] += loss.item()
        num_batches += 1
    
    return {k: v / num_batches for k, v in total_metrics.items()}

def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    model_type: str,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on data loader."""
    model.eval()
    total_metrics = {'loss': 0.0}
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if model_type == 'lstm':
                data, target = batch
                hidden = model.init_hidden(data_loader.batch_size, device)
            else:
                data, target = batch
                data, target = data.to(device), target.to(device)
            
            if model_type == 'lstm':
                output, hidden = model(data, hidden)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), target)
            elif model_type == 'autoencoder':
                output = model(data)
                loss = F.mse_loss(output, data)
            else:  # CNN
                output = model(data)
                loss = F.cross_entropy(output, target)
            
            metrics = compute_metrics(model_type, output, target)
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
            total_metrics['loss'] += loss.item()
            num_batches += 1
    
    return {k: v / num_batches for k, v in total_metrics.items()}

def train_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
) -> None:
    """Train a model with given optimizer and data loaders."""
    # Setup
    torch.manual_seed(config.get('seed', 42))
    model = model.to(device)
    log_file = setup_logging()
    model_dir = os.path.join('models', config['model_type'])
    os.makedirs(model_dir, exist_ok=True)
    
    # Training loop
    best_val_metric = float('inf')
    metric_key = {'autoencoder': 'mse', 'cnn': 'accuracy', 'lstm': 'perplexity'}[config['model_type']]
    patience = config.get('patience', 5)
    patience_counter = 0
    
    for epoch in range(1, config.get('epochs', 100) + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, config['model_type'], device)
        log_metrics(train_metrics, epoch, 'train', log_file)
        
        # Validate
        val_metrics = evaluate(model, val_loader, config['model_type'], device)
        log_metrics(val_metrics, epoch, 'val', log_file)
        
        # Save checkpoint if best
        val_metric = val_metrics[metric_key]
        is_best = (metric_key == 'accuracy' and val_metric > best_val_metric) or \
                 (metric_key != 'accuracy' and val_metric < best_val_metric)
        
        if is_best:
            best_val_metric = val_metric
            patience_counter = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }
            torch.save(checkpoint, os.path.join(model_dir, 'best_model.pt'))
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch} epochs")
            break
