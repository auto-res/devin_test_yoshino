import torch
from torch.utils.data import DataLoader
from typing import Dict, Any
from pathlib import Path


def train_epoch(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                train_loader: DataLoader,
                criterion: torch.nn.Module,
                device: torch.device) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return {"loss": total_loss / len(train_loader)}


def train_model(model: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                train_loader: DataLoader,
                val_loader: DataLoader,
                criterion: torch.nn.Module,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                device: torch.device,
                epochs: int,
                save_dir: Path) -> Dict[str, float]:
    best_val_loss = float('inf')
    metrics = {}
    
    for epoch in range(epochs):
        train_metrics = train_epoch(model, optimizer, train_loader, criterion, device)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        metrics[epoch] = {
            'train_loss': train_metrics['loss'],
            'val_loss': val_loss
        }
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir / 'best_model.pt')
            
        scheduler.step()
        
    return metrics
