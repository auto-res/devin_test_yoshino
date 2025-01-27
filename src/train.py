"""Training utilities for optimizer experiments."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import time
from pathlib import Path
import json
from tqdm import tqdm
import logging
from torch.utils.tensorboard import SummaryWriter

from .optimizers import NewOptimizer

def train_epoch(model: nn.Module,
               train_loader: DataLoader,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in tqdm(train_loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100. * correct / total
    }

def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Validation"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return {
        'loss': total_loss / len(val_loader),
        'accuracy': 100. * correct / total
    }

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer_name: str,
                optimizer_config: Dict[str, Any],
                training_config: Dict[str, Any],
                experiment_name: str,
                device: torch.device) -> Dict[str, Any]:
    """
    Train a model with specified optimizer and configuration.
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer_name: Name of the optimizer to use
        optimizer_config: Optimizer configuration
        training_config: Training configuration
        experiment_name: Name of the experiment
        device: Device to train on
    
    Returns:
        Dictionary containing training history and metrics
    """
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'new_optimizer':
        optimizer = NewOptimizer(model.parameters(), **optimizer_config)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_config)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    # Setup tensorboard
    writer = SummaryWriter(f'logs/runs/{experiment_name}')
    
    best_val_acc = 0
    patience_counter = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_times': []
    }
    
    for epoch in range(training_config['epochs']):
        epoch_start_time = time.time()
        
        # Training phase
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        if epoch % training_config['val_interval'] == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            
            # Log metrics
            writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
            writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('Accuracy/train', train_metrics['accuracy'], epoch)
            writer.add_scalar('Accuracy/val', val_metrics['accuracy'], epoch)
            
            
            # Save metrics to history
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            epoch_time = time.time() - epoch_start_time
            history['epoch_times'].append(epoch_time)
            
            # Early stopping check
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                patience_counter = 0
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': best_val_acc,
                }, f'models/{experiment_name}_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= training_config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Log to file
            with open('logs/logs.txt', 'a') as f:
                log_entry = (f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
                           f"{experiment_name} | "
                           f"Epoch {epoch}/{training_config['epochs']} | "
                           f"Train Loss: {train_metrics['loss']:.4f} | "
                           f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                           f"Val Loss: {val_metrics['loss']:.4f} | "
                           f"Val Acc: {val_metrics['accuracy']:.2f}%\n")
                f.write(log_entry)
    
    writer.close()
    return history
