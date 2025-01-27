"""Training utilities for optimizer experiments."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional
from .optimizers import NewOptimizer

def train_epoch(model: nn.Module,
               train_loader: DataLoader,
               criterion: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
    
    Returns:
        Dictionary containing training metrics
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
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
        'train_loss': total_loss / len(train_loader),
        'train_accuracy': 100. * correct / total
    }

def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: torch.device) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
    
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return {
        'val_loss': total_loss / len(val_loader),
        'val_accuracy': 100. * correct / total
    }

def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                optimizer_name: str,
                optimizer_config: Dict[str, Any],
                training_config: Dict[str, Any],
                experiment_name: str,
                device: torch.device) -> Dict[str, list]:
    """
    Train a model with the specified optimizer.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer_name: Name of the optimizer
        optimizer_config: Optimizer configuration
        training_config: Training configuration
        experiment_name: Name of the experiment
        device: Device to train on
    
    Returns:
        Dictionary containing training history
    """
    logger = logging.getLogger(__name__)
    writer = SummaryWriter(f'logs/runs/{experiment_name}')
    
    # Initialize optimizer
    if optimizer_name == 'new_optimizer':
        optimizer = NewOptimizer(model.parameters(), **optimizer_config)
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_config)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_config)
    else:
        raise ValueError(f'Unknown optimizer: {optimizer_name}')
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop setup
    epochs = training_config['epochs']
    early_stopping_patience = training_config['early_stopping_patience']
    val_interval = training_config['val_interval']
    
    best_val_acc = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_accuracy': [],
              'val_loss': [], 'val_accuracy': []}
    
    # Create directory for model checkpoints
    Path('models').mkdir(exist_ok=True)
    
    for epoch in range(epochs):
        # Training phase
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        for key, value in train_metrics.items():
            history[key].append(value)
            writer.add_scalar(f'{key}/{optimizer_name}', value, epoch)
        
        # Validation phase
        if epoch % val_interval == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            
            for key, value in val_metrics.items():
                history[key].append(value)
                writer.add_scalar(f'{key}/{optimizer_name}', value, epoch)
            
            # Early stopping check
            val_acc = val_metrics['val_accuracy']
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), f'models/{experiment_name}_best.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                logger.info(f'Early stopping triggered at epoch {epoch}')
                break
            
            logger.info(f'Epoch {epoch}: train_loss={train_metrics["train_loss"]:.4f}, '
                       f'train_acc={train_metrics["train_accuracy"]:.2f}%, '
                       f'val_loss={val_metrics["val_loss"]:.4f}, '
                       f'val_acc={val_metrics["val_accuracy"]:.2f}%')
    
    writer.close()
    return history
