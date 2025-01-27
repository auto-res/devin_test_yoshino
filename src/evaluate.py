"""Evaluation utilities for optimizer experiments."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any
import json
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np

def evaluate_model(model: nn.Module,
                  test_loader: DataLoader,
                  device: torch.device) -> Dict[str, float]:
    """
    Evaluate a model on test data.
    
    Args:
        model: The neural network model
        test_loader: Test data loader
        device: Device to evaluate on
    
    Returns:
        Dictionary containing test metrics
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return {
        'test_loss': total_loss / len(test_loader),
        'test_accuracy': 100. * correct / total
    }

def plot_training_curves(histories: Dict[str, Dict[str, list]],
                        save_dir: str = 'logs/figures'):
    """
    Plot training curves for multiple optimizers.
    
    Args:
        histories: Dictionary containing training histories for different optimizers
        save_dir: Directory to save the plots
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    metrics = ['loss', 'accuracy']
    phases = ['train', 'val']
    
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for optimizer_name, history in histories.items():
            for phase in phases:
                key = f'{phase}_{metric}'
                if key in history:
                    plt.plot(history[key], 
                            label=f'{optimizer_name} ({phase})',
                            linestyle='-' if phase == 'train' else '--',
                            alpha=0.8)
        
        plt.title(f'{metric.capitalize()} vs. Epoch')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{save_dir}/{metric}_comparison.png')
        plt.close()

def plot_convergence_speed(histories: Dict[str, Dict[str, list]],
                          target_accuracy: float = 90.0,
                          save_dir: str = 'logs/figures'):
    """
    Plot epochs to reach target accuracy for different optimizers.
    
    Args:
        histories: Dictionary containing training histories for different optimizers
        target_accuracy: Target accuracy to measure convergence speed
        save_dir: Directory to save the plots
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    epochs_to_converge = {}
    for optimizer_name, history in histories.items():
        val_acc = history['val_acc']
        for epoch, acc in enumerate(val_acc):
            if acc >= target_accuracy:
                epochs_to_converge[optimizer_name] = epoch + 1
                break
        else:
            epochs_to_converge[optimizer_name] = len(val_acc)
    
    plt.figure(figsize=(8, 6))
    plt.bar(range(len(epochs_to_converge)), 
            list(epochs_to_converge.values()),
            tick_label=list(epochs_to_converge.keys()))
    plt.title(f'Epochs to Reach {target_accuracy}% Accuracy')
    plt.xlabel('Optimizer')
    plt.ylabel('Epochs')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/convergence_speed.png')
    plt.close()

def save_results(results: Dict[str, Any], 
                experiment_name: str,
                save_dir: str = 'logs/results'):
    """
    Save experimental results to JSON file.
    
    Args:
        results: Dictionary containing experimental results
        experiment_name: Name of the experiment
        save_dir: Directory to save the results
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f'{save_dir}/{experiment_name}.json', 'w') as f:
        json.dump(results, f, indent=4)
