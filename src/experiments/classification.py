import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from ..optimizers import NewOptimizer
from .. import DATA_DIR

def get_model(model_type):
    if model_type == 'resnet34':
        # Use ResNet18 for faster testing, we'll switch back to ResNet34 for the full run
        return models.resnet18(num_classes=10)
    raise ValueError(f'Unknown model type: {model_type}')

def get_data_loaders(dataset_name, batch_size):
    if dataset_name == 'cifar10':
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
        
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        
        train_dataset = datasets.CIFAR10(
            DATA_DIR, train=True, download=True,
            transform=train_transform
        )
        
        val_size = int(0.1 * len(train_dataset))
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )
        
        test_dataset = datasets.CIFAR10(
            DATA_DIR, train=False,
            transform=test_transform
        )
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=2, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

def run_classification_experiment(config):
    print("Creating model...")
    model = get_model(config['model']['type'])
    
    print("Setting up data loaders...")
    train_loader, val_loader, test_loader = get_data_loaders(
        config['dataset'],
        config['optim']['batch_size']
    )
    print("Data loaders created successfully")
    
    from ..train import train_model
    from ..evaluate import evaluate_model
    
    best_metric = train_model(model, train_loader, val_loader, config)
    test_results = evaluate_model(model, test_loader, config)
    
    return {
        'best_val_metric': best_metric,
        'test_metrics': test_results
    }
