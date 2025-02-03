import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from ..optimizers import NewOptimizer
from .. import DATA_DIR

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.input_dim = input_dim
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x.view(-1, self.input_dim))

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dims):
        super().__init__()
        self.output_dim = output_dim
        layers = []
        prev_dim = hidden_dims[-1]
        for dim in reversed(hidden_dims[:-1]):
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU()
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims)
        self.decoder = Decoder(input_dim, hidden_dims)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)

def get_data_loaders(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = datasets.MNIST(
        DATA_DIR, train=True, download=True, transform=transform
    )
    
    val_size = int(0.1 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size]
    )
    
    test_dataset = datasets.MNIST(
        DATA_DIR, train=False, transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size
    )
    
    return train_loader, val_loader, test_loader

def run_autoencoder_experiment(config):
    model = AutoEncoder(
        input_dim=config['model']['input_dim'],
        hidden_dims=config['model']['hidden_dims']
    )
    
    train_loader, val_loader, test_loader = get_data_loaders(
        config['optim']['batch_size']
    )
    
    from ..train import train_model
    from ..evaluate import evaluate_model
    
    best_metric = train_model(model, train_loader, val_loader, config)
    test_results = evaluate_model(model, test_loader, config)
    
    return {
        'best_val_metric': best_metric,
        'test_metrics': test_results
    }
