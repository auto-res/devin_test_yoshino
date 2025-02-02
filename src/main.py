import torch
import yaml
import json
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torchvision import datasets, transforms
from typing import Dict, Any

from optimizers import NewOptimizer
from preprocess import TextDataset, collate_variable_length
from train import train_model
from evaluate import evaluate_model
from src.models import AutoEncoder, ResNet32, LSTM


def get_mnist_loaders(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST('/tmp/data', train=True, download=True, transform=transform)
    val_size = len(train_dataset) // 10
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    test_dataset = datasets.MNIST('/tmp/data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def get_cifar_loaders(batch_size: int, dataset: str = 'cifar10') -> tuple[DataLoader, DataLoader, DataLoader]:
    Dataset = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.ToTensor()
    
    train_dataset = Dataset('/tmp/data', train=True, download=True, transform=transform_train)
    val_size = len(train_dataset) // 10
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    val_dataset.dataset.transform = transform_test
    test_dataset = Dataset('/tmp/data', train=False, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def get_model(config: Dict[str, Any]) -> torch.nn.Module:
    model_type = config['model']['type']
    
    if model_type == 'autoencoder':
        layers = config['model']['layers']
        model = AutoEncoder(layers)
    elif model_type == 'resnet32':
        num_classes = config['model']['num_classes']
        model = ResNet32(num_classes=num_classes)
    elif model_type == 'lstm':
        model = LSTM(
            hidden_size=config['model']['hidden_size'],
            num_layers=config['model']['num_layers'],
            embedding_size=config['model']['embedding_size'],
            dropout=config['model']['dropout']
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
        
    return model


def get_criterion(model_type: str) -> torch.nn.Module:
    if model_type == 'autoencoder':
        return torch.nn.MSELoss()
    else:
        return torch.nn.CrossEntropyLoss()


def get_scheduler(optimizer: torch.optim.Optimizer, config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    schedule_config = config['training']['lr_schedule']
    
    if schedule_config.get('type') == 'reduce_on_plateau':
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=schedule_config['factor'],
            patience=schedule_config['patience']
        )
    else:
        return MultiStepLR(
            optimizer,
            milestones=schedule_config['milestones'],
            gamma=schedule_config['gamma']
        )


def log_metrics(log_path: Path, epoch: str | int, metrics: Dict[str, Dict[str, float]]) -> None:
    if not log_path.exists():
        log_data = {}
    else:
        with open(log_path) as f:
            log_data = json.load(f)
    
    log_data[str(epoch)] = metrics
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)


def run_experiment(config_path: str | Path, optimizer_fn=None) -> Dict[str, Dict[str, float]]:
    config_path = Path(config_path) if isinstance(config_path, str) else config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs") / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config).to(device)
    optimizer = optimizer_fn(model.parameters()) if optimizer_fn else NewOptimizer(model.parameters(), **config['optimizer'])
    criterion = get_criterion(config['model']['type'])
    scheduler = get_scheduler(optimizer, config)
    
    if config['model']['type'] == 'autoencoder':
        train_loader, val_loader, test_loader = get_mnist_loaders(config['training']['batch_size'])
    elif config['model']['type'] == 'lstm':
        # Load text data and create dataloaders with variable sequence lengths
        with open(config['data']['text_path']) as f:
            text = f.read()
        
        train_size = int(len(text) * 0.8)
        val_size = int(len(text) * 0.1)
        train_text = text[:train_size]
        val_text = text[train_size:train_size + val_size]
        test_text = text[train_size + val_size:]
        
        train_dataset = TextDataset(train_text, config['data']['seq_length'])
        val_dataset = TextDataset(val_text, config['data']['seq_length'])
        test_dataset = TextDataset(test_text, config['data']['seq_length'])
        
        train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                                shuffle=True, collate_fn=collate_variable_length)
        val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                              collate_fn=collate_variable_length)
        test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'],
                               collate_fn=collate_variable_length)
    else:
        train_loader, val_loader, test_loader = get_cifar_loaders(
            config['training']['batch_size'],
            'cifar10' if config['model'].get('num_classes', 10) == 10 else 'cifar100'
        )
    
    metrics = train_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=config['training']['epochs'],
        save_dir=log_dir
    )
    
    test_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        model_path=log_dir / 'best_model.pt'
    )
    
    final_metrics = {'train': metrics, 'test': test_metrics}
    log_metrics(log_dir / 'metrics.json', 'final', final_metrics)
    return final_metrics


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <config_path>")
        sys.exit(1)
    run_experiment(sys.argv[1])
