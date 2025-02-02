import torch
import json
import os
import argparse
from typing import Dict, Any
import matplotlib.pyplot as plt

from models.autoencoder import Autoencoder
from models.cnn import CNN5, ResNet32
from models.lstm import LSTM
from optimizers.adam import Adam
from optimizers.aggmo import AggMo
from optimizers.new_optimizer import NewOptimizer
from preprocess import load_mnist, load_cifar, load_ptb
from train import train_model
from evaluate import evaluate_model

def load_config(model_type: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    config_path = os.path.join('config', f'{model_type}.json')
    with open(config_path, 'r') as f:
        return json.load(f)

def setup_model_and_data(config: Dict[str, Any]):
    """Setup model and data loaders based on configuration."""
    model_type = config['model_type']
    batch_size = config['batch_size']
    
    if model_type == 'autoencoder':
        model = Autoencoder()
        train_loader, val_loader, test_loader = load_mnist(batch_size=batch_size)
    elif model_type == 'cnn':
        model = ResNet32()
        train_loader, val_loader, test_loader = load_cifar(batch_size=batch_size)
    else:  # LSTM
        train_loader, val_loader, test_loader, vocab_info = load_ptb(
            batch_size=batch_size,
            seq_length=config.get('seq_length', 35)
        )
        model = LSTM(
            vocab_size=vocab_info['vocab_size'],
            **config.get('model_config', {})
        )
    
    return model, train_loader, val_loader, test_loader

def get_optimizer(name: str, params, config: Dict[str, Any]):
    """Create optimizer instance based on name and config."""
    opt_config = config['optimizers'][name]
    if name == 'adam':
        return Adam(params, **opt_config)
    elif name == 'aggmo':
        return AggMo(params, **opt_config)
    else:  # new_optimizer
        return NewOptimizer(params, **opt_config)

def plot_results(results: Dict[str, Dict[str, float]], model_type: str):
    """Plot comparison of optimizer performances."""
    metric_key = {
        'autoencoder': 'mse',
        'cnn': 'accuracy',
        'lstm': 'perplexity'
    }[model_type]
    
    plt.figure(figsize=(10, 6))
    for optimizer_name, metrics in results.items():
        plt.plot(metrics[metric_key], label=optimizer_name)
    
    plt.xlabel('Epoch')
    plt.ylabel(metric_key.capitalize())
    plt.title(f'{model_type.upper()} - {metric_key.capitalize()} vs Epoch')
    plt.legend()
    
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/{model_type}_{metric_key}_comparison.png')
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True,
                       choices=['autoencoder', 'cnn', 'lstm'])
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.model_type)
    
    # Set random seeds
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['seed'])
    
    # Setup model and data
    model, train_loader, val_loader, test_loader = setup_model_and_data(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train and evaluate with each optimizer
    results = {}
    for opt_name in ['adam', 'aggmo', 'new_optimizer']:
        print(f"\nTraining with {opt_name}...")
        
        # Create optimizer
        optimizer = get_optimizer(opt_name, model.parameters(), config)
        
        # Train model
        train_model(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device
        )
        
        # Load best model and evaluate
        checkpoint_path = os.path.join('models', config['model_type'], 'best_model.pt')
        test_metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            model_type=config['model_type'],
            device=device,
            checkpoint_path=checkpoint_path
        )
        
        results[opt_name] = test_metrics
        print(f"{opt_name} test metrics:", test_metrics)
    
    # Plot results
    plot_results(results, config['model_type'])

if __name__ == '__main__':
    main()
