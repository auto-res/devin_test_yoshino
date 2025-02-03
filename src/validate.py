import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from . import CONFIG_DIR, LOGS_DIR
from .experiments import (
    run_autoencoder_experiment,
    run_classification_experiment,
    run_language_model_experiment
)

def get_optimizer_configs():
    """Get configurations for all optimizers to compare"""
    configs = {
        'adam': {
            'name': 'adam',
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.0
        },
        'aggmo': {
            'name': 'aggmo',
            'lr': 0.001,
            'betas': [0.0, 0.9, 0.99],
            'weight_decay': 0.0
        },
        'new_optimizer': {
            'name': 'new_optimizer',
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'betas_aggmo': [0.0, 0.9, 0.99],
            'weight_decay': 0.0
        }
    }
    
    return configs

def run_experiment_with_optimizer(experiment, config, optimizer_name, optimizer_config):
    """Run a single experiment with the specified optimizer"""
    config = config.copy()
    config['optim']['optimizer'] = optimizer_config
    return globals()[f'run_{experiment}_experiment'](config)

def plot_comparison(results, metric_name, experiment_name, timestamp):
    """Generate comparison plot for the experiment results"""
    plt.figure(figsize=(10, 6))
    
    for optimizer_name, metrics in results.items():
        if isinstance(metrics['test_metrics'][metric_name.lower()], (int, float)):
            plt.bar(optimizer_name, metrics['test_metrics'][metric_name.lower()])
    
    plt.title(f'{experiment_name} - {metric_name} Comparison')
    plt.ylabel(metric_name)
    
    save_path = os.path.join(LOGS_DIR, f'{timestamp}_{experiment_name}_comparison.png')
    plt.savefig(save_path)
    plt.close()

def run_validation():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiments = ['autoencoder', 'classification', 'language_model']
    optimizer_configs = get_optimizer_configs()
    all_results = {}
    
    for experiment in experiments:
        config_path = os.path.join(CONFIG_DIR, f'{experiment}.json')
        with open(config_path) as f:
            base_config = json.load(f)
        
        results = {}
        for opt_name, opt_config in optimizer_configs.items():
            print(f'Running {experiment} with {opt_name}...')
            results[opt_name] = run_experiment_with_optimizer(
                experiment, base_config, opt_name, opt_config
            )
        
        all_results[experiment] = results
        
        # Plot comparisons
        metric_map = {
            'autoencoder': 'MSE',
            'classification': 'Accuracy',
            'language_model': 'Perplexity'
        }
        plot_comparison(results, metric_map[experiment], experiment, timestamp)
    
    # Save numerical results
    results_path = os.path.join(LOGS_DIR, f'{timestamp}_validation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

if __name__ == '__main__':
    run_validation()
