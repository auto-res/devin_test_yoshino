import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from . import CONFIG_DIR, LOGS_DIR
from .experiments.autoencoder import run_autoencoder_experiment
from .experiments.classification import run_classification_experiment
from .experiments.language_model import run_language_model_experiment

def get_optimizer_configs(base_config):
    """Get configurations for all optimizers to compare"""
    weight_decay = base_config['optim'].get('wdecay', 0.0)
    configs = {
        'adam': {
            'name': 'adam',
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': weight_decay
        },
        'aggmo': {
            'name': 'aggmo',
            'lr': 0.001,
            'betas': [0.0, 0.9, 0.99],
            'weight_decay': weight_decay
        },
        'new_optimizer': {
            'name': 'new_optimizer',
            'lr': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'betas_aggmo': [0.0, 0.9, 0.99],
            'weight_decay': weight_decay
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
        training_log = os.path.join(LOGS_DIR, f'{optimizer_name}_{timestamp}', 'training.log')
        if os.path.exists(training_log):
            epochs = []
            values = []
            with open(training_log, 'r') as f:
                for line in f:
                    if metric_name.lower() == 'mse' and 'Val Loss:' in line:
                        val_loss = float(line.split('Val Loss:')[1].strip())
                        values.append(val_loss)
                        epochs.append(len(epochs))
                    elif metric_name.lower() == 'accuracy' and 'Val Accuracy:' in line:
                        acc = float(line.split('Val Accuracy:')[1].strip().rstrip('%'))
                        values.append(acc)
                        epochs.append(len(epochs))
                    elif metric_name.lower() == 'perplexity' and 'Val Loss:' in line:
                        val_loss = float(line.split('Val Loss:')[1].strip())
                        perplexity = np.exp(val_loss)
                        values.append(perplexity)
                        epochs.append(len(epochs))
            if epochs and values:
                plt.plot(epochs, values, label=optimizer_name, marker='o', markersize=4)
    
    plt.title(f'{experiment_name} - {metric_name} Over Time')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set appropriate y-axis limits and scale for each metric
    if metric_name.lower() == 'perplexity':
        plt.yscale('log')
        plt.ylim(bottom=1)  # Perplexity cannot be less than 1
    elif metric_name.lower() == 'accuracy':
        plt.ylim(0, 100)
    elif metric_name.lower() == 'mse':
        plt.ylim(bottom=0)
    
    plt.tight_layout()
    save_path = os.path.join(LOGS_DIR, f'{timestamp}_{experiment_name}_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_validation(test_mode=True):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiments = ['autoencoder', 'classification', 'language_model']
    all_results = {}
    
    if test_mode:
        print("Running in test mode with reduced epochs...")
        test_config = {
            'autoencoder': 5,
            'classification': 5,
            'language_model': 5
        }
    
    for experiment in experiments:
        config_path = os.path.join(CONFIG_DIR, f'{experiment}.json')
        with open(config_path) as f:
            base_config = json.load(f)
            if test_mode:
                base_config['optim']['epochs'] = test_config[experiment]
        
        optimizer_configs = get_optimizer_configs(base_config)
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
    import sys
    test_mode = len(sys.argv) < 2 or sys.argv[1] != '--full'
    run_validation(test_mode)
