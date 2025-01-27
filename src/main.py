"""Main script for running optimizer experiments."""

import torch
import torch.nn as nn
from pathlib import Path
import json
import logging
from datetime import datetime

from .models import ResNet18
from .preprocess import get_cifar10_loaders
from .train import train_model
from .evaluate import evaluate_model, plot_training_curves, plot_convergence_speed, save_results
from config.optimizer_config import OPTIMIZER_CONFIGS, TRAINING_CONFIG, DATA_CONFIG

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/logs.txt'),
            logging.StreamHandler()
        ]
    )

def main():
    """Run the optimizer comparison experiments."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    Path('models').mkdir(exist_ok=True)
    Path('logs/figures').mkdir(parents=True, exist_ok=True)
    Path('logs/results').mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load data
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=TRAINING_CONFIG['batch_size'],
        num_workers=DATA_CONFIG['num_workers'],
        train_val_split=DATA_CONFIG['train_val_split']
    )
    
    # Dictionary to store results for each optimizer
    histories = {}
    test_results = {}
    
    # Run experiments for each optimizer
    for optimizer_name, optimizer_config in OPTIMIZER_CONFIGS.items():
        logger.info(f"\nStarting experiment with {optimizer_name}")
        
        # Initialize model
        model = ResNet18().to(device)
        
        # Create experiment name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{optimizer_name}_{timestamp}"
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_name=optimizer_name,
            optimizer_config=optimizer_config,
            training_config=TRAINING_CONFIG,
            experiment_name=experiment_name,
            device=device
        )
        
        # Evaluate on test set
        test_metrics = evaluate_model(model, test_loader, device)
        
        # Store results
        histories[optimizer_name] = history
        test_results[optimizer_name] = test_metrics
        
        logger.info(f"Test results for {optimizer_name}:")
        logger.info(f"Loss: {test_metrics['test_loss']:.4f}")
        logger.info(f"Accuracy: {test_metrics['test_accuracy']:.2f}%")
    
    # Generate plots
    plot_training_curves(histories)
    plot_convergence_speed(histories)
    
    # Save final results
    final_results = {
        'histories': histories,
        'test_results': test_results,
        'config': {
            'optimizers': OPTIMIZER_CONFIGS,
            'training': TRAINING_CONFIG,
            'data': DATA_CONFIG
        }
    }
    save_results(final_results, 'optimizer_comparison_final')
    
    logger.info("\nExperiment completed successfully!")

if __name__ == '__main__':
    main()
