"""Test script to verify the complete experimental pipeline."""

import torch
import logging
from pathlib import Path
from src.models import ResNet18
from src.optimizers import NewOptimizer
from src.preprocess import get_cifar10_loaders
from src.train import train_model
from src.evaluate import evaluate_model, plot_training_curves
from config.optimizer_config import OPTIMIZER_CONFIGS, TRAINING_CONFIG, DATA_CONFIG

def test_pipeline():
    """Verify the complete experimental pipeline with minimal data."""
    try:
        # Create required directories
        Path('data').mkdir(exist_ok=True)
        Path('logs').mkdir(exist_ok=True)
        Path('models').mkdir(exist_ok=True)
        Path('logs/figures').mkdir(parents=True, exist_ok=True)

        # Configure for quick test
        test_config = TRAINING_CONFIG.copy()
        test_config['epochs'] = 1  # Run only one epoch for testing
        test_config['batch_size'] = 32  # Smaller batch size

        # Initialize model and get data loaders
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNet18().to(device)
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            batch_size=test_config['batch_size'],
            num_workers=1
        )

        # Test training
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer_name='new_optimizer',
            optimizer_config=OPTIMIZER_CONFIGS['new_optimizer'],
            training_config=test_config,
            experiment_name='pipeline_test',
            device=device
        )

        # Test evaluation
        metrics = evaluate_model(model, test_loader, device)
        
        # Test plotting
        plot_training_curves(
            {'new_optimizer': history},
            save_dir='logs/figures'
        )

        print("✓ Pipeline test successful")
        print(f"Test metrics: {metrics}")
        return True

    except Exception as e:
        print(f"Error during pipeline test: {str(e)}")
        return False

if __name__ == "__main__":
    test_pipeline()
