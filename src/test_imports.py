"""Test script to verify all components are properly implemented and importable."""

import torch
import logging
from pathlib import Path

def test_imports():
    """Verify all components can be imported and basic functionality works."""
    try:
        # Test imports
        from src.models import ResNet18
        from src.optimizers import NewOptimizer
        from src.preprocess import get_cifar10_loaders
        from src.train import train_model
        from src.evaluate import evaluate_model, plot_training_curves
        
        # Test basic component initialization
        model = ResNet18()
        optimizer = NewOptimizer(model.parameters())
        
        print("✓ All imports successful")
        print("✓ Components can be initialized")
        return True
        
    except Exception as e:
        print(f"Error during import verification: {str(e)}")
        return False

if __name__ == "__main__":
    test_imports()
