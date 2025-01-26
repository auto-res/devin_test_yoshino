"""Main script for running the complete experiment pipeline."""
import logging
import sys
import os
from preprocess import load_and_preprocess_data
from train import train_model
from evaluate import evaluate_model
from config.experiment_config import Config

def setup_logging():
    """Set up logging configuration."""
    os.makedirs('../logs', exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('../logs/logs.txt'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Run the complete experiment pipeline."""
    # Set up logging
    setup_logging()
    logging.info("Starting experiment...")
    
    # Create necessary directories
    os.makedirs(Config.data_dir, exist_ok=True)
    os.makedirs(Config.model_dir, exist_ok=True)
    
    # Load and preprocess data
    train_loader, val_loader, test_loader = load_and_preprocess_data(Config)
    
    # TODO: Define your model architecture here
    model = None  # Replace with actual model architecture
    
    # Train the model
    logging.info("Starting model training...")
    train_model(model, train_loader, val_loader, Config)
    
    # Evaluate the model
    logging.info("Evaluating model...")
    metrics = evaluate_model(model, test_loader)
    
    logging.info("Experiment completed successfully!")

if __name__ == "__main__":
    main()
