import os
import sys
import torch
import psutil
from torch.utils.data import DataLoader, TensorDataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.experiments.autoencoder import AutoEncoder, run_autoencoder_experiment
from src.experiments.classification import run_classification_experiment
from src.experiments.language_model import run_language_model_experiment
import json

def print_memory_usage():
    process = psutil.Process()
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

def test_autoencoder():
    print("\nTesting Autoencoder...")
    print_memory_usage()
    
    with open('config/autoencoder.json') as f:
        config = json.load(f)
    
    # Reduce batch size and epochs for quick test
    config['optim']['batch_size'] = 32
    config['optim']['epochs'] = 2
    
    try:
        results = run_autoencoder_experiment(config)
        print('Results:', results)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    print_memory_usage()

def test_classification():
    print("\nTesting Classification...")
    print_memory_usage()
    
    with open('config/classification.json') as f:
        config = json.load(f)
    
    # Reduce batch size and epochs for quick test
    config['optim']['batch_size'] = 32
    config['optim']['epochs'] = 2
    
    try:
        results = run_classification_experiment(config)
        print('Results:', results)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    print_memory_usage()

def test_language_model():
    print("\nTesting Language Model...")
    print_memory_usage()
    
    with open('config/language_model.json') as f:
        config = json.load(f)
    
    # Reduce batch size and epochs for quick test
    config['optim']['batch_size'] = 32
    config['optim']['epochs'] = 2
    
    try:
        results = run_language_model_experiment(config)
        print('Results:', results)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    print_memory_usage()

if __name__ == '__main__':
    print("Starting metric verification tests...")
    test_autoencoder()
    test_classification()
    test_language_model()
