import json
import os
from pathlib import Path

def modify_configs_for_testing():
    config_dir = Path('/home/ubuntu/devin_test_yoshino/config')
    configs = ['autoencoder.json', 'classification.json', 'language_model.json']
    
    for config_file in configs:
        config_path = config_dir / config_file
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        config['optim']['epochs'] = 2  # Just 2 epochs for testing
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

if __name__ == '__main__':
    modify_configs_for_testing()
