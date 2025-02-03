import argparse
import json
import os
from . import CONFIG_DIR
from .experiments import (
    run_autoencoder_experiment,
    run_classification_experiment,
    run_language_model_experiment
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    with open(os.path.join(CONFIG_DIR, args.config)) as f:
        config = json.load(f)
    
    if config['experiment'] == 'autoencoder':
        run_autoencoder_experiment(config)
    elif config['experiment'] == 'classification':
        run_classification_experiment(config)
    elif config['experiment'] == 'language_model':
        run_language_model_experiment(config)
    else:
        raise ValueError(f"Unknown experiment: {config['experiment']}")

if __name__ == '__main__':
    main()
