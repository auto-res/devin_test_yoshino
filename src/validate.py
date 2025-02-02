import torch
import yaml
from pathlib import Path
from datetime import datetime
from torch.optim import Adam
from typing import Dict, Any, List

from optimizers import NewOptimizer
from train import train_model
from evaluate import evaluate_model
from main import run_experiment


def run_validation(config_path: str | Path, num_trials: int = 3) -> Dict[str, List[Dict[str, float]]]:
    """Run validation comparing NewOptimizer against Adam and AggMo baselines."""
    config_path = Path(config_path)
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    
    optimizers = {
        'adam': lambda params: Adam(
            params,
            lr=base_config['optimizer']['lr'],
            betas=(base_config['optimizer']['beta1'], base_config['optimizer']['beta2']),
            eps=base_config['optimizer']['epsilon'],
            weight_decay=base_config['optimizer']['weight_decay']
        ),
        'aggmo': lambda params: NewOptimizer(
            params,
            lr=base_config['optimizer']['lr'],
            betas_aggmo=base_config['optimizer']['betas_aggmo'],
            weight_decay=base_config['optimizer']['weight_decay']
        ),
        'new_optimizer': lambda params: NewOptimizer(
            params,
            **base_config['optimizer']
        )
    }
    
    results = {name: [] for name in optimizers}
    
    for trial in range(num_trials):
        timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_trial{trial}")
        
        for opt_name, opt_fn in optimizers.items():
            # Create trial-specific config
            trial_config = base_config.copy()
            trial_config['optimizer']['name'] = opt_name
            trial_config['trial'] = trial
            
            # Save trial config
            trial_dir = Path("logs") / timestamp / opt_name
            trial_dir.mkdir(parents=True, exist_ok=True)
            with open(trial_dir / "config.yaml", 'w') as f:
                yaml.dump(trial_config, f)
            
            # Run experiment
            metrics = run_experiment(trial_dir / "config.yaml", opt_fn)
            results[opt_name].append(metrics)
    
    return results


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python validate.py <config_path>")
        sys.exit(1)
    
    results = run_validation(sys.argv[1])
    print("\nValidation Results:")
    for opt_name, trials in results.items():
        print(f"\n{opt_name.upper()}:")
        test_losses = [trial['test'].get('loss', float('inf')) for trial in trials]
        mean_loss = sum(test_losses) / len(test_losses)
        std_loss = torch.tensor(test_losses).std().item()
        
        if 'accuracy' in trials[0]['test']:
            test_accs = [trial['test']['accuracy'] for trial in trials]
            mean_acc = sum(test_accs) / len(test_accs)
            std_acc = torch.tensor(test_accs).std().item()
            print(f"Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")
            print(f"Test Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
        else:
            print(f"Test Loss: {mean_loss:.4f} ± {std_loss:.4f}")
