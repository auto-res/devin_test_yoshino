"""Configuration for optimizer experiments."""

OPTIMIZER_CONFIGS = {
    'new_optimizer': {
        'lr': 1e-3,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'betas_aggmo': [0.0, 0.9, 0.99],
        'weight_decay': 1e-4
    },
    'adam': {
        'lr': 1e-3,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 1e-4
    },
    'sgd': {
        'lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 1e-4
    }
}

TRAINING_CONFIG = {
    'batch_size': 128,
    'epochs': 200,
    'early_stopping_patience': 20,
    'val_interval': 1
}

DATA_CONFIG = {
    'train_val_split': 0.8,
    'random_seed': 42,
    'num_workers': 4
}
