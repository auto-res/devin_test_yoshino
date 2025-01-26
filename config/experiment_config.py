"""Configuration for the hybrid Adam-AggMo optimizer experiments."""

class Config:
    # Optimizer parameters
    learning_rate = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    betas_aggmo = [0.0, 0.9, 0.99]
    weight_decay = 0.0

    # Training parameters
    batch_size = 128
    num_epochs = 100
    
    # Model parameters
    hidden_size = 256
    num_layers = 2
    
    # Data parameters
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1
    
    # Paths
    data_dir = "../data"
    model_dir = "../models"
    log_file = "../logs/logs.txt"
