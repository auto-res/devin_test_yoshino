"""Training script for the experiment."""
import torch
import torch.nn as nn
import logging
import sys
from optimizer import NewOptimizer
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/logs.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)

def train_model(model, train_loader, val_loader, config):
    """
    Train the model using the hybrid Adam-AggMo optimizer.
    """
    optimizer = NewOptimizer(
        model.parameters(),
        lr=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        epsilon=config.epsilon,
        betas_aggmo=config.betas_aggmo,
        weight_decay=config.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(config.num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                logging.info(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.6f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        logging.info(f'Epoch: {epoch}, Validation Loss: {val_loss:.6f}')
        
        # Save model checkpoint
        torch.save(model.state_dict(), os.path.join(config.model_dir, f'model_epoch_{epoch}.pt'))
