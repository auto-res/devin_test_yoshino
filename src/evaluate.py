import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .experiments.autoencoder import AutoEncoder
from torch.utils.data import DataLoader

def batch_mean_mse(recon, inputs):
    return torch.sum(torch.mean((recon - inputs) ** 2, 0))

def compute_perplexity(loss):
    return math.exp(loss) if loss < 100 else float('inf')

def evaluate_model(model, test_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    criterion_config = config['optim']['criterion']
    
    total_loss = 0.0
    total_metric = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            batch_size = data.size(0)
            
            if isinstance(model, AutoEncoder):
                # For autoencoder, input is the target
                target = data.clone()
                data = data.view(batch_size, -1)  # Flatten input
                target = target.view(batch_size, -1)  # Flatten target
                
                reconstruction = model(data)
                mse = batch_mean_mse(reconstruction, target)
                loss = F.binary_cross_entropy(reconstruction, target)
                
                total_loss += loss.item() * batch_size
                total_metric += mse.item() * batch_size
            else:
                output = model(data)
                if criterion_config['tag'] == 'acc':
                    loss = F.cross_entropy(output, target)
                    pred = output.argmax(dim=1, keepdim=True)
                    total_metric += pred.eq(target.view_as(pred)).sum().item()
                elif criterion_config['tag'] == 'perplexity':
                    loss = F.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
                    total_metric += compute_perplexity(loss.item()) * batch_size
                
                total_loss += loss.item() * batch_size
            
            total_samples += batch_size
    
    avg_loss = total_loss / total_samples
    
    if criterion_config['tag'] == 'acc':
        metric = 100. * total_metric / total_samples
        metric_name = 'Accuracy'
    elif criterion_config['tag'] == 'perplexity':
        metric = total_metric / total_samples
        metric_name = 'Perplexity'
    else:  # mse
        metric = total_metric / total_samples
        metric_name = 'MSE'
    
    results = {
        'loss': avg_loss,
        metric_name.lower(): metric
    }
    
    return results
