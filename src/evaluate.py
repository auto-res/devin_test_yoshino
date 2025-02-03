import torch
import torch.nn as nn
import math
from torch.utils.data import DataLoader

def compute_perplexity(loss):
    return math.exp(loss) if loss < 100 else float('inf')

def evaluate_model(model, test_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    criterion_config = config['optim']['criterion']
    criterion = nn.MSELoss() if criterion_config['tag'] == 'mse' else nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_metric = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)
            
            if criterion_config['tag'] == 'acc':
                pred = output.argmax(dim=1, keepdim=True)
                total_metric += pred.eq(target.view_as(pred)).sum().item()
            elif criterion_config['tag'] == 'perplexity':
                total_metric += compute_perplexity(loss.item()) * data.size(0)
            else:  # mse
                total_metric += loss.item() * data.size(0)
            
            total_samples += data.size(0)
    
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
