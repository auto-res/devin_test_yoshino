import torch
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
from pathlib import Path


def evaluate(model: torch.nn.Module,
            data_loader: DataLoader,
            criterion: torch.nn.Module,
            device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Handle different model types
            if isinstance(criterion, torch.nn.MSELoss):
                # Autoencoder case
                loss = criterion(output, target)
                total_loss += loss.item()
            else:
                # Classification case
                loss = criterion(output, target)
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    metrics = {"loss": total_loss / len(data_loader)}
    if not isinstance(criterion, torch.nn.MSELoss):
        metrics["accuracy"] = 100. * correct / total
        
    return metrics


def evaluate_model(model: torch.nn.Module,
                  test_loader: DataLoader,
                  criterion: torch.nn.Module,
                  device: torch.device,
                  model_path: Optional[Path] = None) -> Dict[str, float]:
    if model_path is not None and model_path.exists():
        model.load_state_dict(torch.load(model_path))
    
    return evaluate(model, test_loader, criterion, device)
