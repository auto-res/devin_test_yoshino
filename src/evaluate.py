import torch
import torch.nn.functional as F
import json
import os
import math
from typing import Dict, Any, Optional

def evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    model_type: str,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    checkpoint_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a model on test data.
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader containing test data
        model_type: One of ['autoencoder', 'cnn', 'lstm']
        device: Device to run evaluation on
        checkpoint_path: Optional path to model checkpoint
    
    Returns:
        Dictionary containing evaluation metrics
    """
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model = model.to(device)
    model.eval()
    
    total_metrics = {'loss': 0.0}
    num_batches = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if model_type == 'lstm':
                data, target = batch
                hidden = model.init_hidden(test_loader.batch_size, device)
                output, _ = model(data, hidden)
                loss = F.cross_entropy(output.view(-1, output.size(-1)), target)
                perplexity = math.exp(loss.item())
                total_metrics['perplexity'] = total_metrics.get('perplexity', 0.0) + perplexity
            else:
                data, target = batch
                data, target = data.to(device), target.to(device)
                
                if model_type == 'autoencoder':
                    output = model(data)
                    loss = F.mse_loss(output, data)
                    total_metrics['mse'] = total_metrics.get('mse', 0.0) + loss.item()
                else:  # CNN
                    output = model(data)
                    loss = F.cross_entropy(output, target)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct = pred.eq(target.view_as(pred)).sum().item()
                    total = target.size(0)
                    total_metrics['accuracy'] = total_metrics.get('accuracy', 0.0) + correct / total
            
            total_metrics['loss'] += loss.item()
            num_batches += 1
    
    # Average metrics
    metrics = {k: v / num_batches for k, v in total_metrics.items()}
    
    # Log results
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'logs.txt')
    
    with open(log_file, 'a') as f:
        if checkpoint_path:
            f.write(f"\nEvaluation results for {checkpoint_path}:\n")
        else:
            f.write("\nEvaluation results:\n")
        f.write(json.dumps(metrics, indent=2) + "\n")
    
    return metrics

if __name__ == '__main__':
    import argparse
    from models.autoencoder import Autoencoder
    from models.cnn import CNN5, ResNet32
    from models.lstm import LSTM
    from preprocess import load_mnist, load_cifar, load_ptb
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, required=True, choices=['autoencoder', 'cnn', 'lstm'])
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    
    # Load appropriate model and dataset
    if args.model_type == 'autoencoder':
        model = Autoencoder()
        _, _, test_loader = load_mnist(batch_size=args.batch_size)
    elif args.model_type == 'cnn':
        model = ResNet32()  # Using ResNet32 as default CNN
        _, _, test_loader = load_cifar(batch_size=args.batch_size)
    else:  # LSTM
        _, _, test_loader, vocab_info = load_ptb(batch_size=args.batch_size)
        model = LSTM(vocab_size=vocab_info['vocab_size'])
    
    metrics = evaluate_model(model, test_loader, args.model_type, checkpoint_path=args.checkpoint)
    print(json.dumps(metrics, indent=2))
