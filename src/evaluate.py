"""Evaluation script for the experiment."""
import torch
import logging
import sys
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/logs.txt'),
        logging.StreamHandler(sys.stdout)
    ]
)

def evaluate_model(model, test_loader):
    """
    Evaluate the trained model on the test set.
    """
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            predictions.extend(pred.numpy())
            targets.extend(target.numpy())
    
    accuracy = accuracy_score(targets, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(targets, predictions, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    logging.info("Test Set Metrics:")
    for metric_name, value in metrics.items():
        logging.info(f"{metric_name}: {value:.4f}")
    
    return metrics
