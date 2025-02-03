import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from ..optimizers import NewOptimizer
from .. import DATA_DIR

class SimpleTextDataset(Dataset):
    def __init__(self, sequence_length=35):
        # Generate simple synthetic data for testing
        vocab_size = 1000
        data_size = 10000
        self.data = torch.randint(0, vocab_size, (data_size,))
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length - 1
        
    def __getitem__(self, idx):
        return (
            self.data[idx:idx + self.sequence_length],
            self.data[idx + 1:idx + self.sequence_length + 1]
        )

class LSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            hidden_size, hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0
        )
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
        self.init_weights()
        
    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)
        
    def forward(self, text, hidden=None):
        emb = self.dropout(self.embedding(text))
        output, hidden = self.lstm(emb, hidden)
        output = self.dropout(output)
        decoded = self.decoder(output)
        return decoded, hidden

def batchify(data, batch_size, device):
    data = torch.tensor([t for t in data], dtype=torch.long)
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].reshape(-1)
    return data, target

def get_data_loaders(batch_size, bptt):
    train_dataset = SimpleTextDataset(sequence_length=bptt)
    val_dataset = SimpleTextDataset(sequence_length=bptt)
    test_dataset = SimpleTextDataset(sequence_length=bptt)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    vocab_size = 1000  # Same as in SimpleTextDataset
    
    return train_loader, val_loader, test_loader, vocab_size

def run_language_model_experiment(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_loader, val_loader, test_loader, vocab_size = get_data_loaders(
        config['optim']['batch_size'],
        config['optim']['bptt']
    )
    
    model = LSTM(
        vocab_size,
        config['model']['hidden_size'],
        config['model']['num_layers'],
        config['model']['dropout']
    ).to(device)
    
    from ..train import train_model
    from ..evaluate import evaluate_model
    
    best_metric = train_model(model, train_loader, val_loader, config)
    test_results = evaluate_model(model, test_loader, config)
    
    return {
        'best_val_metric': best_metric,
        'test_metrics': test_results
    }
