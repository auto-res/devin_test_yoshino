import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, embedding_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(30000, embedding_size)  # Vocabulary size of 30000
        self.embedding_dropout = nn.Dropout(dropout['embedding'])
        
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout['hidden'] if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout['input'])
        self.fc = nn.Linear(hidden_size, 30000)  # Output vocabulary size
        
    def forward(self, x, hidden=None):
        embedded = self.embedding_dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        decoded = self.fc(output)
        return decoded, hidden
        
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device),
                weight.new_zeros(self.num_layers, batch_size, self.hidden_size).to(device))
