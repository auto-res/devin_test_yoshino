import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        
        # Encoder
        encoder_layers = []
        for i in range(len(layers) - 1):
            encoder_layers.extend([
                nn.Linear(layers[i], layers[i + 1]),
                nn.ReLU()
            ])
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        for i in range(len(layers) - 1, 0, -1):
            decoder_layers.extend([
                nn.Linear(layers[i], layers[i - 1]),
                nn.ReLU() if i > 1 else nn.Sigmoid()
            ])
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
