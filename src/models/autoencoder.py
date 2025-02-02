import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 1000), nn.ReLU(),
            nn.Linear(1000, 500), nn.ReLU(),
            nn.Linear(500, 250), nn.ReLU(),
            nn.Linear(250, 30)
        )
        self.decoder = nn.Sequential(
            nn.Linear(30, 250), nn.ReLU(),
            nn.Linear(250, 500), nn.ReLU(),
            nn.Linear(500, 1000), nn.ReLU(),
            nn.Linear(1000, 784), nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 784)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, 1, 28, 28)
