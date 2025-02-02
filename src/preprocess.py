import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
import numpy as np


def create_sequences(text: str, seq_length: int, stride: int = 1) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Create input-target pairs for language modeling."""
    tokens = torch.tensor([ord(c) % 30000 for c in text], dtype=torch.long)  # Simple char-level tokenization
    sequences = []
    
    for i in range(0, len(tokens) - seq_length - 1, stride):
        input_seq = tokens[i:i + seq_length]
        target_seq = tokens[i + 1:i + seq_length + 1]
        sequences.append((input_seq, target_seq))
    
    return sequences


def collate_variable_length(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function for variable length sequences."""
    inputs, targets = zip(*batch)
    inputs = pad_sequence(inputs, batch_first=True)
    targets = pad_sequence(targets, batch_first=True)
    return inputs, targets


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, text: str, seq_length: int, stride: int = 1):
        self.sequences = create_sequences(text, seq_length, stride)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx]
