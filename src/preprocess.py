import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from typing import Tuple, Dict

def load_mnist(batch_size: int = 128, data_dir: str = 'data/mnist') -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load MNIST dataset for autoencoder experiments."""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    os.makedirs(data_dir, exist_ok=True)
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    val_dataset, test_dataset = torch.utils.data.random_split(
        torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform),
        [5000, 5000]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def load_cifar(batch_size: int = 128, data_dir: str = 'data/cifar', cifar100: bool = False) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load CIFAR-10/100 dataset for CNN experiments."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    os.makedirs(data_dir, exist_ok=True)
    
    dataset_class = torchvision.datasets.CIFAR100 if cifar100 else torchvision.datasets.CIFAR10
    
    train_dataset = dataset_class(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = dataset_class(
        root=data_dir, train=False, download=True, transform=transform_test
    )
    
    # Split test set into validation and test
    val_size = 5000
    test_size = len(test_dataset) - val_size
    val_dataset, test_dataset = torch.utils.data.random_split(test_dataset, [val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def load_ptb(batch_size: int = 20, data_dir: str = 'data/ptb', seq_length: int = 35) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    """Load Penn Treebank dataset for LSTM experiments."""
    from torchtext.datasets import PennTreebank
    from torchtext.data.utils import get_tokenizer
    from torchtext.vocab import build_vocab_from_iterator
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Download dataset
    train_iter = PennTreebank(split='train')
    val_iter = PennTreebank(split='valid')
    test_iter = PennTreebank(split='test')
    
    tokenizer = get_tokenizer('basic_english')
    
    def yield_tokens(data_iter):
        for text in data_iter:
            yield tokenizer(text)
    
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])
    
    def data_process(raw_text_iter):
        data = [torch.tensor([vocab[token] for token in tokenizer(item)], dtype=torch.long)
                for item in raw_text_iter]
        return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))
    
    train_data = data_process(train_iter)
    val_data = data_process(val_iter)
    test_data = data_process(test_iter)
    
    def batchify(data, batch_size):
        n_batch = data.size(0) // batch_size
        data = data[:n_batch * batch_size]
        data = data.view(batch_size, -1).t().contiguous()
        return data
    
    def get_batch(source, i, seq_length):
        seq_len = min(seq_length, len(source) - 1 - i)
        data = source[i:i + seq_len]
        target = source[i + 1:i + 1 + seq_len].reshape(-1)
        return data, target
    
    train_data = batchify(train_data, batch_size)
    val_data = batchify(val_data, batch_size)
    test_data = batchify(test_data, batch_size)
    
    class TextDataLoader:
        def __init__(self, data, batch_size, seq_length):
            self.data = data
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.i = 0
            self.n_batches = (len(data) - 1) // seq_length
        
        def __iter__(self):
            self.i = 0
            return self
        
        def __next__(self):
            if self.i >= self.n_batches * self.seq_length:
                raise StopIteration
            batch = get_batch(self.data, self.i, self.seq_length)
            self.i += self.seq_length
            return batch
    
    train_loader = TextDataLoader(train_data, batch_size, seq_length)
    val_loader = TextDataLoader(val_data, batch_size, seq_length)
    test_loader = TextDataLoader(test_data, batch_size, seq_length)
    
    return train_loader, val_loader, test_loader, {'vocab_size': len(vocab)}
