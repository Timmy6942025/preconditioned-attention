import torch
from torch.utils.data import DataLoader, Dataset


class CopyTaskDataset(Dataset):
    """Copy task: learn to reproduce input sequence.

    Input: random token sequence [t1, t2, ..., tN]
    Target: same sequence [t1, t2, ..., tN]
    """

    def __init__(self, vocab_size: int = 64, seq_len: int = 32, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.data[idx]


class ReverseTaskDataset(Dataset):
    """Reverse task: learn to reverse input sequence.

    Input: [t1, t2, ..., tN]
    Target: [tN, tN-1, ..., t1]
    """

    def __init__(self, vocab_size: int = 64, seq_len: int = 32, num_samples: int = 10000):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.num_samples = num_samples
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], torch.flip(self.data[idx], dims=[0])


def create_dataloaders(
    task: str = "copy",
    vocab_size: int = 64,
    seq_len: int = 32,
    train_samples: int = 8000,
    val_samples: int = 2000,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:

    torch.manual_seed(seed)
    DatasetClass = CopyTaskDataset if task == "copy" else ReverseTaskDataset
    train_ds = DatasetClass(vocab_size, seq_len, train_samples)
    val_ds = DatasetClass(vocab_size, seq_len, val_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
