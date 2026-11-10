"""Dataset and loader utilities."""

import torch
from torch.utils.data import DataLoader, Dataset


class ProjectDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = root
        self.split = split
        self.items = sorted((root / split).glob("*.npz"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        import numpy as np

        data = np.load(self.items[index])
        return {
            "inputs": torch.from_numpy(data["inputs"]).float(),
            "targets": torch.from_numpy(data["targets"]).float(),
            "name": self.items[index].stem,
        }


def build_loader(root, batch_size, split="train"):
    return DataLoader(ProjectDataset(root, split), batch_size=batch_size, shuffle="train" == split)
