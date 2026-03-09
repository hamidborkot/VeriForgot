"""
data_utils.py
-------------
Dataset and DataLoader utilities for VeriForgot CIFAR-10 experiments.
Handles forget/retain splits, random-label datasets, and non-member sampling.
"""

import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms
from typing import List, Tuple


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)


def get_transforms(train: bool = True):
    """Standard CIFAR-10 transforms."""
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])


def build_forget_retain_split(
    forget_classes: List[int],
    forget_size: int = 500,
    seed: int = 42,
    data_root: str = "./data",
    save_path: str = None,
) -> Tuple[List[int], List[int]]:
    """
    Build D_forget and D_retain index lists.

    Args:
        forget_classes: Class labels to sample D_forget from (e.g. [0, 1]).
        forget_size:    Number of samples in D_forget.
        seed:           Random seed for reproducibility.
        data_root:      CIFAR-10 download root.
        save_path:      Optional path prefix to save .pkl index files.

    Returns:
        forget_indices, retain_indices
    """
    np.random.seed(seed)
    full_train = torchvision.datasets.CIFAR10(
        data_root, train=True, download=True,
        transform=get_transforms(train=False)
    )
    candidates = [
        i for i, (_, label) in enumerate(full_train)
        if label in forget_classes
    ]
    np.random.shuffle(candidates)
    forget_indices = candidates[:forget_size]
    retain_indices = [i for i in range(len(full_train))
                      if i not in set(forget_indices)]

    if save_path:
        with open(f"{save_path}/forget_indices.pkl", "wb") as f:
            pickle.dump(forget_indices, f)
        with open(f"{save_path}/retain_indices.pkl", "wb") as f:
            pickle.dump(retain_indices, f)

    return forget_indices, retain_indices


def get_dataloaders(
    forget_indices: List[int],
    retain_indices: List[int],
    non_member_classes: List[int] = None,
    non_member_size: int = 500,
    data_root: str = "./data",
    batch_size_forget: int = 64,
    batch_size_retain: int = 128,
    batch_size_eval: int = 256,
):
    """
    Build all DataLoaders needed for VeriForgot experiments.

    Returns a dict with keys:
        forget_train, retain_train, forget_eval,
        retain_eval, test, non_member
    """
    full_train = torchvision.datasets.CIFAR10(
        data_root, True, download=True, transform=get_transforms(True))
    full_eval  = torchvision.datasets.CIFAR10(
        data_root, True, download=True, transform=get_transforms(False))
    testset    = torchvision.datasets.CIFAR10(
        data_root, False, download=True, transform=get_transforms(False))

    non_member_classes = non_member_classes or [2,3,4,5,6,7,8,9]
    nm_indices = [
        i for i, (_, l) in enumerate(testset)
        if l in non_member_classes
    ][:non_member_size]

    return {
        "forget_train":  DataLoader(Subset(full_train, forget_indices),
                                    batch_size=batch_size_forget,
                                    shuffle=True,  num_workers=2),
        "retain_train":  DataLoader(Subset(full_train, retain_indices),
                                    batch_size=batch_size_retain,
                                    shuffle=True,  num_workers=2),
        "forget_eval":   DataLoader(Subset(full_eval,  forget_indices),
                                    batch_size=batch_size_eval,
                                    shuffle=False, num_workers=2),
        "retain_eval":   DataLoader(Subset(full_eval,  retain_indices),
                                    batch_size=batch_size_eval,
                                    shuffle=False, num_workers=2),
        "test":          DataLoader(testset,
                                    batch_size=batch_size_eval,
                                    shuffle=False, num_workers=2),
        "non_member":    DataLoader(Subset(testset, nm_indices),
                                    batch_size=batch_size_eval,
                                    shuffle=False, num_workers=2),
    }


class RandomLabelDataset(Dataset):
    """
    Wraps a Subset and replaces each sample's label with a
    randomly chosen *different* class label.
    Used for Random Label unlearning baseline.
    """
    def __init__(self, subset: Subset, num_classes: int = 10, seed: int = 42):
        self.subset = subset
        np.random.seed(seed)
        orig_labels = [subset.dataset.targets[i] for i in subset.indices]
        self.random_labels = [
            (l + np.random.randint(1, num_classes)) % num_classes
            for l in orig_labels
        ]

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, _ = self.subset[idx]
        return x, self.random_labels[idx]
