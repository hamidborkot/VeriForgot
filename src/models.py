"""
models.py
---------
Model definitions and utilities for VeriForgot.
Uses ResNet-18 as the target model for CIFAR-10 experiments.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


def get_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """Return ResNet-18 configured for CIFAR-10 (512 -> num_classes)."""
    weights = tv_models.ResNet18_Weights.DEFAULT if pretrained else None
    model = tv_models.resnet18(weights=weights)
    model.fc = nn.Linear(512, num_classes)
    return model


def load_resnet18(path: str, num_classes: int = 10,
                  device: str = "cuda") -> nn.Module:
    """
    Load a saved ResNet-18 checkpoint.

    Args:
        path:        Path to .pth state dict file.
        num_classes: Number of output classes (default: 10).
        device:      Device to load onto ('cuda' or 'cpu').

    Returns:
        model:  Loaded ResNet-18 in eval() mode.
    """
    _device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = get_resnet18(num_classes=num_classes)
    model.load_state_dict(torch.load(path, map_location=_device))
    return model.to(_device).eval()


def count_parameters(model: nn.Module) -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_shift(model_a: nn.Module, model_b: nn.Module) -> float:
    """Euclidean L2 distance between two models' parameter vectors."""
    return float(
        torch.sqrt(sum(
            torch.sum((pa.data - pb.data) ** 2)
            for pa, pb in zip(model_a.parameters(), model_b.parameters())
        ))
    )
