"""
unlearning.py
-------------
Machine unlearning algorithm implementations for VeriForgot.

Supported methods:
  - GradientAscent        : Standard gradient ascent on D_forget
  - SelectiveRetrain      : Full retraining on D_retain only
  - StrongGradientAscent  : Aggressive GA with gradient clipping + retain recovery
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Optional


class GradientAscent:
    """
    Gradient Ascent Unlearning.

    Maximizes loss on D_forget samples, then fine-tunes on D_retain
    to recover utility.

    Args:
        model:          Target model to unlearn.
        forget_loader:  DataLoader for D_forget.
        retain_loader:  DataLoader for D_retain (for interleaved fine-tuning).
        device:         Torch device.
        lr:             Learning rate for gradient ascent.
        momentum:       SGD momentum.
        epochs:         Number of gradient ascent epochs.
        interleave:     If True, interleave retain fine-tuning each epoch.
    """
    def __init__(self, model, forget_loader, retain_loader=None,
                 device="cuda", lr=0.0005, momentum=0.9, epochs=10,
                 interleave=True):
        self.model = model
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lr = lr; self.momentum = momentum
        self.epochs = epochs; self.interleave = interleave
        self.criterion = nn.CrossEntropyLoss()

    def run(self, verbose: bool = True):
        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.lr, momentum=self.momentum)
        for epoch in range(self.epochs):
            self.model.train()
            # Gradient ASCENT on forget set
            for x, y in self.forget_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = self.criterion(self.model(x), y)
                (-loss).backward()          # ascent = negate loss
                optimizer.step()

            # Optional: interleaved retain fine-tuning
            if self.interleave and self.retain_loader:
                for x, y in self.retain_loader:
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    self.criterion(self.model(x), y).backward()
                    optimizer.step()

            if verbose and (epoch + 1) % 5 == 0:
                print(f"  [GA] Epoch {epoch+1}/{self.epochs}")
        return self.model


class StrongGradientAscent:
    """
    Strong Gradient Ascent with gradient clipping and retain recovery.
    Achieves MIA AUC below 0.5 (AUC=0.4669 in paper experiments).

    Args:
        model:           Target model.
        forget_loader:   DataLoader for D_forget.
        retain_loader:   DataLoader for D_retain.
        device:          Torch device.
        lr:              Learning rate (default: 0.01).
        momentum:        SGD momentum.
        ascent_epochs:   Gradient ascent epochs (default: 25).
        grad_clip:       Max gradient norm for clipping (default: 1.0).
        finetune_epochs: Retain recovery fine-tuning epochs (default: 5).
        finetune_lr:     Learning rate for fine-tuning (default: 0.001).
    """
    def __init__(self, model, forget_loader, retain_loader,
                 device="cuda", lr=0.01, momentum=0.9,
                 ascent_epochs=25, grad_clip=1.0,
                 finetune_epochs=5, finetune_lr=0.001):
        self.model = model
        self.forget_loader = forget_loader
        self.retain_loader = retain_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lr = lr; self.momentum = momentum
        self.ascent_epochs = ascent_epochs; self.grad_clip = grad_clip
        self.finetune_epochs = finetune_epochs; self.finetune_lr = finetune_lr
        self.criterion = nn.CrossEntropyLoss()

    def run(self, verbose: bool = True):
        # Phase 1: Strong gradient ascent
        opt = optim.SGD(self.model.parameters(),
                        lr=self.lr, momentum=self.momentum)
        for epoch in range(self.ascent_epochs):
            self.model.train()
            for x, y in self.forget_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = self.criterion(self.model(x), y)
                (-loss).backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.grad_clip)
                opt.step()
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  [SGA] Ascent Epoch {epoch+1}/{self.ascent_epochs}")

        # Phase 2: Retain recovery fine-tuning
        opt_ft = optim.SGD(self.model.parameters(),
                            lr=self.finetune_lr, momentum=self.momentum)
        for epoch in range(self.finetune_epochs):
            self.model.train()
            for x, y in self.retain_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt_ft.zero_grad()
                self.criterion(self.model(x), y).backward()
                opt_ft.step()
            if verbose:
                print(f"  [SGA] Finetune Epoch {epoch+1}/{self.finetune_epochs}")

        return self.model


class SelectiveRetrain:
    """
    Selective Retraining from scratch on D_retain only.
    Provides strongest forget guarantee but highest computational cost.

    Args:
        model:          Fresh (randomly initialized) model to train.
        retain_loader:  DataLoader for D_retain.
        device:         Torch device.
        epochs:         Training epochs (default: 25).
        lr:             Initial learning rate.
    """
    def __init__(self, model, retain_loader, device="cuda",
                 epochs=25, lr=0.1, momentum=0.9, weight_decay=5e-4):
        self.model = model
        self.retain_loader = retain_loader
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.epochs = epochs; self.lr = lr
        self.momentum = momentum; self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def run(self, verbose: bool = True):
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr,
                               momentum=self.momentum,
                               weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs)
        for epoch in range(self.epochs):
            self.model.train()
            for x, y in self.retain_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                self.criterion(self.model(x), y).backward()
                optimizer.step()
            scheduler.step()
            if verbose and (epoch + 1) % 5 == 0:
                print(f"  [SR] Epoch {epoch+1}/{self.epochs}")
        return self.model
