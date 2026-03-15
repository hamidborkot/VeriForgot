"""
VeriForgot — Shared Utilities
MIA oracle, UCS metric, model training, GA unlearning, SCRUB unlearning,
fake model factory.

Paper: VeriForgot: Blockchain-Attested Machine Unlearning Verification
Venue: CRBL 2026
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

PASS_THRESHOLD = 0.57   # Calibrated oracle threshold (see Section 4.1)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------
def make_model(arch: str = 'resnet18', num_classes: int = 10) -> nn.Module:
    if arch == 'resnet18':
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(512, num_classes)
    elif arch == 'vgg11':
        m = models.vgg11(weights=None)
        m.classifier[6] = nn.Linear(4096, num_classes)
    elif arch == 'mobilenet':
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = nn.Linear(1280, num_classes)
    else:
        raise ValueError(f"Unknown arch: {arch}")
    return m


# ---------------------------------------------------------------------------
# MIA AUC oracle  (Eq. 2 in paper)
# ---------------------------------------------------------------------------
def mia_auc(model: nn.Module, forget_eval_ld, nm_loader, device) -> float:
    """Compute MIA AUC between forget-set confidence and non-member confidence."""
    model.eval()
    mc, nc = [], []
    with torch.no_grad():
        for x, y in forget_eval_ld:
            x, y = x.to(device), y.to(device)
            p = torch.softmax(model(x), dim=1)
            mc.extend(p[torch.arange(len(y)), y].cpu().numpy())
        for x, y in nm_loader:
            x, y = x.to(device), y.to(device)
            p = torch.softmax(model(x), dim=1)
            nc.extend(p[torch.arange(len(y)), y].cpu().numpy())
    labels = np.concatenate([np.ones(len(mc)), np.zeros(len(nc))])
    scores = np.concatenate([mc, nc])
    if np.isnan(scores).any():
        return 0.5  # degenerate model
    return float(roc_auc_score(labels, scores))


# ---------------------------------------------------------------------------
# Test accuracy
# ---------------------------------------------------------------------------
def test_acc(model: nn.Module, test_loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total   += len(y)
    return 100.0 * correct / total


# ---------------------------------------------------------------------------
# Unlearning Completeness Score  (Eq. 1 in paper)
# ---------------------------------------------------------------------------
def compute_ucs(auc_unlearn: float, auc_orig: float) -> float:
    """UCS = (AUC_orig - AUC_unlearn) / (AUC_orig - 0.5)"""
    if auc_orig <= 0.5:
        return 0.0
    return (auc_orig - auc_unlearn) / (auc_orig - 0.5)


def is_compliant(auc: float, threshold: float = PASS_THRESHOLD) -> bool:
    return auc < threshold


def is_valid_model(model: nn.Module) -> bool:
    """Guard against NaN/Inf weights (common with aggressive GA)."""
    return all(
        not (torch.isnan(p.data).any() or torch.isinf(p.data).any())
        for p in model.parameters()
    )


# ---------------------------------------------------------------------------
# Train original model
# ---------------------------------------------------------------------------
def train_model(
    arch: str, num_classes: int, train_loader, device,
    epochs: int = 40, lr: float = 0.1, seed: int = 42
) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    m    = make_model(arch, num_classes).to(device)
    opt  = optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(m(x), y).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
        sch.step()
        if (ep + 1) % 10 == 0:
            print(f'    ep {ep+1}/{epochs} done')
    return m


# ---------------------------------------------------------------------------
# Gradient Ascent unlearning
# ---------------------------------------------------------------------------
def gradient_ascent(
    model_path: str, arch: str, num_classes: int,
    forget_train_ld, retain_loader, device,
    epochs: int = 20, ga_lr: float = 0.010,
    ret_lr: float = 1e-3, clip: float = 5.0
) -> nn.Module:
    """
    GA Unlearning:
      - Maximise cross-entropy loss on D_forget  (ascent step)
      - Minimise cross-entropy loss on D_retain  (descent step)
    """
    m    = make_model(arch, num_classes).to(device)
    m.load_state_dict(torch.load(model_path, map_location=device))
    crit = nn.CrossEntropyLoss()
    o1   = optim.SGD(m.parameters(), lr=ga_lr,  momentum=0.9)
    o2   = optim.SGD(m.parameters(), lr=ret_lr, momentum=0.9)
    for _ in range(epochs):
        m.train()
        for x, y in forget_train_ld:
            x, y = x.to(device), y.to(device)
            o1.zero_grad()
            (-crit(m(x), y)).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), clip)
            o1.step()
        m.train()
        for x, y in retain_loader:
            x, y = x.to(device), y.to(device)
            o2.zero_grad()
            crit(m(x), y).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            o2.step()
        if not is_valid_model(m):
            print('  WARNING: NaN weights detected — stopping GA early')
            break
    return m


# ---------------------------------------------------------------------------
# SCRUB unlearning  (Kurmanji et al., NeurIPS 2023)
# ---------------------------------------------------------------------------
def scrub_unlearn(
    model_path: str, arch: str, num_classes: int,
    forget_train_ld, retain_loader, device,
    epochs: int = 15, lr: float = 0.005, T: float = 4.0
) -> nn.Module:
    """
    SCRUB — Student-Teacher KL divergence unlearning.
      Forget step : maximise KL(student || teacher) on D_forget
      Retain step : minimise KL + CE on D_retain
    """
    teacher = make_model(arch, num_classes).to(device)
    teacher.load_state_dict(torch.load(model_path, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = make_model(arch, num_classes).to(device)
    student.load_state_dict(torch.load(model_path, map_location=device))
    opt  = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    kl   = nn.KLDivLoss(reduction='batchmean')
    crit = nn.CrossEntropyLoss()

    for _ in range(epochs):
        student.train()
        for x, _ in forget_train_ld:                    # forget: maximise divergence
            x = x.to(device)
            with torch.no_grad():
                tl = teacher(x) / T
            sl   = student(x) / T
            loss = -kl(F.log_softmax(sl, dim=1), F.softmax(tl, dim=1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            opt.step()
        student.train()
        for x, y in retain_loader:                      # retain: minimise divergence
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                tl = teacher(x) / T
            sl   = student(x) / T
            loss = (
                0.5 * kl(F.log_softmax(sl, dim=1), F.softmax(tl, dim=1))
                + 0.5 * crit(student(x), y)
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            opt.step()
        sch.step()
    return student


# ---------------------------------------------------------------------------
# Fake model factory  (adversary simulation)
# ---------------------------------------------------------------------------
def make_fake_model(
    model_path: str, arch: str, num_classes: int, device,
    strategy: str, param, retain_loader=None
) -> nn.Module:
    """
    Simulate an adversarial (fake) unlearned model.
    Strategies: 'gaussian', 'fgsm', 'prune', 'finetune'
    """
    m = make_model(arch, num_classes).to(device)
    m.load_state_dict(torch.load(model_path, map_location=device))
    with torch.no_grad():
        if strategy == 'gaussian':
            for p in m.parameters():
                p.data.add_(torch.randn_like(p) * param)
        elif strategy == 'fgsm':
            for p in m.parameters():
                p.data.add_(torch.sign(torch.randn_like(p)) * param)
        elif strategy == 'prune':
            flat = torch.cat([p.data.abs().flatten()[:100_000]
                               for p in m.parameters()])
            thr  = torch.quantile(flat.float(), param)
            for p in m.parameters():
                p.data[p.data.abs() < thr] = 0.0
    if strategy == 'finetune' and retain_loader is not None:
        crit = nn.CrossEntropyLoss()
        opt  = optim.SGD(m.parameters(), lr=1e-3, momentum=0.9)
        for _ in range(int(param)):
            m.train()
            for x, y in retain_loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad()
                crit(m(x), y).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
                opt.step()
    return m
