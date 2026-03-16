"""
VeriForgot Experiment: SCRUB Cross-Dataset Evaluation
Datasets: CIFAR-100, SVHN
Architecture: ResNet-18
Seeds: 42, 123, 999
Methods: Gradient Ascent (GA) + SCRUB (Kurmanji et al., NeurIPS 2023)

This experiment proves the oracle is method-agnostic across datasets.
Run on Kaggle GPU T4 (~80 min).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import json
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PASS_THRESHOLD = 0.57


def make_resnet10():
    """ResNet-18 for 10-class datasets (SVHN)."""
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, 10)
    return m.to(device)


def make_resnet100():
    """ResNet-18 for CIFAR-100 (100 classes)."""
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(512, 100)
    return m.to(device)


def mia_auc(model, forget_ld, nm_ld):
    """Membership Inference Attack AUC (shadow-free, confidence-based)."""
    model.eval()
    mc, nc = [], []
    with torch.no_grad():
        for x, y in forget_ld:
            x, y = x.to(device), y.to(device)
            p = torch.softmax(model(x), dim=1)
            mc.extend(p[torch.arange(len(y)), y].cpu().numpy())
        for x, y in nm_ld:
            x, y = x.to(device), y.to(device)
            p = torch.softmax(model(x), dim=1)
            nc.extend(p[torch.arange(len(y)), y].cpu().numpy())
    labels = np.concatenate([np.ones(len(mc)), np.zeros(len(nc))])
    scores = np.concatenate([mc, nc])
    if np.isnan(scores).any() or np.std(scores) == 0:
        return None
    return float(roc_auc_score(labels, scores))


def test_acc(model, test_ld):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for x, y in test_ld:
            x, y = x.to(device), y.to(device)
            c += (model(x).argmax(1) == y).sum().item()
            t += len(y)
    return 100 * c / t


def compute_ucs(auc_m, auc_base):
    """Unlearning Completeness Score."""
    if auc_base <= 0.5:
        return 0.0
    return (auc_base - auc_m) / (auc_base - 0.5)


def train_model(make_fn, train_ld, seed, epochs=40, lr=0.1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = make_fn()
    opt = optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        m.train()
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            crit(m(x), y).backward()
            opt.step()
        sch.step()
        if (ep + 1) % 10 == 0:
            print(f"    ep {ep+1}/{epochs}")
    return m


def gradient_ascent(make_fn, orig_path, forget_ld, retain_ld,
                    epochs=20, ga_lr=0.005, ret_lr=1e-3):
    """
    Gradient Ascent unlearning.
    NOTE: Use ga_lr=0.005 for CIFAR-100 (100-class tasks).
          ga_lr=0.010 causes model collapse on 100-class tasks.
    """
    m = make_fn()
    m.load_state_dict(torch.load(orig_path, map_location=device))
    o1 = optim.SGD(m.parameters(), lr=ga_lr, momentum=0.9)
    o2 = optim.SGD(m.parameters(), lr=ret_lr, momentum=0.9, weight_decay=5e-4)
    crit = nn.CrossEntropyLoss()
    for _ in range(epochs):
        m.train()
        for x, y in forget_ld:
            x, y = x.to(device), y.to(device)
            o1.zero_grad()
            (-crit(m(x), y)).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            o1.step()
        m.train()
        for x, y in retain_ld:
            x, y = x.to(device), y.to(device)
            o2.zero_grad()
            crit(m(x), y).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            o2.step()
    return m


def scrub_unlearn(make_fn, orig_path, forget_ld, retain_ld,
                  epochs=15, lr=0.005, T=4.0):
    """
    SCRUB Unlearning — Kurmanji et al., NeurIPS 2023.
    Maximises KL divergence on forget set, minimises on retain set.
    """
    teacher = make_fn()
    teacher.load_state_dict(torch.load(orig_path, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    student = make_fn()
    student.load_state_dict(torch.load(orig_path, map_location=device))
    opt = optim.SGD(student.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    kl = nn.KLDivLoss(reduction='batchmean')
    crit = nn.CrossEntropyLoss()

    for _ in range(epochs):
        student.train()
        for x, _ in forget_ld:           # FORGET: maximise divergence from teacher
            x = x.to(device)
            with torch.no_grad():
                tl = teacher(x) / T
            sl = student(x) / T
            loss = -kl(F.log_softmax(sl, dim=1), F.softmax(tl, dim=1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            opt.step()
        student.train()
        for x, y in retain_ld:          # RETAIN: minimise divergence + CE
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                tl = teacher(x) / T
            sl = student(x) / T
            loss = (0.5 * kl(F.log_softmax(sl, dim=1), F.softmax(tl, dim=1))
                    + 0.5 * crit(student(x), y))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 3.0)
            opt.step()
        sch.step()
    return student


def run_dataset(name, make_fn, trainset, evalset, testset,
               forget_classes, forget_size, seed_list,
               ga_lr=0.005):
    """Run GA + SCRUB for one dataset across multiple seeds."""
    print(f"\n{'='*55}")
    print(f"DATASET: {name}")
    print(f"{'='*55}")

    tr_labels = np.array([y for _, y in trainset])
    te_labels  = np.array([y for _, y in testset])
    forget_idx = np.where(np.isin(tr_labels, forget_classes))[0][:forget_size].tolist()
    retain_idx = np.where(~np.isin(tr_labels, forget_classes))[0].tolist()
    nm_idx     = np.where(~np.isin(te_labels, forget_classes))[0][:500].tolist()
    print(f"Forget:{len(forget_idx)} | Retain:{len(retain_idx)} | NM:{len(nm_idx)}")

    forget_eval = DataLoader(Subset(evalset,   forget_idx), batch_size=256, shuffle=False)
    nm_loader   = DataLoader(Subset(testset,   nm_idx),    batch_size=256, shuffle=False)
    forget_tr   = DataLoader(Subset(trainset,  forget_idx),batch_size=64,  shuffle=True)
    retain_ld   = DataLoader(Subset(trainset,  retain_idx),batch_size=128, shuffle=True)
    test_ld     = DataLoader(testset, batch_size=256, shuffle=False)
    results     = []

    for seed in seed_list:
        print(f"\n  [Seed={seed}] Training original...")
        orig_path = f'/kaggle/working/{name}_orig_s{seed}.pth'
        try:
            m = make_fn()
            m.load_state_dict(torch.load(orig_path, map_location=device))
            print(f"  Loaded cached model")
        except FileNotFoundError:
            tr_ld = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
            m = train_model(make_fn, tr_ld, seed, epochs=40, lr=0.1)
            torch.save(m.state_dict(), orig_path)

        orig_auc = mia_auc(m, forget_eval, nm_loader)
        orig_acc = test_acc(m, test_ld)
        print(f"  Original  Acc={orig_acc:.1f}%  AUC={orig_auc:.4f}")

        # Gradient Ascent
        ga_aucs, ga_accs = [], []
        for ep, lr in [(18, ga_lr), (20, ga_lr), (22, ga_lr)]:
            mg = gradient_ascent(make_fn, orig_path, forget_tr, retain_ld, ep, lr)
            a = mia_auc(mg, forget_eval, nm_loader)
            if a is not None:
                ga_aucs.append(a)
                ga_accs.append(test_acc(mg, test_ld))
        if ga_aucs:
            ga_auc = float(np.median(ga_aucs))
            ga_acc = float(np.median(ga_accs))
            ga_ucs = compute_ucs(ga_auc, orig_auc)
            v = '✅ PASS' if ga_auc < PASS_THRESHOLD else '❌ FAIL'
            print(f"  GA     AUC={ga_auc:.4f}  Acc={ga_acc:.1f}%  UCS={ga_ucs:.3f}  {v}")
        else:
            ga_auc = ga_acc = ga_ucs = float('nan')
            print("  GA     AUC=nan (model collapse — reduce lr)")

        # SCRUB
        sc_aucs, sc_accs = [], []
        for ep, lr in [(12, 0.005), (15, 0.005), (18, 0.005)]:
            ms = scrub_unlearn(make_fn, orig_path, forget_tr, retain_ld, ep, lr)
            a = mia_auc(ms, forget_eval, nm_loader)
            if a is not None:
                sc_aucs.append(a)
                sc_accs.append(test_acc(ms, test_ld))
        sc_auc = float(np.median(sc_aucs))
        sc_acc = float(np.median(sc_accs))
        sc_ucs = compute_ucs(sc_auc, orig_auc)
        v = '✅ PASS' if sc_auc < PASS_THRESHOLD else '❌ FAIL'
        print(f"  SCRUB  AUC={sc_auc:.4f}  Acc={sc_acc:.1f}%  UCS={sc_ucs:.3f}  {v}")

        results.append({
            'seed': seed, 'orig_auc': orig_auc, 'orig_acc': orig_acc,
            'ga_auc': ga_auc, 'ga_acc': ga_acc, 'ga_ucs': ga_ucs,
            'sc_auc': sc_auc, 'sc_acc': sc_acc, 'sc_ucs': sc_ucs
        })
    return results


if __name__ == '__main__':
    all_results = {}

    # CIFAR-100
    T100_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    T100_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    c100_tr = torchvision.datasets.CIFAR100('./data', True,  download=True, transform=T100_tr)
    c100_ev = torchvision.datasets.CIFAR100('./data', True,  download=True, transform=T100_te)
    c100_te = torchvision.datasets.CIFAR100('./data', False, download=True, transform=T100_te)
    all_results['CIFAR100'] = run_dataset(
        'CIFAR100', make_resnet100,
        c100_tr, c100_ev, c100_te,
        forget_classes=list(range(10)), forget_size=500,
        seed_list=[42, 123, 999], ga_lr=0.005)   # NOTE: 0.005 not 0.010

    # SVHN
    Tsv_tr = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])
    Tsv_te = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])
    sv_tr = torchvision.datasets.SVHN('./data', 'train', download=True, transform=Tsv_tr)
    sv_ev = torchvision.datasets.SVHN('./data', 'train', download=True, transform=Tsv_te)
    sv_te = torchvision.datasets.SVHN('./data', 'test',  download=True, transform=Tsv_te)
    all_results['SVHN'] = run_dataset(
        'SVHN', make_resnet10,
        sv_tr, sv_ev, sv_te,
        forget_classes=[0, 1], forget_size=500,
        seed_list=[42, 123, 999], ga_lr=0.010)

    with open('/kaggle/working/scrub_crossdataset.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*55)
    print("CROSS-DATASET SCRUB SUMMARY")
    print("="*55)
    for ds, res in all_results.items():
        ga_aucs = [r['ga_auc'] for r in res if not np.isnan(r['ga_auc'])]
        sc_aucs = [r['sc_auc'] for r in res]
        ga_pass = sum(1 for r in res if r['ga_auc'] < PASS_THRESHOLD)
        sc_pass = sum(1 for r in res if r['sc_auc'] < PASS_THRESHOLD)
        print(f"\n  {ds}:")
        if ga_aucs:
            print(f"    GA    AUC={np.mean(ga_aucs):.4f}+-{np.std(ga_aucs):.4f}  PASS={ga_pass}/3")
        print(f"    SCRUB AUC={np.mean(sc_aucs):.4f}+-{np.std(sc_aucs):.4f}  PASS={sc_pass}/3")
    print("\n Saved: /kaggle/working/scrub_crossdataset.json")
