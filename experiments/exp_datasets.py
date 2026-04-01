"""
VeriForgot — Experiment 2: Cross-Dataset Evaluation
Evaluates VeriForgot across CIFAR-100 and SVHN using ResNet-18 + GA unlearning.

Reproduces Table 2 (Cross-Dataset Oracle Evaluation) from the paper.

Runtime: ~90 min on T4 GPU (CIFAR-100 ~60 min, SVHN ~30 min)
Output:  results/results_datasets.json

Paper: VeriForgot: Blockchain-Attested Verifiable Machine Unlearning
       Using Membership Inference Auditing for GDPR Compliance
Journal: Journal of Information Security and Applications, Elsevier
Author: Md. Hamid Borkot Tulla
"""
import torch, torchvision, json, numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from utils import (
    make_model, mia_auc, test_acc, compute_ucs,
    is_compliant, train_model, gradient_ascent, scrub_unlearn, PASS_THRESHOLD
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
DATASETS = [
    {
        'name': 'CIFAR-100',
        'num_classes': 100,
        'forget_classes': list(range(0, 10)),   # first 10 superclasses
        'arch': 'resnet18',
        'epochs': 60,
        'lr': 0.1,
        'transform_train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))
        ]),
        'transform_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408),
                                  (0.2675, 0.2565, 0.2761))
        ]),
        'loader': torchvision.datasets.CIFAR100,
    },
    {
        'name': 'SVHN',
        'num_classes': 10,
        'forget_classes': [0, 1],
        'arch': 'resnet18',
        'epochs': 40,
        'lr': 0.05,
        'transform_train': transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                  (0.1980, 0.2010, 0.1970))
        ]),
        'transform_test': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                  (0.1980, 0.2010, 0.1970))
        ]),
        'loader': torchvision.datasets.SVHN,
    },
]

SEEDS   = [42, 123, 999]
results = []


def load_dataset(cfg):
    """Load train/test splits with correct API for each dataset."""
    if cfg['name'] == 'SVHN':
        full_train = cfg['loader']('../data', split='train', download=True,
                                   transform=cfg['transform_train'])
        full_eval  = cfg['loader']('../data', split='train', download=True,
                                   transform=cfg['transform_test'])
        testset    = cfg['loader']('../data', split='test',  download=True,
                                   transform=cfg['transform_test'])
        tr_labels  = np.array(full_train.labels)
        te_labels  = np.array(testset.labels)
    else:
        full_train = cfg['loader']('../data', train=True,  download=True,
                                   transform=cfg['transform_train'])
        full_eval  = cfg['loader']('../data', train=True,  download=True,
                                   transform=cfg['transform_test'])
        testset    = cfg['loader']('../data', train=False, download=True,
                                   transform=cfg['transform_test'])
        tr_labels  = np.array(full_train.targets)
        te_labels  = np.array(testset.targets)
    return full_train, full_eval, testset, tr_labels, te_labels


for cfg in DATASETS:
    print(f'\n{"="*60}\nDataset: {cfg["name"]}\n{"="*60}')
    full_train, full_eval, testset, tr_labels, te_labels = load_dataset(cfg)

    forget_indices  = np.where(np.isin(tr_labels, cfg['forget_classes']))[0][:500].tolist()
    retain_indices  = np.where(~np.isin(tr_labels, cfg['forget_classes']))[0].tolist()
    nm_indices      = np.where(~np.isin(te_labels, cfg['forget_classes']))[0][:500].tolist()

    forget_eval_ld  = DataLoader(Subset(full_eval,  forget_indices), batch_size=256, shuffle=False)
    nm_loader       = DataLoader(Subset(testset,    nm_indices),     batch_size=256, shuffle=False)
    forget_train_ld = DataLoader(Subset(full_train, forget_indices), batch_size=64,  shuffle=True)
    retain_loader   = DataLoader(Subset(full_train, retain_indices), batch_size=128, shuffle=True)
    train_loader    = DataLoader(full_train, batch_size=128, shuffle=True, num_workers=2)
    test_loader     = DataLoader(testset,    batch_size=256, shuffle=False)

    dataset_results = {'dataset': cfg['name'], 'ga': [], 'scrub': []}

    for seed in SEEDS:
        print(f'\n  Seed = {seed}')
        orig = train_model(cfg['arch'], cfg['num_classes'], train_loader,
                           device, epochs=cfg['epochs'], lr=cfg['lr'], seed=seed)
        path = f'../results/orig_{cfg["name"].lower()}_s{seed}.pth'
        torch.save(orig.state_dict(), path)

        orig_auc = mia_auc(orig, forget_eval_ld, nm_loader, device)
        orig_acc = test_acc(orig, test_loader, device)
        print(f'  Original  -> Acc={orig_acc:.1f}%  AUC={orig_auc:.4f}')

        # --- Gradient Ascent ---
        ga_aucs, ga_accs = [], []
        for ep in [18, 20, 22]:
            mg = gradient_ascent(path, cfg['arch'], cfg['num_classes'],
                                 forget_train_ld, retain_loader, device,
                                 epochs=ep, ga_lr=0.010)
            ga_aucs.append(mia_auc(mg, forget_eval_ld, nm_loader, device))
            ga_accs.append(test_acc(mg, test_loader, device))
        ga_auc = float(np.median(ga_aucs))
        ga_acc = float(np.median(ga_accs))
        ga_ucs = compute_ucs(ga_auc, orig_auc)
        print(f'  GA   -> AUC={ga_auc:.4f}  Acc={ga_acc:.1f}%  '
              f'UCS={ga_ucs:.3f}  {"PASS" if is_compliant(ga_auc) else "FAIL"}')
        dataset_results['ga'].append({
            'seed': seed, 'orig_auc': orig_auc, 'orig_acc': orig_acc,
            'ga_auc': ga_auc, 'ga_acc': ga_acc, 'ucs': ga_ucs,
            'pass': is_compliant(ga_auc)
        })

        # --- SCRUB ---
        ms = scrub_unlearn(path, cfg['arch'], cfg['num_classes'],
                           forget_train_ld, retain_loader, device)
        sc_auc = mia_auc(ms, forget_eval_ld, nm_loader, device)
        sc_acc = test_acc(ms, test_loader, device)
        sc_ucs = compute_ucs(sc_auc, orig_auc)
        print(f'  SCRUB-> AUC={sc_auc:.4f}  Acc={sc_acc:.1f}%  '
              f'UCS={sc_ucs:.3f}  {"PASS" if is_compliant(sc_auc) else "FAIL"}')
        dataset_results['scrub'].append({
            'seed': seed, 'orig_auc': orig_auc, 'orig_acc': orig_acc,
            'sc_auc': sc_auc, 'sc_acc': sc_acc, 'ucs': sc_ucs,
            'pass': is_compliant(sc_auc)
        })

    results.append(dataset_results)

output = {'experiment': 'Cross-Dataset Evaluation (CIFAR-100 + SVHN)',
          'pass_threshold': PASS_THRESHOLD,
          'results': results}
with open('../results/results_datasets.json', 'w') as f:
    json.dump(output, f, indent=2)
print('\n✅ Saved: results/results_datasets.json')
