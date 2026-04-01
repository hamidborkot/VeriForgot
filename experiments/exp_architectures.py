"""
VeriForgot — Experiment 3: Cross-Architecture Evaluation
Evaluates VeriForgot on VGG-11 and MobileNetV2 using CIFAR-10 + GA unlearning.

Reproduces Table 3 (Cross-Architecture Evaluation) from the paper.

Runtime: ~60 min on T4 GPU (VGG-11 ~35 min, MobileNetV2 ~25 min)
Output:  results/results_architectures.json

Note: VGG-11 seed=123 is excluded from the mean — training failed to converge
      (Orig Acc ~10%, consistent with known VGG sensitivity to initialisation
      on CIFAR-10 without LR warm-up).

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
    is_compliant, train_model, gradient_ascent, PASS_THRESHOLD
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

# ---------------------------------------------------------------------------
# CIFAR-10 data (shared across all architectures)
# ---------------------------------------------------------------------------
T_tr = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
T_te = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

full_train = torchvision.datasets.CIFAR10('../data', True,  download=True, transform=T_tr)
full_eval  = torchvision.datasets.CIFAR10('../data', True,  download=True, transform=T_te)
testset    = torchvision.datasets.CIFAR10('../data', False, download=True, transform=T_te)

all_tr_labels = np.array([y for _, y in full_train])
all_te_labels = np.array([y for _, y in testset])

forget_indices  = np.where(np.isin(all_tr_labels, [0, 1]))[0][:500].tolist()
retain_indices  = np.where(~np.isin(all_tr_labels, [0, 1]))[0].tolist()
nm_indices      = np.where(~np.isin(all_te_labels, [0, 1]))[0][:500].tolist()

forget_eval_ld  = DataLoader(Subset(full_eval,  forget_indices), batch_size=256, shuffle=False)
nm_loader       = DataLoader(Subset(testset,    nm_indices),     batch_size=256, shuffle=False)
forget_train_ld = DataLoader(Subset(full_train, forget_indices), batch_size=64,  shuffle=True)
retain_loader   = DataLoader(Subset(full_train, retain_indices), batch_size=128, shuffle=True)
train_loader    = DataLoader(full_train, batch_size=128, shuffle=True, num_workers=2)
test_loader     = DataLoader(testset,    batch_size=256, shuffle=False)

# ---------------------------------------------------------------------------
# Architecture configuration
# ---------------------------------------------------------------------------
ARCHS = [
    {
        'name': 'VGG-11',
        'arch': 'vgg11',
        'num_classes': 10,
        'epochs': 40,
        'lr': 0.05,
        'seeds': [42, 999],   # seed=123 excluded (convergence failure)
        'excluded_seeds': [123],
        'exclusion_note': 'seed=123 excluded: Orig Acc~10% (random chance). '
                          'Consistent with known VGG sensitivity to '
                          'initialisation on CIFAR-10 without LR warm-up.'
    },
    {
        'name': 'MobileNetV2',
        'arch': 'mobilenet',
        'num_classes': 10,
        'epochs': 40,
        'lr': 0.05,
        'seeds': [42, 123, 999],
        'excluded_seeds': [],
        'exclusion_note': None
    },
]

results = []

for cfg in ARCHS:
    print(f'\n{"="*60}\nArchitecture: {cfg["name"]}\n{"="*60}')
    arch_results = {
        'architecture': cfg['name'],
        'dataset': 'CIFAR-10',
        'seeds': [],
        'excluded_seeds': cfg['excluded_seeds'],
        'exclusion_note': cfg['exclusion_note']
    }

    for seed in cfg['seeds']:
        print(f'\n  Seed = {seed}')
        orig = train_model(cfg['arch'], cfg['num_classes'], train_loader,
                           device, epochs=cfg['epochs'], lr=cfg['lr'], seed=seed)
        path = f'../results/orig_{cfg["arch"]}_s{seed}.pth'
        torch.save(orig.state_dict(), path)

        orig_auc = mia_auc(orig, forget_eval_ld, nm_loader, device)
        orig_acc = test_acc(orig, test_loader, device)
        print(f'  Original  -> Acc={orig_acc:.1f}%  AUC={orig_auc:.4f}')

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
        verdict = 'PASS' if is_compliant(ga_auc) else 'FAIL'
        print(f'  GA   -> AUC={ga_auc:.4f}  Acc={ga_acc:.1f}%  UCS={ga_ucs:.3f}  {verdict}')

        arch_results['seeds'].append({
            'seed': seed, 'orig_auc': orig_auc, 'orig_acc': orig_acc,
            'ga_auc': ga_auc, 'ga_acc': ga_acc, 'ucs': ga_ucs,
            'pass': is_compliant(ga_auc)
        })

    # Summary statistics (over included seeds only)
    aucs = [r['ga_auc'] for r in arch_results['seeds']]
    arch_results['summary'] = {
        'mean_ga_auc': float(np.mean(aucs)),
        'std_ga_auc':  float(np.std(aucs)),
        'oracle_pass': sum(r['pass'] for r in arch_results['seeds']),
        'oracle_total': len(arch_results['seeds'])
    }
    results.append(arch_results)

output = {
    'experiment': 'Cross-Architecture Evaluation (VGG-11 + MobileNetV2 on CIFAR-10)',
    'pass_threshold': PASS_THRESHOLD,
    'results': results
}
with open('../results/results_architectures.json', 'w') as f:
    json.dump(output, f, indent=2)
print('\n✅ Saved: results/results_architectures.json')
