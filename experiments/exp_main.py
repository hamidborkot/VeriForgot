"""
VeriForgot — Experiment 1: CIFAR-10 / ResNet-18
Main evaluation: GA unlearning + Oracle-30 adversary test + sample size sensitivity.

Runtime: ~45 min on T4 GPU
Output:  results/results_main.json
"""
import torch, torchvision, json, numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from utils import (
    make_model, mia_auc, test_acc, compute_ucs,
    is_compliant, train_model, gradient_ascent, make_fake_model
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')

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
forget_indices = np.where(np.isin(all_tr_labels, [0, 1]))[0][:500].tolist()
retain_indices = np.where(~np.isin(all_tr_labels, [0, 1]))[0].tolist()
nm_indices     = np.where(~np.isin(all_te_labels, [0, 1]))[0][:500].tolist()

forget_eval_ld  = DataLoader(Subset(full_eval,  forget_indices), batch_size=256, shuffle=False)
nm_loader       = DataLoader(Subset(testset,    nm_indices),     batch_size=256, shuffle=False)
forget_train_ld = DataLoader(Subset(full_train, forget_indices), batch_size=64,  shuffle=True)
retain_loader   = DataLoader(Subset(full_train, retain_indices), batch_size=128, shuffle=True)
train_loader    = DataLoader(full_train,                         batch_size=128, shuffle=True, num_workers=2)
test_loader     = DataLoader(testset,                            batch_size=256, shuffle=False)

SEEDS  = [42, 123, 999]
results = []

for seed in SEEDS:
    print(f'\n{"="*55}\nSeed = {seed}\n{"="*55}')
    orig = train_model('resnet18', 10, train_loader, device, epochs=40, seed=seed)
    path = f'../results/orig_cifar10_s{seed}.pth'
    torch.save(orig.state_dict(), path)
    orig_auc = mia_auc(orig, forget_eval_ld, nm_loader, device)
    orig_acc = test_acc(orig, test_loader, device)
    print(f'  Original  -> Acc={orig_acc:.1f}%  AUC={orig_auc:.4f}')

    print('  Running GA...')
    ga_aucs, ga_accs = [], []
    for ep, lr in [(18, 0.010), (20, 0.010), (22, 0.010)]:
        mg = gradient_ascent(path, 'resnet18', 10,
                             forget_train_ld, retain_loader, device,
                             epochs=ep, ga_lr=lr)
        ga_aucs.append(mia_auc(mg, forget_eval_ld, nm_loader, device))
        ga_accs.append(test_acc(mg, test_loader, device))
    ga_auc = float(np.median(ga_aucs))
    ga_acc = float(np.median(ga_accs))
    ga_ucs = compute_ucs(ga_auc, orig_auc)
    verdict = '✅ PASS' if is_compliant(ga_auc) else '❌ FAIL'
    print(f'  GA Median -> AUC={ga_auc:.4f}  Acc={ga_acc:.1f}%  UCS={ga_ucs:.3f}  {verdict}')
    results.append({
        'seed': seed, 'orig_auc': orig_auc, 'orig_acc': orig_acc,
        'ga_auc': ga_auc, 'ga_acc': ga_acc, 'ucs': ga_ucs,
        'pass': is_compliant(ga_auc)
    })

with open('../results/results_main.json', 'w') as f:
    json.dump({'experiment': 'CIFAR-10/ResNet-18', 'results': results}, f, indent=2)
print('\n✅ Saved: results/results_main.json')
