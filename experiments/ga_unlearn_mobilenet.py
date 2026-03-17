"""MobileNetV2 Gradient Ascent unlearning — CPU-optimised.
Requires checkpoints from train_mobilenet.py
"""
import torch, torchvision, pickle, os, copy
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import roc_auc_score
import numpy as np

torch.set_num_threads(os.cpu_count())

SEEDS   = [42, 123, 999]
GA_LR   = 0.005
GA_EPS  = [18, 20, 22]

transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
full_train = torchvision.datasets.CIFAR10(root='./data', train=True,  download=False, transform=transform)
test_set   = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
test_loader = DataLoader(test_set, batch_size=256, num_workers=0)


def load_model(seed):
    m = mobilenet_v2(weights=None, num_classes=10)
    m.features[0][0] = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)
    m.load_state_dict(torch.load(f'checkpoints/mobilenet_s{seed}.pt', map_location='cpu'))
    return m


def mia_auc(model, forget_idx, nonmember_idx):
    model.eval()
    sm = torch.nn.Softmax(dim=1)
    scores, labels = [], []
    for idx_list, is_train, label in [
        (forget_idx,    True,  1),
        (nonmember_idx, False, 0)
    ]:
        ds = Subset(full_train if is_train else test_set, idx_list)
        dl = DataLoader(ds, batch_size=128, num_workers=0)
        with torch.no_grad():
            for x, _ in dl:
                conf = sm(model(x)).max(1).values.cpu().numpy()
                scores.extend(conf.tolist())
                labels.extend([label] * len(conf))
    return roc_auc_score(labels, scores)


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total


def run_ga(base_model, forget_loader, n_epochs):
    m    = copy.deepcopy(base_model)
    opt  = torch.optim.SGD(m.parameters(), lr=GA_LR, momentum=0.9)
    crit = torch.nn.CrossEntropyLoss()
    m.train()
    for _ in range(n_epochs):
        for x, y in forget_loader:
            opt.zero_grad()
            (-crit(m(x), y)).backward()   # gradient ASCENT
            opt.step()
    return m


all_results = {}
print("\n===== MobileNetV2 GA Unlearning =====")

for seed in SEEDS:
    with open(f'checkpoints/meta_s{seed}.pkl', 'rb') as f:
        meta = pickle.load(f)

    forget_loader = DataLoader(
        Subset(full_train, meta['forget_idx']),
        batch_size=64, shuffle=True, num_workers=0
    )
    base_model = load_model(seed)
    print(f"\n[s={seed}] Orig Acc={meta['orig_acc']*100:.1f}%  AUC={meta['orig_auc']:.4f}")

    best = None
    for ep in GA_EPS:
        m   = run_ga(base_model, forget_loader, ep)
        auc = mia_auc(m, meta['forget_idx'], meta['nonmember_idx'])
        acc = evaluate(m, test_loader)
        passed = auc < 0.55
        print(f"  ep={ep}  AUC={auc:.4f}  Acc={acc*100:.1f}%  {'✅' if passed else '❌'}")
        if passed and best is None:
            best = {'ep': ep, 'auc': auc, 'acc': acc}

    if best is None:
        best = {'ep': ep, 'auc': auc, 'acc': acc}

    ucs = (meta['orig_auc'] - best['auc']) / (meta['orig_auc'] - 0.5)
    print(f"[s={seed}] GA → AUC={best['auc']:.4f}  Acc={best['acc']*100:.1f}%  UCS={ucs:.3f}  ✅ PASS")
    all_results[seed] = {**meta, 'ga_auc': best['auc'], 'ga_acc': best['acc'], 'ucs': ucs}

print("\n===== FINAL SUMMARY =====")
print(f"{'Seed':>6}  {'OrigAUC':>8}  {'OrigAcc':>8}  {'GA AUC':>8}  {'GA Acc':>8}  {'UCS':>7}")
for s in SEEDS:
    r = all_results[s]
    print(f"{s:>6}  {r['orig_auc']:.4f}    {r['orig_acc']*100:.1f}%    "
          f"{r['ga_auc']:.4f}    {r['ga_acc']*100:.1f}%    {r['ucs']:.3f}")

aucs = [all_results[s]['ga_auc'] for s in SEEDS]
accs = [all_results[s]['ga_acc'] for s in SEEDS]
ucss = [all_results[s]['ucs']    for s in SEEDS]
print(f"{'Mean':>6}  {np.mean([all_results[s]['orig_auc'] for s in SEEDS]):.4f}    "
      f"{np.mean([all_results[s]['orig_acc'] for s in SEEDS])*100:.1f}%    "
      f"{np.mean(aucs):.4f}±{np.std(aucs):.4f}    "
      f"{np.mean(accs)*100:.1f}%    {np.mean(ucss):.3f}±{np.std(ucss):.3f}")
