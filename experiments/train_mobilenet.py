"""MobileNetV2 baseline training — CPU-optimised.
Runs 3 seeds sequentially. Saves checkpoints to ./checkpoints/
"""
import torch, torchvision, os, pickle
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.metrics import roc_auc_score

torch.set_num_threads(os.cpu_count())

SEEDS        = [42, 123, 999]
FORGET_CLASS = [0, 1]
FORGET_SIZE  = 300
EPOCHS       = 25
BATCH        = 256
LR           = 0.01

transform_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding=4),
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
transform_test = T.Compose([
    T.ToTensor(),
    T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

full_train = torchvision.datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform_train)
test_set   = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)


def get_forget_retain_idx(dataset, seed):
    np.random.seed(seed)
    forget_idx = [i for i, (_, l) in enumerate(dataset) if l in FORGET_CLASS]
    forget_idx = np.random.choice(forget_idx, FORGET_SIZE, replace=False).tolist()
    retain_idx = [i for i in range(len(dataset)) if i not in set(forget_idx)]
    return forget_idx, retain_idx


def build_mobilenet(seed):
    torch.manual_seed(seed)
    model = mobilenet_v2(weights=None, num_classes=10)
    model.features[0][0] = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=False)  # adapt for 32x32
    return model


def train_model(model, loader, epochs=EPOCHS):
    opt   = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = torch.nn.CrossEntropyLoss()
    model.train()
    for ep in range(1, epochs + 1):
        for x, y in loader:
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
        sched.step()
        if ep % 5 == 0:
            print(f"  ep {ep}/{epochs}")
    return model


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x).argmax(1) == y).sum().item()
            total   += y.size(0)
    return correct / total


def mia_auc(model, forget_idx, nonmember_idx):
    model.eval()
    sm     = torch.nn.Softmax(dim=1)
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


os.makedirs('checkpoints', exist_ok=True)
test_loader = DataLoader(test_set, batch_size=256, num_workers=0)

for seed in SEEDS:
    print(f"\n[s={seed}] training...")
    forget_idx, retain_idx = get_forget_retain_idx(full_train, seed)

    np.random.seed(seed)
    nm_pool = [i for i, (_, l) in enumerate(test_set) if l in FORGET_CLASS]
    nonmember_idx = np.random.choice(nm_pool, FORGET_SIZE, replace=False).tolist()

    train_loader = DataLoader(full_train, batch_size=BATCH, shuffle=True, num_workers=0)
    model = build_mobilenet(seed)
    model = train_model(model, train_loader)

    acc = evaluate(model, test_loader)
    auc = mia_auc(model, forget_idx, nonmember_idx)
    print(f"[s={seed}] Orig Acc={acc*100:.1f}%  AUC={auc:.4f}")

    torch.save(model.state_dict(), f'checkpoints/mobilenet_s{seed}.pt')
    with open(f'checkpoints/meta_s{seed}.pkl', 'wb') as f:
        pickle.dump({
            'forget_idx': forget_idx, 'retain_idx': retain_idx,
            'nonmember_idx': nonmember_idx, 'orig_acc': acc, 'orig_auc': auc
        }, f)

print("\n✅ All MobileNetV2 baselines saved.")
