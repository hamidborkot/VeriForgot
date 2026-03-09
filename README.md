# 🔐 VeriForgot

> **Blockchain-Attested Verifiable Machine Unlearning Using Membership Inference Oracles for GDPR Compliance**

[![Conference](https://img.shields.io/badge/Conference-CRBL%202026-blue?style=flat-square)](https://crbl2026.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow?style=flat-square)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square)](https://pytorch.org)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-lightgrey?style=flat-square)](https://www.cs.toronto.edu/~kriz/cifar.html)
[![Status](https://img.shields.io/badge/Status-Under%20Review-red?style=flat-square)](#)

---

## 📌 Overview

**VeriForgot** is the first empirically validated framework that provides **independently verifiable GDPR Article 17 compliance** for machine learning models. It addresses a critical gap: while machine unlearning methods exist, *no mechanism currently proves that unlearning genuinely occurred*.

### The Problem
GDPR Article 17 ("Right to Be Forgotten") requires organizations to remove personal data from trained ML models. But:
- Organizations **self-report compliance** — no independent verification
- Data subjects have **no tool to verify** erasure
- Regulators **lack a technical audit framework**

### Our Solution
VeriForgot combines three novel layers:

| Layer | Component | What it does |
|---|---|---|
| 🔍 Verification | MIA Oracle | Quantifies unlearning success via calibrated AUC threshold |
| ⛓️ Attestation | Blockchain Certificate | Tamper-proof, publicly auditable compliance record |
| 🔒 Privacy | ZK Proof Protocol | Proves parameter shift without revealing model weights |

---

## 🏗️ System Architecture

![VeriForgot Architecture](paper/figures/fig1_architecture.png)

*The complete VeriForgot verification pipeline: from GDPR erasure request through MIA oracle and ZK proof verification to immutable blockchain certificate issuance.*

---

## 📊 Key Results

All experiments on **CIFAR-10** with **ResNet-18** (NVIDIA Tesla T4, Kaggle).

### Unlearning Effectiveness (MIA AUC)

| Method | MIA AUC | Δ AUC | % Toward Random | Verdict |
|---|---|---|---|---|
| Original (baseline) | 0.5918 | — | — | ❌ FAIL |
| Gradient Ascent | 0.5272 | −0.0646 | 70.4% | ✅ PASS |
| Selective Retrain | 0.5604 | −0.0314 | 34.2% | ✅ PASS |
| **Strong GA** | **0.4669** | **−0.1249** | **136.1%** | ✅ PASS |

> AUC = 0.50 represents **perfect unlearning** (forget set indistinguishable from non-members)

### Model Accuracy Preservation

| Method | Forget Acc | Retain Acc | Test Acc | Test Drop |
|---|---|---|---|---|
| Original | 97.00% | 92.72% | 85.81% | — |
| Gradient Ascent | 94.60% | 92.01% | **85.78%** | **0.03%** |
| Selective Retrain | 88.40% | 88.51% | 85.15% | 0.66% |
| Strong GA | 96.20% | 92.05% | ~84.9% | <1% |

### Oracle Detection Accuracy

| Metric | Value |
|---|---|
| True Positive Rate (genuine unlearning detected) | **100.0%** |
| True Negative Rate (fake compliance rejected) | **90.0%** |
| Overall Accuracy | **95.0%** |
| Overall Accuracy (τ=0.58) | **100.0%** |

![Results Figure](paper/figures/fig2_results.png)

---

## 🗂️ Repository Structure

```
VeriForgot/
├── 📄 README.md
├── 📄 LICENSE
├── 📄 requirements.txt
├── 📄 .gitignore
│
├── 📁 paper/
│   ├── VeriForgot_CRBL2026.pdf        # Camera-ready paper (add after acceptance)
│   └── figures/
│       ├── fig1_architecture.png       # System architecture diagram
│       ├── fig2_results.png            # MIA AUC + accuracy results
│       └── fig3_oracle.png             # Oracle detection results
│
├── 📁 notebooks/
│   ├── 01_train_baseline.ipynb         # Baseline ResNet-18 training
│   ├── 02_unlearning_methods.ipynb     # GA, SR, SGA unlearning
│   ├── 03_mia_evaluation.ipynb         # MIA oracle evaluation
│   └── 04_oracle_detection.ipynb       # Genuine vs fake unlearning
│
├── 📁 src/
│   ├── models.py                       # ResNet-18 setup
│   ├── unlearning.py                   # Unlearning algorithm implementations
│   ├── mia_oracle.py                   # MIA oracle core logic
│   ├── data_utils.py                   # Dataset and dataloader utilities
│   └── blockchain/
│       ├── certificate.py              # Unlearning Certificate interface
│       └── zk_proof.py                 # ZK proof protocol specification
│
├── 📁 results/
│   └── all_results.json                # Full experimental results
│
└── 📁 configs/
    └── experiment_config.yaml          # Reproducible experiment configuration
```

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/hamidborkot/VeriForgot.git
cd VeriForgot
pip install -r requirements.txt
```

### 2. Run Full Pipeline (Kaggle recommended for free GPU)
```bash
# Open notebooks in order:
# 1. notebooks/01_train_baseline.ipynb     → trains ResNet-18 on CIFAR-10
# 2. notebooks/02_unlearning_methods.ipynb → runs GA, SR, SGA unlearning
# 3. notebooks/03_mia_evaluation.ipynb     → MIA oracle evaluation
# 4. notebooks/04_oracle_detection.ipynb   → genuine vs fake detection
```

### 3. Run MIA Oracle on a Single Model
```python
from src.mia_oracle import MIAOracle
from src.models import load_resnet18

model = load_resnet18('path/to/unlearned_model.pth')
oracle = MIAOracle(threshold=0.57)

result = oracle.evaluate(model, forget_loader, non_member_loader)
print(f"AUC: {result['auc']:.4f}")
print(f"Verdict: {'PASS ✅' if result['passed'] else 'FAIL ❌'}")
```

---

## 📋 Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | CIFAR-10 (50,000 train / 10,000 test) |
| Model | ResNet-18 (11.2M parameters) |
| D_forget size | 500 samples (classes 0 & 1) |
| D_retain size | 49,500 samples |
| Non-member set | 500 test samples (classes 2–9) |
| Hardware | NVIDIA Tesla T4 (Kaggle free tier) |
| Framework | PyTorch 2.x, CUDA |
| Oracle Threshold τ | 0.57 (calibrated; 0.58 for 100% accuracy) |

---

## 📜 Citation

If you use VeriForgot in your research, please cite:

```bibtex
@inproceedings{tulla2026veriforgot,
  title     = {VeriForgot: Blockchain-Attested Verifiable Machine Unlearning
               Using Membership Inference Oracles for GDPR Compliance},
  author    = {Tulla, Md. Hamid Borkot},
  booktitle = {Proceedings of the 6th International Conference on
               Cryptography and Blockchain (CRBL 2026)},
  year      = {2026},
  note      = {Under Review}
}
```

---

## 🛡️ License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**MD Hamid Borkot Tulla**  
Department of Computer Science and Information Engineering  
National Taiwan University of Science and Technology, Taipei, Taiwan  
📧 hamid.borkottulla@ntust.edu.tw  
🔗 [GitHub](https://github.com/hamidborkot)

---

## 🙏 Acknowledgements

Experiments were conducted using Kaggle free GPU resources (NVIDIA Tesla T4). The authors thank the open-source communities behind PyTorch, torchvision, and scikit-learn.

---

*Submitted to CRBL 2026 — 6th International Conference on Cryptography and Blockchain*
