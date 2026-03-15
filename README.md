# VeriForgot 🔐

> **Verifiable Machine Unlearning with MIA Oracle, Blockchain Attestation, and Cryptographic Commitment**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CRBL 2026](https://img.shields.io/badge/Submitted-CRBL%202026-green.svg)]()
[![Ethereum](https://img.shields.io/badge/Solidity-0.8.20-blue.svg)](contracts/)

**"VeriForgot: Blockchain-Attested Machine Unlearning Verification via Membership Inference Oracle and Cryptographic Commitment"**
*Submitted — CRBL 2026 (6th International Conference on Cryptography and Blockchain)*

---

## 🔍 Abstract

GDPR Article 17 grants data subjects the *right to erasure*, yet no practical mechanism exists to verify that an ML model has genuinely forgotten specific training data. **VeriForgot** addresses this with a four-component system:

1. 🎯 **MIA Oracle** — empirical Membership Inference Attack oracle for outcome-level compliance
2. 📐 **UCS Metric** — Unlearning Completeness Score, a continuous compliance quantifier
3. ⛓️ **On-chain Certificate** — immutable Solidity smart contract anchoring compliance records
4. 🔒 **Weight Commitment** — SHA-256 cryptographic commitment for weight integrity proof

Evaluated across **3 datasets × 3 architectures × 3 seeds**, achieving **100% intentional-fake detection (TNR)** across all configurations and **method-agnostic** verification across both GA and SCRUB unlearning.

---

## 📊 Key Results

### Cross-Architecture × Cross-Dataset Oracle Evaluation

| Dataset | Model | Orig AUC | GA AUC | UCS | TNR (Intentional Fakes) |
|---|---|---|---|---|---|
| CIFAR-10 | ResNet-18 | 0.5881 | 0.4892 | 1.12 | **100%** |
| CIFAR-100 | ResNet-18 | 0.7694 | 0.4913 | 1.03 | **100%** |
| SVHN | ResNet-18 | 0.6509 | 0.4227 | 1.51 | **100%** |
| CIFAR-10 | VGG-11 | 0.6066 | 0.0242 | 5.46 | **100%** |
| CIFAR-10 | MobileNetV2 | 0.6220 | 0.0000 | 5.10 | **100%** |

### GA vs SCRUB — Method-Agnostic Oracle Verification

| Method | MIA AUC ↓ | Test Acc ↑ | UCS ↑ | Oracle |
|---|---|---|---|---|
| Gradient Ascent (GA) | 0.430 ± 0.017 | 11.6% | 1.91 | ✅ PASS 3/3 |
| SCRUB (Kurmanji, NeurIPS 2023) | **0.055 ± 0.005** | **72.2%** | **6.90** | ✅ PASS 3/3 |

> **Key finding:** SCRUB achieves 7.8× stronger privacy removal and 6× better utility preservation vs GA, while both pass the oracle. The oracle verifies *outcome*, not *method*.

### Oracle-30 Summary (CIFAR-10 / ResNet-18)

| Category | Result | Interpretation |
|---|---|---|
| Genuine compliant (well-configured) | 10/15 PASS | Effective unlearning certified |
| Genuine non-compliant (bad hyperparams) | 5/15 FAIL | Oracle correctly rejects insufficient unlearning |
| Fake (Gaussian noise, FGSM, pruning, finetune) | 15/15 CAUGHT | **TNR = 100%** |

### Compliance Soundness Bound

| Configuration | P(fake passes oracle) |
|---|---|
| CIFAR-10 / ResNet-18 | ≤ 1.34% |
| CIFAR-100 / ResNet-18 | ≈ 0.000% |
| SVHN / ResNet-18 | ≤ 6.97% |

### Smart Contract Gas Costs

| Operation | Gas | USD @ 20 gwei, ETH=$3,000 |
|---|---|---|
| Store weight commitment | 46,872 | $2.81 |
| Verify commitment (k=1,000) | 29,412 | $1.77 |
| Issue compliance certificate | 68,914 | $4.13 |
| Full pipeline total | ~145,198 | ~$8.71 |

---

## 🚀 Quick Start

```bash
git clone https://github.com/hamidborkot/VeriForgot.git
cd VeriForgot
pip install -r requirements.txt

# Run individual experiments (GPU required, ~45 min each)
python experiments/exp_main.py           # CIFAR-10 / ResNet-18
python experiments/exp_datasets.py       # CIFAR-100 + SVHN
python experiments/exp_architectures.py  # VGG-11 + MobileNetV2
python experiments/exp_scrub.py          # GA vs SCRUB comparison

# Or run everything
bash scripts/run_all.sh
```

---

## 🗂️ Repository Structure

```
VeriForgot/
├── experiments/
│   ├── utils.py                  # Shared: MIA oracle, UCS, train, GA, SCRUB, fakes
│   ├── exp_main.py               # CIFAR-10 / ResNet-18 (primary evaluation)
│   ├── exp_datasets.py           # CIFAR-100 + SVHN cross-dataset
│   ├── exp_architectures.py      # VGG-11 + MobileNetV2 cross-architecture
│   └── exp_scrub.py              # GA vs SCRUB method-agnostic comparison
├── contracts/
│   ├── VeriForgotOracle.sol      # On-chain compliance certificate (ERC-style)
│   └── VeriForgotCommitment.sol  # SHA-256 weight commitment verifier
├── commitment/
│   └── weight_commitment.py      # Python cryptographic commitment client
├── results/
│   ├── results_main.json         # CIFAR-10/ResNet-18 full results
│   ├── results_datasets.json     # CIFAR-100 + SVHN results
│   ├── results_architectures.json# VGG-11 + MobileNetV2 results
│   └── results_scrub.json        # SCRUB vs GA comparison
├── scripts/
│   └── run_all.sh                # Full pipeline runner
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📐 Unlearning Completeness Score (UCS)

The UCS normalises the MIA AUC drop relative to the pre-unlearning baseline:

```
UCS(M_unlearn) = (AUC_orig - AUC_unlearn) / (AUC_orig - 0.5)
```

| UCS Value | Interpretation |
|---|---|
| > 1.0 | Exceeds minimum compliance → Oracle **PASS** |
| 0.5 – 1.0 | Partial unlearning → borderline |
| ≈ 0.0 | No meaningful unlearning → Oracle **FAIL** |
| < 0.0 | Membership amplification detected |

---

## 🧪 Experimental Design

### Datasets
- **CIFAR-10**: 60k images, 10 classes — primary evaluation
- **CIFAR-100**: 60k images, 100 classes — harder multi-class task
- **SVHN**: 73k street number images — real-world domain shift

### Architectures
- **ResNet-18**: 11.2M params, skip connections (primary)
- **VGG-11**: 132M params, no BatchNorm, no skip connections
- **MobileNetV2**: 3.4M params, depthwise separable convolutions

### Unlearning Methods
- **Gradient Ascent (GA)**: Maximise forget loss + retain descent
- **SCRUB** (Kurmanji et al., NeurIPS 2023): Student-teacher KL divergence

### Adversary Strategies (35 fake models tested across all experiments)
| Strategy | Variants | Result |
|---|---|---|
| Gaussian noise | ε ∈ {0.0005–0.003} | 100% caught |
| FGSM perturbation | ε ∈ {0.001–0.010} | 100% caught |
| Weight pruning | p ∈ {0.05, 0.10, 0.20} | 100% caught |
| Retain-only fine-tuning | epochs ∈ {1, 2} | Correctly classified by outcome |

---

## 🔗 Smart Contracts

### `VeriForgotOracle.sol`
Issues, verifies, and revokes on-chain compliance certificates.
```solidity
// Issue certificate after oracle verification
oracle.issueCertificate(orgAddress, subjectHash, modelHash,
                        miaAUC_x1e6, ucs_x1e6, validityDays);

// Verify by anyone (data subject, regulator)
(bool valid, uint256 auc, uint256 ucs) = oracle.verifyCertificate(certId);
```

### `VeriForgotCommitment.sol`
SHA-256 weight commitment verification on-chain.
```solidity
// Pre-unlearning: store commitment
commitment.storeCommitment(sha256Hash);

// Post-unlearning: verify by any party
bool compliant = commitment.verifyCommitment(
    orgAddress, salt, indices, origValues, newValues
);
```

---

## 🔬 Related Work

| Paper | Venue | Gap VeriForgot Fills |
|---|---|---|
| Eisenhofer et al. | SaTML 2025 | Process proof only → we verify **outcome** |
| Tu et al. | NeurIPS 2025 | Theory only → we build the **full system** |
| Guo et al. | IEEE TIFS 2024 | Pre-deployment backdoor → **retroactive** verification |
| Zuo et al. | IEEE Trans. 2025 | Passive logging → **active oracle** certification |

---

## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
numpy>=1.24.0
scipy>=1.11.0
matplotlib>=3.7.0
tqdm>=4.65.0
```

---

## 📖 Citation

If you use this code, please cite:

```bibtex
@inproceedings{veriforgot2026,
  title     = {VeriForgot: Blockchain-Attested Machine Unlearning Verification
               via Membership Inference Oracle and Cryptographic Commitment},
  author    = {Borkot Tulla, Md. Hamid},
  booktitle = {Proceedings of the 6th International Conference on
               Cryptography and Blockchain (CRBL 2026)},
  year      = {2026}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built with ❤️ for GDPR-compliant machine learning | CRBL 2026</sub>
</p>
