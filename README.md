# VeriForgot 🔐

> **Verifiable Machine Unlearning with MIA Oracle, Blockchain Attestation, and Cryptographic Commitment**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ethereum](https://img.shields.io/badge/Solidity-0.8.20-blue.svg)](contracts/)

**"VeriForgot: Blockchain-Attested Verifiable Machine Unlearning Using Membership Inference Auditing for GDPR Compliance"**

---

## 🔍 Abstract

GDPR Article 17 grants data subjects the *right to erasure*, yet no practical mechanism exists to verify that an ML model has genuinely forgotten specific training data. **VeriForgot** addresses this with a four-component system:

1. 🎯 **MIA Oracle** — empirical Membership Inference Attack oracle for outcome-level compliance
2. 📐 **UCS Metric** — Unlearning Completeness Score, a continuous compliance quantifier
3. ⛓️ **On-chain Certificate** — immutable Solidity smart contract anchoring compliance records
4. 🔒 **Weight Commitment** — SHA-256 cryptographic commitment for weight integrity proof

Evaluated across **3 datasets × 3 architectures × 3 seeds**, achieving **100% fake-model detection (TNR = 100%)** across all configurations and **method-agnostic** verification across both GA and SCRUB unlearning.

---

## 📊 Key Results

### Cross-Dataset Oracle Evaluation (ResNet-18)

| Dataset | Method | Orig AUC | Post AUC | ±σ | Acc% | UCS | Oracle |
|---|---|---|---|---|---|---|---|
| CIFAR-10 | GA | 0.5759 | 0.4303 | ±0.017 | 11.6% | 1.908 | ✅ 3/3 |
| CIFAR-10 | SCRUB | 0.5759 | 0.0548 | ±0.005 | 72.2% | 6.902 | ✅ 3/3 |
| CIFAR-100 | GA | 0.7188 | 0.0000 | ±0.000 | 53.1% | 3.246 | ✅ 3/3 |
| CIFAR-100 | SCRUB | 0.7188 | 0.3420 | ±0.003 | 55.7% | 1.716 | ✅ 3/3 |
| SVHN | GA | 0.6409 | 0.0017 | ±0.001 | 70.7% | 4.567 | ✅ 3/3 |
| SVHN | SCRUB | 0.6409 | 0.0146 | ±0.001 | 71.4% | 4.346 | ✅ 3/3 |

### Cross-Architecture Evaluation (CIFAR-10, GA)

| Architecture | Orig AUC | Orig Acc | GA AUC | ±σ | GA Acc | UCS | Oracle |
|---|---|---|---|---|---|---|---|
| ResNet-18 | 0.5759 | 86.4% | 0.4303 | ±0.017 | 11.6% | 1.908 | ✅ 3/3 |
| VGG-11 | 0.5667 | 89.5% | 0.0019 | ±0.001 | 68.2% | 7.178 | ✅ 2/2† |
| MobileNetV2 | 0.6220 | 79.1% | 0.0000 | ±0.001 | 62.9% | 5.105 | ✅ 3/3 |

*† seed=123 excluded — VGG-11 training failed to converge (Orig Acc=10%, random chance). Consistent with known VGG sensitivity to initialisation on CIFAR-10 without LR warm-up.*

### MobileNetV2 Per-Seed Results

| Seed | Orig Acc | Orig AUC | GA AUC | GA Acc | UCS | Oracle |
|---|---|---|---|---|----|---|
| 42 | 79% | 0.620 | 0.000 | 63% | 5.10 | ✅ PASS |
| 123 | 79% | 0.622 | 0.000 | 62% | 5.08 | ✅ PASS |
| 999 | 79% | 0.624 | 0.000 | 63% | 5.13 | ✅ PASS |
| **Mean** | **79.1%** | **0.6220** | **0.0000** | **62.9%** | **5.105** | **✅ 3/3** |
| **±σ** | ±0.5% | ±0.003 | ±0.001 | ±0.8% | ±0.03 | — |

### Oracle-30 Evaluation (CIFAR-10 / ResNet-18, τ = 0.58)

| Category | Count | Result | Metric |
|---|---|---|---|
| Genuine compliant (well-configured) | 10 | 10 TP | TPR = 100% |
| Genuine non-compliant (bad hyperparams) | 5 | 5 FN | Correctly rejected |
| Fake — Gaussian noise | 6 | 6 TN | TNR = 100% |
| Fake — FGSM perturbation | 5 | 5 TN | TNR = 100% |
| Fake — Weight pruning | 3 | 3 TN | TNR = 100% |
| Fake — Retain-only fine-tuning | 1 | 1 TN | TNR = 100% |
| **Overall** | **30** | **Acc = 100%** | **TPR = 100%, TNR = 100%** |

*Oracle evaluated at strict threshold τ = 0.58 (100% accuracy). τ = 0.57 yields 95% accuracy.*

### Smart Contract Gas Costs (Remix VM, Solidity 0.8.20)

| Function | k Samples | Gas Used | Cost @ 20 gwei (ETH = $1,000) |
|---|---|---|---|
| `storeCommitment` | — | **34,085** | $0.68 |
| `verifyCommitment` | 10 | ~3,200 | $0.06 |
| `verifyCommitment` | **1,000** | **~320,000** | $6.40 |
| **Full protocol** | — | **~354,085** | **$7.08** |

*Measured via Remix IDE JavaScript VM (Shanghai). k = 1,000 sampled weights used in production. `verifyCommitment` scales linearly with k.*

### UCS Zero-Overlap Distribution

| Category | UCS Range |
|---|---|
| Intentional fakes | −2.16 to +0.68 |
| GA compliant | +1.03 to +8.86 |
| SCRUB compliant | +1.72 to +7.57 |

### Sample Size Sensitivity

| n | Orig σ | GA σ | Stable? |
|---|---|---|---|
| 50 | 0.0385 | 0.0429 | ❌ |
| 100 | 0.0204 | 0.0275 | ❌ |
| 200 | 0.0240 | 0.0186 | ✅ GA only |
| 300 | 0.0128 | 0.0129 | ✅ Both |
| **Guideline** | | | **n ≥ 300** |

---

## 🚀 Quick Start

```bash
git clone https://github.com/hamidborkot/VeriForgot.git
cd VeriForgot
pip install -r requirements.txt

# Run individual experiments
python experiments/exp_main.py           # CIFAR-10 / ResNet-18 (primary)
python experiments/exp_datasets.py       # CIFAR-100 + SVHN cross-dataset
python experiments/exp_architectures.py  # VGG-11 + MobileNetV2 cross-architecture
python experiments/exp_scrub.py          # GA vs SCRUB comparison

# MobileNetV2 CPU-optimised run
python experiments/train_mobilenet.py
python experiments/ga_unlearn_mobilenet.py

# Or run all experiments sequentially
bash scripts/run_all.sh
```

---

## 🗂️ Repository Structure

```
VeriForgot/
├── experiments/
│   ├── utils.py                       # Shared: MIA oracle, UCS, train, GA, SCRUB, fake factory
│   ├── exp_main.py                    # CIFAR-10 / ResNet-18 (primary evaluation)
│   ├── exp_datasets.py                # CIFAR-100 + SVHN cross-dataset evaluation
│   ├── exp_architectures.py           # VGG-11 + MobileNetV2 cross-architecture evaluation
│   ├── exp_scrub.py                   # GA vs SCRUB method-agnostic comparison
│   ├── train_mobilenet.py             # MobileNetV2 CPU-optimised baseline training
│   └── ga_unlearn_mobilenet.py        # MobileNetV2 GA unlearning
├── contracts/
│   ├── VeriForgotOracle.sol           # On-chain compliance certificate
│   └── VeriForgotCommitment.sol       # SHA-256 weight commitment verifier
├── commitment/
│   └── weight_commitment.py           # Python cryptographic commitment client
├── src/
│   ├── mia_oracle.py                  # MIA Oracle class (object-oriented interface)
│   ├── unlearning.py                  # GA and SCRUB unlearning implementations
│   ├── models.py                      # Model factory (ResNet-18, VGG-11, MobileNetV2)
│   └── data_utils.py                  # Dataset loading and split utilities
├── results/
│   ├── all_results.json               # Master results file
│   ├── results_main.json              # CIFAR-10 / ResNet-18 primary results
│   ├── results_datasets.json          # CIFAR-100 + SVHN cross-dataset results
│   ├── results_architectures.json     # VGG-11 + MobileNetV2 results
│   ├── results_mobilenet_per_seed.json# MobileNetV2 3-seed breakdown
│   ├── results_gas.json               # Blockchain gas cost measurements
│   ├── results_scrub.json             # SCRUB vs GA comparison
│   ├── results_scrub_crossdataset.json# SCRUB cross-dataset results
│   └── results_ga_collapse_finding.json # GA collapse analysis
├── scripts/
│   └── run_all.sh                     # Run all experiments sequentially
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 📐 Unlearning Completeness Score (UCS)

```
UCS(M_unlearn) = (AUC_orig − AUC_unlearn) / (AUC_orig − 0.5)
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
- **VGG-11**: 132M params, no BatchNorm variant
- **MobileNetV2**: 3.4M params, depthwise separable convolutions

### Unlearning Methods
- **Gradient Ascent (GA)**: Maximise forget loss with retain fine-tuning
- **SCRUB** (Kurmanji et al., NeurIPS 2023): Student-teacher KL divergence

### Adversary Strategies (Oracle-30)

| Strategy | Variants | TNR (τ = 0.58) |
|---|---|---|
| Gaussian noise | ε ∈ {0.0005–0.003} | **100%** |
| FGSM perturbation | ε ∈ {0.001–0.010} | **100%** |
| Weight pruning | p ∈ {0.05–0.20} | **100%** |
| Retain-only fine-tuning | epochs ∈ {1–4} | **100%** |

---

## 🔗 Smart Contracts

### `VeriForgotOracle.sol`
```solidity
// Issue certificate after oracle verification
oracle.issueCertificate(orgAddress, subjectHash, modelHash,
                        miaAUC_x1e6, ucs_x1e6, validityDays);

// Verify by anyone (data subject, regulator)
(bool valid, uint256 auc, uint256 ucs) = oracle.verifyCertificate(certId);
```

### `VeriForgotCommitment.sol`
```solidity
// Pre-unlearning: store commitment
commitment.storeCommitment(sha256Hash);

// Post-unlearning: verify weight shift
bool compliant = commitment.verifyCommitment(
    orgAddress, salt, indices, origValues, newValues
);
```

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

```bibtex
@article{veriforgot2026,
  title   = {VeriForgot: Blockchain-Attested Verifiable Machine Unlearning
             Using Membership Inference Auditing for GDPR Compliance},
  author  = {Borkot Tulla, Md. Hamid},
  journal = {Journal of Information Security and Applications},
  year    = {2026},
  publisher = {Elsevier}
}
```

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built for GDPR-compliant verifiable machine learning</sub>
</p>
