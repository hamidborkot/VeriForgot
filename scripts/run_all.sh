#!/bin/bash
# VeriForgot — Full Experiment Pipeline
# GPU required (T4 or better recommended)
# Total runtime: ~3.5 hours

set -e

echo "======================================="
echo " VeriForgot Full Experiment Pipeline"
echo "======================================="

cd experiments/

echo ""
echo "[1/4] CIFAR-10 / ResNet-18 (main evaluation)"
python exp_main.py

echo ""
echo "[2/4] CIFAR-100 + SVHN / ResNet-18 (cross-dataset)"
python exp_datasets.py

echo ""
echo "[3/4] VGG-11 + MobileNetV2 / CIFAR-10 (cross-architecture)"
python exp_architectures.py

echo ""
echo "[4/4] GA vs SCRUB method-agnostic comparison"
python exp_scrub.py

echo ""
echo "======================================="
echo " All experiments complete!"
echo " Results saved in: results/"
echo "======================================="
