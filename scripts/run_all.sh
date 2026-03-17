#!/bin/bash
# VeriForgot — Full Experiment Pipeline
# GPU recommended (T4 or better); CPU also supported
# Total runtime: ~3.5 hours (GPU) / ~5 hours (CPU)

set -e

echo "======================================="
echo " VeriForgot Full Experiment Pipeline"
echo "======================================="

cd experiments/

echo ""
echo "[1/5] CIFAR-10 / ResNet-18 (main evaluation)"
python exp_main.py

echo ""
echo "[2/5] CIFAR-100 + SVHN / ResNet-18 (cross-dataset)"
python exp_datasets.py

echo ""
echo "[3/5] VGG-11 + MobileNetV2 / CIFAR-10 (cross-architecture)"
python exp_architectures.py

echo ""
echo "[4/5] GA vs SCRUB method-agnostic comparison"
python exp_scrub.py

echo ""
echo "[5/5] MobileNetV2 CPU-optimised run (3 seeds)"
python train_mobilenet.py
python ga_unlearn_mobilenet.py

echo ""
echo "======================================="
echo " All experiments complete!"
echo " Results saved in: results/"
echo "======================================="
