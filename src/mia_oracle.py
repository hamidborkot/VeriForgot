"""
mia_oracle.py
-------------
Membership Inference Attack (MIA) Oracle for VeriForgot compliance verification.

Repurposes MIA — traditionally a privacy ATTACK — as a compliance VERIFICATION
tool. Models that have genuinely unlearned D_forget should score near AUC=0.50
(indistinguishable from non-members). Models that fake compliance maintain
AUC close to the original ~0.59.

Key thresholds (calibrated on CIFAR-10 / ResNet-18):
  tau = 0.57  ->  95.0% oracle accuracy (TPR=100%, TNR=90%)
  tau = 0.58  -> 100.0% oracle accuracy (TPR=100%, TNR=100%)
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, Tuple


class MIAOracle:
    """
    Calibrated MIA Oracle for unlearning compliance verification.

    Args:
        threshold:  AUC threshold tau. Models with AUC < threshold PASS.
                    Default: 0.57 (95% accuracy). Use 0.58 for 100% accuracy.
        device:     Torch device for model inference.
    """
    def __init__(self, threshold: float = 0.57, device: str = "cuda"):
        self.threshold = threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss(reduction="none")

    def _get_confidence_scores(self, model, loader) -> np.ndarray:
        """P(true class | x) for each sample in loader."""
        model.eval(); scores = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                probs = torch.softmax(model(x), dim=1)
                scores.extend(probs[torch.arange(len(y)), y].cpu().numpy())
        return np.array(scores)

    def _get_loss_scores(self, model, loader) -> np.ndarray:
        """Per-sample cross-entropy loss for each sample in loader."""
        model.eval(); losses = []
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                losses.extend(
                    self.criterion(model(x), y).cpu().numpy()
                )
        return np.array(losses)

    def evaluate(self,
                 model,
                 forget_loader,
                 non_member_loader) -> Dict:
        """
        Run MIA evaluation and return compliance verdict.

        Args:
            model:              Candidate unlearned model (black-box access).
            forget_loader:      DataLoader for D_forget (membership targets).
            non_member_loader:  DataLoader for verified non-members.

        Returns:
            dict with keys:
                auc_conf   : float  AUROC using confidence scores
                auc_loss   : float  AUROC using loss scores
                mia_rate   : float  % of forget samples above non-member threshold
                conf_gap   : float  mean(member_conf) - mean(nonmember_conf)
                loss_gap   : float  mean(nonmember_loss) - mean(member_loss)
                passed     : bool   True if auc_conf < self.threshold
                verdict    : str    'PASS' or 'FAIL'
                threshold  : float  Oracle threshold used
        """
        mc = self._get_confidence_scores(model, forget_loader)
        nc = self._get_confidence_scores(model, non_member_loader)
        ml = self._get_loss_scores(model, forget_loader)
        nl = self._get_loss_scores(model, non_member_loader)

        labels   = np.concatenate([np.ones(len(mc)), np.zeros(len(nc))])
        auc_conf = float(roc_auc_score(labels, np.concatenate([mc, nc])))
        auc_loss = float(roc_auc_score(labels, np.concatenate([-ml, -nl])))

        threshold_conf = np.percentile(nc, 90)
        mia_rate = float(np.mean(mc > threshold_conf) * 100)

        passed  = auc_conf < self.threshold
        verdict = "PASS" if passed else "FAIL"

        return {
            "auc_conf"  : auc_conf,
            "auc_loss"  : auc_loss,
            "mia_rate"  : mia_rate,
            "conf_gap"  : float(np.mean(mc) - np.mean(nc)),
            "loss_gap"  : float(np.mean(nl) - np.mean(ml)),
            "passed"    : passed,
            "verdict"   : verdict,
            "threshold" : self.threshold,
            "member_conf_scores"     : mc,
            "nonmember_conf_scores"  : nc,
        }

    def batch_evaluate(self, models: Dict, forget_loader,
                       non_member_loader) -> Dict:
        """
        Evaluate multiple models at once.

        Args:
            models: dict {name: model}

        Returns:
            dict {name: evaluation_result}
        """
        results = {}
        for name, model in models.items():
            r = self.evaluate(model, forget_loader, non_member_loader)
            results[name] = r
            print(f"  [{name}]  AUC={r['auc_conf']:.4f}  "
                  f"ConfGap={r['conf_gap']:.4f}  "
                  f"Verdict={r['verdict']}")
        return results
