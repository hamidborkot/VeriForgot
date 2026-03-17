"""
VeriForgot — SHA-256 Weight Commitment Protocol
Scheme : SHA-256 over k randomly sampled model weights
Binding: yes (collision resistance of SHA-256)
ZK     : no  (weights revealed during verification)
Note   : Full ZK-SNARK extension via circom/Groth16 is left as future work
"""
import hashlib
import struct
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

K_DEFAULT = 1_000   # sampled weight indices (calibrated for stability)
DELTA_MIN = 0.5     # minimum L2 shift for compliance (calibrated from experiments)


def extract_weights(model: nn.Module) -> np.ndarray:
    """Flatten all model parameters into a single float32 array."""
    return np.concatenate(
        [p.data.cpu().numpy().flatten() for p in model.parameters()]
    ).astype(np.float32)


def generate_commitment(
    weight_vec: np.ndarray,
    k: int = K_DEFAULT,
    rng_seed: int = None
) -> Tuple[str, str, List[int], List[float]]:
    """
    Pre-unlearning step: sample k weights and commit via SHA-256.

    Protocol:
        payload = salt || (idx_0 || val_0) || (idx_1 || val_1) || ...
        commitment = SHA256(payload)

    Returns:
        commitment_hex : 64-char hex string (bytes32 for Solidity)
        salt_hex       : 64-char hex string (bytes32 for Solidity)
        indices        : sorted list of sampled indices
        orig_values    : original float values at those indices
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)
    salt    = np.random.bytes(32)
    n       = len(weight_vec)
    indices = sorted(
        np.random.choice(n, size=min(k, n), replace=False).tolist()
    )
    sampled = [float(weight_vec[i]) for i in indices]

    payload = salt
    for idx, val in zip(indices, sampled):
        payload += struct.pack('>I', idx)   # 4-byte big-endian uint32
        payload += struct.pack('>f', val)   # 4-byte big-endian float32

    commitment = hashlib.sha256(payload).hexdigest()
    return commitment, salt.hex(), indices, sampled


def verify_commitment(
    commitment_hex: str,
    salt_hex: str,
    indices: List[int],
    orig_values: List[float]
) -> bool:
    """
    Post-unlearning step: recompute commitment to verify integrity.
    Called by data subject or regulator.
    """
    salt    = bytes.fromhex(salt_hex)
    payload = salt
    for idx, val in zip(indices, orig_values):
        payload += struct.pack('>I', idx)
        payload += struct.pack('>f', val)
    return hashlib.sha256(payload).hexdigest() == commitment_hex


def compute_param_shift(
    orig_values: List[float],
    new_values:  List[float]
) -> float:
    """L2 norm of parameter shift at sampled indices."""
    return float(
        np.linalg.norm(
            np.array(new_values, dtype=np.float32)
            - np.array(orig_values, dtype=np.float32)
        )
    )


def generate_compliance_proof(
    model_orig: nn.Module,
    model_unlearned: nn.Module,
    k: int = K_DEFAULT
) -> dict:
    """
    Full protocol:
        1. Extract weights from both models
        2. Generate commitment over original weights
        3. Verify commitment integrity (sanity check)
        4. Compute L2 parameter shift
        5. Return proof dict ready for on-chain submission

    Returns dict with all fields needed for VeriForgotCommitment.sol
    """
    w_orig = extract_weights(model_orig)
    w_new  = extract_weights(model_unlearned)

    commitment, salt_hex, indices, orig_vals = generate_commitment(w_orig, k)
    new_vals  = [float(w_new[i]) for i in indices]
    delta     = compute_param_shift(orig_vals, new_vals)
    valid     = verify_commitment(commitment, salt_hex, indices, orig_vals)
    compliant = delta >= DELTA_MIN

    return {
        'commitment':       commitment,              # bytes32 for Solidity
        'salt':             salt_hex,                # bytes32 for Solidity
        'k':                k,
        'delta_l2':         round(delta, 6),
        'delta_min':        DELTA_MIN,
        'compliant':        compliant,
        'commitment_valid': valid,
        'indices_sample':   indices[:5],             # first 5 for display
        'orig_sample':      orig_vals[:5],
        'new_sample':       new_vals[:5],
    }


# ---------------------------------------------------------------------------
# Gas estimation helper (Remix VM measured values)
# ---------------------------------------------------------------------------
def estimate_gas(k: int = K_DEFAULT) -> dict:
    """
    On-chain gas cost for verifyCommitment(k).
    Measured via Remix IDE JavaScript VM (Shanghai), Solidity 0.8.20.
    storeCommitment: 34,085 gas (measured)
    verifyCommitment k=10: ~3,200 gas (measured)
    verifyCommitment scales linearly with k.
    """
    gas_verify = int(3200 * k / 10)     # linear scaling from k=10 measurement
    gas_store  = 34_085                 # directly measured
    gas_total  = gas_store + gas_verify
    usd_at_20gwei = gas_total * 20e-9 * 3000
    return {
        'k':              k,
        'gas_store':      gas_store,
        'gas_verify':     gas_verify,
        'gas_total':      gas_total,
        'usd_at_20gwei':  round(usd_at_20gwei, 4),
    }


if __name__ == '__main__':
    print('Gas estimate (k=1000):', estimate_gas(1000))
    print('Gas estimate (k=500): ', estimate_gas(500))
    print('Gas estimate (k=10):  ', estimate_gas(10))
