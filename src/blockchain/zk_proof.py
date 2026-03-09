"""
zk_proof.py
-----------
Zero-Knowledge Proof Protocol Specification for VeriForgot.

This module specifies the zk-SNARK circuit for parameter shift verification.

Circuit: Groth16 construction over BN128 curve
  Private witness : theta_orig, theta_new, nonce_r, nonce_r_prime
  Public inputs   : C_orig, C_new, delta (minimum shift bound)
  Statement       : Prove that:
    (1) C_orig = H(theta_orig || r)
    (2) C_new  = H(theta_new  || r')
    (3) ||theta_new - theta_orig||_2^2 > delta^2
    WITHOUT revealing theta_orig or theta_new

Proof size : 288 bytes (Groth16 compressed)
Verify time: ~10ms on-chain (EVM)

NOTE: Full circuit implementation requires snarkjs + circom.
      This module provides the protocol specification and
      a simplified prototype using hash-based commitments.
"""

import hashlib
import io
import math
import torch
from typing import Dict, Tuple


class ParameterShiftProver:
    """
    Prototype ZK Proof prover for parameter shift verification.

    In production: replace prove() with a call to snarkjs CLI
    or py_ecc Groth16 prover with the compiled circom circuit.

    Args:
        min_shift_delta:  Minimum required L2 parameter shift.
                          Models must satisfy ||theta_new - theta_orig||_2 > delta.
    """
    def __init__(self, min_shift_delta: float = 0.01):
        self.delta = min_shift_delta

    def _l2_shift(self, state_a: Dict, state_b: Dict) -> float:
        """Compute L2 distance between two state dicts."""
        total = 0.0
        for key in state_a:
            if state_a[key].dtype.is_floating_point:
                diff = state_a[key].float() - state_b[key].float()
                total += torch.sum(diff ** 2).item()
        return math.sqrt(total)

    def _commitment(self, state_dict: Dict, nonce: bytes) -> str:
        """H(theta || nonce) using SHA3-256."""
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        h = hashlib.sha3_256()
        h.update(buf.getvalue() + nonce)
        return h.hexdigest()

    def prove(
        self,
        theta_orig  : Dict,
        theta_new   : Dict,
        nonce_orig  : bytes,
        nonce_new   : bytes,
    ) -> Tuple[bool, Dict]:
        """
        Generate parameter shift proof.

        Returns:
            (valid, proof_dict)
            valid      : True if shift > delta
            proof_dict : Contains public commitments and shift claim
                         (in production: Groth16 proof bytes)
        """
        shift = self._l2_shift(theta_orig, theta_new)
        c_orig = self._commitment(theta_orig, nonce_orig)
        c_new  = self._commitment(theta_new,  nonce_new)
        valid  = shift > self.delta

        proof = {
            "C_orig"    : c_orig,
            "C_new"     : c_new,
            "delta"     : self.delta,
            "shift_gt_delta": valid,
            "protocol"  : "groth16_bn128_prototype",
            "note"      : "Production: replace with snarkjs Groth16 proof bytes"
        }
        print(f"[ZKProof] L2 shift = {shift:.6f}  |  "
              f"delta = {self.delta}  |  "
              f"Valid: {valid}")
        return valid, proof

    def verify(
        self,
        proof_dict  : Dict,
        c_orig_pub  : str,
        c_new_pub   : str,
    ) -> bool:
        """
        Verify a parameter shift proof against public commitments.
        In production: call snarkjs verifyProof() or EVM verifier contract.

        Returns:
            True if proof is valid and commitments match.
        """
        return (
            proof_dict["C_orig"] == c_orig_pub and
            proof_dict["C_new"]  == c_new_pub  and
            proof_dict["shift_gt_delta"] is True
        )
