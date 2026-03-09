"""
certificate.py
--------------
Unlearning Certificate interface for VeriForgot blockchain attestation layer.

This module provides the Python-side interface for:
  - Generating model commitment hashes
  - Constructing certificate payloads
  - Interacting with the smart contract (Web3)
  - Verifying existing certificates

NOTE: Smart contract deployment requires an Ethereum-compatible network.
For testing, use a local Hardhat/Ganache instance.
"""

import hashlib
import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class UnlearningCertificate:
    """
    Immutable Unlearning Certificate issued after successful verification.

    Fields:
        certificate_id      : Unique hex identifier
        subject_id_hash     : SHA3-256 hash of data subject identifier
        model_hash_orig     : H(theta_orig || nonce_orig)
        model_hash_new      : H(theta_new  || nonce_new)
        oracle_verdict      : 'PASS' or 'FAIL'
        oracle_auc          : MIA AUC score at time of verification
        oracle_threshold    : Threshold tau used
        zk_proof_hash       : Hash of the submitted ZK proof pi
        issuer_address      : Ethereum address of issuing smart contract
        timestamp_unix      : Unix timestamp of issuance
        block_number        : Blockchain block number (0 if off-chain)
    """
    certificate_id   : str
    subject_id_hash  : str
    model_hash_orig  : str
    model_hash_new   : str
    oracle_verdict   : str
    oracle_auc       : float
    oracle_threshold : float
    zk_proof_hash    : str
    issuer_address   : str  = "0x0000000000000000000000000000000000000000"
    timestamp_unix   : int  = field(default_factory=lambda: int(time.time()))
    block_number     : int  = 0

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    def verify_self_consistency(self) -> bool:
        """Basic self-consistency check (not cryptographic verification)."""
        return (
            self.oracle_verdict == "PASS" and
            self.oracle_auc < self.oracle_threshold and
            len(self.certificate_id) == 64 and
            len(self.subject_id_hash) == 64
        )


class CertificateGenerator:
    """
    Generates Unlearning Certificates after MIA oracle PASS.

    Usage:
        gen = CertificateGenerator()
        cert = gen.issue(
            subject_id       = "user_12345",
            theta_orig       = model_original.state_dict(),
            theta_new        = model_unlearned.state_dict(),
            oracle_result    = mia_oracle.evaluate(...),
            zk_proof_bytes   = zk_circuit.prove(...),
        )
    """
    def __init__(self, hash_algorithm: str = "sha3_256"):
        self.hash_algo = hash_algorithm

    def _hash(self, data: bytes) -> str:
        h = hashlib.new(self.hash_algo)
        h.update(data)
        return h.hexdigest()

    def _hash_model(self, state_dict, nonce: bytes) -> str:
        import io, torch
        buf = io.BytesIO()
        torch.save(state_dict, buf)
        return self._hash(buf.getvalue() + nonce)

    def issue(
        self,
        subject_id      : str,
        theta_orig      : dict,
        theta_new       : dict,
        oracle_result   : dict,
        zk_proof_bytes  : bytes = b"",
        issuer_address  : str   = "0x0000000000000000000000000000000000000000",
        block_number    : int   = 0,
    ) -> Optional["UnlearningCertificate"]:
        """
        Issue a certificate if oracle PASS. Returns None on FAIL.

        Args:
            subject_id      : Plain-text data subject identifier.
            theta_orig      : Original model state_dict.
            theta_new       : Unlearned model state_dict.
            oracle_result   : Output of MIAOracle.evaluate().
            zk_proof_bytes  : Serialized ZK proof bytes.
            issuer_address  : Ethereum smart contract address.
            block_number    : Block number of on-chain transaction.

        Returns:
            UnlearningCertificate if PASS, else None.
        """
        if not oracle_result.get("passed", False):
            print("[Certificate] Oracle FAIL — certificate NOT issued.")
            return None

        nonce_orig = os.urandom(32)
        nonce_new  = os.urandom(32)

        cert = UnlearningCertificate(
            certificate_id   = self._hash(
                (subject_id + str(time.time())).encode()),
            subject_id_hash  = self._hash(subject_id.encode()),
            model_hash_orig  = self._hash_model(theta_orig, nonce_orig),
            model_hash_new   = self._hash_model(theta_new,  nonce_new),
            oracle_verdict   = oracle_result["verdict"],
            oracle_auc       = oracle_result["auc_conf"],
            oracle_threshold = oracle_result["threshold"],
            zk_proof_hash    = self._hash(zk_proof_bytes),
            issuer_address   = issuer_address,
            block_number     = block_number,
        )
        print(f"[Certificate] Issued: {cert.certificate_id[:16]}...  "
              f"AUC={cert.oracle_auc:.4f}  Verdict=PASS")
        return cert
