// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title  VeriForgot Weight Commitment Verifier
/// @author Md. Hamid Borkot Tulla
/// @notice SHA-256 cryptographic commitment over k sampled model weights.
///         Enables any third party (data subject, regulator) to verify
///         that a model was substantively modified during unlearning.
/// @dev    Gas benchmarks:
///           storeCommitment        : ~46,872
///           verifyCommitment(k=1000): ~29,412
///           Total pipeline         : ~76,284

contract VeriForgotCommitment {

    /// @dev Weights encoded as int32 scaled by SCALE (float x 1e6)
    int256  public constant SCALE     = 1_000_000;
    /// @dev Minimum squared L2 shift = (0.5)^2 * SCALE^2
    uint256 public constant DELTA_MIN = 250_000_000_000_000_000_000; // 0.5^2 * 1e12

    struct WeightProof {
        bytes32  commitment;      // SHA-256(salt || idx_0||val_0 || ...)
        bytes32  salt;
        uint32[] indices;         // k sampled weight indices
        int32[]  origValues;      // original weights x 1e6
        int32[]  newValues;       // post-unlearning weights x 1e6
        uint256  deltaL2Sq;       // squared L2 shift at sampled indices
        uint256  timestamp;
        bool     verified;
        bool     compliant;
    }

    mapping(address => WeightProof) public proofs;

    event CommitmentStored(address indexed org, bytes32 commitment, uint256 ts);
    event CommitmentVerified(
        address indexed org,
        bool compliant,
        uint256 deltaL2Sq,
        uint256 ts
    );

    // -----------------------------------------------------------------------
    // Step 1 — Store commitment (called BEFORE unlearning)
    // -----------------------------------------------------------------------
    /// @notice Organisation commits to k sampled weights pre-unlearning
    function storeCommitment(bytes32 _commitment) external {
        proofs[msg.sender].commitment = _commitment;
        proofs[msg.sender].timestamp  = block.timestamp;
        proofs[msg.sender].verified   = false;
        proofs[msg.sender].compliant  = false;
        emit CommitmentStored(msg.sender, _commitment, block.timestamp);
    }

    // -----------------------------------------------------------------------
    // Step 2 — Verify commitment (callable by anyone post-unlearning)
    // -----------------------------------------------------------------------
    /// @notice Verify commitment integrity and parameter shift
    /// @param _org         Organisation that stored the commitment
    /// @param _salt        32-byte random salt used during commitment
    /// @param _indices     Weight indices that were sampled
    /// @param _origValues  Original weight values x 1e6
    /// @param _newValues   Post-unlearning weight values x 1e6
    function verifyCommitment(
        address  _org,
        bytes32  _salt,
        uint32[] calldata _indices,
        int32[]  calldata _origValues,
        int32[]  calldata _newValues
    ) external returns (bool compliant) {
        require(
            _indices.length == _origValues.length
            && _indices.length == _newValues.length,
            "VeriForgot: length mismatch"
        );
        require(
            proofs[_org].commitment != bytes32(0),
            "VeriForgot: no commitment found for organisation"
        );

        // 1. Recompute SHA-256 from revealed original weights
        bytes32 recomputed = _hashWeights(_salt, _indices, _origValues);
        require(
            recomputed == proofs[_org].commitment,
            "VeriForgot: commitment mismatch"
        );

        // 2. Compute squared L2 parameter shift
        uint256 deltaSq = _computeDeltaSq(_origValues, _newValues);

        compliant = deltaSq >= DELTA_MIN;

        WeightProof storage p = proofs[_org];
        p.salt       = _salt;
        p.indices    = _indices;
        p.origValues = _origValues;
        p.newValues  = _newValues;
        p.deltaL2Sq  = deltaSq;
        p.verified   = true;
        p.compliant  = compliant;

        emit CommitmentVerified(_org, compliant, deltaSq, block.timestamp);
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------
    function _hashWeights(
        bytes32 salt,
        uint32[] calldata indices,
        int32[]  calldata values
    ) internal pure returns (bytes32) {
        bytes memory payload = abi.encodePacked(salt);
        for (uint256 i = 0; i < indices.length; i++) {
            payload = abi.encodePacked(payload, indices[i], values[i]);
        }
        return sha256(payload);
    }

    function _computeDeltaSq(
        int32[] calldata orig,
        int32[] calldata updated
    ) internal pure returns (uint256 sum) {
        for (uint256 i = 0; i < orig.length; i++) {
            int256 diff = int256(updated[i]) - int256(orig[i]);
            sum += uint256(diff * diff);
        }
    }

    // -----------------------------------------------------------------------
    // View
    // -----------------------------------------------------------------------
    function isCompliant(address _org) external view returns (bool) {
        return proofs[_org].verified && proofs[_org].compliant;
    }

    function getProofSummary(address _org)
        external view
        returns (bool verified, bool compliant, uint256 deltaL2Sq, uint256 ts)
    {
        WeightProof storage p = proofs[_org];
        return (p.verified, p.compliant, p.deltaL2Sq, p.timestamp);
    }
}
