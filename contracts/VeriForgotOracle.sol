// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title  VeriForgot Oracle — On-chain GDPR Machine Unlearning Certificate
/// @author Md. Hamid Borkot Tulla
/// @notice Issues, verifies, and revokes compliance certificates per GDPR Article 17

contract VeriForgotOracle {

    address public immutable owner;

    struct Certificate {
        address  organisation;
        bytes32  dataSubjectHash;   // keccak256(subject identifier)
        bytes32  modelHash;         // keccak256(weight commitment sample)
        uint256  miaAUC;            // AUC scaled x1e6  (e.g. 0.489 -> 489000)
        uint256  ucsScore;          // UCS scaled x1e6  (e.g. 1.12  -> 1120000)
        uint256  issuedAt;
        uint256  expiresAt;         // 0 = perpetual
        bool     valid;
    }

    mapping(bytes32 => Certificate) public certificates;
    mapping(address => bytes32[])   public orgCertificates;

    uint256 public constant AUC_THRESHOLD = 570_000;   // 0.57 x 1e6
    uint256 public constant UCS_THRESHOLD = 1_000_000; // 1.0  x 1e6

    // -----------------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------------
    event CertificateIssued(
        bytes32 indexed certId,
        address indexed organisation,
        bytes32         dataSubjectHash,
        uint256         miaAUC,
        uint256         ucsScore,
        uint256         issuedAt
    );
    event CertificateRevoked(bytes32 indexed certId, address indexed revokedBy);

    // -----------------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------------
    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "VeriForgot: not authorised");
        _;
    }

    // -----------------------------------------------------------------------
    // Issue certificate
    // -----------------------------------------------------------------------
    /// @notice Issue a compliance certificate after off-chain oracle verification
    /// @param _org           Organisation that performed unlearning
    /// @param _subjectHash   keccak256 of data subject identifier
    /// @param _modelHash     keccak256 of sampled model weights (commitment)
    /// @param _miaAUC        MIA AUC x1e6 — must be < AUC_THRESHOLD
    /// @param _ucsScore      UCS x1e6 — must be >= UCS_THRESHOLD
    /// @param _validityDays  Certificate validity in days (0 = no expiry)
    function issueCertificate(
        address _org,
        bytes32 _subjectHash,
        bytes32 _modelHash,
        uint256 _miaAUC,
        uint256 _ucsScore,
        uint256 _validityDays
    ) external onlyOwner returns (bytes32 certId) {
        require(_miaAUC  < AUC_THRESHOLD,  "VeriForgot: MIA AUC exceeds threshold");
        require(_ucsScore >= UCS_THRESHOLD, "VeriForgot: UCS below minimum");

        certId = keccak256(abi.encodePacked(
            _org, _subjectHash, _modelHash, block.timestamp
        ));

        uint256 expiry = _validityDays == 0
            ? 0
            : block.timestamp + _validityDays * 1 days;

        certificates[certId] = Certificate({
            organisation:    _org,
            dataSubjectHash: _subjectHash,
            modelHash:       _modelHash,
            miaAUC:          _miaAUC,
            ucsScore:        _ucsScore,
            issuedAt:        block.timestamp,
            expiresAt:       expiry,
            valid:           true
        });

        orgCertificates[_org].push(certId);

        emit CertificateIssued(
            certId, _org, _subjectHash, _miaAUC, _ucsScore, block.timestamp
        );
    }

    // -----------------------------------------------------------------------
    // Verify certificate
    // -----------------------------------------------------------------------
    /// @notice Verify a certificate — callable by anyone (data subject, regulator)
    function verifyCertificate(bytes32 _certId)
        external view
        returns (bool isValid, uint256 miaAUC, uint256 ucsScore)
    {
        Certificate storage c = certificates[_certId];
        bool notExpired = (c.expiresAt == 0) || (block.timestamp <= c.expiresAt);
        isValid  = c.valid && notExpired;
        miaAUC   = c.miaAUC;
        ucsScore = c.ucsScore;
    }

    // -----------------------------------------------------------------------
    // Revoke certificate
    // -----------------------------------------------------------------------
    /// @notice Revoke a certificate (e.g. model retrained on same subject)
    function revokeCertificate(bytes32 _certId) external onlyOwner {
        require(certificates[_certId].valid, "VeriForgot: already revoked");
        certificates[_certId].valid = false;
        emit CertificateRevoked(_certId, msg.sender);
    }

    // -----------------------------------------------------------------------
    // View helpers
    // -----------------------------------------------------------------------
    function getOrgCertificates(address _org)
        external view returns (bytes32[] memory)
    {
        return orgCertificates[_org];
    }

    function isCompliant(address _org) external view returns (bool) {
        bytes32[] storage ids = orgCertificates[_org];
        if (ids.length == 0) return false;
        bytes32 latest = ids[ids.length - 1];
        Certificate storage c = certificates[latest];
        return c.valid && ((c.expiresAt == 0) || (block.timestamp <= c.expiresAt));
    }
}
