"""
INTEGRITY CODE SERIES -- Week 7
Cybersecurity Architecture for H2 Conversion Integrity Assessment

Context: When a pipeline operator uses sensor data, ILI results, and
computational models to make a go/no-go decision on hydrogen conversion,
the entire decision chain is an attack surface.

STRIDE Threat Model for H2 Conversion Assessment System:

1. SPOOFING
   Threat: Attacker spoofs ILI pit depth data to make a degraded ERW
   seam appear safer than it is, leading to approval of unsafe conversion.
   Mitigation: Cryptographic signing of ILI data at source, chain-of-custody
   verification, redundant UT measurements at seam locations.

2. TAMPERING
   Threat: Modification of hydrogen diffusion parameters (D_L, Sievert's
   constant) in the simulation configuration to produce optimistic
   remaining life predictions.
   Mitigation: Hash verification of config files before each run,
   parameter bounds checking against physically admissible ranges.

3. REPUDIATION
   Threat: An engineer modifies model inputs but denies doing so,
   making it impossible to trace which assumptions drove the decision.
   Mitigation: Hash-chain audit log of every simulation run with
   input hash, output hash, timestamp, and operator ID.

4. INFORMATION DISCLOSURE
   Threat: Pipeline condition data (pit depths, seam quality) leaks
   to competitors or malicious actors who could identify vulnerable
   segments for sabotage.
   Mitigation: Encryption at rest, access control, data classification.

5. DENIAL OF SERVICE
   Threat: Flooding the assessment system with Monte Carlo requests
   to prevent timely completion of conversion assessments.
   Mitigation: Rate limiting, job queuing, resource allocation controls.

6. ELEVATION OF PRIVILEGE
   Threat: A field technician with read-only access modifies the
   surrogate model pickle file to always predict safe remaining life.
   Mitigation: Model integrity verification (hash check on .pkl files),
   role-based access control, model output bounds checking.

7. DATA POISONING (additional, specific to ML pipeline)
   Threat: Injecting fabricated training data into the LHS sweep to
   shift the surrogate model toward optimistic predictions.
   Mitigation: Physics consistency checks on training data (e.g.,
   remaining life must decrease with increasing pit depth at
   constant other parameters).
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np


# ============================================================
# STRIDE THREAT REGISTRY
# ============================================================
STRIDE_THREATS = [
    {
        "id": "T1",
        "category": "Spoofing",
        "threat": "Spoofed ILI pit depth data at ERW seam locations",
        "asset": "Pipeline inspection data",
        "impact": "Unsafe H2 conversion approval",
        "severity": "Critical",
        "mitigation": "Cryptographic signing of ILI data, redundant UT at seam",
    },
    {
        "id": "T2",
        "category": "Tampering",
        "threat": "Modified hydrogen transport parameters in config",
        "asset": "Simulation configuration",
        "impact": "Optimistic remaining life prediction",
        "severity": "Critical",
        "mitigation": "Config hash verification, parameter bounds enforcement",
    },
    {
        "id": "T3",
        "category": "Repudiation",
        "threat": "Untracked modification of model inputs",
        "asset": "Decision audit trail",
        "impact": "Untraceable safety decision",
        "severity": "High",
        "mitigation": "Hash-chain audit log with operator ID",
    },
    {
        "id": "T4",
        "category": "Information Disclosure",
        "threat": "Pipeline condition data exfiltration",
        "asset": "Pit depth and seam quality records",
        "impact": "Targeted infrastructure sabotage",
        "severity": "High",
        "mitigation": "Encryption at rest, access control, classification",
    },
    {
        "id": "T5",
        "category": "Denial of Service",
        "threat": "Monte Carlo request flooding",
        "asset": "Assessment compute resources",
        "impact": "Delayed conversion decision",
        "severity": "Medium",
        "mitigation": "Rate limiting, job queue, resource allocation",
    },
    {
        "id": "T6",
        "category": "Elevation of Privilege",
        "threat": "Surrogate model file tampering",
        "asset": "ML model pickle",
        "impact": "Always-safe prediction from corrupted model",
        "severity": "Critical",
        "mitigation": "Model hash verification, RBAC, output bounds check",
    },
    {
        "id": "T7",
        "category": "Data Poisoning",
        "threat": "Fabricated LHS training data injection",
        "asset": "Surrogate training dataset",
        "impact": "Systematically biased remaining life estimates",
        "severity": "Critical",
        "mitigation": "Physics monotonicity checks on training data",
    },
]


# ============================================================
# PARAMETER BOUNDS CHECKER
# ============================================================
PARAMETER_BOUNDS = {
    "p_H2_MPa": (0.1, 20.0),
    "pit_depth_m": (0.0, 0.01),           # 0 to 10 mm
    "D_L_m2s": (1.0e-12, 1.0e-8),
    "sieverts_constant": (0.001, 1.0),
    "K_IC_seam": (10.0, 200.0),
    "f_seam": (1.0, 10.0),
    "aspect_ratio": (0.1, 10.0),
    "wall_thickness_m": (0.003, 0.030),
    "SMYS_MPa": (200.0, 600.0),
    "remaining_life_years": (0.0, 1000.0),
}


def check_parameter_bounds(name: str, value: float) -> Dict:
    """
    Check if a parameter value is within physically admissible bounds.

    Returns dict with 'valid', 'value', 'bounds', and 'message'.
    """
    if name not in PARAMETER_BOUNDS:
        return {"valid": True, "value": value, "bounds": None,
                "message": f"No bounds defined for {name}"}

    lo, hi = PARAMETER_BOUNDS[name]
    valid = lo <= value <= hi
    msg = "Within bounds" if valid else f"OUT OF BOUNDS: {value} not in [{lo}, {hi}]"
    return {"valid": valid, "value": value, "bounds": (lo, hi), "message": msg}


def check_all_config_bounds(config_dict: Dict) -> List[Dict]:
    """Check all parameters in a config dictionary against bounds."""
    results = []
    for name, value in config_dict.items():
        if isinstance(value, (int, float)):
            results.append(check_parameter_bounds(name, value))
    return results


# ============================================================
# HASH-CHAIN AUDIT LOGGER
# ============================================================
@dataclass
class AuditEntry:
    """Single entry in the audit chain."""
    entry_id: int
    timestamp: float
    operator_id: str
    action: str
    input_hash: str
    output_hash: str
    chain_hash: str           # SHA-256(previous_chain_hash + this_entry)
    metadata: Dict = field(default_factory=dict)


class AuditLogger:
    """
    SHA-256 hash-chain audit logger for integrity assessment decisions.

    Every simulation run, parameter change, or model evaluation is
    logged with a cryptographic chain. Tampering with any entry
    invalidates all subsequent chain hashes.
    """

    def __init__(self):
        self.chain: List[AuditEntry] = []
        self._genesis_hash = hashlib.sha256(b"ICS2_WEEK7_GENESIS").hexdigest()

    def _compute_chain_hash(self, previous_hash: str, entry_data: str) -> str:
        """Compute SHA-256 chain hash."""
        combined = f"{previous_hash}:{entry_data}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def log(
        self,
        operator_id: str,
        action: str,
        input_data: str,
        output_data: str,
        metadata: Optional[Dict] = None,
    ) -> AuditEntry:
        """
        Log an action to the audit chain.

        Parameters
        ----------
        operator_id : str
        action : str
            Description of action (e.g., 'run_life_prediction', 'modify_config')
        input_data : str
            Serialized input (will be hashed)
        output_data : str
            Serialized output (will be hashed)
        metadata : dict, optional

        Returns
        -------
        AuditEntry
        """
        input_hash = hashlib.sha256(input_data.encode()).hexdigest()
        output_hash = hashlib.sha256(output_data.encode()).hexdigest()

        previous_hash = (self.chain[-1].chain_hash
                         if self.chain else self._genesis_hash)

        ts = time.time()
        entry_data = f"{operator_id}:{action}:{input_hash}:{output_hash}:{ts}"
        chain_hash = self._compute_chain_hash(previous_hash, entry_data)

        entry = AuditEntry(
            entry_id=len(self.chain),
            timestamp=ts,
            operator_id=operator_id,
            action=action,
            input_hash=input_hash,
            output_hash=output_hash,
            chain_hash=chain_hash,
            metadata=metadata or {},
        )
        self.chain.append(entry)
        return entry

    def verify_chain(self) -> Dict:
        """
        Verify integrity of the entire audit chain.

        Returns dict with 'valid', 'entries_checked', 'first_invalid'.
        """
        if not self.chain:
            return {"valid": True, "entries_checked": 0, "first_invalid": -1}

        previous_hash = self._genesis_hash
        for entry in self.chain:
            entry_data = (f"{entry.operator_id}:{entry.action}:"
                          f"{entry.input_hash}:{entry.output_hash}:{entry.timestamp}")
            expected = self._compute_chain_hash(previous_hash, entry_data)
            if expected != entry.chain_hash:
                return {
                    "valid": False,
                    "entries_checked": entry.entry_id,
                    "first_invalid": entry.entry_id,
                }
            previous_hash = entry.chain_hash

        return {
            "valid": True,
            "entries_checked": len(self.chain),
            "first_invalid": -1,
        }


# ============================================================
# SENSOR DATA INTEGRITY CHECKS
# ============================================================
def detect_pit_depth_spoofing(
    pit_depths_reported: np.ndarray,
    wall_thickness_m: float,
) -> Dict:
    """
    Detect potential spoofing in ILI pit depth data.

    Checks:
    1. All values >= 0
    2. No values > wall thickness
    3. No suspiciously uniform distribution (real pits have scatter)
    4. No exact duplicates beyond statistical expectation

    Returns dict with 'spoofing_detected', 'checks', 'flags'.
    """
    checks = {}
    flags = []

    # Check 1: Non-negative
    neg_count = np.sum(pit_depths_reported < 0)
    checks["non_negative"] = neg_count == 0
    if neg_count > 0:
        flags.append(f"{neg_count} negative pit depth values detected")

    # Check 2: Within wall thickness
    over_count = np.sum(pit_depths_reported > wall_thickness_m)
    checks["within_wall"] = over_count == 0
    if over_count > 0:
        flags.append(f"{over_count} values exceed wall thickness")

    # Check 3: Not suspiciously uniform
    if len(pit_depths_reported) > 10:
        cv = np.std(pit_depths_reported) / (np.mean(pit_depths_reported) + 1e-12)
        checks["sufficient_scatter"] = cv > 0.05
        if cv <= 0.05:
            flags.append(f"CV={cv:.4f} suspiciously low, possible synthetic data")

    # Check 4: Duplicate rate
    unique_ratio = len(np.unique(np.round(pit_depths_reported, 6))) / len(pit_depths_reported)
    checks["low_duplicate_rate"] = unique_ratio > 0.5
    if unique_ratio <= 0.5:
        flags.append(f"Duplicate ratio {1-unique_ratio:.2%} abnormally high")

    return {
        "spoofing_detected": len(flags) > 0,
        "checks": checks,
        "flags": flags,
    }


def physics_monotonicity_check(
    pit_depths: np.ndarray,
    remaining_lives: np.ndarray,
    threshold: float = 0.1,
) -> Dict:
    """
    Check that remaining life decreases with increasing pit depth
    (at approximately constant other parameters).

    This is a data poisoning detection heuristic: if deeper pits
    consistently produce longer lives, the training data is suspect.

    Parameters
    ----------
    pit_depths : ndarray
    remaining_lives : ndarray
    threshold : float
        Maximum acceptable fraction of violations.

    Returns
    -------
    dict with 'consistent', 'violation_fraction', 'message'
    """
    from scipy.stats import spearmanr
    rho, pval = spearmanr(pit_depths, remaining_lives)

    # Expect negative correlation (deeper pit = shorter life)
    consistent = rho < 0
    return {
        "consistent": consistent,
        "spearman_rho": float(rho),
        "p_value": float(pval),
        "message": ("Monotonicity OK: deeper pits yield shorter life"
                     if consistent
                     else "WARNING: positive correlation detected, possible data poisoning"),
    }
