"""``arx.audit`` — statistical auditing of benchmark claims.

This sub-package deliberately has **no** dependency on the rest of arx: no
LLM client, no agent loop, no network. It installs and runs standalone, which
is what lets a reviewer cite the gate without adopting the framework.

    from arx.audit import audit_leaderboard, Verdict

    report = audit_leaderboard(votes, {"sys-a": 0.91, "sys-b": 0.88})
    if report.verdict is not Verdict.READABLE:
        ...  # do not write this claim into a report
"""

from .auditor import AuditConfig, audit_item_sets, audit_leaderboard
from .floor import (
    DawidSkeneResult,
    column_majority,
    dawid_skene,
    external_gold_floor,
    floor_estimate,
    identifiability_diagnostics,
    loo_floor,
)
from .goodhart import (
    VERDICT_UTILITY,
    GoodhartMonitor,
    ProxyTargetMonitor,
    pearson,
    spearman,
)
from .power import (
    binomial_se,
    cluster_bootstrap_mde,
    mde_from_se,
    paired_diff_se,
    power_at_effect,
    z_for_confidence,
    z_for_power,
)
from .prereg import (
    MultiplicityLedger,
    Preregistration,
    check_preregistration,
    one_sided_p_from_z,
    plan_k_max,
    prereg_fingerprint,
    read_preregistration,
    sha256_text,
    two_sided_p_from_z,
)
from .sequential import (
    SequentialState,
    calibrate_scale,
    effective_alpha,
    fixed_sample_boundary,
    obf_boundaries,
    union_bound_rate,
    z_from_effect,
)
from .verdict import AuditReport, UndecidableReason, Verdict

__version__ = "0.3.0"

__all__ = [
    # verdict / report
    "AuditReport",
    "UndecidableReason",
    "Verdict",
    # auditor
    "AuditConfig",
    "audit_leaderboard",
    "audit_item_sets",
    # floor
    "DawidSkeneResult",
    "column_majority",
    "dawid_skene",
    "external_gold_floor",
    "floor_estimate",
    "identifiability_diagnostics",
    "loo_floor",
    # power
    "binomial_se",
    "cluster_bootstrap_mde",
    "mde_from_se",
    "paired_diff_se",
    "power_at_effect",
    "z_for_confidence",
    "z_for_power",
    # sequential
    "SequentialState",
    "calibrate_scale",
    "effective_alpha",
    "fixed_sample_boundary",
    "obf_boundaries",
    "union_bound_rate",
    "z_from_effect",
    # prereg / multiplicity
    "MultiplicityLedger",
    "Preregistration",
    "check_preregistration",
    "one_sided_p_from_z",
    "plan_k_max",
    "prereg_fingerprint",
    "read_preregistration",
    "sha256_text",
    "two_sided_p_from_z",
    # goodhart
    "VERDICT_UTILITY",
    "GoodhartMonitor",
    "ProxyTargetMonitor",
    "pearson",
    "spearman",
]
