# sphinx/tasks/sequence/config.py
from typing import Dict

SEQ_CONFIG: Dict = {
    # Global sequence knobs
    "min_delta": 0.025,            # visible-change threshold (per step)

    # Limits / ranges used by variants
    "limits": {
        "count_lo_hi": (3, 15),         # for count_arith
        "rowcol_max_1d": 10,            # dot rows/cols max in 1D variants
        "grid_caps": {                  # for grid(r,c) synthetic
            "max_rows": 6, "max_cols": 6, "max_dots": 36
        },
    },

    # Candidate steps for rotation progression
    "rotation_steps": [20, 30, 36, 40, 45, 60, 72, 90],

    # Motif allow/deny (exact or glob patterns). None => allow all.
    "allowed_motifs": {
        "count_arith": None,
        "rotation_arith": None,
    },
    "denied_motifs": {
        "count_arith": [],
        "rotation_arith": [],
    },

    # Toggle variants globally
    "enabled_variants": ["synthetic", "count_arith", "rotation_arith"],
}
