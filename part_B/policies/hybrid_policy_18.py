"""
Hybrid Policy — Group 18
Task 5

Strategy
--------
  t = 0  (first hour) : dummy action — no heating, no ventilation.
  t = 1 … 8           : ADP one-step lookahead MILP (Task 4 policy).
  t = 9  (last hour)  : dummy action — no heating, no ventilation.
"""

import os
import sys
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────────────
_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
_PART_B_DIR = os.path.dirname(_POLICY_DIR)
_BASE_DIR   = os.path.dirname(_PART_B_DIR)

for _p in [_BASE_DIR, _PART_B_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────────────────────────────────────
# Import internals from ADP policy
# ─────────────────────────────────────────────────────────────────────────────
from part_B.policies.ADP_policy_9_features_18 import (
    ETA,
    _get_data,
    _sample_scenarios,
    _solve_milp,
)

DUMMY_ACTION = {"HeatPowerRoom1": 0.0, "HeatPowerRoom2": 0.0, "VentilationON": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Public policy function
# ─────────────────────────────────────────────────────────────────────────────

def select_action(state):
    t = state["current_time"]

    if t == 0 or t == 9:
        return DUMMY_ACTION

    try:
        data     = _get_data()
        n_eta    = len(next(iter(ETA.values())))
        eta_next = ETA.get(t + 1, np.zeros(n_eta))
        prices, occs, price_mean, occ1_mean, occ2_mean = _sample_scenarios(state)
        return _solve_milp(state, data, eta_next,
                           prices, occs, price_mean, occ1_mean, occ2_mean)
    except Exception as e:
        print(f"  [HYBRID ERROR t={t}: {e}]")
        return DUMMY_ACTION