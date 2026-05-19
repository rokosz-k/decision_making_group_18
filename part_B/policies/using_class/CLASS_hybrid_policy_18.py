"""
Hybrid Policy — Group 18
Task 5

Strategy
--------
  t = 0  (first hour) : dummy action — no heating, no ventilation.
                         Electricity prices are typically low at opening;
                         the overrule controller handles any cold start.
  t = 1 … 8           : ADP one-step lookahead MILP (Task 4 policy).
  t = 9  (last hour)  : dummy action — no heating, no ventilation.
                         No future value remains; minimising the last
                         hour's cost means doing nothing.

Everything between the first and last hour uses the full ADP policy
(eta_sp_fvi_v2.pkl weights, K=20 scenarios for binary override terms).
"""

import os
import sys

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
# Imports
# ─────────────────────────────────────────────────────────────────────────────
from part_B.policies.using_class.CLASS_ADP_policy_new_features_18 import ADP_Policy

DUMMY_ACTION = {"HeatPowerRoom1": 0.0, "HeatPowerRoom2": 0.0, "VentilationON": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Public policy class
# ─────────────────────────────────────────────────────────────────────────────

class Hybrid_Policy:
    """
    Hybrid policy:
      t = 0 or t = 9  →  dummy (no heating, no ventilation)
      t = 1 … 8       →  ADP one-step lookahead MILP

    Compatible with v2_Checks.check_and_sanitize_action:
        policy = HybridPolicy()
        action = check_and_sanitize_action(policy, state, PowerMax)
    """

    def __init__(self):
        self._adp = ADP_Policy()

    def select_action(self, state):
        t = state["current_time"]

        # First and last hour: do nothing, leave everything to overrule controllers
        if t == 0 or t == 9:
            return DUMMY_ACTION

        # All other hours: delegate to ADP
        try:
            return self._adp.select_action(state)
        except Exception as e:
            print(f"  [HYBRID ERROR t={t}: {e}]")
            return DUMMY_ACTION