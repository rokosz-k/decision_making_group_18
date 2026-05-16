"""
ADP_policy_18.py

Approximate Dynamic Programming policy with linear value function approximation.

At each hour t, selects action by solving a one-step lookahead MILP:

    min  price_mean * (p1 + p2 + P_vent * v)
       + eta_{t+1} @ features(next_state)

The MILP structure mirrors RestaurantEnv.step_env exactly:
  1. Pre-resolve action overrides in Python (same if-else as environment)
  2. Compute dynamics as linear Pyomo expressions
  3. Detect next-step threshold crossings with binary indicators (OIH naming)
  4. Determine next override state

Exogenous uncertainty (price, occupancy) is handled by sampling K scenarios
from the process models and using their means before building the MILP.

Weights (eta.pkl) are loaded from part_B/ADP_training/ at import time.
"""

import os
import sys
import pickle
import numpy as np

from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint,
    NonNegativeReals, Binary, value, SolverFactory, minimize
)

# ─────────────────────────────────────────────
# Path setup
# File lives at: <BASE>/part_B/policies/ADP_policy_18.py
# ─────────────────────────────────────────────
_POLICY_DIR  = os.path.dirname(os.path.abspath(__file__))
_PART_B_DIR  = os.path.dirname(_POLICY_DIR)
_BASE_DIR    = os.path.dirname(_PART_B_DIR)

for _p in [_BASE_DIR, _PART_B_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────
# Load trained weights
# ─────────────────────────────────────────────
_ETA_PATH = os.path.join(_PART_B_DIR, "ADP_training", "eta_dummy.pkl")
with open(_ETA_PATH, "rb") as _f:
    ETA = pickle.load(_f)

# ─────────────────────────────────────────────
# Project imports
# ─────────────────────────────────────────────
from part_A.SystemCharacteristics import get_fixed_data
from part_B.PriceProcessRestaurant import price_model
from part_B.OccupancyProcessRestaurant import next_occupancy_levels

# ─────────────────────────────────────────────
# System data cache
# ─────────────────────────────────────────────
_DATA_CACHE = None

def _get_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = get_fixed_data()
    return _DATA_CACHE


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
K     = 20    # scenarios for exogenous sampling
M_BIG = 1e6   # big-M (matches OIH formulation)


# ─────────────────────────────────────────────
# Scenario sampling
# ─────────────────────────────────────────────

def _sample_scenarios(state):
    """
    Sample K realisations of next-step exogenous variables.

    price_model(price_t, price_previous) samples the CURRENT step price
    (state carries previous step price, so one call steps us forward).

    Returns:
        prices    : list of K sampled prices
        occs      : list of K (occ1, occ2) tuples
        price_mean: mean price (for linear VF terms that collapse)
        occ1_mean : mean occ1  (for linear VF terms that collapse)
        occ2_mean : mean occ2  (for linear VF terms that collapse)

    Linear terms in the objective collapse to means (expectation passes
    through linear functions). Binary override terms do NOT collapse and
    need the full K-scenario treatment in the MILP.
    """
    prices = [price_model(state["price_t"], state["price_previous"])
              for _ in range(K)]
    occs   = [next_occupancy_levels(state["Occ1"], state["Occ2"])
              for _ in range(K)]

    price_mean = float(np.mean(prices))
    occ1_mean  = float(np.mean([o[0] for o in occs]))
    occ2_mean  = float(np.mean([o[1] for o in occs]))

    return prices, occs, price_mean, occ1_mean, occ2_mean


# ─────────────────────────────────────────────
# MILP one-step lookahead
# ─────────────────────────────────────────────

def _solve_milp(state, data, eta_next, prices, occs, price_mean, occ1_mean, occ2_mean):
    """
    One-step lookahead MILP structured to mirror RestaurantEnv.step_env.

    Logic order matches the environment exactly:
      Step 1 — Pre-resolve action overrides (Python if-else, no Pyomo needed)
      Step 2 — Dynamics (linear Pyomo expressions)
      Step 3 — Low-temperature threshold indicators (OIH: y_low, y_ok)
      Step 4 — Next override state (OIH: u variable equivalent)
      Step 5 — Objective: immediate cost + value function
    """

    # ── Unpack state ──────────────────────────────────────────────────────
    t   = state["current_time"]
    T1  = state["T1"]
    T2  = state["T2"]
    H   = state["H"]
    lr1 = state["low_override_r1"]
    lr2 = state["low_override_r2"]
    vc  = state["vent_counter"]

    # ── System parameters (mirrors RestaurantEnv variable names) ──────────
    P_max    = data["heating_max_power"]
    T_out    = data["outdoor_temperature"][t]
    T_low    = data["temp_min_comfort_threshold"]
    T_ok     = data["temp_OK_threshold"]
    T_high   = data["temp_max_comfort_threshold"]
    H_high   = data["humidity_threshold"]
    P_vent   = data["ventilation_power"]
    vent_min = data["vent_min_up_time"]

    he      = data["heat_exchange_coeff"]
    tl      = data["thermal_loss_coeff"]
    ef      = data["heating_efficiency_coeff"]
    hv      = data["heat_vent_coeff"]
    ho      = data["heat_occupancy_coeff"]
    hu_occ  = data["humidity_occupancy_coeff"]
    hu_vent = data["humidity_vent_coeff"]

    # ── Pyomo model ───────────────────────────────────────────────────────
    m = ConcreteModel()

    # ═════════════════════════════════════════════════════════════════════
    # STEP 1 — Pre-resolve action overrides
    # Mirrors RestaurantEnv lines:
    #   if low_r1 == 1: P1 = heating_max_power
    #   if T1 > T_high: P1 = 0
    #   if H > humidity_threshold: vent_on = 1
    #   if 0 < vent_counter <= vent_min_up_time: vent_on = 1
    #
    # Overrides are known constants → resolved in Python before MILP.
    # Free variables are created only when action is not forced.
    # ═════════════════════════════════════════════════════════════════════

    # Heating power — Room 1
    if lr1 == 1:
        p1 = P_max                              # low override: forced to max
    elif T1 > T_high:
        p1 = 0.0                               # high cutoff: forced to zero
    else:
        m.p1 = Var(within=NonNegativeReals, bounds=(0, P_max))
        p1   = m.p1                            # free decision variable

    # Heating power — Room 2
    if lr2 == 1:
        p2 = P_max
    elif T2 > T_high:
        p2 = 0.0
    else:
        m.p2 = Var(within=NonNegativeReals, bounds=(0, P_max))
        p2   = m.p2

    # Ventilation
    if H > H_high or (0 < vc <= vent_min):
        v = 1                                  # forced on
    else:
        m.v = Var(within=Binary)
        v   = m.v                              # free decision variable

    # Ventilation counter at next step
    # New env: vc_next = vc + 1 if v=1, else vc (no change)
    # Compactly: vc_next = vc + v  (works for both constant and Pyomo v)
    vc_next = vc + v

    # ═════════════════════════════════════════════════════════════════════
    # STEP 2 — Dynamics
    # Direct copy of RestaurantEnv temperature and humidity equations.
    # p1, p2, v are either Python constants or Pyomo expressions —
    # Pyomo handles both transparently.
    # ═════════════════════════════════════════════════════════════════════

    T1_next = (T1
               + he * (T2 - T1)
               - tl * (T1 - T_out)
               + ef * p1
               - hv * v
               + ho * occ1_mean)

    T2_next = (T2
               + he * (T1 - T2)
               - tl * (T2 - T_out)
               + ef * p2
               - hv * v
               + ho * occ2_mean)

    H_next = H + hu_occ * (occ1_mean + occ2_mean) - hu_vent * v

    # ═════════════════════════════════════════════════════════════════════
    # STEP 3 — Per-scenario temperature expressions
    #
    # T1_next and T2_next differ across scenarios because occupancy differs.
    # We precompute the scenario-specific constant part (alpha) so that:
    #
    #   T1_next_k = alpha1[k] + ef*p1 - hv*v
    #
    # alpha1[k] is a Python float (known before MILP).
    # ef*p1 - hv*v is a shared Pyomo expression (same decision for all k).
    #
    # Linear VF terms (T1, T2, H, price, occ) collapse to means because
    # E[linear(x)] = linear(E[x]). Only the BINARY override terms do not
    # collapse — they need per-scenario variables.
    # ═════════════════════════════════════════════════════════════════════

    # Per-scenario constant offsets (floats, precomputed before Pyomo)
    alpha1 = [T1 + he*(T2-T1) - tl*(T1-T_out) + ho*occs[k][0] for k in range(K)]
    alpha2 = [T2 + he*(T1-T2) - tl*(T2-T_out) + ho*occs[k][1] for k in range(K)]

    # ═════════════════════════════════════════════════════════════════════
    # STEP 4 — Per-scenario binary override variables
    #
    # y_low[k] = 1 iff T_next_k <= T_low  (override fires in scenario k)
    # y_ok[k]  = 1 iff T_next_k >= T_ok   (override lifts in scenario k)
    # lr_next[k] = override state in scenario k
    #
    # The K binary variables replace the single binary from before.
    # The objective then AVERAGES over scenarios, which is the correct
    # expected value computation (not the mean-then-evaluate shortcut).
    # ═════════════════════════════════════════════════════════════════════

    m.y1_low   = Var(range(K), within=Binary)
    m.y2_low   = Var(range(K), within=Binary)
    m.lr1_next = Var(range(K), within=Binary)
    m.lr2_next = Var(range(K), within=Binary)

    # Room 1 — low threshold per scenario (mirrors OIH low_ub_rule, low_lb_rule)
    m.c_y1l_ub = Constraint(range(K), rule=lambda m, k:
        alpha1[k] + ef*p1 - hv*v <= T_low + M_BIG * (1 - m.y1_low[k]))
    m.c_y1l_lb = Constraint(range(K), rule=lambda m, k:
        alpha1[k] + ef*p1 - hv*v >= T_low - M_BIG * m.y1_low[k])

    # Room 2 — low threshold per scenario
    m.c_y2l_ub = Constraint(range(K), rule=lambda m, k:
        alpha2[k] + ef*p2 - hv*v <= T_low + M_BIG * (1 - m.y2_low[k]))
    m.c_y2l_lb = Constraint(range(K), rule=lambda m, k:
        alpha2[k] + ef*p2 - hv*v >= T_low - M_BIG * m.y2_low[k])

    # Next override state per scenario (mirrors OIH u variable logic)
    if lr1 == 0:
        # Override fires if T1_next_k < T_low in scenario k
        m.c_lr1 = Constraint(range(K), rule=lambda m, k:
            m.lr1_next[k] == m.y1_low[k])
    else:
        # Override lifts if T1_next_k >= T_ok in scenario k
        m.y1_ok     = Var(range(K), within=Binary)
        m.c_y1ok_lb = Constraint(range(K), rule=lambda m, k:
            alpha1[k] + ef*p1 - hv*v >= T_ok - M_BIG * (1 - m.y1_ok[k]))
        m.c_y1ok_ub = Constraint(range(K), rule=lambda m, k:
            alpha1[k] + ef*p1 - hv*v <= T_ok + M_BIG * m.y1_ok[k])
        m.c_lr1     = Constraint(range(K), rule=lambda m, k:
            m.lr1_next[k] == 1 - m.y1_ok[k])

    if lr2 == 0:
        m.c_lr2 = Constraint(range(K), rule=lambda m, k:
            m.lr2_next[k] == m.y2_low[k])
    else:
        m.y2_ok     = Var(range(K), within=Binary)
        m.c_y2ok_lb = Constraint(range(K), rule=lambda m, k:
            alpha2[k] + ef*p2 - hv*v >= T_ok - M_BIG * (1 - m.y2_ok[k]))
        m.c_y2ok_ub = Constraint(range(K), rule=lambda m, k:
            alpha2[k] + ef*p2 - hv*v <= T_ok + M_BIG * m.y2_ok[k])
        m.c_lr2     = Constraint(range(K), rule=lambda m, k:
            m.lr2_next[k] == 1 - m.y2_ok[k])

    # ═════════════════════════════════════════════════════════════════════
    # STEP 5 — Objective
    #
    # Immediate cost: uses price_mean (collapses correctly since p1,p2,v shared)
    # VF linear terms: use means (collapse correctly for linear functions)
    # VF binary terms: average over K scenarios (do NOT collapse — this is
    #   the key correction over the single-scenario formulation)
    #
    # Value function feature order matches FEATURE_COLS in train_adp.py:
    #   [T1, T2, H, Occ1, Occ2, price_t, price_previous,
    #    vent_counter, low_override_r1, low_override_r2, bias]
    # ═════════════════════════════════════════════════════════════════════

    # Immediate cost (price_mean collapses correctly since action is shared)
    immediate = price_mean * (p1 + p2 + P_vent * v)

    # VF — constant terms (pure floats)
    vf_const = (  eta_next[3]  * occ1_mean
                + eta_next[4]  * occ2_mean
                + eta_next[5]  * price_mean        # price_t at next step
                + eta_next[6]  * state["price_t"]  # price_previous at next step
                + eta_next[10]                     # bias
               )

    # VF — linear Pyomo terms (use mean occ since linear terms collapse)
    vf_linear = (  eta_next[0]  * T1_next
                 + eta_next[1]  * T2_next
                 + eta_next[2]  * H_next
                 + eta_next[7]  * vc_next
                )

    # VF — binary terms averaged over K scenarios (correct expected value)
    vf_binary = (1.0 / K) * sum(
        eta_next[8] * m.lr1_next[k] + eta_next[9] * m.lr2_next[k]
        for k in range(K)
    )

    m.obj = Objective(expr=immediate + vf_const + vf_linear + vf_binary,
                      sense=minimize)

    # ── Solve ─────────────────────────────────────────────────────────────
    solver = SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0
    solver.options["TimeLimit"]  = 12
    solver.solve(m)

    # Extract results — handle pre-resolved constants vs Pyomo variables
    if lr1 == 1:
        p1_val = P_max
    elif T1 > T_high:
        p1_val = 0.0
    else:
        p1_val = float(max(0.0, min(P_max, value(m.p1))))

    if lr2 == 1:
        p2_val = P_max
    elif T2 > T_high:
        p2_val = 0.0
    else:
        p2_val = float(max(0.0, min(P_max, value(m.p2))))

    if H > H_high or (0 < vc <= vent_min):
        v_val = 1
    else:
        v_val = int(round(value(m.v)))

    return {
        "HeatPowerRoom1": p1_val,
        "HeatPowerRoom2": p2_val,
        "VentilationON":  v_val,
    }


# ─────────────────────────────────────────────
# Public policy function
# ─────────────────────────────────────────────

def select_action(state):
    """
    ADP one-step lookahead policy.

    At every hour t (including the last), solves the one-step MILP:
        min  immediate_cost  +  eta_{t+1} @ features(next_state)

    At t=9 there is no ETA[10], so a zero vector is used — the VF term
    vanishes and the MILP minimises immediate cost only.
    This keeps the policy pure ADP with no hand-crafted exceptions.

    Args:
        state : dict from RestaurantEnv.reset_env / step_env
    Returns:
        HereAndNowActions: dict with HeatPowerRoom1, HeatPowerRoom2, VentilationON
    """
    t    = state["current_time"]
    data = _get_data()

    prices, occs, price_mean, occ1_mean, occ2_mean = _sample_scenarios(state)

    # ETA[t+1] for t=0..8; zero vector at t=9 (no future beyond last step)
    eta_next = ETA.get(t + 1, np.zeros(11))

    result = _solve_milp(state, data, eta_next,
                         prices, occs, price_mean, occ1_mean, occ2_mean)

    HereAndNowActions = {
        "HeatPowerRoom1": result["HeatPowerRoom1"],
        "HeatPowerRoom2": result["HeatPowerRoom2"],
        "VentilationON":  result["VentilationON"],
    }
    return HereAndNowActions