"""
ADP_policy_18.py

Approximate Dynamic Programming policy with linear value function approximation.

At each hour t, selects action by solving a one-step lookahead MILP:

    min  price_mean * (p1_eff + p2_eff + P_vent * v)
       + eta_{t+1} @ features(next_state)

    s.t. temperature/humidity dynamics (linear in p1, p2, v)
         overrule controller logic (modelled with big-M binary constraints)
         ventilation inertia (hard constraint on v)

Exogenous uncertainty (next price, next occupancy) is handled by sampling
K scenarios from the process models BEFORE the MILP and using their means.

Trained weights (eta.pkl) are loaded from the repo root at import time.
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
_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
_PART_B_DIR = os.path.dirname(_POLICY_DIR)
_BASE_DIR   = os.path.dirname(_PART_B_DIR)   # repo root

for _p in [_BASE_DIR, _PART_B_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────
# Load trained weights (eta.pkl from repo root)
# ─────────────────────────────────────────────
_ETA_PATH = os.path.join(_BASE_DIR, "eta.pkl")
with open(_ETA_PATH, "rb") as _f:
    ETA = pickle.load(_f)

# ─────────────────────────────────────────────
# Project imports
# ─────────────────────────────────────────────
from part_A.SystemCharacteristics import get_fixed_data
from part_B.PriceProcessRestaurant import price_model
from part_B.OccupancyProcessRestaurant import next_occupancy_levels

# ─────────────────────────────────────────────
# System data (loaded once, cached)
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
K     = 20      # number of scenarios for exogenous sampling
M_BIG = 100.0   # big-M for binary indicator constraints
EPS   = 0.01    # small epsilon to approximate strict inequalities


# ─────────────────────────────────────────────
# Scenario sampling
# ─────────────────────────────────────────────

def _sample_scenarios(state):
    """
    Sample K realisations of next-step exogenous variables from process models.

    price_model(price_t, price_previous) gives samples of the CURRENT step's
    price (the state carries the previous step's price, so one call of
    price_model steps us forward to the price being used right now).

    Returns scalar means used as proxies inside the MILP.
    """
    prices = [price_model(state["price_t"], state["price_previous"])
              for _ in range(K)]
    occs   = [next_occupancy_levels(state["Occ1"], state["Occ2"])
              for _ in range(K)]

    price_mean = float(np.mean(prices))
    occ1_mean  = float(np.mean([o[0] for o in occs]))
    occ2_mean  = float(np.mean([o[1] for o in occs]))

    return price_mean, occ1_mean, occ2_mean


# ─────────────────────────────────────────────
# MILP one-step lookahead
# ─────────────────────────────────────────────

def _solve_milp(state, data, eta_next, price_mean, occ1_mean, occ2_mean):
    """
    Build and solve the one-step lookahead MILP.

    Decision variables  : p1, p2 continuous in [0, P_max]; v binary
    Auxiliary binaries  : b_low, b_high, b_ok, lr_next  (one set per room)
    Auxiliary continuous: p_eff, w  (linearisation of p * binary)

    Objective is linear in all variables — standard MILP solvable by Gurobi.
    """

    # ── Unpack state ────────────────────────────────────────────────────
    t   = state["current_time"]
    T1  = state["T1"]
    T2  = state["T2"]
    H   = state["H"]
    lr1 = state["low_override_r1"]
    lr2 = state["low_override_r2"]
    vc  = state["vent_counter"]

    # ── System parameters ────────────────────────────────────────────────
    P_max    = data["heating_max_power"]
    T_out    = data["outdoor_temperature"][t]
    T_low    = data["temp_min_comfort_threshold"]
    T_ok     = data["temp_OK_threshold"]
    T_high   = data["temp_max_comfort_threshold"]
    P_vent   = data["ventilation_power"]
    vent_min = data["vent_min_up_time"]

    he      = data["heat_exchange_coeff"]
    tl      = data["thermal_loss_coeff"]
    ef      = data["heating_efficiency_coeff"]
    hv      = data["heat_vent_coeff"]
    ho      = data["heat_occupancy_coeff"]
    hu_occ  = data["humidity_occupancy_coeff"]
    hu_vent = data["humidity_vent_coeff"]

    # ── Pyomo model ──────────────────────────────────────────────────────
    m = ConcreteModel()

    # Decision variables
    m.p1 = Var(within=NonNegativeReals, bounds=(0, P_max))
    m.p2 = Var(within=NonNegativeReals, bounds=(0, P_max))
    m.v  = Var(within=Binary)

    # ── Ventilation inertia ──────────────────────────────────────────────
    if vc > 0:
        # Ventilation must stay ON; counter decrements by 1
        m.c_v_forced = Constraint(expr=m.v == 1)
        vc_next = vc - 1                      # constant integer
    else:
        # Fresh start: if v=1, counter becomes vent_min-1 next step; else 0
        vc_next = (vent_min - 1) * m.v        # linear Pyomo expression

    # ── Dynamics (all linear in p1, p2, v) ──────────────────────────────
    T1_next = (T1
               + he * (T2 - T1)
               - tl * (T1 - T_out)
               + ef * m.p1
               - hv * m.v
               + ho * occ1_mean)

    T2_next = (T2
               + he * (T1 - T2)
               - tl * (T2 - T_out)
               + ef * m.p2
               - hv * m.v
               + ho * occ2_mean)

    H_next = H + hu_occ * (occ1_mean + occ2_mean) - hu_vent * m.v

    # ── Binary indicators for temperature thresholds ─────────────────────
    # b_low  = 1  iff  T_next < T_low   (new low override fires)
    # b_high = 1  iff  T_next > T_high  (high cutoff fires)
    # Big-M formulation:
    #   b=1 iff x <= c:   x >= c - M*b        [b=0 forces x >= c]
    #                     x <= c - eps + M*(1-b) [b=1 forces x <= c-eps]

    m.b1_low  = Var(within=Binary)
    m.b2_low  = Var(within=Binary)
    m.b1_high = Var(within=Binary)
    m.b2_high = Var(within=Binary)

    # Room 1 — low threshold  (b=1 iff T1_next < T_low)
    m.c_b1l_1 = Constraint(expr=T1_next >= T_low - M_BIG * m.b1_low)
    m.c_b1l_2 = Constraint(expr=T1_next <= T_low - EPS + M_BIG * (1 - m.b1_low))
    # Room 2 — low threshold
    m.c_b2l_1 = Constraint(expr=T2_next >= T_low - M_BIG * m.b2_low)
    m.c_b2l_2 = Constraint(expr=T2_next <= T_low - EPS + M_BIG * (1 - m.b2_low))

    # Room 1 — high cutoff  (b=1 iff T1_next > T_high)
    m.c_b1h_1 = Constraint(expr=T1_next <= T_high + M_BIG * m.b1_high)
    m.c_b1h_2 = Constraint(expr=T1_next >= T_high + EPS - M_BIG * (1 - m.b1_high))
    # Room 2 — high cutoff
    m.c_b2h_1 = Constraint(expr=T2_next <= T_high + M_BIG * m.b2_high)
    m.c_b2h_2 = Constraint(expr=T2_next >= T_high + EPS - M_BIG * (1 - m.b2_high))

    # A room cannot be simultaneously too cold and too hot
    m.c_excl1 = Constraint(expr=m.b1_low + m.b1_high <= 1)
    m.c_excl2 = Constraint(expr=m.b2_low + m.b2_high <= 1)

    # ── Next override state ──────────────────────────────────────────────
    # lr_next = 1  if override was already active and T_next < T_ok
    #         = 1  if override was inactive and T_next < T_low  (new trigger)
    #         = 0  otherwise

    m.lr1_next = Var(within=Binary)
    m.lr2_next = Var(within=Binary)

    if lr1 == 0:
        # Enter override only if T1_next < T_low
        m.c_lr1 = Constraint(expr=m.lr1_next == m.b1_low)
    else:
        # Already in override; exit only when T1_next >= T_ok
        # b1_ok = 1 iff T1_next >= T_ok  → lr1_next = 1 - b1_ok
        m.b1_ok    = Var(within=Binary)
        m.c_b1ok_1 = Constraint(expr=T1_next >= T_ok - M_BIG * (1 - m.b1_ok))
        m.c_b1ok_2 = Constraint(expr=T1_next <= T_ok - EPS + M_BIG * m.b1_ok)
        m.c_lr1    = Constraint(expr=m.lr1_next == 1 - m.b1_ok)

    if lr2 == 0:
        m.c_lr2 = Constraint(expr=m.lr2_next == m.b2_low)
    else:
        m.b2_ok    = Var(within=Binary)
        m.c_b2ok_1 = Constraint(expr=T2_next >= T_ok - M_BIG * (1 - m.b2_ok))
        m.c_b2ok_2 = Constraint(expr=T2_next <= T_ok - EPS + M_BIG * m.b2_ok)
        m.c_lr2    = Constraint(expr=m.lr2_next == 1 - m.b2_ok)

    # Override and high cutoff are mutually exclusive (physically)
    m.c_excl_lr1 = Constraint(expr=m.lr1_next + m.b1_high <= 1)
    m.c_excl_lr2 = Constraint(expr=m.lr2_next + m.b2_high <= 1)

    # ── Effective powers ─────────────────────────────────────────────────
    # p1_eff = P_max  if lr1_next = 1   (low override → max heating)
    #        = 0      if b1_high  = 1   (high cutoff  → no heating)
    #        = p1     otherwise
    #
    # Written as: p1_eff = P_max * lr1_next + p1 * s1
    #   where s1 = 1 - lr1_next - b1_high  (∈ {0,1}, mutually exclusive)
    #
    # Linearise  w1 = p1 * s1  via McCormick (s1 binary, p1 ∈ [0, P_max]):
    #   w1 <= P_max * s1
    #   w1 >= p1 - P_max*(1-s1)
    #   w1 <= p1
    #   w1 >= 0

    m.p1_eff = Var(within=NonNegativeReals, bounds=(0, P_max))
    m.w1     = Var(within=NonNegativeReals, bounds=(0, P_max))
    m.c_w1_1   = Constraint(expr=m.w1 <= P_max * (1 - m.lr1_next - m.b1_high))
    m.c_w1_2   = Constraint(expr=m.w1 >= m.p1 - P_max * (m.lr1_next + m.b1_high))
    m.c_w1_3   = Constraint(expr=m.w1 <= m.p1)
    m.c_p1_eff = Constraint(expr=m.p1_eff == P_max * m.lr1_next + m.w1)

    m.p2_eff = Var(within=NonNegativeReals, bounds=(0, P_max))
    m.w2     = Var(within=NonNegativeReals, bounds=(0, P_max))
    m.c_w2_1   = Constraint(expr=m.w2 <= P_max * (1 - m.lr2_next - m.b2_high))
    m.c_w2_2   = Constraint(expr=m.w2 >= m.p2 - P_max * (m.lr2_next + m.b2_high))
    m.c_w2_3   = Constraint(expr=m.w2 <= m.p2)
    m.c_p2_eff = Constraint(expr=m.p2_eff == P_max * m.lr2_next + m.w2)

    # ── Objective ────────────────────────────────────────────────────────
    # Immediate cost: sampled price mean proxies for the current step price
    immediate = price_mean * (m.p1_eff + m.p2_eff + P_vent * m.v)

    # Value function at next state.
    # Feature order matches FEATURE_COLS in train_adp.py:
    #   [T1, T2, H, Occ1, Occ2, price_t, price_previous, vent_counter,
    #    low_override_r1, low_override_r2, bias]
    #
    # Separate constant terms (plain floats) from Pyomo expression terms
    # to avoid type-mixing issues.
    vf_const = (  eta_next[3]  * occ1_mean
                + eta_next[4]  * occ2_mean
                + eta_next[5]  * price_mean        # price_t at t+1
                + eta_next[6]  * state["price_t"]  # price_previous at t+1
                + eta_next[10]                     # bias
               )

    vf_expr  = (  eta_next[0]  * T1_next
                + eta_next[1]  * T2_next
                + eta_next[2]  * H_next
                + eta_next[7]  * vc_next
                + eta_next[8]  * m.lr1_next
                + eta_next[9]  * m.lr2_next
               )

    m.obj = Objective(expr=immediate + vf_const + vf_expr, sense=minimize)

    # ── Solve ─────────────────────────────────────────────────────────────
    solver = SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0
    solver.options["TimeLimit"]  = 12
    solver.solve(m)

    p1_val = float(max(0.0, min(P_max, value(m.p1))))
    p2_val = float(max(0.0, min(P_max, value(m.p2))))
    v_val  = int(round(value(m.v)))

    return {
        "HeatPowerRoom1": p1_val,
        "HeatPowerRoom2": p2_val,
        "VentilationON":  v_val,
    }


# ─────────────────────────────────────────────
# Public policy function (imported by running_script)
# ─────────────────────────────────────────────

def select_action(state):
    """
    ADP one-step lookahead policy.

    Steps:
      1. Sample K scenarios from process models to get price/occupancy means
      2. Solve MILP with VF term eta_{t+1}  (or skip VF at last step t=9)
      3. Return optimal (p1, p2, v)

    Args:
        state : dict from RestaurantEnv.reset_env / step_env
    Returns:
        HereAndNowActions: dict with keys HeatPowerRoom1, HeatPowerRoom2, VentilationON
    """
    try:
        t    = state["current_time"]
        data = _get_data()

        price_mean, occ1_mean, occ2_mean = _sample_scenarios(state)

        if t >= 9:
            # Last step: no future value — minimise immediate cost.
            # Best action is to heat as little as possible; the overrule
            # controller handles any temperature constraint violations.
            v = 1 if state["vent_counter"] > 0 else 0
            HereAndNowActions = {
                "HeatPowerRoom1": 0.0,
                "HeatPowerRoom2": 0.0,
                "VentilationON":  v,
            }
            return HereAndNowActions

        eta_next = ETA[t + 1]
        result   = _solve_milp(state, data, eta_next,
                               price_mean, occ1_mean, occ2_mean)

        HereAndNowActions = {
            "HeatPowerRoom1": result["HeatPowerRoom1"],
            "HeatPowerRoom2": result["HeatPowerRoom2"],
            "VentilationON":  result["VentilationON"],
        }
        return HereAndNowActions

    except Exception as e:
        print(f"  [ADP ERROR: {e}]  — using fallback")
        v = 1 if state.get("vent_counter", 0) > 0 else 0
        HereAndNowActions = {
            "HeatPowerRoom1": 0.0,
            "HeatPowerRoom2": 0.0,
            "VentilationON":  v,
        }
        return HereAndNowActions
