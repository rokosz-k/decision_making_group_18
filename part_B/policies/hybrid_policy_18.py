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
import pickle
import numpy as np

from pyomo.environ import (
    ConcreteModel, Var, Objective, Constraint,
    NonNegativeReals, Binary, value, SolverFactory, minimize,
)

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
# Load trained ADP weights
# ─────────────────────────────────────────────────────────────────────────────
_ETA_PATH = os.path.join(_PART_B_DIR, "ADP_training", "eta_sp_fvi_v2.pkl")
with open(_ETA_PATH, "rb") as _f:
    ETA = pickle.load(_f)

# ─────────────────────────────────────────────────────────────────────────────
# Project imports
# ─────────────────────────────────────────────────────────────────────────────
from data.v2_SystemCharacteristics     import get_fixed_data
from part_B.PriceProcessRestaurant     import price_model
from part_B.OccupancyProcessRestaurant import next_occupancy_levels

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
K     = 20
M_BIG = 1e6

_DATA_CACHE = None

DUMMY_ACTION = {"HeatPowerRoom1": 0.0, "HeatPowerRoom2": 0.0, "VentilationON": 0}


def _get_data():
    global _DATA_CACHE
    if _DATA_CACHE is None:
        _DATA_CACHE = get_fixed_data()
    return _DATA_CACHE


# ─────────────────────────────────────────────────────────────────────────────
# Scenario sampling — identical to ADP_policy_new_features_18
# ─────────────────────────────────────────────────────────────────────────────

def _sample_scenarios(state):
    prices = [price_model(state["price_t"], state["price_previous"])
              for _ in range(K)]
    occs   = [next_occupancy_levels(state["Occ1"], state["Occ2"])
              for _ in range(K)]

    price_mean = float(np.mean(prices))
    occ1_mean  = float(np.mean([o[0] for o in occs]))
    occ2_mean  = float(np.mean([o[1] for o in occs]))

    return prices, occs, price_mean, occ1_mean, occ2_mean


# ─────────────────────────────────────────────────────────────────────────────
# ADP MILP — identical to ADP_policy_new_features_18._solve_milp
# ─────────────────────────────────────────────────────────────────────────────

def _solve_milp(state, data, eta_next, prices, occs, price_mean, occ1_mean, occ2_mean):
    t   = state["current_time"]
    T1  = state["T1"]
    T2  = state["T2"]
    H   = state["H"]
    lr1 = state["low_override_r1"]
    lr2 = state["low_override_r2"]
    vc  = state["vent_counter"]

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

    m = ConcreteModel()

    # Step 1 — pre-resolve overrides
    if lr1 == 1:
        p1 = P_max
    elif T1 > T_high:
        p1 = 0.0
    else:
        m.p1 = Var(within=NonNegativeReals, bounds=(0, P_max))
        p1   = m.p1

    if lr2 == 1:
        p2 = P_max
    elif T2 > T_high:
        p2 = 0.0
    else:
        m.p2 = Var(within=NonNegativeReals, bounds=(0, P_max))
        p2   = m.p2

    if H > H_high or (0 < vc <= vent_min):
        v = 1
    else:
        m.v = Var(within=Binary)
        v   = m.v

    vc_next = vc + v

    # Step 2 — dynamics
    T1_next = (T1 + he*(T2-T1) - tl*(T1-T_out) + ef*p1 - hv*v + ho*occ1_mean)
    T2_next = (T2 + he*(T1-T2) - tl*(T2-T_out) + ef*p2 - hv*v + ho*occ2_mean)
    H_next  = H + hu_occ*(occ1_mean + occ2_mean) - hu_vent*v

    # Step 3 — per-scenario base temperatures
    alpha1 = [T1 + he*(T2-T1) - tl*(T1-T_out) + ho*occs[k][0] for k in range(K)]
    alpha2 = [T2 + he*(T1-T2) - tl*(T2-T_out) + ho*occs[k][1] for k in range(K)]

    # Step 4 — per-scenario binary override detection
    m.y1_low   = Var(range(K), within=Binary)
    m.y2_low   = Var(range(K), within=Binary)
    m.lr1_next = Var(range(K), within=Binary)
    m.lr2_next = Var(range(K), within=Binary)

    m.c_y1l_ub = Constraint(range(K), rule=lambda m, k:
        alpha1[k] + ef*p1 - hv*v <= T_low + M_BIG*(1 - m.y1_low[k]))
    m.c_y1l_lb = Constraint(range(K), rule=lambda m, k:
        alpha1[k] + ef*p1 - hv*v >= T_low - M_BIG*m.y1_low[k])

    m.c_y2l_ub = Constraint(range(K), rule=lambda m, k:
        alpha2[k] + ef*p2 - hv*v <= T_low + M_BIG*(1 - m.y2_low[k]))
    m.c_y2l_lb = Constraint(range(K), rule=lambda m, k:
        alpha2[k] + ef*p2 - hv*v >= T_low - M_BIG*m.y2_low[k])

    if lr1 == 0:
        m.c_lr1 = Constraint(range(K), rule=lambda m, k:
            m.lr1_next[k] == m.y1_low[k])
    else:
        m.y1_ok     = Var(range(K), within=Binary)
        m.c_y1ok_lb = Constraint(range(K), rule=lambda m, k:
            alpha1[k] + ef*p1 - hv*v >= T_ok - M_BIG*(1 - m.y1_ok[k]))
        m.c_y1ok_ub = Constraint(range(K), rule=lambda m, k:
            alpha1[k] + ef*p1 - hv*v <= T_ok + M_BIG*m.y1_ok[k])
        m.c_lr1 = Constraint(range(K), rule=lambda m, k:
            m.lr1_next[k] == 1 - m.y1_ok[k])

    if lr2 == 0:
        m.c_lr2 = Constraint(range(K), rule=lambda m, k:
            m.lr2_next[k] == m.y2_low[k])
    else:
        m.y2_ok     = Var(range(K), within=Binary)
        m.c_y2ok_lb = Constraint(range(K), rule=lambda m, k:
            alpha2[k] + ef*p2 - hv*v >= T_ok - M_BIG*(1 - m.y2_ok[k]))
        m.c_y2ok_ub = Constraint(range(K), rule=lambda m, k:
            alpha2[k] + ef*p2 - hv*v <= T_ok + M_BIG*m.y2_ok[k])
        m.c_lr2 = Constraint(range(K), rule=lambda m, k:
            m.lr2_next[k] == 1 - m.y2_ok[k])

    # Step 5 — objective
    immediate = price_mean * (p1 + p2 + P_vent * v)

    vf_const  = (eta_next[3] * price_mean
               + eta_next[4] * state["price_t"]
               + eta_next[8])

    vf_linear = (eta_next[0] * T1_next
               + eta_next[1] * T2_next
               + eta_next[2] * H_next
               + eta_next[5] * vc_next)

    vf_binary = (1.0 / K) * sum(
        eta_next[6] * m.lr1_next[k] + eta_next[7] * m.lr2_next[k]
        for k in range(K)
    )

    m.obj = Objective(expr=immediate + vf_const + vf_linear + vf_binary,
                      sense=minimize)

    solver = SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0
    solver.options["TimeLimit"]  = 12
    solver.solve(m)

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

    return {"HeatPowerRoom1": p1_val, "HeatPowerRoom2": p2_val, "VentilationON": v_val}


# ─────────────────────────────────────────────────────────────────────────────
# Public interface
# ─────────────────────────────────────────────────────────────────────────────

def select_action(state):
    """
    Hybrid policy:
      t = 0 or t = 9  →  dummy (no heating, no ventilation)
      t = 1 … 8       →  ADP one-step lookahead MILP
    """
    t = state["current_time"]

    # First and last hour: do nothing, leave everything to overrule controllers
    if t == 0 or t == 9:
        return DUMMY_ACTION

    # All other hours: ADP
    try:
        data = _get_data()
        n_eta    = len(next(iter(ETA.values())))
        eta_next = ETA.get(t + 1, np.zeros(n_eta))
        prices, occs, price_mean, occ1_mean, occ2_mean = _sample_scenarios(state)
        return _solve_milp(state, data, eta_next,
                           prices, occs, price_mean, occ1_mean, occ2_mean)
    except Exception as e:
        print(f"  [HYBRID ERROR t={t}: {e}]")
        return DUMMY_ACTION