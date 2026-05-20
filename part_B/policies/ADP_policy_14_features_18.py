"""
ADP_policy_18_v4.py
===================

Changes from ADP_policy_new_features_18.py (v3):
--------------------------------------------------

FIX 1 — vc_next corrected for count-UP environment with reset
  The environment counts UP while ventilation is on, resets to 0 on turn-off.
  Original vc+v is correct for vc <= vent_min, but wrong when vc > vent_min
  and v=0: gives vc instead of 0 (the reset value).

  New formula:
    if vc > vent_min:  vc_next = (vc+1)*v  (0 on turn-off reset, vc+1 if on)
    else:              vc_next = vc + v     (original — correct for vc <= vent_min)

FIX 2 — Five new VF features (N_FEAT 9 → 14, bias now at index 13)
  Three piecewise-linear features on the next state (single aux variable each):
    feat 8:  max(0, T_low + T_DANGER_MARGIN - T1_next)   ← T1 cold-danger proximity
    feat 9:  max(0, T_low + T_DANGER_MARGIN - T2_next)   ← T2 cold-danger proximity
    feat 10: max(0, H_next - H_APPROACH_THRESH)           ← humidity threshold approach

  Two per-scenario interaction features (linear in binary lr_next[k]):
    feat 11: (1/K) Σ_k  prices[k] * lr1_next[k]   ← override cost signal room 1
    feat 12: (1/K) Σ_k  prices[k] * lr2_next[k]   ← override cost signal room 2

  MILP validity:
    - Piecewise terms are monotone in p1/p2/v → action boundaries {0, P_max} still optimal.
    - z1_low/z2_low >= 0 and >= (T_low+margin - T1_next).  With η[8]>0 the minimiser
      naturally sets z = max(0, ...).  Bounds (0, margin+10) prevent blowup if η < 0.
    - price_k * lr1_next[k] is constant * binary → linear in Pyomo. ✓

Requires eta_sp_fvi_v4.pkl (14 weights per timestep, produced by train_adp_fvi_v4.py).
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
# ─────────────────────────────────────────────
_POLICY_DIR = os.path.dirname(os.path.abspath(__file__))
_PART_B_DIR = os.path.dirname(_POLICY_DIR)
_BASE_DIR   = os.path.dirname(_PART_B_DIR)

for _p in [_BASE_DIR, _PART_B_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ─────────────────────────────────────────────
# Load trained weights  (v4 — 14 features)
# ─────────────────────────────────────────────
_ETA_PATH = os.path.join(_PART_B_DIR, "ADP_training", "eta_sp_fvi_14_features.pkl")
with open(_ETA_PATH, "rb") as _f:
    ETA = pickle.load(_f)

# ─────────────────────────────────────────────
# Project imports
# ─────────────────────────────────────────────
from data.v2_SystemCharacteristics import get_fixed_data
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
# Danger-zone constants  (must match train_adp_fvi_v4.py)
# ─────────────────────────────────────────────
T_DANGER_MARGIN   = 2.0    # feature fires when T_next < T_low + 2 = 20 °C
H_APPROACH_THRESH = 60.0   # feature fires when H_next > 60 %

# ─────────────────────────────────────────────
# MILP constants
# ─────────────────────────────────────────────
K     = 20
M_BIG = 1e6


# ─────────────────────────────────────────────
# Scenario sampling
# ─────────────────────────────────────────────

def _sample_scenarios(state):
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
    One-step lookahead MILP.

    Feature indices in eta_next (14 elements):
      0:T1  1:T2  2:H  3:price_t  4:price_previous  5:vent_counter
      6:low_override_r1  7:low_override_r2
      8:T1_danger  9:T2_danger  10:H_approach
      11:price_x_ovr1  12:price_x_ovr2
      13:bias
    """

    # ── Unpack state ──────────────────────────────────────────────────────────
    t   = state["current_time"]
    T1  = state["T1"]
    T2  = state["T2"]
    H   = state["H"]
    lr1 = state["low_override_r1"]
    lr2 = state["low_override_r2"]
    vc  = state["vent_counter"]

    # ── System parameters ─────────────────────────────────────────────────────
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

    # ── Pyomo model ───────────────────────────────────────────────────────────
    m = ConcreteModel()

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1 — Pre-resolve action overrides (mirrors RestaurantEnv exactly)
    # ═════════════════════════════════════════════════════════════════════════

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

    if H > H_high or (0 < vc < vent_min):   # strict < matches fixed RestaurantEnv
        v = 1
    else:
        m.v = Var(within=Binary)
        v   = m.v

    # ─────────────────────────────────────────────────────────────────────────
    # vc_next — count-UP environment with reset (v2 RestaurantEnv)
    #
    # Environment logic:
    #   if vent_on == 1:          vc += 1
    #   elif 0 < vc <= vent_min:  vc += 1  (forced on)
    #   else:                     vc  = 0  (reset on successful turn-off)
    #
    # Result per case:
    #   vc=0,            v=0 → vc_next=0        [vc + v = 0 ✓]
    #   vc=0,            v=1 → vc_next=1        [vc + v = 1 ✓]
    #   0 < vc ≤ vent_min, v=1 (forced) → vc+1  [vc + v ✓]
    #   vc > vent_min,   v=1 → vc_next=vc+1     [(vc+1)*v ✓]
    #   vc > vent_min,   v=0 → vc_next=0 (RESET) [(vc+1)*v=0 ✓; vc+v=vc ✗]
    #
    # Fix: use (vc+1)*v when vc > vent_min (linear: vc is a known constant).
    # ─────────────────────────────────────────────────────────────────────────
    if vc >= vent_min:
        # vc == vent_min: 3 hours served, now free. vc > vent_min: kept on voluntarily.
        # Turning off resets to 0; staying on increments. Linear since vc is a constant.
        vc_next = (vc + 1) * v
    else:
        # vc=0 (free) or 0 < vc < vent_min (forced): original formula is correct.
        vc_next = vc + v

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2 — Dynamics  (mean occupancy → correct for linear terms)
    # ═════════════════════════════════════════════════════════════════════════

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

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3 — Per-scenario offsets (occupancy varies across scenarios)
    # ═════════════════════════════════════════════════════════════════════════

    alpha1 = [T1 + he*(T2-T1) - tl*(T1-T_out) + ho*occs[k][0] for k in range(K)]
    alpha2 = [T2 + he*(T1-T2) - tl*(T2-T_out) + ho*occs[k][1] for k in range(K)]

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 4 — Per-scenario override binary indicators
    # ═════════════════════════════════════════════════════════════════════════

    m.y1_low   = Var(range(K), within=Binary)
    m.y2_low   = Var(range(K), within=Binary)
    m.lr1_next = Var(range(K), within=Binary)
    m.lr2_next = Var(range(K), within=Binary)

    m.c_y1l_ub = Constraint(range(K), rule=lambda m, k:
        alpha1[k] + ef*p1 - hv*v <= T_low + M_BIG * (1 - m.y1_low[k]))
    m.c_y1l_lb = Constraint(range(K), rule=lambda m, k:
        alpha1[k] + ef*p1 - hv*v >= T_low - M_BIG * m.y1_low[k])

    m.c_y2l_ub = Constraint(range(K), rule=lambda m, k:
        alpha2[k] + ef*p2 - hv*v <= T_low + M_BIG * (1 - m.y2_low[k]))
    m.c_y2l_lb = Constraint(range(K), rule=lambda m, k:
        alpha2[k] + ef*p2 - hv*v >= T_low - M_BIG * m.y2_low[k])

    if lr1 == 0:
        m.c_lr1 = Constraint(range(K), rule=lambda m, k:
            m.lr1_next[k] == m.y1_low[k])
    else:
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

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 5 — Piecewise-linear auxiliary variables  (FIX 2: new features 8-10)
    #
    # z1_low = max(0, T_low + T_DANGER_MARGIN - T1_next)
    #   With η[8] > 0 (higher danger → higher future cost), the minimiser
    #   drives z1_low to its lower bound = max(0, ...).
    #   Bounds (0, T_DANGER_MARGIN+10) prevent blowup if η[8] < 0 after training.
    # ═════════════════════════════════════════════════════════════════════════

    m.z1_low = Var(within=NonNegativeReals, bounds=(0, T_DANGER_MARGIN + 10))
    m.c_z1   = Constraint(expr=m.z1_low >= T_low + T_DANGER_MARGIN - T1_next)

    m.z2_low = Var(within=NonNegativeReals, bounds=(0, T_DANGER_MARGIN + 10))
    m.c_z2   = Constraint(expr=m.z2_low >= T_low + T_DANGER_MARGIN - T2_next)

    m.z_H_app = Var(within=NonNegativeReals, bounds=(0, 20))
    m.c_z_H   = Constraint(expr=m.z_H_app >= H_next - H_APPROACH_THRESH)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 6 — Objective
    #
    # Feature → index mapping (must match train_adp_fvi_v4.py FEATURE_COLS):
    #   T1[0] T2[1] H[2] price_t[3] price_prev[4] vc[5] lr1[6] lr2[7]
    #   T1_danger[8] T2_danger[9] H_approach[10]
    #   price_x_ovr1[11] price_x_ovr2[12]  bias[13]
    # ═════════════════════════════════════════════════════════════════════════

    # Immediate cost (unchanged)
    immediate = price_mean * (p1 + p2 + P_vent * v)

    # VF — constant terms (pure floats)
    vf_const = (  eta_next[3]  * price_mean        # price_t at next step
               +  eta_next[4]  * state["price_t"]  # price_previous at next step
               +  eta_next[13]                     # bias (now at index 13)
               )

    # VF — linear Pyomo terms (features 0-2, 5; indices 6-7 go in vf_binary)
    vf_linear = (  eta_next[0] * T1_next
               +   eta_next[1] * T2_next
               +   eta_next[2] * H_next
               +   eta_next[5] * vc_next    # uses FIXED vc_next
               )

    # VF — per-scenario binary terms (features 6, 7 + NEW 11, 12)
    vf_binary = (1.0 / K) * sum(
        eta_next[6]  * m.lr1_next[k]           # low_override_r1
      + eta_next[7]  * m.lr2_next[k]           # low_override_r2
      + eta_next[11] * prices[k] * m.lr1_next[k]   # FIX 2: price_x_ovr1
      + eta_next[12] * prices[k] * m.lr2_next[k]   # FIX 2: price_x_ovr2
        for k in range(K)
    )

    # VF — piecewise-linear terms (features 8, 9, 10) — FIX 2
    vf_piecewise = (  eta_next[8]  * m.z1_low
                   +  eta_next[9]  * m.z2_low
                   +  eta_next[10] * m.z_H_app
                   )

    m.obj = Objective(
        expr=immediate + vf_const + vf_linear + vf_binary + vf_piecewise,
        sense=minimize
    )

    # ── Solve ─────────────────────────────────────────────────────────────────
    solver = SolverFactory("gurobi")
    solver.options["OutputFlag"] = 0
    solver.options["TimeLimit"]  = 12
    solver.solve(m)

    # ── Extract results ────────────────────────────────────────────────────────
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

    if H > H_high or (0 < vc < vent_min):
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
    ADP one-step lookahead policy (v4).

    Improvements over v3:
      - vc_next bug fixed (was: vc+v; now: vc-1 if vc>0 else (vent_min-1)*v)
      - Five new danger-zone features: T1/T2 cold proximity, humidity approach,
        price×override interaction for both rooms

    At t=9: ETA[10] absent → zero vector → minimise immediate cost only.
    """
    t    = state["current_time"]
    data = _get_data()

    prices, occs, price_mean, occ1_mean, occ2_mean = _sample_scenarios(state)

    n_eta    = len(next(iter(ETA.values())))
    eta_next = ETA.get(t + 1, np.zeros(n_eta))

    result = _solve_milp(state, data, eta_next,
                         prices, occs, price_mean, occ1_mean, occ2_mean)

    return {
        "HeatPowerRoom1": result["HeatPowerRoom1"],
        "HeatPowerRoom2": result["HeatPowerRoom2"],
        "VentilationON":  result["VentilationON"],
    }