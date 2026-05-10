"""
ADP Policy — Group 18
Task 4: Approximate Dynamic Programming with Linear Value Function Approximation.

One-step lookahead using learned value function V̂_{t+1}(s').

At each hour t:
  1. Enumerate feasible actions
  2. For each action: call step_env with known current values (deterministic)
  3. Sample next-period uncertainty using the official process models
     (PriceProcessRestaurant, OccupancyProcessRestaurant) conditioned on
     current state — matches the distribution used for grader evaluation
  4. Return action minimising immediate cost + expected future value
"""

import os
import sys
import copy

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
_POLICY_DIR  = os.path.dirname(os.path.abspath(__file__))
_PART_B_DIR  = os.path.dirname(_POLICY_DIR)
_PROJECT_DIR = os.path.dirname(_PART_B_DIR)
_ADP_DIR     = os.path.join(_PART_B_DIR, 'adp')
_WEIGHTS_DIR = os.path.join(_ADP_DIR, 'weights')
_PART_A_DIR  = os.path.join(_PROJECT_DIR, 'part_A')

sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _PART_A_DIR)
sys.path.insert(0, _ADP_DIR)
sys.path.insert(0, _PART_B_DIR)

from value_function import (
    compute_features, normalize_features, estimate_value,
    NUM_FEATURES, T_END
)
from SystemCharacteristics import get_fixed_data
from part_B.RestaurantEnv import step_env
from PriceProcessRestaurant import price_model
from OccupancyProcessRestaurant import next_occupancy_levels

# ── Load weights (once at import) ─────────────────────────────────────────────
_thetas = np.load(os.path.join(_WEIGHTS_DIR, 'thetas.npy'))
_mu     = np.load(os.path.join(_WEIGHTS_DIR, 'mu.npy'))
_sigma  = np.load(os.path.join(_WEIGHTS_DIR, 'sigma.npy'))

assert _thetas.shape == (T_END, NUM_FEATURES), (
    f"thetas shape mismatch: expected ({T_END},{NUM_FEATURES}), got {_thetas.shape}"
)

# ── System parameters ─────────────────────────────────────────────────────────
_base_data = get_fixed_data()
_P_MAX     = _base_data['heating_max_power']
_P_VENT    = _base_data['ventilation_power']

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_SCENARIOS   = 20    # more scenarios = more stable estimate
N_HEAT_LEVELS = 5     # 0, 25%, 50%, 75%, 100% of P_MAX

_HEAT_LEVELS = [round(i / (N_HEAT_LEVELS - 1) * _P_MAX, 4)
                for i in range(N_HEAT_LEVELS)]


# ── Candidate actions ─────────────────────────────────────────────────────────

def _candidate_actions(state):
    """
    Return all feasible action dicts given current state constraints.

    Feasibility rules from RestaurantEnv:
      vent_counter > 0    → ventilation must stay ON
      low_override_r1 = 1 → heater r1 will be forced to P_MAX by env anyway,
                            so only submit p1=0 to pass the feasibility check
      low_override_r2 = 1 → same for r2
    """
    vent_counter = state['vent_counter']
    low_r1       = state['low_override_r1']
    low_r2       = state['low_override_r2']

    vent_options = [1] if vent_counter > 0 else [0, 1]
    p1_options   = [0.0] if low_r1 == 1 else _HEAT_LEVELS
    p2_options   = [0.0] if low_r2 == 1 else _HEAT_LEVELS

    return [
        {"HeatPowerRoom1": p1, "HeatPowerRoom2": p2, "VentilationON": v}
        for v  in vent_options
        for p1 in p1_options
        for p2 in p2_options
    ]


# ── Sample next-period uncertainty using process models ───────────────────────

def _sample_next_period(state, n):
    """
    Generate n scenarios for (price_{t+1}, occ1_{t+1}, occ2_{t+1}) using
    the official process models, conditioned on the current state.

    Using the process models (not historical data) ensures scenarios match
    the distribution the graders use when evaluating the policy.
    """
    scenarios = []
    for _ in range(n):
        next_price = price_model(
            state['price_t'],
            state['price_previous']
        )
        next_occ1, next_occ2 = next_occupancy_levels(
            state['Occ1'],
            state['Occ2']
        )
        scenarios.append((float(next_price), float(next_occ1), float(next_occ2)))
    return scenarios


# ── Evaluate one action ───────────────────────────────────────────────────────

def _evaluate_action(state, action, data, occupancy, t):
    """
    Expected total cost for one action under the Bellman equation:

        Q(s, a) = c_t(s, a)  +  (1/N) * Σ_n V̂_{t+1}(s'_n)

    Step 1 — deterministic transition with known current values:
              call step_env → get next_state and immediate cost.
              No scenario patching here — current price and occupancy
              are already known.

    Step 2 — approximate E[V̂_{t+1}(s')] by sampling N scenarios from
              the process models, patching next_state with each sampled
              (price_{t+1}, occ1_{t+1}, occ2_{t+1}), and averaging.
    """
    # Step 1 — deterministic transition (current values are known)
    next_state, cost, _ = step_env(state, action, data, occupancy)

    # Terminal hour — no future value
    if t + 1 >= T_END:
        return float(cost)

    # Step 2 — expected future value via process model sampling
    scenarios = _sample_next_period(state, N_SCENARIOS)
    futures   = []

    for (sc_price, sc_occ1, sc_occ2) in scenarios:
        # Patch next_state with sampled next-period values.
        # These are the uncertain quantities not yet revealed at hour t.
        ns = dict(next_state)
        ns['price_previous'] = next_state['price_t']  # shift price history
        ns['price_t']        = sc_price
        ns['Occ1']           = sc_occ1
        ns['Occ2']           = sc_occ2

        futures.append(estimate_value(ns, _thetas[t + 1], _mu, _sigma))

    return float(cost) + float(np.mean(futures))


# ── Main policy function ──────────────────────────────────────────────────────

def select_action(state):
    """
    ADP one-step lookahead policy.

    Implements the Bellman optimality equation:
        a* = argmin_a [ c_t(s,a) + E[V̂_{t+1}(s'(s,a,ξ))] ]

    Parameters
    ----------
    state : dict — current environment state from RestaurantEnv

    Returns
    -------
    dict with keys HeatPowerRoom1, HeatPowerRoom2, VentilationON
    """
    t = state['current_time']

    # Build data and occupancy dicts for step_env.
    # Only index t is read by step_env — fill all positions with known
    # current values so no patching is needed during action evaluation.
    data = copy.deepcopy(_base_data)
    data['price'] = [state['price_t']] * T_END

    occupancy = {
        'Room1': [state['Occ1']] * T_END,
        'Room2': [state['Occ2']] * T_END,
    }

    candidates  = _candidate_actions(state)
    best_action = candidates[0]
    best_total  = float('inf')

    for action in candidates:
        total = _evaluate_action(state, action, data, occupancy, t)
        if total < best_total:
            best_total  = total
            best_action = action

    return best_action
