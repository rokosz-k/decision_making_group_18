#!/usr/bin/env python3
"""Approximate Backward Induction training for the HVAC control problem."""

import os
import sys
import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
PART_A_DIR  = os.path.join(PROJECT_DIR, 'part_A')
PART_B_DIR  = os.path.join(PROJECT_DIR, 'part_B')

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, PART_A_DIR)
sys.path.insert(0, PART_B_DIR)
sys.path.insert(0, SCRIPT_DIR)

# ── Imports ───────────────────────────────────────────────────────────────────
from SystemCharacteristics import get_fixed_data
from RestaurantEnv import reset_env, step_env
from PriceProcessRestaurant import price_model
from OccupancyProcessRestaurant import next_occupancy_levels
from value_function import compute_features, NUM_FEATURES, T_END

# ── Hyperparameters ───────────────────────────────────────────────────────────
N_ITER      = 5
N_DAYS      = 100
N_SCENARIOS = 10
LAMBDA_REG  = 10.0       # stronger regularization prevents weight blow-up
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, 'weights')

np.random.seed(42)

# ── System parameters ─────────────────────────────────────────────────────────
BASE_DATA     = get_fixed_data()
P_MAX         = BASE_DATA['heating_max_power']
HEATER_LEVELS = np.linspace(0.0, P_MAX, 5).tolist()

# ── Historical data (used only for seeding initial states) ────────────────────
DATA_DIR   = os.path.join(PROJECT_DIR, 'data')
price_hist = pd.read_csv(os.path.join(DATA_DIR, 'PriceData.csv')).values
occ1_hist  = pd.read_csv(os.path.join(DATA_DIR, 'OccupancyRoom1.csv')).values
occ2_hist  = pd.read_csv(os.path.join(DATA_DIR, 'OccupancyRoom2.csv')).values
N_HIST     = price_hist.shape[0]

os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_step_data(state):
    """
    Minimal data dict for step_env in the backward pass.
    step_env reads data['price'][t] — fill all positions with current price.
    """
    d = dict(BASE_DATA)
    d['price'] = [float(state['price_t'])] * T_END
    return d


def make_step_occupancy(state):
    """
    Minimal occupancy dict for step_env in the backward pass.
    step_env reads occupancy['Room1'][t] — fill all with current values.
    """
    return {
        'Room1': [float(state['Occ1'])] * T_END,
        'Room2': [float(state['Occ2'])] * T_END,
    }


def get_feasible_actions(state):
    """
    Enumerate feasible (p1, p2, vent) combinations given state constraints.

    Rules from RestaurantEnv:
      vent_counter > 0    → ventilation must stay ON
      low_override_r1 = 1 → heater r1 forced to P_MAX by env, submit p1=0
      low_override_r2 = 1 → same for r2
    """
    p1_opts   = [0.0] if state['low_override_r1'] == 1 else HEATER_LEVELS
    p2_opts   = [0.0] if state['low_override_r2'] == 1 else HEATER_LEVELS
    vent_opts = [1]   if state['vent_counter'] > 0     else [0, 1]
    return [
        {'HeatPowerRoom1': float(p1), 'HeatPowerRoom2': float(p2), 'VentilationON': v}
        for p1 in p1_opts for p2 in p2_opts for v in vent_opts
    ]


def fit_ridge(Phi, targets):
    """
    Ridge regression: theta = (Phi^T Phi + lambda I)^{-1} Phi^T targets
    Always used — prevents numerical blow-up from near-rank-deficient matrices.
    """
    d   = Phi.shape[1]
    A   = Phi.T @ Phi + LAMBDA_REG * np.eye(d)
    b   = Phi.T @ targets
    sol = np.linalg.solve(A, b)

    if np.any(np.isnan(sol)) or np.any(np.isinf(sol)):
        print(f"    [WARNING] Ridge produced NaN/Inf — using zeros")
        return np.zeros(d)

    return sol


def q_value(state, action, thetas_next, data=None, occ=None):
    """
    Q(s, a) = immediate_cost + mean V_{t+1}(s') over N_SCENARIOS

    Implements one step of the Bellman equation:
      - Transition with known current values is deterministic
      - Uncertainty is over next-period price and occupancy, sampled
        from the official process models conditioned on current state

    Future values are clipped to >= 0 because costs are always non-negative.
    Without the clip, early-iteration thetas produce large negative dot
    products that cascade into negative targets, breaking convergence.
    """
    if data is None:
        data = make_step_data(state)
    if occ is None:
        occ = make_step_occupancy(state)

    next_state, imm_cost, done = step_env(state, action, data, occ)

    # Terminal hour — no future cost
    if done:
        return float(imm_cost)

    # Sample next-period uncertainty from process models
    future_vals = []
    for _ in range(N_SCENARIOS):
        sc_price         = price_model(state['price_t'], state['price_previous'])
        sc_occ1, sc_occ2 = next_occupancy_levels(state['Occ1'], state['Occ2'])

        patched                   = dict(next_state)
        patched['price_previous'] = next_state['price_t']
        patched['price_t']        = float(sc_price)
        patched['Occ1']           = float(sc_occ1)
        patched['Occ2']           = float(sc_occ2)

        # Clip to >= 0: costs are non-negative so future value must be too
        v = float(np.dot(thetas_next, compute_features(patched)))
        future_vals.append(max(0.0, v))

    return float(imm_cost) + float(np.mean(future_vals))


def best_action_and_q(state, thetas_next, data=None, occ=None):
    """Return the action with the lowest Q(s,a) and its value."""
    best_q   = np.inf
    best_act = None
    for act in get_feasible_actions(state):
        q = q_value(state, act, thetas_next, data=data, occ=occ)
        if q < best_q:
            best_q, best_act = q, act
    return best_act, best_q


# ── Trajectory generation ─────────────────────────────────────────────────────

def generate_trajectory(day_idx):
    """
    Build a full-day price and occupancy trajectory.
    Hour 0 seeded from historical data.
    Hours 1-9 generated using the process models — matches grader distribution.
    """
    prices = [float(price_hist[day_idx, 0])]
    occ1s  = [float(occ1_hist[day_idx, 0])]
    occ2s  = [float(occ2_hist[day_idx, 0])]
    prev_p = 0.0

    for _ in range(1, T_END):
        new_p          = price_model(prices[-1], prev_p)
        prev_p         = prices[-1]
        new_o1, new_o2 = next_occupancy_levels(occ1s[-1], occ2s[-1])
        prices.append(float(new_p))
        occ1s.append(float(new_o1))
        occ2s.append(float(new_o2))

    data          = dict(BASE_DATA)
    data['price'] = prices
    occ           = {'Room1': occ1s, 'Room2': occ2s}
    return data, occ


# ── Forward pass ──────────────────────────────────────────────────────────────

def forward_pass(iteration, thetas_prev):
    """
    Simulate N_DAYS days and collect one state per stage per day.

    Iteration 1: dummy policy — broadly explores the state space.
    Iterations 2+: greedy policy using thetas_prev.

    Initial states are perturbed with small noise so the t=0 feature
    matrix is not rank-deficient (all days start from identical reset_env
    initial conditions without perturbation — Ridge cannot fit a line
    through 100 identical points).
    """
    print(f"  [Iter {iteration}] Forward pass ...")
    states_per_stage = [[] for _ in range(T_END)]

    day_indices = np.random.choice(N_HIST, size=N_DAYS, replace=True)

    for day_idx in day_indices:
        data, occ = generate_trajectory(int(day_idx))
        state     = reset_env(data, occ)

        # Perturb initial state to break rank-deficiency at t=0
        # reset_env always returns the same temperature and humidity,
        # making all t=0 feature vectors identical without this noise.
        state['T1'] += np.random.uniform(-1.0, 1.0)
        state['T2'] += np.random.uniform(-1.0, 1.0)
        state['H']  += np.random.uniform(-2.0, 2.0)

        states_per_stage[0].append(dict(state))

        for t in range(T_END - 1):
            if iteration == 1:
                action = {
                    'HeatPowerRoom1': 0.0,
                    'HeatPowerRoom2': 0.0,
                    'VentilationON':  0,
                }
            else:
                thetas_next = thetas_prev[t + 1]
                action, _   = best_action_and_q(state, thetas_next,
                                                data=data, occ=occ)

            next_state, _, _ = step_env(state, action, data, occ)
            state            = next_state
            states_per_stage[t + 1].append(dict(state))

    return states_per_stage


# ── Backward pass ─────────────────────────────────────────────────────────────

def backward_pass(iteration, states_per_stage, thetas_prev):
    """
    Fit one theta per stage from t = T_END-2 down to t = 0.
    thetas[T_END-1] = zeros (terminal: no future cost).

    For each stage t:
      target(s) = min_a Q(s, a, thetas[t+1])   ← Bellman target
      thetas[t] = Ridge(Phi, targets)

    Backward ordering guarantees thetas[t+1] is already fitted
    when computing targets for stage t.
    """
    print(f"  [Iter {iteration}] Backward pass ...")
    thetas       = np.zeros((T_END, NUM_FEATURES))
    mean_targets = {}

    for t in range(T_END - 2, -1, -1):
        thetas_next = thetas[t + 1]
        states_t    = states_per_stage[t]

        targets = []
        for s in states_t:
            actions = get_feasible_actions(s)
            qs      = [q_value(s, a, thetas_next) for a in actions]
            targets.append(min(qs))

        targets = np.array(targets, dtype=float)
        Phi     = np.vstack([compute_features(s) for s in states_t])

        sol = fit_ridge(Phi, targets)

        # Safety fallback if Ridge still produces unusable weights
        if np.linalg.norm(sol) > 1e6:
            print(f"    [FALLBACK] t={t}: keeping previous iteration's theta")
            sol = thetas_prev[t].copy()

        thetas[t]       = sol
        mean_targets[t] = float(np.mean(targets))

        print(
            f"    Iter {iteration}  stage t={t:2d}  "
            f"mean_target={mean_targets[t]:9.4f}  "
            f"theta_norm={np.linalg.norm(sol):.4f}"
        )

    return thetas, mean_targets


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    thetas       = np.zeros((T_END, NUM_FEATURES))
    iter_mean_t0 = []

    for iteration in range(1, N_ITER + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration} / {N_ITER}")
        print(f"{'='*60}")

        thetas_prev       = thetas.copy()
        states_per_stage  = forward_pass(iteration, thetas_prev)
        thetas, mean_tgts = backward_pass(iteration, states_per_stage, thetas_prev)

        np.save(os.path.join(WEIGHTS_DIR, 'thetas.npy'), thetas)
        print(f"  Weights saved → {WEIGHTS_DIR}/thetas.npy")

        mean_t0 = mean_tgts.get(0, float('nan'))
        iter_mean_t0.append(mean_t0)
        print(f"  Iteration {iteration} summary: mean target at t=0 = {mean_t0:.4f}")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print("Per-iteration mean target at t=0:")
    print("(should be largest across stages, trending toward OiH avg ~111)")
    for i, v in enumerate(iter_mean_t0, 1):
        print(f"  Iteration {i}: {v:.4f}")
    print(f"Weights saved to: {WEIGHTS_DIR}")


if __name__ == '__main__':
    main()
