#!/usr/bin/env python3
"""
train_adp_fvi.py  –  Fitted Value Iteration (FVI) training for ADP policy
============================================================================

HOW THIS DIFFERS FROM train_adp.py (Approximate Backward Induction):
----------------------------------------------------------------------
  train_adp.py     → targets = actual cost-to-go from replaying a fixed policy
                     (OiH or dummy). Target quality is capped by that policy.

  train_adp_fvi.py → targets = Bellman optimality equation, solved iteratively.
                     At each iteration, eta_{t+1} (current best estimate) is
                     used to evaluate future value. Targets improve every
                     iteration regardless of any fixed policy.

Algorithm (from lecture):
--------------------------
  Sample N states per timestep  {x_{n,t}}
  Sample K future scenarios per state  w_{n,k,t+1}

  Repeat for I iterations:
    For t = T-1 down to 0:
      For each n:
        target[n] = min_{feasible u} { (1/K) Σ_k [ cost_k + V̂(s'_k ; eta_{t+1}) ] }
      Fit eta_t via Ridge on (features[n], target[n])

CONVENTIONS — identical to train_adp_new_features.py for fair comparison:
--------------------------------------------------------------
  - Feature order  : T1, T2, H, price_t, price_previous,
                     vent_counter, low_override_r1, low_override_r2, bias (LAST)
  - Regularisation : ridge_lambda = 10.0, bias NOT penalised
  - Scaling        : StandardScaler on raw features (not bias),
                     weights converted back to raw space before saving
  - Output format  : eta_fvi.pkl  —  dict {t: np.array shape (9,)} RAW space
                     No mu/sigma files needed — policy uses raw state values

v2 changes (vs original):
  - Imports v2_SystemCharacteristics (fixed initial state for T, H; random for price/occ)
  - Loads v2_PriceData.csv (11 cols: col 0 = price_previous, cols 1-10 = hourly prices)
  - Cold-start only — no warm-start from stale pre-v2 eta pkl

Usage:
    python train_adp_fvi.py                      # defaults
    python train_adp_fvi.py samples_sp.csv       # custom samples
    python train_adp_fvi.py samples_sp.csv 5     # custom lambda
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(os.path.dirname(SCRIPT_DIR))   # repo root
sys.path.append(BASE_DIR)

# ── v2 imports ────────────────────────────────────────────────────────────────
from data.v2_SystemCharacteristics import get_fixed_data   # FIX 1: was part_A.SystemCharacteristics
from part_B.RestaurantEnv import step_env

# ── Hyper-parameters ──────────────────────────────────────────────────────────
T_END        = 10     # hourly timesteps per day
N            = 300    # sampled states per timestep (from samples CSV)
K            = 20     # future scenarios per state (Monte Carlo expectation)
I_ITER       = 10     # FVI iterations
RIDGE_LAMBDA = 10.0   # same as train_adp_new_features.py for fair comparison
SEED         = 42

# ── Feature definition — must match train_adp_new_features.py FEATURE_COLS ───
# Bias is NOT listed here; it is appended as the last column in build_features.
FEATURE_COLS = [
    "T1",
    "T2",
    "H",
    "price_t",
    "price_previous",
    "vent_counter",
    "low_override_r1",
    "low_override_r2",
]
N_RAW  = len(FEATURE_COLS)   # 8 raw features
N_FEAT = N_RAW + 1           # 9 total (raw + bias)

# ── Action grid ───────────────────────────────────────────────────────────────
# Cost and VF are both linear in P1, P2 → optimum is always at a boundary.
# {0, 3} is sufficient; step_env override controllers handle clamping.
P_OPTIONS = [0.0, 3.0]
V_OPTIONS = [0, 1]
ALL_ACTIONS = [
    {"HeatPowerRoom1": p1, "HeatPowerRoom2": p2, "VentilationON": v}
    for p1 in P_OPTIONS
    for p2 in P_OPTIONS
    for v  in V_OPTIONS
]   # 8 combinations


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_features_raw(state):
    """
    Returns the raw feature vector of shape (N_FEAT,) = (9,).
    Order: [T1, T2, H, price_t, price_previous, vent_counter,
            low_override_r1, low_override_r2, bias]
    Bias is last — matches train_adp_new_features.py convention.
    """
    return np.array([
        float(state['T1']),
        float(state['T2']),
        float(state['H']),
        float(state['price_t']),
        float(state['price_previous']),
        float(state['vent_counter']),
        float(state['low_override_r1']),
        float(state['low_override_r2']),
        1.0,   # bias LAST
    ])


def estimate_value(state, eta_raw):
    """
    V̂(state) = eta_raw^T · phi_raw(state).
    Uses RAW feature space — no normalisation needed.
    Returns 0.0 at or beyond the terminal timestep (boundary condition V_T = 0).
    """
    if state['current_time'] >= T_END:
        return 0.0
    phi = compute_features_raw(state)
    return float(np.dot(eta_raw, phi))


def fit_eta(features_raw, targets, ridge_lambda):
    """
    Fits eta via Ridge regression on scaled features, then converts back to
    raw feature space. Identical procedure to train_adp_new_features.py.
    """
    n = len(targets)

    scaler   = StandardScaler()
    X_sc     = scaler.fit_transform(features_raw[:, :N_RAW])
    X_sc_b   = np.hstack([X_sc, np.ones((n, 1))])

    Lambda         = ridge_lambda * np.eye(N_FEAT)
    Lambda[-1, -1] = 0.0
    eta_scaled     = np.linalg.solve(X_sc_b.T @ X_sc_b + Lambda,
                                     X_sc_b.T @ targets)

    eta_raw         = np.zeros(N_FEAT)
    eta_raw[:N_RAW] = eta_scaled[:N_RAW] / scaler.scale_
    eta_raw[N_RAW]  = (eta_scaled[N_RAW]
                       - np.sum(eta_scaled[:N_RAW]
                                * scaler.mean_
                                / scaler.scale_))

    y_pred = X_sc_b @ eta_scaled
    ss_res = np.sum((targets - y_pred) ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    return eta_raw, r2


# ─────────────────────────────────────────────────────────────────────────────
# FVI target computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_target(state, base_data, price_k, occ1_k, occ2_k, eta_next_raw):
    """
    FVI Bellman target for a single state x_{n,t}:

        V*(x_{n,t}) = min_{feasible u} { (1/K) Σ_k [ cost_k + V̂(s'_k ; eta_{t+1}) ] }
    """
    t = state['current_time']

    # Ventilation inertia constraint: if vent_counter > 0, must stay ON
    feasible = (
        [a for a in ALL_ACTIONS if a['VentilationON'] == 1]
        if state['vent_counter'] > 0
        else ALL_ACTIONS
    )

    occ_scenario = {"Room1": [0] * T_END, "Room2": [0] * T_END}

    best_value = np.inf

    for action in feasible:
        total = 0.0
        for k in range(K):
            base_data['price'][t]    = float(price_k[k])
            occ_scenario["Room1"][t] = float(occ1_k[k])
            occ_scenario["Room2"][t] = float(occ2_k[k])

            next_state, cost, _ = step_env(state, action, base_data, occ_scenario)
            total += cost + estimate_value(next_state, eta_next_raw)

        avg = total / K
        if avg < best_value:
            best_value = avg

    return best_value


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(samples_path, output_path, ridge_lambda=RIDGE_LAMBDA):

    rng = np.random.default_rng(seed=SEED)

    # ── Load system data (v2) ─────────────────────────────────────────────────
    base_data = get_fixed_data()

    # FIX 2: use v2_PriceData.csv (11 cols: col 0 = price_previous, cols 1-10 = hourly prices)
    v2_price_raw = pd.read_csv(
        os.path.join(BASE_DIR, "data", "v2_PriceData.csv")
    ).values                              # shape (100, 11)
    price_data = v2_price_raw[:, 1:]     # cols 1-10 → hourly prices, shape (100, 10)

    occ1_data  = pd.read_csv(os.path.join(BASE_DIR, "data", "OccupancyRoom1.csv")).values
    occ2_data  = pd.read_csv(os.path.join(BASE_DIR, "data", "OccupancyRoom2.csv")).values
    num_days   = price_data.shape[0]

    print(f"Loading samples from : {samples_path}")
    print(f"Ridge lambda         : {ridge_lambda}")
    print(f"FVI iterations       : {I_ITER}  |  K scenarios : {K}")
    print(f"Price data shape     : {price_data.shape}  (after stripping price_previous col)")

    # ── Load sampled states ───────────────────────────────────────────────────
    if not os.path.exists(samples_path):
        raise FileNotFoundError(
            f"Samples file not found: {samples_path}\n"
            f"Run collect_samples.py first."
        )
    df = pd.read_csv(samples_path)

    print(f"Hours found  : {sorted(df['hour'].unique())}")
    print(f"Days found   : {df['day'].nunique()}")
    print(f"Total rows   : {len(df)}")

    states_by_t = {}
    for t in range(T_END):
        df_t = states_by_t[t] = df[df['hour'] == t].head(N).to_dict(orient='records')
        print(f"  t={t}: {len(df_t)} states loaded")

    # Pre-sample scenario day indices
    scenario_days = rng.integers(0, num_days, size=(T_END, N, K))

    # FIX 3: always cold-start — no warm-start from pre-v2 stale eta
    eta = {t: np.zeros(N_FEAT) for t in range(T_END)}
    print("\nCold start: etas initialised to zeros (v2 — no stale warm-start).")

    # price placeholder — compute_target overwrites base_data['price'][t] per scenario
    base_data['price'] = [0.0] * T_END

    # ── FVI main loop ─────────────────────────────────────────────────────────
    for iteration in range(I_ITER):
        print(f"\n{'='*58}")
        print(f"  FVI Iteration {iteration + 1} / {I_ITER}")
        print(f"{'='*58}")
        print(f"  {'t':>3}  {'N':>6}  {'R2':>8}  {'target_mean':>12}  {'target_std':>11}")
        print(f"  {'-'*46}")

        new_eta = {}

        for t in range(T_END - 1, -1, -1):
            states   = states_by_t[t]
            n_states = len(states)
            t_next   = min(t + 1, T_END - 1)

            # TRUE FVI: use PREVIOUS iteration's eta for future value estimate.
            # Using new_eta here would make every iteration identical (single-pass BI).
            eta_next_raw = eta.get(t + 1, np.zeros(N_FEAT))

            targets      = np.zeros(n_states)
            features_raw = np.zeros((n_states, N_FEAT))

            for n, state in enumerate(states):
                days_k  = scenario_days[t, n]
                # price_data is already sliced to cols 1-10 → index directly by t_next
                price_k = price_data[days_k, t_next]
                occ1_k  = occ1_data[days_k,  t_next]
                occ2_k  = occ2_data[days_k,  t_next]

                targets[n]      = compute_target(
                    state, base_data,
                    price_k, occ1_k, occ2_k,
                    eta_next_raw
                )
                features_raw[n] = compute_features_raw(state)

            new_eta[t], r2 = fit_eta(features_raw, targets, ridge_lambda)

            print(f"  t={t:2d}  {n_states:>6}  {r2:>8.4f}  "
                  f"{targets.mean():>12.4f}  {targets.std():>11.4f}")

        eta = new_eta

    # ── Save ──────────────────────────────────────────────────────────────────
    with open(output_path, 'wb') as f:
        pickle.dump(eta, f)

    print(f"\nSaved ETA (raw space) to: {output_path}")
    print("\nWeights per hour (raw feature space — use directly with unscaled features):")
    print("Feature order: " + ", ".join(FEATURE_COLS) + ", bias")
    for t, eta_t in eta.items():
        print(f"  t={t}: {np.round(eta_t, 4).tolist()}")

    return eta


if __name__ == "__main__":
    samples_path = (sys.argv[1] if len(sys.argv) > 1
                    else os.path.join(SCRIPT_DIR, "samples/samples_sp_fvi_v2.csv"))
    ridge_lambda = float(sys.argv[2]) if len(sys.argv) > 2 else RIDGE_LAMBDA
    output_path  = os.path.join(SCRIPT_DIR, "eta_sp_fvi_v2.pkl")

    train(samples_path, output_path, ridge_lambda)