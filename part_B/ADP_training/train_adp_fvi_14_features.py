#!/usr/bin/env python3
"""
train_adp_fvi_v4.py  -  FVI training with danger-zone feature engineering
============================================================================

Changes from train_adp_fvi.py (v3):
--------------------------------------
  1. Five new features added (N_FEAT 9 → 14):
       T1_danger    = max(0, T_low + T_DANGER_MARGIN - T1)
       T2_danger    = max(0, T_low + T_DANGER_MARGIN - T2)
       H_approach   = max(0, H - H_APPROACH_THRESH)
       price_x_ovr1 = price_t * low_override_r1
       price_x_ovr2 = price_t * low_override_r2

     Rationale:
       - Raw T1/T2 cannot represent the nonlinear cost cliff at T_low=18°C.
         A linear VF treats T1=19.5 and T1=21.5 as proportionally different,
         but the real cost has a kink: crossing T_low forces P_max heating
         at whatever price happens to be current.
       - T1_danger/T2_danger fire within T_DANGER_MARGIN=3°C of T_low, giving
         the VFA a continuous signal that pre-emptive heating is valuable.
       - H_approach fires within 10 units of H_high=70, capturing the risk
         of a forced 3-hour ventilation lock-in at bad prices.
       - price_x_ovr1/2 directly measure the cost of being trapped in override
         at the next step's price — the single largest driver of high max-cost days.

  2. Output: eta_sp_fvi_v4.pkl   (raw weight space, 14 values per timestep)
  3. No other hyper-parameters changed for fair comparison.

No new samples needed — all 5 new features are computable from the existing
state columns already present in samples_sp_fvi_v3.csv (or any v3 sample file).

Usage:
    python train_adp_fvi_v4.py                       # uses v3 samples by default
    python train_adp_fvi_v4.py path/to/samples.csv   # custom samples
    python train_adp_fvi_v4.py samples.csv 10        # custom ridge lambda
"""

import os
import sys
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.append(BASE_DIR)

from data.v2_SystemCharacteristics import get_fixed_data
from part_B.RestaurantEnv import step_env

# ── Hyper-parameters ──────────────────────────────────────────────────────────
T_END        = 10
N            = 300
K            = 20
I_ITER       = 50
RIDGE_LAMBDA = 4.0
SEED         = 42

# ── Danger-zone thresholds (must match ADP_policy_18_v4.py) ──────────────────
T_LOW             = 18.0   # from SystemCharacteristics
T_DANGER_MARGIN   = 3.0    # feature fires when T < T_LOW + margin = 21 °C
H_APPROACH_THRESH = 60.0   # feature fires when H > 60 (threshold is 70)

# ── Feature definition ────────────────────────────────────────────────────────
# Bias appended as the LAST column in compute_features_raw (index N_FEAT-1).
FEATURE_COLS = [
    "T1",               # 0  — raw temperature room 1
    "T2",               # 1  — raw temperature room 2
    "H",                # 2  — raw humidity
    "price_t",          # 3  — current electricity price
    "price_previous",   # 4  — previous electricity price
    "vent_counter",     # 5  — ventilation countdown (0 = free to turn off)
    "low_override_r1",  # 6  — overrule controller active room 1 (0/1)
    "low_override_r2",  # 7  — overrule controller active room 2 (0/1)
    # ── NEW ──────────────────────────────────────────────────────────────────
    "T1_danger",        # 8  — max(0, T_low + margin - T1): proximity to cold danger
    "T2_danger",        # 9  — max(0, T_low + margin - T2): proximity to cold danger
    "H_approach",       # 10 — max(0, H - H_APPROACH_THRESH): proximity to vent trigger
    "price_x_ovr1",     # 11 — price_t * low_override_r1: override cost signal room 1
    "price_x_ovr2",     # 12 — price_t * low_override_r2: override cost signal room 2
]
N_RAW  = len(FEATURE_COLS)   # 13 raw features
N_FEAT = N_RAW + 1           # 14 total (13 raw + 1 bias)

# ── Action grid ───────────────────────────────────────────────────────────────
# VF is piecewise-linear (not just linear) in p1, p2 after new features.
# BUT the kink in max(0, c - ef*p1) is still monotone → optimum still at boundary.
# Proof: d/dp1 max(0,c-ef*p1) = -ef when p1<kink, 0 when p1>kink — always ≤ 0.
# So {0, P_max} enumeration remains sufficient. ✓
P_OPTIONS = [0.0, 3.0]
V_OPTIONS = [0, 1]
ALL_ACTIONS = [
    {"HeatPowerRoom1": p1, "HeatPowerRoom2": p2, "VentilationON": v}
    for p1 in P_OPTIONS
    for p2 in P_OPTIONS
    for v  in V_OPTIONS
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_features_raw(state):
    """
    Returns raw feature vector of shape (N_FEAT,) = (14,).
    Order matches FEATURE_COLS above; bias is last (index 13).
    """
    T1      = float(state['T1'])
    T2      = float(state['T2'])
    H       = float(state['H'])
    price_t = float(state['price_t'])
    lr1     = float(state['low_override_r1'])
    lr2     = float(state['low_override_r2'])

    return np.array([
        # ── original 8 raw features ──────────────────────────────────────────
        T1,
        T2,
        H,
        price_t,
        float(state['price_previous']),
        float(state['vent_counter']),
        lr1,
        lr2,
        # ── 5 new engineered features ────────────────────────────────────────
        max(0.0, T_LOW + T_DANGER_MARGIN - T1),   # T1_danger  [idx 8]
        max(0.0, T_LOW + T_DANGER_MARGIN - T2),   # T2_danger  [idx 9]
        max(0.0, H - H_APPROACH_THRESH),           # H_approach [idx 10]
        price_t * lr1,                             # price_x_ovr1 [idx 11]
        price_t * lr2,                             # price_x_ovr2 [idx 12]
        # ── bias last ────────────────────────────────────────────────────────
        1.0,                                       # bias [idx 13]
    ])


def estimate_value(state, eta_raw):
    """V̂(state) = eta_raw · phi_raw(state). Returns 0 at terminal step."""
    if state['current_time'] >= T_END:
        return 0.0
    return float(np.dot(eta_raw, compute_features_raw(state)))


def fit_eta(features_raw, targets, ridge_lambda):
    """
    Ridge regression in scaled space, weights converted back to raw space.
    Identical procedure to v3 — only N_RAW/N_FEAT change.
    """
    n = len(targets)

    scaler   = StandardScaler()
    X_sc     = scaler.fit_transform(features_raw[:, :N_RAW])
    X_sc_b   = np.hstack([X_sc, np.ones((n, 1))])

    Lambda         = ridge_lambda * np.eye(N_FEAT)
    Lambda[-1, -1] = 0.0                               # bias not penalised
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
# FVI target computation  (unchanged from v3 — uses step_env directly)
# ─────────────────────────────────────────────────────────────────────────────

def compute_target(state, base_data, price_k, occ1_k, occ2_k, eta_next_raw):
    """
    FVI Bellman target:
        V*(x_{n,t}) = min_u { (1/K) Σ_k [ cost_k + V̂(s'_k ; eta_{t+1}) ] }

    step_env handles vc_next correctly — no manual vc logic needed here.
    New features are computed inside estimate_value → compute_features_raw.
    """
    t = state['current_time']

    feasible = (
        [a for a in ALL_ACTIONS if a['VentilationON'] == 1]
        if state['vent_counter'] > 0
        else ALL_ACTIONS
    )

    occ_scenario = {"Room1": [0] * T_END, "Room2": [0] * T_END}
    best_value   = np.inf

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

    base_data = get_fixed_data()

    v2_price_raw = pd.read_csv(
        os.path.join(BASE_DIR, "data", "v2_PriceData.csv")
    ).values
    price_data = v2_price_raw[:, 1:]     # shape (100, 10)

    occ1_data = pd.read_csv(os.path.join(BASE_DIR, "data", "OccupancyRoom1.csv")).values
    occ2_data = pd.read_csv(os.path.join(BASE_DIR, "data", "OccupancyRoom2.csv")).values
    num_days  = price_data.shape[0]

    print(f"Loading samples from : {samples_path}")
    print(f"Ridge lambda         : {ridge_lambda}")
    print(f"FVI iterations       : {I_ITER}  |  K scenarios : {K}")
    print(f"N_FEAT               : {N_FEAT}  ({N_RAW} raw + 1 bias)")
    print(f"New features         : T1_danger [8], T2_danger [9], H_approach [10],")
    print(f"                       price_x_ovr1 [11], price_x_ovr2 [12]")
    print(f"Danger margin        : T_DANGER_MARGIN={T_DANGER_MARGIN} (fires when T < {T_LOW+T_DANGER_MARGIN}°C)")
    print(f"H approach thresh    : H_APPROACH_THRESH={H_APPROACH_THRESH} (fires when H > {H_APPROACH_THRESH}%)")

    if not os.path.exists(samples_path):
        raise FileNotFoundError(
            f"Samples file not found: {samples_path}\n"
            f"Pass the v3 samples file — no new samples needed."
        )
    df = pd.read_csv(samples_path)

    print(f"\nHours found  : {sorted(df['hour'].unique())}")
    print(f"Days found   : {df['day'].nunique()}")
    print(f"Total rows   : {len(df)}")

    states_by_t = {}
    for t in range(T_END):
        df_t = states_by_t[t] = df[df['hour'] == t].head(N).to_dict(orient='records')
        print(f"  t={t}: {len(df_t)} states loaded")

    scenario_days = rng.integers(0, num_days, size=(T_END, N, K))

    eta = {t: np.zeros(N_FEAT) for t in range(T_END)}
    print("\nCold start: etas initialised to zeros.")

    base_data['price'] = [0.0] * T_END

    for iteration in range(I_ITER):
        print(f"\n{'='*62}")
        print(f"  FVI Iteration {iteration + 1} / {I_ITER}")
        print(f"{'='*62}")
        print(f"  {'t':>3}  {'N':>6}  {'R2':>8}  {'target_mean':>12}  {'target_std':>11}")
        print(f"  {'-'*50}")

        new_eta = {}

        for t in range(T_END - 1, -1, -1):
            states   = states_by_t[t]
            n_states = len(states)
            t_next   = min(t + 1, T_END - 1)

            eta_next_raw = eta.get(t + 1, np.zeros(N_FEAT))

            targets      = np.zeros(n_states)
            features_raw = np.zeros((n_states, N_FEAT))

            for n, state in enumerate(states):
                days_k  = scenario_days[t, n]
                price_k = price_data[days_k, t_next]
                occ1_k  = occ1_data[days_k, t_next]
                occ2_k  = occ2_data[days_k, t_next]

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

    with open(output_path, 'wb') as f:
        pickle.dump(eta, f)

    print(f"\nSaved ETA (raw space, 14 weights per timestep) to: {output_path}")
    print("\nFeature order: " + ", ".join(FEATURE_COLS) + ", bias")
    print("\nWeights per hour:")
    for t, eta_t in eta.items():
        print(f"  t={t}: {np.round(eta_t, 4).tolist()}")

    # Quick sanity check: new feature weights should be positive (more danger = more cost)
    print("\n=== Sanity check: new feature weight signs ===")
    print("(Expected positive — higher danger signal = higher future cost)")
    for t, eta_t in eta.items():
        signs = {
            "T1_danger":    "+" if eta_t[8]  > 0 else "-",
            "T2_danger":    "+" if eta_t[9]  > 0 else "-",
            "H_approach":   "+" if eta_t[10] > 0 else "-",
            "price_x_ovr1": "+" if eta_t[11] > 0 else "-",
            "price_x_ovr2": "+" if eta_t[12] > 0 else "-",
        }
        print(f"  t={t}: {signs}")

    return eta


if __name__ == "__main__":
    samples_path = (sys.argv[1] if len(sys.argv) > 1
                    else os.path.join(SCRIPT_DIR, "samples/samples_sp_fvi_v3.csv"))
    ridge_lambda = float(sys.argv[2]) if len(sys.argv) > 2 else RIDGE_LAMBDA
    output_path  = os.path.join(SCRIPT_DIR, "eta_sp_fvi_14_features.pkl")

    train(samples_path, output_path, ridge_lambda)