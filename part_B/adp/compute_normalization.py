"""
Loads oih_states.csv, computes per-feature mean and standard deviation
from the full feature matrix, and saves mu.npy and sigma.npy.

Input  : part_B/adp/weights/oih_states.csv
Outputs: part_B/adp/weights/mu.npy     — shape (NUM_FEATURES,)
         part_B/adp/weights/sigma.npy  — shape (NUM_FEATURES,)
"""

import os
import sys

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, 'weights')

sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, SCRIPT_DIR)   # so value_function.py is importable directly

from value_function import compute_features, NUM_FEATURES

# Feature names — must match the order in compute_features exactly
FEATURE_NAMES = [
    'bias',
    'T1',
    'T2',
    'T1 - T_low',
    'T2 - T_low',
    'T_high - T1',
    'T_high - T2',
    'max(0, T_low - T1)^2',
    'max(0, T_low - T2)^2',
    'low_override_r1',
    'low_override_r2',
    'H',
    'H_high - H',
    'vent_counter',
    'time_remaining',
    'price_t',
    'time_remaining * price_t',
    'price_t - price_prev',
    'occ1',
    'occ2',
]

assert len(FEATURE_NAMES) == NUM_FEATURES, (
    f"FEATURE_NAMES has {len(FEATURE_NAMES)} entries but NUM_FEATURES={NUM_FEATURES}"
)


def row_to_state(row) -> dict:
    """
    Reconstruct a state dict from a CSV row (pandas Series).
    Casts integer fields explicitly — pandas reads them as float64 from CSV.
    """
    return {
        'T1':              float(row['T1']),
        'T2':              float(row['T2']),
        'H':               float(row['H']),
        'Occ1':            float(row['Occ1']),
        'Occ2':            float(row['Occ2']),
        'price_t':         float(row['price_t']),
        'price_previous':  float(row['price_previous']),
        'vent_counter':    int(row['vent_counter']),
        'low_override_r1': int(row['low_override_r1']),
        'low_override_r2': int(row['low_override_r2']),
        'current_time':    int(row['current_time']),
    }


def main():
    # ── Load CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(WEIGHTS_DIR, 'oih_states.csv')
    assert os.path.exists(csv_path), (
        f"oih_states.csv not found at {csv_path}\n"
        "Run collect_oih_states.py first."
    )

    df = pd.read_csv(csv_path)
    n_rows = len(df)
    print(f"Loaded {n_rows} rows from oih_states.csv")

    # ── Build feature matrix ──────────────────────────────────────────────────
    print("Computing features...")
    Phi = np.zeros((n_rows, NUM_FEATURES), dtype=float)

    for i, (_, row) in enumerate(df.iterrows()):
        state     = row_to_state(row)
        Phi[i, :] = compute_features(state)

    print(f"Feature matrix shape: {Phi.shape}")

    # ── Compute mu and sigma ──────────────────────────────────────────────────
    mu    = Phi.mean(axis=0)
    sigma = Phi.std(axis=0)

    # Guard: constant features (sigma ~ 0) would cause division by zero
    # in normalize_features. Set their sigma to 1.0 so they pass through unchanged.
    constant_features = np.where(sigma < 1e-8)[0]
    sigma = np.where(sigma < 1e-8, 1.0, sigma)

    # ── Save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(WEIGHTS_DIR, 'mu.npy'),    mu)
    np.save(os.path.join(WEIGHTS_DIR, 'sigma.npy'), sigma)

    # ── Print summary table ───────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  {'idx':>3}  {'feature':<26}  {'mu':>10}  {'sigma':>10}  {'note'}")
    print(f"{'─'*65}")

    for i in range(NUM_FEATURES):
        note = '⚠ constant — sigma set to 1.0' if i in constant_features else ''
        print(f"  {i:>3}  {FEATURE_NAMES[i]:<26}  {mu[i]:>10.4f}  {sigma[i]:>10.4f}  {note}")

    print(f"{'─'*65}")

    if len(constant_features) > 0:
        print(f"\n  ⚠  {len(constant_features)} constant feature(s) detected: "
              f"indices {list(constant_features)}")
        print("  These features carry no information and can be removed from")
        print("  compute_features() to simplify the model.")
    else:
        print("\n  All features have non-zero variance. No issues found.")

    print(f"\n  Saved: mu.npy    — shape {mu.shape}")
    print(f"  Saved: sigma.npy — shape {sigma.shape}")
    print(f"  Location: {WEIGHTS_DIR}")


if __name__ == '__main__':
    main()