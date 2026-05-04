"""
Phase 1 regression: fits one theta vector per stage using OiH states as
training data. Targets are the cost-to-go values recorded by collect_oih_states.py.

Uses Ridge regression instead of plain least squares to handle multicollinearity
between features (T1 vs T1-T_low vs T_high-T1, H vs H_high-H etc.)

Input  : part_B/adp/weights/oih_states.csv
         part_B/adp/weights/mu.npy
         part_B/adp/weights/sigma.npy
Output : part_B/adp/weights/thetas.npy   — shape (T_END, NUM_FEATURES)
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
sys.path.insert(0, os.path.join(PROJECT_DIR, 'part_B', 'adp'))

from value_function import compute_features, normalize_features, NUM_FEATURES, T_END

# ── Ridge regularization strength ─────────────────────────────────────────────
# Prevents numerical blow-up from collinear features.
# Increase if theta norms are still large; decrease if training RMSE is too high.
LAMBDA_REG = 1.0


def row_to_state(row) -> dict:
    """Reconstruct a state dict from a CSV row (pandas Series)."""
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


def fit_ridge(Phi, targets, lambda_reg):
    """
    Ridge regression: minimise ||Phi @ theta - targets||^2 + lambda ||theta||^2

    Closed-form solution: theta = (Phi^T Phi + lambda I)^{-1} Phi^T targets

    Parameters
    ----------
    Phi        : np.ndarray (n, d)
    targets    : np.ndarray (n,)
    lambda_reg : float — regularization strength

    Returns
    -------
    theta : np.ndarray (d,)
    rmse  : float — training RMSE after fit
    """
    d     = Phi.shape[1]
    A     = Phi.T @ Phi + lambda_reg * np.eye(d)
    b     = Phi.T @ targets
    theta = np.linalg.solve(A, b)
    rmse  = float(np.sqrt(np.mean((Phi @ theta - targets) ** 2)))
    return theta, rmse


def main():
    # ── Load inputs ───────────────────────────────────────────────────────────
    csv_path = os.path.join(WEIGHTS_DIR, 'oih_states.csv')
    assert os.path.exists(csv_path), (
        f"oih_states.csv not found at {csv_path}\n"
        "Run collect_oih_states.py first."
    )

    mu    = np.load(os.path.join(WEIGHTS_DIR, 'mu.npy'))
    sigma = np.load(os.path.join(WEIGHTS_DIR, 'sigma.npy'))
    assert mu.shape    == (NUM_FEATURES,), f"mu shape mismatch: {mu.shape}"
    assert sigma.shape == (NUM_FEATURES,), f"sigma shape mismatch: {sigma.shape}"

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from oih_states.csv")
    print(f"Fitting thetas for {T_END} stages  (lambda_reg={LAMBDA_REG})...\n")

    # ── Fit one theta per stage ───────────────────────────────────────────────
    thetas = np.zeros((T_END, NUM_FEATURES), dtype=float)

    print(f"  {'stage':>5}  {'n':>5}  {'train_RMSE':>12}  {'|theta|':>10}  note")
    print(f"  {'─'*58}")

    for t in range(T_END - 1, -1, -1):   # T_END-1 down to 0

        stage_df = df[df['hour'] == t]
        n        = len(stage_df)

        # Last stage: no future — value is always 0 by definition
        if t == T_END - 1:
            thetas[t] = np.zeros(NUM_FEATURES)
            print(f"  {t:>5}  {n:>5}  {'—':>12}  {'—':>10}  terminal — set to zeros")
            continue

        # Build normalized feature matrix and target vector
        Phi     = np.zeros((n, NUM_FEATURES), dtype=float)
        targets = np.zeros(n, dtype=float)

        for i, (_, row) in enumerate(stage_df.iterrows()):
            state      = row_to_state(row)
            phi_raw    = compute_features(state)
            Phi[i, :]  = normalize_features(phi_raw, mu, sigma)
            targets[i] = float(row['cost_to_go'])

        # Fit with Ridge regression
        theta, rmse = fit_ridge(Phi, targets, LAMBDA_REG)

        # Sanity checks
        note = ''
        if np.any(np.isnan(theta)) or np.any(np.isinf(theta)):
            note = '⚠ NaN/Inf — falling back to zeros'
            theta = np.zeros(NUM_FEATURES)
        elif np.linalg.norm(theta) > 1e4:
            note = f'⚠ large norm — consider increasing lambda_reg'

        thetas[t] = theta

        print(f"  {t:>5}  {n:>5}  {rmse:>12.4f}  "
              f"{np.linalg.norm(theta):>10.4f}  {note}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(WEIGHTS_DIR, 'thetas.npy')
    np.save(out_path, thetas)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*58}")
    print(f"  Saved : thetas.npy — shape {thetas.shape}")
    print(f"  Lambda: {LAMBDA_REG}")
    print(f"  Location: {WEIGHTS_DIR}")

    feature_names = [
        'bias', 'T1', 'T2', 'T1-T_low', 'T2-T_low',
        'T_high-T1', 'T_high-T2', 'cold_risk_r1', 'cold_risk_r2',
        'low_override_r1', 'low_override_r2', 'H', 'H_high-H',
        'vent_counter', 'time_remaining', 'price_t',
        'time*price', 'price_momentum', 'occ1', 'occ2',
    ]

    print(f"\n  Top 5 features by mean |theta| across stages (excluding bias):")
    mean_abs_theta = np.abs(thetas).mean(axis=0)
    top5 = np.argsort(mean_abs_theta)[::-1][1:6]
    for idx in top5:
        print(f"    [{idx:>2}] {feature_names[idx]:<20}  "
              f"mean |theta| = {mean_abs_theta[idx]:.4f}")


if __name__ == '__main__':
    main()