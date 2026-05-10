"""
train_adp.py

Trains a linear value function approximation for each hour t.
Reads samples from a CSV (produced by collect_samples.py) and fits:

    V̂(x_t; η_t) = η_t^T · x_t

where x_t = [T1, T2, H, Occ1, Occ2, price_t, price_previous,
             vent_counter, low_override_r1, low_override_r2, 1 (bias)]

and the target is cost_to_go (remaining cost under the sampled policy).

Uses Ridge regression to avoid unstable weights from multicollinearity
(e.g. features that are constant across all samples at a given hour).

Output: eta.pkl  —  dict {t: np.array of shape (11,)} for t = 0..9

Usage:
    python train_adp.py                          # uses samples_dummy.csv, lambda=1.0
    python train_adp.py samples_adp_v1.csv       # custom samples file
    python train_adp.py samples_dummy.csv 10.0   # custom lambda
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
# Feature columns (order matters — must match policy)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "T1",
    "T2",
    "H",
    "Occ1",
    "Occ2",
    "price_t",
    "price_previous",
    "vent_counter",
    "low_override_r1",
    "low_override_r2",
]


def build_features(df_hour):
    """
    Build feature matrix X (N x 11) and target vector y (N,) for one hour.
    Last column of X is the bias term (all ones).
    """
    X = df_hour[FEATURE_COLS].values.astype(float)
    X = np.hstack([X, np.ones((len(X), 1))])   # append bias
    y = df_hour["cost_to_go"].values.astype(float)
    return X, y


def train(samples_path="samples_dummy.csv", output_path="eta.pkl", ridge_lambda=10.0):
    """
    Fit eta_t for each hour t via Ridge regression and save to output_path.

    Ridge regression solves:
        eta = (X^T X + lambda*I)^{-1} X^T y

    This prevents multicollinearity from blowing up weights when features
    are constant or correlated across samples (e.g. all low_override=0 at t=0).

    Args:
        samples_path  : path to CSV produced by collect_samples.py
        output_path   : where to save eta.pkl
        ridge_lambda  : regularisation strength - larger means more shrinkage.
                        Try 0.1, 1.0, 10.0 and compare RMSE.
    Returns:
        eta dict {t: np.array of shape (11,)}
    """
    print(f"Loading samples from : {samples_path}")
    print(f"Ridge lambda         : {ridge_lambda}")
    df = pd.read_csv(samples_path)

    print(f"Hours found  : {sorted(df['hour'].unique())}")
    print(f"Days found   : {df['day'].nunique()}")
    print(f"Total rows   : {len(df)}")
    print()

    eta = {}

    print(f"{'Hour':>4}  {'N samples':>9}  {'R2':>8}  {'RMSE':>10}")
    print("-" * 40)

    for t in sorted(df["hour"].unique()):
        df_t = df[df["hour"] == t]
        X, y = build_features(df_t)

        n_features = X.shape[1]

        # Ridge: eta = (X^T X + lambda*I)^{-1} X^T y
        # No penalty on the bias term (last entry)
        Lambda = ridge_lambda * np.eye(n_features)
        Lambda[-1, -1] = 0.0

        eta_t = np.linalg.solve(X.T @ X + Lambda, X.T @ y)
        eta[t] = eta_t

        # Diagnostics
        y_pred = X @ eta_t
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2   = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rmse = np.sqrt(ss_res / len(y))

        print(f"  t={t}  {len(df_t):>9}  {r2:>8.4f}  {rmse:>10.4f}")

    print()

    # ── Save ──
    with open(output_path, "wb") as f:
        pickle.dump(eta, f)
    print(f"Saved ETA to: {output_path}")

    # ── Print weights for reference ──
    print("\nWeights per hour (feature order: " + ", ".join(FEATURE_COLS) + ", bias):")
    for t, eta_t in eta.items():
        print(f"  t={t}: {np.round(eta_t, 4).tolist()}")

    return eta


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    samples_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(base_dir, "samples_dummy.csv")
    ridge_lambda = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    output_path  = os.path.join(base_dir, "eta.pkl")

    train(samples_path, output_path, ridge_lambda)