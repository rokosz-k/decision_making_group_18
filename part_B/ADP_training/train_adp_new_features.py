"""
train_adp.py

Trains a linear value function approximation for each hour t.
Reads samples from a CSV (produced by collect_samples.py) and fits:

    V̂(x_t; η_t) = η_t^T · x_t

where x_t = [T1, T2, H, Occ1, Occ2, price_t, price_previous,
             vent_counter, low_override_r1, low_override_r2, 1 (bias)]

Features are standardised (zero mean, unit variance) before fitting so
that Ridge regularisation penalises all weights fairly regardless of
feature scale. The trained weights are then converted back to raw feature
space so the policy can use raw state values directly — no scaling needed
at decision time.

Conversion:
    V̂ = eta_scaled @ X_scaled
      = eta_scaled @ (X_raw - mean) / std
      = (eta_scaled / std) @ X_raw  -  sum(eta_scaled * mean / std)  +  bias
      = eta_raw @ X_raw_with_bias

Output: eta.pkl  —  dict {t: np.array of shape (11,)} for t = 0..9
        Weights are in RAW feature space — use directly with unscaled features.

Usage:
    python train_adp.py                          # uses samples_dummy.csv, lambda=10
    python train_adp.py samples_adp_v1.csv       # custom samples file
    python train_adp.py samples_dummy.csv 5.0    # custom lambda
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# Feature columns (order matters — must match policy)
# ─────────────────────────────────────────────
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


def build_features(df_hour):
    """
    Build raw feature matrix X (N x 11) and target vector y (N,).
    Last column of X is the bias term (all ones).
    """
    X = df_hour[FEATURE_COLS].values.astype(float)
    X = np.hstack([X, np.ones((len(X), 1))])
    y = df_hour["cost_to_go"].values.astype(float)
    return X, y


def train(samples_path="samples_dummy.csv", output_path="eta.pkl", ridge_lambda=10.0):
    """
    Fit eta_t for each hour t via Ridge regression on scaled features,
    then convert weights back to raw feature space.

    Scaling ensures Ridge penalises all weights fairly regardless of
    the raw scale of each feature (e.g. price 0-12 vs temperature 15-25).

    The stored eta works directly with unscaled (raw) state features —
    the policy does not need to know about scaling.

    Args:
        samples_path  : path to CSV produced by collect_samples.py
        output_path   : where to save eta.pkl
        ridge_lambda  : regularisation strength
    Returns:
        eta dict {t: np.array of shape (11,)} in raw feature space
    """
    print(f"Loading samples from : {samples_path}")
    print(f"Ridge lambda         : {ridge_lambda}")
    df = pd.read_csv(samples_path)

    print(f"Hours found  : {sorted(df['hour'].unique())}")
    print(f"Days found   : {df['day'].nunique()}")
    print(f"Total rows   : {len(df)}")
    print()

    eta = {}
    n_raw = len(FEATURE_COLS)   # number of features excluding bias

    print(f"{'Hour':>4}  {'N':>6}  {'R2 scaled':>10}  {'R2 raw':>8}  {'RMSE':>10}")
    print("-" * 50)

    for t in sorted(df["hour"].unique()):
        df_t = df[df["hour"] == t]
        X_raw, y = build_features(df_t)       # X_raw includes bias column

        # ── Scale features (not the bias column) ─────────────────────────
        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X_raw[:, :n_raw])   # scale raw features
        X_sc_b  = np.hstack([X_sc, np.ones((len(X_sc), 1))])  # add bias back

        n_total = X_sc_b.shape[1]   # n_raw + 1

        # ── Ridge on scaled features ──────────────────────────────────────
        # eta = (X^T X + lambda*I)^{-1} X^T y
        # No penalty on bias (last entry)
        Lambda = ridge_lambda * np.eye(n_total)
        Lambda[-1, -1] = 0.0

        eta_scaled = np.linalg.solve(X_sc_b.T @ X_sc_b + Lambda,
                                     X_sc_b.T @ y)

        # ── Convert back to raw feature space ────────────────────────────
        # V̂ = eta_scaled @ X_scaled
        #    = eta_scaled[:n] @ (X_raw - mean)/std  +  eta_scaled[n]
        #    = (eta_scaled[:n]/std) @ X_raw
        #      - sum(eta_scaled[:n] * mean / std)
        #      + eta_scaled[n]
        #
        # So: eta_raw[:n] = eta_scaled[:n] / std
        #     eta_raw[n]  = eta_scaled[n] - sum(eta_scaled[:n]*mean/std)
        eta_raw          = np.zeros(n_total)
        eta_raw[:n_raw]  = eta_scaled[:n_raw] / scaler.scale_
        eta_raw[n_raw]   = (eta_scaled[n_raw]
                            - np.sum(eta_scaled[:n_raw]
                                     * scaler.mean_
                                     / scaler.scale_))

        eta[t] = eta_raw

        # ── Diagnostics ───────────────────────────────────────────────────
        # R² on scaled fit (what Ridge actually optimised)
        y_pred_sc = X_sc_b @ eta_scaled
        ss_res_sc = np.sum((y - y_pred_sc) ** 2)
        ss_tot    = np.sum((y - y.mean()) ** 2)
        r2_sc     = 1.0 - ss_res_sc / ss_tot if ss_tot > 0 else float("nan")

        # R² using raw weights on raw features (verify conversion is correct)
        y_pred_raw = X_raw @ eta_raw
        ss_res_raw = np.sum((y - y_pred_raw) ** 2)
        r2_raw     = 1.0 - ss_res_raw / ss_tot if ss_tot > 0 else float("nan")

        rmse = np.sqrt(ss_res_sc / len(y))

        print(f"  t={t}  {len(df_t):>6}  {r2_sc:>10.4f}  {r2_raw:>8.4f}  {rmse:>10.4f}")

    print()

    # ── Save ──────────────────────────────────────────────────────────────
    with open(output_path, "wb") as f:
        pickle.dump(eta, f)
    print(f"Saved ETA (raw space) to: {output_path}")

    # ── Print weights for reference ───────────────────────────────────────
    print("\nWeights per hour (raw feature space — use directly with unscaled features):")
    print("Feature order: " + ", ".join(FEATURE_COLS) + ", bias")
    for t, eta_t in eta.items():
        print(f"  t={t}: {np.round(eta_t, 4).tolist()}")

    return eta


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    samples_path = (sys.argv[1] if len(sys.argv) > 1
                    else os.path.join(base_dir, "samples_mixed_v2.csv"))
    ridge_lambda = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    output_path  = os.path.join(base_dir, "eta_new_features_v2.pkl")

    train(samples_path, output_path, ridge_lambda)