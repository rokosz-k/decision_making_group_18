import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from data.v2_SystemCharacteristics import get_fixed_data

# ── Load system parameters once at import time ────────────────────────────────
_params = get_fixed_data()
T_LOW  = _params['temp_min_comfort_threshold']
T_HIGH = _params['temp_max_comfort_threshold']
H_HIGH = _params['humidity_threshold']
T_END  = _params['num_timeslots']               # 10 hours

# ── Design note ───────────────────────────────────────────────────────────────
# Following the course's Approximate Backward Induction approach, we maintain
# one theta vector PER TIMESTEP (theta_0 ... theta_{T-1}). estimate_value()
# always receives the theta for the NEXT stage (t+1), not the current one.
# This is because at stage t we use V_{t+1}(s') as the future cost estimate.
#
# Feature vector length: 20
# [0]  bias
# [1]  T1
# [2]  T2
# [3]  T1 - T_low              signed distance to cold threshold, room 1
# [4]  T2 - T_low              signed distance to cold threshold, room 2
# [5]  T_high - T1             signed distance to hot threshold, room 1
# [6]  T_high - T2             signed distance to hot threshold, room 2
# [7]  max(0, T_low - T1)^2   nonlinear cold risk, room 1
# [8]  max(0, T_low - T2)^2   nonlinear cold risk, room 2
# [9]  low_override_r1         cold overrule active room 1 (0 or 1)
# [10] low_override_r2         cold overrule active room 2 (0 or 1)
# [11] H                       current humidity
# [12] H_high - H              signed distance to humidity threshold
# [13] vent_counter            hours of forced ventilation remaining (0-3)
# [14] T_end - t               hours remaining in the day
# [15] price_t                 current electricity price
# [16] (T_end - t) * price_t  time-remaining x price interaction
# [17] price_t - price_prev    price momentum
# [18] occ1                    current occupancy room 1
# [19] occ2                    current occupancy room 2
#
# Note: features [14] and [16] (time-remaining) are kept even though we use
# separate thetas per stage. They add robustness and cost nothing.
NUM_FEATURES = 20


def compute_features(state: dict) -> np.ndarray:
    """
    Map a state dict (from RestaurantEnv) to a raw feature vector phi
    of length NUM_FEATURES.

    Parameters
    ----------
    state : dict
        Keys expected: T1, T2, H, Occ1, Occ2, price_t, price_previous,
                       vent_counter, low_override_r1, low_override_r2,
                       current_time

    Returns
    -------
    phi : np.ndarray, shape (NUM_FEATURES,), dtype float64
    """
    T1           = state['T1']
    T2           = state['T2']
    H            = state['H']
    occ1         = state['Occ1']
    occ2         = state['Occ2']
    price        = state['price_t']
    price_prev   = state['price_previous']
    vent_counter = state['vent_counter']       # raw int: 0, 1, 2
    low_r1       = state['low_override_r1']    # 0 or 1
    low_r2       = state['low_override_r2']    # 0 or 1
    t            = state['current_time']

    time_remaining = float(T_END - t)

    # Nonlinear cold risk: zero when temp is safely above T_LOW,
    # grows quadratically as temp drops below the threshold
    cold_risk_r1 = max(0.0, T_LOW - T1) ** 2
    cold_risk_r2 = max(0.0, T_LOW - T2) ** 2

    phi = np.array([
        1.0,                            # [0]  bias
        T1,                             # [1]  room 1 temperature
        T2,                             # [2]  room 2 temperature
        T1 - T_LOW,                     # [3]  signed distance cold threshold r1
        T2 - T_LOW,                     # [4]  signed distance cold threshold r2
        T_HIGH - T1,                    # [5]  signed distance hot threshold r1
        T_HIGH - T2,                    # [6]  signed distance hot threshold r2
        cold_risk_r1,                   # [7]  nonlinear cold risk r1
        cold_risk_r2,                   # [8]  nonlinear cold risk r2
        float(low_r1),                  # [9]  cold overrule flag r1
        float(low_r2),                  # [10] cold overrule flag r2
        H,                              # [11] humidity
        H_HIGH - H,                     # [12] signed distance humidity threshold
        float(vent_counter),            # [13] hours of forced ventilation remaining
        time_remaining,                 # [14] hours remaining in the day
        price,                          # [15] current electricity price
        time_remaining * price,         # [16] time-remaining x price interaction
        price - price_prev,             # [17] price momentum
        occ1,                           # [18] occupancy room 1
        occ2,                           # [19] occupancy room 2
    ], dtype=float)

    assert len(phi) == NUM_FEATURES, (
        f"Feature length mismatch: expected {NUM_FEATURES}, got {len(phi)}"
    )

    return phi


def normalize_features(
    phi: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Standardize a raw feature vector using pre-computed mean and std.

    Safe against constant features (std ~ 0): those dimensions are left
    unchanged rather than producing NaN or Inf.

    Parameters
    ----------
    phi   : np.ndarray (NUM_FEATURES,) — raw features from compute_features
    mu    : np.ndarray (NUM_FEATURES,) — per-feature mean from training data
    sigma : np.ndarray (NUM_FEATURES,) — per-feature std  from training data

    Returns
    -------
    phi_norm : np.ndarray (NUM_FEATURES,)
    """
    safe_sigma = np.where(sigma < 1e-8, 1.0, sigma)
    return (phi - mu) / safe_sigma


def estimate_value(
    state: dict,
    theta: np.ndarray,
    mu: np.ndarray = None,
    sigma: np.ndarray = None,
) -> float:
    """
    Evaluate the approximate value function V_{t+1}(s') at a given state.

    IMPORTANT — which theta to pass:
        At stage t, pass theta_{t+1} (the weights fitted for the next stage).
        This gives V_{t+1}(s'), used when computing targets at stage t.
        In practice: theta = all_thetas[t + 1]

    Returns 0.0 at the final stage (t >= T_END - 1): no future cost remains.

    Parameters
    ----------
    state : dict         — next state s' returned by step_env / transition
    theta : np.ndarray   — weight vector shape (NUM_FEATURES,) for stage t+1
    mu    : np.ndarray or None — feature means  (pass both or neither)
    sigma : np.ndarray or None — feature stdevs (pass both or neither)

    Returns
    -------
    value : float, clipped to >= 0.0
    """
    # Terminal condition: no future cost beyond the last hour
    if state['current_time'] >= T_END - 1:
        return 0.0

    phi = compute_features(state)

    if mu is not None and sigma is not None:
        phi = normalize_features(phi, mu, sigma)

    value = float(np.dot(theta, phi))

    # Costs are always non-negative
    return max(0.0, value)


def load_thetas(weights_dir: str):
    """
    Load all per-stage weight vectors saved by train_adp.py.

    File layout expected in weights_dir:
        thetas.npy  — shape (T_END, NUM_FEATURES), one row per stage t
        mu.npy      — shape (NUM_FEATURES,)
        sigma.npy   — shape (NUM_FEATURES,)

    Usage in the policy:
        thetas, mu, sigma = load_thetas(weights_dir)
        future = estimate_value(next_state, thetas[t + 1], mu, sigma)

    Parameters
    ----------
    weights_dir : str — path to folder containing the .npy files

    Returns
    -------
    thetas : np.ndarray (T_END, NUM_FEATURES)
    mu     : np.ndarray (NUM_FEATURES,)
    sigma  : np.ndarray (NUM_FEATURES,)
    """
    thetas = np.load(os.path.join(weights_dir, 'thetas.npy'))
    mu     = np.load(os.path.join(weights_dir, 'mu.npy'))
    sigma  = np.load(os.path.join(weights_dir, 'sigma.npy'))

    assert thetas.shape == (T_END, NUM_FEATURES), (
        f"thetas.npy shape mismatch: expected ({T_END}, {NUM_FEATURES}), "
        f"got {thetas.shape}"
    )
    assert mu.shape    == (NUM_FEATURES,), "mu.npy shape mismatch"
    assert sigma.shape == (NUM_FEATURES,), "sigma.npy shape mismatch"

    return thetas, mu, sigma