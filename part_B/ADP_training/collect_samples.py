"""
collect_samples.py  (v2)

Runs a policy over all available days and saves, for each (day, hour):
  - all state variables
  - cost_to_go = sum of costs from that hour to end of day

Output: samples.csv

Usage:
    python collect_samples.py

To swap the policy, change the import at the bottom of this file.
"""

import os
import sys
import copy
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from data.v2_SystemCharacteristics import get_fixed_data            # v2
from part_B.RestaurantEnv import step_env, reset_env                # updated RestaurantEnv


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_all_days(path):
    return pd.read_csv(path).values.tolist()


DUMMY_ACTION = {"HeatPowerRoom1": 0, "HeatPowerRoom2": 0, "VentilationON": 0}

def dummy_policy(state):
    return DUMMY_ACTION


def is_feasible(action):
    try:
        p1 = action["HeatPowerRoom1"]
        p2 = action["HeatPowerRoom2"]
        v  = action["VentilationON"]
        return (0 <= p1 <= 3 and 0 <= p2 <= 3 and v in (0, 1))
    except (KeyError, TypeError):
        return False


def get_action(policy_fn, state, timeout=15):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(policy_fn, state)
        try:
            action = future.result(timeout=timeout)
        except FuturesTimeout:
            print(f"    [TIMEOUT] — using dummy action")
            return DUMMY_ACTION
        except Exception as e:
            print(f"    [ERROR: {e}] — using dummy action")
            return DUMMY_ACTION
    if not is_feasible(action):
        print(f"    [INFEASIBLE: {action}] — using dummy action")
        return DUMMY_ACTION
    return action


# ─────────────────────────────────────────────
# Sample collection
# ─────────────────────────────────────────────

def collect_samples(policy_fn, output_path="samples.csv"):
    """
    Run policy_fn over all 100 days.
    At each hour, record the full state and the cost_to_go.
    Saves results to output_path as a CSV.
    """

    # ── Data loading ──────────────────────────────────────────────────────
    price_path = os.path.join(BASE_DIR, "data", "v2_PriceData.csv")  # v2: 11 cols
    occ1_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom1.csv")
    occ2_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom2.csv")

    base_data  = get_fixed_data()
    price_data = load_all_days(price_path)   # 100 rows × 11 cols
    occ1_data  = load_all_days(occ1_path)
    occ2_data  = load_all_days(occ2_path)

    num_days = len(price_data)
    all_rows = []

    for day in range(num_days):
        print(f"Day {day + 1}/{num_days}")

        data = copy.deepcopy(base_data)

        # v2_PriceData layout:
        #   col 0      → price at t = -1  (previous price for the initial state)
        #   cols 1-10  → hourly prices for t = 0 .. 9
        data["price_previous"] = price_data[day][0]   # feeds into reset_env
        data["price"]          = price_data[day][1:]  # 10 hourly prices for step_env

        occupancy = {
            "Room1": occ1_data[day],
            "Room2": occ2_data[day],
        }

        state = reset_env(data, occupancy)   # uses updated v2-compatible reset_env
        done  = False

        # ── Collect (state, cost) pairs for one day ──
        day_states = []
        day_costs  = []

        while not done:
            day_states.append(dict(state))

            action            = get_action(policy_fn, state)
            state, cost, done = step_env(state, action, data, occupancy)

            day_costs.append(cost)

        # ── Compute cost_to_go[t] = sum of costs from t onwards ──
        T = len(day_costs)
        cost_to_go = [0.0] * T
        running = 0.0
        for t in reversed(range(T)):
            running       += day_costs[t]
            cost_to_go[t]  = running

        # ── Build rows ──
        for t, (s, ctg) in enumerate(zip(day_states, cost_to_go)):
            row = {
                "day":              day,
                "hour":             t,
                "cost_to_go":       ctg,
                "T1":               s["T1"],
                "T2":               s["T2"],
                "H":                s["H"],
                "Occ1":             s["Occ1"],
                "Occ2":             s["Occ2"],
                "price_t":          s["price_t"],
                "price_previous":   s["price_previous"],
                "vent_counter":     s["vent_counter"],
                "low_override_r1":  s["low_override_r1"],
                "low_override_r2":  s["low_override_r2"],
                "current_time":     s["current_time"],
            }
            all_rows.append(row)

        total_cost = sum(day_costs)
        print(f"  Total cost: {total_cost:.4f}   cost_to_go[0]: {cost_to_go[0]:.4f}")

    # ── Save ──
    df = pd.DataFrame(all_rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")
    return df


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
# To use a different policy, replace the import below, e.g.:
#   from part_B.policies.ADP_policy_18 import select_action as policy_fn
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from part_B.policies.ADP_policy_new_features_18 import select_action as policy_fn
    # policy_fn = dummy_policy   # if you want to use dummy policy
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples/samples_fvi_v4.csv")

    collect_samples(policy_fn, output_path)