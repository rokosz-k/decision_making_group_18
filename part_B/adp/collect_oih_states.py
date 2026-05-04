"""
Runs the Optimal-in-Hindsight MILP on all 100 historical days and records,
for each (day, hour) pair:
  - the state dict at that hour (BEFORE the action is taken)
  - the cost-to-go  (sum of remaining hourly costs from that hour to end of day)

Output: part_B/adp/weights/oih_states.csv
        One row per (day, hour). Columns:
            day, hour, cost_to_go,
            T1, T2, H, Occ1, Occ2,
            price_t, price_previous,
            vent_counter, low_override_r1, low_override_r2,
            current_time
"""

import os
import sys
import copy

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))          # part_B/adp/
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))  # project root
PART_A_DIR  = os.path.join(PROJECT_DIR, 'part_A')
DATA_DIR    = os.path.join(PROJECT_DIR, 'data')
WEIGHTS_DIR = os.path.join(SCRIPT_DIR, 'weights')
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# part_A must be on the path so that main.py's own imports resolve correctly
# (main.py does: from SystemCharacteristics import get_fixed_data)
sys.path.insert(0, PART_A_DIR)
sys.path.insert(0, PROJECT_DIR)

from main import solve_day                        # OiH MILP — from part_A/main.py
from SystemCharacteristics import get_fixed_data  # same file main.py uses
from part_B.RestaurantEnv import reset_env, step_env

# ── System parameters ─────────────────────────────────────────────────────────
# Use the same get_fixed_data() as main.py to guarantee parameter consistency.
# We verify the two keys that reset_env needs.
_params = get_fixed_data()

assert 'initial_temperature' in _params, (
    "get_fixed_data() is missing 'initial_temperature' — "
    "check SystemCharacteristics.py key names"
)
assert 'initial_humidity' in _params, (
    "get_fixed_data() is missing 'initial_humidity' — "
    "check SystemCharacteristics.py key names"
)
assert isinstance(_params.get('outdoor_temperature'), (list, tuple)) and \
       len(_params['outdoor_temperature']) >= _params['num_timeslots'], (
    "'outdoor_temperature' must be a list of at least num_timeslots values"
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_csv(path):
    """Load a CSV and return as a list of rows (list of lists)."""
    return pd.read_csv(path, header=0).values.tolist()


# ── Main collection loop ──────────────────────────────────────────────────────

def main():
    price_data = load_csv(os.path.join(DATA_DIR, 'PriceData.csv'))
    occ1_data  = load_csv(os.path.join(DATA_DIR, 'OccupancyRoom1.csv'))
    occ2_data  = load_csv(os.path.join(DATA_DIR, 'OccupancyRoom2.csv'))

    num_days = len(price_data)
    print(f"Loaded {num_days} days. Starting OiH collection...\n")

    # Each entry is one flat row — all state fields unpacked at collection time
    rows    = []
    skipped = 0

    for day in range(num_days):
        occ1  = [float(x) for x in occ1_data[day]]
        occ2  = [float(x) for x in occ2_data[day]]
        price = [float(x) for x in price_data[day]]

        print(f"Day {day + 1:3d}/{num_days} — solving MILP ...", end=" ", flush=True)

        res = solve_day(day, occ1, occ2, price, verbose=False)

        if res is None:
            print("SKIPPED (infeasible or no solution)")
            skipped += 1
            continue

        # Build data dict that step_env accepts.
        # Deep-copy _params so the shared dict is never mutated across days.
        day_data          = copy.deepcopy(_params)
        day_data['price'] = price   # step_env reads data['price'][t]

        occupancy = {"Room1": occ1, "Room2": occ2}

        # ── Replay optimal actions through the environment ────────────────────
        # We need the pre-action state at each hour (what the policy sees)
        # and the cost incurred at each hour.
        state        = reset_env(day_data, occupancy)
        hourly_costs = []
        pre_states   = []   # state BEFORE action at hour t

        done = False
        while not done:
            t = state['current_time']

            # Record state BEFORE action (this is what a policy would observe)
            pre_states.append(copy.deepcopy(state))

            # Optimal action from the MILP solution.
            # res['h_r1'][t] is the heater power for room 1 at hour t,
            # matching the MILP variable p[0,t].X (0-indexed rooms).
            action = {
                "HeatPowerRoom1": float(res['h_r1'][t]),
                "HeatPowerRoom2": float(res['h_r2'][t]),
                "VentilationON":  int(round(res['v'][t])),
            }

            state, cost, done = step_env(state, action, day_data, occupancy)
            hourly_costs.append(float(cost))

        # Sanity check: should have exactly num_timeslots records
        T = _params['num_timeslots']
        if len(pre_states) != T or len(hourly_costs) != T:
            print(f"  WARNING: Day {day} — expected {T} steps, "
                  f"got {len(pre_states)} states and {len(hourly_costs)} costs. Skipping.")
            skipped += 1
            continue

        # ── Compute cost-to-go and flatten into rows ──────────────────────────
        # cost_to_go[t] = sum of hourly costs from hour t to end of day (inclusive)
        # cost_to_go[T-1] = hourly_costs[T-1]  (last hour: just that hour's cost)
        # cost_to_go[0]   = sum of all hourly costs
        for t_idx in range(T):
            s = pre_states[t_idx]
            rows.append({
                'day':              day,
                'hour':             t_idx,
                'cost_to_go':       float(sum(hourly_costs[t_idx:])),
                # state fields — unpacked so the CSV is human-readable
                'T1':               s['T1'],
                'T2':               s['T2'],
                'H':                s['H'],
                'Occ1':             s['Occ1'],
                'Occ2':             s['Occ2'],
                'price_t':          s['price_t'],
                'price_previous':   s['price_previous'],
                'vent_counter':     s['vent_counter'],
                'low_override_r1':  s['low_override_r1'],
                'low_override_r2':  s['low_override_r2'],
                'current_time':     s['current_time'],
            })

        day_total = sum(hourly_costs)
        print(f"cost = {day_total:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(WEIGHTS_DIR, 'oih_states.csv')
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    # ── Summary ───────────────────────────────────────────────────────────────
    solved_days = num_days - skipped
    print(f"\n{'='*55}")
    print(f"  Days solved     : {solved_days} / {num_days}  ({skipped} skipped)")
    print(f"  Total records   : {len(rows)}")
    print(f"  Output          : {out_path}")
    print(f"{'='*55}")

    print(f"\n  Mean cost-to-go by stage (should decrease toward t=9):")
    ctg_by_t = {}
    for row in rows:
        ctg_by_t.setdefault(row['hour'], []).append(row['cost_to_go'])

    for t in sorted(ctg_by_t):
        vals = ctg_by_t[t]
        print(f"    t={t}:  mean={np.mean(vals):.4f}  "
              f"min={np.min(vals):.4f}  max={np.max(vals):.4f}")

    # Extra check: warn if cost-to-go is not monotonically decreasing on average
    means = [np.mean(ctg_by_t[t]) for t in sorted(ctg_by_t)]
    if any(means[i] < means[i + 1] for i in range(len(means) - 1)):
        print("\n  WARNING: mean cost-to-go is not monotonically decreasing.")
        print("  This may indicate a time-indexing mismatch between the MILP")
        print("  and step_env. Check that res['h_r1'][t] is the action at hour t.")


if __name__ == '__main__':
    main()