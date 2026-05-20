"""
debug_adp.py — run this to diagnose why ADP cost is high

Prints the state at each step, the action chosen, and the cost,
for a single day. Lets you verify prices, temperatures, and actions
are all sensible.
"""

import os, sys, copy
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# ── Import whichever SystemCharacteristics your running_script uses ──
from data.v2_SystemCharacteristics import get_fixed_data   # adjust if using v2

from part_B.RestaurantEnv import step_env, reset_env
from part_B.policies.ADP_policy_9_features_18 import select_action

def load_all_days(path):
    return pd.read_csv(path).values.tolist()

# ── Load data ────────────────────────────────────────────────────────
price_path = os.path.join(BASE_DIR, "data", "PriceData.csv")   # adjust if v2
occ1_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom1.csv")
occ2_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom2.csv")

base_data  = get_fixed_data()
price_data = load_all_days(price_path)
occ1_data  = load_all_days(occ1_path)
occ2_data  = load_all_days(occ2_path)

print(f"Price row 0 length : {len(price_data[0])}  (expect 10 for old, 11 for v2)")
print(f"Occ1 row 0 length  : {len(occ1_data[0])}")
print()

# ── Run day 0 ────────────────────────────────────────────────────────
DAY = 0
data = copy.deepcopy(base_data)

# ---- adjust this block to match your running_script exactly ----
data['price'] = price_data[DAY]          # old: 10 values
# data['price'] = price_data[DAY][1:]    # v2:  10 values (skip col 0)
# ----------------------------------------------------------------

occupancy = {"Room1": occ1_data[DAY], "Room2": occ2_data[DAY]}
state = reset_env(data, occupancy)

print(f"Initial state:")
for k, v in state.items():
    print(f"  {k:20s} = {v}")
print(f"\n  data['price'] (first 3) = {data['price'][:3]}")
print()

total_cost = 0.0
while True:
    t = state["current_time"]
    action = select_action(state)
    state, cost, done = step_env(state, action, data, occupancy)
    total_cost += cost

    print(f"t={t}  P1={action['HeatPowerRoom1']:.2f}  P2={action['HeatPowerRoom2']:.2f}"
          f"  V={action['VentilationON']}"
          f"  price={data['price'][t]:.3f}"
          f"  cost={cost:.3f}"
          f"  T1={state['T1']:.2f}  T2={state['T2']:.2f}"
          f"  H={state['H']:.2f}"
          f"  lr1={state['low_override_r1']}  lr2={state['low_override_r2']}")

    if done:
        break

print(f"\nTotal cost day {DAY}: {total_cost:.4f}")
