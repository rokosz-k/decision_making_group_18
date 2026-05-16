import os
import sys
import copy
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from part_A.SystemCharacteristics import get_fixed_data
from part_B.policies.SP_policy_18 import select_action
from part_B.RestaurantEnv import step_env, reset_env
from part_B.dummy_policy import dummy_action, DUMMY_ACTION


def load_all_days(path):
    return pd.read_csv(path).values.tolist()


def is_feasible(action):
    try:
        p1 = action["HeatPowerRoom1"]
        p2 = action["HeatPowerRoom2"]
        v  = action["VentilationON"]
        return (
            0 <= p1 <= 3 and
            0 <= p2 <= 3 and
            v in (0, 1)
        )
    except (KeyError, TypeError):
        return False


def get_action(policy_fn, state, timeout=15):
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(policy_fn, state)
        try:
            action = future.result(timeout=timeout)
        except FuturesTimeout:
            print(f"  [TIMEOUT] — using dummy action")
            return DUMMY_ACTION
        except Exception as e:
            print(f"  [ERROR: {e}] — using dummy action")
            return DUMMY_ACTION

    if not is_feasible(action):
        print(f"  [INFEASIBLE: {action}] — using dummy action")
        return DUMMY_ACTION

    return action


# Data loading

price_path = os.path.join(BASE_DIR, "data", "PriceData.csv")
occ1_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom1.csv")
occ2_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom2.csv")

base_data  = get_fixed_data()
price_data = load_all_days(price_path)
occ1_data  = load_all_days(occ1_path)
occ2_data  = load_all_days(occ2_path)


num_days    = 100
total_costs = []

for day in range(num_days):
    print(f"\nDay {day + 1}")

    data = copy.deepcopy(base_data)
    data['price'] = price_data[day]

    occupancy = {
        "Room1": occ1_data[day],
        "Room2": occ2_data[day]
    }

    state      = reset_env(data, occupancy)
    done       = False
    total_cost = 0.0

    while not done:
        action            = get_action(select_action, state)
        state, cost, done = step_env(state, action, data, occupancy)
        total_cost       += cost

    total_costs.append(total_cost)
    print(f"\nTotal cost: {total_cost:.4f}")

print("\n", "=" * 30, "SUMMARY", "=" * 30)
print(f"Average cost : {sum(total_costs) / len(total_costs):.4f}")
print(f"Min cost     : {min(total_costs):.4f}")
print(f"Max cost     : {max(total_costs):.4f}")