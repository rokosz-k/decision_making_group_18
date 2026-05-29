import os
import sys
import copy
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from data.v2_SystemCharacteristics import get_fixed_data            # v2
from part_B.RestaurantEnv import step_env, reset_env                # updated RestaurantEnv
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


def run_script(select_action):
    # ── Data loading ───────────────────────────────────────────────────────────
    price_path = os.path.join(BASE_DIR, "data", "v2_PriceData.csv")    # v2: 11 cols
    occ1_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom1.csv")
    occ2_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom2.csv")

    base_data  = get_fixed_data()
    price_data = load_all_days(price_path)   # 100 rows × 11 cols
    occ1_data  = load_all_days(occ1_path)
    occ2_data  = load_all_days(occ2_path)

    num_days    = 100
    total_costs = []

    # price_days = []
    # for day in range(num_days):
    #     price_day = sum(price_data[day][1:])/10
    #     price_days.append(price_day)

    # import seaborn as sns
    # sns.histplot(price_days, bins=20, kde=True, color='skyblue', edgecolor='black')
    # plt.xlabel('Price [DKK]')
    # plt.ylabel('Frequency')
    # plt.title('Average Daily Price')
    # plt.show()

    for day in range(num_days):
        print(f"\nDay {day + 1}")

        data = copy.deepcopy(base_data)

        # v2_PriceData layout:
        #   col 0      → price at t = -1  (previous price for the initial state)
        #   cols 1-10  → hourly prices for t = 0 .. 9
        data["price_previous"] = price_data[day][0]   # feeds into reset_env
        data["price"]          = price_data[day][1:]  # 10 hourly prices for step_env

        occupancy = {
            "Room1": occ1_data[day],
            "Room2": occ2_data[day]
        }

        state      = reset_env(data, occupancy)
        done       = False
        total_cost = 0.0

        while not done:
            action            = get_action(select_action, state)
            #print(action)
            state, cost, done = step_env(state, action, data, occupancy)
            total_cost       += cost

        total_costs.append(total_cost)
        print(f"\nTotal cost: {total_cost:.4f}")

    print("\n", "=" * 30, "SUMMARY", "=" * 30)
    print(f"Average cost : {sum(total_costs) / len(total_costs):.4f}")
    print(f"Min cost     : {min(total_costs):.4f}")
    print(f"Max cost     : {max(total_costs):.4f}")

    # Plot histogram
    # plt.figure(figsize=(10, 5))
    # plt.hist(total_costs, bins=20, edgecolor='black')
    # plt.axvline(sum(total_costs)/len(total_costs), color='red', linestyle='--', 
    #             linewidth=1.5, label=f'Mean: {sum(total_costs)/len(total_costs):.4f}')
    # plt.xlabel('Daily Cost')
    # plt.ylabel('Frequency')
    # plt.title('Distribution of Daily Costs over 100 Days - Dummy')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig('daily_cost_histogram_deterministic.png', dpi=150)
    # plt.show()

    return sum(total_costs) / len(total_costs)

if __name__ == "__main__":
    # from part_B.policies.SP_policy_18 import select_action
    # from part_B.policies.ADP_policy_9_features_18 import select_action
    from part_B.policies.hybrid_policy_18 import select_action

    run_script(select_action)