import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from part_A.SystemCharacteristics import get_fixed_data

from part_B.policies.SP_policy_18 import select_action
from part_B.RestaurantEnv import step_env, reset_env

# CSV loader
def load_all_days(path):
    return pd.read_csv(path).values.tolist()

# File paths

price_path = os.path.join(BASE_DIR, "data", "PriceData.csv")
occ1_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom1.csv")
occ2_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom2.csv")


base_data = get_fixed_data()

price_data = load_all_days(price_path)
occ1_data  = load_all_days(occ1_path)
occ2_data  = load_all_days(occ2_path)
num_days = 100
total_costs = []

for day in range(num_days):
    print(f"\nDay {day+1}")
    
    data = base_data.copy()
    
    data['price'] = price_data[day]
    
    occupancy = {
        "Room1": occ1_data[day],
        "Room2": occ2_data[day]
    }
    
    state = reset_env(data, occupancy)
    
    done = False
    total_cost = 0

    while not done:
        action = select_action(state)
        state, cost, done = step_env(state, action, data, occupancy)
        total_cost += cost

    total_costs.append(total_cost)
    print('\n', '='*50)
    print("Total cost:", total_cost)

print("\n", "="*30, "SUMMARY", "="*30)
print("Average cost:", sum(total_costs) / len(total_costs))
print("Min cost:", min(total_costs))
print("Max cost:", max(total_costs))