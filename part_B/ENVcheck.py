# ------------------------------
# Imports
# ------------------------------
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from part_A.SystemCharacteristics import get_fixed_data
from RestaurantEnv import reset_env, step_env   


# ------------------------------
# CSV loader
# ------------------------------
def load_first_day(path):
    return pd.read_csv(path).iloc[0].tolist()


# ------------------------------
# File paths
# ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
price_path = os.path.join(BASE_DIR, "data", "PriceData.csv")
occ1_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom1.csv")
occ2_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom2.csv")


# ------------------------------
# Load data
# ------------------------------
data = get_fixed_data()

price = load_first_day(price_path)
occ1  = load_first_day(occ1_path)
occ2  = load_first_day(occ2_path)

data['price'] = price

occupancy = {
    "Room1": occ1,
    "Room2": occ2
}


# ------------------------------
# Run simulation
# ------------------------------
state = reset_env(data, occupancy)

print("Initial state:", state)

total_cost = 0
done = False

while not done:

    action = {
        "HeatPowerRoom1": 3.0,
        "HeatPowerRoom2": 3.0,
        "VentilationON": 0
    }

    state, cost, done = step_env(state, action, data, occupancy)
    total_cost += cost

    print(f"t={state['current_time']}, T1={state['T1']:.2f}, T2={state['T2']:.2f}, H={state['H']:.2f}, cost={cost:.2f}")

print("Total cost:", total_cost)