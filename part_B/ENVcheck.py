# ------------------------------
# Imports
# ------------------------------
import pandas as pd
from SystemCharacteristics import get_fixed_data
from RestaurantEnv import reset_env, step_env   


# ------------------------------
# CSV loader
# ------------------------------
def load_first_day(path):
    return pd.read_csv(path).iloc[0].tolist()


# ------------------------------
# File paths
# ------------------------------
price_path = '/Users/jaredbutler/Desktop/Masters/Winter 2026/2) Decision Making/Assignment/Part B/PriceData.csv'
occ1_path  = '/Users/jaredbutler/Desktop/Masters/Winter 2026/2) Decision Making/Assignment/Part B/OccupancyRoom1.csv'
occ2_path  = '/Users/jaredbutler/Desktop/Masters/Winter 2026/2) Decision Making/Assignment/Part B/OccupancyRoom2.csv'


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