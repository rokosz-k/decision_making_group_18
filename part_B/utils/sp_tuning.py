# ...code for hyperparameter tuning in Stochastic Programming policy goes here...
import sys, os
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from pyomo.environ import value

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from part_B.policies.SP_policy_18 import build_model, build_scenario_tree, build_OMEGA_tw, solve_model
from part_B.running_script import run_script
from data.v2_SystemCharacteristics import get_fixed_data



# 1. Lookahead horizon and branching factor to test
L_test = list(range(2, 5 + 1))   # [2, 3, 4, 5]
B_test = list(range(2, 5 + 1))   # [2, 3, 4, 5]

# Store results in a 2D array (rows = L, cols = B)
avg_cost_grid = np.zeros((len(L_test), len(B_test)))

for i, l in enumerate(L_test):
    for j, b in enumerate(B_test):
        # Define select_action function
        def select_action(state, L=l, B=b):
            """Copy from SP_policy_18.py file, but takes L and B as arguments."""
            params = get_fixed_data()
            K = int(params['num_timeslots'])
            R = [1,2]
            S = 10  # Sampling factor

            # 2. ...
            L = min(L, K - state["current_time"])
            L_set = list(range(state["current_time"], state["current_time"] + L))
            OMEGA = pow(B,L-1)  # Number of scenarios generated
            OMEGA_set = list(range(1, OMEGA+1)) # Set of scenarios

            # 3. ...
            root, leaves, probabilities, (price, occ) = build_scenario_tree(state, L, B, S)

            # 4. ...
            OMEGA_tw = build_OMEGA_tw(state, leaves, OMEGA, OMEGA_set, L_set)

            # Setup the data dictionary
            data={
                "L": L_set,
                "R": R,
                "Omega": OMEGA_set,
                "Omega_tw": OMEGA_tw,
                "pi": probabilities,
                "price": price,
                "occ": occ
            }

            start_time = time.time() # NEW
            m = build_model(data, state)
            m, results = solve_model(m, start_time)

            HereAndNowActions = {
            "HeatPowerRoom1" : value(m.p[1, 1, state["current_time"]]),
            "HeatPowerRoom2" : value(m.p[1, 2, state["current_time"]]),
            "VentilationON" : value(m.v[1, state["current_time"]])
            }
            
            return HereAndNowActions

        # Run the script with select_action function with defined L and B
        print("L =", l, "B =", b)
        avg_cost = run_script(select_action)
        avg_cost_grid[i, j] = avg_cost

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    avg_cost_grid,
    xticklabels=B_test,
    yticklabels=L_test,
    annot=True,
    fmt=".2f",
    cmap="coolwarm_r",   # blue = low cost (good), red = high cost (bad)
    linewidths=0.5,
    linecolor="gray"
)

# Mark the minimum
min_idx = np.unravel_index(np.argmin(avg_cost_grid), avg_cost_grid.shape)
# plt.gca().add_patch(plt.Rectangle(
#     (min_idx[1], min_idx[0]), 1, 1,
#     fill=False, edgecolor="gold", lw=3, label="Optimum"
# ))

plt.xlabel("B")
plt.ylabel("L")
plt.title("Hyperparameter Tuning — Average Cost")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=150)
plt.show()

print(f"Best L = {L_test[min_idx[0]]}, Best B = {B_test[min_idx[1]]}, Avg cost = {avg_cost_grid[min_idx]:.3f}")