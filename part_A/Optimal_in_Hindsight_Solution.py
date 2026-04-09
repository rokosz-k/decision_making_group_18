# -*- coding: utf-8 -*-

"""
Solves one MILP per day for each of the 100 days in the dataset.
Reports the average daily electricity cost.

Each day is fully independent
The plot shows results for a representative day (defined later in the code).
"""

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
from SystemCharacteristics import get_fixed_data
from PlotsRestaurant import plot_HVAC_results

# Load data from the file

data = get_fixed_data()

# Sets
num_timeslots = data['num_timeslots']
T_slots = list(range(num_timeslots))
rooms = [0, 1]

# Parameters 
price_df = pd.read_csv('PriceData.csv', header=0) # Electricity price

occ1_df = pd.read_csv('OccupancyRoom1.csv', header=0) # Occupancy in room 1 
occ2_df = pd.read_csv('OccupancyRoom2.csv', header=0) # )ccupancy in room 2

L = num_timeslots # Number of timeslots in the Lookahead Horizon

T_out = data['outdoor_temperature'] # Outdoor temperature

P_vent = data['ventilation_power'] # Power consumption of ventilation

P_max = data['heating_max_power'] # Maximum heating power (for both rooms)

T_low = data['temp_min_comfort_threshold'] # Lower comfort temperature threshold (for both rooms)
T_high = data['temp_max_comfort_threshold'] # Upper comfort temperature threshold (for both rooms)

H_high = data['humidity_threshold'] # Humidity threshold for triggering ventilation

M_temp = 1e6 # Big-M constant for temperature logic
M_hum = 1e6 # Big-M constant for humidity logic

U_vent = data['vent_min_up_time'] # Minimum ventilation up-time 

# Thermal dynamics coefficients:
zeta_exch = data['heat_exchange_coeff']
zeta_loss = data['thermal_loss_coeff']
zeta_conv = data['heating_efficiency_coeff']
zeta_cool = data['heat_vent_coeff']
zeta_occ = data['heat_occupancy_coeff']

# Humidity dynamics coefficients:
eta_occ = data['humidity_occupancy_coeff']
eta_vent = data['humidity_vent_coeff']

# THIS IS LEFT FROM PREVIOUS CODE:
T_init = data['initial_temperature']
H_init = data['initial_humidity']
T_ok = data['temp_OK_threshold']

num_days = len(occ1_df)   # 100
print(f"Loaded {num_days} days of data.\n")

# SINGLE-DAY MILP

def solve_day(day_idx, occ1, occ2, price, verbose=False):

    mdl = gp.Model(f"hvac_day_{day_idx}")
    mdl.Params.OutputFlag = int(verbose)
    mdl.Params.MIPGap = 0.01
    mdl.Params.TimeLimit = 120

    # OCC[r, t]
    OCC = {(0, t): occ1[t] for t in T_slots}
    OCC.update({(1, t): occ2[t] for t in T_slots})


    # VARIABLES:
    p = mdl.addVars(rooms, T_slots, lb=0.0 ,name="p") # Heating power in room r at time t

    T = mdl.addVars(rooms, T_slots, lb=-GRB.INFINITY, name="T") # Indoor temperature in room r at time t

    H = mdl.addVars(T_slots, lb=0.0, name="H") #Indoor humidity at time t 

    v = mdl.addVars(T_slots, vtype=GRB.BINARY, name="v") # Binary variable indicating whether ventilation is active at time t 

    s = mdl.addVars(T_slots, vtype=GRB.BINARY, name="s")# Binary variable indicating ventilation startup is active at time t 

    y_high = mdl.addVars(rooms, T_slots, vtype=GRB.BINARY, name="y_high") # Binary variable indicating wherther the indoor temperature in room r at time t exceeds a predefined cutoff level

    # Auxiliary binary variables for detecting when room temperature is below or above thresholds:
    y_low = mdl.addVars(rooms, T_slots, vtype=GRB.BINARY, name="y_low")
    y_ok  = mdl.addVars(rooms, T_slots, vtype=GRB.BINARY, name="y_ok")

    u = mdl.addVars(rooms, T_slots, vtype=GRB.BINARY, name="u") # Binary variable indicating if the heater’s overrule controller is active at time t.


    # OBJECTIVE:
    mdl.setObjective(
        gp.quicksum(price[t] * P_vent * v[t] for t in T_slots)
        + gp.quicksum(price[t] * p[r, t] for r in rooms for t in T_slots),
        GRB.MINIMIZE
    )

    # previous-step values (use initial conditions at t=0)
    def T_prev(r, t): return T_init if t == 0 else T[r, t-1] # temperature (for each room)
    def v_prev(t): return 0 if t == 0 else v[t-1] # ventilation 
    def p_prev(r, t): return 0 if t == 0 else p[r, t-1] # heating (for each room)


    # CONSTRAINTS: 
    # Temperature dynamics (t=0 is set by initial condition)
    for r in rooms:
        other = [q for q in rooms if q != r]
        for t in T_slots:
            if t == 0:
                mdl.addConstr(T[r, 0] == T_init, name=f"temp_init[{r}]")
            else:
                exchange = gp.quicksum(T_prev(q, t) - T_prev(r, t) for q in other)
                mdl.addConstr(
                    T[r, t] == T_prev(r, t)
                        + zeta_exch * exchange
                        - zeta_loss * (T_prev(r, t) - T_out[t-1])
                        + zeta_conv * p_prev(r, t)
                        - zeta_cool * v_prev(t)
                        + zeta_occ  * OCC[r, t-1],
                    name=f"temp_dyn[{r},{t}]"
                )

    # Humidity dynamics
    for t in T_slots:
        if t == 0:
            mdl.addConstr(H[t] == H_init, name=f"hum_init")
        else:
            mdl.addConstr(
                H[t] == H[t-1]
                    + eta_occ  * gp.quicksum(OCC[r, t - 1] for r in rooms)
                    - eta_vent * v_prev(t),
                name=f"hum_dyn[{t}]"
            )

    # Heater upper power limits
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(p[r, t] <= P_max, name=f"p_max[{r},{t}]")

    # Detecting when Temperature is above threshold T_high
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(T[r, t] >= T_high - M_temp * (1 - y_high[r, t]), name=f"high_lb[{r},{t}]")
            mdl.addConstr(T[r, t] <= T_high + M_temp * y_high[r, t], name=f"high_ub[{r},{t}]")

    # Overrule controller forcing heater to zero:
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(p[r, t] <= P_max * (1 - y_high[r, t]), name=f"high_cutoff[{r},{t}]")

    # Detecting when Temperature is below threshold T_low
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(T[r, t] <= T_low + M_temp * (1 - y_low[r, t]), name=f"low_ub[{r},{t}]")
            mdl.addConstr(T[r, t] >= T_low - M_temp * y_low[r, t], name=f"low_lb[{r},{t}]")
            
    # Detecting when Temperature is above the "OK" threshold T_ok
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(T[r, t] >= T_ok - M_temp * (1 - y_ok[r, t]), name=f"ok_lb[{r},{t}]")
            mdl.addConstr(T[r, t] <= T_ok + M_temp * y_ok[r, t], name=f"ok_ub[{r},{t}]")

    # Triggering Overrule (only) if Temperature is low
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(u[r, t] >= y_low[r, t], name=f"recovery_start{r},{t}")
            if t > 0:
                mdl.addConstr(u[r, t] <= u[r, t-1] + y_low[r, t], name=f"recovery_persist{r},{t}")

    # Overrule controller forcing heater to maximum
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(p[r, t] >= P_max * u[r, t], name=f"recovery_power{r},{t}]")
    
    # De-activating Overrule (only) if Temperature is above the “OK” level
    for r in rooms:
        for t in T_slots:
            if t > 0:
                mdl.addConstr(u[r, t] >= u[r, t-1] - y_ok[r, t], name=f"recovery_complete{r},{t}")
                mdl.addConstr(u[r, t] <= 1 - y_ok[r, t], name=f"recovery_exit{r},{t}")

    # Ventilation Startup and Minimum Up-Time
    for t in T_slots:
        if t > 0:
            mdl.addConstr(s[t] >= v[t] - v[t-1])
            mdl.addConstr(s[t] <= v[t])
            mdl.addConstr(s[t] <= 1 - v[t-1])
            mdl.addConstr(gp.quicksum(v[tau] for tau in range(t, min(t+U_vent, L))) >= min(U_vent, L-t) * s[t])

    # Humidity-Triggered Ventilation
    for t in T_slots:
        mdl.addConstr(H[t] <= H_high + M_hum * v[t])
  
    # SOLVE

    mdl.optimize()

    if mdl.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and mdl.SolCount > 0:
        return {
            'cost': mdl.ObjVal,
            'Temp_r1': [T[0, t].X for t in T_slots],
            'Temp_r2': [T[1, t].X for t in T_slots],
            'h_r1': [p[0, t].X for t in T_slots],
            'h_r2': [p[1, t].X for t in T_slots],
            'v': [v[t].X for t in T_slots],
            'Hum': [H[t].X for t in T_slots],
            'Occ_r1': [OCC[0, t] for t in T_slots],
            'Occ_r2': [OCC[1, t] for t in T_slots],
            'price': price,
        }
    else:
        print(f"  WARNING: Day {day_idx} — no solution found (status {mdl.Status})")
        return None


# SOLVE ALL 100 DAYS

daily_costs = []
results_per_day = []

for day in range(num_days):
    occ1 = [float(occ1_df.iloc[day, t]) for t in T_slots]
    occ2 = [float(occ2_df.iloc[day, t]) for t in T_slots]
    price = [float(price_df.iloc[day, t]) for t in T_slots]

    print(f"Solving day {day + 1:3d} / {num_days} ...", end=" ")
    res = solve_day(day, occ1, occ2, price, verbose=False)

    if res is not None:
        daily_costs.append(res['cost'])
        results_per_day.append(res)
        print(f"cost = {res['cost']:.4f} €")
    else:
        results_per_day.append(None)
        print("FAILED")



# SUMMARY

solved_costs = [c for c in daily_costs]
print(f"\n{'='*45}")
print(f"  Days solved:          {len(solved_costs)} / {num_days}")
print(f"  Average daily cost:   {np.mean(solved_costs):.4f} €")
print(f"  Std deviation:        {np.std(solved_costs):.4f} €")
print(f"  Min daily cost:       {np.min(solved_costs):.4f} €")
print(f"  Max daily cost:       {np.max(solved_costs):.4f} €")
print(f"{'='*45}\n")



# PLOT — representative day (change DAY_TO_PLOT as needed)

DAY_TO_PLOT = 0   # change to any day index 0-99 (value 0 corresponds to the first day in the dataset)

res = results_per_day[DAY_TO_PLOT]
if res is not None:
    HVAC_results = {
        'T': T_slots,
        'Temp_r1': res['Temp_r1'],
        'Temp_r2': res['Temp_r2'],
        'h_r1': res['h_r1'],
        'h_r2': res['h_r2'],
        'v': res['v'],
        'Hum': res['Hum'],
        'Occ_r1': res['Occ_r1'],
        'Occ_r2': res['Occ_r2'],
        'price': res['price'],
    }
    print(f"Plotting results for day {DAY_TO_PLOT}...")
    # plot_HVAC_results(HVAC_results)

    
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Plotting a basic histogram
# plt.hist(solved_costs, bins=15, color='skyblue', edgecolor='black')
sns.histplot(solved_costs, bins=10, kde=True, color='skyblue', edgecolor='black')

# Adding labels and title
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Basic Histogram')

# Display the plot
plt.show()