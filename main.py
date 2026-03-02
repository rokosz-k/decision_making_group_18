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

num_timeslots = data['num_timeslots']
T_slots = list(range(num_timeslots))
rooms = [0, 1]

T_init = data['initial_temperature']
H_init = data['initial_humidity']

P_max = data['heating_max_power']
zeta_exch = data['heat_exchange_coeff']
zeta_conv = data['heating_efficiency_coeff']
zeta_loss = data['thermal_loss_coeff']
zeta_cool = data['heat_vent_coeff']
zeta_occ = data['heat_occupancy_coeff']

T_low = data['temp_min_comfort_threshold']
T_ok = data['temp_OK_threshold']
T_high = data['temp_max_comfort_threshold']
T_out = data['outdoor_temperature']

T_vent = data['vent_min_up_time']
H_high = data['humidity_threshold']
P_vent = data['ventilation_power']
eta_occ = data['humidity_occupancy_coeff']
eta_vent = data['humidity_vent_coeff']

M = 1e4

# Load time series data for occupancy and electricity prices

occ1_df = pd.read_csv('OccupancyRoom1.csv', header=0)
occ2_df = pd.read_csv('OccupancyRoom2.csv', header=0)
price_df = pd.read_csv('PriceData.csv',      header=0)

num_days = len(occ1_df)   # 100
print(f"Loaded {num_days} days of data.\n")


# SINGLE-DAY MILP

def solve_day(day_idx, occ1, occ2, price, verbose=False):
    """
    Solve the HVAC MILP for a single day.

    Parameters
    ----------
    day_idx : int   — used only for labelling
    occ1    : list  — occupancy room 1, length num_timeslots
    occ2    : list  — occupancy room 2, length num_timeslots
    price   : list  — electricity price (€/kWh), length num_timeslots
    verbose : bool  — show Gurobi log

    Returns
    -------
    dict with cost and result time series, or None if infeasible
    """

    mdl = gp.Model(f"hvac_day_{day_idx}")
    mdl.Params.OutputFlag = int(verbose)
    mdl.Params.MIPGap = 0.01
    mdl.Params.TimeLimit = 120

    # OCC[r, t]
    OCC = {(0, t): occ1[t] for t in T_slots}
    OCC.update({(1, t): occ2[t] for t in T_slots})


    # Variables

    p = mdl.addVars(rooms, T_slots, lb=0.0 ,name="p")
    T_var = mdl.addVars(rooms, T_slots, lb=-GRB.INFINITY, name="T")
    H = mdl.addVars(T_slots, lb=0.0, name="H")

    v = mdl.addVars(T_slots, vtype=GRB.BINARY, name="v")
    y = mdl.addVars(T_slots, vtype=GRB.BINARY, name="y")

    alpha = mdl.addVars(rooms, T_slots, vtype=GRB.BINARY, name="alpha")
    beta  = mdl.addVars(rooms, T_slots, vtype=GRB.BINARY, name="beta")
    delta = mdl.addVars(rooms, T_slots, vtype=GRB.BINARY, name="delta")
    nu = mdl.addVars(rooms, T_slots, vtype=GRB.BINARY, name="nu")


    # Objective

    mdl.setObjective(
        gp.quicksum(price[t] * p[r, t] for r in rooms for t in T_slots)
        + gp.quicksum(price[t] * P_vent * v[t] for t in T_slots),
        GRB.MINIMIZE
    )

    # previous-step values (use initial conditions at t=0)

    def T_prev(r, t): return T_init if t == 0 else T_var[r, t - 1]
    def v_prev(t): return 0 if t == 0 else v[t - 1]
    def p_prev(r, t): return 0 if t == 0 else p[r, t - 1]


    # Constraints

    # Temperature dynamics (t=0 is set by initial condition)
    for r in rooms:
        other = [q for q in rooms if q != r]
        for t in T_slots:
            if t == 0:
                mdl.addConstr(T_var[r, 0] == T_init, name=f"temp_init[{r}]")
            else:
                exchange = gp.quicksum(T_prev(q, t) - T_prev(r, t) for q in other)
                mdl.addConstr(
                    T_var[r, t] == T_prev(r, t)
                        + zeta_exch * exchange
                        - zeta_loss * (T_prev(r, t) - T_out[t - 1])
                        + zeta_conv * p_prev(r, t)
                        - zeta_cool * v_prev(t)
                        + zeta_occ  * OCC[r, t - 1],
                    name=f"temp_dyn[{r},{t}]"
                )

    # Humidity dynamics
    for t in T_slots:
        if t == 0:
            mdl.addConstr(H[t] == H_init, name=f"hum_init")
        else:
            H_prev = H[t - 1]
            occ_hum_prev = gp.quicksum(OCC[r, t - 1] for r in rooms)
            mdl.addConstr(
                H[t] == H_prev
                    + eta_occ  * occ_hum_prev
                    - eta_vent * v_prev(t),
                name=f"hum_dyn[{t}]"
            )

    # Low-temperature trigger: alpha[r,t] = 1 when T < T_low
    # One-sided Big-M: if alpha=0 then T >= T_low
    # The objective keeps alpha=0 unless temperature actually drops below T_low
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(T_var[r, t] >= T_low - M * alpha[r, t], name=f"low_lb[{r},{t}]")

    # Recovery mode
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(delta[r, t] >= alpha[r, t], name=f"recov_start[{r},{t}]")
            if t > 0:
                mdl.addConstr(delta[r, t] >= delta[r, t-1] - nu[r, t], name=f"recov_persist[{r},{t}]")

    # Recovery completion
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(T_var[r, t] >= T_ok - M * (1 - nu[r, t]), name=f"recov_complete[{r},{t}]")
            mdl.addConstr(nu[r, t] <= 1 - delta[r, t], name=f"nu_exit[{r},{t}]")
            if t == 0:
                mdl.addConstr(nu[r, t] <= 0, name=f"nu_init[{r}]")
            else:
                mdl.addConstr(nu[r, t] <= delta[r, t-1], name=f"nu_need_prev[{r},{t}]")

    # Force max power during recovery
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(p[r, t] >= P_max * delta[r, t], name=f"recov_power[{r},{t}]")

    # High-temperature trigger: beta[r,t] = 1 when T > T_high
    # One-sided Big-M: if beta=0 then T <= T_high
    # The objective keeps beta=0 unless temperature actually exceeds T_high
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(T_var[r, t] <= T_high + M * beta[r, t], name=f"high_ub[{r},{t}]")

    # Force zero heating when too hot
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(p[r, t] <= P_max * (1 - beta[r, t]), name=f"high_cutoff[{r},{t}]")

    # Heater upper bound
    for r in rooms:
        for t in T_slots:
            mdl.addConstr(p[r, t] <= P_max, name=f"p_max[{r},{t}]")

    # Ventilation humidity trigger
    for t in T_slots:
        mdl.addConstr(H[t] <= H_high + M * v[t], name=f"vent_hum[{t}]")

    # Ventilation startup indicator
    mdl.addConstr(y[0] >= v[0], name="vent_startup_0")
    for t in T_slots:
        if t > 0:
            mdl.addConstr(y[t] >= v[t] - v[t - 1], name=f"vent_startup[{t}]")

    # Ventilation inertia
    for t in T_slots:
        if t <= num_timeslots - T_vent:
            # Standard case: sum from t to t+T_vent-1 >= T_vent * y[t]
            mdl.addConstr(
                gp.quicksum(v[k] for k in range(t, t + T_vent)) >= T_vent * y[t],
                name=f"vent_inertia[{t}]"
            )
        else:
            # End-of-horizon: sum from t to T-1 >= T_vent * y[t]
            mdl.addConstr(
                gp.quicksum(v[k] for k in range(t, num_timeslots)) >= T_vent * y[t],
                name=f"vent_inertia[{t}]"
            )


    # Solve

    mdl.optimize()

    if mdl.Status in (GRB.OPTIMAL, GRB.TIME_LIMIT) and mdl.SolCount > 0:
        return {
            'cost': mdl.ObjVal,
            'Temp_r1': [T_var[0, t].X for t in T_slots],
            'Temp_r2': [T_var[1, t].X for t in T_slots],
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

DAY_TO_PLOT = 30   # change to any day index 0-99 (value 0 corresponds to the first day in the dataset)

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
    plot_HVAC_results(HVAC_results)