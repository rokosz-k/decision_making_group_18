# -*- coding: utf-8 -*-

"""
Solves one MILP per day for each of the 100 days in the dataset.
Reports the average daily electricity cost.

Each day is fully independent
The plot shows results for a representative day (defined later in the code).
"""

import sys
import os
import numpy as np
import pandas as pd
import pyomo.environ as pyo

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from part_A.SystemCharacteristics import get_fixed_data
from part_A.PlotsRestaurant import plot_HVAC_results

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
price_path = os.path.join(BASE_DIR, "data", "PriceData.csv")
occ1_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom1.csv")
occ2_path  = os.path.join(BASE_DIR, "data", "OccupancyRoom2.csv")

data = get_fixed_data()

num_timeslots = data['num_timeslots']
T_slots = list(range(num_timeslots))
rooms = [0, 1]

price_df = pd.read_csv(price_path, header=0)
occ1_df  = pd.read_csv(occ1_path,  header=0)
occ2_df  = pd.read_csv(occ2_path,  header=0)

L = num_timeslots

T_out  = data['outdoor_temperature']
P_vent = data['ventilation_power']
P_max  = data['heating_max_power']
T_low  = data['temp_min_comfort_threshold']
T_high = data['temp_max_comfort_threshold']
H_high = data['humidity_threshold']

M_temp = 1e6
M_hum  = 1e6

U_vent = data['vent_min_up_time']

zeta_exch = data['heat_exchange_coeff']
zeta_loss = data['thermal_loss_coeff']
zeta_conv = data['heating_efficiency_coeff']
zeta_cool = data['heat_vent_coeff']
zeta_occ  = data['heat_occupancy_coeff']

eta_occ  = data['humidity_occupancy_coeff']
eta_vent = data['humidity_vent_coeff']

T_init = data['initial_temperature']
H_init = data['initial_humidity']
T_ok   = data['temp_OK_threshold']

num_days = len(occ1_df)
print(f"Loaded {num_days} days of data.\n")


# SINGLE-DAY MILP

def solve_day(day_idx, occ1, occ2, price, verbose=False):

    mdl = pyo.ConcreteModel(name=f"hvac_day_{day_idx}")

    OCC = {(0, t): occ1[t] for t in T_slots}
    OCC.update({(1, t): occ2[t] for t in T_slots})

    mdl.rooms   = pyo.Set(initialize=rooms)
    mdl.T_slots = pyo.Set(initialize=T_slots)

    # VARIABLES
    mdl.p      = pyo.Var(mdl.rooms, mdl.T_slots, domain=pyo.NonNegativeReals)
    mdl.T      = pyo.Var(mdl.rooms, mdl.T_slots, domain=pyo.Reals)
    mdl.H      = pyo.Var(mdl.T_slots, domain=pyo.NonNegativeReals)
    mdl.v      = pyo.Var(mdl.T_slots, domain=pyo.Binary)
    mdl.s      = pyo.Var(mdl.T_slots, domain=pyo.Binary)
    mdl.y_high = pyo.Var(mdl.rooms, mdl.T_slots, domain=pyo.Binary)
    mdl.y_low  = pyo.Var(mdl.rooms, mdl.T_slots, domain=pyo.Binary)
    mdl.y_ok   = pyo.Var(mdl.rooms, mdl.T_slots, domain=pyo.Binary)
    mdl.u      = pyo.Var(mdl.rooms, mdl.T_slots, domain=pyo.Binary)

    # OBJECTIVE
    def obj_rule(m):
        return (sum(price[t] * P_vent * m.v[t] for t in T_slots)
              + sum(price[t] * m.p[r, t] for r in rooms for t in T_slots))
    mdl.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    def T_prev(m, r, t): return T_init if t == 0 else m.T[r, t-1]
    def v_prev(m, t):    return 0      if t == 0 else m.v[t-1]
    def p_prev(m, r, t): return 0      if t == 0 else m.p[r, t-1]

    # CONSTRAINTS

    # Temperature dynamics
    def temp_init_rule(m, r):
        return m.T[r, 0] == T_init
    mdl.temp_init = pyo.Constraint(mdl.rooms, rule=temp_init_rule)

    def temp_dyn_rule(m, r, t):
        if t == 0:
            return pyo.Constraint.Skip
        other    = [q for q in rooms if q != r]
        exchange = sum(T_prev(m, q, t) - T_prev(m, r, t) for q in other)
        return m.T[r, t] == (T_prev(m, r, t)
                              + zeta_exch * exchange
                              - zeta_loss * (T_prev(m, r, t) - T_out[t-1])
                              + zeta_conv * p_prev(m, r, t)
                              - zeta_cool * v_prev(m, t)
                              + zeta_occ  * OCC[r, t-1])
    mdl.temp_dyn = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=temp_dyn_rule)

    # Humidity dynamics
    def hum_init_rule(m):
        return m.H[0] == H_init
    mdl.hum_init = pyo.Constraint(rule=hum_init_rule)

    def hum_dyn_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.H[t] == (m.H[t-1]
                          + eta_occ  * sum(OCC[r, t-1] for r in rooms)
                          - eta_vent * v_prev(m, t))
    mdl.hum_dyn = pyo.Constraint(mdl.T_slots, rule=hum_dyn_rule)

    # Heater upper power limits
    def p_max_rule(m, r, t):
        return m.p[r, t] <= P_max
    mdl.p_max = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=p_max_rule)

    # Detecting when Temperature is above threshold T_high
    def high_lb_rule(m, r, t):
        return m.T[r, t] >= T_high - M_temp * (1 - m.y_high[r, t])
    mdl.high_lb = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=high_lb_rule)

    def high_ub_rule(m, r, t):
        return m.T[r, t] <= T_high + M_temp * m.y_high[r, t]
    mdl.high_ub = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=high_ub_rule)

    # Overrule controller forcing heater to zero
    def high_cutoff_rule(m, r, t):
        return m.p[r, t] <= P_max * (1 - m.y_high[r, t])
    mdl.high_cutoff = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=high_cutoff_rule)

    # Detecting when Temperature is below threshold T_low
    def low_ub_rule(m, r, t):
        return m.T[r, t] <= T_low + M_temp * (1 - m.y_low[r, t])
    mdl.low_ub = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=low_ub_rule)

    def low_lb_rule(m, r, t):
        return m.T[r, t] >= T_low - M_temp * m.y_low[r, t]
    mdl.low_lb = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=low_lb_rule)

    # Detecting when Temperature is above the "OK" threshold T_ok
    def ok_lb_rule(m, r, t):
        return m.T[r, t] >= T_ok - M_temp * (1 - m.y_ok[r, t])
    mdl.ok_lb = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=ok_lb_rule)

    def ok_ub_rule(m, r, t):
        return m.T[r, t] <= T_ok + M_temp * m.y_ok[r, t]
    mdl.ok_ub = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=ok_ub_rule)

    # Triggering Overrule (only) if Temperature is low
    def recovery_start_rule(m, r, t):
        return m.u[r, t] >= m.y_low[r, t]
    mdl.recovery_start = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=recovery_start_rule)

    def recovery_persist_rule(m, r, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.u[r, t] <= m.u[r, t-1] + m.y_low[r, t]
    mdl.recovery_persist = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=recovery_persist_rule)

    # Overrule controller forcing heater to maximum
    def recovery_power_rule(m, r, t):
        return m.p[r, t] >= P_max * m.u[r, t]
    mdl.recovery_power = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=recovery_power_rule)

    # De-activating Overrule (only) if Temperature is above the "OK" level
    def recovery_complete_rule(m, r, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.u[r, t] >= m.u[r, t-1] - m.y_ok[r, t]
    mdl.recovery_complete = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=recovery_complete_rule)

    def recovery_exit_rule(m, r, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.u[r, t] <= 1 - m.y_ok[r, t]
    mdl.recovery_exit = pyo.Constraint(mdl.rooms, mdl.T_slots, rule=recovery_exit_rule)

    # Ventilation Startup and Minimum Up-Time
    def vent_s_lb_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.s[t] >= m.v[t] - m.v[t-1]
    mdl.vent_s_lb = pyo.Constraint(mdl.T_slots, rule=vent_s_lb_rule)

    def vent_s_ub1_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.s[t] <= m.v[t]
    mdl.vent_s_ub1 = pyo.Constraint(mdl.T_slots, rule=vent_s_ub1_rule)

    def vent_s_ub2_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return m.s[t] <= 1 - m.v[t-1]
    mdl.vent_s_ub2 = pyo.Constraint(mdl.T_slots, rule=vent_s_ub2_rule)

    def vent_min_up_rule(m, t):
        if t == 0:
            return pyo.Constraint.Skip
        return sum(m.v[tau] for tau in range(t, min(t + U_vent, L))) >= min(U_vent, L - t) * m.s[t]
    mdl.vent_min_up = pyo.Constraint(mdl.T_slots, rule=vent_min_up_rule)

    # Humidity-Triggered Ventilation
    def hum_trigger_rule(m, t):
        return m.H[t] <= H_high + M_hum * m.v[t]
    mdl.hum_trigger = pyo.Constraint(mdl.T_slots, rule=hum_trigger_rule)

    # SOLVE
    solver = pyo.SolverFactory('gurobi')
    solver.options['MIPGap']    = 0.01
    solver.options['TimeLimit'] = 120
    result = solver.solve(mdl, tee=verbose)

    optimal    = pyo.TerminationCondition.optimal
    time_limit = pyo.TerminationCondition.maxTimeLimit
    tc = result.solver.termination_condition

    if tc in (optimal, time_limit) and result.solver.status == pyo.SolverStatus.ok:
        return {
            'cost':    pyo.value(mdl.obj),
            'Temp_r1': [pyo.value(mdl.T[0, t]) for t in T_slots],
            'Temp_r2': [pyo.value(mdl.T[1, t]) for t in T_slots],
            'h_r1':    [pyo.value(mdl.p[0, t]) for t in T_slots],
            'h_r2':    [pyo.value(mdl.p[1, t]) for t in T_slots],
            'v':       [pyo.value(mdl.v[t])    for t in T_slots],
            'Hum':     [pyo.value(mdl.H[t])    for t in T_slots],
            'Occ_r1':  [OCC[0, t] for t in T_slots],
            'Occ_r2':  [OCC[1, t] for t in T_slots],
            'price':   price,
        }
    else:
        print(f"  WARNING: Day {day_idx} — no solution found (termination: {tc})")
        return None


# SOLVE ALL 100 DAYS

daily_costs     = []
results_per_day = []

for day in range(num_days):
    occ1  = [float(occ1_df.iloc[day, t])  for t in T_slots]
    occ2  = [float(occ2_df.iloc[day, t])  for t in T_slots]
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

DAY_TO_PLOT = 0

res = results_per_day[DAY_TO_PLOT]
if res is not None:
    HVAC_results = {
        'T':       T_slots,
        'Temp_r1': res['Temp_r1'],
        'Temp_r2': res['Temp_r2'],
        'h_r1':    res['h_r1'],
        'h_r2':    res['h_r2'],
        'v':       res['v'],
        'Hum':     res['Hum'],
        'Occ_r1':  res['Occ_r1'],
        'Occ_r2':  res['Occ_r2'],
        'price':   res['price'],
    }
    print(f"Plotting results for day {DAY_TO_PLOT}...")
    # plot_HVAC_results(HVAC_results)


import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(solved_costs, bins=10, kde=True, color='skyblue', edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Basic Histogram')
plt.show()