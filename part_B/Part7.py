#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 14 14:49:58 2026

@author: jaredbutler
"""
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from DataTask7 import fetch_data

data = fetch_data()

occ_df = pd.read_csv("Task7Occupancies.csv", usecols=range(10))
occ = occ_df.values.astype(float)   # shape (2, 10): row=room, col=hour



N_STORES  = 15
STORES    = list(range(N_STORES))
ROOMS     = [0, 1]
HOURS     = list(range(data['num_timeslots']))
 
P_mall    = data['P_mall']
T_ref     = data['Temperature_reference']
T_init    = data['initial_temperature']
P_max     = data['heating_max_power']
 
zeta_exch = data['heat_exchange_coeff']
zeta_conv = data['heating_efficiency_coeff']
zeta_loss = data['thermal_loss_coeff']
zeta_cool = data['heat_vent_coeff']
zeta_occ  = data['heat_occupancy_coeff']
 
T_out     = np.array(data['outdoor_temperature'])   # shape (10,)
 
# Weights: w_n = n+1 for 1-indexed → n+2 for 0-indexed stores
weights   = {n: n + 2 for n in STORES}


# occ[r, t] = occupancy of room r at hour t
print("Occupancy shape:", occ.shape)
print("Room 1:", np.round(occ[0], 1))
print("Room 2:", np.round(occ[1], 1))
print("Outdoor temps:", np.round(T_out, 2))
print("Weights:", weights)



def solve_store(n, lam, occ, data):
    """
    Solve one store's QP given current lambda vector.
    
    Parameters
    ----------
    n    : store index (0-indexed)
    lam  : np.array of shape (10,) - current dual variables λ_t
    occ  : np.array of shape (2, 10) - occupancy[room, hour]
    data : dict from fetch_data()
    
    Returns
    -------
    p_opt : np.array (2, 10) - optimal heater powers
    T_opt : np.array (2, 10) - resulting temperatures (hours 1..10)
    """
    w  = weights[n]
    m  = pyo.ConcreteModel()

    m.R = pyo.Set(initialize=ROOMS)
    m.T = pyo.Set(initialize=HOURS)   # 0..9 = decision epochs

    # --- Variables ---
    m.p = pyo.Var(m.R, m.T, bounds=(0, P_max))          # heater power
    m.T_room = pyo.Var(m.R, m.T)                         # temperature AFTER each hour

    # --- Objective: minimize temp deviation + lambda penalty on power ---
    def obj_rule(m):
        temp_cost  = sum(w * (m.T_room[r, t] - T_ref)**2
                         for r in m.R for t in m.T)
        power_cost = sum(lam[t] * (m.p[0, t] + m.p[1, t])
                         for t in m.T)
        return temp_cost + power_cost
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- Temperature dynamics ---
    def temp_dynamics(m, r, t):
        other = 1 - r   # the other room
        T_prev       = T_init      if t == 0 else m.T_room[r,     t-1]
        T_other_prev = T_init      if t == 0 else m.T_room[other, t-1]
        T_out_prev   = T_out[t]    # outdoor temp at this hour
        return m.T_room[r, t] == (T_prev
                                  + zeta_exch * (T_other_prev - T_prev)
                                  + zeta_loss * (T_out_prev   - T_prev)
                                  + zeta_conv *  m.p[r, t]
                                  - zeta_cool
                                  + zeta_occ  *  occ[r, t])
    m.dynamics = pyo.Constraint(m.R, m.T, rule=temp_dynamics)

    # --- Solve ---
    solver = pyo.SolverFactory('gurobi')
    solver.solve(m, tee=False)

    p_opt = np.array([[pyo.value(m.p[r, t]) for t in HOURS] for r in ROOMS])
    T_opt = np.array([[pyo.value(m.T_room[r, t]) for t in HOURS] for r in ROOMS])

    return p_opt, T_opt


def run_dual_decomposition(alpha, n_iter=100):
    lam = np.zeros(len(HOURS))
    
    obj_history = []
    lam_history = []
    violation_history = []

    for k in range(n_iter):
        print(f"Iteration {k+1}/{n_iter} | lambda max: {lam.max():.4f}")
        all_p = np.zeros((N_STORES, 2, len(HOURS)))
        all_T = np.zeros((N_STORES, 2, len(HOURS)))

        # Each store solves independently
        for n in STORES:
            print(f"  Solving store {n+1}/{N_STORES}", end='\r')
            p_opt, T_opt = solve_store(n, lam, occ, data)
            all_p[n] = p_opt
            all_T[n] = T_opt

        # Compute true objective (no lambda term)
        obj = sum(weights[n] * (all_T[n, r, t] - T_ref)**2
                  for n in STORES for r in ROOMS for t in HOURS)

        # Constraint violation per hour
        g = all_p.sum(axis=(0,1)) - P_mall   # shape (10,)

        # Update multipliers
        lam = np.maximum(0, lam + alpha * g)

        obj_history.append(obj)
        lam_history.append(lam.copy())
        violation_history.append(g.copy())

    return np.array(obj_history), np.array(lam_history), np.array(violation_history)



alphas = [0.001, 0.01, 0.1, 1, 10]
results = {}

for alpha in alphas:
    obj_hist, lam_hist, viol_hist = run_dual_decomposition(alpha, n_iter=100)
    results[alpha] = {
        'obj': obj_hist,
        'lam': lam_hist,
        'viol': viol_hist
    }

#%% 

# --- Adaptive step size: α_k = α0 / (1 + k) ---
alpha0 = 5
lam = np.zeros(len(HOURS))

obj_history = []
lam_history = []
violation_history = []

for k in range(100):
    print(f"Adaptive iteration {k+1}/100 | lambda max: {lam.max():.4f}")
    all_p = np.zeros((N_STORES, 2, len(HOURS)))
    all_T = np.zeros((N_STORES, 2, len(HOURS)))

    for n in STORES:
        print(f"  Solving store {n+1}/{N_STORES}", end='\r')
        p_opt, T_opt = solve_store(n, lam, occ, data)
        all_p[n] = p_opt
        all_T[n] = T_opt

    obj = sum(weights[n] * (all_T[n, r, t] - T_ref)**2
              for n in STORES for r in ROOMS for t in HOURS)

    g = all_p.sum(axis=(0, 1)) - P_mall

    step = alpha0 / (1 + k)
    lam = np.maximum(0, lam + step * g)

    obj_history.append(obj)
    lam_history.append(lam.copy())
    violation_history.append(g.copy())

results['adaptive'] = {
    'obj': np.array(obj_history),
    'lam': np.array(lam_history),
    'viol': np.array(violation_history)
}

#%%
def solve_centralized(occ, data):
    m = pyo.ConcreteModel()
    m.N = pyo.Set(initialize=STORES)
    m.R = pyo.Set(initialize=ROOMS)
    m.T = pyo.Set(initialize=HOURS)

    m.p = pyo.Var(m.N, m.R, m.T, bounds=(0, P_max))
    m.T_room = pyo.Var(m.N, m.R, m.T)

    # Objective
    def obj_rule(m):
        return sum(weights[n] * (m.T_room[n, r, t] - T_ref)**2
                   for n in m.N for r in m.R for t in m.T)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Power constraint
    def power_constraint(m, t):
        return sum(m.p[n, r, t] for n in m.N for r in m.R) <= P_mall
    m.power_con = pyo.Constraint(m.T, rule=power_constraint)

    # Temperature dynamics
    def temp_dynamics(m, n, r, t):
        other = 1 - r
        T_prev       = T_init if t == 0 else m.T_room[n, r,     t-1]
        T_other_prev = T_init if t == 0 else m.T_room[n, other, t-1]
        return m.T_room[n, r, t] == (T_prev
                                    + zeta_exch * (T_other_prev - T_prev)
                                    + zeta_loss * (T_out[t]     - T_prev)
                                    + zeta_conv *  m.p[n, r, t]
                                    - zeta_cool
                                    + zeta_occ  *  occ[r, t])
    m.dynamics = pyo.Constraint(m.N, m.R, m.T, rule=temp_dynamics)

    solver = pyo.SolverFactory('gurobi')
    solver.solve(m)

    return pyo.value(m.obj)

central_obj = solve_centralized(occ, data)
print(f"Centralized optimal objective: {central_obj:.2f}")

#%%

fig, axes = plt.subplots(3, 1, figsize=(10, 10))

for alpha, res in results.items():
    axes[0].plot(res['obj'], label=f'α={alpha}')
    axes[1].plot(np.max(np.abs(res['viol']), axis=1), label=f'α={alpha}')
    axes[2].plot(np.max(res['lam'], axis=1), label=f'α={alpha}')

axes[0].set_ylabel('Objective')
axes[0].set_title('Objective vs Iteration')
axes[0].axhline(y=central_obj, color='black', linestyle='--', linewidth=1.5, label='Centralized optimal')
axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

axes[1].set_ylabel('Max Constraint Violation')
axes[1].set_title('Violation vs Iteration')
axes[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

axes[2].set_ylabel('Max Lambda')
axes[2].set_title('Dual Variable vs Iteration')
axes[2].legend(loc='upper left', bbox_to_anchor=(1, 1))

for ax in axes:
    ax.set_xlabel('Iteration')
    ax.grid(True)

plt.tight_layout()
plt.savefig('dual_decomposition.png', dpi=150)
plt.show()

#%%


# Run one final solve with converged lambda from alpha=0.1
best_lam = results[0.1]['lam'][-1]
total_power_per_store = []

for n in STORES:
    p_opt, _ = solve_store(n, best_lam, occ, data)
    total_power_per_store.append(p_opt.sum())
    
for n, p in enumerate(total_power_per_store):
    print(f"Store {n} (w={weights[n]}): total power = {p:.2f}")
    

    
