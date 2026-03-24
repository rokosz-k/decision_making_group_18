#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 15:54:50 2026

@author: jaredbutler
"""

import pandas as pd

def reset_env(data, occupancy):
    """
    Initializes the environment state as a dictionary.
    """
    state = {
        "T1": data['initial_temperature'],
        "T2": data['initial_temperature'],
        "H": data['initial_humidity'],
        "Occ1": occupancy["Room1"][0],
        "Occ2": occupancy["Room2"][0],
        "price_t": data['price'][0],
        "price_previous": 0.0,  # assume 0 at t=0
        "vent_counter": 0,
        "low_override_r1": 0,
        "low_override_r2": 0,
        "current_time": 0
    }
    return state

def step_env(state, action, data, occupancy):
    """
    Updates the state dictionary based on the action and system dynamics.
    
    Inputs:
        state: current state dict
        action: dict with keys "HeatPowerRoom1", "HeatPowerRoom2", "VentilationON"
        data: system parameters
        occupancy: dict of occupancy per room per hour
    Returns:
        new_state: updated state dictionary
        cost: cost for this step
        done: True if simulation is finished
    """
    # -----------------------------
    # Extract old state
    # -----------------------------
    t = state["current_time"]
    T1 = state["T1"]
    T2 = state["T2"]
    H  = state["H"]
    price_previous = state["price_t"]
    vent_counter = state["vent_counter"]
    low_r1 = state["low_override_r1"]
    low_r2 = state["low_override_r2"]

    P1 = action["HeatPowerRoom1"]
    P2 = action["HeatPowerRoom2"]
    vent_on = action["VentilationON"]

    # Occupancy this timestep
    Occ1 = occupancy["Room1"][t]
    Occ2 = occupancy["Room2"][t]

    # -----------------------------
    # Temperature dynamics
    # -----------------------------
    T1_new = (
        T1
        + data['heat_exchange_coeff'] * (T2 - T1)
        - data['thermal_loss_coeff'] * (T1 - data['outdoor_temperature'][t])
        + data['heating_efficiency_coeff'] * P1
        - data['heat_vent_coeff'] * vent_on
        + data['heat_occupancy_coeff'] * Occ1
    )

    T2_new = (
        T2
        + data['heat_exchange_coeff'] * (T1 - T2)
        - data['thermal_loss_coeff'] * (T2 - data['outdoor_temperature'][t])
        + data['heating_efficiency_coeff'] * P2
        - data['heat_vent_coeff'] * vent_on
        + data['heat_occupancy_coeff'] * Occ2
    )

    # -----------------------------
    # Humidity dynamics
    # -----------------------------
    H_new = H + data['humidity_occupancy_coeff'] * (Occ1 + Occ2) - data['humidity_vent_coeff'] * vent_on

    # -----------------------------
    # Low-temperature overrule controller
    # -----------------------------
    T_low = data['temp_min_comfort_threshold']
    T_ok  = data['temp_OK_threshold']

    # Room1
    if T1_new < T_low:
        low_r1 = 1
    elif low_r1 == 1 and T1_new >= T_ok:
        low_r1 = 0

    # Room2
    if T2_new < T_low:
        low_r2 = 1
    elif low_r2 == 1 and T2_new >= T_ok:
        low_r2 = 0

    # -----------------------------
    # Enforce max heating during low override
    # -----------------------------
    if low_r1 == 1:
        P1 = data['heating_max_power']
    if low_r2 == 1:
        P2 = data['heating_max_power']

    # -----------------------------
    # High temperature cutoff
    # -----------------------------
    T_high = data['temp_max_comfort_threshold']
    if T1_new > T_high:
        P1 = 0
    if T2_new > T_high:
        P2 = 0

    # -----------------------------
    # Ventilation inertia
    # -----------------------------
    if vent_on == 1 and vent_counter == 0:
        vent_counter = data['vent_min_up_time']
    if vent_counter > 0:
        vent_on = 1
        vent_counter -= 1

    # -----------------------------
    # Cost function
    # -----------------------------
    P_vent = data['ventilation_power']
    price_t = data['price'][t]
    cost = price_t * (P1 + P2) + price_t * P_vent * vent_on

    # -----------------------------
    # Update state dictionary
    # -----------------------------
    new_state = {
        "T1": T1_new,
        "T2": T2_new,
        "H": H_new,
        "Occ1": Occ1,
        "Occ2": Occ2,
        "price_t": price_t,
        "price_previous": price_previous,
        "vent_counter": vent_counter,
        "low_override_r1": low_r1,
        "low_override_r2": low_r2,
        "current_time": t + 1
    }

    done = t + 1 >= data['num_timeslots']

    return new_state, cost, done