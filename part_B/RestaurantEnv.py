#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RestaurantEnv.py  (v2)

Changes from the original:
  reset_env:
    - Uses v2_SystemCharacteristics key names: data['T1'], data['T2'], data['H']
      instead of data['initial_temperature'] / data['initial_humidity'].
    - price_previous is read from data['price_previous'] instead of being
      hardcoded to 0.0.  The caller is responsible for setting this key:
        * When using v2_SystemCharacteristics directly (evaluation environment),
          data['price_previous'] is already set to a random draw U[2,8].
        * When replaying historical data (collect_samples, running_script),
          set  data['price_previous'] = v2_PriceData[day][0]  before calling.
    - vent_counter / low_override_r1 / low_override_r2 are also read from
      data (all fixed at 0 in v2_SystemCharacteristics).

  step_env (uploaded / corrected version):
    - Action overrides (low-override, high-temp cutoff, humidity-forced vent,
      ventilation inertia) are resolved BEFORE temperature/humidity dynamics,
      so the power values that enter the dynamics equations are always correct.
    - Humidity overrule controller added: forces ventilation ON when H > H_high.
    - Ventilation inertia counter counts UP while vent is on; turn-off is blocked
      while 0 < counter <= vent_min_up_time.
    - Low-temperature override flags are updated AFTER dynamics (they take effect
      from the next step onward, not the current one).
"""


def reset_env(data, occupancy):
    """
    Initializes the environment state from a v2_SystemCharacteristics data dict.

    The caller must ensure data['price_previous'] is set appropriately:
      - evaluation environment  → already present as U[2,8] from get_fixed_data()
      - historical replay       → set data['price_previous'] = price_prev_from_csv
                                  before calling this function.
    """
    state = {
        # ── Fixed initial conditions (v2_SystemCharacteristics) ──────────
        "T1":              data["T1"],               # 21.0 °C
        "T2":              data["T2"],               # 21.0 °C
        "H":               data["H"],                # 40.0 %
        "vent_counter":    data["vent_counter"],     # 0
        "low_override_r1": data["low_override_r1"],  # 0
        "low_override_r2": data["low_override_r2"],  # 0

        # ── Not-fixed: drawn per day / from data ─────────────────────────
        "Occ1":            occupancy["Room1"][0],   # first-hour occupancy from data
        "Occ2":            occupancy["Room2"][0],
        "price_t":         data["price"][0],        # first hourly price
        "price_previous":  data["price_previous"],  # was hardcoded 0.0 — now from data

        "current_time":    0,
    }
    return state


def step_env(state, action, data, occupancy):
    """
    Advances the environment by one hour.

    Inputs:
        state    : current state dict
        action   : dict with keys "HeatPowerRoom1", "HeatPowerRoom2", "VentilationON"
        data     : system parameters (v2_SystemCharacteristics dict)
        occupancy: {"Room1": [...], "Room2": [...]}  per-hour occupancy lists

    Returns:
        new_state : updated state dict
        cost      : electricity cost for this step
        done      : True when the day (num_timeslots) is complete
    """

    # ── Extract current state ─────────────────────────────────────────────
    t              = state["current_time"]
    T1             = state["T1"]
    T2             = state["T2"]
    H              = state["H"]
    price_previous = state["price_t"]           # this step's price becomes next step's previous
    vent_counter   = state["vent_counter"]
    low_r1         = state["low_override_r1"]
    low_r2         = state["low_override_r2"]

    P1      = action["HeatPowerRoom1"]
    P2      = action["HeatPowerRoom2"]
    vent_on = action["VentilationON"]

    # Occupancy this timestep
    Occ1 = occupancy["Room1"][t]
    Occ2 = occupancy["Room2"][t]

    # =========================================================
    # ACTION OVERRIDES  (resolved before dynamics)
    # =========================================================

    # Low-temperature override: force heater to max while room is cold
    if low_r1 == 1:
        P1 = data['heating_max_power']
    if low_r2 == 1:
        P2 = data['heating_max_power']

    # High-temperature cutoff: kill heater if room is already too hot
    T_high = data['temp_max_comfort_threshold']
    if T1 > T_high:
        P1 = 0
    if T2 > T_high:
        P2 = 0

    # Humidity overrule controller: force ventilation ON when humidity too high
    if H > data['humidity_threshold']:
        vent_on = 1

    # Ventilation inertia: once started, must stay ON for vent_min_up_time hours
    if vent_on == 1:
        vent_counter += 1
    else:  # vent_on == 0 requested
        if 0 < vent_counter <= data['vent_min_up_time']:
            vent_on = 1          # override: cannot turn off yet
            vent_counter += 1

    # =========================================================
    # DYNAMICS
    # =========================================================

    # Temperature
    T1_new = (
        T1
        + data['heat_exchange_coeff']    * (T2 - T1)
        - data['thermal_loss_coeff']     * (T1 - data['outdoor_temperature'][t])
        + data['heating_efficiency_coeff'] * P1
        - data['heat_vent_coeff']        * vent_on
        + data['heat_occupancy_coeff']   * Occ1
    )

    T2_new = (
        T2
        + data['heat_exchange_coeff']    * (T1 - T2)
        - data['thermal_loss_coeff']     * (T2 - data['outdoor_temperature'][t])
        + data['heating_efficiency_coeff'] * P2
        - data['heat_vent_coeff']        * vent_on
        + data['heat_occupancy_coeff']   * Occ2
    )

    # Humidity
    H_new = (
        H
        + data['humidity_occupancy_coeff'] * (Occ1 + Occ2)
        - data['humidity_vent_coeff']      * vent_on
    )

    # =========================================================
    # LOW-TEMPERATURE OVERRIDE FLAGS  (updated after dynamics,
    # take effect from the NEXT step onward)
    # =========================================================
    T_low = data['temp_min_comfort_threshold']
    T_ok  = data['temp_OK_threshold']

    if T1_new < T_low:
        low_r1 = 1
    elif low_r1 == 1 and T1_new >= T_ok:
        low_r1 = 0

    if T2_new < T_low:
        low_r2 = 1
    elif low_r2 == 1 and T2_new >= T_ok:
        low_r2 = 0

    # =========================================================
    # COST
    # =========================================================
    price_t = data['price'][t]
    cost = price_t * (P1 + P2 + data['ventilation_power'] * vent_on)

    # =========================================================
    # NEW STATE
    # =========================================================
    new_state = {
        "T1":              T1_new,
        "T2":              T2_new,
        "H":               H_new,
        "Occ1":            Occ1,
        "Occ2":            Occ2,
        "price_t":         price_t,
        "price_previous":  price_previous,
        "vent_counter":    vent_counter,
        "low_override_r1": low_r1,
        "low_override_r2": low_r2,
        "current_time":    t + 1,
    }

    done = (t + 1) >= data['num_timeslots']

    return new_state, cost, done