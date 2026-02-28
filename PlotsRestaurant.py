# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 13:14:45 2025

@author: geots
"""

# -*- coding: utf-8 -*-
"""
Added unpacking of the variables for plots
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_HVAC_results(HVAC_results):


    # ---- unpack results ----
    Temp_r1 = HVAC_results["Temp_r1"]
    Temp_r2 = HVAC_results["Temp_r2"]
    h_r1    = HVAC_results["h_r1"]
    h_r2    = HVAC_results["h_r2"]
    v       = HVAC_results["v"]
    Hum     = HVAC_results["Hum"]
    price   = HVAC_results["price"]
    Occ_r1  = HVAC_results["Occ_r1"]
    Occ_r2  = HVAC_results["Occ_r2"]

    # Prefer time index from main if provided
    if "T" in HVAC_results and HVAC_results["T"] is not None:
        T = np.asarray(HVAC_results["T"])
    else:
        T = np.arange(len(Temp_r1))

    fig, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=True)

    # ---- Room Temperatures ----
    axes[0].plot(T, Temp_r1, label="Room 1 Temp", marker="o")
    axes[0].plot(T, Temp_r2, label="Room 2 Temp", marker="s")
    axes[0].axhline(18, linestyle="--", alpha=0.5)
    axes[0].axhline(20, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Temperature (°C)")
    axes[0].set_title("Room Temperatures")
    axes[0].legend()
    axes[0].grid(True)

    # ---- Heater consumption ----
    axes[1].bar(T, h_r1, width=0.4, label="Room 1 Heater", alpha=0.7)
    axes[1].bar(T, h_r2, width=0.4, bottom=h_r1, label="Room 2 Heater", alpha=0.7)
    axes[1].set_ylabel("Heater Power (kW)")
    axes[1].set_title("Heater Consumption")
    axes[1].legend()
    axes[1].grid(True)

    # ---- Ventilation and Humidity ----
    axes[2].step(T, v, where="mid", label="Ventilation ON")
    axes[2].plot(T, Hum, label="Humidity (%)", marker="o")
    axes[2].axhline(45, linestyle="--", alpha=0.5)
    axes[2].axhline(60, linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Ventilation / Humidity")
    axes[2].set_title("Ventilation Status and Humidity")
    axes[2].legend()
    axes[2].grid(True)

    # ---- Electricity price and occupancy ----
    axes[3].plot(T, price, label="TOU Price (€/kWh)", marker="x")
    axes[3].bar(T, Occ_r1, label="Occupancy Room 1", alpha=0.5)
    axes[3].bar(T, Occ_r2, bottom=Occ_r1, label="Occupancy Room 2", alpha=0.5)
    axes[3].set_ylabel("Price / Occupancy")
    axes[3].set_xlabel("Time (hours)")
    axes[3].set_title("Electricity Price and Occupancy")
    axes[3].legend()
    axes[3].grid(True)

    plt.tight_layout()
    plt.show()