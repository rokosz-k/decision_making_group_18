# 02435 Decision-Making Under Uncertainty – Group 18

This repository contains the code and data for the course assignment for *02435 Decision-Making Under Uncertainty*, Spring 2026, at the Technical University of Denmark (DTU).

> Repository made public on **02.03.2026 at 11:59 PM**.

---

## Project Overview

The project tackles **HVAC control optimization** for a two-room restaurant. The goal is to minimize daily electricity costs while satisfying thermal comfort and ventilation constraints, under uncertain electricity prices and occupancy levels.

**System setup:**
- 2 heating zones with continuous power control (0–3 kW each)
- 1 ventilation unit (binary ON/OFF, 0.5 kW)
- 1 humidity sensor with automatic safety override
- 10 hourly decision steps per day, evaluated over 100 historical days

**Uncertainties:**
- Electricity prices (mean-reverting AR(1) process)
- Room occupancy levels (coupled Markovian process, 2 rooms)

---

## Repository Structure

```
decision_making_group_18/
├── data/
│   ├── v2_PriceData.csv          # 100-day historical electricity prices (DKK/kWh)
│   ├── OccupancyRoom1.csv        # 100-day occupancy for room 1
│   └── OccupancyRoom2.csv        # 100-day occupancy for room 2
│
├── part_A/                        # Part A: Stochastic modelling & basic optimization
│
└── part_B/                        # Part B: Online control policies
    ├── RestaurantEnv.py           # Environment simulator (dynamics, cost, safety overrides)
    ├── PriceProcessRestaurant.py  # Stochastic price process model
    ├── OccupancyProcessRestaurant.py  # Stochastic occupancy process model
    ├── running_script.py          # Main evaluation loop (100 days)
    ├── dummy_policy.py            # Baseline: no-action policy
    ├── Optimal_in_Hindsight_Solution.py  # Oracle benchmark (full-day MILP)
    ├── requirements.txt           # Python dependencies
    │
    ├── policies/
    │   ├── ADP_policy_9_features_18.py  # ADP one-step lookahead (main policy)
    │   ├── SP_policy_18.py              # Multi-stage stochastic programming
    │   └── hybrid_policy_18.py          # Hybrid: ADP with dummy bookends
    │
    └── ADP_training/
        ├── collect_samples.py            # Generate training trajectories
        ├── train_adp_fvi_9_features.py   # Fitted Value Iteration training
        └── eta_sp_fvi_v3.pkl             # Trained ADP weights (currently used)
```

---

## Part A

Part A covers stochastic modelling and scenario-based optimization. See [part_A/](part_A/) for details.

**Tech stack:**
- **NumPy** – numerical computations
- **pandas** – data handling
- **Gurobi (gurobipy)** – optimization modeling and solving
- **Matplotlib** – data visualization

---

## Part B

### Problem

At each of the 10 hourly steps in a day, a controller observes the current state and selects heating powers (P1, P2) and ventilation (vent). Safety overrule controllers enforce:
- Heating forced ON if room temperature drops below 18°C
- Heating forced OFF above 25°C
- Ventilation forced ON if humidity exceeds 60%
- Ventilation must stay ON for at least 2 consecutive hours once activated

The objective is to minimize expected total electricity cost per day.

### Policies Implemented

| Policy | Description | Solver |
|--------|-------------|--------|
| **Dummy** | No action; relies entirely on safety overrides | — |
| **SP** | Multi-stage MILP over a receding horizon (L=5, K=2 branches) | Gurobi |
| **ADP** | One-step lookahead MILP with a learned linear value function | Gurobi |
| **Hybrid** | ADP for hours 1–8, dummy for hour 0 and 9 | Gurobi |
| **OiH** | Optimal-in-Hindsight oracle (full-day MILP with perfect information) | Gurobi |

### ADP Value Function

The ADP policy approximates the cost-to-go as a linear function of a 9-dimensional feature vector:

```
φ(state) = [T1, T2, H, price_t, price_previous, vent_counter,
             low_override_r1, low_override_r2, bias]
```

Weights `η_t` (one per timestep) are trained offline via **Fitted Value Iteration (FVI)**:

1. Collect sample trajectories using an existing policy (SP or dummy)
2. For each iteration and each timestep (backward):
   - Sample states, solve one-step lookahead MILPs to get Bellman targets
   - Fit `η_t` via Ridge regression
3. Save weights to `eta_sp_fvi_v3.pkl`

### Expected Performance (100-day evaluation)

| Policy | Avg. Daily Cost | Notes |
|--------|-----------------|-------|
| Dummy | ~5.0 DKK | No control; relies on overrides |
| SP | ~3.8–4.0 DKK | Multi-stage scenario tree |
| ADP | ~3.7–3.9 DKK | Trained on SP trajectories |
| Hybrid | ~3.8–4.0 DKK | Avoids cold-start bias |
| OiH (oracle) | ~3.2–3.5 DKK | Perfect-information upper bound |

---

## Installation

```bash
pip install -r part_B/requirements.txt
```

**Dependencies:**

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 2.2.6 | Numerical computations |
| pandas | 2.3.3 | Data handling |
| matplotlib | 3.10.9 | Visualization |
| pyomo | 6.10.0 | Algebraic modeling (MILP) |
| scikit-learn | 1.7.2 | Feature scaling, K-means clustering |
| scipy | 1.15.3 | Scientific computing |

> **Gurobi is required** for all non-dummy policies. A free academic license is available at [gurobi.com](https://www.gurobi.com/academia/academic-program-and-licenses/).

---

## Usage

### Evaluate a policy over 100 days

```bash
cd part_B
python running_script.py
```

To switch policies, edit the import at the bottom of `running_script.py`:

```python
# Swap in any of these:
from policies.ADP_policy_9_features_18 import select_action   # ADP (default)
from policies.SP_policy_18 import select_action                # Stochastic Programming
from policies.hybrid_policy_18 import select_action            # Hybrid
from dummy_policy import select_action                         # Baseline
```

### Retrain ADP weights

```bash
cd part_B/ADP_training
python collect_samples.py          # Generate training data
python train_adp_fvi_9_features.py # Train and save eta_sp_fvi_v3.pkl
```
