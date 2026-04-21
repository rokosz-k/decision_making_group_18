import sys
import os
import numpy as np
import pandas as pd
import pyomo.environ as pyo
from sklearn.cluster import KMeans
from pyomo.environ import (
    ConcreteModel, Set, Param, Var, Binary, NonNegativeReals,
    Objective, Constraint, minimize, value
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from data.v2_SystemCharacteristics import get_fixed_data

# The state will be provided by the environment as the following dictionary

# state = {
#     "T1": ..., #Temperature of room 1
#     "T2": ..., #Temperature of room 2
#     "H": ..., #Humidity
#     "Occ1": ..., #Occupancy of room 1
#     "Occ2": ..., #Occupancy of room 2
#     "price_t": ..., #Price
#     "price_previous": ..., #Previous Price
#     "vent_counter": ..., #For how many consecutive hours has the ventilation been on 
#     "low_override_r1": ..., #Is the low-temperature overrule controller of room 1 active 
#     "low_override_r2": ..., #Is the low-temperature overrule controller of room 2 active 
#     "current_time": ... #What is the hour of the day
# }

class Node:
    """This is part of scpecifying the tree structure (Step 2) but used in Step 3+"""
    def __init__(self, value, cond_prob, parent, stage):
        self.value = value
        self.cond_prob = cond_prob
        self.parent = parent
        self.stage = stage
        self.children = []

    def path_from_root(self):
        """Return list of nodes from root down to this node."""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def scenario_probability(self):
        """
        p_omega = product of conditional probabilities along the root-to-leaf path.
        The root node has cond_prob=1 (probability 1 at initialization).
        """
        path = self.path_from_root()
        p_omega = 1.0
        for node in path:
            p_omega *= node.cond_prob
        return p_omega

def build_model(data: dict, state: dict) -> ConcreteModel:
    """
    Build the multi-stage stochastic MILP.

    Expected keys in `state`:
        current_time         : int, first time slot of the horizon
        T_init      : dict[(r, tau)] -> float, initial room temperatures
        H_init      : float, initial humidity at tau
        u_init      : dict[r] -> int {0,1}, overrule controller state before horizon
        c0          : int, consecutive hours ventilation was already on
        v_prev      : int {0,1}, ventilation state before horizon (= 1 if c0 > 0)
    Expected keys in `data`:
        L           : list[int], set of time slots in the horizon [tau, tau+1, ..., tau+K-1]
        R           : list[int], set of room indices, e.g. [1, 2]
        Omega       : list[int], set of scenario indices
        Omega_tw    : dict[(t, omega)] -> list[omega], indistinguishable scenarios
        pi          : dict[omega] -> float, scenario probabilities
        price       : dict[(omega, t)] -> float, electricity price
        occ         : dict[(omega, r, t)] -> float, occupancy
    """
    m = ConcreteModel()

    # ------------------------------------------------------------------
    # Sets
    # ------------------------------------------------------------------
    tau = state["current_time"]
    L   = data["L"]          # list of time slots
    R   = data["R"]
    Om  = data["Omega"]

    m.L     = Set(initialize=L)
    m.L_not_tau = Set(initialize=[t for t in L if t != tau])
    m.R     = Set(initialize=R)
    m.Omega = Set(initialize=Om)

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    sys_characteristics = get_fixed_data()

    m.pi       = Param(m.Omega, initialize=data["pi"])
    m.price    = Param(m.Omega, m.L, initialize=data["price"])
    m.occ      = Param(m.Omega, m.R, m.L, initialize=data["occ"])
    m.T_out    = Param(m.L, initialize=sys_characteristics["outdoor_temperature"][tau - 1])
    m.P_bar    = Param(m.R, initialize=sys_characteristics["heating_max_power"])
    m.P_vent   = Param(initialize=sys_characteristics["ventilation_power"])

    m.T_low    = Param(initialize=sys_characteristics["temp_min_comfort_threshold"])
    m.T_ok     = Param(initialize=sys_characteristics["temp_OK_threshold"])
    m.T_high   = Param(initialize=sys_characteristics["temp_max_comfort_threshold"])
    m.H_high   = Param(initialize=sys_characteristics["humidity_threshold"])
    m.U_vent   = Param(initialize=sys_characteristics["vent_min_up_time"], within=pyo.NonNegativeIntegers)
    m.M_temp   = Param(initialize=1e6)
    m.M_hum    = Param(initialize=1e6)

    m.zeta_exch = Param(initialize=sys_characteristics["heat_exchange_coeff"])
    m.zeta_loss = Param(initialize=sys_characteristics["thermal_loss_coeff"])
    m.zeta_conv = Param(initialize=sys_characteristics["heating_efficiency_coeff"])
    m.zeta_cool = Param(initialize=sys_characteristics["heat_vent_coeff"])
    m.zeta_occ  = Param(initialize=sys_characteristics["heat_occupancy_coeff"])
    m.eta_occ   = Param(initialize=sys_characteristics["humidity_occupancy_coeff"])
    m.eta_vent  = Param(initialize=sys_characteristics["humidity_vent_coeff"])

    # Initial conditions
    m.T_init = Param(m.R, initialize={1: state["T1"], 2: state["T2"]})
    m.H_init   = Param(initialize=state["H"])
    m.u_init   = Param(m.R, initialize={1: state["low_override_r1"], 2: state["low_override_r2"]}, within=pyo.Binary)
    m.c0       = Param(initialize=state["vent_counter"], within=pyo.NonNegativeIntegers)
    m.v_prev = Param(initialize=1 if state["vent_counter"] > 0 else 0, within=pyo.Binary)

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------
    m.p       = Var(m.Omega, m.R, m.L, domain=NonNegativeReals)
    m.v       = Var(m.Omega, m.L, domain=Binary)
    m.T       = Var(m.Omega, m.R, m.L, domain=pyo.Reals)
    m.H       = Var(m.Omega, m.L, domain=pyo.Reals)
    m.s       = Var(m.Omega, m.L, domain=Binary)
    m.u       = Var(m.Omega, m.R, m.L, domain=Binary)
    m.y_low   = Var(m.Omega, m.R, m.L, domain=Binary)
    m.y_ok    = Var(m.Omega, m.R, m.L, domain=Binary)
    m.y_high  = Var(m.Omega, m.R, m.L, domain=Binary)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    m.obj = Objective(
        expr=sum(
            m.pi[om] * sum(
                m.price[om, t] * (
                    m.P_vent * m.v[om, t] +
                    sum(m.p[om, r, t] for r in m.R)
                )
                for t in m.L
            )
            for om in m.Omega
        ),
        sense=minimize
    )

    # ------------------------------------------------------------------
    # Helper: get T or H at t-1, using initial condition at tau
    # ------------------------------------------------------------------
    def T_prev(om, r, t):
        return m.T_init[r] if t == tau else m.T[om, r, t - 1]

    def H_prev(om, t):
        return m.H_init if t == tau else m.H[om, t - 1]

    def v_prev_val(om, t):
        return value(m.v_prev) if t == tau else m.v[om, t - 1]

    # ------------------------------------------------------------------
    # Fix initial temperature and humidity at tau
    # ------------------------------------------------------------------
    def fix_T_init_rule(m, om, r):
        return m.T[om, r, tau] == m.T_init[r]
    m.fix_T_init = Constraint(m.Omega, m.R, rule=fix_T_init_rule)

    def fix_H_init_rule(m, om):
        return m.H[om, tau] == m.H_init
    m.fix_H_init = Constraint(m.Omega, rule=fix_H_init_rule)

    # ------------------------------------------------------------------
    # Temperature Dynamics  (t in L \ {tau})
    # ------------------------------------------------------------------
    def temp_dynamics_rule(m, om, r, t):
        r_other = [rr for rr in R if rr != r][0]   # the other room
        return m.T[om, r, t] == (
            T_prev(om, r, t)
            + m.zeta_exch * (T_prev(om, r_other, t) - T_prev(om, r, t))
            - m.zeta_loss * (T_prev(om, r, t) - m.T_out[t - 1])
            + m.zeta_conv * m.p[om, r, t - 1]
            - m.zeta_cool * v_prev_val(om, t)
            + m.zeta_occ  * m.occ[om, r, t - 1]
        )
    m.temp_dynamics = Constraint(m.Omega, m.R, m.L_not_tau, rule=temp_dynamics_rule)

    # ------------------------------------------------------------------
    # Humidity Dynamics  (t in L \ {tau})
    # ------------------------------------------------------------------
    def hum_dynamics_rule(m, om, t):
        return m.H[om, t] == (
            H_prev(om, t)
            + m.eta_occ  * sum(m.occ[om, r, t - 1] for r in m.R)
            - m.eta_vent * v_prev_val(om, t)
        )
    m.hum_dynamics = Constraint(m.Omega, m.L_not_tau, rule=hum_dynamics_rule)

    # ------------------------------------------------------------------
    # Heating Power Limits
    # ------------------------------------------------------------------
    def power_ub_rule(m, om, r, t):
        return m.p[om, r, t] <= m.P_bar[r]
    m.power_ub = Constraint(m.Omega, m.R, m.L, rule=power_ub_rule)
    # lower bound (>= 0) already enforced by NonNegativeReals domain

    # ------------------------------------------------------------------
    # High-Temperature Cutoff
    # ------------------------------------------------------------------
    def high_lb_rule(m, om, r, t):
        return m.T[om, r, t] >= m.T_high - m.M_temp * (1 - m.y_high[om, r, t])
    m.high_lb = Constraint(m.Omega, m.R, m.L, rule=high_lb_rule)

    def high_ub_rule(m, om, r, t):
        return m.T[om, r, t] <= m.T_high + m.M_temp * m.y_high[om, r, t]
    m.high_ub = Constraint(m.Omega, m.R, m.L, rule=high_ub_rule)

    def heater_off_rule(m, om, r, t):
        return m.p[om, r, t] <= m.P_bar[r] * (1 - m.y_high[om, r, t])
    m.heater_off = Constraint(m.Omega, m.R, m.L, rule=heater_off_rule)

    # ------------------------------------------------------------------
    # Low-Temperature Detection
    # ------------------------------------------------------------------
    def low_ub_rule(m, om, r, t):
        return m.T[om, r, t] <= m.T_low + m.M_temp * (1 - m.y_low[om, r, t])
    m.low_ub = Constraint(m.Omega, m.R, m.L, rule=low_ub_rule)

    def low_lb_rule(m, om, r, t):
        return m.T[om, r, t] >= m.T_low - m.M_temp * m.y_low[om, r, t]
    m.low_lb = Constraint(m.Omega, m.R, m.L, rule=low_lb_rule)

    # ------------------------------------------------------------------
    # OK-Temperature Detection
    # ------------------------------------------------------------------
    def ok_lb_rule(m, om, r, t):
        return m.T[om, r, t] >= m.T_ok - m.M_temp * (1 - m.y_ok[om, r, t])
    m.ok_lb = Constraint(m.Omega, m.R, m.L, rule=ok_lb_rule)

    def ok_ub_rule(m, om, r, t):
        return m.T[om, r, t] <= m.T_ok + m.M_temp * m.y_ok[om, r, t]
    m.ok_ub = Constraint(m.Omega, m.R, m.L, rule=ok_ub_rule)

    # ------------------------------------------------------------------
    # Overrule Controller — Activation
    # ------------------------------------------------------------------
    # (12): u >= y_low, for all t
    def overrule_on_rule(m, om, r, t):
        return m.u[om, r, t] >= m.y_low[om, r, t]
    m.overrule_on = Constraint(m.Omega, m.R, m.L, rule=overrule_on_rule)

    # (13): u <= u_prev + y_low, for t > tau
    def overrule_persist_rule(m, om, r, t):
        return m.u[om, r, t] <= m.u[om, r, t - 1] + m.y_low[om, r, t]
    m.overrule_persist = Constraint(m.Omega, m.R, m.L_not_tau, rule=overrule_persist_rule)

    # (13 init): u_tau <= u_init + y_low_tau
    def overrule_persist_init_rule(m, om, r):
        return m.u[om, r, tau] <= m.u_init[r] + m.y_low[om, r, tau]
    m.overrule_persist_init = Constraint(m.Omega, m.R, rule=overrule_persist_init_rule)

    # (14): p >= P_bar * u
    def heater_max_rule(m, om, r, t):
        return m.p[om, r, t] >= m.P_bar[r] * m.u[om, r, t]
    m.heater_max = Constraint(m.Omega, m.R, m.L, rule=heater_max_rule)

    # ------------------------------------------------------------------
    # Overrule Controller — Deactivation
    # ------------------------------------------------------------------
    # (15): u >= u_prev - y_ok, for t > tau
    def overrule_off_lb_rule(m, om, r, t):
        return m.u[om, r, t] >= m.u[om, r, t - 1] - m.y_ok[om, r, t]
    m.overrule_off_lb = Constraint(m.Omega, m.R, m.L_not_tau, rule=overrule_off_lb_rule)

    # (15 init): u_tau >= u_init - y_ok_tau
    def overrule_off_lb_init_rule(m, om, r):
        return m.u[om, r, tau] >= m.u_init[r] - m.y_ok[om, r, tau]
    m.overrule_off_lb_init = Constraint(m.Omega, m.R, rule=overrule_off_lb_init_rule)

    # (16): u <= 1 - y_ok, for all t
    def overrule_off_ub_rule(m, om, r, t):
        return m.u[om, r, t] <= 1 - m.y_ok[om, r, t]
    m.overrule_off_ub = Constraint(m.Omega, m.R, m.L, rule=overrule_off_ub_rule)

    # ------------------------------------------------------------------
    # Ventilation Startup
    # ------------------------------------------------------------------
    # (17): s >= v - v_prev, for t > tau
    def startup1_rule(m, om, t):
        return m.s[om, t] >= m.v[om, t] - m.v[om, t - 1]
    m.startup1 = Constraint(m.Omega, m.L_not_tau, rule=startup1_rule)

    # (17 init): s_tau >= v_tau - v_prev
    def startup1_init_rule(m, om):
        return m.s[om, tau] >= m.v[om, tau] - m.v_prev
    m.startup1_init = Constraint(m.Omega, rule=startup1_init_rule)

    # (18): s <= v, for all t
    def startup2_rule(m, om, t):
        return m.s[om, t] <= m.v[om, t]
    m.startup2 = Constraint(m.Omega, m.L, rule=startup2_rule)

    # (19): s <= 1 - v_prev, for t > tau
    def startup3_rule(m, om, t):
        return m.s[om, t] <= 1 - m.v[om, t - 1]
    m.startup3 = Constraint(m.Omega, m.L_not_tau, rule=startup3_rule)

    # (19 init): s_tau <= 1 - v_prev
    def startup3_init_rule(m, om):
        return m.s[om, tau] <= 1 - m.v_prev
    m.startup3_init = Constraint(m.Omega, rule=startup3_init_rule)

    # ------------------------------------------------------------------
    # Minimum Up-Time — startups within horizon  (20)
    # ------------------------------------------------------------------
    horizon_end = tau + len(L) - 1   # last time slot index

    def min_uptime_rule(m, om, t):
        window_end = min(t + value(m.U_vent) - 1, horizon_end)
        rhs_count  = min(value(m.U_vent), horizon_end - t + 1)
        return (
            sum(m.v[om, t_] for t_ in range(t, window_end + 1))
            >= rhs_count * m.s[om, t]
        )
    m.min_uptime = Constraint(m.Omega, m.L, rule=min_uptime_rule)

    # ------------------------------------------------------------------
    # Minimum Up-Time — carry-over from before horizon  (20 init)
    # ------------------------------------------------------------------
    remaining = max(value(m.U_vent) - value(m.c0), 0)
    remaining = min(remaining, len(L))   # clip to horizon length

    if remaining > 0:
        window_end_init = tau + remaining - 1

        def min_uptime_init_rule(m, om):
            return (
                sum(m.v[om, t_] for t_ in range(tau, window_end_init + 1))
                >= remaining * m.v_prev
            )
        m.min_uptime_init = Constraint(m.Omega, rule=min_uptime_init_rule)

    # ------------------------------------------------------------------
    # Humidity-Triggered Ventilation  (21)
    # ------------------------------------------------------------------
    def hum_vent_rule(m, om, t):
        return m.H[om, t] <= m.H_high + m.M_hum * m.v[om, t]
    m.hum_vent = Constraint(m.Omega, m.L, rule=hum_vent_rule)

    # ------------------------------------------------------------------
    # Non-Anticipativity Constraints
    # ------------------------------------------------------------------
    Omega_tw = data["Omega_tw"]   # dict[(t, omega)] -> list of indistinguishable scenarios

    def nac_p_rule(m, om, r, t):
        return [
            m.p[om, r, t] == m.p[om2, r, t]
            for om2 in Omega_tw[(t, om)] if om2 != om
        ]

    def nac_v_rule(m, om, t):
        return [
            m.v[om, t] == m.v[om2, t]
            for om2 in Omega_tw[(t, om)] if om2 != om
        ]

    # Build NAC constraints explicitly to avoid duplicate pairs
    nac_p_constraints = {}
    nac_v_constraints = {}
    for t in L:
        for om in Om:
            for om2 in Omega_tw[(t, om)]:
                if om2 > om:   # avoid duplicates (om, om2) and (om2, om)
                    for r in R:
                        nac_p_constraints[(om, om2, r, t)] = (
                            m.p[om, r, t] == m.p[om2, r, t]
                        )
                    nac_v_constraints[(om, om2, t)] = (
                        m.v[om, t] == m.v[om2, t]
                    )

    m.nac_p = Constraint(
        list(nac_p_constraints.keys()),
        rule=lambda m, om, om2, r, t: nac_p_constraints[(om, om2, r, t)]
    )
    m.nac_v = Constraint(
        list(nac_v_constraints.keys()),
        rule=lambda m, om, om2, t: nac_v_constraints[(om, om2, t)]
    )

    return m

def sample(S, t, PRICE_FILE, OCC1_FILE, OCC2_FILE):
    """
    Monte Carlo Sampling of State Variable Combinations
    ---
    State variables: electricity price, occupancy room 1, occupancy room 2
    S: Number of Monte Carlo scenarios
    t: Timeslot of interest (1-10, corresponds to hour column)
    Data: 100 days x 10 hourly timeslots
    Output: S scenario combinations for a given timeslot t
    """
    SEED = 42     # for reproducibility

    # --- Load data ---
    # Price file: first column is "previous price", columns 1–10 are hourly timeslots
    price_df = pd.read_csv(PRICE_FILE, header=0)
    price_df.columns = ["t0"] + [f"t{i}" for i in range(1, 11)]   # t0 = prev price

    # Occupancy files: 10 columns corresponding to timeslots 1–10 (0-indexed as 0–9)
    occ1_df = pd.read_csv(OCC1_FILE, header=0)
    occ2_df = pd.read_csv(OCC2_FILE, header=0)
    occ1_df.columns = [f"t{i}" for i in range(1, 11)]
    occ2_df.columns = [f"t{i}" for i in range(1, 11)]

    # ── Extract the empirical distribution for timeslot t ────────────────────────
    # Each column holds 100 observed values (one per day) → empirical population
    price_at_t = price_df[f"t{t}"].values   # shape (100,)
    occ1_at_t  = occ1_df[f"t{t}"].values   # shape (100,)
    occ2_at_t  = occ2_df[f"t{t}"].values   # shape (100,)

    # ── Monte Carlo sampling (independent, with replacement) ─────────────────────
    rng = np.random.default_rng(SEED)

    sampled_price = rng.choice(price_at_t, size=S, replace=True)
    sampled_occ1  = rng.choice(occ1_at_t,  size=S, replace=True)
    sampled_occ2  = rng.choice(occ2_at_t,  size=S, replace=True)

    # ── Assemble results ──────────────────────────────────────────────────────────
    # Stack into (S, 3) — each row is one [price, occ1, occ2] scenario
    return np.column_stack([sampled_price, sampled_occ1, sampled_occ2])

def build_scenario_tree(state, L, K, S, PRICE_FILE, OCC1_FILE, OCC2_FILE):
    """
    state : ...
    L     : number of stages to branch forward
    S     : number of samples
    K     : number of clusters (branches per node)
    
    Returns ...
    """
    # --- Initialize: root node at stage tau with probability 1 ---
    root = Node(value=[state["price_t"],state["Occ1"],state["Occ2"]], cond_prob=1.0, parent=None, stage=state["current_time"])
    current_level = [root]

    # --- Build the tree ---
    for t in range(state["current_time"] + 1, state["current_time"] + L):
        next_level = []

        # --- Branching out: for each node n at stage t ---
        for node in current_level:

            # Generate S samples from this node
            samples = sample(S, t, PRICE_FILE, OCC1_FILE, OCC2_FILE)
            sample_probs = np.full(S, 1.0 / S) # Each sample gets probability 1/S

            # --- Clustering (stage-wise): cluster into K clusters ---
            k_actual = min(K, S)  # can't have more clusters than samples
            kmeans = KMeans(n_clusters=k_actual, random_state=42, n_init="auto")
            labels = kmeans.fit_predict(samples)
            centroids = kmeans.cluster_centers_

            # --- For each cluster k: create child node ---
            for k in range(k_actual):
                mask = (labels == k)
                if not np.any(mask):
                    continue

                # p(k|n) = sum of member probabilities
                cond_prob = np.sum(sample_probs[mask])
                child = Node(
                    value=centroids[k], # [price, occ1, occ2]
                    cond_prob=cond_prob,
                    parent=node,
                    stage=t
                )
                node.children.append(child)
                next_level.append(child)

        current_level = next_level
        #print(f"Stage t={t}: {len(next_level)} nodes")

    # First product
    leaves = current_level # Leaves are the nodes in the final stage
    
    # --- Get probabilities dictionary ---
    def get_probabilities(leaves):
        probabilities = {}
        for i, node in enumerate(leaves):
            probabilities[i+1] = node.scenario_probability()
        return probabilities
    
    # --- Build forecast dictionaries by walking each scenario's path from root ---
    def get_forecasts(leaves):
        price = {}
        occ = {}
        for i, node in enumerate(leaves):
            path = node.path_from_root()
            for node2 in path:
                price[(i + 1, node2.stage)] = node2.value[0]
                occ[(i + 1, 1, node2.stage)] = node2.value[1]
                occ[(i + 1, 2, node2.stage)] = node2.value[2]
        return price, occ

    probabilities = get_probabilities(leaves)
    price, occ = get_forecasts(leaves)
    return root, leaves, probabilities, (price, occ)

def build_OMEGA_tw(state, leaves, OMEGA, OMEGA_set, L_set):
    """"""
    OMEGA_tw = {}

    def get_leaf_indices(node, scenarios_nodes):
        """Recursively collect all leaf scenario indices reachable from the node."""
        if not node.children:
            return [scenarios_nodes.index(node)+1]
        result = []
        for child in node.children:
            result.extend(get_leaf_indices(child, scenarios_nodes))
        return result

    # Initialize OMEGA_tw for t = tau to save computational time (first stage node is common for all scenarios)
    for s in range(1, OMEGA + 1): 
        OMEGA_tw[state["current_time"], s] = list(OMEGA_set)

    for t in L_set[1:]: # Rest of the stages
        for s in OMEGA_set:
            node = leaves[s-1]   # Initialize the leaf node of scenario s
            while node.stage != t:  # Walk back up to stage t 
                node = node.parent
            end_scenarios = get_leaf_indices(node,leaves)
            end_scenarios.remove(s)
            OMEGA_tw[t, s] = end_scenarios
            # Print check
            #print(f"For t={t} s={s}, {len(end_scenarios)} scenarios share this node in their path (excluding s).")  
    return OMEGA_tw

def solve_model(m):
    """Build and solve the model using Gurobi."""
    solver = pyo.SolverFactory("gurobi")
    solver.options["MIPGap"]   = 1e-4
    solver.options["TimeLimit"] = 300     # seconds
    solver.options["OutputFlag"] = False

    results = solver.solve(m, tee=False)

    if results.solver.termination_condition == pyo.TerminationCondition.optimal:
        print(f"Optimal cost: {value(m.obj):.4f}")
    else:
        print(f"Solver status: {results.solver.termination_condition}")

    return m, results

def select_action(state):
    # Files used for sampling
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) # 2 folders up
    PRICE_FILE = os.path.join(ROOT_DIR, 'data', 'v2_PriceData.csv')
    OCC1_FILE = os.path.join(ROOT_DIR, "data", "OccupancyRoom1.csv")
    OCC2_FILE = os.path.join(ROOT_DIR, "data", "OccupancyRoom2.csv")
    K = 10 # Simulation time horizon
    R = [1,2] # Set of rooms

    # 1. Specify the Lookahead Horizon Length
    L = 3   # Lookahead horizon length
    B = 3   # Branching factor
    S = 10  # Number of samples

    # 2. Specify the structure of the scenario tree
    L = min(L, K - state["current_time"] + 1)   # Update lookahead horizon length for the current timeslot
    L_set = list(range(state["current_time"], state["current_time"] + L - 1 + 1))  # Lookahead Horizon (e.g. {1, 2, 3, 4})
    # print("L =", L_set)
    # print(f"Lookahead time horizon length is {L}.")
    
    OMEGA = pow(B,L-1)  # Number of scenarios generated
    OMEGA_set = list(range(1, OMEGA+1)) # Set of scenarios
    # print("There are", OMEGA, "scenarios generated in total.")

    # 3. Create scenarios and probabilities
    root, leaves, probabilities, (price, occ) = build_scenario_tree(state, L, B, S, PRICE_FILE, OCC1_FILE, OCC2_FILE)

    # sanity check
    total_prob = sum(probabilities.values())
    # print(f"\nSum of all p_omega: {total_prob:.6f}  (should be ≈ 1.0)")

    # 4. Create non-anticipativity sets (OMEGA_tw set)
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
    
    # --- Build and solve the optimization model ---
    m = build_model(data, state)
    m, results = solve_model(m)
    #print(results)

    # --- Results DataFrame ---
    def results_to_dataframe(m) -> pd.DataFrame:
        rows = []
        for t in sorted(m.L):
            for om in m.Omega:
                rows.append({
                    "omega": om,
                    "t":     t,
                    "p_r1":  value(m.p[om, 1, t]),
                    "p_r2":  value(m.p[om, 2, t]),
                    "v":     int(value(m.v[om, t])),
                })

        df = pd.DataFrame(rows).set_index(["t", "omega"])
        return df
    results = results_to_dataframe(m)

    # --- Print results ---
    def print_example_solution(m):
        # Pick any one scenario to read the here-and-now decision at tau
        print("\n=== Heating Power (scenario 1) ===")
        for r in m.R:
            vals = [f"t={t}: {value(m.p[1, r, t]):.2f}" for t in m.L]
            print(f"  Room {r}: {', '.join(vals)}")

        print("\n=== Ventilation (scenario 1) ===")
        vals = [f"t={t}: {int(value(m.v[1, t]))}" for t in m.L]
        print(f"  {', '.join(vals)}")

    #print_example_solution(m)  # Compact results
    # print(results)              # Full results
    #results.to_csv("results.csv") # Save a file

    HereAndNowActions = {
    "HeatPowerRoom1" : value(m.p[1, 1, state["current_time"]]),
    "HeatPowerRoom2" : value(m.p[1, 2, state["current_time"]]),
    "VentilationON" : value(m.v[1, state["current_time"]])
    }
    
    return HereAndNowActions


# Example use (to be deleted)
state = {
    "T1": 21, #Temperature of room 1
    "T2": 21, #Temperature of room 2
    "H": 40, #Humidity
    "Occ1": 34.098941577065816, #Occupancy of room 1
    "Occ2": 19.416994739003783, #Occupancy of room 2
    "price_t": 4.311188506349689, #Price
    "price_previous": 5.831303587171684, #Previous Price
    "vent_counter": 0, #For how many consecutive hours has the ventilation been on 
    "low_override_r1": 0, #Is the low-temperature overrule controller of room 1 active 
    "low_override_r2": 0, #Is the low-temperature overrule controller of room 2 active 
    "current_time": 1 #What is the hour of the day
}