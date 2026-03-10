import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. SYSTEM PARAMETERS (From SystemCharacteristics)
# ==========================================
def get_fixed_data():
    """
    Returns the fixed data for the heating + ventilation system.
    """
    num_timeslots = 10
    return {
        'num_timeslots': num_timeslots,
        'initial_temperature': 21.0,
        'initial_humidity': 40.0,
        
        'heating_max_power': 3.0,
        'heat_exchange_coeff': 0.6,
        'heating_efficiency_coeff': 1.0,
        'thermal_loss_coeff': 0.1,
        'heat_vent_coeff': 0.7,
        'heat_occupancy_coeff': 0.02,
        
        'temp_min_comfort_threshold': 18.0,
        'temp_OK_threshold': 22.0,
        'temp_max_comfort_threshold': 26.0,
        
        'outdoor_temperature': [
            3 * np.sin(2 * np.pi * t / num_timeslots - np.pi/2)
            for t in range(num_timeslots)
        ],
        
        'vent_min_up_time': 3,
        'humidity_threshold': 70.0,
        'ventilation_power': 2.0,
        'humidity_occupancy_coeff': 0.18,
        'humidity_vent_coeff': 15
    }

# ==========================================
# 2. DUMMY POLICY (From PolicyRestaurant2 template)
# ==========================================
def dummy_policy(state):
    """
    Dummy policy: never turns on heating or ventilation voluntarily.
    It relies 100% on the Overrule Controllers to maintain constraints.
    """
    HereAndNowActions = {
        "HeatPowerRoom1": 0.0, 
        "HeatPowerRoom2": 0.0, 
        "VentilationON": 0 
    }
    return HereAndNowActions

# ==========================================
# 3. SIMULATION ENVIRONMENT
# ==========================================
def run_simulation(policy_func, prices_df, occ1_df, occ2_df, params):
    num_days = prices_df.shape[0]
    num_hours = params['num_timeslots']
    
    daily_costs = []
    
    for day in range(num_days):
        # State initialization at the beginning of the day (t=0)
        T1 = params['initial_temperature']
        T2 = params['initial_temperature']
        H = params['initial_humidity']
        
        # Flags for tracking the status of overrule controllers and inertia
        low_override_r1 = False
        low_override_r2 = False
        vent_counter = 0  # Counter for consecutive hours the ventilation is ON
        
        cost_for_day = 0.0
        
        for t in range(num_hours):
            # Read external data for the current hour
            price_t = prices_df.iloc[day, t]
            # No previous price at t=0, so we just use the current one
            price_prev = prices_df.iloc[day, t-1] if t > 0 else price_t  
            occ1_t = occ1_df.iloc[day, t]
            occ2_t = occ2_df.iloc[day, t]
            T_out_t = params['outdoor_temperature'][t]
            
            # --- 3.1. Build the state dictionary for the Policy ---
            state = {
                "T1": T1,
                "T2": T2,
                "H": H,
                "Occ1": occ1_t,
                "Occ2": occ2_t,
                "price_t": price_t,
                "price_previous": price_prev,
                "vent_counter": vent_counter,
                "low_override_r1": low_override_r1,
                "low_override_r2": low_override_r2,
                "current_time": t
            }
            
            # --- 3.2. Execute the policy ---
            actions = policy_func(state)
            
            # Extract the policy's intended actions (clipped to physical limits just in case)
            p1_policy = np.clip(actions.get("HeatPowerRoom1", 0), 0, params['heating_max_power'])
            p2_policy = np.clip(actions.get("HeatPowerRoom2", 0), 0, params['heating_max_power'])
            v_policy = 1 if actions.get("VentilationON", 0) > 0.5 else 0
            
            # --- 3.3. Apply Overrule Controllers & Inertia ---
            v_actual = v_policy
            p1_actual = p1_policy
            p2_actual = p2_policy
            
            # A. Humidity Overrule Controller
            # Forces ventilation ON if humidity exceeds the threshold
            if H > params['humidity_threshold']:
                v_actual = 1
                
            # B. Ventilation Inertia
            # If ventilation is currently ON but has been running for less than the minimum required time (3h), it MUST remain ON
            if 0 < vent_counter < params['vent_min_up_time']:
                v_actual = 1
            
            # Update the ventilation operation counter for the next hour's state
            if v_actual == 1:
                vent_counter += 1
            else:
                vent_counter = 0

            # C. Temperature Overrule Controllers
            # ROOM 1
            if low_override_r1:
                # If emergency heating is active, it stops only when T > T_OK
                if T1 > params['temp_OK_threshold']:
                    low_override_r1 = False
            else:
                # If temperature drops below the minimum limit, activate emergency heating
                if T1 < params['temp_min_comfort_threshold']:
                    low_override_r1 = True
            
            # Apply actual power based on the overrule status
            if low_override_r1:
                p1_actual = params['heating_max_power']
            elif T1 > params['temp_max_comfort_threshold']:
                p1_actual = 0.0  # Force heater OFF for this hour

            # ROOM 2
            if low_override_r2:
                if T2 > params['temp_OK_threshold']:
                    low_override_r2 = False
            else:
                if T2 < params['temp_min_comfort_threshold']:
                    low_override_r2 = True
            
            if low_override_r2:
                p2_actual = params['heating_max_power']
            elif T2 > params['temp_max_comfort_threshold']:
                p2_actual = 0.0

            # --- 3.4. Calculate costs using the actual (overruled) actions ---
            cost_for_day += price_t * (p1_actual + p2_actual + v_actual * params['ventilation_power'])
            
            # --- 3.5. System state update (Dynamics for step t+1) ---
            T1_next = (T1 
                       + params['heat_exchange_coeff'] * (T2 - T1) 
                       + params['thermal_loss_coeff'] * (T_out_t - T1)
                       + params['heating_efficiency_coeff'] * p1_actual 
                       - params['heat_vent_coeff'] * v_actual 
                       + params['heat_occupancy_coeff'] * occ1_t)
            
            T2_next = (T2 
                       + params['heat_exchange_coeff'] * (T1 - T2) 
                       + params['thermal_loss_coeff'] * (T_out_t - T2)
                       + params['heating_efficiency_coeff'] * p2_actual 
                       - params['heat_vent_coeff'] * v_actual 
                       + params['heat_occupancy_coeff'] * occ2_t)
            
            H_next = (H 
                      + params['humidity_occupancy_coeff'] * (occ1_t + occ2_t) 
                      - params['humidity_vent_coeff'] * v_actual)
            
            # Move to the next state
            T1, T2, H = T1_next, T2_next, H_next
            
        daily_costs.append(cost_for_day)
        
    return daily_costs

# ==========================================
# 4. MAIN EXECUTION SCRIPT
# ==========================================
if __name__ == "__main__":
    # Load system parameters
    params = get_fixed_data()
    
    try:
        # Load CSV files assuming no headers (header=None) so indices [day, t] align correctly
        prices_df = pd.read_csv('PriceData.csv', header=None)
        occ1_df = pd.read_csv('OccupancyRoom1.csv', header=None)
        occ2_df = pd.read_csv('OccupancyRoom2.csv', header=None)
        
        print("Starting environment simulation for the Dummy Policy...")
        dummy_costs = run_simulation(dummy_policy, prices_df, occ1_df, occ2_df, params)
        
        avg_dummy_cost = np.mean(dummy_costs)
        print(f"Average daily energy cost for the Dummy Policy: {avg_dummy_cost:.2f}")
        
        # Task 6 requirement: Plotting the histogram and saving it to a file
        plt.figure(figsize=(10, 6))
        plt.hist(dummy_costs, bins=20, alpha=0.75, color='#2c7bb6', edgecolor='black')
        plt.axvline(avg_dummy_cost, color='#d7191c', linestyle='dashed', linewidth=2, 
                    label=f'Average Cost: {avg_dummy_cost:.2f}')
        plt.title('Histogram of Daily Costs - Dummy Policy (100 days)')
        plt.xlabel('Daily Cost')
        plt.ylabel('Frequency (Number of days)')
        plt.legend()
        plt.grid(axis='y', alpha=0.4)
        plt.tight_layout()
        
        output_filename = 'Task6_DummyPolicy_Histogram.png'
        plt.savefig(output_filename, dpi=300)
        print(f"Histogram saved successfully as: {output_filename}")
        plt.show()
        
    except FileNotFoundError as e:
        print(f"ERROR: File not found! Make sure the CSV files are in the same directory. Details: {e}")
