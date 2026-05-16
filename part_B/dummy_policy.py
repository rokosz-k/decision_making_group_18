"""
Python code implementing a dummy policy that never turns on the ventilation nor any heater, and leaves
everything up to the overrule controllers.
"""


DUMMY_ACTION = {
    "HeatPowerRoom1": 0.0,
    "HeatPowerRoom2": 0.0,
    "VentilationON":  0
}

def dummy_action(state):
    return DUMMY_ACTION