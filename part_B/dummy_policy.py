"""
Python code implementing a dummy policy that never turns on the ventilation nor any heater, and leaves
everything up to the overrule controllers.
"""


def dummy_action(state):

    HereAndNowActions = {
        "HeatPowerRoom1": 0,   # No heating in room 1
        "HeatPowerRoom2": 0,   # No heating in room 2
        "VentilationON":  0    # Ventilation OFF
    }

    return HereAndNowActions