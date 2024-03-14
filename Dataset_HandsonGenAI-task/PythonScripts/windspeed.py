
"""
Given a wind speed, solve the fatigue load for a generic wind turbine blade load
"""

import numpy


#%%
def bladefatigue(wind_speed):
    """ Define a generic relation between blade flapwise fatigue and wind speed"""

    # Wind speed to fatigue function
    bld_flap_root_fatigue = -0.0512*wind_speed**4 + 6.0206*wind_speed**3 - 223.92*wind_speed**2 + 2936.4*wind_speed - 4595.7


    #returns 
    return bld_flap_root_fatigue

#%%
def main():
    data_input = 23
    fatigue_estimate = bladefatigue(data_input)
    print(fatigue_estimate)

if __name__ == "__main__":
    main()