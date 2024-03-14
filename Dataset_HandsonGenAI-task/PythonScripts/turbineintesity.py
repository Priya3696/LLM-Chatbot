"""
Given turbulence intensity and yaw error, solve the fatigue load of the yaw system of a generic wind turbine
"""

import numpy


#%%
def yawfatigue(turbulence_intensity, yaw_error):
    """ Define a generic relation between yaw syst fatigue and turbulence intensity and yaw error"""

    x = turbulence_intensity
    y = yaw_error
    # turbulence intensity and yaw error to fatigue function
    # Coefficients:
    p00 =        1339
    p10 =       162.2
    p01 =       445.1
    p20 =       -24.7
    p11 =   1.239e-13
    p02 =      -21.78
    p30 =       1.287
    p21 =  -7.281e-15
    p12 =   1.016e-15
    p03 =      0.3555
    p40 =    -0.01426
    p31 =   6.988e-17
    p22 =   9.675e-17
    p13 =  -8.285e-17
       
    yaw_sys_fatigue = p00 + p10*x + p01*y + p20*x**2 + p11*x*y + p02*y**2 + p30*x**3 + p21*x**2*y + p12*x*y**2 + p03*y**3 + p40*x**4 + p31*x**3*y + p22*x**2*y**2 + p13*x*y**3
    

    #returns 
    return yaw_sys_fatigue

#%%
def main():
    ti = 10
    ye = 23
    yaw_sys_fatigue = yawfatigue(ti,ye)
    print(yaw_sys_fatigue)

if __name__ == "__main__":
    main()