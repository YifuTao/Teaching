import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# animate plots?
animate=False # True / False

# define model
def vehicle(v,t,u,load):
    # inputs
    #  v    = vehicle velocity (m/s)
    #  t    = time (sec)
    #  u    = gas pedal position (-50% to 100%)
    #  load = passenger load + cargo (kg)
    Cd = 0.24    # drag coefficient
    rho = 1.225  # air density (kg/m^3)
    A = 5.0      # cross-sectional area (m^2)
    Fp = 30      # thrust parameter (N/%pedal)
    m = 500      # vehicle mass (kg)
    # calculate derivative of the velocity
    dv_dt = (1.0/(m+load)) * (Fp*u - 0.5*rho*Cd*A*v**2)
    return dv_dt

tf = 300.0                 # final time for simulation
nsteps = 300 * 10 + 1               # number of time steps
delta_t = tf/(nsteps-1)   # how long is each time step?
ts = np.linspace(0,tf,nsteps) # linearly spaced time vector

# simulate step test operation
step = np.zeros(nsteps) # u = valve % open
step[11:] = 50.0       # step up pedal position
# passenger(s) + cargo load
load = 200.0 # kg
# velocity initial condition
v0 = 0.0
# set point
sp = 25.0
# for storing the results
vs = np.zeros(nsteps)
sps = np.zeros(nsteps)

ubias = 0.0
Kc = 5.0 / 1.2
tauI = 10
Kd = 10
sum_int = 0.0
es = np.zeros(nsteps)
ies = np.zeros(nsteps)
sensor_lag = 2

plt.figure(1,figsize=(5,4))
if animate:
    plt.ion()
    plt.show()

v_sensed = 0
# simulate with ODEINT
for i in range(nsteps-1):
    # u = step[i]

    sps[i+1] = sp
    error = sp - v0
    es[i+1] = error
    sum_int = sum_int + error * delta_t
    # de_dt = (es[i+1] - es[i]) / delta_t
    u = ubias + Kc * error + Kc/tauI * sum_int  # + Kd * de_dt
    #u = u + np.random.normal(0, 1)
    # clip inputs to -50% to 100%
    
    if u >= 100.0:
        u = 100.0
    if u <= -50.0:
        u = -50.0
    
    ies[i+1] = sum_int
    step[i+1] = u
    v = odeint(vehicle,v0,[0,de

    

    # plot results
    if animate:
        plt.clf()
        plt.subplot(2,1,1)
        plt.plot(ts[0:i+1],vs[0:i+1],'b-',linewidth=3)
        plt.plot(ts[0:i+1],sps[0:i+1],'k--',linewidth=2)
        plt.ylabel('Velocity (m/s)')
        plt.legend(['Velocity','Set Point'],loc=2)
        plt.subplot(2,1,2)
        plt.plot(ts[0:i+1],step[0:i+1],'r--',linewidth=3)
        plt.ylabel('Gas Pedal')    
        plt.legend(['Gas Pedal (%)'])
        plt.xlabel('Time (sec)')
        plt.pause(0.1)    

print(es)
if not animate:
    # plot results
    plt.subplot(2,1,1)
    plt.plot(ts,vs,'b-',linewidth=3)
    plt.plot(ts,sps,'k--',linewidth=2)
    plt.ylabel('Velocity (m/s)')
    plt.legend(['Velocity','Set Point'],loc=2)
    plt.subplot(2,1,2)
    plt.plot(ts,step,'r--',linewidth=3)
    plt.ylabel('Gas Pedal')    
    plt.legend(['Gas Pedal (%)'])
    plt.xlabel('Time (sec)')
    plt.show()
