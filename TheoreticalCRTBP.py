############################## Iets koels ##################
import numpy as np
from tudatpy.kernel import constants
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import Parameters
"""
    This model shows a L2 orbiter in a CRTBP and a LLO in simulated in a circular Two-Body Problem. Both position 
    vectors are tranformed into non-normalized Earth_Moon barycentric. Maybe I'll rewrite it to Earth centric since
    The models using tudatpy will be Earth centric, so comparing would be easier. The inter-satellite distance is
    measured and plotted.
"""

####################################################################
# Defining the Normalized crtbp
# Primary masses
m1 = Parameters.Earth_Mass
m2 = Parameters.Moon_Mass
mu_i = m2/(m1+m2)       # Normalized mass unit [-]
# Normalized time
T_normalized = np.sqrt(Parameters.EarthMoon_Distance**3/Parameters.EarthMoon_GM)    # [s]
T_normalized_day = T_normalized/constants.JULIAN_DAY    # [day]

##### Defining the Circular Restricted Three-Body Problem #####
def crtbp(x, t, mu):
    # Distance primaries to S/C
    r1 = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)
    r2 = np.sqrt((x[0] + mu - 1)**2 + x[1]**2 +x[2]**2)
    # Normalized masses of the primaries
    mu = m2/(m1+m2)

    # From Wakker
    xdot = [x[3],
            x[4],
            x[5],
            x[0]+2 * x[4] - (1-mu)*(x[0]+mu)/r1**3 - mu * (x[0]+mu-1)/r2**3,
            -2*x[3] + (1 - (1-mu)/r1**3 - mu/r2**3)*x[1],
            ((mu-1)/r1**3 - mu/r2**3)*x[2]
    ]
    return xdot
##### Defining the Circular Restricted Two-Body Problem #####
def twobody_dyn(y, t, mu):
    # Unpack state
    rx, ry, rz, vx, vy, vz = y
    r = np.array([rx, ry, rz])

    norm_r = np.linalg.norm(r)

    # Two-Body acceleration
    ax, ay, az = -r * mu/norm_r**3

    return [vx, vy, vz, ax, ay, az]
if __name__ == '__main__':

    # Initial conditions
    r_scalar = Parameters.Moon_R + Parameters.h_orbit   # [m]
    v_scalar = np.sqrt(Parameters.Moon_GM/r_scalar)     # [m/s]

    # Initial position vector

    r0 = [0, r_scalar, 0]
    v0 = [0, 0, v_scalar]
    #r0 = [r_scalar, 0, 0]
    #v0 = [0, v_scalar, 0]

    # Simulation time setup
    t0 = 0.0
    TAU_LLO = 14.7*constants.JULIAN_DAY
    TAU_L2 = 14.7/T_normalized_day
    dt = 10
    n_steps = int(np.ceil(TAU_LLO/dt))
    t_span_LLO = np.linspace(t0, TAU_LLO, n_steps)
    t_span_L2 = np.linspace(t0, TAU_L2, n_steps)

    # Initial conditions, y for LLO and x for L2
    y0 = r0 + v0
    x0 = Parameters.x_norm_0

    # Initialize solver
    states_wrt_Moon_LLO = odeint(twobody_dyn, y0, t_span_LLO, args=(Parameters.Moon_GM,), rtol=1e-12, atol=1e-12)
    states_norm_L2 = odeint(crtbp, x0, t_span_L2, args=(mu_i,), rtol=1e-12, atol=1e-12)

    # Transforming states to Non-Normalized Earth-Moon barycenter
    states_L2 = states_norm_L2 * Parameters.EarthMoon_Distance
    states_LLO = [(1-mu_i)*Parameters.EarthMoon_Distance, 0, 0, 0, 0, 0] + states_wrt_Moon_LLO
    Earth_Position = (-1 * mu_i * Parameters.EarthMoon_Distance) * 10 ** -3         # [km]
    Moon_Position = ((1 - mu_i) * Parameters.EarthMoon_Distance) * 10 ** -3         # [km]
    Moon_vector = [Moon_Position*10**3, 0, 0]

    # Inter-satellite distance
    pos_LLO = states_LLO[:, :3]         #wrt barycenter
    pos_L2 = states_L2[:, :3]           #wrt barycenter

    delta_pos_vector = pos_L2 - pos_LLO             #Intersatellite position vector
    inter_satellite_distance = np.linalg.norm(delta_pos_vector, axis=1)     #absolute intersatelitte distance
    L2_Moondistance_vector = pos_L2 - Moon_vector                           #L2 Moon distance vector
    L2_Moondistance = np.linalg.norm(L2_Moondistance_vector, axis=1)        #L2 distance to Moon

    # Time in days
    time = t_span_LLO / constants.JULIAN_DAY

    #########################################################################
    # PLOTS #
    plt.figure()
    plot = plt.axes(projection='3d')
    plot.plot3D(states_L2[:, 0] * 10 ** -3, states_L2[:, 1] * 10 ** -3, states_L2[:, 2] * 10 ** -3)
    plot.plot3D(states_LLO[:, 0]*10**-3, states_LLO[:, 1]*10**-3, states_LLO[:, 2]*10**-3)
    plot.plot3D([Earth_Position], [0], [0], marker='o', markersize=10, color='blue')
    plot.plot3D([Moon_Position], [0], [0], marker='o', markersize=3, color='grey')
    plt.xlabel("X-axis direction [km]")
    plt.ylabel("Y-axis direction [km]")


    plt.figure()
    plt.plot(states_L2[:, 0]*10**-3, states_L2[:, 2]*10**-3)
    plt.plot(states_LLO[:, 0]*10**-3, states_LLO[:, 2]*10**-3)
    plt.plot([Earth_Position], [0], marker='o', markersize=10, color='blue')
    plt.plot([Moon_Position], [0], marker='o', markersize=3, color='grey')
    plt.xlabel("X-axis direction [km]")
    plt.ylabel("Y-axis direction [km]")


    plt.figure()
    plt.title("Inter-satellite distance and L2-Moon distance over one period L2 orbit")
    plt.xlabel("Time [days]")
    plt.ylabel("Inter-satellite distance [km]")
    plt.plot(time, inter_satellite_distance*10**-3)
    plt.plot(time, L2_Moondistance*10**-3)
    plt.legend(['Inter-satellite distance', 'L2orbiter-Moon distance'])
    plt.grid()

    plt.figure()
    plot = plt.axes(projection='3d')
    plot.plot3D(states_wrt_Moon_LLO[:, 0]*10**-3, states_wrt_Moon_LLO[:, 1]*10**-3, states_wrt_Moon_LLO[:, 2]*10**-3)
    plot.plot3D([0], [0], [0], marker='o', markersize=10, color='grey')
    plt.xlabel("X-axis direction [km]")
    plt.ylabel("Y-axis direction [km]")

    print(max(inter_satellite_distance))
    print(min(inter_satellite_distance))
    print(max(L2_Moondistance))
    print(min(L2_Moondistance))
    plt.show()








