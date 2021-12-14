import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.astro import element_conversion
spice_interface.load_standard_kernels()

########### Environmental Constants ###########
# General
G = constants.GRAVITATIONAL_CONSTANT

# Earth
Earth_GM = spice_interface.get_body_gravitational_parameter("Earth")
Earth_Mass = Earth_GM / G
Earth_R = spice_interface.get_average_radius("Earth")
# Moon
Moon_GM = spice_interface.get_body_gravitational_parameter("Moon")
Moon_Mass = Moon_GM / G
Moon_R = spice_interface.get_average_radius("Moon")
# Earth-Moon system
EarthMoon_Distance = 384400E3
EarthMoon_GM = G*(Earth_Mass+Moon_Mass)
# Sun

# Moon circular
Moon_v = np.sqrt(Earth_GM/EarthMoon_Distance)
x_Moon_i = np.array([EarthMoon_Distance, 0, 0, 0, Moon_v, 0])
############################################################################
# Orbiters
########### Earth-Moon L2 Orbiter Properties ###########
L2_Mass = 24

# Normalized initial values
#x_norm_i = 1.1785867766
#y_norm_i = 0
#z_norm_i = -0.04686057902
#vx_norm_i = 0
#vy_norm_i = -0.16739715056
#vz_norm_i = 0
x_norm_i = 1.1435
y_norm_i = 0
z_norm_i = -0.1579
vx_norm_i = 0
vy_norm_i = -0.2220
vz_norm_i = 0

x_norm_0 = np.array([x_norm_i, y_norm_i, z_norm_i, vx_norm_i, vy_norm_i, vz_norm_i])
x_L2_i = x_norm_0*EarthMoon_Distance

########### Low Lunar Orbiter Properties ###########
# LLO circular
h_orbit = 50E3
r_scalar = Moon_R+h_orbit
v_scalar = np.sqrt(Moon_GM/r_scalar)
x_LLO_i = np.array([0, r_scalar, 0, 0, 0, v_scalar])

# Lunar Pathfinder
initial_state_LunarPathfinder = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=EarthMoon_GM,
    semi_major_axis=5737.4E3,
    eccentricity=0.61,
    inclination=np.deg2rad(57.82),
    argument_of_periapsis=np.deg2rad(90),
    longitude_of_ascending_node=np.rad2deg(61.552),
    true_anomaly=np.deg2rad(30)
)

circ_ini_states = np.vstack(
    [x_Moon_i.reshape(-1, 1),
     x_L2_i.reshape(-1, 1),
     x_LLO_i.reshape(-1, 1)]
)
#print("Parameters")
#print(circ_ini_states)

########### Lunar Reconnaissance Orbiter Properties ##########


#################################################################
