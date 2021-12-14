""""
    In this script a CRTBP using tudatpy is made. The Moon and satellites will be propagated by the Earth, so that the
    properties of the Moon can be simplified as in a CRTBP.
    Using Earth as central body for both L2 orbiter and Moon
    Trying to propagate the LLO orbiter around the Moon in CRTBP instead of real Moon ephemeris
"""
import numpy as np
from tudatpy.kernel import constants
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.numerical_simulation import propagation
import matplotlib.pyplot as plt
import Parameters

spice_interface.load_standard_kernels()
simulation_start_epoch = 0.0
simulation_end_epoch = 14.7*constants.JULIAN_DAY

#################
# Environment set-up
#################
bodies_to_create = ["Earth", "Moon"]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)

bodies = environment_setup.create_system_of_bodies(body_settings)

#############
# System of bodies
#############
bodies.create_empty_body("L2orbiter")
bodies.create_empty_body("LLOorbiter")
#bodies.create_empty_body("Moon")
bodies.get("L2orbiter").mass = 24
bodies.get("LLOorbiter").mass = 1000
bodies.get("Moon").mass = Parameters.Moon_Mass

bodies_to_propagate = ["Moon", "L2orbiter", "LLOorbiter"]
central_bodies = ["Earth", "Earth", "Moon"]


###############
# Acceleration settings
###############
acceleration_settings_Moon = dict(
    Earth=[
        propagation_setup.acceleration.point_mass_gravity()
    ]
)
acceleration_settings_L2orbiter = dict(
    Earth=[
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Moon=[
        propagation_setup.acceleration.point_mass_gravity()
    ]
)
acceleration_Settings_LLOorbiter = dict(
    Moon=[
        propagation_setup.acceleration.point_mass_gravity()
    ]
)
acceleration_settings = {
    "Moon": acceleration_settings_Moon,
    "L2orbiter": acceleration_settings_L2orbiter,
    "LLOorbiter": acceleration_Settings_LLOorbiter
}

acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, bodies_to_propagate, central_bodies
)

##################
# Propagation Settings
##################
initial_state_Moon = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=Parameters.Earth_GM,
    semi_major_axis=384400E3,
    eccentricity=0.0,
    inclination=np.deg2rad(0),
    argument_of_periapsis=np.deg2rad(0),
    longitude_of_ascending_node=np.deg2rad(0),
    true_anomaly=np.deg2rad(0)
)

initial_state_L2orbiter = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=Parameters.EarthMoon_GM,
    semi_major_axis=Parameters.EarthMoon_Distance*Parameters.x_norm_i,
    eccentricity=0.0,
    inclination=np.arctan(Parameters.z_norm_i/Parameters.x_norm_i),
    argument_of_periapsis=np.deg2rad(0),
    longitude_of_ascending_node=np.deg2rad(0),
    true_anomaly=np.deg2rad(0)
)

initial_state_LLOorbiter = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=Parameters.Moon_GM,
    semi_major_axis= Parameters.Moon_R + Parameters.h_orbit,
    eccentricity=0.0,
    inclination=np.deg2rad(90),
    argument_of_periapsis=np.deg2rad(0),
    longitude_of_ascending_node=np.deg2rad(90),
    true_anomaly=np.deg2rad(0)
)

#################################### VERY IMPORTANT how to stack initial states ############################
initial_states = np.vstack(
    [initial_state_Moon.reshape(-1, 1),
     initial_state_L2orbiter.reshape(-1, 1),
     initial_state_LLOorbiter.reshape(-1, 1)]
)
print(initial_states)


#initial_states = Parameters.circ_ini_states

Moon_dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("Moon"),
    propagation_setup.dependent_variable.central_body_fixed_cartesian_position("Moon", "Earth"),
    propagation_setup.dependent_variable.keplerian_state("Moon", "Earth"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Moon", "Earth"
    )
]
L2orbiter_dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("L2orbiter"),
    propagation_setup.dependent_variable.central_body_fixed_cartesian_position("L2orbiter", "Earth"),
    propagation_setup.dependent_variable.keplerian_state("L2orbiter", "Earth"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "L2orbiter", "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "L2orbiter", "Moon"
    )
]
LLOorbiter_dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("LLOorbiter"),
    propagation_setup.dependent_variable.central_body_fixed_cartesian_position("LLOorbiter", "Moon"),
    propagation_setup.dependent_variable.keplerian_state("LLOorbiter", "Moon"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LLOorbiter", "Moon"
    )
]

################################################ Very important how to stack list of variables ################
complete_list_of_dependent_variables = \
    Moon_dependent_variables_to_save + L2orbiter_dependent_variables_to_save + LLOorbiter_dependent_variables_to_save

###########
# Propagating
###########
termination_condition = propagation_setup.propagator.time_termination(simulation_end_epoch)
propagation_setup = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_states,
    termination_condition,
    output_variables=complete_list_of_dependent_variables
)

########### Integrating ############
fixed_step_size = 1000
integrator_settings = numerical_simulation.propagation_setup.integrator.runge_kutta_4(
    simulation_start_epoch, fixed_step_size
)

############ Simulation ###############
dynamic_simulator = numerical_simulation.SingleArcSimulator(
    bodies, integrator_settings, propagation_setup
)
states = dynamic_simulator.state_history
dependent_variables = dynamic_simulator.dependent_variable_history

######### Rewritting Data #############
width = states[0.0].size          # This is the length of np array
height = len(states)                # This is the amount of key/value pairs fo dict

states_array = np.empty(shape=(height,width))        # Ini 2d matrix
# Loop over entries in dictionair getting both key and value
for x, (key, np_array) in enumerate(states.items()):
    # Looping over elements in the np array
    for y, np_value in enumerate(np_array):
        #print("i {}: key: {}, np.array {}".format(x, key, np_value))
        states_array[x, y] = np_value

### in Kilometer
states_Moon = states_array[:, 0:6]*10**-3
states_L2orbiter = states_array[:, 6:12]*10**-3
states_LLOorbiter_wrt_Moon = states_array[:, 12:18]*10**-3
states_LLOorbiter = states_LLOorbiter_wrt_Moon + states_Moon


############################## Plottings ###############################
plt.figure()
ax = plt.axes(projection='3d')
plt.plot([0], [0], [0], marker='o', markersize = 10, color='blue')
plt.plot([Parameters.EarthMoon_Distance*10**-3], [0], [0], marker='o', markersize=3, color='grey')
ax.plot3D(states_Moon[:, 0], states_Moon[:, 1], states_Moon[:, 2])
ax.plot3D(states_L2orbiter[:, 0], states_L2orbiter[:, 1], states_L2orbiter[:, 2])
ax.plot3D(states_LLOorbiter[:, 0], states_LLOorbiter[:, 1], states_LLOorbiter[:, 2])
plt.legend(['Earth', 'Initial Position Moon', 'Position Moon', 'Position L2orbiter', 'Position LLOorbiter'])

plt.figure()
ax = plt.axes(projection='3d')
plt.plot([0], [0], [0], marker='o', markersize=3, color='grey')
ax.plot3D(states_LLOorbiter_wrt_Moon[:, 0], states_LLOorbiter_wrt_Moon[:, 1], states_LLOorbiter_wrt_Moon[:, 2])
plt.show()
