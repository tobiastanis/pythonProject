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
simulation_end_epoch = constants.JULIAN_DAY

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
bodies.get("Moon").mass = Parameters.Moon_Mass
bodies.get("L2orbiter").mass = 24
bodies.get("LLOorbiter").mass = 1000


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
    semi_major_axis=384400E3 + 60000E3,
    eccentricity=0.0,
    inclination=np.deg2rad(0),
    argument_of_periapsis=np.deg2rad(0),
    longitude_of_ascending_node=np.deg2rad(0),
    true_anomaly=np.deg2rad(0)
)
initial_state_LLOorbiter = element_conversion.keplerian_to_cartesian_elementwise(
    gravitational_parameter=Parameters.Moon_GM,
    semi_major_axis=50E3 + Parameters.Moon_R,
    eccentricity=0.0,
    inclination=np.deg2rad(0),
    argument_of_periapsis=np.deg2rad(0),
    longitude_of_ascending_node=np.deg2rad(0),
    true_anomaly=np.deg2rad(0)
)

initial_states = np.stack((initial_state_Moon, initial_state_L2orbiter, initial_state_LLOorbiter), axis=0)
initial_states = initial_states.astype(np.float64)

dependent_variables_to_save = [
    propagation_setup.dependent_variable.total_acceleration("Moon"),
    propagation_setup.dependent_variable.keplerian_state("Moon", "Earth"),
    propagation_setup.dependent_variable.latitude("Moon", "Earth"),
    propagation_setup.dependent_variable.longitude("Moon", "Earth"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "Moon", "Earth"
    ),
    propagation_setup.dependent_variable.total_acceleration("L2orbiter"),
    propagation_setup.dependent_variable.keplerian_state("L2orbiter", "Earth"),
    propagation_setup.dependent_variable.latitude("L2orbiter", "Earth"),
    propagation_setup.dependent_variable.longitude("L2orbiter", "Earth"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "L2orbiter", "Earth"
    ),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "L2orbiter", "Moon"
    ),
    propagation_setup.dependent_variable.total_acceleration("LLOorbiter"),
    propagation_setup.dependent_variable.keplerian_state("LLOorbiter", "Moon"),
    propagation_setup.dependent_variable.latitude("LLOorbiter", "Moon"),
    propagation_setup.dependent_variable.longitude("LLOorbiter", "Moon"),
    propagation_setup.dependent_variable.single_acceleration_norm(
        propagation_setup.acceleration.point_mass_gravity_type, "LLOorbiter", "Moon"
    )
]


###########
# Propagating
###########
termination_condition = propagation_setup.propagator.time_termination( simulation_end_epoch)
#print(type(central_bodies))
#print(type(acceleration_models))
#print(type(bodies_to_propagate))
#print(type(initial_states))
#print(type(termination_condition))
#print(type(dependent_variables_to_save))
propagation_setup = propagation_setup.propagator.translational(
    central_bodies,
    acceleration_models,
    bodies_to_propagate,
    initial_states,
    termination_condition,
    output_variables=dependent_variables_to_save
)
print(propagation_setup)
print(type(propagation_setup))

fixed_step_size = 1000.0
integrator_settings = propagation_setup.integrator.runge_kutta_4(
    simulation_start_epoch,
    fixed_step_size
)
