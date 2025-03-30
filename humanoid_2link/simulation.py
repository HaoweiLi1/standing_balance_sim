import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from typing import Callable, Optional, Union, List
import scipy.linalg as linalg
import mediapy as media
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from queue import Queue
import xml.etree.ElementTree as ET
import yaml


from xml_utilities import calculate_kp_and_geom, set_geometry_params

from renderer import MujocoRenderer
from data_logger import DataLogger
from data_plotter import DataPlotter
from controllers import create_human_controller, create_exo_controller, HumanLQRController
from perturbation import create_perturbation


plt.rcParams['text.usetex'] = True

mpl.rcParams.update(mpl.rcParamsDefault)

perturbation_queue = Queue()
control_log_queue = Queue()
counter_queue = Queue()
perturbation_datalogger_queue = Queue()

class AnkleExoSimulation:
    """
    Main simulation class for ankle exoskeleton.
    Handles physics simulation, controllers, and data logging.
    """
    
    def __init__(self, config_file='config.yaml'):
        """
        Initialize the simulation.
        
        Args:
            config_file: Path to configuration YAML file
        """
        print('Simulation initialized')
        self.config_file = config_file
        self.params = self.load_params_from_yaml(config_file)
        self.config = self.params['config']
        
        # Setup flags
        self.plot_flag = self.config['plotter_flag']
        self.mp4_flag = self.config['mp4_flag']
        
        # Controllers
        self.human_controller = None
        self.exo_controller = None
        self.show_exoskeleton = True  # Add flag for exoskeleton visibility
        
        # Renderer
        self.renderer = None
        
        # Data logging
        self.logger = DataLogger()
        self.plotter = None
        
        self.visualization_flag = self.config.get('visualization_flag', True)

        # MuJoCo objects
        self.model = None
        self.data = None
        self.ankle_joint_id = None
        self.human_body_id = None
        
        # Setpoints
        self.ankle_position_setpoint = self.config['ankle_position_setpoint_radians']
        
        # Perturbation
        self.perturbation = None
        self.perturbation_thread = None

    def load_params_from_yaml(self, file_path):
        """Load configuration from YAML file."""
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        return params

    def initialize_controllers(self, model, data):
        """Initialize controllers for ankle and hip."""
        # Get human controller configuration for ankle
        human_config = self.config['controllers']['human']
        human_type = human_config['type']
        
        # Get hip controller configuration
        hip_config = self.config['controllers']['hip']
        
        print(f"Initializing controllers with type: {human_type}")
        
        # Create ankle human controller parameters
        ankle_params = {
            'max_torque_df': human_config['max_torque_df'],
            'max_torque_pf': human_config['max_torque_pf'],
            'mrtd_df': human_config.get('mrtd_df'),
            'mrtd_pf': human_config.get('mrtd_pf'),
        }
        
        # Add controller-specific parameters for ankle
        if human_type == "LQR":
            lqr_params = human_config['lqr_params']
            ankle_params.update({
                'Q_angle': lqr_params['Q_angle'],
                'Q_velocity': lqr_params['Q_velocity'],
                'R': lqr_params['R'],
                'mass': self.config['M_total'] - 2*0.0145*self.config['M_total'],
                'leg_length': 0.575 * self.config['H_total']
            })
            # For LQR, also pass exo configuration if exo is enabled
            exo_config = self.config['controllers']['exo']
            if exo_config['type'] != "None":
                ankle_params['exo_config'] = exo_config
        elif human_type == "PD":
            pd_params = human_config['pd_params']
            ankle_params.update({
                'kp': pd_params['kp'],
                'kd': pd_params['kd']
            })
        
        # Create human ankle controller
        self.human_ankle_controller = create_human_controller(
            human_type, model, data, ankle_params
        )
        
        # Create hip controller parameters
        hip_params = {
            'max_torque_df': hip_config.get('max_torque_ext', 150),
            'max_torque_pf': hip_config.get('max_torque_flex', -150),
        }
        
        # Add controller-specific parameters for hip
        if human_type == "LQR":
            hip_lqr_params = hip_config.get('lqr_params', human_config['lqr_params'])
            hip_params.update({
                'Q_angle': hip_lqr_params['Q_angle'],
                'Q_velocity': hip_lqr_params['Q_velocity'],
                'R': hip_lqr_params['R'],
                'mass': 0.43 * self.config['M_total'],  # Torso is approximately 43% of body mass
                'leg_length': 0.35 * self.config['H_total']  # Approximate COM height from hip
            })
        elif human_type == "PD":
            hip_pd_params = hip_config.get('pd_params', {'kp': 50, 'kd': 5})
            hip_params.update({
                'kp': hip_pd_params['kp'],
                'kd': hip_pd_params['kd']
            })
        
        # Create human hip controller (using same controller type as ankle)
        self.hip_controller = create_human_controller(
            human_type, model, data, hip_params
        )
        
        # Create exo controller for ankle only (unchanged)
        exo_config = self.config['controllers']['exo']
        exo_type = exo_config['type']
        
        # Set show_exoskeleton flag based on controller type
        self.show_exoskeleton = exo_type != "None"
        
        # Prepare parameters for exo controller
        exo_params = {
            'max_torque': exo_config.get('max_torque', 0)
        }
        
        # Add controller-specific parameters
        if exo_type == "PD":
            pd_params = exo_config['pd_params']
            exo_params.update({
                'kp': pd_params.get('kp', 400),
                'kd': pd_params.get('kd', 10)
            })

        # Create exo controller using factory function
        self.exo_controller = create_exo_controller(
            exo_type, model, data, exo_params
        )
        
        # Set exoskeleton visibility based on controller type
        self.toggle_exoskeleton_visibility(model, self.show_exoskeleton)

    # def initialize_controllers(self, model, data):
    #     """Initialize controllers based on configuration."""
    #     # Get human controller configuration
    #     human_config = self.config['controllers']['human']
    #     human_type = human_config['type']
        
    #     # Debug prints
    #     print(f"MRTD from config - DF: {human_config.get('mrtd_df')}, PF: {human_config.get('mrtd_pf')}")
    #     print(f"Human config keys: {list(human_config.keys())}")

    #     # Get exo controller configuration
    #     exo_config = self.config['controllers']['exo']
    #     exo_type = exo_config['type']

    #     # Prepare parameters for human controller
    #     human_params = {
    #         'max_torque_df': human_config['max_torque_df'],
    #         'max_torque_pf': human_config['max_torque_pf'],
    #         'mass': self.config['M_total'] - 2*0.0145*self.config['M_total'],
    #         'leg_length': 0.575 * self.config['H_total'],
    #         'mrtd_df': human_config.get('mrtd_df'),
    #         'mrtd_pf': human_config.get('mrtd_pf')
    #     }

    #     # Debug print
    #     print(f"Human params for controller: {human_params}")
        
    #     # Add controller-specific parameters
    #     if human_type == "LQR":
    #         lqr_params = human_config['lqr_params']
    #         human_params.update({
    #             'Q_angle': lqr_params['Q_angle'],
    #             'Q_velocity': lqr_params['Q_velocity'],
    #             'R': lqr_params['R']
    #         })
    #         # For LQR, also pass exo configuration if exo is enabled
    #         if exo_type != "None":
    #             human_params['exo_config'] = exo_config
                
    #             # Get the pre-calculated values from xml_utilities for dynamic gains
    #             if exo_type == "PD" and exo_config.get('pd_params', {}).get('use_dynamic_gains', False):
    #                 _, m_feet, m_body, l_COM, _, _, K_p = calculate_kp_and_geom(
    #                     self.config['M_total'], 
    #                     self.config['H_total']
    #                 )
                    
    #                 # Calculate dynamic gains
    #                 kp = K_p  # Already calculated as m_body * g * l_COM
    #                 kd = 0.3 * np.sqrt(m_body * l_COM**2 * kp)
                    
    #                 # Pass dynamic gains to LQR controller
    #                 human_params['dynamic_gains'] = {
    #                     'kp': kp,
    #                     'kd': kd
    #                 }
    #                 print(f"Passing exo dynamic gains to LQR - Kp: {kp:.2f}, Kd: {kd:.2f}")
    #     elif human_type == "PD":
    #         pd_params = human_config['pd_params']
    #         human_params.update({
    #             'kp': pd_params['kp'],
    #             'kd': pd_params['kd']
    #         })
    #     elif human_type == "Precomputed":
    #         human_params['precomputed_params'] = human_config.get('precomputed_params', {})
    #         # Log information about the trajectory file
    #         trajectory_file = human_params['precomputed_params'].get('trajectory_file')
    #         if trajectory_file:
    #             print(f"Using precomputed trajectory from: {trajectory_file}")
            
    #     # Create human controller using factory function
    #     self.human_controller = create_human_controller(
    #         human_type, model, data, human_params
    #     )

    #     # Set show_exoskeleton flag based on controller type
    #     self.show_exoskeleton = exo_type != "None"
        
    #     # Get the pre-calculated values from xml_utilities
    #     _, m_feet, m_body, l_COM, _, _, K_p = calculate_kp_and_geom(
    #         self.config['M_total'], 
    #         self.config['H_total']
    #     )

    #     # Prepare parameters for exo controller
    #     exo_params = {
    #         'max_torque': exo_config.get('max_torque', 0)
    #     }
        
    #     # Add controller-specific parameters
    #     if exo_type == "PD" and exo_config.get('pd_params', {}).get('use_dynamic_gains', False):
    #         # Calculate Kp and Kd dynamically using the pre-calculated values
    #         kp = K_p  # Already calculated as m_body * g * l_COM
    #         kd = 0.3 * np.sqrt(m_body * l_COM**2 * kp)
            
    #         exo_params.update({
    #             'kp': kp,
    #             'kd': kd
    #         })
            
    #         print(f"Exo PD controller configured with dynamic gains - Kp: {kp:.2f}, Kd: {kd:.2f}")

    #     elif exo_type == "PD":
    #         # Use the values from config if dynamic gains are not enabled
    #         pd_params = exo_config['pd_params']
    #         exo_params.update({
    #             'kp': pd_params.get('kp', 400),
    #             'kd': pd_params.get('kd', 10)
    #         })

    #     # Create exo controller using factory function
    #     self.exo_controller = create_exo_controller(
    #         exo_type, model, data, exo_params
    #     )
        
    #     # Set exoskeleton visibility based on controller type
    #     self.toggle_exoskeleton_visibility(model, self.show_exoskeleton)

    def toggle_exoskeleton_visibility(self, model, show_exoskeleton):
        """
        Toggle the visibility of exoskeleton components in the model.
        
        Args:
            model: MuJoCo model
            show_exoskeleton: Boolean indicating whether to show the exoskeleton
        """
        # List of all exoskeleton component names
        exo_geom_names = [
            "exo_heel_attachment", 
            "exo_housing", 
            "exo_connecting_rod",
            "exo_joint_upper", 
            "exo_joint_lower",
            "exo_calf_strap_left",
            "exo_calf_strap_right",
            "exo_calf_strap_top"
        ]
        
        # Set the visibility of each component
        for name in exo_geom_names:
            geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            if geom_id >= 0:  # Check if the geom exists
                # Set alpha to 0 (invisible) or 1 (visible)
                model.geom_rgba[geom_id][3] = 1.0 if show_exoskeleton else 0.0
                
                # Additionally, if not showing, move the geom far away to ensure no interactions
                if not show_exoskeleton:
                    model.geom_pos[geom_id] = np.array([1000.0, 1000.0, 1000.0])
                
        print(f"Exoskeleton visibility set to: {show_exoskeleton}")
    
    def controller(self, model, data):
        """Controller function for ankle and hip."""
        # Get ankle state
        ankle_state = np.array([
            data.qpos[self.ankle_joint_id],  # Ankle angle directly from qpos
            data.qvel[self.ankle_joint_id]   # Ankle velocity directly from qvel
        ])
        
        # Get hip state
        hip_state = np.array([
            data.qpos[self.hip_joint_id],    # Hip angle directly from qpos
            data.qvel[self.hip_joint_id]     # Hip velocity directly from qvel
        ])
        
        # Compute ankle control
        human_ankle_torque = self.human_ankle_controller.compute_control(
            state=ankle_state,
            target=self.ankle_position_setpoint
        )
        
        # Find the human ankle actuator ID
        human_ankle_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "human_ankle_actuator")
        data.ctrl[human_ankle_id] = human_ankle_torque  # Apply torque
        
        # Compute hip control
        hip_torque = self.hip_controller.compute_control(
            state=hip_state,
            target=self.hip_position_setpoint
        )
        
        # Find the hip actuator ID
        hip_actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "human_hip_actuator")
        data.ctrl[hip_actuator_id] = hip_torque  # Apply torque
        
        # Compute and apply exo control for ankle only
        exo_torque = self.exo_controller.compute_control(
            state=ankle_state,
            target=self.ankle_position_setpoint
        )
        
        # Find the exo ankle actuator ID
        exo_ankle_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "exo_ankle_actuator")
        data.ctrl[exo_ankle_id] = exo_torque  # Apply torque

    # def controller(self, model, data):
    #     """Controller function for ankle and hip."""
    #     # Get current states using joint IDs directly (more robust)
    #     ankle_state = np.array([
    #         data.qpos[self.ankle_joint_id],  # Ankle angle directly from qpos
    #         data.qvel[self.ankle_joint_id]   # Ankle velocity directly from qvel
    #     ])
        
    #     hip_state = np.array([
    #         data.qpos[self.hip_joint_id],    # Hip angle directly from qpos
    #         data.qvel[self.hip_joint_id]     # Hip velocity directly from qvel
    #     ])
        
    #     # Compute human ankle control
    #     human_ankle_torque = self.human_ankle_controller.compute_control(
    #         state=ankle_state,
    #         target=self.ankle_position_setpoint
    #     )
        
    #     # Get human ankle actuator ID explicitly
    #     human_ankle_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "human_ankle_actuator")
    #     data.ctrl[human_ankle_id] = human_ankle_torque  # This should be index 0
        
    #     # Compute human hip control
    #     human_hip_torque = self.human_hip_controller.compute_control(
    #         state=hip_state,
    #         target=self.hip_position_setpoint
    #     )
        
    #     # Get hip actuator ID
    #     hip_actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "human_hip_actuator")
    #     data.ctrl[hip_actuator_id] = human_hip_torque  # This should be index 2
        
    #     # Compute exo control for ankle only
    #     exo_torque = self.exo_controller.compute_control(
    #         state=ankle_state,
    #         target=self.ankle_position_setpoint
    #     )
        
    #     # Get exo ankle actuator ID explicitly
    #     exo_ankle_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "exo_ankle_actuator")
    #     data.ctrl[exo_ankle_id] = exo_torque  # This should be index 1
        
    #     # Print actuator indices the first time (for debugging)
    #     if not hasattr(self, '_printed_actuator_indices'):
    #         print(f"Actuator indices: human_ankle={human_ankle_id}, exo_ankle={exo_ankle_id}, hip={hip_actuator_id}")
    #         self._printed_actuator_indices = True

    def initialize_model(self):
        """Initialize MuJoCo model and data."""
        # Get model parameters
        xml_path = self.config['xml_path']
        translation_friction_constant = self.config['translation_friction_constant']
        rolling_friction_constant = self.config['rolling_friction_constant']
        M_total = self.config['M_total']
        H_total = self.config['H_total']
        
        # Prepare model parameters
        tree = ET.parse(xml_path)
        root = tree.getroot()
        h_f, m_feet, m_body, l_COM, l_foot, a, self.K_p = calculate_kp_and_geom(M_total, H_total)
        
        # Set geometry parameters
        set_geometry_params(
            root, 
            m_feet, 
            m_body, 
            l_COM, 
            l_foot, 
            a, 
            H_total, 
            h_f, 
            translation_friction_constant, 
            rolling_friction_constant
        )

        # Write modified model
        literature_model = self.config['lit_xml_file']
        tree.write(literature_model)
        
        # Load model and create data
        self.model = mj.MjModel.from_xml_path(literature_model)
        self.data = mj.MjData(self.model)
        
        # Set model parameters
        self.model.opt.timestep = self.config['simulation_timestep']
        self.model.opt.gravity = np.array([0, 0, self.config['gravity']])
        
        # Get joint IDs
        self.ankle_joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "ankle_hinge")
        self.hip_joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "hip_hinge")
        
        # Set foot initial conditions
        self.data.qvel[0] = self.config['foot_rotation_initial_velocity']
        self.data.qvel[1] = self.config['foot_x_initial_velocity']
        self.data.qvel[2] = self.config['foot_z_initial_velocity']
        self.data.qpos[0] = self.config['foot_angle_initial_position_radians']
        
        # Set ankle initial conditions
        self.data.qpos[self.ankle_joint_id] = self.config['ankle_initial_position_radians']
        self.data.qvel[self.ankle_joint_id] = self.config['ankle_initial_velocity']
        
        # Set hip initial conditions
        self.data.qpos[self.hip_joint_id] = self.config['hip_initial_position_radians']
        self.data.qvel[self.hip_joint_id] = self.config['hip_initial_velocity']
        
        # Set setpoints
        self.ankle_position_setpoint = self.config.get('ankle_position_setpoint_radians', 0.0)
        self.hip_position_setpoint = self.config.get('hip_position_setpoint_radians', 0.0)

        # Call forward to update derived quantities
        mj.mj_forward(self.model, self.data)
        
        # Initialize controllers
        self.initialize_controllers(self.model, self.data)
        
        print("Model initialized successfully")

    def initialize_renderer(self):
        """Initialize renderer if visualization is enabled."""
        if not self.visualization_flag:
            print("Visualization disabled, skipping renderer initialization")
            return
            
        self.renderer = MujocoRenderer()
        self.renderer.setup_visualization(self.model, self.config)
        
        if self.mp4_flag:
            self.renderer.start_recording()
            
        print("Renderer initialized successfully")

    def initialize_perturbation(self):
        """Initialize perturbation if enabled."""
        if not self.config['apply_perturbation']:
            print("Perturbation disabled")
            return
            
        self.perturbation = create_perturbation(self.config)
        self.perturbation_thread = self.perturbation.start(perturbation_queue)
        print(f"Perturbation initialized: {type(self.perturbation).__name__}")
        
    def initialize(self):
        """Initialize all simulation components."""
        # Initialize logger and plotter
        self.logger.create_standard_datasets()
        self.logger.save_config(self.params)
        self.plotter = DataPlotter(self.logger.run_dir)
        
        # Initialize model, renderer, and perturbation
        self.initialize_model()
        original_time = self.data.time
        self.data.time = 0.0
        # Log initial state at t=0 before any physics steps
        self._log_simulation_data()

        # Restore original time
        self.data.time = original_time

        self.initialize_renderer()
        self.initialize_perturbation()
        
        print(f"Simulation duration: {self.config['simend']} seconds")
        
    def _log_simulation_data(self, custom_time=None):
        """Log all simulation data for the current timestep."""

        # Use custom time if provided, otherwise use simulation time
        time_value = custom_time if custom_time is not None else self.data.time
        # Extract actuator IDs
        human_actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "human_ankle_actuator")
        exo_actuator_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, "exo_ankle_actuator")

        # Get torque values
        human_torque_executed = self.data.actuator_force[human_actuator_id] 
        exo_torque_executed = self.data.actuator_force[exo_actuator_id] 
        ankle_torque_executed = self.data.qfrc_actuator[self.ankle_joint_id]
        gravity_torque = self.data.qfrc_bias[self.ankle_joint_id]
        
        # Log all the data with the precise time
        self.logger.log_data("human_torque", np.array([time_value, human_torque_executed]))
        self.logger.log_data("exo_torque", np.array([self.data.time, exo_torque_executed]))
        self.logger.log_data("ankle_torque", np.array([self.data.time, ankle_torque_executed]))
        self.logger.log_data("gravity_torque", np.array([self.data.time, gravity_torque]))
        self.logger.log_data("control_torque", np.array([self.data.time, self.data.qfrc_actuator[self.ankle_joint_id]]))
        
        # Joint data
        self.logger.log_data("joint_position", np.array([self.data.time, 180/np.pi*self.data.qpos[self.ankle_joint_id]]))
        self.logger.log_data("joint_velocity", np.array([self.data.time, 180/np.pi*self.data.qvel[self.ankle_joint_id]]))
        self.logger.log_data("goal_position", np.array([self.data.time, 180/np.pi*self.ankle_position_setpoint]))
        
        # Constraint and contact forces
        self.logger.log_data("constraint_force", np.array([
            self.data.time, 
            self.data.qfrc_constraint[0], 
            self.data.qfrc_constraint[1], 
            self.data.qfrc_constraint[2], 
            self.data.qfrc_constraint[3]
        ]))
        self.logger.log_data("contact_force", np.array([
            self.data.time, 
            self.data.sensordata[1], 
            self.data.sensordata[2]
        ]))
        
        # COM data
        com = self.data.xipos[self.human_body_id]
        self.logger.log_data("body_com", np.array([self.data.time, com[0], com[1], com[2]]))

        # Log RTD data - add this at the end
        if hasattr(self.human_controller, 'current_rtd'):
            self.logger.log_data("human_rtd", np.array([
                self.data.time, 
                self.human_controller.current_rtd, 
                self.human_controller.current_rtd_limit
            ]))
        
    def _simulation_step(self):
        """Execute one step of the simulation."""
        # Handle perturbation
        if not perturbation_queue.empty():
            x_perturbation = perturbation_queue.get()
            self.data.xfrc_applied[2] = [x_perturbation, 0, 0, 0., 0., 0.]
        else:
            self.data.xfrc_applied[2] = [0, 0, 0, 0., 0., 0.]

        # MODIFIED: First compute control based on current state
        # This ensures we're using the apply-then-step pattern like MATLAB
        if self.config['controller_flag']:
            self.controller(self.model, self.data)
            
        # Log perturbation before stepping
        if not perturbation_queue.empty():
            self.logger.log_data("perturbation", np.array([self.data.time, x_perturbation]))
        else:
            self.logger.log_data("perturbation", np.array([self.data.time, 0.]))

        # Step physics (now using the torque we just computed)
        mj.mj_step(self.model, self.data)
        
        # Log data after stepping
        self._log_simulation_data()
            
    def _generate_plots(self):
        """Generate plots from the collected data."""
        print("Generating plots...")
        
        # Load data from logger to plotter
        for name, data_array in self.logger.data_arrays.items():
            # Skip the first empty row when passing to plotter
            self.plotter.set_data(name, data_array[1:])
        
        # Generate dashboard plot
        self.plotter.plot_dashboard(show=self.plot_flag, save=True)
        
        # Generate individual plots
        self.plotter.plot_joint_state(show=self.plot_flag, save=True)
        self.plotter.plot_torques(show=self.plot_flag, save=True)
        self.plotter.plot_gravity_compensation(show=self.plot_flag, save=True)
        
        # If perturbations were applied, plot the response
        perturbation_data = self.logger.get_dataset("perturbation")
        if perturbation_data is not None and np.any(perturbation_data[:, 1] != 0):
            self.plotter.plot_perturbation_response(show=self.plot_flag, save=True)
            
        print(f"Plots saved to: {os.path.join(self.logger.run_dir, 'plots')}")

    def run(self):
        """Run the simulation."""
        simend = self.config['simend']

        # Record initial state at t=0 before any physics steps
        original_time = self.data.time  # Save current time
        self.data.time = 0.0           # Explicitly set to t=0

        if self.config['controller_flag']:
            self.controller(self.model, self.data)
        
        # -- (2) Optionally do a forward pass so MuJoCo sees the new ctrl
        mj.mj_forward(self.model, self.data)

        self._log_simulation_data()    # Log initial state
        self.data.time = original_time # Restore original time
        
        # # Apply initial control before any steps occur
        # if self.config['controller_flag']:
        #     self.controller(self.model, self.data)
        
        start_time = time.time()
        
        # Main simulation loop with visualization
        if self.renderer:
            while not self.renderer.window_should_close():
                simstart = self.data.time
                
                # Run physics at a higher rate than rendering (60 fps)
                while (self.data.time - simstart < 1.0/60.0):
                    # Step physics and record data
                    self._simulation_step()
                    
                    # Check if simulation time exceeded
                    if self.data.time >= simend:
                        break
                
                # Check if simulation time exceeded
                if self.data.time >= simend:
                    break
                    
                # Render the current state
                self.renderer.render(self.model, self.data)
        
        # Without visualization - run the simulation faster
        else:
            while self.data.time < simend:
                self._simulation_step()
        
        print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
        
        # Clean up
        if self.perturbation_thread:
            self.perturbation.stop()
            self.perturbation_thread.join()
            print("Perturbation thread terminated")
            
        if self.renderer:
            if self.mp4_flag:
                self.renderer.save_video(
                    self.config['mp4_file_name'], 
                    self.config['mp4_fps']
                )
            self.renderer.close()
        
        # Save all collected data
        self.logger.save_all()
        
        # Generate plots if enabled
        if self.plot_flag:
            self._generate_plots()


if __name__ == "__main__":
    simulation = AnkleExoSimulation('config.yaml')
    simulation.initialize()
    simulation.run()