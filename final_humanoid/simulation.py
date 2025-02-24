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
import threading
import csv
import time
from queue import Queue
import xml.etree.ElementTree as ET
import imageio
import yaml


from xml_utilities import calculate_kp_and_geom, set_geometry_params

from data_logger import DataLogger
from data_plotter import DataPlotter
from controllers import create_human_controller, create_exo_controller
from perturbation import create_perturbation


plt.rcParams['text.usetex'] = True

mpl.rcParams.update(mpl.rcParamsDefault)

perturbation_queue = Queue()
control_log_queue = Queue()
counter_queue = Queue()
perturbation_datalogger_queue = Queue()

class ankleTorqueControl:

    def __init__(self, plot_flag):
        print('simulation class object created')
        self.plot_flag = plot_flag
        self.mp4_flag = False
        self.pertcount = 0
        self.human_controller = None
        self.exo_controller = None
        self.logger = None
        self.plotter = None

    def initialize_controllers(self, model, data, config):
        """Initialize controllers based on configuration."""
        # Get human controller configuration
        human_config = config['controllers']['human']
        human_type = human_config['type']
        
        # Prepare parameters for human controller
        human_params = {
            'max_torque_df': human_config['max_torque_df'],
            'max_torque_pf': human_config['max_torque_pf'],
            'mass': config['M_total'],
            'leg_length': 0.575 * config['H_total'],
            'mrtd_df': human_config.get('mrtd_df'),
            'mrtd_pf': human_config.get('mrtd_pf')
        }
        
        # Add controller-specific parameters
        if human_type == "LQR":
            lqr_params = human_config['lqr_params']
            human_params.update({
                'Q_angle': lqr_params['Q_angle'],
                'Q_velocity': lqr_params['Q_velocity'],
                'R': lqr_params['R']
            })
        elif human_type == "PD":
            pd_params = human_config['pd_params']
            human_params.update({
                'kp': pd_params['kp'],
                'kd': pd_params['kd']
            })
            
        # Create human controller using factory function
        self.human_controller = create_human_controller(
            human_type, model, data, human_params
        )
        
        # Get exo controller configuration
        exo_config = config['controllers']['exo']
        exo_type = exo_config['type']
        
        # Prepare parameters for exo controller
        exo_params = {
            'max_torque': exo_config.get('max_torque', 0)
        }
        
        # Add controller-specific parameters
        if exo_type == "PD":
            pd_params = exo_config['pd_params']
            exo_params.update({
                'kp': pd_params['kp'],
                'kd': pd_params['kd']
            })
        elif exo_type == "GC":
            gc_params = exo_config['gc_params']
            exo_params.update({
                'compensation_factor': gc_params['compensation_factor']
            })
            
        # Create exo controller using factory function
        self.exo_controller = create_exo_controller(
            exo_type, model, data, exo_params
        )

    def controller(self, model, data):
        """Controller function for the leg."""
        # Get current state
        state = np.array([
            data.sensordata[0],  # Joint angle
            data.qvel[3]         # Joint velocity
        ])
        
        # Compute human control
        human_torque = self.human_controller.compute_control(
            state=state,
            target=self.ankle_position_setpoint
        )
        data.ctrl[0] = human_torque
        
        # Compute exo control
        exo_torque = self.exo_controller.compute_control(
            state=state,
            target=self.ankle_position_setpoint
        )
        data.ctrl[1] = exo_torque


    def load_params_from_yaml(self, file_path):
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    
    def _initialize_simulation(self, params):
        """
        Initialize the simulation environment, model, and controllers.
        
        Args:
            params: Configuration parameters from config.yaml
            
        Returns:
            tuple: (model, data, ankle_joint_id, human_body_id, window, simend, perturbation_thread)
        """
        # Get key configuration parameters
        xml_path = params['config']['xml_path']
        simend = params['config']['simend']
        M_total = params['config']['M_total']
        H_total = params['config']['H_total']
        perturbation_time = params['config']['perturbation_time']
        perturbation_magnitude = params['config']['perturbation_magnitude']
        perturbation_period = params['config']['perturbation_period']
        translation_friction_constant = params['config']['translation_friction_constant']
        rolling_friction_constant = params['config']['rolling_friction_constant']
        
        # Set ankle position setpoint
        self.ankle_position_setpoint = params['config']['ankle_position_setpoint_radians']
        ankle_joint_initial_position = params['config']['ankle_initial_position_radians']
        
        # Prepare model parameters
        tree = ET.parse(xml_path)
        root = tree.getroot()
        h_f, m_feet, m_body, l_COM, l_foot, a, self.K_p = calculate_kp_and_geom(M_total, H_total)
        set_geometry_params(root, 
                          m_feet, 
                          m_body, 
                          l_COM, 
                          l_foot, 
                          a, 
                          H_total, 
                          h_f, 
                          translation_friction_constant, 
                          rolling_friction_constant)

        # Write modified model
        literature_model = params['config']['lit_xml_file']
        tree.write(literature_model)
        
        # Load model and create data
        model = mj.MjModel.from_xml_path(literature_model)
        data = mj.MjData(model)
        
        # Setup visualization
        cam = mj.MjvCamera()
        opt = mj.MjvOption()
        
        # Initialize GLFW
        glfw.init()
        window = glfw.create_window(1200, 912, "Demo", None, None)
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        # Initialize visualization settings
        mj.mjv_defaultCamera(cam)
        mj.mjv_defaultOption(opt)
        opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = params['config']['visualize_contact_force']
        opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = params['config']['visualize_perturbation_force']
        opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = params['config']['visualize_joints']
        opt.flags[mj.mjtVisFlag.mjVIS_ACTUATOR] = params['config']['visualize_actuators']
        opt.flags[mj.mjtVisFlag.mjVIS_COM] = params['config']['visualize_center_of_mass']
        
        # Set model parameters
        model.opt.timestep = params['config']['simulation_timestep']
        model.opt.gravity = np.array([0, 0, params['config']['gravity']])
        
        # Configure visualization parameters
        model.vis.map.force = 0.25
        model.vis.map.torque = 0.1
        model.vis.scale.contactwidth = 0.05
        model.vis.scale.contactheight = 0.01
        model.vis.scale.forcewidth = 0.03
        model.vis.scale.com = 0.2
        model.vis.scale.actuatorwidth = 0.1
        model.vis.scale.actuatorlength = 0.1
        model.vis.scale.jointwidth = 0.025
        model.vis.scale.framelength = 0.25
        model.vis.scale.framewidth = 0.05
        model.vis.rgba.contactforce = np.array([0.7, 0., 0., 0.5], dtype=np.float32)
        model.vis.rgba.force = np.array([0., 0.7, 0., 0.5], dtype=np.float32)
        model.vis.rgba.joint = np.array([0.2, 1, 0.1, 0.8])
        model.vis.rgba.actuatorpositive = np.array([0., 0.9, 0., 0.5])
        model.vis.rgba.actuatornegative = np.array([0.9, 0., 0., 0.5])
        model.vis.rgba.com = np.array([1.,0.647,0.,0.5])

        # Create scene and context
        scene = mj.MjvScene(model, maxgeom=10000)
        context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # Set camera configuration
        cam.azimuth = params['config']['camera_azimuth']
        cam.distance = params['config']['camera_distance']
        cam.elevation = params['config']['camera_elevation']
        lookat_string_xyz = params['config']['camera_lookat_xyz']
        res = tuple(map(float, lookat_string_xyz.split(', ')))
        cam.lookat = np.array([res[0], res[1], res[2]])

        # Set initial conditions
        data.qvel[0] = 0  # hinge joint at top of body
        data.qvel[1] = 0  # slide / prismatic joint at top of body in x direction
        data.qvel[2] = 0  # slide / prismatic joint at top of body in z direction
        data.qvel[3] = 0  # hinge joint at ankle
        data.qpos[0] = params['config']['foot_angle_initial_position_radians']
        data.qpos[3] = ankle_joint_initial_position

        # Get important joint and body IDs
        ankle_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "ankle_hinge")
        human_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "long_link_body")

        # Initialize controllers
        self.initialize_controllers(model, data, params['config'])
        
        # Setup controller if enabled
        control_flag = params['config']['controller_flag']
        if control_flag:
            mj.set_mjcb_control(self.controller)

        perturbation_object = create_perturbation(params['config'])

        # Setup perturbation thread if enabled
        perturbation_thread = None
        if params['config']['apply_perturbation']:
            perturbation_thread = perturbation_object.start(perturbation_queue)
            print(f'Started {type(perturbation_object).__name__}')
            
        # Print simulation duration
        print(f'simulation duration: {simend} seconds')
        
        return model, data, ankle_joint_id, human_body_id, window, scene, context, cam, opt, simend, perturbation_thread

    def _log_simulation_data(self, model, data, ankle_joint_id, human_body_id):
        """Log all simulation data to the data logger."""
        # Extract actuator IDs
        human_actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "human_ankle_actuator")
        exo_actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "exo_ankle_actuator")
        
        # Get torque values
        human_torque_executed = data.actuator_force[human_actuator_id] * 15
        exo_torque_executed = data.actuator_force[exo_actuator_id] * 10
        ankle_torque_executed = data.qfrc_actuator[ankle_joint_id]
        gravity_torque = data.qfrc_bias[ankle_joint_id]
        
        # Log all the data
        self.logger.log_data("human_torque", np.array([data.time, human_torque_executed]))
        self.logger.log_data("exo_torque", np.array([data.time, exo_torque_executed]))
        self.logger.log_data("ankle_torque", np.array([data.time, ankle_torque_executed]))
        self.logger.log_data("gravity_torque", np.array([data.time, gravity_torque]))
        self.logger.log_data("control_torque", np.array([data.time, data.qfrc_actuator[ankle_joint_id]]))
        
        # Joint data
        self.logger.log_data("joint_position", np.array([data.time, 180/np.pi*data.qpos[ankle_joint_id]]))
        self.logger.log_data("joint_velocity", np.array([data.time, 180/np.pi*data.qvel[ankle_joint_id]]))
        self.logger.log_data("goal_position", np.array([data.time, 180/np.pi*self.ankle_position_setpoint]))
        
        # Constraint and contact forces
        self.logger.log_data("constraint_force", np.array([
            data.time, 
            data.qfrc_constraint[0], 
            data.qfrc_constraint[1], 
            data.qfrc_constraint[2], 
            data.qfrc_constraint[3]
        ]))
        self.logger.log_data("contact_force", np.array([
            data.time, 
            data.sensordata[1], 
            data.sensordata[2]
        ]))
        
        # COM data
        com = data.xipos[human_body_id]
        self.logger.log_data("body_com", np.array([data.time, com[0], com[1], com[2]]))
        
    def _generate_plots(self):
        """Generate plots from the collected data."""
        print("Generating plots...")
        
        # Load data from logger to plotter
        for name, data_array in self.logger.data_arrays.items():
            # Skip the first empty row when passing to plotter
            self.plotter.set_data(name, data_array[1:])
        
        # Create plots subfolder
        plots_dir = os.path.join(self.logger.run_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Generate dashboard plot
        self.plotter.plot_dashboard(show=True, save=True)
        
        # Generate individual plots
        self.plotter.plot_joint_state(show=True, save=True)
        self.plotter.plot_torques(show=True, save=True)
        self.plotter.plot_gravity_compensation(show=True, save=True)
        
        # If perturbations were applied, plot the response
        perturbation_data = self.logger.get_dataset("perturbation")
        if perturbation_data is not None and np.any(perturbation_data[:, 1] != 0):
            self.plotter.plot_perturbation_response(show=True, save=True)
            
        print(f"Plots saved to: {plots_dir}")


    def run(self):

        # precallocating arrays and queues for data sharing and data collection
        # Initialize the data logger
        self.logger = DataLogger()
        # Create standard datasets for ankle exo simulation
        self.logger.create_standard_datasets()

        # Initialize the data plotter
        self.plotter = DataPlotter(self.logger.run_dir)
        
        # Rest of initialization code remains unchanged
        params = self.load_params_from_yaml('config.yaml')
        
        # Save the configuration
        self.logger.save_config(params)
        
        params = self.load_params_from_yaml('config.yaml')

        self.plot_flag = params['config']['plotter_flag']
        self.mp4_flag = params['config']['mp4_flag']
        
        self.ankle_position_setpoint = params['config']['ankle_position_setpoint_radians']
        ankle_joint_initial_position = params['config']['ankle_initial_position_radians'] # ankle joint initial angle
        
        # Define parameters for creating and storing an MP4 file of the rendered simulation
        video_file = params['config']['mp4_file_name']
        video_fps = params['config']['mp4_fps']
        
        frames = [] # list to store frames
        
        # Initialize the simulation environment and get required objects
        model, data, ankle_joint_id, human_body_id, window, scene, context, cam, opt, simend, perturbation_thread = self._initialize_simulation(params)
        
        # Main simulation loop
        start_time = time.time()

        
        while not glfw.window_should_close(window): # glfw.window_should_close() indicates whether or not the user has closed the window
            simstart = data.time
            # print(data.time)
            
            while (data.time - simstart < 1.0/60.0):
                
                if not perturbation_queue.empty():
                    # if not perturbation_queue.empty():
                    # print(f"perturbation: {perturbation_queue.get()}, time: {time.time()-start}")
            
                    x_perturbation = perturbation_queue.get()
                    
                    data.xfrc_applied[2] = [x_perturbation, 0, 0, 0., 0., 0.]
                    self.logger.log_data("perturbation", np.array([data.time, x_perturbation]))
                else:
                    # if the perturbation queue is empty then the perturbation is zero
                    # print(time.time())
                    data.xfrc_applied[2] = [0, 0, 0, 0., 0., 0.]
                    self.logger.log_data("perturbation", np.array([data.time, 0.]))

                mj.mj_step(model, data)

                self._log_simulation_data(model, data, ankle_joint_id, human_body_id)
                
            
            if (data.time>=simend):
                
                self.impulse_thread_exit_flag = True
                if params['config']['apply_perturbation']:
                    perturbation_thread.join()
                    print('perturbation thread terminated')
                break;
            
            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Update scene and render
            mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
            mj.mjr_render(viewport, scene, context)


            ###### CODE TO CAPTURE FRAMES FOR MP4 VIDEO GENERATION ##########

            if self.mp4_flag:
                rgb_array = np.empty((viewport_height, viewport_width, 3), dtype=np.uint8)
                depth_array = np.empty((viewport_height, viewport_width), dtype=np.float32)
                mj.mjr_readPixels(rgb=rgb_array, depth=depth_array, viewport=viewport, con=context)
                rgb_array = np.flipud(rgb_array)
                frames.append(rgb_array)
            #################################################################

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        # this writes the list of frames we collected to an mp4 file and makes it a video
        if self.mp4_flag:
            imageio.mimwrite(video_file, frames, fps=video_fps)
        print('terminated')
        glfw.terminate()

        # Save all collected data
        self.logger.save_all()
        
        # Handle plotting if enabled
        if self.plot_flag:
            self._generate_plots()


if __name__ == "__main__":

    sim_class = ankleTorqueControl(True)
    sim_class.run()
 