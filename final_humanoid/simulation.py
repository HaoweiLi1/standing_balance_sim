import mujoco
from mujoco.glfw import glfw
import numpy as np
import threading
import time
from pathlib import Path
import imageio
from typing import Optional, Tuple
import xml.etree.ElementTree as ET

from config_manager import ConfigManager
from controller_module import (create_human_controller, create_exo_controller)
from data_logger import SimulationLogger
from visualization import Visualizer
from xml_utilities import calculate_kp_and_geom, set_geometry_params

class MujocoSimulation:
    """Main simulation class for the ankle exoskeleton system."""
    
    def __init__(self, config_path: str):
        """Initialize simulation.
        
        Args:
            config_path: Path to configuration file.
        """
        # Load configurations
        self.config_manager = ConfigManager(config_path)
        (self.sim_config, self.model_config, self.human_controller_config,
         self.exo_controller_config, self.mrtd_config, self.pert_config,
         self.vis_config) = self.config_manager.get_all_configs()
        
        # Initialize components
        self._init_simulation()
        self._init_visualization()
        self._init_controller()
        self._init_data_logger()
        
        # State variables
        self.frame_buffer = []
        self.current_step = 0
        self.perturbation_thread = None
        self.perturbation_exit_flag = False
        self.current_perturbation = 0.0
        
    def _init_simulation(self):
        """Initialize MuJoCo simulation."""
        # Load and modify XML
        tree = ET.parse(self.sim_config.xml_path)
        root = tree.getroot()
        
        # Calculate model parameters from literature equations
        h_f, m_feet, m_body, l_COM, l_foot, a, K_p = calculate_kp_and_geom(
            self.model_config.M_total,
            self.model_config.H_total
        )
        
        # Modify XML with calculated parameters and friction settings
        set_geometry_params(
            root,
            m_feet,
            m_body,
            l_COM,
            l_foot,
            a,
            self.model_config.H_total,
            h_f,
            self.sim_config.translation_friction_constant,
            self.sim_config.rolling_friction_constant
        )
        
        # Save the modified XML to lit_xml_file
        tree.write(self.sim_config.lit_xml_file)
        
        # Load model and create data
        self.model = mujoco.MjModel.from_xml_path(self.sim_config.lit_xml_file)
        self.data = mujoco.MjData(self.model)
        
        # Set simulation parameters
        self.model.opt.timestep = self.sim_config.simulation_timestep
        self.model.opt.gravity = np.array([0, 0, self.sim_config.gravity])
        
        # Set visualization parameters
        self._set_visualization_parameters()
        
        # Get essential IDs from MuJoCo model
        self._get_mujoco_ids()
        
        # Set initial conditions for joint positions and velocities
        self._set_initial_state()
        
    def _set_visualization_parameters(self):
        """Set visualization parameters for MuJoCo model."""
        self.model.vis.map.force = 0.25
        self.model.vis.map.torque = 0.1
        self.model.vis.scale.contactwidth = 0.05
        self.model.vis.scale.contactheight = 0.01
        self.model.vis.scale.forcewidth = 0.03
        self.model.vis.scale.com = 0.2
        self.model.vis.scale.actuatorwidth = 0.1
        self.model.vis.scale.actuatorlength = 0.1
        self.model.vis.scale.jointwidth = 0.025
        self.model.vis.scale.framelength = 0.25
        self.model.vis.scale.framewidth = 0.05
        
        self.model.vis.rgba.contactforce = np.array([0.7, 0., 0., 0.5], dtype=np.float32)
        self.model.vis.rgba.force = np.array([0., 0.7, 0., 0.5], dtype=np.float32)
        self.model.vis.rgba.joint = np.array([0.2, 1, 0.1, 0.8])
        self.model.vis.rgba.actuatorpositive = np.array([0., 0.9, 0., 0.5])
        self.model.vis.rgba.actuatornegative = np.array([0.9, 0., 0., 0.5])
        self.model.vis.rgba.com = np.array([1., 0.647, 0., 0.5])
        
    def _get_mujoco_ids(self):
        """Get MuJoCo IDs for joints and actuators."""
        self.ankle_joint_id = mujoco.mj_name2id(self.model, 
                                            mujoco.mjtObj.mjOBJ_JOINT, 
                                            "ankle_hinge")
        self.human_body_id = mujoco.mj_name2id(self.model, 
                                           mujoco.mjtObj.mjOBJ_BODY, 
                                           "long_link_body")
        self.human_actuator_id = mujoco.mj_name2id(self.model, 
                                               mujoco.mjtObj.mjOBJ_ACTUATOR, 
                                               "human_ankle_actuator")
        self.exo_actuator_id = mujoco.mj_name2id(self.model, 
                                             mujoco.mjtObj.mjOBJ_ACTUATOR, 
                                             "exo_ankle_actuator")
        
    def _set_initial_state(self):
        """Set initial simulation state."""
        # Set all joint velocities to zero
        self.data.qvel[0:4] = 0.0
        
        # Set initial joint positions: foot angle and ankle joint angle
        self.data.qpos[0] = self.model_config.foot_angle_initial_position_radians
        self.data.qpos[3] = self.model_config.ankle_initial_position_radians
        
    def _init_visualization(self):
        """Initialize visualization components."""
        glfw.init()
        self.window = glfw.create_window(1200, 912, "Ankle Exoskeleton Simulation", None, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        
        # Set camera configuration from visualization config
        self.cam.azimuth = self.vis_config.camera_azimuth
        self.cam.distance = self.vis_config.camera_distance
        self.cam.elevation = self.vis_config.camera_elevation
        lookat = [float(x) for x in self.vis_config.camera_lookat_xyz.split(',')]
        self.cam.lookat = np.array(lookat)
        
        # Set visualization flags
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = self.vis_config.visualize_contact_force
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = self.vis_config.visualize_perturbation_force
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = self.vis_config.visualize_joints
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = self.vis_config.visualize_actuators
        self.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = self.vis_config.visualize_center_of_mass
        
        # Create scene and rendering context
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        
    def _init_controller(self):
        """Initialize controllers."""
        self.human_controller = create_human_controller(
            self.human_controller_config,
            self.mrtd_config
        )
        self.exo_controller = create_exo_controller(
            self.exo_controller_config
        )
        
    def _init_data_logger(self):
        """Initialize data logger."""
        # Allocate one extra step to cover simulation time from 0 to simend inclusive.
        expected_steps = int(self.sim_config.simend / self.sim_config.simulation_timestep) + 1
        self.logger = SimulationLogger()
        self.logger.initialize_storage(expected_steps)
        
    def _start_perturbation_thread(self):
        """Start perturbation generation thread if enabled."""
        if self.pert_config.apply_perturbation:
            self.perturbation_exit_flag = False
            self.perturbation_thread = threading.Thread(
                target=self._generate_perturbation,
                daemon=True
            )
            self.perturbation_thread.start()
            
    def _generate_perturbation(self):
        """Generate perturbation forces."""
        while not self.perturbation_exit_flag:
            wait_time = np.random.uniform(
                self.pert_config.perturbation_period,
                self.pert_config.perturbation_period + 1
            )
            time.sleep(wait_time)
            
            perturbation = np.random.uniform(self.pert_config.perturbation_magnitude,
                                             self.pert_config.perturbation_magnitude)
            direction = np.random.choice([-1, 1])
            
            start_time = time.time()
            end_time = start_time + self.pert_config.perturbation_time
            
            while time.time() < end_time and not self.perturbation_exit_flag:
                self.current_perturbation = direction * perturbation
            self.current_perturbation = 0.0
            
    def _apply_control(self) -> Tuple[float, float]:
        """
        Apply controllers and get control signals.

        Returns:
            Tuple of (human_torque, exo_torque) in physical units (Nm).
        """
        target_pos = self.model_config.ankle_position_setpoint_radians
        
        human_torque = (self.human_controller.compute_control(self.model, self.data, target_pos)
                        if self.human_controller is not None else 0.0)
        exo_torque = self.exo_controller.compute_control(self.model, self.data, target_pos)
        
        # Convert physical torque (Nm) to control command using actuator gear:
        # Human actuator gear = 15, exoskeleton actuator gear = 10.
        self.data.ctrl[0] = human_torque 
        self.data.ctrl[1] = exo_torque 
        
        return human_torque, exo_torque
        
    def _log_data(self, step: int, human_torque: float, exo_torque: float):
        """Log simulation data for the current step."""
        if step >= self.logger.joint_data["position"].time.shape[0]:
            return
        self.logger.log_step(
            step=step,
            time_val=self.data.time,
            data={
                "joint_position": self.data.sensordata[0],
                "target_position": self.model_config.ankle_position_setpoint_radians,
                "joint_velocity": self.data.qvel[3],
                "human_torque": human_torque,
                "exo_torque": exo_torque,
                "total_torque": self.data.qfrc_actuator[3],
                "perturbation": self.current_perturbation,
                "contact_forces": [self.data.sensordata[1], self.data.sensordata[2]],
                "gravity_torque": self.data.qfrc_bias[3]
            }
        )
        
    def _render_frame(self):
        """Render a single frame and capture it if video recording is enabled."""
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)
        
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene
        )
        mujoco.mjr_render(viewport, self.scene, self.context)
        
        if self.vis_config.mp4_flag:
            rgb_array = np.empty((viewport_height, viewport_width, 3), dtype=np.uint8)
            depth_array = np.empty((viewport_height, viewport_width), dtype=np.float32)
            mujoco.mjr_readPixels(rgb=rgb_array, depth=depth_array, viewport=viewport, con=self.context)
            self.frame_buffer.append(np.flipud(rgb_array))
            
        glfw.swap_buffers(self.window)
        glfw.poll_events()
        
    def run(self):
        """Run the simulation."""
        try:
            self._start_perturbation_thread()
            
            while not glfw.window_should_close(self.window):
                sim_start = self.data.time
                
                # Run simulation steps for approximately 1/60 second (one frame)
                while (self.data.time - sim_start < 1.0/60.0):
                    # Apply perturbation if active
                    if self.current_perturbation != 0.0:
                        self.data.xfrc_applied[2] = [self.current_perturbation, 0, 0, 0., 0., 0.]
                    else:
                        self.data.xfrc_applied[2] = [0, 0, 0, 0., 0., 0.]
                    
                    human_torque, exo_torque = self._apply_control()
                    mujoco.mj_step(self.model, self.data)
                    self._log_data(self.current_step, human_torque, exo_torque)
                    self.current_step += 1
                    
                    if self.data.time >= self.sim_config.simend:
                        return
                
                self._render_frame()
                
        finally:
            self._cleanup()
            
    def _cleanup(self):
        """Cleanup resources after simulation."""
        self.perturbation_exit_flag = True
        if self.perturbation_thread:
            self.perturbation_thread.join()
            
        self.logger.save_data()
        
        if self.vis_config.mp4_flag and self.frame_buffer:
            imageio.mimwrite(
                self.vis_config.mp4_file_name,
                self.frame_buffer,
                fps=self.vis_config.mp4_fps
            )
            
        glfw.terminate()
        
        if self.vis_config.plotter_flag:
            visualizer = Visualizer(self.logger)
            visualizer.plot_all()

if __name__ == "__main__":
    sim = MujocoSimulation("config_v1.yaml")
    sim.run()
