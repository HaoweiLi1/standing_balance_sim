from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import mujoco
from typing import Tuple, Optional
from config_manager import HumanControllerConfig, ExoControllerConfig, MRTDConfig

@dataclass
class ControllerState:
    """State information for controllers"""
    time: float
    joint_pos: float
    joint_vel: float
    target_pos: float
    human_torque: float
    exo_torque: float

class HumanController(ABC):
    """Abstract base class for human controllers"""
    def __init__(self, config: HumanControllerConfig, mrtd_config: MRTDConfig):
        """Initialize controller
        
        Args:
            config: Human controller configuration
            mrtd_config: MRTD configuration
        """
        self.config = config
        self.mrtd_config = mrtd_config
        self._prev_torque = 0.0
        self._prev_error = 0.0
        
    def apply_mrtd_constraint(self, desired_torque: float, delta_time: float) -> float:
        """Apply MRTD constraint to desired torque
        
        Args:
            desired_torque: Desired torque before MRTD constraint
            delta_time: Time step
            
        Returns:
            Constrained torque value
        """
        if not self.mrtd_config.enable:
            return desired_torque
            
        # Calculate torque change
        delta_torque = desired_torque - self._prev_torque
        
        # Apply MRTD constraints
        if delta_torque > 0:
            max_increase = self.mrtd_config.dorsiflexion * delta_time
            delta_torque = min(delta_torque, max_increase)
        else:
            max_decrease = -self.mrtd_config.plantarflexion * delta_time
            delta_torque = max(delta_torque, max_decrease)
            
        # Update torque
        constrained_torque = self._prev_torque + delta_torque
        self._prev_torque = constrained_torque
        
        return constrained_torque
    
    @abstractmethod
    def compute_control(self, model: mujoco.MjModel, data: mujoco.MjData,
                       target_pos: float) -> float:
        """Compute human control torque
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            target_pos: Target position
            
        Returns:
            Computed human torque
        """
        pass

class GravityCompensationController(HumanController):
    """Gravity compensation controller implementation"""
    def compute_control(self, model: mujoco.MjModel, data: mujoco.MjData,
                       target_pos: float) -> float:
        error = target_pos - data.sensordata[0]
        gravity_torque = data.qfrc_bias[3]
        
        # Calculate desired torque
        desired_torque = self.config.gravity_compensation.Kp * error - gravity_torque
        
        # Apply MRTD constraint
        return self.apply_mrtd_constraint(desired_torque, model.opt.timestep)

class PDController(HumanController):
    """PD controller implementation"""
    def compute_control(self, model: mujoco.MjModel, data: mujoco.MjData,
                       target_pos: float) -> float:
        # Calculate error and derivative
        error = target_pos - data.sensordata[0]
        delta_error = error - self._prev_error
        delta_time = model.opt.timestep
        derivative = delta_error / delta_time
        
        # Calculate desired torque
        desired_torque = (self.config.pd.Kp * error + 
                         self.config.pd.Kd * derivative)
        
        # Store current error for next iteration
        self._prev_error = error
        
        # Apply MRTD constraint
        return self.apply_mrtd_constraint(desired_torque, delta_time)

class LQRController(HumanController):
    """LQR controller implementation"""
    def __init__(self, config: HumanControllerConfig, mrtd_config: MRTDConfig):
        super().__init__(config, mrtd_config)
        self._K = None
        
    def _initialize_lqr(self, m: float, l: float, g: float, b: float):
        """Initialize LQR gains"""
        # System parameters
        I = m * l**2  # Moment of inertia
        
        # State space matrices
        A = np.array([[0, 1],
                     [-m*g*l/I, -b/I]])
        B = np.array([[0],
                     [1/I]])
        
        # LQR weights
        Q = np.diag([self.config.lqr.Q_angle, 
                     self.config.lqr.Q_velocity])
        R = np.array([[self.config.lqr.R]])
        
        # Solve Riccati equation
        P = np.linalg.solve_continuous_are(A, B, Q, R)
        
        # Compute LQR gain
        self._K = np.linalg.solve(R, B.T @ P)
    
    def compute_control(self, model: mujoco.MjModel, data: mujoco.MjData,
                       target_pos: float) -> float:
        # Initialize LQR gains if needed
        if self._K is None:
            m = 80.0 - 2 * 0.0145 * 80.0
            l = 0.575 * 1.78
            g = 9.81
            b = 2.5
            self._initialize_lqr(m, l, g, b)
        
        # Current state
        current_state = np.array([
            data.sensordata[0],
            data.qvel[3]
        ])
        
        # Error state
        error_state = np.array([
            current_state[0] - target_pos,
            current_state[1]
        ])
        
        # Calculate desired torque
        desired_torque = -float(np.dot(self._K, error_state))
        
        # Apply MRTD constraint
        return self.apply_mrtd_constraint(desired_torque, model.opt.timestep)

class ExoController:
    """Exoskeleton controller implementation"""
    def __init__(self, config: ExoControllerConfig):
        """Initialize controller
        
        Args:
            config: Exoskeleton controller configuration
        """
        self.config = config
        
    def compute_control(self, model: mujoco.MjModel, data: mujoco.MjData,
                       target_pos: float) -> float:
        """Compute exoskeleton control torque
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            target_pos: Target position
            
        Returns:
            Computed exoskeleton torque
        """
        if self.config.type == "none":
            return 0.0
            
        if self.config.type == "pd":
            # Calculate error
            error = target_pos - data.sensordata[0]
            current_velocity = data.qvel[3]
            
            # PD control
            return (self.config.pd.Kp * error - 
                   self.config.pd.Kd * current_velocity)
        
        return 0.0

def create_human_controller(config: HumanControllerConfig, 
                          mrtd_config: MRTDConfig) -> Optional[HumanController]:
    """Create human controller based on configuration
    
    Args:
        config: Human controller configuration
        mrtd_config: MRTD configuration
        
    Returns:
        Controller instance or None if type is "none"
    """
    if config.type == "none":
        return None
        
    if config.type == "gravity_compensation":
        return GravityCompensationController(config, mrtd_config)
    elif config.type == "pd":
        return PDController(config, mrtd_config)
    elif config.type == "lqr":
        return LQRController(config, mrtd_config)
    else:
        raise ValueError(f"Unknown controller type: {config.type}")

def create_exo_controller(config: ExoControllerConfig) -> ExoController:
    """Create exoskeleton controller
    
    Args:
        config: Exoskeleton controller configuration
        
    Returns:
        Exoskeleton controller instance
    """
    return ExoController(config)