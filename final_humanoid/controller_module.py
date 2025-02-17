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
    """Abstract base class for human controllers."""
    def __init__(self, config: HumanControllerConfig, mrtd_config: MRTDConfig):
        """
        Initialize the human controller.

        Args:
            config: Human controller configuration.
            mrtd_config: MRTD configuration.
        """
        self.config = config
        self.mrtd_config = mrtd_config
        self._prev_torque = 0.0
        self._prev_error = 0.0

    def apply_mrtd_constraint(self, desired_torque: float, delta_time: float) -> float:
        """
        Apply MRTD constraint to the desired torque.

        Args:
            desired_torque: Desired torque before constraint.
            delta_time: Time step.

        Returns:
            Constrained torque value.
        """
        if not self.mrtd_config.enable:
            self._prev_torque = desired_torque
            return desired_torque

        # Calculate torque change
        delta_torque = desired_torque - self._prev_torque

        # Apply MRTD constraint for increasing torque
        if delta_torque > 0:
            max_increase = self.mrtd_config.dorsiflexion * delta_time
            delta_torque = min(delta_torque, max_increase)
        else:
            max_decrease = -self.mrtd_config.plantarflexion * delta_time
            delta_torque = max(delta_torque, max_decrease)

        constrained_torque = self._prev_torque + delta_torque
        self._prev_torque = constrained_torque

        return constrained_torque

    @abstractmethod
    def compute_control(self, model: mujoco.MjModel, data: mujoco.MjData,
                        target_pos: float) -> float:
        """
        Compute the human control torque.

        Args:
            model: MuJoCo model.
            data: MuJoCo data.
            target_pos: Target joint position.

        Returns:
            Computed human torque.
        """
        pass

class GravityCompensationController(HumanController):
    """Implementation of a gravity compensation controller."""
    def compute_control(self, model: mujoco.MjModel, data: mujoco.MjData,
                        target_pos: float) -> float:
        error = target_pos - data.sensordata[0]
        gravity_torque = data.qfrc_bias[3]
        desired_torque = self.config.gravity_compensation.Kp * error - gravity_torque
        return self.apply_mrtd_constraint(desired_torque, model.opt.timestep)

class PDController(HumanController):
    """Implementation of a PD controller."""
    def compute_control(self, model: mujoco.MjModel, data: mujoco.MjData,
                        target_pos: float) -> float:
        error = target_pos - data.sensordata[0]
        delta_time = model.opt.timestep
        delta_error = error - self._prev_error
        derivative = delta_error / delta_time if delta_time > 0 else 0.0
        desired_torque = self.config.pd.Kp * error + self.config.pd.Kd * derivative
        self._prev_error = error
        return self.apply_mrtd_constraint(desired_torque, delta_time)

class LQRController(HumanController):
    """Implementation of an LQR controller."""
    def __init__(self, config: HumanControllerConfig, mrtd_config: MRTDConfig):
        super().__init__(config, mrtd_config)
        self._K = None

    def _initialize_lqr(self, m: float, l: float, g: float, b: float):
        """
        Initialize the LQR gain matrix using hardcoded system parameters.

        Args:
            m: Effective mass.
            l: Effective COM distance.
            g: Gravitational acceleration.
            b: Damping coefficient.
        """
        I = m * l**2  # Moment of inertia
        A = np.array([[0, 1],
                      [-m * g * l / I, -b / I]])
        B = np.array([[0],
                      [1 / I]])
        Q = np.diag([self.config.lqr.Q_angle, self.config.lqr.Q_velocity])
        R = np.array([[self.config.lqr.R]])
        P = np.linalg.solve_continuous_are(A, B, Q, R)
        self._K = np.linalg.solve(R, B.T @ P)

    def compute_control(self, model: mujoco.MjModel, data: mujoco.MjData,
                        target_pos: float) -> float:
        if self._K is None:
            # Use hardcoded values as in the original version
            m = 80.0 - 2 * 0.0145 * 80.0
            l = 0.575 * 1.78
            g = 9.81
            b = 2.5
            self._initialize_lqr(m, l, g, b)
        current_state = np.array([data.sensordata[0], data.qvel[3]])
        error_state = np.array([current_state[0] - target_pos, current_state[1]])
        desired_torque = -float(np.dot(self._K, error_state))
        return self.apply_mrtd_constraint(desired_torque, model.opt.timestep)

class ExoController:
    """Implementation of an exoskeleton controller."""
    def __init__(self, config: ExoControllerConfig):
        self.config = config

    def compute_control(self, model: mujoco.MjModel, data: mujoco.MjData,
                        target_pos: float) -> float:
        if self.config.type == "none":
            return 0.0
        if self.config.type == "pd":
            error = target_pos - data.sensordata[0]
            current_velocity = data.qvel[3]
            return self.config.pd.Kp * error - self.config.pd.Kd * current_velocity
        return 0.0

def create_human_controller(config: HumanControllerConfig, 
                            mrtd_config: MRTDConfig) -> Optional[HumanController]:
    """
    Create a human controller based on configuration.

    Args:
        config: Human controller configuration.
        mrtd_config: MRTD configuration.

    Returns:
        A human controller instance or None if type is "none".
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
    """
    Create an exoskeleton controller based on configuration.

    Args:
        config: Exoskeleton controller configuration.

    Returns:
        An exoskeleton controller instance.
    """
    return ExoController(config)
