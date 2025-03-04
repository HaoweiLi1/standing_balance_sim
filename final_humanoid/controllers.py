from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg as linalg
import mujoco as mj

class Controller(ABC):
    """Abstract base class for all controllers."""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
    @abstractmethod
    def compute_control(self, state, target):
        """Compute control output based on current state and target."""
        pass

class HumanController(Controller):
    """Base class for human joint controllers."""
    
    def __init__(self, model, data, max_torque_df, max_torque_pf, mrtd_df=None, mrtd_pf=None):
        super().__init__(model, data)
        # Maximum torque limits for dorsiflexion (df) and plantarflexion (pf)
        self.max_torque_df = max_torque_df
        self.max_torque_pf = max_torque_pf
        # Maximum rate of torque development limits if specified
        self.mrtd_df = mrtd_df
        print(self.mrtd_df)
        self.mrtd_pf = mrtd_pf
        print(self.mrtd_pf)
        self.prev_torque = 0.0
        self.current_rtd = 0.0     # Store the current RTD
        self.current_rtd_limit = 0.0  # Store the applied limit

        # Get the gear ratio for the human actuator
        human_actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "human_ankle_actuator")
        self.gear_ratio = model.actuator_gear[human_actuator_id][0]
        # print(f"Human actuator gear ratio: {self.gear_ratio}")
        
    def apply_torque_limits(self, torque):
        """Apply maximum torque limits."""
        max_df = self.max_torque_df / self.gear_ratio
        max_pf = self.max_torque_pf / self.gear_ratio
        return np.clip(torque, max_pf, max_df)
    
    def apply_mrtd_limits(self, desired_torque, timestep):
        """Apply rate of torque development limits if enabled."""
        if self.mrtd_df is None or self.mrtd_pf is None:
            self.current_rtd = 0.0
            self.current_rtd_limit = float('inf')  # No limit
            return desired_torque
            
        delta_torque = desired_torque - self.prev_torque
        self.current_rtd = delta_torque / timestep
        
        if delta_torque > 0:
            max_increase = (self.mrtd_df / self.gear_ratio) * timestep
            rtd_limit = self.mrtd_df
            delta_torque = min(delta_torque, max_increase)
        else:
            max_decrease = -(self.mrtd_pf / self.gear_ratio) * timestep
            rtd_limit = -self.mrtd_pf
            delta_torque = max(delta_torque, max_decrease)
            
        self.current_rtd_limit = rtd_limit    
        limited_torque = self.prev_torque + delta_torque


        self.prev_torque = limited_torque
        return limited_torque

class HumanLQRController(HumanController):
    """LQR controller for human joint."""
    
    def __init__(self, model, data, max_torque_df, max_torque_pf, mass, leg_length, 
                 Q=None, R=None, damping=2.5, mrtd_df=None, mrtd_pf=None):
        # Pass mrtd parameters to parent class
        super().__init__(model, data, max_torque_df, max_torque_pf, mrtd_df, mrtd_pf)
        
        # System parameters
        self.m = mass
        self.l = leg_length
        self.g = 9.81
        self.b = damping
        self.I = mass * leg_length**2
        
        # Default LQR weights if not specified
        if Q is None:
            Q = np.diag([5000, 100])
        if R is None:
            R = np.array([[0.01]])
            
        # Calculate LQR gains
        self.K = self._compute_lqr_gains(Q, R)
        
    def _compute_lqr_gains(self, Q, R):
        """Compute LQR gain matrix."""
        # System matrices
        A = np.array([[0, 1],
                     [-self.m*self.g*self.l/self.I, -self.b/self.I]])
        B = np.array([[0],
                     [1/self.I]])
                     
        # Solve Riccati equation
        P = linalg.solve_continuous_are(A, B, Q, R)
        
        # Compute gains
        return np.linalg.solve(R, B.T @ P)
        
    def compute_control(self, state, target):
        """Compute LQR control based on current state and target."""
        # Current state [theta, theta_dot]
        current_state = np.array([
            state[0],  # Joint angle
            state[1]   # Joint velocity
        ])
        
        # Error state
        error_state = current_state - np.array([target, 0])
        
        # Compute control
        u = -np.dot(self.K, error_state)
        torque = float(u[0])
        
        # Apply limits
        torque = self.apply_torque_limits(torque)
        torque = self.apply_mrtd_limits(torque, self.model.opt.timestep)
        
        return torque

class HumanPDController(HumanController):
    """PD controller for human joint."""
    
    def __init__(self, model, data, max_torque_df, max_torque_pf, 
                 kp, kd, mrtd_df=None, mrtd_pf=None):
        super().__init__(model, data, max_torque_df, max_torque_pf, mrtd_df, mrtd_pf)
        self.kp = kp
        self.kd = kd
        self.prev_error = 0
        
    def compute_control(self, state, target):
        """Compute PD control based on current state and target."""
        # Compute error and derivative
        error = target - state[0]
        delta_error = error - self.prev_error
        delta_time = self.model.opt.timestep
        derivative = delta_error / delta_time
        
        # PD control
        torque = self.kp * error + self.kd * derivative
        
        # Apply limits
        torque = self.apply_torque_limits(torque)
        torque = self.apply_mrtd_limits(torque, delta_time)
        
        self.prev_error = error
        return torque

class ExoController(Controller):
    """Base class for exoskeleton controllers."""
    
    def __init__(self, model, data, max_torque):
        super().__init__(model, data)
        self.max_torque = max_torque
        
    def apply_torque_limits(self, torque):
        """Apply maximum torque limits."""
        return np.clip(torque, -self.max_torque, self.max_torque)

    @property
    def is_visible(self):
        """Property indicating whether the exoskeleton should be visible."""
        return True

class ExoPDController(ExoController):
    """PD controller for exoskeleton."""
    
    def __init__(self, model, data, max_torque, kp, kd):
        super().__init__(model, data, max_torque)
        self.kp = kp
        self.kd = kd
        
    def compute_control(self, state, target):
        """Compute PD control based on current state and target."""
        error = target - state[0]
        torque = self.kp * error - self.kd * state[1]
        return self.apply_torque_limits(torque)


class ExoNoneController(ExoController):
    """Controller that always returns zero torque (disabled exoskeleton)."""
    
    def __init__(self, model, data, max_torque=0):
        # Initialize with zero actual max torque
        super().__init__(model, data, 0)
        
    def compute_control(self, state, target):
        """Always return zero torque."""
        return 0.0
    
    @property
    def is_visible(self):
        """Property indicating whether the exoskeleton should be visible."""
        return False

# Factory functions to create controllers by type name

def create_human_controller(controller_type, model, data, params):
    """
    Factory function to create a human controller instance based on type.
    
    Args:
        controller_type: String identifier for the controller type
        model: MuJoCo model
        data: MuJoCo data
        params: Dict containing controller parameters
        
    Returns:
        An instance of a HumanController subclass
    """
    if controller_type == "LQR":
        return HumanLQRController(
            model=model,
            data=data,
            max_torque_df=params.get('max_torque_df', 43),
            max_torque_pf=params.get('max_torque_pf', -181),
            mass=params.get('mass', 80),
            leg_length=params.get('leg_length', 1.0),
            Q=np.diag([params.get('Q_angle', 5000), params.get('Q_velocity', 100)]),
            R=np.array([[params.get('R', 0.01)]]),
            # Add these explicitly:
            mrtd_df=params.get('mrtd_df'),
            mrtd_pf=params.get('mrtd_pf')
        )
    elif controller_type == "PD":
        return HumanPDController(
            model=model,
            data=data,
            max_torque_df=params.get('max_torque_df', 43),
            max_torque_pf=params.get('max_torque_pf', -181),
            kp=params.get('kp', 800),
            kd=params.get('kd', 10),
            mrtd_df=params.get('mrtd_df', None),
            mrtd_pf=params.get('mrtd_pf', None)
        )
    else:
        raise ValueError(f"Unknown human controller type: {controller_type}")

def create_exo_controller(controller_type, model, data, params):
    """
    Factory function to create an exo controller instance based on type.
    
    Args:
        controller_type: String identifier for the controller type
        model: MuJoCo model
        data: MuJoCo data
        params: Dict containing controller parameters
        
    Returns:
        An instance of an ExoController subclass
    """
    if controller_type == "PD":
        return ExoPDController(
            model=model,
            data=data,
            max_torque=params.get('max_torque', 50),
            kp=params.get('kp', 10),
            kd=params.get('kd', 1)
        )

    elif controller_type == "None":
        return ExoNoneController(
            model=model,
            data=data
        )
    else:
        raise ValueError(f"Unknown exo controller type: {controller_type}")