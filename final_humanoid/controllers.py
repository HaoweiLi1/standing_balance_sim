from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg as linalg

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
        self.mrtd_pf = mrtd_pf
        self.prev_torque = 0.0
        
    def apply_torque_limits(self, torque):
        """Apply maximum torque limits."""
        return np.clip(torque, self.max_torque_pf, self.max_torque_df)
    
    def apply_mrtd_limits(self, desired_torque, timestep):
        """Apply rate of torque development limits if enabled."""
        if self.mrtd_df is None or self.mrtd_pf is None:
            return desired_torque
            
        delta_torque = desired_torque - self.prev_torque
        
        if delta_torque > 0:
            max_increase = self.mrtd_df * timestep
            delta_torque = min(delta_torque, max_increase)
        else:
            max_decrease = -self.mrtd_pf * timestep
            delta_torque = max(delta_torque, max_decrease)
            
        limited_torque = self.prev_torque + delta_torque
        self.prev_torque = limited_torque
        return limited_torque

# Add a new multi-joint controller class
class HumanMultiJointController:
    """Base class for human controllers that handle multiple joints."""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.controllers = {}  # Dictionary to hold individual joint controllers
        
    def add_joint_controller(self, joint_name, controller):
        """Add a controller for a specific joint."""
        self.controllers[joint_name] = controller
        
    def compute_control(self, states, targets):
        """Compute control outputs for all joints.
        
        Args:
            states: Dictionary of joint states {joint_name: [angle, velocity]}
            targets: Dictionary of joint targets {joint_name: target_angle}
            
        Returns:
            Dictionary of control outputs {joint_name: torque}
        """
        torques = {}
        for joint_name, controller in self.controllers.items():
            if joint_name in states and joint_name in targets:
                torques[joint_name] = controller.compute_control(
                    state=states[joint_name],
                    target=targets[joint_name]
                )
        return torques

class HumanCoordinatedController(HumanMultiJointController):
    """Controller that coordinates ankle and hip joints for balance."""
    
    def __init__(self, model, data, ankle_controller, hip_controller, 
                 coordination_gain=0.2):
        """
        Initialize the coordinated controller.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            ankle_controller: Controller for ankle joint
            hip_controller: Controller for hip joint
            coordination_gain: Gain factor for coordination (0-1)
        """
        super().__init__(model, data)
        self.add_joint_controller('ankle', ankle_controller)
        self.add_joint_controller('hip', hip_controller)
        self.coordination_gain = coordination_gain
        
    def compute_control(self, states, targets):
        """
        Compute coordinated control for ankle and hip.
        
        This implements a coordination strategy where:
        1. Each joint computes its own control
        2. The hip assists the ankle based on ankle state
        3. The ankle adapts based on hip action
        
        Args:
            states: Dictionary with 'ankle' and 'hip' states
            targets: Dictionary with 'ankle' and 'hip' targets
            
        Returns:
            Dictionary with 'ankle' and 'hip' torques
        """
        # Get individual joint torques first
        base_torques = super().compute_control(states, targets)
        
        # Only proceed with coordination if we have both joints
        if 'ankle' in base_torques and 'hip' in base_torques:
            ankle_torque = base_torques['ankle']
            hip_torque = base_torques['hip']
            
            # Get ankle and hip states
            ankle_angle, ankle_velocity = states['ankle']
            hip_angle, hip_velocity = states['hip']
            
            # Coordination: Hip assists ankle based on ankle state
            # When ankle is working hard (high velocity/displacement), hip provides assistance
            ankle_effort = abs(ankle_velocity) + abs(ankle_angle - targets['ankle'])
            hip_assistance = self.coordination_gain * ankle_effort * np.sign(ankle_torque)
            
            # Coordination: Ankle reduces effort when hip is active
            hip_activity = abs(hip_velocity) + abs(hip_angle - targets['hip'])
            ankle_reduction = self.coordination_gain * hip_activity * 0.5
            
            # Apply coordination adjustments
            coordinated_hip_torque = hip_torque + hip_assistance
            coordinated_ankle_torque = ankle_torque * (1.0 - ankle_reduction)
            
            # Apply joint torque limits from the original controllers
            ankle_controller = self.controllers['ankle']
            hip_controller = self.controllers['hip']
            
            coordinated_ankle_torque = ankle_controller.apply_torque_limits(coordinated_ankle_torque)
            coordinated_hip_torque = hip_controller.apply_torque_limits(coordinated_hip_torque)
            
            # Apply MRTD limits if available
            if hasattr(ankle_controller, 'apply_mrtd_limits'):
                coordinated_ankle_torque = ankle_controller.apply_mrtd_limits(
                    coordinated_ankle_torque, 
                    self.model.opt.timestep
                )
            
            if hasattr(hip_controller, 'apply_mrtd_limits'):
                coordinated_hip_torque = hip_controller.apply_mrtd_limits(
                    coordinated_hip_torque,
                    self.model.opt.timestep
                )
            
            return {
                'ankle': coordinated_ankle_torque,
                'hip': coordinated_hip_torque
            }
        
        # If we don't have both joints, return the base torques
        return base_torques


class HumanMultiJointLQRController(HumanMultiJointController):
    """Multivariable LQR controller for controlling ankle and hip joints together."""
    
    def __init__(self, model, data, mass, leg_length, torso_length,
                 ankle_torque_limits, hip_torque_limits,
                 ankle_mrtd=None, hip_mrtd=None,
                 Q=None, R=None, damping_ankle=2.5, damping_hip=3.0):
        """
        Initialize the multi-joint LQR controller.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            mass: Total body mass (kg)
            leg_length: Length of leg segment (m)
            torso_length: Length of torso segment (m)
            ankle_torque_limits: Tuple of (min, max) ankle torque
            hip_torque_limits: Tuple of (min, max) hip torque
            ankle_mrtd: Tuple of (min, max) ankle rate of torque development
            hip_mrtd: Tuple of (min, max) hip rate of torque development
            Q: State cost matrix (4x4)
            R: Control cost matrix (2x2)
            damping_ankle: Ankle joint damping
            damping_hip: Hip joint damping
        """
        super().__init__(model, data)
        
        # System parameters
        self.m = mass
        self.l1 = leg_length
        self.l2 = torso_length
        self.g = 9.81
        self.b1 = damping_ankle
        self.b2 = damping_hip
        
        # Torque limits
        self.ankle_torque_min, self.ankle_torque_max = ankle_torque_limits
        self.hip_torque_min, self.hip_torque_max = hip_torque_limits
        
        # MRTD limits
        self.ankle_mrtd = ankle_mrtd
        self.hip_mrtd = hip_mrtd
        
        # Previous torques for MRTD limiting
        self.prev_ankle_torque = 0.0
        self.prev_hip_torque = 0.0
        
        # Default LQR weights if not specified
        if Q is None:
            # State cost matrix [ankle_angle, ankle_vel, hip_angle, hip_vel]
            Q = np.diag([5000, 100, 1000, 50])
        if R is None:
            # Control cost matrix [ankle_torque, hip_torque]
            R = np.diag([0.01, 0.05])
            
        # Calculate LQR gains
        self.K = self._compute_lqr_gains(Q, R)
        
    def _compute_lqr_gains(self, Q, R):
        """Compute LQR gain matrix for double inverted pendulum."""
        # Linearized system matrices for a double inverted pendulum
        # State = [θ_ankle, θ_ankle_dot, θ_hip, θ_hip_dot]
        # Control = [τ_ankle, τ_hip]
        
        # Simplified double pendulum linearized about the upright position
        A = np.zeros((4, 4))
        A[0, 1] = 1.0
        A[2, 3] = 1.0
        
        # Simplified dynamics at the upright equilibrium
        # These are approximations - a full derivation would be more complex
        A[1, 0] = -self.m * self.g * self.l1 / (self.m * self.l1**2)
        A[1, 1] = -self.b1 / (self.m * self.l1**2)
        A[1, 2] = 0.5 * self.m * self.g * self.l2 / (self.m * self.l1**2)
        A[3, 0] = 0
        A[3, 2] = -self.m * self.g * self.l2 / (self.m * self.l2**2)
        A[3, 3] = -self.b2 / (self.m * self.l2**2)
        
        B = np.zeros((4, 2))
        B[1, 0] = 1.0 / (self.m * self.l1**2)
        B[3, 1] = 1.0 / (self.m * self.l2**2)
        
        # Solve Riccati equation
        P = linalg.solve_continuous_are(A, B, Q, R)
        
        # Compute gains
        K = np.linalg.solve(R, B.T @ P)
        
        return K
        
    def compute_control(self, states, targets):
        """
        Compute LQR control for both ankle and hip joints.
        
        Args:
            states: Dictionary with 'ankle' and 'hip' states
            targets: Dictionary with 'ankle' and 'hip' targets
            
        Returns:
            Dictionary with 'ankle' and 'hip' torques
        """
        # Extract states
        ankle_state = states.get('ankle', [0, 0])
        hip_state = states.get('hip', [0, 0])
        
        # Extract targets
        ankle_target = targets.get('ankle', 0)
        hip_target = targets.get('hip', 0)
        
        # Form full state vector [ankle_angle, ankle_vel, hip_angle, hip_vel]
        current_state = np.array([
            ankle_state[0],  # Ankle angle
            ankle_state[1],  # Ankle velocity
            hip_state[0],    # Hip angle
            hip_state[1]     # Hip velocity
        ])
        
        # Form target state vector
        target_state = np.array([
            ankle_target,  # Ankle angle target
            0,             # Ankle velocity target (zero)
            hip_target,    # Hip angle target
            0              # Hip velocity target (zero)
        ])
        
        # Error state
        error_state = current_state - target_state
        
        # Compute control
        u = -np.dot(self.K, error_state)
        ankle_torque, hip_torque = u
        
        # Apply ankle torque limits
        ankle_torque = np.clip(ankle_torque, self.ankle_torque_min, self.ankle_torque_max)
        
        # Apply hip torque limits
        hip_torque = np.clip(hip_torque, self.hip_torque_min, self.hip_torque_max)
        
        # Apply ankle MRTD limits if enabled
        if self.ankle_mrtd is not None:
            ankle_mrtd_min, ankle_mrtd_max = self.ankle_mrtd
            delta_torque = ankle_torque - self.prev_ankle_torque
            max_delta = ankle_mrtd_max * self.model.opt.timestep
            min_delta = ankle_mrtd_min * self.model.opt.timestep
            
            delta_torque = np.clip(delta_torque, min_delta, max_delta)
            ankle_torque = self.prev_ankle_torque + delta_torque
            self.prev_ankle_torque = ankle_torque
            
        # Apply hip MRTD limits if enabled
        if self.hip_mrtd is not None:
            hip_mrtd_min, hip_mrtd_max = self.hip_mrtd
            delta_torque = hip_torque - self.prev_hip_torque
            max_delta = hip_mrtd_max * self.model.opt.timestep
            min_delta = hip_mrtd_min * self.model.opt.timestep
            
            delta_torque = np.clip(delta_torque, min_delta, max_delta)
            hip_torque = self.prev_hip_torque + delta_torque
            self.prev_hip_torque = hip_torque
        
        return {
            'ankle': ankle_torque,
            'hip': hip_torque
        }


class HumanLQRController(HumanController):
    """LQR controller for human joint."""
    
    def __init__(self, model, data, max_torque_df, max_torque_pf, mass, leg_length, 
                 Q=None, R=None, damping=2.5):
        super().__init__(model, data, max_torque_df, max_torque_pf)
        
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

class ExoGravityCompensation(ExoController):
    """Gravity compensation controller for exoskeleton."""
    
    def __init__(self, model, data, max_torque, compensation_factor=0.5):
        super().__init__(model, data, max_torque)
        self.compensation_factor = compensation_factor
        
    def compute_control(self, state, target):
        """Compute gravity compensation torque."""
        gravity_torque = -self.compensation_factor * self.data.qfrc_bias[3]
        return self.apply_torque_limits(gravity_torque)

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
    if controller_type == "MultiJointLQR":
        # Create a multi-joint LQR controller
        return HumanMultiJointLQRController(
            model=model,
            data=data,
            mass=params.get('mass', 80),
            leg_length=params.get('leg_length', 0.5),
            torso_length=params.get('torso_length', 0.5),
            ankle_torque_limits=(
                params.get('ankle_max_torque_pf', -181),
                params.get('ankle_max_torque_df', 43)
            ),
            hip_torque_limits=(
                params.get('hip_max_torque_min', -150),
                params.get('hip_max_torque_max', 150)
            ),
            ankle_mrtd=(
                params.get('ankle_mrtd_pf', -957),
                params.get('ankle_mrtd_df', 309)
            ),
            hip_mrtd=(
                params.get('hip_mrtd_min', -400),
                params.get('hip_mrtd_max', 400)
            ),
            Q=np.diag([
                params.get('Q_ankle_angle', 5000),
                params.get('Q_ankle_velocity', 100),
                params.get('Q_hip_angle', 1000),
                params.get('Q_hip_velocity', 50)
            ]),
            R=np.diag([
                params.get('R_ankle', 0.01),
                params.get('R_hip', 0.05)
            ])
        )
    elif controller_type == "Coordinated":
        # Create separate controllers for ankle and hip
        ankle_controller = HumanLQRController(
            model=model,
            data=data,
            max_torque_df=params.get('ankle_max_torque_df', 43),
            max_torque_pf=params.get('ankle_max_torque_pf', -181),
            mass=params.get('mass', 80),
            leg_length=params.get('leg_length', 0.5),
            Q=np.diag([params.get('ankle_Q_angle', 5000), params.get('ankle_Q_velocity', 100)]),
            R=np.array([[params.get('ankle_R', 0.01)]])
        )
        
        hip_controller = HumanLQRController(
            model=model,
            data=data,
            max_torque_df=params.get('hip_max_torque_max', 150),
            max_torque_pf=params.get('hip_max_torque_min', -150),
            mass=params.get('mass', 80),
            leg_length=params.get('torso_length', 0.5),
            Q=np.diag([params.get('hip_Q_angle', 1000), params.get('hip_Q_velocity', 50)]),
            R=np.array([[params.get('hip_R', 0.05)]])
        )
        
        # Create coordinated controller with both joint controllers
        return HumanCoordinatedController(
            model=model,
            data=data,
            ankle_controller=ankle_controller,
            hip_controller=hip_controller,
            coordination_gain=params.get('coordination_gain', 0.2)
        )
        
    # Fall back to original controller types
    elif controller_type == "LQR":
        return HumanLQRController(
            model=model,
            data=data,
            max_torque_df=params.get('max_torque_df', 43),
            max_torque_pf=params.get('max_torque_pf', -181),
            mass=params.get('mass', 80),
            leg_length=params.get('leg_length', 1.0),
            Q=np.diag([params.get('Q_angle', 5000), params.get('Q_velocity', 100)]),
            R=np.array([[params.get('R', 0.01)]])
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