from abc import ABC, abstractmethod
import numpy as np
import scipy.linalg as linalg
import mujoco as mj
import os

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
    
    def __init__(self, model, data, mrtd_df=None, mrtd_pf=None):
        super().__init__(model, data)

        self.mrtd_df = mrtd_df
        self.mrtd_pf = mrtd_pf

        self.prev_torque = 0.0
        self.current_rtd = 0.0     # Store the current RTD
        self.current_rtd_limit = 0.0  # Store the applied limit
  
    def apply_mrtd_limits(self, desired_torque, timestep):
        """Apply rate of torque development limits if enabled."""
        if self.mrtd_df is None or self.mrtd_pf is None:
            self.current_rtd = 0.0
            self.current_rtd_limit = float('inf')  # No limit
            return desired_torque
            
        delta_torque = desired_torque - self.prev_torque
        # Calculate raw RTD before limiting
        raw_rtd = delta_torque / timestep
        
        # Apply rate limits based on direction
        if delta_torque > 0:  # Dorsiflexion (positive direction)
            max_increase = self.mrtd_df * timestep
            rtd_limit = self.mrtd_df
            
            if delta_torque > max_increase:
                delta_torque = max_increase
                # If we're limiting the torque change, RTD equals the limit
                self.current_rtd = self.mrtd_df
            else:
                # Otherwise RTD equals the raw calculated value
                self.current_rtd = raw_rtd
        else:  # Plantarflexion (negative direction)
            max_decrease = -self.mrtd_pf * timestep
            rtd_limit = -self.mrtd_pf
            
            if delta_torque < max_decrease:
                delta_torque = max_decrease
                # If we're limiting the torque change, RTD equals the limit
                self.current_rtd = -self.mrtd_pf
            else:
                # Otherwise RTD equals the raw calculated value
                self.current_rtd = raw_rtd
        
        self.current_rtd_limit = rtd_limit
        limited_torque = self.prev_torque + delta_torque

        self.prev_torque = limited_torque
        return limited_torque

class HumanPrecomputedController(HumanController):
    """Controller that uses pre-computed torque values from a CSV file."""
    
    def __init__(self, model, data, precomputed_params, mrtd_df=None, mrtd_pf=None):
        super().__init__(model, data, mrtd_df, mrtd_pf)
        
        # Get trajectory file path from precomputed_params
        self.trajectory_file = precomputed_params.get('trajectory_file', 'trajectory.csv')
        
        # Flag to enable one-step look-ahead (compensate for MuJoCo control delay)
        self.use_time_offset = True
        
        # Track if this is the first control step
        self.first_step = True
        
        # Load precomputed curve from file
        try:
            # Check if the path is absolute or relative
            if os.path.isabs(self.trajectory_file):
                file_path = self.trajectory_file
            else:
                # If relative, assume it's relative to the current working directory
                file_path = os.path.join(os.getcwd(), self.trajectory_file)
            
            print(f"Loading precomputed trajectory from: {file_path}")
            self.curve_data = np.loadtxt(file_path, delimiter=',')
            
            # Verify data format
            if self.curve_data.shape[1] < 2:
                raise ValueError(f"CSV file must have at least 2 columns (time, torque). Found {self.curve_data.shape[1]} columns.")
            
            # Store the first torque value for special initial handling
            self.initial_torque = self.curve_data[0, 1]
            
            # Create a direct time-to-torque mapping for faster lookup
            self.torque_map = {}
            for i in range(len(self.curve_data)):
                time_val = self.curve_data[i, 0]
                torque_val = self.curve_data[i, 1]
                self.torque_map[round(time_val, 6)] = torque_val
            
            # Store min/max time values for bound checking
            self.min_time = self.curve_data[0, 0]
            self.max_time = self.curve_data[-1, 0]
            
            # Calculate the time step in the data
            time_diffs = np.diff(self.curve_data[:, 0])
            self.avg_time_step = np.mean(time_diffs)
            time_step_std = np.std(time_diffs)
            
            # Get the model timestep
            self.model_timestep = model.opt.timestep
            
            print(f"Successfully loaded trajectory with {len(self.torque_map)} points.")
            print(f"Time range: [{self.min_time:.6f}, {self.max_time:.6f}] s")
            print(f"Initial torque value: {self.initial_torque:.6f} Nm")
            print(f"Average time step in data: {self.avg_time_step:.6f} s (std: {time_step_std:.6f} s)")
            print(f"Model time step: {self.model_timestep:.6f} s")
            print(f"Time offset (look-ahead) enabled: {self.use_time_offset}")
            print(f"Special first step handling enabled")
            
            # Check if time steps are consistent
            if time_step_std > 1e-6:
                print("Warning: Time steps in the CSV file are not uniform. This may cause inaccuracies.")
                
            # Compare with model time step
            if abs(self.avg_time_step - self.model_timestep) > 1e-6:
                print(f"Warning: CSV time step ({self.avg_time_step:.6f} s) differs from model time step ({self.model_timestep:.6f} s).")
                print("This may cause timing mismatches when applying the precomputed torques.")
            
        except Exception as e:
            print(f"Error loading control curve: {e}")
            # Create a fallback empty dictionary
            self.torque_map = {0.0: 0.0, 1.0: 0.0}
            self.min_time = 0.0
            self.max_time = 1.0
            self.avg_time_step = 0.001
            self.model_timestep = 0.001
            self.initial_torque = 0.0
            print("Using fallback zero control curve")
        
    def compute_control(self, state, target):
        """
        Use the precomputed torque value with special handling for the initial state.
        """
        # Get current time
        current_time = self.data.time
        
        # Special handling for the first control step
        if self.first_step:
            torque = self.initial_torque
            print(f"First step: Using initial torque {torque:.6f} Nm at time {current_time:.6f}s")
            self.first_step = False
        else:
            # After first step, apply the time offset compensation
            if self.use_time_offset:
                lookup_time = current_time + self.model_timestep
            else:
                lookup_time = current_time
                
            # Round to 6 decimal places for more accurate dictionary lookup
            lookup_time = round(lookup_time, 6)
            
            # Find torque value using direct lookup
            if lookup_time in self.torque_map:
                # Exact time match found
                torque = self.torque_map[lookup_time]
            else:
                # No exact match - handle bounds
                if lookup_time <= self.min_time:
                    torque = self.torque_map[round(self.min_time, 6)]
                elif lookup_time >= self.max_time:
                    torque = self.torque_map[round(self.max_time, 6)]
                else:
                    # Find the closest time
                    closest_time = min(self.torque_map.keys(), key=lambda x: abs(x - lookup_time))
                    torque = self.torque_map[closest_time]
        
        # Store the current torque for reference
        self.prev_torque = torque
        self.current_rtd = 0.0
        self.current_rtd_limit = float('inf')
        
        return torque

class HumanLQRController(HumanController):
    """LQR controller for human joint."""
    
    def __init__(self, model, data, mass, leg_length, 
                 Q=None, R=None, damping=2.5, mrtd_df=None, mrtd_pf=None, exo_config=None,
                 dynamic_gains=None):
        # Pass mrtd parameters to parent class
        super().__init__(model, data, mrtd_df, mrtd_pf)
        
        # System parameters
        self.m = mass
        self.l = leg_length
        self.g = 9.81
        self.b = damping
        self.I = mass * leg_length**2

        # Store the exoskeleton configuration
        self.exo_config = exo_config
        self.exo_type = exo_config.get('type', 'None') if exo_config else 'None'
        
        # Store dynamic gains if provided
        self.dynamic_gains = dynamic_gains
        
        # Default LQR weights if not specified
        if Q is None:
            Q = np.diag([4000, 400])
        if R is None:
            R = np.array([[0.01]])
            
        # Store Q and R for gain calculations
        self.Q = Q
        self.R = R
            
        # Calculate LQR gains based on whether exo is enabled
        if self.exo_type != 'None':
            self.K = self._compute_human_exo_lqr_gains()
            print(f"LQR controller initialized for human-exo system with gains: {self.K}")
        else:
            self.K = self._compute_human_only_lqr_gains()
            print(f"LQR controller initialized for human-only system with gains: {self.K}")
    
    def _compute_human_only_lqr_gains(self):
        """Compute LQR gain matrix for human-only system."""
        # System matrices for human-only case
        A = np.array([[0, 1],
                      [self.m*self.g*self.l/self.I, -self.b/self.I]])
        B = np.array([[0],
                      [1/self.I]])
                     
        # Solve Riccati equation
        P = linalg.solve_continuous_are(A, B, self.Q, self.R)
        
        # Compute gains
        return np.linalg.solve(self.R, B.T @ P)
    
    def _compute_human_exo_lqr_gains(self):
        """
        Compute LQR gain matrix for human-exo system with PD controller.
        This accounts for the exoskeleton's effect on system dynamics.
        """
        # Get exoskeleton PD controller parameters
        if self.exo_config['type'] == 'PD':
            # Check if dynamic gains are being used
            if self.exo_config.get('pd_params', {}).get('use_dynamic_gains', False) and self.dynamic_gains:
                # Use dynamically calculated gains
                exo_kp = self.dynamic_gains['kp']
                exo_kd = self.dynamic_gains['kd']
                print(f"Using dynamic exo gains - Kp: {exo_kp:.2f}, Kd: {exo_kd:.2f}")
            else:
                # Use static gains from config
                exo_kp = self.exo_config.get('pd_params', {}).get('kp', 0)
                exo_kd = self.exo_config.get('pd_params', {}).get('kd', 0)
                print(f"Using static exo gains - Kp: {exo_kp}, Kd: {exo_kd}")
        else:
            # Default to zero gains if exo is not using PD control
            exo_kp = 0
            exo_kd = 0
        
        # Modify system matrices to account for exoskeleton PD control
        # The exo PD controller effectively changes the system dynamics:
        # 1. Adds damping (b + Kd)
        # 2. Changes effective stiffness (mgl - Kp)
        A_exo = np.array([
            [0, 1],
            [(self.m*self.g*self.l - exo_kp)/self.I, -(self.b + exo_kd)/self.I]
        ])
        
        B_exo = np.array([
            [0],
            [1/self.I]
        ])
        
        # Solve Riccati equation with modified system matrices
        P_exo = linalg.solve_continuous_are(A_exo, B_exo, self.Q, self.R)
        
        # Compute gains for human-exo system
        K_exo = np.linalg.solve(self.R, B_exo.T @ P_exo)
        
        print(f"Human-exo LQR system matrices: A={A_exo}, B={B_exo}")
        
        return K_exo
    # def _compute_lqr_gains(self, Q, R):
    #     """Compute LQR gain matrix."""
    #     # System matrices
    #     A = np.array([[0, 1],
    #                  [-self.m*self.g*self.l/self.I, -self.b/self.I]])
    #     B = np.array([[0],
    #                  [1/self.I]])
                     
    #     # Solve Riccati equation
    #     P = linalg.solve_continuous_are(A, B, Q, R)
        
    #     # Compute gains
    #     return np.linalg.solve(R, B.T @ P)
        
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
        torque = self.apply_mrtd_limits(torque, self.model.opt.timestep)
        
        return torque

class HumanPDController(HumanController):
    """PD controller for human joint."""
    
    def __init__(self, model, data, kp, kd, mrtd_df=None, mrtd_pf=None):
        super().__init__(model, data, mrtd_df, mrtd_pf)
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
        torque = self.apply_mrtd_limits(torque, delta_time)
        
        self.prev_error = error
        return torque

class ExoController(Controller):
    """Base class for exoskeleton controllers."""
    
    def __init__(self, model, data):
        super().__init__(model, data)

    @property
    def is_visible(self):
        """Property indicating whether the exoskeleton should be visible."""
        return True

class ExoPDController(ExoController):
    """PD controller for exoskeleton."""
    
    def __init__(self, model, data, kp, kd):
        super().__init__(model, data)
        self.kp = kp
        self.kd = kd
        
    def compute_control(self, state, target):
        """Compute PD control based on current state and target."""
        error = target - state[0]
        torque = self.kp * error - self.kd * state[1]
        return torque


class ExoNoneController(ExoController):
    """Controller that always returns zero torque (disabled exoskeleton)."""
    
    def __init__(self, model, data):
        # Initialize with zero actual max torque
        super().__init__(model, data)
        
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
            mass=params.get('mass', 80),
            leg_length=params.get('leg_length', 1.0),
            Q=np.diag([params.get('Q_angle', 5000), params.get('Q_velocity', 100)]),
            R=np.array([[params.get('R', 0.01)]]),
            # Get MRTD params explicitly
            mrtd_df=params.get('mrtd_df'),
            mrtd_pf=params.get('mrtd_pf'),
            exo_config=params.get('exo_config'),
            dynamic_gains=params.get('dynamic_gains')
        )
    elif controller_type == "PD":
        return HumanPDController(
            model=model,
            data=data,
            kp=params.get('kp', 800),
            kd=params.get('kd', 10),
            mrtd_df=params.get('mrtd_df', None),
            mrtd_pf=params.get('mrtd_pf', None)
        )
    elif controller_type == "Precomputed":
        # Get precomputed-specific parameters
        precomputed_params = params.get('precomputed_params', {})
        
        return HumanPrecomputedController(
            model=model,
            data=data,
            precomputed_params=precomputed_params,
            mrtd_df=params.get('mrtd_df'),
            mrtd_pf=params.get('mrtd_pf')
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