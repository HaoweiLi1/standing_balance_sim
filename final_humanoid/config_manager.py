import yaml
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class SimulationConfig:
    """Simulation parameters configuration"""
    simend: float
    simulation_timestep: float
    gravity: float
    xml_path: str
    lit_xml_file: str
    translation_friction_constant: float
    rolling_friction_constant: float

@dataclass
class ModelConfig:
    """Human model parameters configuration"""
    M_total: float
    H_total: float
    ankle_position_setpoint_radians: float
    ankle_initial_position_radians: float
    foot_angle_initial_position_radians: float

@dataclass
class GravityCompensationParams:
    """Gravity compensation controller parameters"""
    Kp: float

@dataclass
class PDParams:
    """PD controller parameters"""
    Kp: float
    Kd: float

@dataclass
class LQRParams:
    """LQR controller parameters"""
    Q_angle: float
    Q_velocity: float
    R: float

@dataclass
class HumanControllerConfig:
    """Human controller configuration"""
    type: str  # "gravity_compensation", "pd", "lqr", "none"
    gravity_compensation: Optional[GravityCompensationParams] = None
    pd: Optional[PDParams] = None
    lqr: Optional[LQRParams] = None

@dataclass
class ExoControllerConfig:
    """Exoskeleton controller configuration"""
    type: str  # "pd", "none"
    pd: Optional[PDParams] = None

@dataclass
class MRTDConfig:
    """Maximum Rate of Torque Development configuration"""
    enable: bool
    dorsiflexion: float  # Nm/s
    plantarflexion: float  # Nm/s

@dataclass
class PerturbationConfig:
    """Perturbation parameters configuration"""
    apply_perturbation: bool
    perturbation_time: float
    perturbation_magnitude: float
    perturbation_period: float

@dataclass
class VisualizationConfig:
    """Visualization parameters configuration"""
    plotter_flag: bool
    mp4_flag: bool
    mp4_file_name: str
    mp4_fps: int
    camera_azimuth: float
    camera_distance: float
    camera_elevation: float
    camera_lookat_xyz: str
    visualize_contact_force: bool
    visualize_perturbation_force: bool
    visualize_joints: bool
    visualize_actuators: bool
    visualize_center_of_mass: bool

class ConfigManager:
    """Configuration manager for the simulation"""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file.
        """
        self.config_path = config_path
        self._load_config()
        
    def _load_config(self):
        """Load configuration from YAML file and construct config objects."""
        with open(self.config_path, 'r', encoding='utf-8') as file:
            config_yaml = yaml.safe_load(file)
        
        if 'config' not in config_yaml:
            raise KeyError("Missing top-level 'config' key in configuration file.")
        config = config_yaml['config']
        
        # Helper to ensure required keys are present
        def require(key: str):
            if key not in config:
                raise KeyError(f"Missing required configuration key: '{key}'")
            return config[key]
        
        self.simulation = SimulationConfig(
            simend=require('simend'),
            simulation_timestep=require('simulation_timestep'),
            gravity=require('gravity'),
            xml_path=require('xml_path'),
            lit_xml_file=require('lit_xml_file'),
            translation_friction_constant=require('translation_friction_constant'),
            rolling_friction_constant=require('rolling_friction_constant')
        )
        
        self.model = ModelConfig(
            M_total=require('M_total'),
            H_total=require('H_total'),
            ankle_position_setpoint_radians=require('ankle_position_setpoint_radians'),
            ankle_initial_position_radians=require('ankle_initial_position_radians'),
            foot_angle_initial_position_radians=require('foot_angle_initial_position_radians')
        )
        
        human_ctrl = config.get('human_controller', {})
        self.human_controller = HumanControllerConfig(
            type=human_ctrl.get('type', 'none'),
            gravity_compensation=GravityCompensationParams(**human_ctrl['gravity_compensation'])
                if human_ctrl.get('gravity_compensation') else None,
            pd=PDParams(**human_ctrl['pd'])
                if human_ctrl.get('pd') else None,
            lqr=LQRParams(**human_ctrl['lqr'])
                if human_ctrl.get('lqr') else None
        )
        
        exo_ctrl = config.get('exo_controller', {})
        self.exo_controller = ExoControllerConfig(
            type=exo_ctrl.get('type', 'none'),
            pd=PDParams(**exo_ctrl['pd'])
                if exo_ctrl.get('pd') else None
        )
        
        self.mrtd = MRTDConfig(**config.get('mrtd'))
        
        self.perturbation = PerturbationConfig(
            apply_perturbation=require('apply_perturbation'),
            perturbation_time=require('perturbation_time'),
            perturbation_magnitude=require('perturbation_magnitude'),
            perturbation_period=require('perturbation_period')
        )
        
        self.visualization = VisualizationConfig(
            plotter_flag=require('plotter_flag'),
            mp4_flag=require('mp4_flag'),
            mp4_file_name=require('mp4_file_name'),
            mp4_fps=require('mp4_fps'),
            camera_azimuth=require('camera_azimuth'),
            camera_distance=require('camera_distance'),
            camera_elevation=require('camera_elevation'),
            camera_lookat_xyz=require('camera_lookat_xyz'),
            visualize_contact_force=require('visualize_contact_force'),
            visualize_perturbation_force=require('visualize_perturbation_force'),
            visualize_joints=require('visualize_joints'),
            visualize_actuators=require('visualize_actuators'),
            visualize_center_of_mass=require('visualize_center_of_mass')
        )
    
    def get_all_configs(self):
        """Return all configuration objects as a tuple."""
        return (
            self.simulation,
            self.model,
            self.human_controller,
            self.exo_controller,
            self.mrtd,
            self.perturbation,
            self.visualization
        )
