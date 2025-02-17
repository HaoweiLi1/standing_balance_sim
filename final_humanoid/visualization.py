import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
from data_logger import SimulationLogger

class Visualizer:
    """Visualization class for simulation data"""
    
    def __init__(self, logger: SimulationLogger):
        """Initialize visualizer
        
        Args:
            logger: Data logger containing simulation data
        """
        self.logger = logger
        plt.rcParams['text.usetex'] = True
        
    def plot_joint_data(self):
        """Plot joint position and velocity data"""
        # Plot position data
        pos_data = self.logger.get_plot_data("position")
        self._create_plot(
            pos_data["time"],
            pos_data["values"],
            ["Actual Position", "Goal Position"],
            "Joint Position vs Time",
            "Time [sec]",
            "Position [deg]"
        )
        
        # Plot velocity data
        vel_data = self.logger.get_plot_data("velocity")
        self._create_plot(
            vel_data["time"],
            vel_data["values"],
            ["Joint Velocity"],
            "Joint Velocity vs Time",
            "Time [sec]",
            "Velocity [deg/s]"
        )
        
    def plot_torque_data(self):
        """Plot torque data"""
        torque_data = self.logger.get_plot_data("ankle")
        gravity_data = self.logger.get_plot_data("gravity_torque")
        
        # Plot all torques
        plt.figure(figsize=(10, 6))
        plt.plot(torque_data["time"], torque_data["values"][:, 0], 
                label="Human Torque")
        plt.plot(torque_data["time"], torque_data["values"][:, 1], 
                label="Exo Torque")
        plt.plot(torque_data["time"], torque_data["values"][:, 2], 
                label="Total Torque")
        plt.plot(gravity_data["time"], gravity_data["values"], 
                '--', label="Gravity Torque")
        
        plt.title("Torques vs Time")
        plt.xlabel("Time [sec]")
        plt.ylabel("Torque [Nm]")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_gravity_compensation(self):
        """Plot gravity compensation related data"""
        # Get torque and gravity data
        torque_data = self.logger.get_plot_data("ankle", 
            ["human_torque", "exo_torque"])
        gravity_data = self.logger.get_plot_data("gravity_torque")
        
        plt.figure(figsize=(10, 6))
        plt.plot(torque_data["time"], 
                -(torque_data["values"][:, 0] + torque_data["values"][:, 1]), 
                label="Net Control Torque")
        plt.plot(gravity_data["time"], gravity_data["values"], 
                '--', label="Gravity Torque")
        
        plt.title("Gravity Compensation Performance")
        plt.xlabel("Time [sec]")
        plt.ylabel("Torque [Nm]")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_perturbation_data(self):
        """Plot perturbation force data"""
        if self.logger.perturbation_data is not None:
            pert_data = self.logger.get_plot_data("perturbation")
            self._create_plot(
                pert_data["time"],
                pert_data["values"],
                ["Perturbation Force"],
                "Perturbation Force vs Time",
                "Time [sec]",
                "Force [N]"
            )
            
    def plot_contact_forces(self):
        """Plot contact force data"""
        if self.logger.contact_data is not None:
            contact_data = self.logger.get_plot_data("contact")
            self._create_plot(
                contact_data["time"],
                contact_data["values"],
                ["Front Contact", "Back Contact"],
                "Contact Forces vs Time",
                "Time [sec]",
                "Force [N]"
            )
    
    def plot_stability_analysis(self):
        """Plot data relevant to stability analysis"""
        # Get position and velocity data
        pos_data = self.logger.get_plot_data("position")
        vel_data = self.logger.get_plot_data("velocity")
        
        # Create phase portrait
        plt.figure(figsize=(8, 8))
        plt.plot(pos_data["values"][:, 0], vel_data["values"], 'b.')
        plt.axvline(x=pos_data["values"][:, 1][0], color='r', linestyle='--', 
                   label='Target Position')
        
        plt.title("Phase Portrait")
        plt.xlabel("Position [deg]")
        plt.ylabel("Velocity [deg/s]")
        plt.legend()
        plt.grid(True)
        plt.show()
            
    def plot_all(self):
        """Create all plots"""
        self.plot_joint_data()
        self.plot_torque_data()
        self.plot_perturbation_data()
        self.plot_contact_forces()
        self.plot_stability_analysis()
            
    def _create_plot(self, time: np.ndarray, values: np.ndarray, 
                    labels: List[str], title: str, xlabel: str, ylabel: str):
        """Create a single plot
        
        Args:
            time: Time data
            values: Values to plot
            labels: Labels for each line
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
        """
        plt.figure()
        for i in range(values.shape[1]):
            plt.plot(time, values[:, i], label=labels[i])
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()