from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Any
import csv
from pathlib import Path

@dataclass
class TimeSeriesData:
    """Container for time series data"""
    time: np.ndarray
    values: np.ndarray
    columns: List[str]
    
    def __post_init__(self):
        """Validate data dimensions"""
        if len(self.time) != len(self.values):
            raise ValueError("Time and values must have the same length")
        if self.values.shape[1] != len(self.columns):
            raise ValueError("Number of columns must match values shape")

@dataclass
class SimulationLogger:
    """Main data logging class for simulation"""
    # Pre-allocate storage for different data types
    joint_data: Dict[str, TimeSeriesData] = field(default_factory=dict)
    torque_data: Dict[str, TimeSeriesData] = field(default_factory=dict)
    perturbation_data: Optional[TimeSeriesData] = None
    contact_data: Optional[TimeSeriesData] = None
    gravity_torque_data: Optional[TimeSeriesData] = None
    
    # Configuration
    save_path: Path = field(default_factory=lambda: Path("."))
    
    def initialize_storage(self, expected_steps: int):
        """Initialize data storage arrays
        
        Args:
            expected_steps: Expected number of simulation steps
        """
        # Time array for all series
        time = np.zeros(expected_steps)
        
        # Joint position and velocity
        joint_pos = np.zeros((expected_steps, 2))  # [actual, target]
        joint_vel = np.zeros((expected_steps, 1))
        
        self.joint_data["position"] = TimeSeriesData(
            time=time.copy(),
            values=joint_pos,
            columns=["actual_position", "target_position"]
        )
        
        self.joint_data["velocity"] = TimeSeriesData(
            time=time.copy(),
            values=joint_vel,
            columns=["velocity"]
        )
        
        # Torque data
        torque_data = np.zeros((expected_steps, 3))  # [human, exo, total]
        self.torque_data["ankle"] = TimeSeriesData(
            time=time.copy(),
            values=torque_data,
            columns=["human_torque", "exo_torque", "total_torque"]
        )
        
        # Perturbation data
        pert_data = np.zeros((expected_steps, 1))
        self.perturbation_data = TimeSeriesData(
            time=time.copy(),
            values=pert_data,
            columns=["perturbation_force"]
        )
        
        # Contact force data
        contact_data = np.zeros((expected_steps, 2))  # [front, back]
        self.contact_data = TimeSeriesData(
            time=time.copy(),
            values=contact_data,
            columns=["front_contact", "back_contact"]
        )

        # Gravity torque data
        gravity_data = np.zeros((expected_steps, 1))
        self.gravity_torque_data = TimeSeriesData(
            time=time.copy(),
            values=gravity_data,
            columns=["gravity_torque"]
        )
    
    def log_step(self, step: int, time: float, data: Dict[str, Any]):
        """Log data for a single simulation step
        
        Args:
            step: Current simulation step
            time: Current simulation time
            data: Dictionary containing data to log
        """
        # Update time arrays
        for series in self.joint_data.values():
            series.time[step] = time
        for series in self.torque_data.values():
            series.time[step] = time
        if self.perturbation_data:
            self.perturbation_data.time[step] = time
        if self.contact_data:
            self.contact_data.time[step] = time
        if self.gravity_torque_data:
            self.gravity_torque_data.time[step] = time
            
        # Log joint data
        if "joint_position" in data:
            self.joint_data["position"].values[step] = [
                data["joint_position"],
                data["target_position"]
            ]
        if "joint_velocity" in data:
            self.joint_data["velocity"].values[step] = [data["joint_velocity"]]
            
        # Log torque data
        if "human_torque" in data:
            self.torque_data["ankle"].values[step] = [
                data["human_torque"],
                data["exo_torque"],
                data["total_torque"]
            ]
            
        # Log perturbation
        if "perturbation" in data:
            self.perturbation_data.values[step] = [data["perturbation"]]
            
        # Log contact forces
        if "contact_forces" in data:
            self.contact_data.values[step] = data["contact_forces"]

        # Log gravity torque
        if "gravity_torque" in data:
            self.gravity_torque_data.values[step] = [data["gravity_torque"]]
    
    def save_data(self):
        """Save all logged data to CSV files"""
        # Save joint data
        for name, data in self.joint_data.items():
            filename = self.save_path / f"joint_{name}_data.csv"
            self._save_timeseries(filename, data)
            
        # Save torque data
        for name, data in self.torque_data.items():
            filename = self.save_path / f"{name}_torque_data.csv"
            self._save_timeseries(filename, data)
            
        # Save perturbation data
        if self.perturbation_data:
            filename = self.save_path / "perturbation_data.csv"
            self._save_timeseries(filename, self.perturbation_data)
            
        # Save contact data
        if self.contact_data:
            filename = self.save_path / "contact_force_data.csv"
            self._save_timeseries(filename, self.contact_data)

        # Save gravity torque data
        if self.gravity_torque_data:
            filename = self.save_path / "gravity_torque_data.csv"
            self._save_timeseries(filename, self.gravity_torque_data)
    
    def _save_timeseries(self, filename: Path, data: TimeSeriesData):
        """Save a single time series to CSV
        
        Args:
            filename: Path to save the CSV file
            data: TimeSeriesData object to save
        """
        # Create header
        header = ["time"] + data.columns
        
        # Combine time and values
        full_data = np.column_stack((data.time, data.values))
        
        # Save to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(full_data)
    
    def get_plot_data(self, data_type: str, columns: Optional[List[str]] = None) -> dict:
        """Get data formatted for plotting
        
        Args:
            data_type: Type of data to retrieve (e.g., "joint_position", "torque")
            columns: Optional list of column names to retrieve
            
        Returns:
            Dictionary with time and values arrays for plotting
        """
        # Find the requested data
        if data_type in self.joint_data:
            data = self.joint_data[data_type]
        elif data_type in self.torque_data:
            data = self.torque_data[data_type]
        elif data_type == "perturbation" and self.perturbation_data:
            data = self.perturbation_data
        elif data_type == "contact" and self.contact_data:
            data = self.contact_data
        elif data_type == "gravity_torque" and self.gravity_torque_data:
            data = self.gravity_torque_data
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Filter columns if requested
        if columns:
            col_indices = [data.columns.index(col) for col in columns]
            values = data.values[:, col_indices]
            plot_columns = columns
        else:
            values = data.values
            plot_columns = data.columns
            
        return {
            "time": data.time,
            "values": values,
            "columns": plot_columns
        }