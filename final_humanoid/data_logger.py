from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Any
import csv
from pathlib import Path

@dataclass
class TimeSeriesData:
    """Container for time series data."""
    time: np.ndarray
    values: np.ndarray
    columns: List[str]
    
    def __post_init__(self):
        """Validate data dimensions."""
        if len(self.time) != self.values.shape[0]:
            raise ValueError("Time and values must have the same length.")
        if self.values.ndim < 2:
            raise ValueError("Values array must be 2-dimensional.")
        if self.values.shape[1] != len(self.columns):
            raise ValueError("Number of columns must match values shape.")

@dataclass
class SimulationLogger:
    """Main data logging class for simulation."""
    joint_data: Dict[str, TimeSeriesData] = field(default_factory=dict)
    torque_data: Dict[str, TimeSeriesData] = field(default_factory=dict)
    perturbation_data: Optional[TimeSeriesData] = None
    contact_data: Optional[TimeSeriesData] = None
    gravity_torque_data: Optional[TimeSeriesData] = None

    # 保存数据的路径（默认为当前目录）
    save_path: Path = field(default_factory=lambda: Path("."))

    def initialize_storage(self, expected_steps: int):
        """Initialize data storage arrays.
        
        Args:
            expected_steps: Expected number of simulation steps.
        """
        # 预分配时间数组
        time = np.zeros(expected_steps)
        
        # 关节数据：位置数据包含实际值和目标值
        joint_pos = np.zeros((expected_steps, 2))  # [actual_position, target_position]
        joint_vel = np.zeros((expected_steps, 1))    # [velocity]
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
        
        # 扭矩数据：存储人类扭矩、外骨骼扭矩和总扭矩
        torque_data = np.zeros((expected_steps, 3))  # [human_torque, exo_torque, total_torque]
        self.torque_data["ankle"] = TimeSeriesData(
            time=time.copy(),
            values=torque_data,
            columns=["human_torque", "exo_torque", "total_torque"]
        )
        
        # 扰动数据
        pert_data = np.zeros((expected_steps, 1))
        self.perturbation_data = TimeSeriesData(
            time=time.copy(),
            values=pert_data,
            columns=["perturbation_force"]
        )
        
        # 接触力数据：记录前后接触力
        contact_data = np.zeros((expected_steps, 2))
        self.contact_data = TimeSeriesData(
            time=time.copy(),
            values=contact_data,
            columns=["front_contact", "back_contact"]
        )
        
        # 重力扭矩数据
        gravity_data = np.zeros((expected_steps, 1))
        self.gravity_torque_data = TimeSeriesData(
            time=time.copy(),
            values=gravity_data,
            columns=["gravity_torque"]
        )
    
    def log_step(self, step: int, time_val: float, data: Dict[str, Any]):
        """Log data for a single simulation step.
        
        Args:
            step: Current simulation step.
            time_val: Current simulation time.
            data: Dictionary containing data to log. Expected keys include:
                  "joint_position", "target_position", "joint_velocity",
                  "human_torque", "exo_torque", "total_torque", "perturbation",
                  "contact_forces", "gravity_torque".
        """
        # 更新所有系列的时间数据
        for series in self.joint_data.values():
            series.time[step] = time_val
        for series in self.torque_data.values():
            series.time[step] = time_val
        if self.perturbation_data:
            self.perturbation_data.time[step] = time_val
        if self.contact_data:
            self.contact_data.time[step] = time_val
        if self.gravity_torque_data:
            self.gravity_torque_data.time[step] = time_val
        
        # 记录关节位置数据
        if "joint_position" in data and "target_position" in data:
            self.joint_data["position"].values[step] = [
                data["joint_position"],
                data["target_position"]
            ]
        # 记录关节速度数据
        if "joint_velocity" in data:
            self.joint_data["velocity"].values[step] = [data["joint_velocity"]]
        
        # 记录扭矩数据
        if "human_torque" in data and "exo_torque" in data and "total_torque" in data:
            self.torque_data["ankle"].values[step] = [
                data["human_torque"],
                data["exo_torque"],
                data["total_torque"]
            ]
        
        # 记录扰动数据
        if "perturbation" in data:
            self.perturbation_data.values[step] = [data["perturbation"]]
        
        # 记录接触力数据
        if "contact_forces" in data:
            self.contact_data.values[step] = data["contact_forces"]
        
        # 记录重力扭矩数据
        if "gravity_torque" in data:
            self.gravity_torque_data.values[step] = [data["gravity_torque"]]
    
    def save_data(self):
        """Save all logged data to CSV files."""
        # 保存关节数据
        for name, ts_data in self.joint_data.items():
            filename = self.save_path / f"joint_{name}_data.csv"
            self._save_timeseries(filename, ts_data)
        # 保存扭矩数据
        for name, ts_data in self.torque_data.items():
            filename = self.save_path / f"{name}_torque_data.csv"
            self._save_timeseries(filename, ts_data)
        # 保存扰动数据
        if self.perturbation_data:
            filename = self.save_path / "perturbation_data.csv"
            self._save_timeseries(filename, self.perturbation_data)
        # 保存接触力数据
        if self.contact_data:
            filename = self.save_path / "contact_force_data.csv"
            self._save_timeseries(filename, self.contact_data)
        # 保存重力扭矩数据
        if self.gravity_torque_data:
            filename = self.save_path / "gravity_torque_data.csv"
            self._save_timeseries(filename, self.gravity_torque_data)
    
    def _save_timeseries(self, filename: Path, data: TimeSeriesData):
        """Save a single time series to a CSV file.
        
        Args:
            filename: Path to the CSV file.
            data: TimeSeriesData object.
        """
        header = ["time"] + data.columns
        full_data = np.column_stack((data.time, data.values))
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(full_data)
    
    def get_plot_data(self, data_type: str, columns: Optional[List[str]] = None) -> dict:
        """
        Get data formatted for plotting.
        
        Args:
            data_type: The key for the data series (e.g., "position", "ankle", "perturbation", "contact", "gravity_torque").
            columns: Optional list of column names to filter.
        
        Returns:
            Dictionary with keys "time", "values", and "columns".
        """
        if data_type in self.joint_data:
            ts_data = self.joint_data[data_type]
        elif data_type in self.torque_data:
            ts_data = self.torque_data[data_type]
        elif data_type == "perturbation" and self.perturbation_data is not None:
            ts_data = self.perturbation_data
        elif data_type == "contact" and self.contact_data is not None:
            ts_data = self.contact_data
        elif data_type == "gravity_torque" and self.gravity_torque_data is not None:
            ts_data = self.gravity_torque_data
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        if columns:
            col_indices = [ts_data.columns.index(col) for col in columns]
            values = ts_data.values[:, col_indices]
            plot_columns = columns
        else:
            values = ts_data.values
            plot_columns = ts_data.columns
        
        return {
            "time": ts_data.time,
            "values": values,
            "columns": plot_columns
        }
