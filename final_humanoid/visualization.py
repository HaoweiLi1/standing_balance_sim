import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from data_logger import SimulationLogger

class Visualizer:
    """Visualization class for simulation data."""
    
    def __init__(self, logger: SimulationLogger):
        """
        Initialize visualizer.
        
        Args:
            logger: Data logger containing simulation data.
        """
        self.logger = logger
        plt.rcParams['text.usetex'] = True
        
    def plot_joint_data(self):
        """Plot joint position and velocity data (converted to degrees)."""
        # 获取关节位置数据（假设原始数据为弧度，转换为度）
        pos_data = self.logger.get_plot_data("position")
        # 复制数据并进行转换：第一列为实际位置，第二列为目标位置
        pos_deg = pos_data["values"] * (180.0 / np.pi)
        self._create_plot(
            pos_data["time"],
            pos_deg,
            ["Actual Position", "Target Position"],
            "Joint Position vs Time",
            "Time [sec]",
            "Position [deg]"
        )
        
        # 绘制关节速度（转换为 deg/s）
        vel_data = self.logger.get_plot_data("velocity")
        vel_deg = vel_data["values"] * (180.0 / np.pi)
        self._create_plot(
            vel_data["time"],
            vel_deg,
            ["Joint Velocity"],
            "Joint Velocity vs Time",
            "Time [sec]",
            "Velocity [deg/s]"
        )
        
    def plot_torque_data(self):
        """Plot torque data."""
        torque_data = self.logger.get_plot_data("ankle")
        gravity_data = self.logger.get_plot_data("gravity_torque")
        
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
        """Plot gravity compensation related data."""
        torque_data = self.logger.get_plot_data("ankle", ["human_torque", "exo_torque"])
        gravity_data = self.logger.get_plot_data("gravity_torque")
        
        plt.figure(figsize=(10, 6))
        net_control = -(torque_data["values"][:, 0] + torque_data["values"][:, 1])
        plt.plot(torque_data["time"], net_control, label="Net Control Torque")
        plt.plot(gravity_data["time"], gravity_data["values"], '--', label="Gravity Torque")
        
        plt.title("Gravity Compensation Performance")
        plt.xlabel("Time [sec]")
        plt.ylabel("Torque [Nm]")
        plt.legend()
        plt.grid(True)
        plt.show()
        
    def plot_perturbation_data(self):
        """Plot perturbation force data."""
        try:
            pert_data = self.logger.get_plot_data("perturbation")
        except ValueError:
            return
        self._create_plot(
            pert_data["time"],
            pert_data["values"],
            ["Perturbation Force"],
            "Perturbation Force vs Time",
            "Time [sec]",
            "Force [N]"
        )
            
    def plot_contact_forces(self):
        """Plot contact force data."""
        try:
            contact_data = self.logger.get_plot_data("contact")
        except ValueError:
            return
        self._create_plot(
            contact_data["time"],
            contact_data["values"],
            ["Front Contact", "Back Contact"],
            "Contact Forces vs Time",
            "Time [sec]",
            "Force [N]"
        )
    
    def plot_stability_analysis(self):
        """
        Plot a phase portrait of joint position vs. velocity.
        Joint positions and velocities are converted to degrees.
        """
        pos_data = self.logger.get_plot_data("position")
        vel_data = self.logger.get_plot_data("velocity")
        # 假定第一列为实际关节位置、单位为弧度；转换为度
        actual_pos_deg = pos_data["values"][:, 0] * (180.0 / np.pi)
        # 同理，速度转换为 deg/s
        vel_deg = vel_data["values"] * (180.0 / np.pi)
        
        plt.figure(figsize=(8, 8))
        plt.plot(actual_pos_deg, vel_deg, 'b.')
        # 绘制目标位置的垂直线（转换为度）
        target_deg = pos_data["values"][0, 1] * (180.0 / np.pi)
        plt.axvline(x=target_deg, color='r', linestyle='--', label='Target Position')
        
        plt.title("Phase Portrait")
        plt.xlabel("Position [deg]")
        plt.ylabel("Velocity [deg/s]")
        plt.legend()
        plt.grid(True)
        plt.show()
            
    def plot_3d_pose_trajectory(self, positions: np.ndarray, orientations: List[np.ndarray]):
        """
        Plot 3D trajectory of the body center of mass and its orientation.
        
        Args:
            positions: NumPy array of shape (N, 4), where columns: [time, x, y, z].
            orientations: List or array of orientation matrices flattened to 9 elements per time step.
        """
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_zlabel('$Z$')
        ax.set_title('Body Center of Mass Trajectory')
        
        # 绘制轨迹，假定 positions 数组第一列为时间，其余为 x,y,z
        ax.plot(positions[1:, 1], positions[1:, 2], positions[1:, 3], marker=".", color='k')
        
        # 每隔一定步数绘制方向向量
        for i in range(1, len(positions), 10):
            # 假定 orientations[i] 为 1D 数组长度9，构成3x3矩阵
            ori = orientations[i].reshape((3, 3))
            # 绘制三个轴向量，从当前质心位置出发
            pos = positions[i, 1:4]
            ax.quiver(pos[0], pos[1], pos[2],
                      ori[0, 0], ori[1, 0], ori[2, 0],
                      color='r', length=0.1)
            ax.quiver(pos[0], pos[1], pos[2],
                      ori[0, 1], ori[1, 1], ori[2, 1],
                      color='g', length=0.1)
            ax.quiver(pos[0], pos[1], pos[2],
                      ori[0, 2], ori[1, 2], ori[2, 2],
                      color='b', length=0.1)
        
        ax.set_ylim(-0.5, 0.5)
        plt.show()
        
    def plot_all(self):
        """Generate all plots."""
        self.plot_joint_data()
        self.plot_torque_data()
        self.plot_perturbation_data()
        self.plot_contact_forces()
        self.plot_stability_analysis()
        # 如果需要3D绘图，且 logger 中有相应数据，可调用 plot_3d_pose_trajectory
        # 例如：
        # com_data = self.logger.get_plot_data("com")  # 假设记录了COM数据
        # orientation_data = self.logger.get_plot_data("orientation")  # 假设记录了方向数据
        # self.plot_3d_pose_trajectory(com_data["values"], orientation_data["values"])
        
    def _create_plot(self, time: np.ndarray, values: np.ndarray, 
                     labels: List[str], title: str, xlabel: str, ylabel: str):
        """
        Create a single plot.
        
        Args:
            time: Array of time values.
            values: 2D array of data to plot (columns correspond to different series).
            labels: List of labels for each series.
            title: Plot title.
            xlabel: X-axis label.
            ylabel: Y-axis label.
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
