import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob
import argparse
from datetime import datetime

# Configure matplotlib settings
plt.rcParams['text.usetex'] = True
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 12,
    'font.size': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})

class DataPlotter:
    """
    Data plotter for ankle exoskeleton simulation.
    Can be used standalone to load and plot data from saved files,
    or can be used directly with data from a running simulation.
    """
    
    def __init__(self, data_dir=None):
        """
        Initialize the data plotter.
        
        Args:
            data_dir: Directory containing data files. If None, will use the 
                     most recent directory in 'data/' (default: None)
        """
        self.data = {}
        
        if data_dir:
            self.data_dir = data_dir
        else:
            # Find the most recent data directory
            all_dirs = glob.glob('data/*/')
            if all_dirs:
                # Sort by creation time (newest first)
                all_dirs.sort(key=lambda x: os.path.getctime(x), reverse=True)
                self.data_dir = all_dirs[0]
            else:
                self.data_dir = None
                print("Warning: No data directory found.")
        
        # Load data if a directory was found or provided
        if self.data_dir:
            print(f"Loading data from: {self.data_dir}")
            self.load_all_data()
    
    def load_all_data(self):
        """Load all CSV files from the data directory."""
        if not self.data_dir or not os.path.exists(self.data_dir):
            print(f"Error: Data directory {self.data_dir} does not exist.")
            return
        
        # Find all CSV files
        csv_files = glob.glob(os.path.join(self.data_dir, '*.csv'))
        
        for file_path in csv_files:
            # Get dataset name from filename (without .csv extension)
            name = os.path.basename(file_path).replace('.csv', '')
            
            try:
                # Load the data
                data = np.loadtxt(file_path, delimiter=',')
                self.data[name] = data
                # print(f"Loaded dataset: {name} with shape {data.shape}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    def set_data(self, name, data):
        """
        Set data directly without loading from file.
        Used when plotting directly from a running simulation.
        
        Args:
            name: Name of the dataset
            data: Numpy array containing the data
        """
        self.data[name] = data
    
    def plot_simple(self, data_name, y_axis_label=None, title=None, show=True, save=False):
        """
        Create a simple time series plot.
        
        Args:
            data_name: Name of the dataset to plot
            y_axis_label: Label for y-axis
            title: Plot title
            show: Whether to show the plot
            save: Whether to save the plot
        """
        if data_name not in self.data:
            print(f"Error: Dataset '{data_name}' not found.")
            return
        
        data = self.data[data_name]
        
        if data.shape[1] < 2:
            print(f"Error: Dataset '{data_name}' does not have enough columns.")
            return
        
        # Extract time and value columns
        time = data[:, 0]
        values = data[:, 1]
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.plot(time, values, linestyle='-', color='b', 
                 label=y_axis_label if y_axis_label else data_name)
        
        # Add labels and title
        plt.xlabel('Time [s]')
        plt.ylabel(y_axis_label if y_axis_label else data_name)
        plt.title(title if title else f"{data_name} vs Time")
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend if there's a label
        if y_axis_label:
            plt.legend()
        
        # Save the plot if requested
        if save:
            plot_dir = os.path.join(self.data_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f"{data_name}_plot.png"), dpi=300, bbox_inches='tight')
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_comparison(self, data_names, labels=None, title=None, show=True, save=False):
        """
        Create a comparison plot of multiple datasets.
        
        Args:
            data_names: List of dataset names to plot
            labels: List of labels for each dataset
            title: Plot title
            show: Whether to show the plot
            save: Whether to save the plot
        """
        plt.figure(figsize=(12, 6))
        
        for i, name in enumerate(data_names):
            if name not in self.data:
                print(f"Warning: Dataset '{name}' not found, skipping.")
                continue
                
            data = self.data[name]
            
            if data.shape[1] < 2:
                print(f"Warning: Dataset '{name}' does not have enough columns, skipping.")
                continue
            
            # Get label
            label = labels[i] if labels and i < len(labels) else name
            
            # Plot data
            plt.plot(data[:, 0], data[:, 1], label=label)
        
        plt.xlabel('Time [s]')
        plt.ylabel('Value')
        plt.title(title if title else "Comparison Plot")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the plot if requested
        if save:
            plot_dir = os.path.join(self.data_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            filename = '_'.join(data_names)
            if len(filename) > 50:  # Avoid overly long filenames
                filename = f"comparison_plot_{datetime.now().strftime('%H%M%S')}"
            plt.savefig(os.path.join(plot_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_joint_state(self, show=True, save=False):
        """
        Plot joint position, goal position, and velocity for ankle and hip.
        
        Args:
            show: Whether to show the plot
            save: Whether to save the plot
        """
        # Check if required datasets exist
        ankle_required = ["joint_position", "goal_position", "joint_velocity"]
        hip_required = ["hip_joint_position", "hip_goal_position", "hip_joint_velocity"]
        
        ankle_data_exists = all(name in self.data for name in ankle_required)
        hip_data_exists = all(name in self.data for name in hip_required)
        
        if not ankle_data_exists and not hip_data_exists:
            print("Error: Required datasets not found.")
            return
        
        # Determine how many subplots we need
        num_joints = 0
        if ankle_data_exists:
            num_joints += 1
        if hip_data_exists:
            num_joints += 1
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, num_joints, figsize=(6*num_joints, 8), sharex=True)
        
        # Handle case where there's only one joint (makes axes indexing consistent)
        if num_joints == 1:
            axes = np.array([axes]).transpose()
        
        plot_idx = 0
        
        # Plot ankle data if available
        if ankle_data_exists:
            # Get data
            pos_data = self.data["joint_position"]
            goal_data = self.data["goal_position"]
            vel_data = self.data["joint_velocity"]
            
            # Plot position and goal
            axes[0, plot_idx].plot(pos_data[:, 0], pos_data[:, 1], label="Actual Position")
            axes[0, plot_idx].plot(goal_data[:, 0], goal_data[:, 1], label="Goal Position", linestyle='--')
            axes[0, plot_idx].set_ylabel("Angle [deg]")
            axes[0, plot_idx].set_title("Ankle Joint Position and Goal")
            axes[0, plot_idx].grid(True, linestyle='--', alpha=0.7)
            axes[0, plot_idx].legend()
            
            # Plot velocity
            axes[1, plot_idx].plot(vel_data[:, 0], vel_data[:, 1], label="Joint Velocity", color='g')
            axes[1, plot_idx].set_xlabel("Time [s]")
            axes[1, plot_idx].set_ylabel("Angular Velocity [deg/s]")
            axes[1, plot_idx].set_title("Ankle Joint Velocity")
            axes[1, plot_idx].grid(True, linestyle='--', alpha=0.7)
            axes[1, plot_idx].legend()
            
            plot_idx += 1
        
        # Plot hip data if available
        if hip_data_exists:
            # Get data
            pos_data = self.data["hip_joint_position"]
            goal_data = self.data["hip_goal_position"]
            vel_data = self.data["hip_joint_velocity"]
            
            # Plot position and goal
            axes[0, plot_idx].plot(pos_data[:, 0], pos_data[:, 1], label="Actual Position")
            axes[0, plot_idx].plot(goal_data[:, 0], goal_data[:, 1], label="Goal Position", linestyle='--')
            axes[0, plot_idx].set_ylabel("Angle [deg]")
            axes[0, plot_idx].set_title("Hip Joint Position and Goal")
            axes[0, plot_idx].grid(True, linestyle='--', alpha=0.7)
            axes[0, plot_idx].legend()
            
            # Plot velocity
            axes[1, plot_idx].plot(vel_data[:, 0], vel_data[:, 1], label="Joint Velocity", color='g')
            axes[1, plot_idx].set_xlabel("Time [s]")
            axes[1, plot_idx].set_ylabel("Angular Velocity [deg/s]")
            axes[1, plot_idx].set_title("Hip Joint Velocity")
            axes[1, plot_idx].grid(True, linestyle='--', alpha=0.7)
            axes[1, plot_idx].legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            plot_dir = os.path.join(self.data_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, "joint_state.png"), dpi=300, bbox_inches='tight')
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_torques(self, show=True, save=False):
        """
        Plot human, exo, and total ankle torques.
        
        Args:
            show: Whether to show the plot
            save: Whether to save the plot
        """
        # Check if required datasets exist
        required = ["human_torque", "exo_torque", "ankle_torque"]
        for name in required:
            if name not in self.data:
                print(f"Error: Required dataset '{name}' not found.")
                return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Get data
        human_data = self.data["human_torque"]
        exo_data = self.data["exo_torque"]
        ankle_data = self.data["ankle_torque"]
        
        # Plot torques
        plt.plot(human_data[:, 0], human_data[:, 1], label="Human Torque")
        plt.plot(exo_data[:, 0], exo_data[:, 1], label="Exo Torque")
        plt.plot(ankle_data[:, 0], ankle_data[:, 1], label="Total Ankle Torque")
        
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.title("Ankle Joint Torques")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the plot if requested
        if save:
            plot_dir = os.path.join(self.data_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, "torques.png"), dpi=300, bbox_inches='tight')
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_gravity_compensation(self, show=True, save=False):
        """
        Plot gravity torque and controller torque for comparison.
        
        Args:
            show: Whether to show the plot
            save: Whether to save the plot
        """
        # Check if required datasets exist
        if "gravity_torque" not in self.data or "control_torque" not in self.data:
            print("Error: Required datasets for gravity compensation not found.")
            return
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Get data
        gravity_data = self.data["gravity_torque"]
        control_data = self.data["control_torque"]
        
        # Plot torques
        plt.plot(gravity_data[:, 0], gravity_data[:, 1], label="Gravity Torque")
        plt.plot(control_data[:, 0], control_data[:, 1], label="Control Torque")
        
        plt.xlabel("Time [s]")
        plt.ylabel("Torque [Nm]")
        plt.title("Gravity vs Control Torque")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Save the plot if requested
        if save:
            plot_dir = os.path.join(self.data_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, "gravity_compensation.png"), dpi=300, bbox_inches='tight')
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_multi_joint_torques(self, show=True, save=False):
        """
        Plot torques for both ankle and hip joints.
        
        Args:
            show: Whether to show the plot
            save: Whether to save the plot
        """
        # Check if datasets exist
        ankle_data_exists = "ankle_torque" in self.data
        hip_data_exists = "human_hip_torque" in self.data
        
        if not ankle_data_exists and not hip_data_exists:
            print("Error: No joint torque data found.")
            return
            
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot ankle torques if available
        if ankle_data_exists and "human_torque" in self.data and "exo_torque" in self.data:
            human_data = self.data["human_torque"]
            exo_data = self.data["exo_torque"]
            ankle_data = self.data["ankle_torque"]
            
            axes[0].plot(human_data[:, 0], human_data[:, 1], label="Human")
            axes[0].plot(exo_data[:, 0], exo_data[:, 1], label="Exoskeleton")
            axes[0].plot(ankle_data[:, 0], ankle_data[:, 1], label="Total", linestyle='-.')
            axes[0].set_xlabel("Time [s]")
            axes[0].set_ylabel("Torque [Nm]")
            axes[0].set_title("Ankle Joint Torques")
            axes[0].grid(True, linestyle='--', alpha=0.7)
            axes[0].legend()
        
        # Plot hip torques if available
        if hip_data_exists:
            hip_data = self.data["human_hip_torque"]
            
            axes[1].plot(hip_data[:, 0], hip_data[:, 1], label="Human Hip")
            if "hip_gravity_torque" in self.data:
                gravity_data = self.data["hip_gravity_torque"]
                axes[1].plot(gravity_data[:, 0], gravity_data[:, 1], label="Gravity", linestyle='--')
            axes[1].set_xlabel("Time [s]")
            axes[1].set_ylabel("Torque [Nm]")
            axes[1].set_title("Hip Joint Torques")
            axes[1].grid(True, linestyle='--', alpha=0.7)
            axes[1].legend()
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            plot_dir = os.path.join(self.data_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, "multi_joint_torques.png"), dpi=300, bbox_inches='tight')
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()

    def plot_perturbation_response(self, show=True, save=False):
        """
        Plot perturbation and joint response.
        
        Args:
            show: Whether to show the plot
            save: Whether to save the plot
        """
        # Check if required datasets exist
        required = ["perturbation", "joint_position", "joint_velocity"]
        for name in required:
            if name not in self.data:
                print(f"Error: Required dataset '{name}' not found.")
                return
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Get data
        pert_data = self.data["perturbation"]
        pos_data = self.data["joint_position"]
        vel_data = self.data["joint_velocity"]
        
        # Plot perturbation
        ax1.plot(pert_data[:, 0], pert_data[:, 1], label="Perturbation Force", color='r')
        ax1.set_ylabel("Force [N]")
        ax1.set_title("Applied Perturbation")
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot position
        ax2.plot(pos_data[:, 0], pos_data[:, 1], label="Joint Position", color='b')
        ax2.set_ylabel("Angle [deg]")
        ax2.set_title("Joint Position Response")
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot velocity
        ax3.plot(vel_data[:, 0], vel_data[:, 1], label="Joint Velocity", color='g')
        ax3.set_xlabel("Time [s]")
        ax3.set_ylabel("Angular Velocity [deg/s]")
        ax3.set_title("Joint Velocity Response")
        ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot if requested
        if save:
            plot_dir = os.path.join(self.data_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, "perturbation_response.png"), dpi=300, bbox_inches='tight')
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_dashboard(self, show=True, save=False):
        """
        Create a comprehensive dashboard with multiple plots.
        
        Args:
            show: Whether to show the plot
            save: Whether to save the plot
        """
        # Create figure with grid layout
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(4, 2, figure=fig)
        
        # Define subplots
        ax1 = fig.add_subplot(gs[0, 0])  # Ankle joint position
        ax2 = fig.add_subplot(gs[0, 1])  # Hip joint position
        ax3 = fig.add_subplot(gs[1, 0])  # Ankle joint velocity
        ax4 = fig.add_subplot(gs[1, 1])  # Hip joint velocity
        ax5 = fig.add_subplot(gs[2, 0])  # Ankle torques
        ax6 = fig.add_subplot(gs[2, 1])  # Hip torques
        ax7 = fig.add_subplot(gs[3, :])  # Perturbation and response
        
        # Plot ankle joint position
        if "joint_position" in self.data and "goal_position" in self.data:
            pos_data = self.data["joint_position"]
            goal_data = self.data["goal_position"]
            ax1.plot(pos_data[:, 0], pos_data[:, 1], label="Actual")
            ax1.plot(goal_data[:, 0], goal_data[:, 1], label="Goal", linestyle='--')
            ax1.set_ylabel("Angle [deg]")
            ax1.set_title("Ankle Joint Position")
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend()
        
        # Plot hip joint position
        if "hip_joint_position" in self.data and "hip_goal_position" in self.data:
            pos_data = self.data["hip_joint_position"]
            goal_data = self.data["hip_goal_position"]
            ax2.plot(pos_data[:, 0], pos_data[:, 1], label="Actual")
            ax2.plot(goal_data[:, 0], goal_data[:, 1], label="Goal", linestyle='--')
            ax2.set_ylabel("Angle [deg]")
            ax2.set_title("Hip Joint Position")
            ax2.grid(True, linestyle='--', alpha=0.7)
            ax2.legend()
        
        # Plot ankle velocity
        if "joint_velocity" in self.data:
            vel_data = self.data["joint_velocity"]
            ax3.plot(vel_data[:, 0], vel_data[:, 1], label="Velocity", color='g')
            ax3.set_ylabel("Angular Velocity [deg/s]")
            ax3.set_title("Ankle Joint Velocity")
            ax3.grid(True, linestyle='--', alpha=0.7)
            ax3.legend()
        
        # Plot hip velocity
        if "hip_joint_velocity" in self.data:
            vel_data = self.data["hip_joint_velocity"]
            ax4.plot(vel_data[:, 0], vel_data[:, 1], label="Velocity", color='g')
            ax4.set_ylabel("Angular Velocity [deg/s]")
            ax4.set_title("Hip Joint Velocity")
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.legend()
        
        # Plot ankle torques
        if "human_torque" in self.data and "exo_torque" in self.data:
            human_data = self.data["human_torque"]
            exo_data = self.data["exo_torque"]
            ax5.plot(human_data[:, 0], human_data[:, 1], label="Human")
            ax5.plot(exo_data[:, 0], exo_data[:, 1], label="Exo")
            if "ankle_torque" in self.data:
                ankle_data = self.data["ankle_torque"]
                ax5.plot(ankle_data[:, 0], ankle_data[:, 1], label="Total", linestyle='-.')
            ax5.set_ylabel("Torque [Nm]")
            ax5.set_title("Ankle Joint Torques")
            ax5.grid(True, linestyle='--', alpha=0.7)
            ax5.legend()
        
        # Plot hip torques
        if "human_hip_torque" in self.data:
            hip_data = self.data["human_hip_torque"]
            ax6.plot(hip_data[:, 0], hip_data[:, 1], label="Human")
            if "hip_gravity_torque" in self.data:
                gravity_data = self.data["hip_gravity_torque"]
                ax6.plot(gravity_data[:, 0], gravity_data[:, 1], label="Gravity")
            ax6.set_ylabel("Torque [Nm]")
            ax6.set_title("Hip Joint Torques")
            ax6.grid(True, linestyle='--', alpha=0.7)
            ax6.legend()
        
        # Plot perturbation and response
        if "perturbation" in self.data:
            pert_data = self.data["perturbation"]
            if np.any(pert_data[:, 1] != 0):
                ax7.plot(pert_data[:, 0], pert_data[:, 1], label="Perturbation", color='r')
                
                # Create twin axes for joint positions
                ax7_twin = ax7.twinx()
                
                if "joint_position" in self.data:
                    pos_data = self.data["joint_position"]
                    ax7_twin.plot(pos_data[:, 0], pos_data[:, 1], label="Ankle", color='b', alpha=0.7)
                
                if "hip_joint_position" in self.data:
                    hip_data = self.data["hip_joint_position"]
                    ax7_twin.plot(hip_data[:, 0], hip_data[:, 1], label="Hip", color='g', alpha=0.7)
                    
                ax7.set_xlabel("Time [s]")
                ax7.set_ylabel("Force [N]", color='r')
                ax7_twin.set_ylabel("Angle [deg]")
                ax7.set_title("Perturbation and Joint Response")
                ax7.grid(True, linestyle='--', alpha=0.7)
                
                # Create combined legend
                lines1, labels1 = ax7.get_legend_handles_labels()
                lines2, labels2 = ax7_twin.get_legend_handles_labels()
                ax7.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # Set common x-label for all subplots
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.set_xlabel("Time [s]")
        
        # Add title for the whole figure
        plt.suptitle("Double-Pendulum with Ankle Exoskeleton Simulation Dashboard", fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leave space for suptitle
        
        # Save the plot if requested
        if save:
            plot_dir = os.path.join(self.data_dir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, "dashboard.png"), dpi=300, bbox_inches='tight')
        
        # Show the plot if requested
        if show:
            plt.show()
        else:
            plt.close()
    
    def generate_all_plots(self, show=False, save=True):
        """
        Generate all standard plots and save them.
        
        Args:
            show: Whether to show plots (default: False)
            save: Whether to save plots (default: True)
        """
        print("Generating all plots...")
        
        # Create individual plots
        self.plot_joint_state(show=show, save=save)
        self.plot_torques(show=show, save=save)
        self.plot_multi_joint_torques(show=show, save=save)  # Add new multi-joint plot
        self.plot_gravity_compensation(show=show, save=save)
        self.plot_perturbation_response(show=show, save=save)
        
        # Create dashboard
        self.plot_dashboard(show=show, save=save)
        
        print("Done generating plots.")


# Function to run the plotter directly
def main():
    parser = argparse.ArgumentParser(description='Plot data from ankle exoskeleton simulation.')
    parser.add_argument('--dir', type=str, help='Data directory to load')
    parser.add_argument('--dashboard', action='store_true', help='Generate dashboard plot')
    parser.add_argument('--joint', action='store_true', help='Generate joint state plot')
    parser.add_argument('--torque', action='store_true', help='Generate torque plot')
    parser.add_argument('--gravity', action='store_true', help='Generate gravity compensation plot')
    parser.add_argument('--perturbation', action='store_true', help='Generate perturbation response plot')
    parser.add_argument('--all', action='store_true', help='Generate all plots')
    parser.add_argument('--save', action='store_true', help='Save plots to file')
    parser.add_argument('--no-show', action='store_true', help='Do not display plots')
    
    args = parser.parse_args()
    
    # Create plotter
    plotter = DataPlotter(args.dir)
    
    # Determine which plots to generate
    if args.all:
        plotter.generate_all_plots(show=not args.no_show, save=args.save)
    else:
        show = not args.no_show
        save = args.save
        
        if args.dashboard:
            plotter.plot_dashboard(show=show, save=save)
        
        if args.joint:
            plotter.plot_joint_state(show=show, save=save)
        
        if args.torque:
            plotter.plot_torques(show=show, save=save)
        
        if args.gravity:
            plotter.plot_gravity_compensation(show=show, save=save)
        
        if args.perturbation:
            plotter.plot_perturbation_response(show=show, save=save)
        
        # If no specific plots were requested, show the dashboard
        if not any([args.dashboard, args.joint, args.torque, args.gravity, args.perturbation]):
            plotter.plot_dashboard(show=show, save=save)


if __name__ == "__main__":
    main()