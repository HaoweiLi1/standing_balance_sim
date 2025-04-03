import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_csv_data(file_path, delimiter=','):
    """Load data from a CSV file."""
    try:
        data = np.loadtxt(file_path, delimiter=delimiter)
        print(f"Loaded {file_path} with shape {data.shape}")
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def plot_trajectories(matlab_file, mujoco_file, applied_file, output_file="torque_comparison.png"):
    """Plot three torque trajectories with distinct line styles and markers."""
    # Load the three data files
    matlab_data = load_csv_data(matlab_file)
    mujoco_data = load_csv_data(mujoco_file)
    applied_data = load_csv_data(applied_file)
    
    if matlab_data is None or mujoco_data is None or applied_data is None:
        print("Error: Failed to load one or more data files.")
        return
    
    # Create the figure
    plt.figure(figsize=(12, 8))
    
    # Plot with distinct line styles and markers
    plt.plot(matlab_data[:, 0], matlab_data[:, 1], 'b--', linewidth=2.5, label="MATLAB Generated")
    plt.plot(mujoco_data[:, 0], mujoco_data[:, 1], 'r-', linewidth=2, label="MuJoCo Recorded")
    plt.plot(applied_data[:, 0], applied_data[:, 1], 'g-.', linewidth=2, label="Actually Applied")
    
    # Add markers at regular intervals
    marker_indices = np.arange(0, len(matlab_data), max(1, len(matlab_data) // 20))
    plt.plot(matlab_data[marker_indices, 0], matlab_data[marker_indices, 1], 'bo', markersize=5, fillstyle='none')
    plt.plot(mujoco_data[marker_indices, 0], mujoco_data[marker_indices, 1], 'r^', markersize=5, fillstyle='none')
    plt.plot(applied_data[marker_indices, 0], applied_data[marker_indices, 1], 'gs', markersize=5, fillstyle='none')
    
    # Add labels and legend
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Torque (Nm)", fontsize=12)
    plt.title("Torque Trajectory Comparison", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add statistics box
    stats_text = "Statistics:\n"
    stats_text += f"MATLAB max: {np.min(matlab_data[:, 1]):.2f} Nm, final: {matlab_data[-1, 1]:.2f} Nm\n"
    stats_text += f"MuJoCo max: {np.min(mujoco_data[:, 1]):.2f} Nm, final: {mujoco_data[-1, 1]:.2f} Nm\n"
    stats_text += f"Applied max: {np.min(applied_data[:, 1]):.2f} Nm, final: {applied_data[-1, 1]:.2f} Nm"
    
    plt.figtext(0.5, 0.01, stats_text, ha="center", fontsize=10, 
               bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot three torque trajectories together.')
    parser.add_argument('--matlab', type=str, required=True, help='Path to MATLAB generated torque file')
    parser.add_argument('--mujoco', type=str, required=True, help='Path to MuJoCo recorded torque file')
    parser.add_argument('--applied', type=str, required=True, help='Path to actually applied torque file')
    parser.add_argument('--output', type=str, default='torque_comparison.png', help='Output file path')
    
    args = parser.parse_args()
    
    plot_trajectories(args.matlab, args.mujoco, args.applied, args.output)

if __name__ == "__main__":
    main()