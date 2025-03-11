import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def analyze_torque_csv(csv_path, title="Torque Trajectory from CSV"):
    """
    Analyze a torque trajectory CSV file and plot the data.
    
    Args:
        csv_path: Path to the CSV file
        title: Title for the plot
    """
    print(f"Analyzing CSV file: {csv_path}")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return
        
    try:
        # Load data
        # First try with header=None assuming no header
        data = pd.read_csv(csv_path, header=None)
        
        # Basic file info
        print(f"CSV file size: {os.path.getsize(csv_path)/1024:.2f} KB")
        print(f"Number of rows: {len(data)}")
        print(f"Number of columns: {data.shape[1]}")
        
        # If only 1 column, maybe it's space or tab delimited
        if data.shape[1] == 1 and (' ' in str(data.iloc[0, 0]) or '\t' in str(data.iloc[0, 0])):
            print("Detected space or tab delimiter, trying alternative loading...")
            if ' ' in str(data.iloc[0, 0]):
                data = pd.read_csv(csv_path, sep=' ', header=None)
            else:
                data = pd.read_csv(csv_path, sep='\t', header=None)
            print(f"After reloading - Number of columns: {data.shape[1]}")
        
        # Check if we have the expected 2 columns (time and torque)
        if data.shape[1] < 2:
            print("Warning: Expected at least 2 columns (time and torque)")
            print("Column found:")
            print(data.head())
            return
            
        # Assign column names for clarity
        if data.shape[1] == 2:
            data.columns = ['Time', 'Torque']
        else:
            # If more columns, name the first two and leave others as is
            cols = ['Time', 'Torque'] + [f'Column_{i+3}' for i in range(data.shape[1]-2)]
            data.columns = cols
            print(f"Note: Found {data.shape[1]} columns, expected 2 (time, torque)")
            
        # Display basic statistics
        print("\nData Statistics:")
        print(f"Time range: {data['Time'].min():.4f} to {data['Time'].max():.4f} seconds")
        print(f"Torque range: {data['Torque'].min():.4f} to {data['Torque'].max():.4f} Nm")
        print(f"Time step: {data['Time'].diff().mean():.6f} seconds")
        print(f"Max torque rate of change: {data['Torque'].diff().abs().max() / data['Time'].diff().mean():.2f} Nm/s")
        
        # Create detailed plot
        plt.figure(figsize=(12, 8))
        
        # Main plot - Torque vs Time
        plt.subplot(2, 1, 1)
        plt.plot(data['Time'], data['Torque'], 'b-', linewidth=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.title(f"{title} - Full Duration")
        
        # Calculate the range for the zoomed plot (focus on the first 20% of the time or first 2 seconds)
        zoom_end = min(data['Time'].max() * 0.2, 2.0)
        zoom_data = data[data['Time'] <= zoom_end]
        
        # Zoomed plot - Focus on initial motion
        plt.subplot(2, 1, 2)
        plt.plot(zoom_data['Time'], zoom_data['Torque'], 'r-', linewidth=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.title(f"Zoomed View (0 to {zoom_end:.1f} s)")
        
        plt.tight_layout()
        plt.savefig('torque_trajectory_analysis.png', dpi=300)
        print(f"Plot saved to torque_trajectory_analysis.png")
        plt.show()
        
        # Return data for further analysis if needed
        return data
        
    except Exception as e:
        print(f"Error analyzing CSV file: {e}")
        return None

def compare_with_mujoco_data(csv_path, mujoco_data_path=None):
    """
    Compare the CSV torque trajectory with actual MuJoCo output.
    
    Args:
        csv_path: Path to the original CSV trajectory file
        mujoco_data_path: Path to the CSV containing MuJoCo human_torque data
    """
    if mujoco_data_path is None:
        print("No MuJoCo data file provided for comparison")
        return
        
    try:
        # Load data from both files
        orig_data = pd.read_csv(csv_path, header=None)
        orig_data.columns = ['Time', 'Torque']
        
        mujoco_data = pd.read_csv(mujoco_data_path, header=None)
        mujoco_data.columns = ['Time', 'Torque']
        
        # Create comparison plot
        plt.figure(figsize=(12, 10))
        
        # Plot both trajectories
        plt.subplot(3, 1, 1)
        plt.plot(orig_data['Time'], orig_data['Torque'], 'b-', linewidth=2, label='Original CSV')
        plt.plot(mujoco_data['Time'], mujoco_data['Torque'], 'r-', linewidth=1, label='MuJoCo Output')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.title('Torque Trajectories Comparison')
        plt.legend()
        
        # Zoom on the initial part
        plt.subplot(3, 1, 2)
        zoom_end = min(orig_data['Time'].max() * 0.2, 2.0)
        plt.plot(orig_data[orig_data['Time'] <= zoom_end]['Time'], 
                orig_data[orig_data['Time'] <= zoom_end]['Torque'], 
                'b-', linewidth=2, label='Original CSV')
        plt.plot(mujoco_data[mujoco_data['Time'] <= zoom_end]['Time'], 
                mujoco_data[mujoco_data['Time'] <= zoom_end]['Torque'], 
                'r-', linewidth=1, label='MuJoCo Output')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Torque (Nm)')
        plt.title(f'Zoomed View (0 to {zoom_end:.1f} s)')
        plt.legend()
        
        # Difference plot
        plt.subplot(3, 1, 3)
        # Interpolate the datasets to a common time base for comparison
        # Use the original data time points as reference
        from scipy.interpolate import interp1d
        
        # Create interpolator for MuJoCo data
        mujoco_interp = interp1d(mujoco_data['Time'], mujoco_data['Torque'], 
                               kind='linear', bounds_error=False, fill_value=0)
                               
        # Interpolate MuJoCo data to original data time points
        mujoco_resampled = mujoco_interp(orig_data['Time'])
        
        # Calculate difference
        difference = orig_data['Torque'] - mujoco_resampled
        
        # Plot difference
        plt.plot(orig_data['Time'], difference, 'g-', linewidth=2)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xlabel('Time (s)')
        plt.ylabel('Torque Difference (Nm)')
        plt.title('Difference (Original - MuJoCo)')
        
        plt.tight_layout()
        plt.savefig('torque_comparison.png', dpi=300)
        print(f"Comparison plot saved to torque_comparison.png")
        plt.show()
        
        # Calculate and print some statistics
        print("\nComparison Statistics:")
        print(f"Max absolute difference: {np.abs(difference).max():.4f} Nm")
        print(f"Mean absolute difference: {np.abs(difference).mean():.4f} Nm")
        print(f"RMS difference: {np.sqrt(np.mean(difference**2)):.4f} Nm")
        
    except Exception as e:
        print(f"Error comparing CSV files: {e}")

if __name__ == "__main__":
    # Get the CSV file path from the user
    csv_path = input("Enter the path to your ankle_torque_trajectory.csv file: ")
    if not csv_path:
        csv_path = "ankle_torque_trajectory.csv"  # Default path
        
    # Analyze the CSV file
    data = analyze_torque_csv(csv_path)
    
    # Ask if user wants to compare with MuJoCo output
    mujoco_compare = input("\nDo you want to compare with MuJoCo output? (y/n): ")
    if mujoco_compare.lower() == 'y':
        mujoco_path = input("Enter the path to MuJoCo human_torque.csv file: ")
        if mujoco_path:
            compare_with_mujoco_data(csv_path, mujoco_path)