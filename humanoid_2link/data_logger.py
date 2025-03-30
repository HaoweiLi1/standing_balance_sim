import numpy as np
import os
import time
from datetime import datetime

class DataLogger:
    """
    Data logger for ankle exoskeleton simulation.
    Handles creation, collection, and saving of various data streams.
    """
    
    def __init__(self, output_dir='data'):
        """
        Initialize the data logger.
        
        Args:
            output_dir: Base directory for saving data (default: 'data')
        """
        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory with timestamp
        self.run_dir = os.path.join(output_dir, self.timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        
        # Dictionary to store all data arrays
        self.data_arrays = {}
        
        print(f"Data logger initialized. Saving data to: {self.run_dir}")
    
    def create_dataset(self, name, columns):
        """
        Create a new dataset for logging.
        
        Args:
            name: Name of the dataset (used for filename when saving)
            columns: Number of columns in the dataset
        """
        self.data_arrays[name] = np.empty((0, columns))
    
    def log_data(self, name, data_row):
        """
        Log a row of data to a specific dataset.
        
        Args:
            name: Name of the dataset to log to
            data_row: Numpy array containing the data row
        """
        if name not in self.data_arrays:
            # If dataset doesn't exist, create it with the right number of columns
            self.create_dataset(name, len(data_row))
        
        self.data_arrays[name] = np.vstack((self.data_arrays[name], data_row))
    
    def save_dataset(self, name, delimiter=",", fmt="%.4f"):
        """
        Save a specific dataset to a CSV file.
        
        Args:
            name: Name of the dataset to save
            delimiter: Delimiter character for CSV (default: comma)
            fmt: Format string for numbers (default: 3 decimal places)
        """
        if name in self.data_arrays:
            filepath = os.path.join(self.run_dir, f"{name}.csv")
            np.savetxt(filepath, self.data_arrays[name][1:], delimiter=delimiter, fmt=fmt)
            print(f"Saved dataset '{name}' to {filepath}")
        else:
            print(f"Warning: Dataset '{name}' not found")
    
    def save_all(self, delimiter=",", fmt="%.4f"):
        """
        Save all datasets to CSV files.
        
        Args:
            delimiter: Delimiter character for CSV (default: comma)
            fmt: Format string for numbers (default: 3 decimal places)
        """
        for name in self.data_arrays:
            self.save_dataset(name, delimiter, fmt)
    
    def get_dataset(self, name):
        """
        Get a dataset array.
        
        Args:
            name: Name of the dataset to retrieve
            
        Returns:
            The numpy array containing the dataset (without the first empty row),
            or None if the dataset doesn't exist
        """
        if name in self.data_arrays:
            # return self.data_arrays[name][1:]  # Skip the first empty row
            return self.data_arrays[name]  # Return the full array, not sliced
        return None
    
    def save_config(self, config):
        """
        Save the configuration used for this run.
        
        Args:
            config: Configuration dictionary or YAML string
        """
        import yaml
        
        # If config is already a string, use it directly
        if isinstance(config, str):
            config_str = config
        else:
            # Otherwise, convert to YAML
            config_str = yaml.dump(config, default_flow_style=False)
        
        # Save to file
        config_path = os.path.join(self.run_dir, "config.yaml")
        with open(config_path, "w") as f:
            f.write(config_str)
        
        print(f"Saved configuration to {config_path}")
        
    def create_standard_datasets(self):
        """
        Create standard datasets used in ankle exo simulation.
        """
        self.create_dataset("human_torque", 2)  # time, torque
        self.create_dataset("exo_torque", 2)    # time, torque
        self.create_dataset("ankle_torque", 2)  # time, torque
        self.create_dataset("joint_position", 2)  # time, position
        self.create_dataset("joint_velocity", 2)  # time, velocity
        self.create_dataset("goal_position", 2)   # time, goal
        self.create_dataset("gravity_torque", 2)  # time, torque
        self.create_dataset("perturbation", 2)    # time, magnitude
        self.create_dataset("control_torque", 2)  # time, torque
        self.create_dataset("body_com", 4)        # time, x, y, z
        self.create_dataset("constraint_force", 5)  # time, f1, f2, f3, f4
        self.create_dataset("contact_force", 3)   # time, front, back
        self.create_dataset("human_rtd", 3)       # time, rtd_value, rtd_limit