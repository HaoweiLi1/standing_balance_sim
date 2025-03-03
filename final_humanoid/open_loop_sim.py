import os
import yaml
from simulation import AnkleExoSimulation

def main():
    """Run the simulation with open-loop control."""
    # Check if the optimal torques file exists
    if not os.path.exists("optimal_human_torques.csv"):
        print("ERROR: optimal_human_torques.csv file not found.")
        print("Please run the MATLAB script generate_optimal_torques.m first.")
        return
    
    # Make a copy of the original config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Make a copy of the original config
    modified_config = config.copy()
    
    # Update the controller configuration
    modified_config['config']['controllers']['human']['type'] = "OpenLoop"
    modified_config['config']['controllers']['human']['torque_file'] = "optimal_human_torques.csv"
    modified_config['config']['controllers']['human']['end_behavior'] = "zero"
    
    # Set a different ankle initial position to match the MATLAB optimization
    # This should be the same as theta0 in the MATLAB script
    modified_config['config']['ankle_initial_position_radians'] = 0.0872665
    
    # Set simulation time to match MATLAB optimization
    # This should match T in the MATLAB script
    modified_config['config']['simend'] = 10
    
    # Save the modified config to a temporary file
    temp_config_file = "temp_openloop_config.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(modified_config, f)
    
    print(f"Starting simulation with open-loop control from optimal_human_torques.csv")
    print(f"Initial ankle position: {modified_config['config']['ankle_initial_position_radians']} rad")
    print(f"Simulation time: {modified_config['config']['simend']} seconds")
    
    # Create and run the simulation
    sim = AnkleExoSimulation(temp_config_file)
    sim.initialize()
    sim.run()
    
    print("Simulation complete!")
    print(f"Results and plots saved to: {sim.logger.run_dir}")
    
    # Clean up temporary config file
    os.remove(temp_config_file)

if __name__ == "__main__":
    main()