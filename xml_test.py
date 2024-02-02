import os
import mujoco

def load_mujoco_model(xml_file):
    # Get the absolute path of the XML file in the same directory as the script
    script_directory = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(script_directory, xml_file)

    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_path(xml_path)  # MuJoCo model
    return model

def main():
    # Specify the XML file name
    xml_file = "leg.xml"

    # Load MuJoCo model
    mujoco_model = load_mujoco_model(xml_file)

    # Create MuJoCo simulation
    sim = mujoco.MjSim(mujoco_model)

    # Run the simulation
    while True:
        # Your simulation logic here
        sim.step()

if __name__ == "__main__":
    main()
