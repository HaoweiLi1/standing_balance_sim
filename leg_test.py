import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from typing import Callable, Optional, Union, List
import scipy.linalg
import mediapy as media
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import time
from queue import Queue
import xml.etree.ElementTree as ET
import imageio

xml_path = 'xml_files\leg.xml'
simend = 5
K_p = 0

def plot_3d_pose_trajectory(positions, orientations):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Body Center of Mass Pose Trajectory')

    # Plot the trajectory
    ax.plot(positions[1:, 1], positions[1:, 2], positions[1:, 3], marker=".",color='k')

    # Draw the orientation vectors
    for i in range(1,len(positions),10):
        ax.quiver(positions[i, 1], positions[i, 2], positions[i, 3],
                  orientations[i][0], orientations[i][3], orientations[i][6],
                  color='r', length=0.1)
        
        ax.quiver(positions[i, 1], positions[i, 2], positions[i, 3],
                  orientations[i][1], orientations[i][4], orientations[i][7],
                  color='g', length=0.1)

        ax.quiver(positions[i, 1], positions[i, 2], positions[i, 3],
                  orientations[i][2], orientations[i][5], orientations[i][8],
                  color='b', length=0.1)
    ax.set_ylim(-0.5,0.5)
    legend_entries = ["Body Cartesian Position", "Body X-Orientation", "Body Y-Orientation", "Body Z-Orientation"]
    plt.legend(labels=legend_entries)
    plt.show()

def plot_columns(data_array, y_axis):
    """
    Plot the data in the first column versus the data in the second column.

    Parameters:
    - data_array: NumPy array with at least two columns.

    Returns:
    - None
    """
    # Check if the array has at least two columns
    if data_array.shape[1] < 2:
        print("Error: The input array must have at least two columns.")
        return

    # Extract the columns for plotting
    x_values = data_array[1:, 0]
    y_values = data_array[1:, 1]

    # Plot the data
    plt.plot(x_values, y_values, linestyle='-', color='b', label=y_axis)

    # Add labels and a title
    plt.xlabel('Time [sec]')
    plt.ylabel(y_axis)
    plt.title(y_axis + " versus Time")

    # Add a legend
    plt.legend()

    # Show the plot
    plt.show()

def generate_large_impulse(perturbation_queue, impulse_time):
    
    while True: # run continuously once we start a thread 
                # for this function
        
        # empty the queue from the previous impulse generation
        while not perturbation_queue.empty():
            perturbation_queue.get()

        wait_time = np.random.uniform(1,2)
        time.sleep(wait_time)
        
        # Generate a large impulse
        perturbation = np.random.uniform(400, 401)

        # this will generate either -1 or 1
        direction_bool = np.random.choice([-1,1])
        # print(direction_bool)
        start_time = time.time()
        while (time.time() - start_time) < impulse_time:

            # Put the generated impulse into the result queue
            perturbation_queue.put(direction_bool*perturbation)

def controller(model, data):
    """
    Controller function for the leg. The only two parameters 
    for this function can be mujoco.model and mujoco.data
    otherwise it doesn't seem to run properly
    """

    if actuator_type == "torque":
        # model.actuator_gainprm[1, 0] = 1
        # print(model.actuator_gainprm)
        # model.actuator_gainprm[0, 0] = 1 
        # print(str(data.sensordata[0]) + ", " + str(data.sensordata[1]))
        # print()

        # GRAVITY COMPENSATION #
        human_torque = K_p * \
            (data.sensordata[0] - 0) #5*np.pi/180 )
        exo_torque = -1*(human_torque)
        data.ctrl[0] = human_torque
        # data.ctrl[1] = exo_torque
        # print(data.ctrl[0])
        # data.ctrl[0] = -0.5 * \
        #     (data.sensordata[0] - 0.0) - \
        #     1 * (data.sensordata[1] - 0.0)
        # print(model.actuator_gainprm)
        # pass
        
    elif actuator_type == "servo":
        kp = 100.0
        # model.actuator_gainrpm are the gain parameters for the actuators
        # Each N*3 rows of model.actuator_gainrpm represents the gains for
        # distinct actuators. I think the gains are structure to be a PID
        # controller (with the default "affine").
        # I think the structure of the gain parameters is as followed
        # gain_term = gain_prm[0] + gain_prm[1]*length + gain_prm[2]*velocity

        model.actuator_gainprm[1, 0] = kp
        # print(model.actuator_gainprm)
        model.actuator_biasprm[1, 1] = -kp
        # print(model.actuator_biasprm)
        data.ctrl[1] = -np.pi/8

        kv = 10.0
        model.actuator_gainprm[2, 0] = kv
        model.actuator_biasprm[2, 2] = -kv
        data.ctrl[2] = 0.0
    
    # Log the actuator torque and timestep to an array for later use
    control_torque_time_array = np.array([data.time, human_torque])
    control_log_queue.put(control_torque_time_array)

    counter = 0
    if not counter_queue.empty():
        counter = counter_queue.get()
    counter += 1
    counter_queue.put(counter)

    x_perturbation=0
    
    if not perturbation_queue.empty():
        # print(f"perturbation: {perturbation_queue.get()}, time: {time.time()-start}")
        x_perturbation = perturbation_queue.get()
    
    # data.xfrc_applied[i] = [ F_x, F_y, F_z, R_x, R_y, R_z]
    data.xfrc_applied[1] = [x_perturbation, 0, 0, 0., 0., 0.]

    # Apply joint perturbations in Joint Space
    # data.qfrc_applied = [ q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8]
    # data.qfrc_applied[3] = rx_perturbation
    # data.qfrc_applied[4] = ry_perturbation

    ###############################################################
    ###############################################################
    
    # data.xfrc_applied specifies the Cartesion Space forces applied to the
    # BODIES' Center of Gravities in our model. 

    # Each entry of data.xfrc_applied represents the forces and torques
    # applied to distinct bodies in our model. 
    # The structure is an R6 vector:
    # data.xfrc_applied[i] = [ F_x, F_y, F_z, R_x, R_y, R_z]

    # Indices of data.xfrc_applied and relation to model bodies:
    # i = 0 ; Worldframe body --> Doesn't appear to move when subjected to forces?
    # i = 1 ; Shin body
    # i = 2 ; Foot body

    ###############################################################
    ###############################################################
    
    # data.qfrc_applied represents the Joint Space forces applied to the
    # JOINTS of our model

    # Each entry of data.qfrc_applied represents a degree of freedom
    # of the joints. Since we have different types of joints in our model
    # the structure of data.qfrc_applied is less rigidly defined than
    # data.xfrc_applied.

    # Indies of data.qfrc_applied and relation to model joints:
    # First joint is the knee
    # i = 0 ; knee joint's associated body x coord
    # i = 1 ; knee joint's associated body y coord
    # i = 2 ; knee joint's associated body z coord
    # i = 3 ; Knee R_x
    # i = 4 ; Knee R_y
    # i = 5 ; Knee R_z
    # i = 6 ; X-axis ankle rotation
    # i = 7 ; Y-axis ankle rotation

    ###############################################################
    ###############################################################

def calculate_kp_and_geom(weight, height):
    
    M_total = weight # [kg]
    H_total = height # [meters]
    m_feet = 2 * 0.0145 * M_total
    m_body = M_total - m_feet
    l_COM = 0.575*H_total
    l_foot = 0.152*H_total
    a = 0.19*l_foot
    K_p = m_body * 9.81 * l_COM

    return m_feet, m_body, l_COM, l_foot, a, K_p

    
tree = ET.parse(xml_path)
root = tree.getroot()
# MODIFY THE XML FILE AND INSERT THE MASS AND LENGTH
# PROPERTIES OF INTEREST WHICH ARE CALCULATED BY 
M_total = 80 # kg
H_total = 1.78 # meters
m_feet, m_body, l_COM, l_foot, a, K_p = calculate_kp_and_geom \
                                        (M_total, H_total)

## SETTING XML GEOMETRIES TO MATCH LITERATURE VALUES ###
for geom in root.iter('geom'):
        if geom.get('name') == "shin_geom":

            geom.set('fromto', f'0 0 0 0 0 {-H_total}')
            # geom.set('pos', f'0 0 {H_total-l_COM}')
            geom.set('mass', str(m_body))
            
        elif geom.get('name') == "foot1_right":

            geom.set('fromto', f'0 .02 0 {l_foot} .02 0')
            geom.set('mass', str(m_feet))

for body in root.iter('body'):
        if body.get('name') == "le_foot":
             body.set('pos',  f'-{a} 0 -{H_total+0.02}')

tree.write('xml_files\modified_model.xml')
########

#get the full path
modified_xml_path = 'xml_files\modified_model.xml'
script_directory = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_directory, xml_path)
model = mj.MjModel.from_xml_path(modified_xml_path)  # MuJoCo XML model

data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = True
# opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
opt.flags[mj.mjtVisFlag.mjVIS_ACTUATOR] = True
opt.flags[mj.mjtVisFlag.mjVIS_COM] = True

# tweak map parameters for visualization
model.vis.map.force = 0.1 # scaling parameter for force vector's length
model.vis.map.torque = 0.1 # scaling parameter for control torque

# tweak scales of contact visualization elements
model.vis.scale.contactwidth = 0.05 # width of the floor contact point
model.vis.scale.contactheight = 0.01 # height of the floor contact point
model.vis.scale.forcewidth = 0.03 # width of the force vector
model.vis.scale.com = 0.2 # com radius

# tweaking colors of stuff
model.vis.rgba.contactforce = np.array([0.7, 0., 0., 0.5], dtype=np.float32)
model.vis.rgba.force = np.array([0., 0.7, 0., 0.5], dtype=np.float32)
model.vis.rgba.joint = np.array([0.7, 0.7, 0.7, 0.5])
model.vis.rgba.actuatorpositive = np.array([0., 0.9, 0., 0.5])
model.vis.rgba.actuatornegative = np.array([0.9, 0., 0., 0.5])
model.vis.rgba.com = np.array([1.,0.647,0.,0.5])

scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

### IDK WHAT THIS STUFF DOES EXACTLY, IT IS AN ARTIFACT FROM
### THE MUJOCO TUTORIAL I USED TO START THIS SCRIPT

# install GLFW mouse and keyboard callbacks
# glfw.set_key_callback(window, keyboard)
# glfw.set_cursor_pos_callback(window, mouse_move)
# glfw.set_mouse_button_callback(window, mouse_button)
# glfw.set_scroll_callback(window, scroll)

#############################

#set initial conditions
data.qpos[0]= 0 #5*np.pi/180
# data.qpos[1]=0
# data.qpos[2]=-np.pi/6
# data.qpos[3]=-np.pi/6

# Set camera configuration
cam.azimuth = 90.0
cam.distance = 5.0
cam.elevation = -5
cam.lookat = np.array([0.012768, -0.000000, 1.254336])

# Use torque control mode
actuator_type = "torque"
start = time.time()

control_flag = True
if control_flag:
    # PASS CONTROLLER METHOD
    # TO MUJOCO THING THAT DOES CONTROL... DONT FULLY
    # UNDERSTAND WHAT IS GOING ON HERE BUT IT DOES SEEM
    # TO WORK.
    mj.set_mjcb_control(controller)
else:
    # use prerecorded torque values if this is the case
    torque_csv_file_path = "csv_files\recorded_torques.csv"
    recorded_torques = np.loadtxt(torque_csv_file_path, delimiter=',')
    # print(recorded_torques)

# parameter that sets length of impulse
impulse_time = 0.5

# precallocating Queues and arrays for data sharing and data collection
perturbation_queue = Queue()
control_log_queue = Queue()
control_log_array = np.empty((1,2))
counter_queue = Queue()
joint_position_data = np.empty((1,2))
joint_velocity_data = np.empty((1,2))
body_com_data = np.empty((1,4))
body_orientation_data = np.empty((1,9))

# ID number for the ankle joint, used for data collection
ankle_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "ankle_y_right")

# ID number for the body geometry
human_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "shin_right")

# start the thread that generates the perturbation impulses
perturbation_thread = threading.Thread \
    (target=generate_large_impulse, 
     daemon=True, 
     args=(perturbation_queue,impulse_time) )
perturbation_thread.start()

# Define the video file parameters
video_file = 'pose_trajectory.mp4'
video_fps = 60  # Frames per second
frames = [] # list to store frames
desired_viewport_height = 912

recorded_control_counter = 0

while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0/60.0):
        if not control_flag:
            data.ctrl[0] = recorded_torques[recorded_control_counter,1]
            recorded_control_counter+=1
        mj.mj_step(model, data)
    
        if not control_log_queue.empty():
            control_log_array = np.vstack((control_log_array, control_log_queue.get()))
        
        joint_position_data = np.vstack((joint_position_data, np.array([data.time, data.qpos[ankle_joint_id]]) ))
        joint_velocity_data = np.vstack((joint_velocity_data, np.array([data.time, data.qvel[ankle_joint_id]]) ))
        com = data.xipos[human_body_id]
        orientation_vector = data.ximat[human_body_id]
        body_com_data = np.vstack((body_com_data, np.array([data.time, com[0], com[1], com[2]]) ))
        # orientation_matrix_3d = np.reshape()
        # print(orientation_vector)
        orientation_matrix = np.reshape(orientation_vector, (3,3))
        # print(orientation_matrix)
        body_orientation_data = np.vstack((body_orientation_data, orientation_vector))

    if (data.time>=simend):
        break;

    # get framebuffer viewport
    viewport_width, viewport_height = glfw.get_framebuffer_size(
        window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update scene and render
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # Capture the frame
    # rgb_array = np.empty((viewport_height, viewport_width, 3), dtype=np.uint8)
    # depth_array = np.empty((viewport_height, viewport_width), dtype=np.float32)

    # mj.mjr_readPixels(rgb=rgb_array, depth=depth_array, viewport=viewport, con=context)
    
    # rgb_array = np.flipud(rgb_array)

    # # # Append the frame to the list
    # frames.append(rgb_array) 

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

# imageio.mimwrite(video_file, frames, fps=video_fps)

glfw.terminate()

plot_3d_pose_trajectory(body_com_data, body_orientation_data)
# if control_flag:
#     torque_csv_file_path = "recorded_torques.csv"
#     np.savetxt(torque_csv_file_path, control_log_array[1:,:], delimiter=",")
#     plot_columns(control_log_array, 'Control Torque')
# else:
#     plot_columns(recorded_torques, 'Control Torque')

# plot_columns(joint_position_data, "Joint Position")
# plot_columns(joint_velocity_data, "Joint Velocity")

