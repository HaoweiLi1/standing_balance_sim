import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from typing import Callable, Optional, Union, List
import scipy.linalg
import mediapy as media
import matplotlib.pyplot as plt
import threading
import time
from queue import Queue
import xml.etree.ElementTree as ET
import imageio

xml_path = 'leg.xml'
simend = 5
K_p = 0

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

        wait_time = np.random.uniform(2, 3)
        time.sleep(wait_time)
        
        # Generate a large impulse
        perturbation = np.random.uniform(350, 358)
        direction_bool = np.random.choice([-1,1])
        # print(f"perturbation: {perturbation}")
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
            (data.sensordata[0] - 5*np.pi/180 )
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
    # data.xfrc_applied[1] = [x_perturbation, 0, 0, 0., 0., 0.]

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

tree.write('modified_model.xml')
########

#get the full path
modified_xml_path = 'modified_model.xml'
script_directory = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_directory, xml_path)
model = mj.MjModel.from_xml_path(modified_xml_path)  # MuJoCo model
# model = mj.MjData.from_xml_string(xml_path)

data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# visualize contact frames and forces, make body transparent
# presently this code isn't working :,)
mj.mjv_defaultOption(opt)
opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = True
opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = True
opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = True
# tweak scales of contact visualization elements
model.vis.scale.contactwidth = 0.1
model.vis.scale.contactheight = 0.03
model.vis.scale.forcewidth = 0.05
model.vis.map.force = 0.3

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
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
data.qpos[0]=5*np.pi/180
# data.qpos[1]=0
# data.qpos[2]=-np.pi/6
# data.qpos[3]=-np.pi/6

# Set camera configuration
cam.azimuth = 90.0
cam.distance = 5.0
cam.elevation = -5
cam.lookat = np.array([0.012768, -0.000000, 1.254336])

# SET CONTROL MODE AND PASS CONTROLLER METHOD
# TO MUJOCO THING THAT DOES CONTROL... DONT FULLY
# UNDERSTAND WHAT IS GOING ON HERE BUT IT DOES SEEM
# TO WORK.
actuator_type = "torque"
start = time.time()

control_flag = False
if control_flag:
    mj.set_mjcb_control(controller)
else:
    # use prerecorded torque values if this is the case
    torque_csv_file_path = "recorded_torques.csv"
    recorded_torques = np.loadtxt(torque_csv_file_path, delimiter=',')
    # print(recorded_torques)

perturbation_queue = Queue()
control_log_queue = Queue()
control_log_array = np.empty((1,2))
counter_queue = Queue()

impulse_time = 0.25

joint_position_data = np.empty((1,2))
joint_velocity_data = np.empty((1,2))
ankle_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "ankle_y_right")

perturbation_thread = threading.Thread \
    (target=generate_large_impulse, 
     daemon=True, 
     args=(perturbation_queue,impulse_time) )
perturbation_thread.start()


# Define the video file parameters
video_file = 'output_video.mp4'
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

# print(counter_queue.get())

# print(control_log_array)

if control_flag:
    torque_csv_file_path = "recorded_torques.csv"
    np.savetxt(torque_csv_file_path, control_log_array[1:,:], delimiter=",")
    plot_columns(control_log_array, 'Control Torque')
else:
    plot_columns(recorded_torques, 'Control Torque')

plot_columns(joint_position_data, "Joint Position")
plot_columns(joint_velocity_data, "Joint Velocity")

