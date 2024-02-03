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
simend = 10
K_p = 0

def generate_large_impulse(perturbation_queue):
    while True:
        # Wait for a random amount of time (e.g., between 5 and 10 seconds)
        wait_time = np.random.uniform(2, 3)
        time.sleep(wait_time)

        # Generate a large impulse
        perturbation = np.random.uniform(30, 50)

        # Put the generated impulse into the result queue
        perturbation_queue.put(perturbation)
        # print(perturbation)

def controller(model, data):
    """
    This function implements a PD controller

    Since there are no gravity compensation,
    it will not be very accurate at tracking
    the set point. It will be accurate if
    gravity is turned off.
    """

    if actuator_type == "torque":
        # model.actuator_gainprm[1, 0] = 1
        # print(model.actuator_gainprm)
        # model.actuator_gainprm[0, 0] = 1 
        # print(str(data.sensordata[0]) + ", " + str(data.sensordata[1]))
        # print()

        # GRAVITY COMPENSATION #
        data.ctrl[0] = K_p * \
            (data.sensordata[0] - 5*np.pi/180 )
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
    
    x_perturbation=np.random.normal(loc=50, scale=2.5)
    # if not perturbation_queue.empty():
    #     x_perturbation = perturbation_queue.get()
    
    # print(x_perturbation)
    # z_perburbation = np.random.normal(loc=12.5, scale=2.5)
    # Apply the body COM pertubations in Cartersian Space
        
    # data.xfrc_applied[i] = [ F_x, F_y, F_z, R_x, R_y, R_z]
    data.xfrc_applied[1] = [0, 0, 0, 0., 0., 0.]

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
# calculate_kp_and_geom(weight, height)
M_total = 80 # kg
H_total = 1.78 # meters
m_feet, m_body, l_COM, l_foot, a, K_p = calculate_kp_and_geom( \
                                                M_total, H_total)
print(m_feet, m_body, l_COM, l_foot, a, K_p)

for geom in root.iter('geom'):
        if geom.get('name') == "shin_geom":
            # Modify the mass attribute of the geom
            geom.set('fromto', f'0 0 0 0 0 {-H_total}')
            # geom.set('pos', f'0 0 {H_total-l_COM}')
            geom.set('mass', str(m_body))
            
        elif geom.get('name') == "foot1_right":
            geom.set('fromto', f'0 .02 0 {l_foot} .02 0')
            geom.set('mass', str(m_feet))
            # parent_body = geom.getparent()
            # parent_body.set('pos', f'-{a} 0 -0.32')

for body in root.iter('body'):
        if body.get('name') == "le_foot":
             body.set('pos',  f'-{a} 0 -{H_total+0.02}')

tree.write('modified_model.xml')

#get the full path
modified_xml_path = 'modified_model.xml'
script_directory = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_directory, xml_path)
model = mj.MjModel.from_xml_path(modified_xml_path)  # MuJoCo model
# model = mj.MjData.from_xml_string(xml_path)

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
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
# glfw.set_key_callback(window, keyboard)
# glfw.set_cursor_pos_callback(window, mouse_move)
# glfw.set_mouse_button_callback(window, mouse_button)
# glfw.set_scroll_callback(window, scroll)

#set initial conditions
data.qpos[0]=5*np.pi/180
# data.qpos[1]=0
# data.qpos[2]=-np.pi/6
# data.qpos[3]=-np.pi/6
# print(data.qpos)
# Set camera configuration
cam.azimuth = 90.0
cam.distance = 5.0
cam.elevation = -5
cam.lookat = np.array([0.012768, -0.000000, 1.254336])

# perturbation_force = [0.0, 0.0, 1.0]  # Modify as needed
# mj.MjData.xfrc_applied['shin_right'] = perturbation_force




actuator_type = "torque"
mj.set_mjcb_control(controller)

perturbation_queue = Queue()

perturbation_thread = threading.Thread \
    (target=generate_large_impulse, 
     daemon=True, 
     args=(perturbation_queue,) )
perturbation_thread.start()


# Define the video file parameters
# video_file = 'output_video.mp4'
# video_fps = 60  # Frames per second
# frames = [] # list to store frames

while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0/60.0):
        mj.mj_step(model, data)
        # print(data.efc_J)

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

    # # Capture the frame
    # mj.mjr_readPixels(rgb=np.array([0,0,0]),depth=np.array([1]), \
    #                   viewport=viewport, con=context)
    
    # # Convert the frame to uint8
    # frame = (frame * 255).astype(np.uint8)

    # # Append the frame to the list
    # frames.append(frame)

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

# video_writer.close()

glfw.terminate()