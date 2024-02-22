import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
from numpy.linalg import inv
from scipy.linalg import solve_continuous_are
import os
import xml.etree.ElementTree as ET
# xml_path = 'doublependulum.xml'
simend = 5

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

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
        
        error = data.sensordata[0] - ankle_position_setpoint 
        # GRAVITY COMPENSATION #
        human_torque = -K_p * error

        # exo_torque = -1*(human_torque)
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
    # control_torque_time_array = np.array([data.time, human_torque])
    # control_log_queue.put(control_torque_time_array)

    # counter = 0
    # if not counter_queue.empty():
    #     counter = counter_queue.get()
    # counter += 1
    # counter_queue.put(counter)

    # x_perturbation=0
    
    # if not perturbation_queue.empty():
    #     # print(f"perturbation: {perturbation_queue.get()}, time: {time.time()-start}")
    #     x_perturbation = perturbation_queue.get()
    #     perturbation_datalogger_queue.put(x_perturbation)
    
    # data.xfrc_applied[i] = [ F_x, F_y, F_z, R_x, R_y, R_z]
    # data.xfrc_applied[1] = [x_perturbation, 0, 0, 0., 0., 0.]
    

# def controller(model, data):
#     """
#     This function implements a LQR controller for balancing.
#     """
#     state = np.array([
#         [data.qpos[0]],
#         [data.qvel[0]],
#         [data.qpos[1]],
#         [data.qvel[1]],
#     ])
#     data.ctrl[0] = (K @ state)[0, 0]

#     # Apply noise to shoulder
#     noise = mj.mju_standardNormal(0.0)
#     data.qfrc_applied[0] = noise

#get the full path
xml_path = "xml_files\leg copy.xml"

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

            geom.set('fromto', f'0 0 0 0 0 {H_total}')
            # geom.set('pos', f'0 0 {H_total-l_COM}')
            geom.set('mass', str(m_body))

        elif geom.get('name') == "foot1_right":

            geom.set('fromto', f'0 .02 0 {l_foot} .02 0')
            geom.set('mass', str(m_feet))

for body in root.iter('body'):
        if body.get('name') == "foot":
             body.set('pos',  f'0 0 0.035')

        elif body.get('name') == "shin_body":
            body.set('pos', f'{a} 0 0.')

for joint in root.iter('joint'):
        if joint.get('name') == "ankle_hinge":
            joint.set("pos", f"{a} 0 0")

        elif joint.get('name') == "rotation_dof":
            joint.set('pos', f'0 0 {H_total*0.575}')

        elif joint.get('name') == "joint_slide_x":
            joint.set('pos', f"0 0 {H_total*0.575}")

        elif joint.get('name') == "joint_slide_z":
            joint.set('pos', f"0 0 {H_total*0.575}")
            

tree.write('xml_files\modified_model.xml')

modified_xml_path = 'xml_files\modified_model.xml'
script_directory = os.path.dirname(os.path.abspath(__file__))
xml_path = os.path.join(script_directory, xml_path)
model = mj.MjModel.from_xml_path(modified_xml_path)  # MuJoCo XML model
# MuJoCo data structures
# model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
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
opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = True
# opt.flags[mj.mjtVisFlag.mjVIS_ACTUATOR] = True
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

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

#set the camera
cam.azimuth = 40
cam.elevation = -20  
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 0])

data.qpos[3] = 5*np.pi/180
ankle_position_setpoint = 5*np.pi/180  #5*np.pi/180
actuator_type = "torque"
#set the controller
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    simstart = data.time

    while (data.time - simstart < 1.0/60.0):
        mj.mj_step(model, data)
        # data.ctrl[0] = -100

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

    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)

    # process pending GUI events, call GLFW callbacks
    glfw.poll_events()

glfw.terminate()
