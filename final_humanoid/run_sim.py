import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
from typing import Callable, Optional, Union, List
import scipy.linalg as linalg
import mediapy as media
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import threading
import csv
import time
from queue import Queue
import xml.etree.ElementTree as ET
import imageio
import yaml
from plotting_utilities import plot_3d_pose_trajectory, \
                               plot_columns, \
                               plot_four_columns, \
                               plot_two_columns, \
                               plot_three_columns

from xml_utilities import calculate_kp_and_geom, set_geometry_params
from controllers import (HumanLQRController, HumanPDController, 
                       ExoPDController, ExoGravityCompensation)


plt.rcParams['text.usetex'] = True
# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': 'sans-serif',  # You can choose 'serif', 'sans-serif', or other font families
#     'font.sans-serif': 'Helvetica',  # Specify the font you want to use
#     'axes.labelsize': 12,
#     'font.size': 12,
#     'legend.fontsize': 10,
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     })

mpl.rcParams.update(mpl.rcParamsDefault)

perturbation_queue = Queue()
control_log_queue = Queue()
counter_queue = Queue()
perturbation_datalogger_queue = Queue()

class ankleTorqueControl:

    def __init__(self, plot_flag):
        print('simulation class object created')
        self.plot_flag = plot_flag
        self.impulse_threaad_exit_flag = True
        self.mp4_flag = False
        self.K_p = 1
        self.pertcount = 0
        # self.prev_error = 0  # 存储上一个时间步的误差
        # self.Kp_exo = 10
        # self.Kd_exo = 1
        self.human_controller = None
        self.exo_controller = None

    def initialize_controllers(self, model, data, config):
        """Initialize controllers based on configuration."""
        # Get human controller configuration
        human_config = config['controllers']['human']
        if human_config['type'] == "LQR":
            lqr_params = human_config['lqr_params']
            Q = np.diag([lqr_params['Q_angle'], lqr_params['Q_velocity']])
            R = np.array([[lqr_params['R']]])
            
            self.human_controller = HumanLQRController(
                model=model,
                data=data,
                max_torque_df=human_config['max_torque_df'],
                max_torque_pf=human_config['max_torque_pf'],
                mass=config['M_total'],
                leg_length=0.575 * config['H_total'],
                Q=Q,
                R=R
            )
        elif human_config['type'] == "PD":
            pd_params = human_config['pd_params']
            self.human_controller = HumanPDController(
                model=model,
                data=data,
                max_torque_df=human_config['max_torque_df'],
                max_torque_pf=human_config['max_torque_pf'],
                kp=pd_params['kp'],
                kd=pd_params['kd'],
                mrtd_df=human_config['mrtd_df'],
                mrtd_pf=human_config['mrtd_pf']
            )
            
        # Get exo controller configuration
        exo_config = config['controllers']['exo']
        if exo_config['type'] == "PD":
            pd_params = exo_config['pd_params']
            self.exo_controller = ExoPDController(
                model=model,
                data=data,
                max_torque=exo_config['max_torque'],
                kp=pd_params['kp'],
                kd=pd_params['kd']
            )
        elif exo_config['type'] == "GC":
            gc_params = exo_config['gc_params']
            self.exo_controller = ExoGravityCompensation(
                model=model,
                data=data,
                max_torque=exo_config['max_torque'],
                compensation_factor=gc_params['compensation_factor']
            )
    def controller(self, model, data):
        """Controller function for the leg."""
        # Get current state
        state = np.array([
            data.sensordata[0],  # Joint angle
            data.qvel[3]         # Joint velocity
        ])
        
        # Compute human control
        human_torque = self.human_controller.compute_control(
            state=state,
            target=self.ankle_position_setpoint
        )
        data.ctrl[0] = human_torque
        
        # Compute exo control
        exo_torque = self.exo_controller.compute_control(
            state=state,
            target=self.ankle_position_setpoint
        )
        data.ctrl[1] = exo_torque

    # # this function has to stay in this script per MuJoCo documentation
    # def controller(self, model, data):
    #     """
    #     Controller function for the leg. The only two parameters 
    #     for this function can be mujoco.model and mujoco.data
    #     otherwise it doesn't seem to run properly
    #     """
        
    #     error = self.ankle_position_setpoint - data.sensordata[0]
    #     # # # GRAVITY COMPENSATION #
    #     # human_torque = self.K_p * error 
    #     # data.ctrl[0] = human_torque
    #     # print(data.ctrl[0])
    #     # # PD #
    #     # Kp =800
    #     # Kd =10
    #     # delta_error = error - self.prev_error
    #     # delta_time = model.opt.timestep  # MuJoCo 仿真步长
    #     # derivative = delta_error / delta_time  # 计算微分项
    #     # human_torque = Kp * error + Kd * derivative
    #     # data.ctrl[0] = human_torque
    #     # self.prev_error = error

    #     if not hasattr(self, 'K'):
    #         # 系统参数
    #         m = 80.0 - 2 * 0.0145 * 80.0  # 总质量(kg)
    #         l = 0.575 * 1.78   # 到COM的距离(m)
    #         g = 9.81  # 重力加速度(m/s^2)
    #         b = 2.5   # 阻尼系数
    #         I = m * l**2  # 转动惯量
            
    #         # 状态空间矩阵
    #         # A = np.array([[0, 1],
    #         #             [g/l, -b/(m*l**2)]])
    #         # B = np.array([[0],
    #         #             [1/(m*l**2)]])
    #         A = np.array([[0, 1],
    #                     [-m*g*l/I, -b/I]])
    #         B = np.array([[0],
    #                     [1/I]])

            
    #         # LQR权重矩阵
    #         Q = np.diag([300, 100])  # 状态权重：角度误差权重大，角速度误差权重小
    #         R = np.array([[0.1]])   # 控制输入权重
            
    #         # 求解连续时间黎卡提方程
    #         P = linalg.solve_continuous_are(A, B, Q, R)
            
    #         # 计算LQR增益矩阵
    #         # self.K = np.dot(np.linalg.inv(R), np.dot(B.T, P))
    #         self.K = np.linalg.solve(R, B.T @ P)

            
    #         # 为了后续计算初始化上一时刻状态
    #         self.prev_state = np.array([0.0, 0.0])
            
    #         print("LQR controller initialized with gains:", self.K)
        
    #     # 获取当前状态 [theta, theta_dot]
    #     current_state = np.array([
    #         data.sensordata[0],  # 踝关节角度
    #         data.qvel[3]         # 踝关节角速度
    #     ])
        
    #     # 计算误差状态
    #     error_state = np.array([
    #         current_state[0] - self.ankle_position_setpoint,  # 角度误差
    #         current_state[1]                                  # 角速度
    #     ])
        
    #     # 计算LQR控制输入
    #     human_torque = -np.dot(self.K, error_state)
    #     human_torque = float(human_torque[0])
        
    #     # 更新控制输入
    #     data.ctrl[0] = human_torque
        
    #     # 存储当前状态用于下一时刻计算
    #     self.prev_state = current_state


    #     # ################## test MRTD ##########
    #     # PD 控制计算 human torque
    #     # Kp = 500
    #     # Kd = 20
    #     # delta_error = error - self.prev_error
    #     # delta_time = model.opt.timestep
    #     # derivative = delta_error / delta_time
    #     # desired_torque = Kp * error + Kd * derivative

    #     # # 确保 prev_torque 初始化
    #     # if not hasattr(self, 'prev_torque'):
    #     #     self.prev_torque = 0.0
    #     #     self.prev_error = 0.0

    #     # prev_torque = self.prev_torque

    #     # # MRTD 约束（单位 Nm/s）
    #     # MRTD_df = 148/15  # Dorsiflexion（背屈）
    #     # MRTD_pf = 389/15  # Plantarflexion（跖屈）

    #     # # 计算 torque 变化量
    #     # delta_torque = desired_torque - prev_torque

    #     # # 施加 MRTD 约束
    #     # if delta_torque > 0:
    #     #     max_increase = MRTD_df * delta_time
    #     #     delta_torque = min(delta_torque, max_increase)
    #     # else:
    #     #     max_decrease = -MRTD_pf * delta_time
    #     #     delta_torque = max(delta_torque, max_decrease)

    #     # # 更新 human torque
    #     # human_torque = prev_torque + delta_torque

    #     # # 存储数据
    #     # self.prev_torque = human_torque
    #     # self.prev_error = error

    #     # # 施加到 MuJoCo 控制
    #     # data.ctrl[0] = human_torque

    #     ##############################################
        

    #     # k_exo = 0.5  # 读取 config 里的补偿系数
    #     # exo_torque = - k_exo * data.qfrc_bias[3]  # 计算外骨骼力矩
    #     # data.ctrl[1] = exo_torque

    #     # 计算外骨骼力矩
    #     # 计算外骨骼的 PD 控制
    #     # 获取当前踝关节角度和角速度
    #     current_angle = data.qpos[3]   # 踝关节当前角度
    #     current_velocity = data.qvel[3]  # 踝关节当前角速度

    #     # PD 控制计算
    #     exo_error = self.ankle_position_setpoint - current_angle  # 角度误差
    #     exo_torque = self.Kp_exo * exo_error - self.Kd_exo * current_velocity

    #     # 施加外骨骼力矩
    #     # data.ctrl[1] = exo_torque
    #     data.ctrl[1] = 0
    #     # print(data.ctrl[1])
            
    #     # print("Timestep:", data.time)
    #     # print("Control torque:", data.ctrl[0]*15 + data.ctrl[1]*5)  # 控制器指定的总力矩
    #     # print("Actual joint torque:", data.qfrc_actuator[3])        # 实际执行的关节力矩
    #     # print("Passive damping:", data.qfrc_passive[3])             # 被动阻尼力矩
    #     # print("Constraint force:", data.qfrc_constraint[3])         # 约束力产生的力矩
    #     # # print("Contact force:", data.qfrc_contact[3])               # 接触力产生的力矩
    #     # Log the actuator torque and timestep to an array for later use
    #     # control_torque_time_array = np.array([data.time, human_torque])
    #     # control_log_queue.put(control_torque_time_array)

    #     # x_perturbation=0

    #     # if not perturbation_queue.empty():
    #     #     # print(f"perturbation: {perturbation_queue.get()}, time: {time.time()-start}")
    #     #     x_perturbation = perturbation_queue.get()
    #     #     perturbation_datalogger_queue.put(x_perturbation)
        
    #     # # data.xfrc_applied[i] = [ F_x, F_y, F_z, R_x, R_y, R_z]
    #     # data.xfrc_applied[2] = [x_perturbation, 0, 0, 0., 0., 0.]

        
    #     # if x_perturbation < -100:
    #     #     self.pertcount+=1
        
    #     # if x_perturbation == 0:
    #     #     # print(f'x_perturbation: {x_perturbation}; percounter: {self.pertcount}')
    #     #     self.pertcount=0
    #     # print(f"perturbation: {x_perturbation}")
    #     # print(f'frc data: {data.cfrc_ext}')
    #     # counter = 0
    #     # if not counter_queue.empty():
    #     #     counter = counter_queue.get()
    #     # counter += 1
    #     # counter_queue.put(counter)
        
    #     # Apply joint perturbations in Joint Space
    #     # data.qfrc_applied = [ q_1, q_2, q_3, q_4, q_5, q_6, q_7, q_8]
    #     # data.qfrc_applied[3] = rx_perturbation
    #     # data.qfrc_applied[4] = ry_perturbation

    # this function is a thread that uses imulse_thread_exit_flag (global variable)
    # not sure if there is a way to move this function to a utility script without 
    # preventing function from accessing impulse_thread_exit_flag
    def generate_large_impulse(self, perturbation_queue, impulse_time, perturbation_magnitude, impulse_period):
    
        while not self.impulse_thread_exit_flag: # run continuously once we start a thread 
                    # for this function
            
            # empty the queue from the previous impulse generation
            while not perturbation_queue.empty():
                perturbation_queue.get()

            wait_time = np.random.uniform(impulse_period,impulse_period+1)
            time.sleep(wait_time)
            
            # Generate a large impulse
            perturbation = np.random.uniform(perturbation_magnitude, perturbation_magnitude)

            # this will generate either -1 or 1
            direction_bool = np.random.choice([-1,1])
            # print(direction_bool)
            # start_time = time.time()
            # delta = time.time() - start_time
            start_time = time.time()
            end_time = start_time + impulse_time
            # print(time.time())
            # print(end_time)
            i=1
            while time.time() < end_time:
                # delta = time.time() - start_time
                # time.sleep(0.0001)
                
                # pert__force_data = np.vstack((com_ext_force_data, np.array([data.time,data.cfrc_ext[1,:]])))
                # print('')
                # time.sleep(0.0000000001)
                # print('impulse thread')
                print(f"impulse duration: {(impulse_time+end_time-time.time())}")
                # Put the generated impulse into the result queue
                # time.sleep(0.001)
                i+=1
                perturbation_queue.put(direction_bool*perturbation)
            # perturbation_queue = Queue.Queue()
            # with perturbation_queue.mutex:
            #     perturbation_queue.queue.clear()
            # print(f'pert counter: {i}')

    def load_params_from_yaml(self, file_path):
        with open(file_path, 'r') as file:
            params = yaml.safe_load(file)
        return params

    def run(self):

        # precallocating arrays and queues for data sharing and data collection
        human_torque_data = np.empty((1,2))
        exo_torque_data = np.empty((1,2))
        ankle_torque_data = np.empty((1,2))
        control_log_array = np.empty((1,2))
        goal_position_data = np.empty((1,2))
        joint_position_data = np.empty((1,2))
        joint_velocity_data = np.empty((1,2))
        body_com_data = np.empty((1,4))
        body_orientation_data = np.empty((1,9))
        perturbation_data_array = np.empty((1,2))
        com_ext_force_data = np.empty((1,7))
        constraint_frc_data = np.empty((1,5))
        contact_force_sensor = np.empty((1,3))

        gravity_torque_data = np.empty((1,2))  # 修改这里
        
        params = self.load_params_from_yaml('config.yaml')

        self.plot_flag = params['config']['plotter_flag']
        self.mp4_flag = params['config']['mp4_flag']
        
        self.ankle_position_setpoint = params['config']['ankle_position_setpoint_radians']
        ankle_joint_initial_position = params['config']['ankle_initial_position_radians'] # ankle joint initial angle
        
        # Define parameters for creating and storing an MP4 file of the rendered simulation
        video_file = params['config']['mp4_file_name']
        video_fps = params['config']['mp4_fps']
        frames = [] # list to store frames
        desired_viewport_height = 912

        translation_friction_constant = params['config']['translation_friction_constant']
        rolling_friction_constant = params['config']['rolling_friction_constant']

        perturbation_time = params['config']['perturbation_time']
        perturbation_magnitude = params['config']['perturbation_magnitude']   # size of impulse
        perturbation_period = params['config']['perturbation_period']  # period at which impulse occurs

        xml_path = params['config']['xml_path']
        simend = params['config']['simend']   # duration of simulation
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # MODIFY THE XML FILE AND INSERT THE MASS AND LENGTH
        # PROPERTIES OF INTEREST 
        M_total = params['config']['M_total'] # kilograms
        H_total = params['config']['H_total'] # meters
        h_f, m_feet, m_body, l_COM, l_foot, a, self.K_p = calculate_kp_and_geom \
                                                (M_total, H_total)
        set_geometry_params(root, 
                            m_feet, 
                            m_body, 
                            l_COM, 
                            l_foot, 
                            a, 
                            H_total, 
                            h_f, 
                            translation_friction_constant, 
                            rolling_friction_constant) # call utility functoin to set parameters of xml model

        literature_model = params['config']['lit_xml_file']
        tree.write(literature_model)
        #######        
        model = mj.MjModel.from_xml_path(literature_model)  # MuJoCo XML model

        data = mj.MjData(model)                # MuJoCo data
        cam = mj.MjvCamera()                        # Abstract camera
        opt = mj.MjvOption()                        # visualization options

        # Init GLFW, create window, make OpenGL context current, request v-sync
        glfw.init()
        window = glfw.create_window(1200, 912, "Demo", None, None) # Modify the video size to solve the warning
        glfw.make_context_current(window)
        glfw.swap_interval(1)

        # initialize visualization data structures
        mj.mjv_defaultCamera(cam)
        mj.mjv_defaultOption(opt)
        # opt.flags[mj.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        opt.flags[mj.mjtVisFlag.mjVIS_CONTACTFORCE] = params['config']['visualize_contact_force']
        opt.flags[mj.mjtVisFlag.mjVIS_PERTFORCE] = params['config']['visualize_perturbation_force']
        opt.flags[mj.mjtVisFlag.mjVIS_JOINT] = params['config']['visualize_joints']
        opt.flags[mj.mjtVisFlag.mjVIS_ACTUATOR] = params['config']['visualize_actuators']
        opt.flags[mj.mjtVisFlag.mjVIS_COM] = params['config']['visualize_center_of_mass']
        # opt.flags[mj.mjtFrame.mjFRAME_GEOM] = True
        # opt.flags[mj.mjtLabel.mjLABEL_JOINT] = True
        # opt.flags[mj.mjtFrame.mjFRAME_GEOM] = True
        # opt.flags[mj.mjtFrame.mjFRAME_WORLD] = True 
        # opt.flags[mj.mjtFrame.mjFRAME_CONTACT] = True
        # opt.flags[mj.mjtFrame.mjFRAME_BODY] = True
        # opt.mjVIS_COM = True
        # model.opt.gravity=np.array([0, 0, params['config']['gravity']])
        model.opt.timestep = params['config']['simulation_timestep']

        # gravity_string = params['config']['camera_lookat_xyz']
        # grav_tuple = tuple(map(float, gravity_string.split(', ')))
        model.opt.gravity = np.array([0, 0, params['config']['gravity']])
        # tweak map parameters for visualization
        model.vis.map.force = 0.25 # scaling parameter for force vector's length
        model.vis.map.torque = 0.1 # scaling parameter for control torque

        # tweak scales of contact visualization elements
        model.vis.scale.contactwidth = 0.05 # width of the floor contact point
        model.vis.scale.contactheight = 0.01 # height of the floor contact point
        model.vis.scale.forcewidth = 0.03 # width of the force vector


        model.vis.scale.com = 0.2 # com radius
        model.vis.scale.actuatorwidth = 0.1 # diameter of visualized actuator
        model.vis.scale.actuatorlength = 0.1 # thickness of visualized actuator
        model.vis.scale.jointwidth = 0.025 # diameter of joint arrows
        model.vis.scale.framelength = 0.25
        model.vis.scale.framewidth = 0.05

        # tweaking colors of stuff, attribute names are pretty intuitive
        model.vis.rgba.contactforce = np.array([0.7, 0., 0., 0.5], dtype=np.float32)
        model.vis.rgba.force = np.array([0., 0.7, 0., 0.5], dtype=np.float32)
        model.vis.rgba.joint = np.array([0.2, 1, 0.1, 0.8])
        model.vis.rgba.actuatorpositive = np.array([0., 0.9, 0., 0.5]) ## color when actuator is exerting positive force
        model.vis.rgba.actuatornegative = np.array([0.9, 0., 0., 0.5]) ## color when actuator is exerting negative force
        model.vis.rgba.com = np.array([1.,0.647,0.,0.5]) # center of mass color

        scene = mj.MjvScene(model, maxgeom=10000)
        context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # Set camera configuration
        cam.azimuth = params['config']['camera_azimuth']
        cam.distance = params['config']['camera_distance']
        cam.elevation = params['config']['camera_elevation']

        lookat_string_xyz = params['config']['camera_lookat_xyz']
        print(f'lookat string: {lookat_string_xyz}')
        # print(lookat_string_xyz)
        # do some stuff to take a comma separated string and convert it to a tuple
        res = tuple(map(float, lookat_string_xyz.split(', ')))
        # print(f'res: {res}')
        # get the x, y, z camera coords from the res tuple
        cam.lookat = np.array([res[0], res[1], res[2]])

        # INITIAL CONDITIONS FOR JOINT VELOCITIES
        data.qvel[0]= 0              # hinge joint at top of body
        data.qvel[1]= 0              # slide / prismatic joint at top of body in x direction
        data.qvel[2]= 0              # slide / prismatic joint at top of body in z direction
        data.qvel[3]= 0              # hinge joint at ankle

        # INITIAL CONDITIONS FOR JOINT POSITION
        data.qpos[0]=params['config']['foot_angle_initial_position_radians'] # hinge joint at top of body
        # data.qpos[1]=0          # slide / prismatic joint at top of body in x direction
        # data.qpos[2]=0   # slide / prismatic joint at top of body in z direction
        data.qpos[3]= ankle_joint_initial_position # hinge joint at ankle

        # ID number for the ankle joint, used for data collection
        ankle_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "ankle_hinge")
        # print(ankle_joint_id)
        # ID number for the body geometry
        human_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "long_link_body")


        # human_actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "ankle_actuator")
        # exo_actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "exo_ankle_actuator")
        # print(human_actuator_id)
        # print(exo_actuator_id)

        recorded_control_counter = 0

        params = self.load_params_from_yaml('config.yaml')
        
        # Initialize controllers
        self.initialize_controllers(model, data, params['config'])
        

        control_flag = params['config']['controller_flag']
        if control_flag:
            # PASS CONTROLLER METHOD
            # TO MUJOCO THING THAT DOES CONTROL... DONT FULLY
            # UNDERSTAND WHAT IS GOING ON HERE BUT IT DOES SEEM
            # TO WORK.
            mj.set_mjcb_control(self.controller)
            pass
        # else:
        #     # use prerecorded torque values if this is the case
        #     torque_csv_file_path = os.path.join(script_directory, "recorded_torques_test.csv")
        #     recorded_torques = np.loadtxt(torque_csv_file_path, delimiter=',')
            # print(recorded_torques)

        # start the thread that generates the perturbation impulses
        self.impulse_thread_exit_flag = False
        perturbation_thread = threading.Thread \
            (target=self.generate_large_impulse, 
            daemon=True, 
            args=(perturbation_queue,perturbation_time, perturbation_magnitude, perturbation_period) )
        if params['config']['apply_perturbation']:
            print('perturbation thread started')
            perturbation_thread.start()


        start_time = time.time()
        print(f'simulation duration: {simend} seconds')

        
        while not glfw.window_should_close(window): # glfw.window_should_close() indicates whether or not the user has closed the window
            simstart = data.time
            # print(data.time)
            
            while (data.time - simstart < 1.0/60.0):
                # print(data.qpos[0])
                # print(f'time delta: {time.time() - start_time}')
                # print(f'foot angle: {data.qpos[0]}')

                # if data.qpos[0] > np.pi/4 or data.qpos[0] < -np.pi/4:
                #     simend = data.time 
                    # if params['config']['apply_perturbation']:
                    #     print('perturbation thread terminated')
                    #     perturbation_thread.join()
                    
                #print(f'sensor data: {data.sensordata[0]}')
                # if we aren't using the PD control mode and instead using prerecorded data, then
                # we should set the control input to the nth value of the recorded torque array
                # if not control_flag:
                #     data.ctrl[0] = recorded_torques[recorded_control_counter,1]
                #print(f"simulation time: {data.time}")
                # step the simulation forward in time
                # print('running')
                if not perturbation_queue.empty():
                    # if not perturbation_queue.empty():
                    # print(f"perturbation: {perturbation_queue.get()}, time: {time.time()-start}")
            
                    x_perturbation = perturbation_queue.get()
                    # print(x_perturbation)
                    # perturbation_datalogger_queue.put(x_perturbation)
        
                    # data.xfrc_applied[i] = [ F_x, F_y, F_z, R_x, R_y, R_z]
                    data.xfrc_applied[2] = [x_perturbation, 0, 0, 0., 0., 0.]
                    perturbation_data_array = np.vstack((perturbation_data_array, np.array([data.time, x_perturbation]) ))
                else:
                    # if the perturbation queue is empty then the perturbation is zero
                    # print(time.time())
                    data.xfrc_applied[2] = [0, 0, 0, 0., 0., 0.]
                    perturbation_data_array = np.vstack((perturbation_data_array, np.array([data.time, 0.]) ))

                mj.mj_step(model, data)
                # input_torque = data.ctrl[0]   # 期望控制输入
                # actuator_torque = data.qfrc_actuator[3]  # 执行器的实际力矩
                # print(f"Time: {data.time:.3f}, Input Torque: {input_torque:.4f}, Actuator Torque: {actuator_torque:.4f}")
                gravity_torque = data.qfrc_bias[3] 
                gravity_torque_data = np.vstack((gravity_torque_data, np.array([data.time, gravity_torque])))  # 修改这里
                # collect data from the control log if it isn't empty, this is used in the PD control mode
                # print(f'actuator force: {data.qfrc_actuator[3]}')
                control_log_array = np.vstack((control_log_array, np.array([data.time,data.qfrc_actuator[3]]) ))
                
                # collect data from perturbation for now perturbation is 1D so we have (n,2) array of pert. data
                # print("Timestep:", data.time)
                # print("human torque:", data.ctrl[0]*15)  # 控制器指定的总力矩
                # print("Actual joint torque:", data.qfrc_actuator[3])        # 实际执行的关节力矩
                # print("Passive damping:", data.qfrc_passive[3])             # 被动阻尼力矩
                # print("Constraint force:", data.qfrc_constraint[3])         # 约束力产生的力矩
                    
                # human_torque_executed = 15 * data.ctrl[0]
                # exo_torque_executed = 5 * data.ctrl[1]
                human_actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "human_ankle_actuator")
                exo_actuator_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "exo_ankle_actuator")
                human_torque_executed = data.actuator_force[human_actuator_id] * 15
                exo_torque_executed = data.actuator_force[exo_actuator_id] * 10
                ankle_torque_executed = data.qfrc_actuator[ankle_joint_id]

                human_torque_data = np.vstack((human_torque_data, np.array([data.time, human_torque_executed])))
                exo_torque_data = np.vstack((exo_torque_data, np.array([data.time, exo_torque_executed])))
                ankle_torque_data = np.vstack((ankle_torque_data, np.array([data.time, ankle_torque_executed])))

                # collect joint position and velocity data from simulation for visualization
                joint_position_data = np.vstack((joint_position_data, np.array([data.time, 180/np.pi*data.qpos[ankle_joint_id]]) ))
                joint_velocity_data = np.vstack((joint_velocity_data, np.array([data.time, 180/np.pi*data.qvel[ankle_joint_id]]) ))
                goal_position_data = np.vstack((goal_position_data, np.array([data.time,180/np.pi*self.ankle_position_setpoint]) ))
                # print(f'constraint force: {data.qfrc_constraint}')
                # print(np.array([data.time,data.qfrc_constraint[:]]))
                # print(np.array([data.time,data.qfrc_constraint[0],data.qfrc_constraint[1],data.qfrc_constraint[2],data.qfrc_constraint[3]]))
                constraint_frc_data = np.vstack((constraint_frc_data, np.array([data.time,data.qfrc_constraint[0],data.qfrc_constraint[1],data.qfrc_constraint[2],data.qfrc_constraint[3]])))
                contact_force_sensor = np.vstack((contact_force_sensor, np.array([data.time, data.sensordata[1], data.sensordata[2]]) ))
                # print(f'frc data: {data.cfrc_int[0,:]}, {data.cfrc_int[1,:]}, {data.cfrc_int[2,:]}')
                # pert__force_data = np.vstack((com_ext_force_data, np.array([data.time,data.cfrc_ext[1,:]])))
                
                # collect center of mass data for ID associated with human body element of XML model
                com = data.xipos[human_body_id]
                # collect 1x9 orientation vector for ID associated with human body element of XML model
                # format is [r_11, r_12, r_13, r_21, r_22, r_23, r_31, r_32, r_33]
                orientation_vector = data.ximat[human_body_id]
                # add center of mass data to array
                body_com_data = np.vstack((body_com_data, np.array([data.time, com[0], com[1], com[2]]) ))
                # add orientation data to array
                # orientation_matrix = np.reshape(orientation_vector, (3,3))
                body_orientation_data = np.vstack((body_orientation_data, orientation_vector))
            
            if (data.time>=simend):
                
                self.impulse_thread_exit_flag = True
                if params['config']['apply_perturbation']:
                    perturbation_thread.join()
                    print('perturbation thread terminated')
                break;
            
            # get framebuffer viewport
            viewport_width, viewport_height = glfw.get_framebuffer_size(
                window)
            viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

            # Update scene and render
            mj.mjv_updateScene(model, data, opt, None, cam,
                            mj.mjtCatBit.mjCAT_ALL.value, scene)
            mj.mjr_render(viewport, scene, context)


            ###### CODE TO CAPTURE FRAMES FOR MP4 VIDEO GENERATION ##########
            # Capture the frame
            rgb_array = np.empty((viewport_height, viewport_width, 3), dtype=np.uint8)
            depth_array = np.empty((viewport_height, viewport_width), dtype=np.float32)
            mj.mjr_readPixels(rgb=rgb_array, depth=depth_array, viewport=viewport, con=context)
            rgb_array = np.flipud(rgb_array)
            # # Append the frame to the list
            frames.append(rgb_array) 
            #################################################################

            # swap OpenGL buffers (blocking call due to v-sync)
            glfw.swap_buffers(window)

            # process pending GUI events, call GLFW callbacks
            glfw.poll_events()

        # this writes the list of frames we collected to an mp4 file and makes it a video
        if self.mp4_flag:
            imageio.mimwrite(video_file, frames, fps=video_fps)
        print('terminated')
        glfw.terminate()
        np.savetxt('joint_position_data', joint_position_data, delimiter=",", fmt='%.3f')
        np.savetxt('joint_velocity_data', joint_velocity_data, delimiter=",", fmt='%.3f')
        np.savetxt('constraint_frc_data', constraint_frc_data, delimiter=',', fmt='%.3f')
        np.savetxt('perturbation_data', perturbation_data_array, delimiter=",",fmt="%.3f")
        np.savetxt('contact_force_sensor', contact_force_sensor, delimiter=',', fmt="%.3f")
        np.savetxt('gravity_torque_data', gravity_torque_data, delimiter=",", fmt='%.3f')
        np.savetxt('control_torque', control_log_array, delimiter=",", fmt='%.3f')
        np.savetxt('human_torque_data.csv', human_torque_data, delimiter=",", fmt='%.3f')
        np.savetxt('exo_torque_data.csv', exo_torque_data, delimiter=",", fmt='%.3f')
        # plot_columns(contact_force_sensor[:,0:2], "front contact force versus time")
        # plot_two_columns(contact_force_sensor[:,0:2], contact_force_sensor[:,[0,2]], "Front Contact Force", "Back Contact Force")
        ##### PLOTTING CODE ######################
        ##########################################
        if self.plot_flag:
            # plot_3d_pose_trajectory(body_com_data, body_orientation_data)
            # if control_flag:
                # torque_csv_file_path = os.path.join(script_directory, "recorded_torques_test.csv")
                # np.savetxt(torque_csv_file_path, control_log_array[1:,:], delimiter=",")
            # plot_columns(control_log_array, '$\\bf{Control\;Torque, \\it{\\tau_{ankle}}}$')
            # # else:
            # #     plot_columns(recorded_torques, 'Control Torque')
            np.savetxt('joint_position_data', joint_position_data, delimiter=",", fmt='%.3f')
            np.savetxt('joint_velocity_data', joint_velocity_data, delimiter=",", fmt='%.3f')
            np.savetxt('com_ext_force_data', com_ext_force_data, delimiter=",", fmt='%.3f')
            np.savetxt('perturbation_data', perturbation_data_array, delimiter="'",fmt="%.3f")
            # plot_columns(perturbation_data_array, "perturbation versus time")
            plot_two_columns(joint_position_data, goal_position_data, "Actual Position", "Goal Position")
            plot_columns(joint_velocity_data, "Joint Velocity")
            # plot_four_columns(joint_position_data, 
            #                 goal_position_data, 
            #                 joint_velocity_data, 
            #                 control_log_array,
            #                 "joint actual pos.",
            #                 "joint goal pos.",
            #                 "joint vel.",
            #                 "control torque")
            # plot_two_columns(control_log_array, gravity_torque_data, "Control Torque [Nm]", "Gravitational Torque [Nm]")
            # plot_columns(gravity_torque_data, "Gravitational Torque [Nm]")
            # plot_columns(exo_torque_data, "Exo Torque (Executed) [Nm]")
            # plot_two_columns(human_torque_data, exo_torque_data, "Human Torque (Executed) [Nm]", "Exo Torque (Executed) [Nm]")
            plot_three_columns( human_torque_data, 
                                exo_torque_data, 
                                ankle_torque_data, 
                                "Human Torque (Executed) [Nm]", 
                                "Exo Torque (Executed) [Nm]", 
                                "Ankle Torque (Executed) [Nm]")
            # plot_four_columns (human_torque_data, 
            #                     exo_torque_data, 
            #                     ankle_torque_data,
            #                     gravity_torque_data, 
            #                     "Human Torque (Executed) [Nm]", 
            #                     "Exo Torque (Executed) [Nm]", 
            #                     "Ankle Torque (Executed) [Nm]",
            #                     "Gravitational Torque [Nm]") 
        ###########################################
        ###########################################

if __name__ == "__main__":

    sim_class = ankleTorqueControl(True)
    sim_class.run()
 