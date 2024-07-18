# %%
import time
import mujoco
import mujoco.viewer
import numpy as np
import math


xml_path2 = "scene_right_cylinder.xml"

# Load model
model = mujoco.MjModel.from_xml_path(xml_path2)
data = mujoco.MjData(model)
mujoco.mj_kinematics(model, data)
renderer = mujoco.Renderer(model)


error = 0.001
hand_initial_pos = data.body('palm').xpos
target_initial_pos = data.body('cylinder_object').xpos
# Calculate the distance to the target
distance_to_target =  target_initial_pos - hand_initial_pos
distance_to_open = 1


mujoco.mj_resetData(model, data)

# Init translational and rotational component of Jacobian
ff_contact_jacp = np.zeros((3, model.nv))  # position part of geometric Jacobian
ff_contact_jacr = np.zeros((3, model.nv))  # rotation part of geometric Jacobian
mf_contact_jacp = np.zeros((3, model.nv))  # position part of geometric Jacobian
mf_contact_jacr = np.zeros((3, model.nv))  # rotation part of geometric Jacobian
rf_contact_jacp = np.zeros((3, model.nv))  # position part of geometric Jacobian
rf_contact_jacr = np.zeros((3, model.nv))  # rotation part of geometric Jacobian
th_contact_jacp = np.zeros((3, model.nv))  # position part of geometric Jacobian
th_contact_jacr = np.zeros((3, model.nv))  # rotation part of geometric Jacobian


jacp = np.zeros((3, model.nv))  # position part of geometric Jacobian
jacr = np.zeros((3, model.nv))  # rotation part of geometric Jacobian
#Set target positions based on contact points for each finger

radius = 0.05
beta = 7
sin_beta = math.sin(beta * math.pi / 180)
cos_beta = math.cos(beta * math.pi / 180)
ff_contact_target = np.asarray([ -radius * sin_beta, -radius * cos_beta,  0.15]) + np.array([1 ,1, 0]) * 0.0002
mf_contact_target = np.asarray([ -radius * sin_beta, -radius * cos_beta,  0.10]) + np.array([1, 1, 0]) * 0.0002
rf_contact_target = np.asarray([ -radius * sin_beta, -radius * cos_beta,  0.05]) + np.array([1, 1, 0]) * 0.0002
th_contact_target = np.asarray([  radius * sin_beta,  radius * cos_beta,  0.115]) - np.array([1, 1, 0]) * 0.0002
 
# radius = 0.02
# beta = 0
# sin_beta = math.sin(beta * math.pi / 180)
# cos_beta = math.cos(beta * math.pi / 180)
# ff_contact_target = np.asarray([ -radius * sin_beta, -radius * cos_beta,  0.15]) 
# mf_contact_target = np.asarray([ -radius * sin_beta, -radius * cos_beta,  0.10]) 
# rf_contact_target = np.asarray([ -radius * sin_beta, -radius * cos_beta,  0.05]) 
# th_contact_target = np.asarray([  radius * sin_beta,  radius * cos_beta,  0.115]) 

#Set target positions based on contact points for each finger



# Get indexes for each finger and object so we can compute Jacobian matrix for each of them
ff_tip_idx = model.body('ff_tip').id
mf_tip_idx = model.body('mf_tip').id
rf_tip_idx = model.body('rf_tip').id
th_tip_idx = model.body('th_tip').id
cylinder_object_idx = model.body('cylinder_object').id

# model.opt.timestep = 0.002
# Function that calculates the desired joint positions to move the hand towards the target.

class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_error = 0
        self.prev_error = 0

    def compute_control_signals(self, error):
        # Proportional term
        p = self.kp * error

        # Integral term
        self.integral_error += error
        i = self.ki * self.integral_error

        # Derivative term
        derivative_error = error - self.prev_error
        d = self.kd * derivative_error

        # Calculate the control signals
        control_signals = p + i + d

        # Update previous error
        self.prev_error = error

        return control_signals
    


# open_pid = PIDController(4,0.08,0.01)
# initial_pid = PIDController(1,0.001,0.01)
# ff_pid = PIDController(1.8, 0.18, 0.01)
# mf_pid = PIDController(1.8, 0.2, 0.01)
# rf_pid = PIDController(1.8, 0.18, 0.01)
# th_pid = PIDController(2, 0.1, 0.01)
# target_pid = PIDController(0.02,0.0005,0.0001)

open_pid = PIDController(4,0.08,0.01)
initial_pid = PIDController(1,0.001,0.01)
ff_pid = PIDController(1.8, 0.18, 0.01)
mf_pid = PIDController(1.8, 0.2, 0.01)
rf_pid = PIDController(1.8, 0.18, 0.01)
th_pid = PIDController(0.5, 0.1, 0.01)
target_pid = PIDController(0.02,0.0005,0.0001)

# model flags
open_flag = False
init_flag = True
grasp_flag = False
lift_flag = False
ff_flag = False
mf_flag = False
rf_flag = False
th_flag = False
error_fingers = 0.00008

print(data.contact)



with mujoco.viewer.launch_passive(model, data) as viewer:
    step_start = time.time()
    last_print_time = step_start
    start = step_start

    while viewer.is_running() and time.time() - start < 1000:
        step_start = time.time()
        current_time = time.time()


 
        if current_time - last_print_time >= 1:
          print("time=", current_time)
          print("ff_contact", data.site('ff_contact').xpos)
          print("mf_contact", data.site('mf_contact').xpos)
          print("rf_contact", data.site('rf_contact').xpos)
          print("th_contact", data.site('th_contact').xpos)
          
          last_print_time = current_time        

        if open_flag == True:
          distance_to_open = np.array([1.2]) - data.qpos[15]
          data.ctrl[15] = open_pid.compute_control_signals(distance_to_open)
          if distance_to_open < 0.005:
             open_flag = False
             init_flag = True

        if init_flag == True:
 
          hand_initial_pos = data.body('palm').xpos
          target_initial_pos = data.body('cylinder_object').xpos
          # Calculate the distance to the target
          distance_to_target =  target_initial_pos - hand_initial_pos + [-0.102,0.01,0]

          data.ctrl[0:3] = initial_pid.compute_control_signals(distance_to_target)
          if np.sum(distance_to_target**2) < error:
             init_flag = False
             grasp_flag = True


        if grasp_flag == True:
          
          


          ff_contact_current_pos = data.site('ff_contact').xpos
          mf_contact_current_pos = data.site('mf_contact').xpos
          rf_contact_current_pos = data.site('rf_contact').xpos
          th_contact_current_pos = data.site('th_contact').xpos

          # Calculate the distance to the target
          ff_contact_distance_to_target =  ff_contact_target - ff_contact_current_pos
          mf_contact_distance_to_target =  mf_contact_target - mf_contact_current_pos
          rf_contact_distance_to_target =  rf_contact_target - rf_contact_current_pos
          th_contact_distance_to_target =  th_contact_target - th_contact_current_pos
          if np.sum(ff_contact_distance_to_target**2) < error_fingers:
            ff_flag = True
            
          if np.sum(mf_contact_distance_to_target**2) < error_fingers:
            mf_flag = True
            
          if np.sum(rf_contact_distance_to_target**2) < error_fingers:
            rf_flag = True
            
          if np.sum(th_contact_distance_to_target**2) < error_fingers:
            th_flag = True
            
          if ff_flag == True and mf_flag == True and rf_flag == True and th_flag == True:
            grasp_flag = False
            lift_flag = True
            target_final_pos = data.body("palm").xpos[2] + 0.1
            hand_initial_z = data.body('palm').xpos[2]

            # target_final_pos = np.asarray([0.0, 0.0, 0.5])
            distance_to_final_target =  target_initial_pos - hand_initial_pos
            


          ff_contact_control_signals = ff_pid.compute_control_signals(ff_contact_distance_to_target)
          mf_contact_control_signals = mf_pid.compute_control_signals(mf_contact_distance_to_target)
          rf_contact_control_signals = rf_pid.compute_control_signals(rf_contact_distance_to_target)
          th_contact_control_signals = th_pid.compute_control_signals(th_contact_distance_to_target)
          # th_contact_control_signals = compute_control_signals(0.9,0.03,0.01,th_contact_distance_to_target)
 

          mujoco.mj_jac(model, data, ff_contact_jacp, ff_contact_jacr, data.site('ff_contact').xpos, ff_tip_idx)
          mujoco.mj_jac(model, data, mf_contact_jacp, mf_contact_jacr, data.site('mf_contact').xpos, mf_tip_idx)
          mujoco.mj_jac(model, data, rf_contact_jacp, rf_contact_jacr, data.site('rf_contact').xpos, rf_tip_idx)
          mujoco.mj_jac(model, data, th_contact_jacp, th_contact_jacr, data.site('th_contact').xpos, th_tip_idx)
 
          # Reshape Jacobians to 3D matrices
          ff_contact_jacp = ff_contact_jacp.reshape((3, model.nv))
          mf_contact_jacp = mf_contact_jacp.reshape((3, model.nv))
          rf_contact_jacp = rf_contact_jacp.reshape((3, model.nv))
          th_contact_jacp = th_contact_jacp.reshape((3, model.nv))
  
          jacobi_all = np.vstack([ff_contact_jacp, mf_contact_jacp, rf_contact_jacp, th_contact_jacp])
          #jacobi_pinv_all = np.linalg.pinv()
          control_signal_all = np.vstack([ff_contact_control_signals, mf_contact_control_signals, rf_contact_control_signals, th_contact_control_signals])
  


          J_pinv_ff = np.linalg.pinv(ff_contact_jacp)
          J_pinv_mf = np.linalg.pinv(mf_contact_jacp)
          J_pinv_rf = np.linalg.pinv(rf_contact_jacp)  
          J_pinv_th = np.linalg.pinv(th_contact_jacp)
          H_q_ff = np.zeros((model.nv, 1))
          H_q_mf = np.zeros((model.nv, 1))
          H_q_rf = np.zeros((model.nv, 1))
          H_q_th = np.zeros((model.nv, 1))

          alpha_ff = 0.3
          alpha_mf = 0.3
          alpha_rf = 0.3
          alpha_th = 0.2

          H_q_ff[3:7] = data.qpos[3:7].reshape(4, 1) - np.array([[0], [1.], [1.5], [1.5]])
          H_q_mf[7:11] = data.qpos[7:11].reshape(4, 1) - np.array([[0], [1], [1.5], [1.5]])
          H_q_rf[11:15] = data.qpos[11:15].reshape(4, 1) - np.array([[0], [1], [1.5], [1.5]])
          H_q_th[15:19] = data.qpos[15:19].reshape(4, 1) - np.array([[1.4], [0], [0.8], [0.8]])
          # H_q_ff[3:7] = data.qpos[3:7].reshape(4, 1) - np.array([[0], [1.], [1.8], [1.8]])
          # H_q_mf[7:11] = data.qpos[7:11].reshape(4, 1) - np.array([[0], [1], [1.8], [1.8]])
          # H_q_rf[11:15] = data.qpos[11:15].reshape(4, 1) - np.array([[0], [1], [1.8], [1.8]])
          # H_q_th[15:19] = data.qpos[15:19].reshape(4, 1) - np.array([[1.4], [0], [1], [1.8]])


          ff_contact_control_signals_joint_space = J_pinv_ff @ ff_contact_control_signals - (alpha_ff * (np.eye(model.nv) - J_pinv_ff @ ff_contact_jacp) @ H_q_ff).reshape(25,)
          mf_contact_control_signals_joint_space = J_pinv_mf @ mf_contact_control_signals - (alpha_mf * (np.eye(model.nv) - J_pinv_mf @ mf_contact_jacp) @ H_q_mf).reshape(25,) 
          rf_contact_control_signals_joint_space = J_pinv_rf @ rf_contact_control_signals - (alpha_rf * (np.eye(model.nv) - J_pinv_rf @ ff_contact_jacp) @ H_q_rf).reshape(25,) 
          th_contact_control_signals_joint_space = J_pinv_th @ th_contact_control_signals - (alpha_th * (np.eye(model.nv) - J_pinv_th @ th_contact_jacp) @ H_q_th).reshape(25,)  

          # control signals
          control_signals_joint_space = np.concatenate((ff_contact_control_signals_joint_space[3:7], mf_contact_control_signals_joint_space[7:11],  rf_contact_control_signals_joint_space[11:15], th_contact_control_signals_joint_space[15:19]))

          data.ctrl[3:] = control_signals_joint_space



        
        
        if lift_flag == True:
    
          
          hand_initial_z = data.body('palm').xpos[2]
          # Calculate the distance to the target
          distance_to_final_target =  target_final_pos - hand_initial_z
          
          data.ctrl[2] = target_pid.compute_control_signals(distance_to_final_target)
          if distance_to_final_target < 0.0001:
             lift_flag = False
        
          
        mujoco.mj_step(model, data)
        renderer.update_scene(data)
        
        with viewer.lock():
          viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1

        # Pick up changes to the physics state, apply perturbations, update options from GUI.
        viewer.sync()


        time_until_next_step = model.opt.timestep- (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
            
