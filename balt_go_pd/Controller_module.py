#!/usr/bin/env python

import sys
sys.path.append('/home/user/work/ros2_ws/src/balt_go_pd/balt_go_pd')

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy, QoSPresetProfiles

from geometry_msgs.msg import PoseStamped
from px4_msgs.msg import *
from std_msgs.msg import Int32
from Controller_lib import Controller

import linecache
import ast
import math
import time
from cvxopt import matrix, solvers
from scipy.spatial.transform import Rotation as R
from transforms3d.quaternions import quat2mat, mat2quat, qmult, qinverse
from transforms3d.euler import euler2mat, euler2quat


class Controller_module(Node):

    def __init__(self):
        super().__init__('Controller_module')

        # QoS Profile
        qos_profile = QoSProfile(
            history=QoSPresetProfiles.SENSOR_DATA.value.history,
            depth=5,
            reliability=QoSPresetProfiles.SENSOR_DATA.value.reliability,
            durability=QoSPresetProfiles.SENSOR_DATA.value.durability
            )

        # Define subscribers
        self.status_subscriber_           = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)
        
        
        self.vehicle_odometry_subscriber_ = self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.vehicle_odometry_callback,
            qos_profile)
        
        self.command_pose_subscriber_     = self.create_subscription(
            PoseStamped,
            '/command/pose', 
            self.command_pose_callback,
            10)
               
        # Define publishers
        self.offboard_mode_publisher_   = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode',
            10)

        self.actuator_motors_publisher_ = self.create_publisher(
            ActuatorMotors,
            '/fmu/in/actuator_motors',  
            10)
        
        self.motor_failure_publisher_ = self.create_publisher(
            Int32, '/motor_failure/motor_number', 10)        
        
        # Initialize 
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)
        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        
        # Initialize odometry
        self.time_temp = np.uint64(0)
        self.pos_odo = np.zeros(3, dtype=np.float32)
        self.vel_odo = np.zeros(3, dtype=np.float32)
        self.quat_odo = np.zeros(4, dtype=np.float32)
        self.angvel_odo = np.zeros(3, dtype=np.float32)

        # Initialize reference command
        self.pos_cmd = np.zeros(3, dtype=np.float32)
        self.ori_cmd = np.zeros(4, dtype=np.float32)

        # Initialize control output
        self.throttles = np.zeros(4)

        # Inintialize UAV parameters
        self.zero_position_armed = 100
        self.input_scaling = 1000
        self.thrust_constant = 5.84e-06
        self.moment_constant = 0.06       
        self.arm_length = 0.25
        self._inertia_matrix = np.array([0.029125, 0.029125, 0.055125])  # Inertia matrix
        self._gravity = 9.81  # Gravity
        self._uav_mass = 1.725  # UAV mass

        # Initialize matrices
        self.torques_and_thrust_to_rotor_velocities_ = np.zeros((4, 4))
        self.throttles_to_normalized_torques_and_thrust_ = np.zeros((4, 4))

        # Attack parameters
        self.callback_counter                   =   int(0)
        self.motors_to_fail                     =   int(0)
        self.attack_on_off                      =   int(0)

    def cmdloop_callback(self):
        # Publish offboard control modes
        self.offboard_mode_publish_(pos_cont=False, vel_cont=False, acc_cont=False, att_cont=False, act_cont=True)
        
        wrench = np.zeros(4)
        desired_quat = np.zeros(4)
        normalized_torque_thrust = np.zeros(4)
        throttles = np.zeros(4)
        pos_odo1 = np.zeros(3)
        vel_odo1 = np.zeros(3)
        quat_odo1 = np.zeros(4)
        angvel_odo1 = np.zeros(3)

        controller_ = Controller()

        # Initialize UAV parameters with placeholder values
        controller_.set_uav_parameters(self._uav_mass, self._inertia_matrix, self._gravity)
        controller_.set_control_gains(np.array([7.0, 7.0, 6.0]), np.array([6.0, 6.0, 3.0]), np.array([3.5, 3.5, 0.3]), np.array([0.5, 0.5, 0.1]))
        
        pos_odo1, vel_odo1, quat_odo1, angvel_odo1 = self.eigen_odometry_from_PX4_msg(self.pos_odo, self.vel_odo, self.quat_odo, self.angvel_odo)

        # Set odometry and trajectory point
        controller_.set_odometry(pos_odo1, vel_odo1, quat_odo1, angvel_odo1)
        controller_.set_trajectory_point(self.pos_cmd, self.ori_cmd)   

        # Calculate controller output
        wrench, desired_quat = controller_.calculate_controller_output()
        normalized_torque_thrust, throttles = self.px4InverseSITL(wrench)

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            current_time = int(Clock().now().nanoseconds / 1000)

            self.callback_counter += 1

            if self.callback_counter == 901:
                self.motors_to_fail = int(1)                    # Randomly select one motors from 1 to 4
                self.attack_on_off = 1
                self.publish_motor_failure(self.motors_to_fail)

            elif self.callback_counter == 950:
                self.publish_motor_failure(0)
                self.callback_counter = 0
                self.attack_on_off = 0

            # Calculate controller output
            self.actuator_motors_publish_(throttles)
            # self.attitude_setpoint_publish_(wrench[3]/1000, desired_quat)


        # self.get_logger().info(f'time: {self.time_temp}, \n \
        #                          Odometry Position: {self.pos_odo}, \n Quaternion: {self.quat_odo}, \n Velocity: {self.vel_odo}, \n \
        #                          Angular Velocity: {self.angvel_odo}, \n Desired Position: {self.pos_cmd}, \n Desired Orientation: {self.ori_cmd}, \n \
        #                          Navigation State: {self.nav_state}, \n \
        #                          Desired Quaternion: {controller_.r_R_B_W}, \n Desired Yaw: {controller_.r_yaw}')
    
    def compute_ControlAllocation_and_ActuatorEffect_matrices(self):
        kDegToRad = np.pi / 180.0
        kS = np.sin(45 * kDegToRad)
        rotor_velocities_to_torques_and_thrust = np.zeros((4, 4))
        rotor_velocities_to_torques_and_thrust = np.array([
                [-kS, kS, kS, -kS],
                [-kS, kS, -kS, kS],
                [-1, -1, 1, 1],
                [1, 1, 1, 1]
                ])
        # mixing_matrix = np.array([
        #         [-0.495384, -0.707107, -0.765306, 1.0],
        #         [0.495384, 0.707107, -1.0, 1.0],
        #         [0.495384, -0.707107, 0.765306, 1.0],
        #         [-0.495384, 0.707107, 1.0, 1.0]
        #         ])
        
        ## Hardcoded because the calculation of pesudo-inverse is not accurate
        # self.throttles_to_normalized_torques_and_thrust_ = np.array([
        #         [-0.5718, 0.4376, 0.5718, -0.4376],
        #         [-0.3536, 0.3536, -0.3536, 0.3536],
        #         [-0.2832, -0.2832, 0.2832, 0.2832],
        #         [0.2500, 0.2500, 0.2500, 0.2500]
        #         ])

        # Calculate Control allocation matrix: Wrench to Rotational velocities / k: helper matrix
        k = np.array([self.thrust_constant * self.arm_length,
                      self.thrust_constant * self.arm_length,
                      self.moment_constant * self.thrust_constant,
                      self.thrust_constant])
        
        # Element-wise multiplication
        rotor_velocities_to_torques_and_thrust = np.diag(k) @ rotor_velocities_to_torques_and_thrust
        self.torques_and_thrust_to_rotor_velocities_ = np.linalg.pinv(rotor_velocities_to_torques_and_thrust)

    def px4InverseSITL(self, wrench):
        # Initialize vectors
        omega = np.zeros(4)
        throttles = np.zeros(4)
        normalized_torque_and_thrust = np.zeros(4)
        ones_temp = np.ones(4)
        
        array = np.array([
                [-242159.856570736, -242159.856570736, -713470.319634703, 42808.2191780822],
                [242159.856570736, 242159.856570736, -713470.319634703, 42808.2191780822],
                [242159.856570736, -242159.856570736, 713470.319634703, 42808.2191780822],
                [-242159.856570736, 242159.856570736, 713470.319634703, 42808.2191780822]
                ])

        quat_odo_euler = self.rotate_quaternion_from_to_ENU_NED(self.quat_odo)


        # Control allocation: Wrench to Rotational velocities (omega)
        omega = array @ wrench
        omega = np.sqrt(np.abs(omega))  # Element-wise square root, handle negative values
        
        # CBF
        indv_forces = omega * np.abs(omega) * self.thrust_constant
        # print("indv_forces")
        # print(indv_forces)

        if self.attack_on_off == 0:
            u_safe = self.control_barrier_function_Noattack(indv_forces)
        else:
            u_safe = self.control_barrier_function_Attack(indv_forces)
        
        u_safe = np.array([u_safe[0], u_safe[1], u_safe[2], u_safe[3]], dtype=np.float32)
        # print("u_safe")
        # print(u_safe)

        # Calculate throttles
        omega = np.sqrt(np.abs(u_safe)/self.thrust_constant)   
    
        # Calculate hrottles from omega (rotor velocities)
        throttles = (omega - (self.zero_position_armed * ones_temp))
        throttles = throttles / self.input_scaling

        # print("throttles")
        # print(throttles)
        
        # Inverse Mixing: throttles to normalized torques and thrust
        normalized_torque_and_thrust = self.throttles_to_normalized_torques_and_thrust_ @ throttles

        return normalized_torque_and_thrust, throttles

    # CBF Function (No attack)
    def control_barrier_function_Noattack(self, indv_f):
        x_state = np.zeros(12)
        f_x = np.zeros(12)
        barrier_phi_max = 15 / 180 * np.pi
        barrier_theta_max = 15 / 180 * np.pi
        barrier_zdelta_max = 10
        barrier_vz_max = 10
        u_max = 1.2 * self._uav_mass * self._gravity / 4
        u_min = 0

        rotated_q = self.rotate_quaternion_from_to_ENU_NED(self.quat_odo)
        # rotated_q = qinverse(rotated_q)
        if np.linalg.norm(rotated_q) == 0:
            rotated_q = np.array([0, 0, 0, 1])

        phi, theta, psi = self.euler_from_quaternion(rotated_q[1], rotated_q[2], rotated_q[3], rotated_q[0]) # Expected Error Part
        p, q, r = self.rotate_vector_from_to_FRD_FLU(self.angvel_odo) 
        pos_r = self.rotate_vector_from_to_ENU_NED(self.pos_cmd)
        pos_odo_rotated = self.rotate_vector_from_to_ENU_NED(self.pos_odo).flatten()
        vel_odo_rotated = self.rotate_vector_from_to_ENU_NED(self.vel_odo).flatten()

        cp          =   np.cos(phi)
        ct          =   np.cos(theta)
        cs          =   np.cos(psi)
        sp          =   np.sin(phi)
        st          =   np.sin(theta)
        ss          =   np.sin(psi)
        tt          =   np.tan(theta)
        sect        =   1 / (np.cos(theta) + 10e-6)

        B = np.array([[1, sp*tt, cp*tt], \
                        [0, cp, -sp], \
                        [0, sp/ct, cp/ct]])
        
        Euler_dot = B @ np.array([p, q, r])

        x_state = np.array([pos_odo_rotated[0], pos_odo_rotated[1], pos_odo_rotated[2], \
                            vel_odo_rotated[0], vel_odo_rotated[1], vel_odo_rotated[2], \
                            phi, theta, psi,                      \
                            p, q, r])        
        
        f_x = np.array([vel_odo_rotated[0], vel_odo_rotated[1], vel_odo_rotated[2],   \
                          0, 0, -self._gravity,                             \
                          Euler_dot[0], Euler_dot[1], Euler_dot[2],                                        \
                          q*r*(self._inertia_matrix[1] - self._inertia_matrix[2])/self._inertia_matrix[0], \
                          p*r*(self._inertia_matrix[2] - self._inertia_matrix[0])/self._inertia_matrix[1], \
                          p*q*(self._inertia_matrix[0] - self._inertia_matrix[1])/self._inertia_matrix[2]]) # Expected Error Part (p, q, r)

        G_mat = np.array([[0, 0, 0, 0], \
                            [0, 0, 0, 0], \
                            [0, 0, 0, 0],
                            [self.thrust_constant*(cp*ss*st-cs*sp)/self._uav_mass, self.thrust_constant*(cp*ss*st-cs*sp)/self._uav_mass, self.thrust_constant*(cp*ss*st-cs*sp)/self._uav_mass, self.thrust_constant*(cp*ss*st-cs*sp)/self._uav_mass],
                            [self.thrust_constant*(sp*ss+cp*cs*st)/self._uav_mass, self.thrust_constant*(sp*ss+cp*cs*st)/self._uav_mass, self.thrust_constant*(sp*ss+cp*cs*st)/self._uav_mass, self.thrust_constant*(sp*ss+cp*cs*st)/self._uav_mass], 
                            [self.thrust_constant*(cp*ct)/self._uav_mass, self.thrust_constant*(cp*ct)/self._uav_mass, self.thrust_constant*(cp*ct)/self._uav_mass, self.thrust_constant*(cp*ct)/self._uav_mass], 
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [self.thrust_constant*self.arm_length / (self._inertia_matrix[0]*np.sqrt(2)), -self.thrust_constant*self.arm_length / (self._inertia_matrix[0]*np.sqrt(2)), -self.thrust_constant*self.arm_length / (self._inertia_matrix[0]*np.sqrt(2)), self.thrust_constant*self.arm_length / (self._inertia_matrix[0]*np.sqrt(2))],
                            [-self.thrust_constant*self.arm_length / (self._inertia_matrix[1]*np.sqrt(2)), self.thrust_constant*self.arm_length / (self._inertia_matrix[1]*np.sqrt(2)), -self.thrust_constant*self.arm_length / (self._inertia_matrix[1]*np.sqrt(2)), self.thrust_constant*self.arm_length / (self._inertia_matrix[1]*np.sqrt(2))],
                            [self.moment_constant*self.thrust_constant / self._inertia_matrix[2], self.moment_constant*self.thrust_constant / self._inertia_matrix[2], -self.moment_constant*self.thrust_constant / self._inertia_matrix[2], -self.moment_constant*self.thrust_constant / self._inertia_matrix[2]]])
                             # Expected Error Part (v1. Changed the x, y part)
        
        # Barrier Function (roll) # Expected Error Part 
        Broll           =   (x_state[6]/barrier_phi_max) **2 - 1
        LfBroll         =   2*x_state[6]/(barrier_phi_max**2)*f_x[6]
        etaBroll        =   np.array([1, 1]) @ np.array([Broll, LfBroll])
        Lf2Broll        =   2/(barrier_phi_max**2)*f_x[6]**2 + 2*x_state[6]/(barrier_phi_max**2)*(f_x[9]+f_x[10]*sp*tt+f_x[11]*cp*tt+ \
                            x_state[10]*cp*Euler_dot[0]*tt+x_state[10]*sp*(sect**2)*Euler_dot[1]-x_state[11]*sp*Euler_dot[0]*tt+x_state[11]*cp*(sect**2)*Euler_dot[1])
        LgLfBroll       =   2*x_state[6]/(barrier_phi_max**2)*(G_mat[9,:]+G_mat[10,:]*sp*tt+G_mat[11,:]*cp*tt) # Expected Error Part (Matrix Indexing)

        # Barrier Funciton (pitch) # Expected Error Part 
        Bpitch          =   (x_state[7]/barrier_theta_max) **2 - 1
        LfBpitch        =   2*x_state[7]/(barrier_theta_max**2)*f_x[7]
        etaBpitch       =   np.array([1, 1]) @ np.array([Bpitch, LfBpitch])
        Lf2Bpitch       =   2/(barrier_theta_max**2)*f_x[7]**2 + 2*x_state[7]/(barrier_theta_max**2)*(f_x[10]*cp-f_x[11]*sp-x_state[10]*sp* \
                                                                                                      Euler_dot[0]-x_state[11]*cp*Euler_dot[0])
        LgLfBpitch      =   2*x_state[7]/(barrier_theta_max**2)*(G_mat[10,:]*cp-G_mat[11,:]*sp)

        # Barrier Function (altitude) # Expected Error Part 
        Baltitude       =   (x_state[2]-pos_r[2]) ** 4/(barrier_zdelta_max**4) + x_state[5] ** 4 / (barrier_vz_max**4) - 1 # Expected Error Part 
        LfBaltitude     =   4*(x_state[2]-pos_r[2]) **3/(barrier_zdelta_max**4)*f_x[2] + 4*x_state[5] ** 3 / (barrier_vz_max**4)*f_x[5]
        LgBaltitude     =   4*(x_state[2]-pos_r[2]) **3/(barrier_zdelta_max**4)*G_mat[5,:]

        A_ineq = np.array([
                    [LgLfBroll[0], LgLfBroll[1], LgLfBroll[2], LgLfBroll[3], etaBroll, 0, 0, 1, 0, 0],
                    [LgLfBpitch[0], LgLfBpitch[1], LgLfBpitch[2], LgLfBpitch[3], 0, etaBpitch, 0, 0, 1, 0],
                    [LgBaltitude[0], LgBaltitude[1], LgBaltitude[2], LgBaltitude[3], 0, 0, Baltitude, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [-1,0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0,-1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, -1,0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0,-1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0,-1, 0, 0, 0]
                    ], dtype=np.float64)
        
        B_ineq = np.array([
                    [-Lf2Broll],
                    [-Lf2Bpitch],
                    [-LfBaltitude],
                    [u_max],        
                    [u_min],
                    [u_max],
                    [u_min],
                    [u_max],
                    [u_min],
                    [u_max],
                    [u_min],
                    [-0.01],
                    [-0.01],
                    [-0.01]
                    ], dtype=np.float64)
        
        Q = np.eye(10, dtype=np.float64)
        Q[7,7] = 20
        Q[8,8] = 20
        Q[9,9] = 20

        f = np.array([-indv_f[0], -indv_f[1], -indv_f[2], -indv_f[3], 0, 0, 0, 0, 0, 0], dtype=np.float64)
        
        # Compute the control barrier function
        P = matrix(Q)
        q = matrix(f)
        G = matrix(A_ineq)
        h = matrix(B_ineq)
        initvals = matrix([indv_f[0], indv_f[1], indv_f[2], indv_f[3], 0, 0, 0, 0, 0, 0])

        # Solve QP problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, initvals=initvals)

        u_temp = solution['x']

        return u_temp[0:4]

 # CBF Function (Attack)
    def control_barrier_function_Attack(self, indv_f):
        x_state = np.zeros(12)
        f_x = np.zeros(12)
        barrier_phi_max = 15 / 180 * np.pi
        barrier_theta_max = 15 / 180 * np.pi
        barrier_zdelta_max = 10
        barrier_vz_max = 10
        u_max = 1.2 * self._uav_mass * self._gravity / 4
        u_min = 0

        rotated_q = self.rotate_quaternion_from_to_ENU_NED(self.quat_odo)
        # rotated_q = qinverse(rotated_q)
        if np.linalg.norm(rotated_q) == 0:
            rotated_q = np.array([0, 0, 0, 1])

        phi, theta, psi = self.euler_from_quaternion(rotated_q[1], rotated_q[2], rotated_q[3], rotated_q[0]) # Expected Error Part
        p, q, r = self.rotate_vector_from_to_FRD_FLU(self.angvel_odo) 
        pos_r = self.rotate_vector_from_to_ENU_NED(self.pos_cmd)
        pos_odo_rotated = self.rotate_vector_from_to_ENU_NED(self.pos_odo).flatten()
        vel_odo_rotated = self.rotate_vector_from_to_ENU_NED(self.vel_odo).flatten()

        cp          =   np.cos(phi)
        ct          =   np.cos(theta)
        cs          =   np.cos(psi)
        sp          =   np.sin(phi)
        st          =   np.sin(theta)
        ss          =   np.sin(psi)
        tt          =   np.tan(theta)
        sect        =   1 / (np.cos(theta) + 10e-6)

        B = np.array([[1, sp*tt, cp*tt], \
                        [0, cp, -sp], \
                        [0, sp/ct, cp/ct]])
        
        Euler_dot = B @ np.array([p, q, r])

        x_state = np.array([pos_odo_rotated[0], pos_odo_rotated[1], pos_odo_rotated[2], \
                            vel_odo_rotated[0], vel_odo_rotated[1], vel_odo_rotated[2], \
                            phi, theta, psi,                      \
                            p, q, r])        
        
        f_x = np.array([vel_odo_rotated[0], vel_odo_rotated[1], vel_odo_rotated[2],   \
                          0, 0, -self._gravity,                             \
                          Euler_dot[0], Euler_dot[1], Euler_dot[2],                                        \
                          q*r*(self._inertia_matrix[1] - self._inertia_matrix[2])/self._inertia_matrix[0], \
                          p*r*(self._inertia_matrix[2] - self._inertia_matrix[0])/self._inertia_matrix[1], \
                          p*q*(self._inertia_matrix[0] - self._inertia_matrix[1])/self._inertia_matrix[2]]) # Expected Error Part (p, q, r)

        G_mat = np.array([[0, 0, 0, 0], \
                            [0, 0, 0, 0], \
                            [0, 0, 0, 0],
                            [self.thrust_constant*(cp*ss*st-cs*sp)/self._uav_mass, self.thrust_constant*(cp*ss*st-cs*sp)/self._uav_mass, self.thrust_constant*(cp*ss*st-cs*sp)/self._uav_mass, self.thrust_constant*(cp*ss*st-cs*sp)/self._uav_mass],
                            [self.thrust_constant*(sp*ss+cp*cs*st)/self._uav_mass, self.thrust_constant*(sp*ss+cp*cs*st)/self._uav_mass, self.thrust_constant*(sp*ss+cp*cs*st)/self._uav_mass, self.thrust_constant*(sp*ss+cp*cs*st)/self._uav_mass], 
                            [self.thrust_constant*(cp*ct)/self._uav_mass, self.thrust_constant*(cp*ct)/self._uav_mass, self.thrust_constant*(cp*ct)/self._uav_mass, self.thrust_constant*(cp*ct)/self._uav_mass], 
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [self.thrust_constant*self.arm_length / (self._inertia_matrix[0]*np.sqrt(2)), -self.thrust_constant*self.arm_length / (self._inertia_matrix[0]*np.sqrt(2)), -self.thrust_constant*self.arm_length / (self._inertia_matrix[0]*np.sqrt(2)), self.thrust_constant*self.arm_length / (self._inertia_matrix[0]*np.sqrt(2))],
                            [-self.thrust_constant*self.arm_length / (self._inertia_matrix[1]*np.sqrt(2)), self.thrust_constant*self.arm_length / (self._inertia_matrix[1]*np.sqrt(2)), -self.thrust_constant*self.arm_length / (self._inertia_matrix[1]*np.sqrt(2)), self.thrust_constant*self.arm_length / (self._inertia_matrix[1]*np.sqrt(2))],
                            [self.moment_constant*self.thrust_constant / self._inertia_matrix[2], self.moment_constant*self.thrust_constant / self._inertia_matrix[2], -self.moment_constant*self.thrust_constant / self._inertia_matrix[2], -self.moment_constant*self.thrust_constant / self._inertia_matrix[2]]])
                             # Expected Error Part (v1. Changed the x, y part)
        
        # Barrier Function (roll) # Expected Error Part 
        Broll           =   (x_state[6]/barrier_phi_max) **2 - 1
        LfBroll         =   2*x_state[6]/(barrier_phi_max**2)*f_x[6]
        etaBroll        =   np.array([1, 1]) @ np.array([Broll, LfBroll])
        Lf2Broll        =   2/(barrier_phi_max**2)*f_x[6]**2 + 2*x_state[6]/(barrier_phi_max**2)*(f_x[9]+f_x[10]*sp*tt+f_x[11]*cp*tt+ \
                            x_state[10]*cp*Euler_dot[0]*tt+x_state[10]*sp*(sect**2)*Euler_dot[1]-x_state[11]*sp*Euler_dot[0]*tt+x_state[11]*cp*(sect**2)*Euler_dot[1])
        LgLfBroll       =   2*x_state[6]/(barrier_phi_max**2)*(G_mat[9,:]+G_mat[10,:]*sp*tt+G_mat[11,:]*cp*tt) # Expected Error Part (Matrix Indexing)

        # Barrier Funciton (pitch) # Expected Error Part 
        Bpitch          =   (x_state[7]/barrier_theta_max) **2 - 1
        LfBpitch        =   2*x_state[7]/(barrier_theta_max**2)*f_x[7]
        etaBpitch       =   np.array([1, 1]) @ np.array([Bpitch, LfBpitch])
        Lf2Bpitch       =   2/(barrier_theta_max**2)*f_x[7]**2 + 2*x_state[7]/(barrier_theta_max**2)*(f_x[10]*cp-f_x[11]*sp-x_state[10]*sp* \
                                                                                                      Euler_dot[0]-x_state[11]*cp*Euler_dot[0])
        LgLfBpitch      =   2*x_state[7]/(barrier_theta_max**2)*(G_mat[10,:]*cp-G_mat[11,:]*sp)

        # Barrier Function (altitude) # Expected Error Part 
        Baltitude       =   (x_state[2]-pos_r[2]) ** 4/(barrier_zdelta_max**4) + x_state[5] ** 4 / (barrier_vz_max**4) - 1 # Expected Error Part 
        LfBaltitude     =   4*(x_state[2]-pos_r[2]) **3/(barrier_zdelta_max**4)*f_x[2] + 4*x_state[5] ** 3 / (barrier_vz_max**4)*f_x[5]
        LgBaltitude     =   4*(x_state[2]-pos_r[2]) **3/(barrier_zdelta_max**4)*G_mat[5,:]

        A_ineq = np.array([
                    [LgLfBroll[0], LgLfBroll[1], LgLfBroll[2], 0, etaBroll, 0, 0, 1, 0, 0],
                    [LgLfBpitch[0], LgLfBpitch[1], LgLfBpitch[2], 0, 0, etaBpitch, 0, 0, 1, 0],
                    [LgBaltitude[0], LgBaltitude[1], LgBaltitude[2], 0, 0, 0, Baltitude, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [-1,0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0,-1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0,-1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, -1,0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0,-1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0,-1, 0, 0, 0]
                    ], dtype=np.float64)
        
        B_ineq = np.array([
                    [-Lf2Broll-max(LgLfBroll[3]*u_max, LgLfBroll[3]*u_min)],
                    [-Lf2Bpitch-max(LgLfBpitch[3]*u_max, LgLfBpitch[3]*u_min)],
                    [-LfBaltitude-max(LgBaltitude[3]*u_max, LgBaltitude[3]*u_min)],
                    [u_max],        
                    [u_min],
                    [u_max],
                    [u_min],
                    [u_max],
                    [u_min],
                    [u_max],
                    [u_min],
                    [-0.01],
                    [-0.01],
                    [-0.01]
                    ], dtype=np.float64)
        
        Q = np.eye(10, dtype=np.float64)
        Q[7,7] = 100
        Q[8,8] = 100
        Q[9,9] = 100

        f = np.array([-indv_f[0], -indv_f[1], -indv_f[2], -indv_f[3], 0, 0, 0, 0, 0, 0], dtype=np.float64)
        
        # Compute the control barrier function
        P = matrix(Q)
        q = matrix(f)
        G = matrix(A_ineq)
        h = matrix(B_ineq)
        initvals = matrix([indv_f[0], indv_f[1], indv_f[2], indv_f[3], 0, 0, 0, 0, 0, 0])

        # Solve QP problem
        solvers.options['show_progress'] = False
        solution = solvers.qp(P, q, G, h, initvals=initvals)

        u_temp = solution['x']

        return u_temp[0:4]

    # Function for Sliding Mode Conrol Barrier Function
    def hyper_tangent(self, input_signal, gain=1.0):
        
        return np.tanh(gain * input_signal)

    # Subscribers
    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state

    def command_pose_callback(self, msg):
        # Extract position and orientation from the message
        self.pos_cmd = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        self.ori_cmd = np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w])

    def vehicle_odometry_callback(self, msg):
        self.time_temp  = msg.timestamp
        self.pos_odo = np.array([msg.position[0], msg.position[1], msg.position[2]], dtype=np.float32)
        self.vel_odo = np.array([msg.velocity[0], msg.velocity[1], msg.velocity[2]], dtype=np.float32)
        self.quat_odo = np.array([msg.q[0], msg.q[1], msg.q[2], msg.q[3]], dtype=np.float32)
        self.angvel_odo = np.array([msg.angular_velocity[0], msg.angular_velocity[1], msg.angular_velocity[2]], dtype=np.float32)

    # Helper functions
    def rotate_vector_from_to_ENU_NED(self, vec_in):
        # NED (X North, Y East, Z Down) & ENU (X East, Y North, Z Up)
        vec_out = np.array([vec_in[1], vec_in[0], -vec_in[2]])
        return vec_out

    def rotate_vector_from_to_FRD_FLU(self, vec_in):
        # FRD (X Forward, Y Right, Z Down) & FLU (X Forward, Y Left, Z Up)
        vec_out = np.array([vec_in[0], -vec_in[1], -vec_in[2]])
        return vec_out

    def eigen_odometry_from_PX4_msg(self, pos, vel, quat, ang_vel):
        position_W = self.rotate_vector_from_to_ENU_NED(pos)
        velocity_B = self.rotate_vector_from_to_ENU_NED(vel)
        orientation_B_W = self.rotate_quaternion_from_to_ENU_NED(quat)  # ordering (w, x, y, z)
        
        angular_velocity_B = self.rotate_vector_from_to_FRD_FLU(ang_vel)
    
        return position_W, velocity_B, orientation_B_W, angular_velocity_B

    # Publisher
    def offboard_mode_publish_(self, pos_cont, vel_cont, acc_cont, att_cont, act_cont):
        msg = OffboardControlMode()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.position = bool(pos_cont)
        msg.velocity = bool(vel_cont)
        msg.acceleration = bool(acc_cont)
        msg.attitude = bool(att_cont)
        msg.direct_actuator = bool(act_cont)

        self.offboard_mode_publisher_.publish(msg)

    def actuator_motors_publish_(self, throttles):
        msg = ActuatorMotors()
        msg.control[0] = np.float32(throttles[0])
        msg.control[1] = np.float32(throttles[1])
        msg.control[2] = np.float32(throttles[2])
        msg.control[3] = np.float32(throttles[3])
        msg.control[4] = math.nan
        msg.control[5] = math.nan
        msg.control[6] = math.nan
        msg.control[7] = math.nan
        msg.control[8] = math.nan
        msg.control[9] = math.nan
        msg.control[10] = math.nan
        msg.control[11] = math.nan
        msg.reversible_flags = int(0)
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.timestamp_sample = msg.timestamp

        self.actuator_motors_publisher_.publish(msg)
    
    def attitude_setpoint_publish_(self, thrust, q):
        msg = AttitudeSetpoint()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.q_d = q
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = thrust

        self.attitude_setpoint_publisher_.publish(msg)

    def publish_motor_failure(self, motor_id):
        msg = Int32()
        msg.data = motor_id
        self.motor_failure_publisher_.publish(msg)
        self.get_logger().info(f'Publishing motor failure on motor: {msg.data}')

    def rotate_quaternion_from_to_ENU_NED(self, quat_in):
        # Transform from orientation represented in ROS format to PX4 format and back.
        # quat_in_reordered = [quat_in[1], quat_in[2], quat_in[3], quat_in[0]]
    
        # NED to ENU conversion Euler angles
        euler_1 = np.array([np.pi, 0.0, np.pi/2])
        NED_ENU_Q = euler2quat(euler_1[2], euler_1[1], euler_1[0], 'szyx')
        # print("NED_ENU_Q")
        # print(NED_ENU_Q)

        # Aircraft to baselink conversion Euler angles
        euler_2 = np.array([np.pi, 0.0, 0.0])
        AIRCRAFT_BASELINK_Q = euler2quat(euler_2[2], euler_2[1], euler_2[0], 'szyx')
        # print("AIRCRAFT_BASELINK_Q")
        # print(AIRCRAFT_BASELINK_Q)

        # Perform the quaternion multiplications to achieve the desired rotation
        # Note: the multiply function from transforms3d takes quaternions in [w, x, y, z] format
        result_quat = qmult(NED_ENU_Q, quat_in)
        result_quat = qmult(result_quat, AIRCRAFT_BASELINK_Q)

        # # Convert the rotated quaternion back to w, x, y, z order before returning
        return result_quat          # ordering (w, x, y, z)
    
    def quaternion_rotation_matrix(self,Q):
        # Extract the values from Q   (w-x-y-z) #### NEED TRANSPOSE
        q0 = Q[0]
        q1 = Q[1]
        q2 = Q[2]
        q3 = Q[3]
     
        # First row of the rotation matrix
        r00 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
     
        # Second row of the rotation matrix
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = q0 * q0 - q1 * q1 + q2 * q2 - q3 * q3
        r12 = 2 * (q2 * q3 - q0 * q1)
     
        # Third row of the rotation matrix
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = q0 * q0 - q1 * q1 - q2 * q2 + q3 * q3 
     
        # 3x3 rotation matrix
        rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
        return rot_matrix    

    def euler_from_quaternion(self, x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z # in radians

def main():
    rclpy.init(args=None)

    Controller_module_ = Controller_module()

    rclpy.spin(Controller_module_)

    Controller_module_.destroy_node()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()