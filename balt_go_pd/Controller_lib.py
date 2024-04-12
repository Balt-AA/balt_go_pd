#!/usr/bin/env python

import math
import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R

class Controller():
    def __init__(self):
        # Initialize UAV parameters with placeholder values
        self._uav_mass = 1.725  # UAV mass
        self._inertia_matrix = np.array([0.029125, 0.029125, 0.055125])  # Inertia matrix
        self._gravity = 9.81  # Gravity

        # Initialize control gains with placeholder values
        self.position_gain = np.array([7.0, 7.0, 6.0])
        self.velocity_gain = np.array([6.0, 6.0, 3.0])
        self.attitude_gain = np.array([3.5, 3.5, 0.3])
        self.angular_rate_gain = np.array([0.5, 0.5, 0.1])

        # Initialize state and reference values with zeros
        self.position_W = np.zeros(3)
        self.velocity_W = np.zeros(3)
        self.orientation_B_W_temp = np.array([0.0, 0.0, 0.0, 1.0])
        self.R_B_W = np.eye(3)
        self.angular_velocity_B = np.zeros(3)
        
        self.r_position_W = np.zeros(3)
        self.r_velocity_W = np.zeros(3)
        self.r_acceleration_W = np.zeros(3)
        #self.orientation_W_temp = np.zeros(4)
        self.r_R_B_W = np.eye(3)
        self.r_yaw = 0.0
        self.r_yaw_rate = 0.0

        # Initialize controller output with zeros
        self.controller_torque_thrust = np.zeros(4)
        self.desired_quaternion = np.zeros(4)

    def set_odometry(self, position_W, velocity_B, orientation_B_W, angular_velocity_B):
        self.orientation_B_W_temp = np.array([orientation_B_W[1], orientation_B_W[2], orientation_B_W[3], orientation_B_W[0]])  #Quaternion reordering (x, y, z, w)
        if norm(self.orientation_B_W_temp) == 0:
            self.orientation_B_W_temp = np.array([0.0, 0.0, 0.0, 1.0])
        self.R_B_W = self.quaternion_rotation_matrix(orientation_B_W)
        self.position_W = position_W
        self.velocity_W = velocity_B #self.R_B_W @ 
        self.angular_velocity_B = angular_velocity_B

    def set_trajectory_point(self, position_W, orientation_W):
        self.r_position_W = position_W
        self.r_velocity_W = np.zeros(3)
        self.r_acceleration_W = np.zeros(3)
        if norm(orientation_W) == 0:
            orientation_W = np.array([0.0, 0.0, 0.0, 1.0])
        self.r_R_B_W = R.from_quat(orientation_W).as_matrix()
        self.r_yaw = R.from_matrix(self.r_R_B_W).as_euler('ZYX')[0]
        self.r_yaw_rate = 0.0

    def quaternion_rotation_matrix(self,Q):
        # Extract the values from Q   (w-x-y-z) 
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
    # def quaternion_matrix(self, quaternion):
    #     """Return homogeneous rotation matrix from quaternion.

    #     True (w-x-y-z)

    #     """
    #     q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    #     nq = np.dot(q, q)
    #     if nq < np.finfo(float).eps * 4.0:
    #         return np.identity(4)
    #     q *= math.sqrt(2.0 / nq)
    #     q = np.outer(q, q)
    #     return np.array((
    #         (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3]),
    #         (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3]),
    #         (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1],)
    #         ), dtype=np.float64)

    def set_control_gains(self, position_gain, velocity_gain, attitude_gain, angular_rate_gain):
        self.position_gain = position_gain
        self.velocity_gain = velocity_gain
        self.attitude_gain = attitude_gain
        self.angular_rate_gain = angular_rate_gain

    def set_uav_parameters(self, uav_mass, inertia_matrix, gravity):
        self._uav_mass = uav_mass
        self._inertia_matrix = inertia_matrix
        self._gravity = gravity

    def calculate_controller_output(self):
        assert self.controller_torque_thrust is not None
        assert self.desired_quaternion is not None

        # Trajectory tracking
        thrust = 0.0
        tau = np.zeros(3)
        R_d_w = np.zeros((3, 3))
        e_p = np.zeros(3)
        e_v = np.zeros(3)
        I_a_d = np.zeros(3)
        B_x_d = np.zeros(3)
        B_y_d = np.zeros(3)
        B_z_d = np.zeros(3)
        q_temp = np.zeros(4)

        # Compute translational tracking errors
        e_p = self.position_W - self.r_position_W
        e_v = self.velocity_W - self.r_velocity_W
        e_a = np.multiply(self.position_gain, e_p) + np.multiply(self.velocity_gain, e_v)
        if norm(e_a) > 7.0:
            e_a = 7.0 * e_a / norm(e_a)        
        I_a_d = (- e_a + \
                self._uav_mass * self._gravity * np.array([0.0, 0.0, 1.0]) + self._uav_mass * self.r_acceleration_W)           # Change sign if NED, mass WAS INCLUDED
        thrust = np.dot(I_a_d, self.R_B_W[:, 2])                                                                            # Change sign if NED
        B_z_d = I_a_d / norm(I_a_d)

        # Calculate Desired Rotational Matrix
        B_x_d = np.array([np.cos(self.r_yaw), np.sin(self.r_yaw), 0.0])
        B_y_d = np.cross(B_z_d, B_x_d)
        B_y_d1 = B_y_d / norm(B_y_d)
        R_d_w = np.array([np.cross(B_y_d1, B_z_d), B_y_d, B_z_d]) # /norm(np.cross(B_y_d1, B_z_d))  

        q_temp = R.from_matrix(R_d_w).as_quat()
        q_temp = np.array([q_temp[3], q_temp[0], q_temp[1], q_temp[2]])  # Quaternion reordering (w, x, y, z)
        self.desired_quaternion[:] = q_temp

        # Attitude tracking
        e_R_matrix = 0.5 * (np.transpose(R_d_w) @ self.R_B_W - np.transpose(self.R_B_W) @ R_d_w)
        e_R = np.array([e_R_matrix[2, 1], e_R_matrix[0, 2], e_R_matrix[1, 0]])
        omega_ref = self.r_yaw_rate * np.array([0.0, 0.0, 1.0])
        e_omega = self.angular_velocity_B - np.transpose(self.R_B_W) @ R_d_w @ omega_ref
        tau = - np.multiply(self.attitude_gain, e_R) - np.multiply(self.angular_rate_gain, e_omega) + \
              np.cross(self.angular_velocity_B, np.diag(self._inertia_matrix) @ self.angular_velocity_B)

        # Output the wrench
        self.controller_torque_thrust[:3] = tau
        self.controller_torque_thrust[3] = thrust

        # print("Tau")
        # print(tau)
        # print("Thrust")
        # print(thrust)


        return self.controller_torque_thrust, self.desired_quaternion