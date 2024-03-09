#!/usr/bin/env python

import numpy as np
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R

class Controller():
    def __init__(self):
        # Initialize UAV parameters with placeholder values
        self._uav_mass = 1.725  # UAV mass
        self._inertia_matrix = np.array([0.029125, 0.029125, 0.055225])  # Inertia matrix
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

    def set_odometry(self, position_W, velocity_W, orientation_B_W, angular_velocity_B):
        self.position_W = position_W
        self.velocity_W = velocity_W
        self.orientation_B_W_temp = np.array([orientation_B_W[1], orientation_B_W[2], orientation_B_W[3], orientation_B_W[0]])  #Quaternion reordering (x, y, z, w)
        if norm(self.orientation_B_W_temp) == 0:
            self.orientation_B_W_temp = np.array([0.0, 0.0, 0.0, 1.0])
        self.R_B_W = R.from_quat(self.orientation_B_W_temp).as_matrix()
        self.angular_velocity_B = angular_velocity_B

    def set_trajectory_point(self, position_W, orientation_W):
        self.r_position_W = position_W
        self.r_velocity_W = np.zeros(3)
        self.r_acceleration_W = np.zeros(3)
        #self.orientation_W_temp = np.array([orientation_W[3], orientation_W[0], orientation_W[1], orientation_W[2]])  #Quaternion reordering (w, x, y, z)
        if norm(orientation_W) == 0:
            orientation_W = np.array([0.0, 0.0, 0.0, 1.0])
        self.r_R_B_W = R.from_quat(orientation_W).as_matrix()
        self.r_yaw = R.from_quat(orientation_W).as_euler('zyx', degrees=False)[0]
        self.r_yaw_rate = 0.0

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

        # Compute translational tracking errors
        e_p = self.position_W - self.r_position_W
        e_v = self.velocity_W - self.r_velocity_W
        I_a_d = -np.multiply(self.position_gain, e_p) - np.multiply(self.velocity_gain, e_v) + \
                self._uav_mass * self._gravity * np.array([0.0, 0.0, 1.0]) + self._uav_mass * self.r_acceleration_W
        thrust = np.dot(I_a_d, self.R_B_W[:, 2])
        B_z_d = I_a_d / norm(I_a_d)

        # Calculate Desired Rotational Matrix
        B_x_d = np.array([np.cos(self.r_yaw), np.sin(self.r_yaw), 0.0])
        B_y_d = np.cross(B_z_d, B_x_d)
        B_y_d /= norm(B_y_d)
        R_d_w[:, 0] = np.cross(B_y_d, B_z_d)
        R_d_w[:, 1] = B_y_d
        R_d_w[:, 2] = B_z_d

        q_temp = R.from_matrix(R_d_w).as_quat()
        q_temp = np.array([q_temp[3], q_temp[0], q_temp[1], q_temp[2]])  # Quaternion reordering (w, x, y, z)
        self.desired_quaternion[:] = q_temp

        # Attitude tracking
        e_R_matrix = 0.5 * (np.transpose(R_d_w) @ self.R_B_W - np.transpose(self.R_B_W) @ R_d_w)
        e_R = np.array([e_R_matrix[2, 1], e_R_matrix[0, 2], e_R_matrix[1, 0]])
        omega_ref = self.r_yaw_rate * np.array([0.0, 0.0, 1.0])
        e_omega = self.angular_velocity_B - np.transpose(self.R_B_W) @ R_d_w @ omega_ref
        tau = -np.multiply(self.attitude_gain, e_R) - np.multiply(self.angular_rate_gain, e_omega) + \
              np.cross(self.angular_velocity_B, np.diag(self._inertia_matrix) @ self.angular_velocity_B)

        # Output the wrench
        self.controller_torque_thrust[:3] = tau
        self.controller_torque_thrust[3] = thrust

        return self.controller_torque_thrust, self.desired_quaternion