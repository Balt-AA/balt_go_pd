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
from Controller_lib import Controller

import linecache
import ast
import math
import time
from cvxopt import matrix, solvers
from scipy.spatial.transform import Rotation as R
from transforms3d.quaternions import quat2mat, mat2quat, qmult
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

        # qos_profile = QoSProfile(
        #     reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
        #     durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
        #     history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        #     depth=5
        # )

        # Define subscribers
        self.status_subscriber_           = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile)
        
        # self.local_position_subscriber_   = self.create_subscription(
        #     VehicleLocalPosition, '/fmu/out/vehicle_local_position',
        #     self.subscribe_vehicle_local_position,
        #     qos_profile)
        
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

        # self.local_position_subscriber_ =   self.create_subscription(VehicleLocalPosition, 'fmu/out/vehicle_local_position', self.subscribe_vehicle_local_position, 10) 
               
        # Define publishers
        self.offboard_mode_publisher_   = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode',
            10)

        self.actuator_motors_publisher_ = self.create_publisher(
            ActuatorMotors,
            '/fmu/in/actuator_motors',  
            10)
        
        # self.attitude_setpoint_publisher_ = self.create_publisher(
        #     AttitudeSetpoint,
        #     '/fmu/in/attitude_setpoint',
        #     10)
        
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
        #self.wrench = np.zeros(4)
        #self.desired_quat = np.zeros(4)
        #self.normalized_torque_thrust = np.zeros(4)
        self.throttles = np.zeros(4)

        # Inintialize UAV parameters
        self.zero_position_armed = 100
        self.input_scaling = 1000
        self.thrust_constant = 5.84e-06
        self.moment_constant = 0.06       
        self.arm_length = 0.25
        #self.compute_ControlAllocation_and_ActuatorEffect_matrices()

        # Initialize matrices
        self.torques_and_thrust_to_rotor_velocities_ = np.zeros((4, 4))
        self.throttles_to_normalized_torques_and_thrust_ = np.zeros((4, 4))

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
        controller_.set_uav_parameters(1.725, np.array([0.029125, 0.029125, 0.055225]), 9.81)
        controller_.set_control_gains(np.array([7.0, 7.0, 6.0]), np.array([6.0, 6.0, 3.0]), np.array([3.5, 3.5, 0.3]), np.array([0.5, 0.5, 0.1]))
        
        pos_odo1, vel_odo1, quat_odo1, angvel_odo1 = self.eigen_odometry_from_PX4_msg(self.pos_odo, self.vel_odo, self.quat_odo, self.angvel_odo)

        # Set odometry and trajectory point
        controller_.set_odometry(pos_odo1, vel_odo1, quat_odo1, angvel_odo1)
        controller_.set_trajectory_point(self.pos_cmd, self.ori_cmd)   

        # Calculate controller output
        wrench, desired_quat = controller_.calculate_controller_output()
        print("wrench")
        print(wrench)
        normalized_torque_thrust, throttles = self.px4InverseSITL(wrench)

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            
            current_time = int(Clock().now().nanoseconds / 1000)

            # Calculate controller output
            self.actuator_motors_publish_(throttles)
            # self.attitude_setpoint_publish_(wrench[3]/1000, desired_quat)


        self.get_logger().info(f'time: {self.time_temp}, \n \
                                 Odometry Position: {self.pos_odo}, \n Quaternion: {self.quat_odo}, \n Velocity: {self.vel_odo}, \n \
                                 Angular Velocity: {self.angvel_odo}, \n Desired Position: {self.pos_cmd}, \n Desired Orientation: {self.ori_cmd}, \n \
                                 Navigation State: {self.nav_state}, \n \
                                 Desired Quaternion: {controller_.r_R_B_W}, \n Desired Yaw: {controller_.r_yaw}')
    
        # print('=============== State check panel ===============')
        # print('Position Odometry[m]: %.3f %.3f %.3f' %(self.pos_odo[0], self.pos_odo[1], self.pos_odo[2]))
        # print('Position Desired [m]: %.3f %.3f %.3f' %(self.pos_cmd[0], self.pos_cmd[1], self.pos_cmd[2]))
        # print('Quarternion [-]: %.3f %.3f %.3f' %(self.pos_cmd[0], self.pos_cmd[1], self.pos_cmd[2]))

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

        # Control allocation: Wrench to Rotational velocities (omega)
        omega = array @ wrench
        omega = np.sqrt(np.abs(omega))  # Element-wise square root, handle negative values
        print("omega")
        print(omega)
        
        # CBF
        indv_forces = omega * np.abs(omega) * self.thrust_constant  
        
        # Calculate hrottles from omega (rotor velocities)
        throttles = (omega - (self.zero_position_armed * ones_temp))
        throttles = throttles / self.input_scaling

        #np.clip(throttles, 0.0, 0.9, out=throttles)


        print("throttles")
        print(throttles)
        
        # Inverse Mixing: throttles to normalized torques and thrust
        normalized_torque_and_thrust = self.throttles_to_normalized_torques_and_thrust_ @ throttles

        return normalized_torque_and_thrust, throttles


    # Function for Sliding Mode Conrol Barrier Function
    def hyper_tangent(self, input_signal, gain=1.0):
        
        return np.tanh(gain * input_signal)

    # Subscribers
    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state

    # def subscribe_vehicle_local_position(self, msg):
    #     self.pos_x_                                 =   np.float32(msg.x)
    #     self.pos_y_                                 =   np.float32(msg.y)
    #     self.pos_z_                                 =   np.float32(msg.z)
    #     self.vel_x_                                 =   np.float32(msg.vx)
    #     self.vel_y_                                 =   np.float32(msg.vy)
    #     self.vel_z_                                 =   np.float32(msg.vz)

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

    def rotate_quaternion_from_to_ENU_NED(self, quat_in):
        # Transform from orientation represented in ROS format to PX4 format and back.
        # Using scipy's Rotation module for quaternion operations.
        # quat_in_reordered = [quat_in[1], quat_in[2], quat_in[3], quat_in[0]]
    
        # NED to ENU conversion Euler angles
        euler_1 = np.array([np.pi, 0.0, np.pi/2])
        NED_ENU_Q = euler2quat(euler_1[2], euler_1[1], euler_1[0], 'szyx')
        #NED_ENU_Q = mat2quat(NED_ENU_Q)
        print("NED_ENU_Q")
        print(NED_ENU_Q)

        # Aircraft to baselink conversion Euler angles
        euler_2 = np.array([np.pi, 0.0, 0.0])
        AIRCRAFT_BASELINK_Q = euler2quat(euler_2[2], euler_2[1], euler_2[0], 'szyx')
        #AIRCRAFT_BASELINK_Q = mat2quat(AIRCRAFT_BASELINK_Q)
        print("AIRCRAFT_BASELINK_Q")
        print(AIRCRAFT_BASELINK_Q)

        # Perform the quaternion multiplications to achieve the desired rotation
        # Note: the multiply function from transforms3d takes quaternions in [w, x, y, z] format
        result_quat = qmult(NED_ENU_Q, quat_in)
        result_quat = qmult(result_quat, AIRCRAFT_BASELINK_Q)

        #quat_in_reordered = np.array([quat_in[1], quat_in[2], quat_in[3], quat_in[0]])
        #rotated_R_quat = self.quaternion_multiply(self.quaternion_multiply(NED_ENU_Q, quat_in), AIRCRAFT_BASELINK_Q)

        # # Convert the rotated quaternion back to w, x, y, z order before returning
        return result_quat          # ordering (w, x, y, z)

    def quaternion_multiply(self, q0, q1):
        """
        Multiplies two quaternions.

        Input
        :param q0: A 4 element array containing the first quaternion (q01, q11, q21, q31)
        :param q1: A 4 element array containing the second quaternion (q02, q12, q22, q32)

        Output
        :return: A 4 element array containing the final quaternion (q03,q13,q23,q33)

        """
        # Extract the values from q0
        w0 = q0[3]
        x0 = q0[0]
        y0 = q0[1]
        z0 = q0[2]

        # Extract the values from q1
        w1 = q1[3]
        x1 = q1[0]
        y1 = q1[1]
        z1 = q1[2]

        # Computer the product of the two quaternions, term by term
        q0q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        q0q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        q0q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        q0q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

        # Create a 4 element array containing the final quaternion
        final_quaternion = np.array([q0q1_w, q0q1_x, q0q1_y, q0q1_z])

        # Return a 4 element array containing the final quaternion (q02,q12,q22,q32)
        return final_quaternion
    
def main():
    rclpy.init(args=None)

    Controller_module_ = Controller_module()

    rclpy.spin(Controller_module_)

    Controller_module_.destroy_node()
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()