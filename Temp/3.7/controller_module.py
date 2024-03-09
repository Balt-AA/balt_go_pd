import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from tf_transformations import quaternion_from_euler, euler_from_quaternion, quaternion_inverse

from px4_msgs.msg import *
from std_msgs.msg import Int32

import linecache
import ast
import math
import time
from cvxopt import matrix, solvers


class controller_module(Node):

    def __init__(self):
        super().__init__('controller_module')
        qos_profile_1 = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )
        qos_profile_2 = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )       

        # Define subscribers
        self.status_sub                         = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile_1)
        self.hover_thrust_estimate_subscriber_  = self.create_subscription(HoverThrustEstimate, "fmu/out/hover_thrust_estimate", self.process_hover_thrust_estimate, qos_profile_2)
        self.local_position_subscriber_         =   self.create_subscription(VehicleLocalPosition, 'fmu/out/vehicle_local_position', self.subscribe_vehicle_local_position, qos_profile_2)
        self.attitude_setpoint_publisher_       =   self.create_publisher(VehicleAttitudeSetpoint, "fmu/in/vehicle_attitude_setpoint", qos_profile_2)
        # Define publishers
        self.publisher_offboard_mode            = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile_2)
        self.publisher_trajectory               = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile_2)

        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.dt = timer_period
        self.theta = 0.0 # t is THETA, modified to match svgeq.jar output
        self.radius = 10.0
        self.omega = 0.5
        
        ## Initalize
        self.pos_x_                             =   np.float32(0.0)
        self.pos_y_                             =   np.float32(0.0)
        self.pos_z_                             =   np.float32(0.0)
        self.vel_x_                             =   np.float32(0.0)
        self.vel_y_                             =   np.float32(0.0)
        self.vel_z_                             =   np.float32(0.0)
        self.vel_dot_x                          =   np.float32(0.0)
        self.vel_dot_y                          =   np.float32(0.0)
        self.vel_dot_z                          =   np.float32(0.0)
        self.params_g                           =   np.float32(9.80665)
        self.quadrotor_mass                     =   np.float32(2.02)     # kg

        self.A_vel_max                          =   np.float32(0.5)

        self._lim_vel_horizontal                =   np.float32(12.0)
        self._lim_vel_up                        =   np.float32(3.0)

        self._lim_tilt                          =   np.float32(45.0)
        self._lim_thr_min                       =   np.float32(0.12)
        self._lim_thr_max                       =   np.float32(1.0)
        self.body_z                             =   np.zeros(3, dtype=np.float32)
        self._hover_thrust                      =   np.float32(0.0)
        self._hover_thrust_var_                 =   np.float32(0.0)
        self._hover_thrust_previous             =   np.float32(0.0)
        self._thr_sp                            =   np.zeros(3, dtype=np.float32)
        self._lim_thr_xy_margin                 =   np.float32(0.3)

        self.euler_d_                           =   np.zeros(3,dtype=np.float32)
        self.q_d_                               =   np.zeros(4,dtype=np.float32)
        self.thrust_body_                       =   np.zeros(3,dtype=np.float32)    

        # Position controller P Gain
        self.k_pos_gain                         =   np.float32(0.95)
        self.k_pos_z_gain                       =   np.float32(1.0)

        # Velocity Controller PID Gain
        self.k_vel_gain                         =   np.float32(3.0)
        self.k_vel_z_gain                       =   np.float32(4.0)
        self.i_vel_gain                         =   np.float32(4.0)
        self.i_vel_z_gain                       =   np.float32(2.0)
        self.d_vel_gain                         =   np.float32(0.20)
        self.d_vel_z_gain                       =   np.float32(0.0)

        self.i_vel_x_instant                    =   np.float32(0.0)
        self.i_vel_y_instant                    =   np.float32(0.0)
        self.i_vel_z_instant                    =   np.float32(0.0)


        
    def cmdloop_callback(self):
        # Publish offboard control modes
        self.publish_offboard_mode(pos_cont=False, vel_cont=False, acc_cont=False, att_cont=True)

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            
            current_time = int(Clock().now().nanoseconds / 1000)

            if self.omega == 0.5:
                #Initialize
                x = np.array([
                [self.pos_x_],
                [self.pos_y_],
                [self.pos_z_],
                [self.vel_x_],
                [self.vel_y_],
                [self.vel_z_]
                ])
               
                # F 행렬 정의
                F = np.array([
                [self.vel_x_],
                [self.vel_y_],
                [self.vel_z_],
                [0],
                [0],
                [-self.params_g]
                ])

                # G 행렬 정의
                G_k = np.array([
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
                ])

                x_ref = np.array([
                    [self.radius * np.cos(self.theta)],
                    [self.radius * np.sin(self.theta)],
                    [-10]
                ], dtype = np.float32)

                vel_ref = np.array([
                    [self.k_pos_gain * (x_ref[0] - x[0])],
                    [self.k_pos_gain * (x_ref[1] - x[1])],
                    [self.k_pos_z_gain * (x_ref[2] - x[2])]
                ], dtype=np.float32)
                
                vel_ref[0] = np.clip(vel_ref[0], -self._lim_vel_horizontal, self._lim_vel_horizontal)
                vel_ref[1] = np.clip(vel_ref[1], -self._lim_vel_horizontal, self._lim_vel_horizontal)
                vel_ref[2] = np.clip(vel_ref[2], -self._lim_vel_up, self._lim_vel_up)          

                # u_nominal == Acceleration (x, y, z)
                self.i_vel_z_instant = np.clip(self.i_vel_z_instant, -self.params_g, self.params_g)

                vel_error = np.array([
                    [vel_ref[0] - self.vel_x_],
                    [vel_ref[1] - self.vel_y_],
                    [vel_ref[2] - self.vel_z_]
                ], dtype=np.float32)     

                self.acceleration_cmd_ = np.array([
                    [self.k_vel_gain * vel_error[0] + self.i_vel_x_instant - self.d_vel_gain * self.vel_dot_x],
                    [self.k_vel_gain * vel_error[1] + self.i_vel_y_instant - self.d_vel_gain * self.vel_dot_y],
                    [self.k_vel_z_gain * vel_error[2] + self.i_vel_z_instant - self.d_vel_z_gain * self.vel_dot_z]
                ], dtype=np.float32)

                self._hover_thrust_previous = self._hover_thrust

                self.body_z = np.array([-self.acceleration_cmd_[0], -self.acceleration_cmd_[1], self.params_g], dtype=np.float32)
                self.body_z /= np.linalg.norm(self.body_z)  # Normalizing vector

                if self._hover_thrust < 0.1:
                    self._hover_thrust = 0.1
                
                np.where(np.isnan(self._hover_thrust), 0.1, self._hover_thrust)

                self.body_z = self.limit_tilt(self.body_z, np.array([0, 0, 1], dtype=np.float32), self._lim_tilt)
                collective_thrust = self.acceleration_cmd_[2] * (self._hover_thrust / self.params_g) - self._hover_thrust
                collective_thrust /= np.dot(np.array([0, 0, 1]), self.body_z)
                collective_thrust = min(collective_thrust, -self._lim_thr_min)
                self._thr_sp = self.body_z * collective_thrust

            
                self.i_vel_z_instant += (self.acceleration_cmd_[2] - self.params_g) * self._hover_thrust_previous / self._hover_thrust \
                            + self.params_g - self.acceleration_cmd_[2]

                self._thr_sp = self._thr_sp.reshape(3,1)

                # Integrator anti-windup in vertical direction
                if ((self._thr_sp[2] >= -self._lim_thr_min and vel_error[2] >= 0.0) or
                        (self._thr_sp[2] <= -self._lim_thr_max and vel_error[2] <= 0.0)):
                        vel_error[2] = 0
            
                # Prioritize vertical control while keeping a horizontal margin
                thrust_sp_xy = self._thr_sp[:2]
                thrust_sp_xy_norm = np.linalg.norm(thrust_sp_xy)
                thrust_max_squared = self.params_g ** 2 * self._lim_thr_max ** 2

                allocated_horizontal_thrust = min(thrust_sp_xy_norm, self._lim_thr_xy_margin)
                thrust_z_max_squared = thrust_max_squared - allocated_horizontal_thrust ** 2
                
                # Saturate maximal vertical thrust
                self._thr_sp[2] = max(self._thr_sp[2], -np.sqrt(thrust_z_max_squared))

                # Determine how much horizontal thrust is left after prioritizing vertical control
                thrust_max_xy_squared = thrust_max_squared - self._thr_sp[2] ** 2
                thrust_max_xy = 0

                if thrust_max_xy_squared > 0:
                    thrust_max_xy = np.sqrt(thrust_max_xy_squared)

                # Saturate thrust in horizontal direction
                if thrust_sp_xy_norm > thrust_max_xy:
                    self._thr_sp[:2] = thrust_sp_xy / thrust_sp_xy_norm * thrust_max_xy

                # Use tracking Anti-Windup for horizontal direction
                acc_sp_xy_produced = self._thr_sp[:2] * (self.params_g / self._hover_thrust)
                arw_gain = 2.0 / self.k_vel_gain

                # The ARW loop
                acc_sp_xy = self.acceleration_cmd_[:2]
                acc_limited_xy = acc_sp_xy_produced if np.linalg.norm(acc_sp_xy)**2 > np.linalg.norm(acc_sp_xy_produced)**2 else acc_sp_xy

                vel_error[0] = vel_error[0] - arw_gain * (acc_sp_xy[0] - acc_limited_xy[0])
                vel_error[1] = vel_error[1] - arw_gain * (acc_sp_xy[1] - acc_limited_xy[1])

                # Make sure integral doesn't get NAN
                vel_error = np.where(np.isnan(vel_error), 0, vel_error)

                self.i_vel_x_instant += self.i_vel_gain * (vel_error[0]) * self.dt
                self.i_vel_y_instant += self.i_vel_gain * (vel_error[1]) * self.dt
                self.i_vel_z_instant += self.i_vel_gain * (vel_error[2]) * self.dt


                total_acceleration_     =   np.sqrt(np.power(self.acceleration_cmd_[0],2) \
                                            +np.power(self.acceleration_cmd_[1],2) \
                                            +np.power(self.acceleration_cmd_[2]- \
                                            self.params_g,2))
                self.unit_thrust_level_ =   self._hover_thrust/ \
                                            (self.quadrotor_mass*self.params_g)                 

                self.euler_d_[0]        =   np.arcsin(self.acceleration_cmd_[1]/total_acceleration_)    # [deg] Roll angle
                self.euler_d_[1]        =   np.arcsin(-self.acceleration_cmd_[0]/ \
                                            np.cos(self.euler_d_[0])/total_acceleration_)               # [deg] Pitch angle
                self.euler_d_[2]        =   0.0/180.0*np.pi                                             # [deg] Yaw angle

                temp_q_d_               =   np.float32(quaternion_from_euler(self.euler_d_[2], \
                                            self.euler_d_[1],self.euler_d_[0],axes='rzyx'))             # [-] Quaternion command (x-y-z-w order)
                self.q_d_               =   temp_q_d_[[3,0,1,2]]                                        # [-] Quaternion command re-ordering (w-x-y-z order)        
                
                self.thrust_body_[0]    =   0
                self.thrust_body_[1]    =   0
                self.thrust_body_[2]    =   -self.quadrotor_mass*total_acceleration_ \
                                            *self.unit_thrust_level_                                    # [%] Body-z thrust level  

                self.publish_attitude_setpoint()
      
                
                self.theta = self.theta + self.omega * self.dt

                #else:
                    # Handle the error (e.g., log, raise an exception, use default values, etc.)
                    #print("Error: solve_qp did not return a solution.")
                
            else:
                # Non CBF
                pass

            #offboard_msg = OffboardControlMode()
            #offboard_msg.timestamp = int(Clock().now().nanoseconds / 1000)
            #self.publisher_offboard_mode.publish(offboard_msg)
    
    def limit_tilt(self, body_unit, world_unit, max_angle):
        """
        Limit the tilt of the body_unit vector towards the world_unit vector
        to a maximum angle of max_angle radians.
        """
        dot_product_unit = np.dot(body_unit, world_unit)
        angle = np.arccos(dot_product_unit)
        # Limit tilt
        angle = min(angle, max_angle)
        rejection = body_unit - (dot_product_unit * world_unit)

        # Handle the corner case of exactly parallel vectors
        if np.linalg.norm(rejection)**2 < np.finfo(np.float32).eps:
            rejection[0] = np.float32(1.0)

        body_unit_new = np.cos(angle) * world_unit + np.sin(angle) * rejection / np.linalg.norm(rejection)
    
        return body_unit_new

    def hyper_tangent(self, input_signal, gain=1.0):
        """
        Applies a hyperbolic tangent function to reduce chattering.

        Parameters:
        - input_signal: The input signal to the control system.
        - gain: The gain factor to adjust the steepness of the tanh function.

        Returns:
        - The output signal after applying the tanh function.
        """
        return np.tanh(gain * input_signal)

    # Subscribers
    def vehicle_status_callback(self, msg):
        # TODO: handle NED->ENU transformation
        print("NAV_STATUS: ", msg.nav_state)
        print("  - offboard status: ", VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        self.nav_state = msg.nav_state

    def subscribe_vehicle_local_position(self, msg):
        self.pos_x_                                 =   np.float32(msg.x)
        self.pos_y_                                 =   np.float32(msg.y)
        self.pos_z_                                 =   np.float32(msg.z)
        self.vel_x_                                 =   np.float32(msg.vx)
        self.vel_y_                                 =   np.float32(msg.vy)
        self.vel_z_                                 =   np.float32(msg.vz)
        self.vel_dot_x                              =   np.float32(msg.ax)
        self.vel_dot_y                              =   np.float32(msg.ay)
        self.vel_dot_z                              =   np.float32(msg.az)
    
    def process_hover_thrust_estimate(self,msg):
        self._hover_thrust                          =   msg.hover_thrust
        self._hover_thrust_var_                     =   msg.hover_thrust_var

    # Publisher
    def publish_offboard_mode(self, pos_cont, vel_cont, acc_cont, att_cont):
        msg = OffboardControlMode()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.position = bool(pos_cont)
        msg.velocity = bool(vel_cont)
        msg.acceleration = bool(acc_cont)
        msg.attitude = bool(att_cont)
        self.publisher_offboard_mode.publish(msg)

    def publish_attitude_setpoint(self):
        msg                                         =   VehicleAttitudeSetpoint()
        msg.timestamp                               =   int(Clock().now().nanoseconds / 1000)
        msg.q_d                                     =   self.q_d_
        msg.thrust_body                             =   self.thrust_body_

        self.attitude_setpoint_publisher_.publish(msg)  

def main(args=None):
    rclpy.init(args=args)

    controller_module_ = controller_module()

    rclpy.spin(controller_module_)

    controller_module_.destroy_node()
    
    rclpy.shutdown()


if __name__ == '__main__':
    main()