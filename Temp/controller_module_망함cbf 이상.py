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

        self.status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile_1)
        self.publisher_offboard_mode = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile_2)
        self.publisher_trajectory = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile_2)

        self.local_position_subscriber_ =   self.create_subscription(VehicleLocalPosition, 'fmu/out/vehicle_local_position', self.subscribe_vehicle_local_position, qos_profile_2)
        timer_period = 0.01  # seconds
        self.timer = self.create_timer(timer_period, self.cmdloop_callback)

        self.nav_state = VehicleStatus.NAVIGATION_STATE_MAX
        self.dt = timer_period
        self.theta = 0.0 # t is THETA, modified to match svgeq.jar output
        self.radius = 15.0
        self.omega = 0.5
        
        ## Initalize
        self.pos_x_                             =   np.float32(0.0)
        self.pos_y_                             =   np.float32(0.0)
        self.pos_z_                             =   np.float32(0.0)
        self.vel_x_                             =   np.float32(0.0)
        self.vel_y_                             =   np.float32(0.0)
        self.vel_z_                             =   np.float32(0.0)

        self.A_vel_max                          =   np.float32(0.5)
        self.k_pos_gain                         =   np.float32(0.95)
        self.k_pos_z_gain                       =   np.float32(1.0)
        self.alpha_x_gain                       =   np.float32(2.0)
        self.alpha_y_gain                       =   np.float32(2.0)
        self.alpha_z_gain                       =   np.float32(2.0)

        
    def cmdloop_callback(self):
        # Publish offboard control modes
        self.publish_offboard_mode(pos_cont=False, vel_cont=True, act_cont=False)

        if self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
            
            current_time = int(Clock().now().nanoseconds / 1000)

            if self.omega == 0.5:
                #Initialize
                x = np.array([
                [self.pos_x_],
                [self.pos_y_],
                [self.pos_z_]
                ])
               
                # F 행렬 정의
                #F = np.array([
                #[self.vel_x_],
                #[self.vel_y_],
                #[self.vel_z_]
                #])

                # G 행렬 정의
                G_k = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
                ])

                x_ref = np.array([
                    [self.radius * np.cos(self.theta)],
                    [self.radius * np.sin(self.theta)],
                    [-15.0]
                ], dtype = np.float32)

                u_nominal = np.array([
                    [self.k_pos_gain * (x[0] - x_ref[0])],
                    [self.k_pos_gain * (x[1] - x_ref[1])],
                    [self.k_pos_z_gain * (x[2] - x_ref[2])]
                ], dtype=np.float32)

                # CBF
                # B_v_x
                epsilon = 0.5
                B_v_x = 1 - ((x[0] - x_ref[0]) / (x_ref[0] * self.A_vel_max + epsilon)) ** 2
                LfB_v_x = 0 
                LgB_v_x = -2 * ((x[0] - x_ref[0]) / (x_ref[0] * self.A_vel_max + epsilon))

                # B_v_y
                B_v_y = 1 - ((x[1] - x_ref[1]) / (x_ref[1] * self.A_vel_max + epsilon)) ** 2
                LfB_v_y = 0 
                LgB_v_y = -2 * ((x[1] - x_ref[1]) / (x_ref[1] * self.A_vel_max + epsilon))

                # B_v_z
                B_v_z = 1 - ((x[2] - x_ref[2]) / (x_ref[2] * self.A_vel_max + epsilon)) ** 2
                LfB_v_z = 0 
                LgB_v_z = -2 * ((x[2] - x_ref[2]) / (x_ref[2] * self.A_vel_max + epsilon))

                # Define A_ineq as a numpy array
                A_ineq = np.array([
                    [-LgB_v_x, 0, 0, -1],
                    [0, -LgB_v_y, 0, 0],
                    [0, 0, -LgB_v_z, 0]
                ], dtype=np.float64)

                # Define B_ineq
                B_ineq = np.array([
                    [self.alpha_x_gain * B_v_x],
                    [self.alpha_y_gain * B_v_y],
                    [self.alpha_z_gain * B_v_z]
                ], dtype=np.float64)
                B_ineq = B_ineq.reshape(3, 1)

                Q = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 0.1]
                ], dtype=np.float64)
                #Q = csc_matrix(Q)

                # Define f
                f = np.array([
                    [-u_nominal[0]],
                    [-u_nominal[1]],
                    [-u_nominal[2]],
                    [0]
                ], dtype=np.float64)
                
                # Convert to cvxopt matrices
                P = matrix(Q)
                q = matrix(f)
                G = matrix(A_ineq)
                h = matrix(B_ineq)

                # Solve QP problem
                solvers.options['show_progress'] = False
                solution = solvers.qp(P, q, G, h)

                u_temp = solution['x']
                
                trajectory_msg = TrajectorySetpoint()
                trajectory_msg.velocity[0] = u_temp[1]
                trajectory_msg.velocity[1] = u_temp[0]
                trajectory_msg.velocity[2] = -u_temp[2]
                self.publisher_trajectory.publish(trajectory_msg)

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
                                           
    # Publisher
    def publish_offboard_mode(self, pos_cont, vel_cont, act_cont):
        msg = OffboardControlMode()
        msg.timestamp = int(Clock().now().nanoseconds / 1000)
        msg.position = bool(pos_cont)
        msg.velocity = bool(vel_cont)
        msg.acceleration = False
        msg.direct_actuator= bool(act_cont)
        self.publisher_offboard_mode.publish(msg)
    

def main(args=None):
    rclpy.init(args=args)

    controller_module_ = controller_module()

    rclpy.spin(controller_module_)

    controller_module_.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()