#!/usr/bin/env python

# Load Public Libraries
import numpy as np
import random
import time

# Load ROS2 related Libraries
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

# Load ROS2 Messages
from px4_msgs.msg import *
from std_msgs.msg import Int32

# Load Internal messages
#from internal_interfaces.msg from *

class attacker_module(Node):

    def __init__(self):
        super().__init__('attacker_module')

        # Define Publisher
        self.motor_failure_publisher_ = self.create_publisher(Int32, '/motor_failure/motor_number', 10)

        # Define Subscriber to a hypothetical autopilot engagement topic
        self.status_sub = self.create_subscription(VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, 10)

        self.motors_to_fail = np.int32(0)
        self.nav_state_ = np.uint(0)
        self.callback_group = ReentrantCallbackGroup()
        timer_period = 0.01  # seconds
        self.dt = timer_period
        self.failure_timer = self.create_timer(timer_period, self.fail_motors, callback_group=self.callback_group)
        self.callback_counter                   =   int(0)
        self.motors_to_fail                     =   int(0)

    def fail_motors(self):
        if self.nav_state_ == 15:                                             # If the vehicle is in OFFBOARD mode (14) / else (15)                   
            self.callback_counter += 1

            if self.callback_counter == 701:
                self.motors_to_fail = int(random.choice(range(0, 6)))         # Randomly select one motors from 1 to 6
                self.publish_motor_failure(self.motors_to_fail)
            
            elif self.callback_counter == 1000:
                self.publish_motor_failure(0)
                self.callback_counter = 0
        else:
            pass
        
    def vehicle_status_callback(self, msg):
        self.nav_state_ = np.uint8(msg.nav_state)
  
    def publish_motor_failure(self, motor_id):
        msg = Int32()
        msg.data = motor_id
        self.motor_failure_publisher_.publish(msg)
        self.get_logger().info(f'Publishing motor failure on motor: {msg.data}')
        
def main():

    # Initialize python rclpy library
    rclpy.init(args=None)

    # Create attacker node
    attacker_module_ = attacker_module()
    executor = MultiThreadedExecutor()
    executor.add_node(attacker_module_)

    try:
        executor.spin()

    finally:
        attacker_module_.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()