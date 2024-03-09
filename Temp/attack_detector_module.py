# Load Public Libraries
import numpy as np
import random

# Load ROS2 related Libraries
import rclpy
from rclpy.node import Node
from rclpy.clock import Clock

# Load ROS2 Messages
from px4_msgs.msg import *
from std_msgs.msg import Int32

# Load Internal messages
from internal_interfaces.msg from *

class attacker_module(Node,attacker):

    def __init__(self):
        super().__init__('attacker_module')

        #super(). load_parameters()
    
        # Define Publisher
        inputattack

        # Define Subscriber
        afterarming

        # Initialize
        timer

    def attacker_callback(self):
        










def main():

    # Initialize python rclpy library
    rclpy.init(args=None)

    # Create extGCU attitude control node
    attacker_module_    =   attacker_module()

    # Spin the created control node
    rclpy.spin(attacker_module_)

    # After spinning, destroy the node and shutdown rclpy library
    attacker_module_.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()