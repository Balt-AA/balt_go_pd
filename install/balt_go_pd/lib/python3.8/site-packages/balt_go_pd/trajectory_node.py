#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import math
from rclpy.clock import ROSClock

class trajectory_node(Node):
    def __init__(self):
        super().__init__('trajectory_node')
        self.publisher_ = self.create_publisher(PoseStamped, 'command/pose', 10)
        self.timer_period = 0.01  # seconds
        self.timer = self.create_timer(self.timer_period, self.publish_figure_eight_pose)
        self.angle = 0.0
        self.A = 2.0  # Scale factor for x direction
        self.B = 5.0  # Scale factor for y direction

    def publish_figure_eight_pose(self):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = ROSClock().now().to_msg()
        pose_stamped.header.frame_id = "base_link"

        pose_stamped.pose.position.x = self.A * math.sin(self.angle)
        pose_stamped.pose.position.y = self.A * math.cos(self.angle)
        pose_stamped.pose.position.z = 2.0  # Adjust Z as needed
        pose_stamped.pose.orientation.w = 1.0  # Assuming no rotation

        self.publisher_.publish(pose_stamped)

        self.angle += 0.01  # Adjust this to control speed of the trajectory

def main(args=None):
    rclpy.init(args=args)
    trajectory_node_ = trajectory_node()
    rclpy.spin(trajectory_node_)

    trajectory_node_.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
