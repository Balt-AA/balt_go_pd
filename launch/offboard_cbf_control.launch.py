#!/usr/bin/env python

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_dir = get_package_share_directory('balt_go_pd')
    resource_folder = os.path.expanduser('~/px4_ws/src/balt_go_pd/resource')
    return LaunchDescription([
        Node(
            package='balt_go_pd',
            namespace='balt_go_pd',
            executable='visualizer',
            name='visualizer'
        ),
        Node(
            package='balt_go_pd',
            namespace='balt_go_pd',
            executable='Controller_module',
            name='Controller_module'
        ),
        Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', [os.path.join(package_dir, 'visualize.rviz')]]
        )
        # Node(
        #     package='plotjuggler',
        #     namespace='plotjuggler_with_layout',
        #     executable='plotjuggler',
        #     name='plotjuggler',
        #     arguments=['--layout', [os.path.join(resource_folder, 'Pos_z_act.xml')]]
        # )        
    ])
