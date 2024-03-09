import os
from glob import glob
from setuptools import setup

package_name = 'balt_go_pd'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/ament_index/resource_index/packages',
            ['resource/' + 'visualize.rviz']),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*launch.[pxy][yma]*')),
        (os.path.join('share', package_name), glob('resource/*rviz'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='balt',
    maintainer_email='kejty5345@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'Controller_lib = balt_go_pd.Controller_lib:main', 
            'attacker_module = balt_go_pd.attacker_module:main',
            'Controller_module = balt_go_pd.Controller_module:main',
            'trajectory_node = balt_go_pd.trajectory_node:main',
            'visualizer = balt_go_pd.visualizer:main',
        ],
    },
)
