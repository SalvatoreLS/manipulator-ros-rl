from setuptools import find_packages, setup

package_name = 'keyboard_movement'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='salvatore',
    maintainer_email='losardosalvatorejr@gmail.com',
    description='This ROS2 package handles the movement of the robotic manipulator through keyboard inputs.',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'move_keyboard = keyboard_movement.move_keyboard:main',
        ],
    },
)
