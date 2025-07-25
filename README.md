# human_trajectory_predictor

This ROS2 package for real-time human detection from LiDAR data and future position prediction using Kalman filter.

## lidar_human_detection
"This packgae utilizes the Velodyne LiDAR package."
 
## Environment
ROS2 Humble
Ubuntu 22.04

## How to use
sudo apt install ros-humble-velodyne ros-humble-pcl-ros ros-humble-visualization-msgs

pip install scikit-learn

colcon build

source install/setup.bash

ros2 launch lidar_human_detection human_detection_launch.py

rviz2 -> frame_id : velodyne, ADD Point cloud2 / marker_array
