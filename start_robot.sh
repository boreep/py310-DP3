#!/usr/bin/env bash

ros2 launch hand_control_cpp right_driver.launch.py  &
sleep 1

ros2 launch l515_pcl_processor l515_launch.py  &

wait