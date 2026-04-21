#!/bin/bash

source /opt/ros/humble/setup.bash
source /home/dev/ws/src/install/setup.bash

echo "Building packages..."
cd /home/dev/ws/src
colcon build --packages-select cev_msgs obstacle

echo "Sourcing workspace..."
source /home/dev/ws/src/install/setup.bash

echo "Starting TF publishers and costmap..."

# Run each in background
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map base_link &
TF1_PID=$!

ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link rslidar &
TF2_PID=$!

# Small delay to let TF come up before costmap starts
sleep 1

ros2 run obstacle costmap_node &
COSTMAP_PID=$!

echo "Running. PIDs: tf1=$TF1_PID tf2=$TF2_PID costmap=$COSTMAP_PID"
echo "Press Ctrl+C to stop all."

# Wait and kill all on Ctrl+C
trap "kill $TF1_PID $TF2_PID $COSTMAP_PID; exit" SIGINT SIGTERM
wait