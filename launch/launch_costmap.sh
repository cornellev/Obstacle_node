#!/bin/bash

source /opt/ros/humble/setup.bash
source /home/dev/ws/src/install/setup.bash

echo "Checking and building packages if needed..."

cd /home/dev/ws/src

build_if_missing () {
    PKG=$1

    if ros2 pkg list | grep -qx "$PKG"; then
        echo "Package '$PKG' already available, skipping build."
    else
        echo "Package '$PKG' not found, building..."
        colcon build --packages-select "$PKG"
    fi
}

build_if_missing cev_msgs
build_if_missing obstacle

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