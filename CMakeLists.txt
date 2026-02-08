cmake_minimum_required(VERSION 3.8)
project(obstacle)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cev_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Open3D REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Obstacle.msg"
  "msg/ObstacleArray.msg"
  "msg/Box.msg"
  "msg/Boxes.msg"
  DEPENDENCIES std_msgs geometry_msgs cev_msgs nav_msgs
)

add_executable(obstacle_node src/obstacle_node.cpp)
add_executable(occupancy_grid src/occupancy_grid.cpp)
add_executable(obstacle_tracker src/obstacle_tracker.cpp)

ament_target_dependencies(
  obstacle_node 
  rclcpp 
  cev_msgs
  sensor_msgs 
  std_msgs 
  pcl_conversions
  pcl_ros
  geometry_msgs
  tf2
  tf2_geometry_msgs
  visualization_msgs
)

ament_target_dependencies(
  occupancy_grid
  rclcpp 
  cev_msgs
  sensor_msgs 
  std_msgs
  nav_msgs
  pcl_conversions
  pcl_ros
  geometry_msgs
  tf2
  tf2_geometry_msgs
  visualization_msgs
)

ament_target_dependencies(
  obstacle_tracker
  rclcpp 
  cev_msgs
  sensor_msgs 
  std_msgs
  nav_msgs
  pcl_conversions
  pcl_ros
  geometry_msgs
  tf2
  tf2_geometry_msgs
  visualization_msgs
)

set(CEV_ICP_INCLUDE lib/cev_icp/include)
set(CEV_ICP_LIB lib/cev_icp/lib)

include_directories(${CEV_ICP_INCLUDE})
link_directories(${CEV_ICP_LIB})
include_directories(
  ${PCL_INCLUDE_DIRS}
)

target_link_libraries(obstacle_node
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  Open3D::Open3D
)

target_link_libraries(occupancy_grid
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
)

target_link_libraries(obstacle_tracker
  ${OpenCV_LIBS}
  ${PCL_LIBRARIES}
  cevicp
  Open3D::Open3D
)

ament_export_dependencies(rosidl_default_runtime)

rosidl_target_interfaces(obstacle_node
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

rosidl_target_interfaces(occupancy_grid
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

rosidl_target_interfaces(obstacle_tracker
  ${PROJECT_NAME} "rosidl_typesupport_cpp")

install(TARGETS
  obstacle_node
  occupancy_grid
  obstacle_tracker
  DESTINATION lib/${PROJECT_NAME})


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
