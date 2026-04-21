// costmap_node.cpp
#include "rclcpp/rclcpp.hpp"
#include "nav2_costmap_2d/costmap_2d_ros.hpp"
#include "nav2_costmap_2d/costmap_2d.hpp"

class CostmapNode : public rclcpp::Node
{
public:
  CostmapNode() : Node("costmap_node")
  {
    rclcpp::NodeOptions costmap_options;
    costmap_options.parameter_overrides({
      {"global_frame",       rclcpp::ParameterValue(std::string("map"))},
      {"robot_base_frame",   rclcpp::ParameterValue(std::string("base_link"))},
      {"update_frequency",   rclcpp::ParameterValue(10.0)},
      {"publish_frequency",  rclcpp::ParameterValue(5.0)},
      {"resolution",         rclcpp::ParameterValue(0.05)},
      {"width",              rclcpp::ParameterValue(50)},
      {"height",             rclcpp::ParameterValue(50)},
      {"robot_radius",       rclcpp::ParameterValue(0.3)},
      {"rolling_window",     rclcpp::ParameterValue(false)},
      {"plugins",            rclcpp::ParameterValue(std::vector<std::string>{"obstacle_layer", "inflation_layer"})},

      {"origin_x",   rclcpp::ParameterValue(-25.0)},
      {"origin_y",   rclcpp::ParameterValue(-25.0)},

      {"obstacle_layer.plugin",                                rclcpp::ParameterValue(std::string("nav2_costmap_2d::ObstacleLayer"))},
      {"obstacle_layer.enabled",                               rclcpp::ParameterValue(true)},
      {"obstacle_layer.observation_sources",                   rclcpp::ParameterValue(std::string("pointcloud_sensor"))},
      {"obstacle_layer.pointcloud_sensor.topic",               rclcpp::ParameterValue(std::string("/rslidar_points"))},
      {"obstacle_layer.pointcloud_sensor.data_type",           rclcpp::ParameterValue(std::string("PointCloud2"))},
      {"obstacle_layer.pointcloud_sensor.marking",             rclcpp::ParameterValue(true)},
      {"obstacle_layer.pointcloud_sensor.clearing",            rclcpp::ParameterValue(true)},
      {"obstacle_layer.pointcloud_sensor.min_obstacle_height", rclcpp::ParameterValue(-0.2)},
      {"obstacle_layer.pointcloud_sensor.max_obstacle_height", rclcpp::ParameterValue(2.0)},
      {"obstacle_layer.pointcloud_sensor.obstacle_max_range",  rclcpp::ParameterValue(15.0)},
      {"obstacle_layer.pointcloud_sensor.raytrace_max_range",  rclcpp::ParameterValue(20.0)},

      {"inflation_layer.plugin",               rclcpp::ParameterValue(std::string("nav2_costmap_2d::InflationLayer"))},
      {"inflation_layer.enabled",              rclcpp::ParameterValue(true)},
      {"inflation_layer.inflation_radius",     rclcpp::ParameterValue(0.55)},
      {"inflation_layer.cost_scaling_factor",  rclcpp::ParameterValue(10.0)},
    });

    // Humble only has 3-arg constructor: (name, namespace, param_namespace)
    // Use NodeOptions constructor instead — it's the standalone node form
    costmap_ros_ = std::make_shared<nav2_costmap_2d::Costmap2DROS>(costmap_options);

    costmap_ros_->configure();
    costmap_ros_->activate();

    RCLCPP_INFO(get_logger(), "Costmap running.");

    timer_ = create_wall_timer(
      std::chrono::seconds(1),
      std::bind(&CostmapNode::printStats, this));
  }

  ~CostmapNode()
  {
    costmap_ros_->deactivate();
    costmap_ros_->cleanup();
  }

private:
  void printStats()
  {
    nav2_costmap_2d::Costmap2D * costmap = costmap_ros_->getCostmap();
    std::unique_lock<nav2_costmap_2d::Costmap2D::mutex_t> lock(*(costmap->getMutex()));

    unsigned int width  = costmap->getSizeInCellsX();
    unsigned int height = costmap->getSizeInCellsY();

    int free = 0, occupied = 0, unknown = 0;
    for (unsigned int y = 0; y < height; ++y) {
      for (unsigned int x = 0; x < width; ++x) {
        unsigned char c = costmap->getCost(x, y);
        if      (c == nav2_costmap_2d::FREE_SPACE)       free++;
        else if (c == nav2_costmap_2d::LETHAL_OBSTACLE)  occupied++;
        else if (c == nav2_costmap_2d::NO_INFORMATION)   unknown++;
      }
    }
    RCLCPP_INFO(get_logger(), "Costmap %dx%d | free=%d occupied=%d unknown=%d",
      width, height, free, occupied, unknown);
  }

  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<CostmapNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
