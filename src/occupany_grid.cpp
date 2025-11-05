// src/occupancy_grid.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include "cev_msgs/msg/obstacles.hpp"
#include "obstacle/msg/obstacle_array.hpp"

#include <unordered_map>
#include <vector>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

using obstacle::msg::ObstacleArray;

struct PointXYZCluster {
  float x;
  float y;
  float z;
  int32_t cluster_id;
};

class OccupancyGrid : public rclcpp::Node {
public:
  OccupancyGrid()
  : Node("occupancy_grid")
  {
    obs_sub_ = this->create_subscription<cev_msgs::msg::Obstacles>(
      "/rslidar_obstacles", 10,
      std::bind(&OccupancyGrid::obstaclesCallback, this, std::placeholders::_1));

    pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "input_points", 10,
      std::bind(&OccupancyGrid::pcCallback, this, std::placeholders::_1));

    bev_pub_ = this->create_publisher<geometry_msgs::msg::Point32[]>("bev_obstacles", 10);

    RCLCPP_INFO(this->get_logger(), "OccupancyGrid started - waiting for PointCloud2 on 'input_points'");
  }

private:
  void obstaclesCallback(const cev_msgs::msg::Obstacles::SharedPtr msg)
  {
    RCLCPP_INFO(this->get_logger(), "Received Obstacles message with %zu clouds", msg->obstacles.size());
    
    sensor_msgs::msg::PointCloud2 merged_cloud = msg->obstacles[0];

    for (size_t i = 1; i < msg->obstacles.size(); ++i)
    {
        const auto &cloud = msg->obstacles[i];

        pcl::concatenatePointCloud(merged_cloud, cloud, merged_cloud);
    }

    auto cloud_ptr = std::make_shared<sensor_msgs::msg::PointCloud2>(merged_cloud);
    pcCallback(cloud_ptr);
  }

  void pcCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // check required fields
    bool has_x=false, has_y=false, has_z=false, has_id=false;
    for (const auto &f : msg->fields) {
      if (f.name == "x") has_x = true;
      if (f.name == "y") has_y = true;
      if (f.name == "z") has_z = true;
      if (f.name == "id" || f.name == "cluster_id") has_id = true;
    }
    if (!(has_x && has_y && has_z && has_id)) {
      RCLCPP_ERROR(this->get_logger(),
                   "PointCloud2 missing required fields (need x,y,z and id/cluster_id). Skipping this message.");
      return;
    }

    sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
    sensor_msgs::PointCloud2ConstIterator<int32_t> iter_id(*msg, "id");

    std::unordered_map<int32_t, std::vector<PointXYZCluster>> clusters;
    size_t total_points = 0;
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_id) {
      PointXYZCluster p;
      p.x = *iter_x;
      p.y = *iter_y;
      p.z = *iter_z;
      p.cluster_id = *iter_id;
      clusters[p.cluster_id].push_back(p);
      ++total_points;
    }

    // z-axis filtering
    const float z_min_allowed = 0.0f;   
    const float z_max_allowed = 2.5f;   

    std::unordered_map<int32_t, std::vector<PointXYZCluster>> filtered_clusters;

    for (const auto &kv : clusters) {
      int cid = kv.first;
      const auto &pts = kv.second;

      std::vector<PointXYZCluster> kept;
      float z_min = std::numeric_limits<float>::max();
      float z_max = std::numeric_limits<float>::lowest();

      for (const auto &p : pts) {
        z_min = std::min(z_min, p.z);
        z_max = std::max(z_max, p.z);
        if (p.z >= z_min_allowed && p.z <= z_max_allowed) {
          kept.push_back(p);
        }
      }

      if (!kept.empty()) {
        filtered_clusters[cid] = kept;
      } 

    }

    // === BEV projection + OBB ===
    ObstacleArray out;
    out.header = msg->header;

    for (const auto &kv : filtered_clusters) {
        int cid = kv.first;
        const auto &pts = kv.second;

        if (pts.size() < 3) {
            RCLCPP_WARN(this->get_logger(),
                        "Cluster %d has only %zu points, skipping OBB",
                        cid, pts.size());
            continue;
        }

        std::vector<cv::Point2f> cv_points;
        cv_points.reserve(pts.size());
        float z_min = std::numeric_limits<float>::max();
        float z_max = std::numeric_limits<float>::lowest();
        for (const auto &p : pts) {
            cv_points.emplace_back(p.x, p.y);
            z_min = std::min(z_min, p.z);
            z_max = std::max(z_max, p.z);
        }
    }

    // === publish ObstacleArray ===
    obs_pub_->publish(out);
  }

  rclcpp::Subscription<cev_msgs::msg::Obstacles>::SharedPtr obs_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
  rclcpp::Publisher<geometry_msgs::msg::PointCloud32[]>::SharedPtr bev_pub_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OccupancyGrid>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
