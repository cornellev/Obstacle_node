// src/obstacle_node.cpp
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
#include <open3d/Open3D.h>
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

class ObstacleNode : public rclcpp::Node {
public:
  ObstacleNode()
  : Node("obstacle_node")
  {
    obs_sub_ = this->create_subscription<cev_msgs::msg::Obstacles>(
      "/rslidar_obstacles", 10,
      std::bind(&ObstacleNode::obstaclesCallback, this, std::placeholders::_1));

    pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/rslidar_clusters", 10,
      std::bind(&ObstacleNode::pcCallback, this, std::placeholders::_1));

    obs_pub_ = this->create_publisher<ObstacleArray>("obstacles", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_markers", 10);

    RCLCPP_INFO(this->get_logger(), "ObstacleNode started - waiting for PointCloud2 on 'input_points'");
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

    // RCLCPP_INFO(this->get_logger(),
    //             "Received PointCloud2: points=%zu clusters=%zu",
    //             total_points, clusters.size());

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
        kept.push_back(p);
      }

      if (!kept.empty()) {
        filtered_clusters[cid] = kept;
      } 
    }


    // === BEV projection + OBB ===
    ObstacleArray out;
    out.header = msg->header;
    visualization_msgs::msg::MarkerArray obb_markers;

    for (const auto &kv : filtered_clusters) {
        int cid = kv.first;
        const auto &pts = kv.second;

        if (pts.size() < 3) {
          RCLCPP_WARN(this->get_logger(),
                      "Cluster %d has only %zu points, skipping OBB",
                      cid, pts.size());
          continue;
        } else {
          auto maybe_m = generateOBB(cid, pts);
          if (maybe_m) { 
            visualization_msgs::msg::Marker m = *maybe_m;
            m.header = msg->header;
            m.header.frame_id = "rslidar";
            obb_markers.markers.push_back(m) ;
          };
        }
    }

    marker_pub_->publish(obb_markers);
    obs_pub_->publish(out);
  }

  std_msgs::msg::ColorRGBA getColorFromId(int cluster_id)
  {
    std_msgs::msg::ColorRGBA color;
    uint32_t hash = static_cast<uint32_t>(cluster_id * 2654435761 % 4294967296); // Knuth's multiplicative hash
    color.r = ((hash & 0xFF0000) >> 16) / 255.0f;
    color.g = ((hash & 0x00FF00) >> 8) / 255.0f;
    color.b = (hash & 0x0000FF) / 255.0f;
    color.a = 0.5f;

    return color;
  }

  std::optional<visualization_msgs::msg::Marker> generateOBB(int cid, std::vector<PointXYZCluster> pts) 
  {
    auto cloud = std::make_shared<open3d::geometry::PointCloud>();

    float z_min = std::numeric_limits<float>::max();
    float z_max = std::numeric_limits<float>::lowest();

    for (const auto &p : pts) {
        cloud->points_.push_back(Eigen::Vector3d(p.x, p.y, p.z));
        z_min = std::min(z_min, p.z);
        z_max = std::max(z_max, p.z);
    }

    if (cloud && cloud->points_.size() > 3) {
        cloud->RemoveNonFinitePoints();
        if (cloud->points_.size() > 3) {
            auto obb = cloud->GetOrientedBoundingBox();

            Eigen::Vector3d center = obb.center_;

            Eigen::Matrix3d R = obb.R_;
            double yaw = std::atan2(R(1,0), R(0,0));
            Eigen::AngleAxisd yaw_rot(yaw, Eigen::Vector3d::UnitZ());
            Eigen::Quaterniond q(yaw_rot);
            q.normalize();

            Eigen::Vector3d extent = obb.extent_;

            visualization_msgs::msg::Marker m;
            m.ns = "obstacle";
            m.id = cid;
            m.type = visualization_msgs::msg::Marker::CUBE;
            m.action = visualization_msgs::msg::Marker::ADD;

            // we will describe an OBB with: center, orientation, scale
            // centroid
            m.pose.position.x = center.x();
            m.pose.position.y = center.y();
            m.pose.position.z = (z_max + z_min) / 2;

            m.pose.orientation.x = q.x();
            m.pose.orientation.y = q.y();
            m.pose.orientation.z = q.z();
            m.pose.orientation.w = q.w();

            m.scale.x = extent.x();
            m.scale.y = extent.z();
            m.scale.z = z_max - z_min;

            m.color = getColorFromId(m.id);
            return m;
        }
    }
    return std::nullopt;
  }

  rclcpp::Subscription<cev_msgs::msg::Obstacles>::SharedPtr obs_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
  rclcpp::Publisher<ObstacleArray>::SharedPtr obs_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

#include "visualization_msgs/msg/marker_array.hpp"

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ObstacleNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}