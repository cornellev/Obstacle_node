// src/obstacle_tracker.cpp
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

class ObstacleTracker : public rclcpp::Node {
public:
  ObstacleTracker()
  : Node("obstacle_tracker")
  {
    // nearest neighbor association method -> MHT
    pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/rslidar_clusters", 10,
      std::bind(&ObstacleTracker::pcCallback, this, std::placeholders::_1));

    // maybe implement joint probability data association (JPDA) also for clustering

    RCLCPP_INFO(this->get_logger(), "ObstacleTracker started - waiting for PointCloud2 on 'input_points'");
  }

private:
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

    // get transformation matrix from 3d icp between *msg and *msg_prev_;

    std::unordered_map<int32_t, std::vector<PointXYZCluster>> C;
    size_t total_points = 0;
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_id) {
      PointXYZCluster c;
      c.x = *iter_x;
      c.y = *iter_y;
      c.z = *iter_z;
      c.cluster_id = *iter_id;
      C[c.cluster_id].push_back(c);
      ++total_points;
    }

    // after all the cluster matching and whatever is done, update C_prev
    // we can also use this to determine static v. dynamic obstacles

    msg_prev_ = msg;
    C_prev_ = C;
  }

  void multi_hypothesis_tracking(std::unordered_map<int32_t, std::vector<PointXYZCluster>> C_prev, 
                                 std::unordered_map<int32_t, std::vector<PointXYZCluster>> C) {
    // takes in ??? vector<clusters> or the entire point cloud idrk
    // tbh for both MHT and max bipartite matching, might need to consider all
    // possible matches
    // or like icp_cost_bipartite_matching is a way to generate all possible 
    // association hypothesis -> run SHT -> reduce number of hypotheses
  }

  void icp_cost_bipartite_matching(std::unordered_map<int32_t, std::vector<PointXYZCluster>> C_prev, 
                                   std::unordered_map<int32_t, std::vector<PointXYZCluster>> C,
                                   double cost_threshold)  {
    // a way to implement MHT for clusters

    // matching on a bipartite graph where red vertices correspond to previous
    // clusters at timestamp t-1 and black vertices correspond to current
    // clusters at timestamp t
    // we construct a fully connected (WLOG, each red vertex connected to all 
    // black vertices) bipartite graph where the weight of edge connecting each
    // red-black vertex pair is defined by the quantitative ICP overlaying cost
    // thing between the two clusters (maybe like determined by the returned
    // transformation matrix [R | t])
    // somehow think about this?
    // unmatched black (t) vertices are maybe categorized as new obstacles?
    // unique id for clusters so we can do more long term obstacle tracking for
    // the same cluster

    // complexity reduction / pruning: 
    // there should be a cost_threshold -> don't even create an edge if 
    // weight < cost_threshold
    // cost func should consider diff in shape, position, orientation, # points -> ICP

  }

  sensor_msgs::msg::PointCloud2::SharedPtr msg_prev_;
  std::unordered_map<int32_t, std::vector<PointXYZCluster>> C_prev_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ObstacleTracker>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
