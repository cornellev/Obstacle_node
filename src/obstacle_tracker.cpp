// src/obstacle_tracker.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include "cev_msgs/msg/obstacles.hpp"
#include "obstacle/msg/obstacle_array.hpp"
#include "icp/icp.h"
#include "icp/geo.h"
#include "icp/driver.h"

#include <fstream>
#include <iostream>
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

struct Edge {
    std::tuple<int, int> edge;
    float weight;
};

struct Box {
  float cx, cy;
  float vx, vy;
  float length, width;
  float yaw, yaw_rate;
  int cid;
};

struct OBB {
  Eigen::Vector3d center;
  Eigen::Vector3d extent;
  float yaw;
};

struct Transform {
  std::tuple<float, float> vel;
  float yaw_rate;
};


class ObstacleTracker : public rclcpp::Node {
public:
  ObstacleTracker()
  : Node("obstacle_tracker")
  {
    bev_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/bev_obstacles", 10);
    match_pub_ = this->create_publisher<cev_msgs::msg::Obstacles>("/rslidar_matches", 10);
    prev_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/prev", 10);
    curr_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/curr", 10);
    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("obstacle_markers", 10);
    // nearest neighbor association method -> MHT
    // pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    //   "/rslidar_clusters", 10,
    //   std::bind(&ObstacleTracker::pcCallback, this, std::placeholders::_1));

    obs_sub_ = this->create_subscription<cev_msgs::msg::Obstacles>(
      "/rslidar_obstacles", 10,
      std::bind(&ObstacleTracker::obsCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "ObstacleTracker started - waiting for PointCloud2 on 'input_points'");
  }

private:
  void pcCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // rclcpp::Time curr_stamp(msg->header.stamp);
    // rclcpp::Time prev_stamp(msg_prev_->header.stamp);

    // rclcpp::Duration dt_sec = curr_stamp - prev_stamp;
    // dt = dt_sec.seconds();

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

    if (C_prev_.size() * C.size() != 0) {
      prev_pub_->publish(*msg_prev_);
      curr_pub_->publish(*msg);
    }
    msg_prev_ = msg;
    C_prev_ = C;
  }
  

  icp::PointCloud<icp::ThreeD> pc_to_icp_pc(sensor_msgs::msg::PointCloud2 c) {
    sensor_msgs::PointCloud2ConstIterator<float> iter_x(c, "x");
    sensor_msgs::PointCloud2ConstIterator<float> iter_y(c, "y");
    sensor_msgs::PointCloud2ConstIterator<float> iter_z(c, "z");

    icp::PointCloud<icp::ThreeD> eigen_c(3, c.width * c.height);

    for (size_t i = 0; iter_x != iter_x.end(); ++i, ++iter_x, ++iter_y, ++iter_z) {
        eigen_c(0, i) = *iter_x;
        eigen_c(1, i) = *iter_y;
        eigen_c(2, i) = *iter_z;
    }
    
    return eigen_c;
  }

  inline float wrapAngle(float angle) {
      while (angle > M_PI) angle -= 2.0f * M_PI;
      while (angle < -M_PI) angle += 2.0f * M_PI;
      return angle;
  }

  std::tuple<float, float> computeVelocity(const Box& a, const Box& b) {
    float dx = a.cx - b.cx;
    float dy = a.cy - b.cy;

    return std::make_tuple(dx/0.1f, dy/0.1f);
  }

  float computeYawRate(const Box& a, const Box& b) {
    float d = a.yaw - b.yaw;
    while (d > M_PI) d -= 2.0 * M_PI;
    while (d < -M_PI) d += 2.0 * M_PI;
    return d / 0.1f;
  }

  // some version of iou, which is sometimes bad. 
  float boxLoss(const Box& a, const Box& b) {
    // RCLCPP_INFO(this->get_logger(), "BOX A: cid (%d) => (cx, cy) : (%f, %f), (width, length) : (%f, %f)", a.cid, a.cx, a.cy, a.length, a.width);
    // RCLCPP_INFO(this->get_logger(), "BOX B: cid (%d) => (cx, cy) : (%f, %f), (width, length) : (%f, %f)", b.cid, b.cx, b.cy, b.length, b.width);
    // Tunable normalization constants
    const float d_max = 10.0f;   // meters
    const float l_max = 5.0f;    // meters
    const float w_max = 5.0f;    // meters

    // Weights
    const float w_d = 0.5f;
    const float w_s = 0.3f;
    const float w_y = 0.2f;

    // Position distance
    float dx = a.cx - b.cx;
    float dy = a.cy - b.cy;
    float D_pos = std::sqrt(dx*dx + dy*dy) / d_max;

    // Size difference
    float D_size = std::abs(a.length - b.length) / l_max +
                   std::abs(a.width  - b.width)  / w_max;

    // Yaw difference
    float d_yaw = std::abs(wrapAngle(a.yaw - b.yaw));
    float D_yaw = d_yaw / static_cast<float>(M_PI);
    // RCLCPP_INFO(this->get_logger(), "LOSS IS %f", w_d * D_pos + w_s * D_size + w_y * D_yaw);
    return w_d * D_pos + w_s * D_size + w_y * D_yaw;
  }

  float obbLoss(const OBB& a, const OBB& b)
  {
      // --- Tunable parameters ---
      const double d_max = 0.0;   // max distance for normalization
      const double w_d   = 0.4;    // distance weight
      const double w_iou = 0.4;    // IoU weight
      const double w_v   = 0.2;    // volume weight

      // --- Center distance ---
      Eigen::Vector3d d = a.center - b.center;
      double D_pos = d.norm() / d_max;
      D_pos = std::min(D_pos, 1.0);

      // --- Approximate 3D IoU using centers and extents ---
      double dx = std::abs(a.center.x() - b.center.x());
      double dy = std::abs(a.center.y() - b.center.y());
      double dz = std::abs(a.center.z() - b.center.z());

      double overlap_x = std::max(0.0, (a.extent.x() + b.extent.x())/2 - dx);
      double overlap_y = std::max(0.0, (a.extent.y() + b.extent.y())/2 - dy);
      double overlap_z = std::max(0.0, (a.extent.z() + b.extent.z())/2 - dz);

      double interVol = overlap_x * overlap_y * overlap_z;

      double volA = a.extent.x() * a.extent.y() * a.extent.z();
      double volB = b.extent.x() * b.extent.y() * b.extent.z();
      double unionVol = volA + volB - interVol;

      double approxIoU = (unionVol > 0.0) ? interVol / unionVol : 0.0;
      double D_iou = 1.0 - approxIoU;

      // --- Volume difference term ---
      double D_vol = std::abs(volA - volB) / std::max(volA, volB);
      D_vol = std::min(D_vol, 1.0);

      // --- Final cost ---
      return static_cast<float>(w_d * D_pos + w_iou * D_iou + w_v * D_vol);
  }


  Box predictBox(const Box& prev, Transform transform) {
      Box pred = prev;
      float dt = 0.5f;
      pred.cx += std::get<0>(transform.vel) * dt;
      pred.cy += std::get<1>(transform.vel) * dt;
      pred.yaw += transform.yaw_rate * dt;
      return pred;
  }

  void obsCallback(const cev_msgs::msg::Obstacles::SharedPtr msg) {
    // initialize the bipartite graph
    std::vector<sensor_msgs::msg::PointCloud2> C_PREV = obs_msg_prev_;
    std::vector<sensor_msgs::msg::PointCloud2> C_CURR = msg->obstacles;
    cev_msgs::msg::Obstacles obstacles_msg;
    obstacles_msg.obstacles.reserve(C_CURR.size());

    // map i_curr to j_prev
    std::unordered_map<int32_t, int32_t> predefined_matches;

    // or maybe a mapping from (c_prev, c) -> edge weight, we'll see
    // hungarian algorithm takes in cost (adjacency) matrix where C_CURR is row
    // C_PREV is col
    std::vector<Edge> E;
    int max_size = std::max(C_CURR.size(), C_PREV.size());
    std::vector<std::vector<float>> C(max_size, std::vector<float>(max_size, std::numeric_limits<float>::max()));
    std::vector<std::vector<Transform>> T(
        max_size,
        std::vector<Transform>(max_size, Transform{std::make_tuple(0.0f, 0.0f), 0.0f})
    );

    // bipartite graph construction:
    // C_PREV (t-1), C_CURR (t): sets of clusters at time t-1 and t, C_PREV \intersect C_CURR = \emptyset
    // E: set of edges with elements (c_prev, c) s.t. c_prev \in C_PREV and c \in C_CURR
    // set cluster_ids of c \in C_CURR to be the bipartite matched cluster_ids of c_prev \in C_PREV:
    // i.e. if (c_prev, c) is a match in max bipartite match, then set cluster_id of c_prev to be cluster_id of c
    if (max_size == 0) return;

    auto start = std::chrono::high_resolution_clock::now();

    visualization_msgs::msg::MarkerArray obb_markers;

    // for (int j = 0; j < C_PREV.size(); ++j) {
    //     RCLCPP_INFO(this->get_logger(), "c_prev %d", getClusterId(C_PREV[j]));
    // }

    for (int i = 0; i < C_CURR.size(); ++i) {
        int j = 0;
        // a current cluster
        sensor_msgs::msg::PointCloud2 c = C_CURR[i];
        if (c.width * c.height == 0) continue;
        
        // icp::PointCloud<icp::ThreeD> icp_c = pc_to_icp_pc(c);

        std::vector<cv::Point2f> cv_points_curr;
        cv_points_curr.reserve(c.width * c.height);

        sensor_msgs::PointCloud2ConstIterator<float> iter_x(c, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(c, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(c, "z");

        auto cloud = std::make_shared<open3d::geometry::PointCloud>();

        float z_min = std::numeric_limits<float>::max();
        float z_max = std::numeric_limits<float>::lowest();

        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
            cloud->points_.push_back(Eigen::Vector3d(*iter_x, *iter_y, *iter_z));
            z_min = std::min(z_min, *iter_z);
            z_max = std::max(z_max, *iter_z);
        };

        auto maybe_m = generateOBB(getClusterId(C_CURR[i]), cloud, z_min, z_max);
        if (maybe_m) { 
          visualization_msgs::msg::Marker m = *maybe_m;
          m.header.frame_id = "rslidar";
          obb_markers.markers.push_back(m) ;
        };

        if (cloud && cloud->points_.size() > 3) {
            cloud->RemoveNonFinitePoints();
            if (cloud->points_.size() > 3) {
                auto obb = cloud->GetOrientedBoundingBox();

                Eigen::Vector3d center = obb.center_;

                Eigen::Matrix3d R = obb.R_;
                double yaw = std::atan2(R(1,0), R(0,0));

                Eigen::Vector3d extent = obb.extent_;

                Box box_curr{center.x(), center.y(), 0.0f, 0.0f, extent.x(), extent.z(), yaw, 0.0f, getClusterId(C_CURR[i])};  
                // OBB box_curr{center, extent, yaw};

                for (int j = 0; j < C_PREV.size(); ++j) {
                    sensor_msgs::msg::PointCloud2 c_prev = C_PREV[j];

                    auto cloud_prev = std::make_shared<open3d::geometry::PointCloud>();

                    sensor_msgs::PointCloud2ConstIterator<float> iter_x(c_prev, "x");
                    sensor_msgs::PointCloud2ConstIterator<float> iter_y(c_prev, "y");
                    sensor_msgs::PointCloud2ConstIterator<float> iter_z(c_prev, "z");

                    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z) {
                        cloud_prev->points_.push_back(Eigen::Vector3d(*iter_x, *iter_y, *iter_z));
                    };

                    if (cloud_prev && cloud_prev->points_.size() > 3) {
                      cloud_prev->RemoveNonFinitePoints();
                      if (cloud_prev->points_.size() > 3) {
                          auto obb_prev = cloud_prev->GetOrientedBoundingBox();

                          Eigen::Vector3d center_prev = obb_prev.center_;
                          
                          Eigen::Matrix3d R_prev = obb_prev.R_;
                          double yaw_prev = std::atan2(R(1,0), R(0,0));

                          Eigen::Vector3d extent_prev = obb_prev.extent_;
                          
                          Box box_prev{center_prev.x(), center_prev.y(), 0.0f, 0.0f, extent_prev.x(), extent_prev.z(), yaw_prev, 0.0f, getClusterId(c_prev)};  
                          // OBB box_prev{center_prev, extent_prev, yaw_prev};
                          // THIS IS A VERY BAD FUNCTION THAT IS TURNING FLOATS INTO INTS
                          // if (transform.count(getClusterId(c_prev)) > 0) {
                          //   box_prev = predictBox(box_prev, transform[getClusterId(c_prev)]);
                          // }

                          Edge edge;
                          edge.edge = std::make_tuple(j, i);
                          edge.weight = boxLoss(box_curr, box_prev);
                          // edge.weight = obbLoss(box_curr, box_prev);

                          E.push_back(edge);
                          // try to force some predefined matches
                          C[i][j] = edge.weight > 3.0f ? std::numeric_limits<float>::max() : edge.weight;
                          if (edge.weight < 0.1f) {
                            if (predefined_matches.find(i) == predefined_matches.end() || edge.weight < C[i][predefined_matches[i]]) {
                                predefined_matches[i] = j;
                                // for (int ii = 0; ii < C_CURR.size(); ++ii) {
                                //   if (ii != i) {
                                //     C[i][j] = std::numeric_limits<float>::max();
                                //   }
                                // }
                                // for (int jj = 0; jj < C_PREV.size(); ++jj) {
                                //   if (jj != j) {
                                //     C[i][j] = std::numeric_limits<float>::max();
                                //   }
                                // }
                            }
      
                            // RCLCPP_INFO(this->get_logger(), "BOX A: cid (%d) => (cx, cy) : (%f, %f), (width, length) : (%f, %f)", box_curr.cid, box_curr.cx, box_curr.cy, box_curr.length, box_curr.width);
                            // RCLCPP_INFO(this->get_logger(), "BOX B: cid (%d) => (cx, cy) : (%f, %f), (width, length) : (%f, %f)", box_prev.cid, box_prev.cx, box_prev.cy, box_prev.length, box_prev.width);
                            // RCLCPP_INFO(this->get_logger(), "i_curr (%d) -> j_prev (%d), LOSS IS %f", i, getClusterId(C_PREV[j]), edge.weight);
                          }
                          // T[i][j] = Transform{computeVelocity(box_curr, box_prev), computeYawRate(box_curr, box_prev)};
                      }
                    }
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    RCLCPP_INFO(this->get_logger(), "RAN FOR: %ld ms", duration.count());

    std::vector<int> matchings = hungarian_assignment(C, predefined_matches);

    std::vector<uint8_t> inactive_ids{}; 

    for (int i = 0; i < matchings.size(); ++i) {
      // prev_idx may not be the previous cluster's cluster_id
      // matchings[j_prev] = i_curr;
      int prev_idx = i;
      int curr_idx = matchings[i];

      if (curr_idx != -1 && curr_idx < C_CURR.size() && prev_idx != -1 && prev_idx < C_PREV.size()) {
        // this is a valid matching
        if (prev_idx > max_active_id) {
          max_active_id = prev_idx;
        }
      }

      if ((curr_idx == -1 || curr_idx >= C_CURR.size()) && prev_idx != -1 && prev_idx < C_PREV.size()) {
        uint8_t cid = getClusterId(C_PREV[prev_idx]);
        // if c_prev is not matched to some c_curr:
        // then c_prev --- invalid c_curr
        // add prev_idx to inactive ids, s.t. we can reuse this id if necessary
        inactive_ids.push_back(cid);
      }
    }

    writeClusterIds(C_PREV, C_CURR, matchings, inactive_ids, T, predefined_matches);

    // given cluster_id of past C_PREV, output C_CURR s.t. the cluster_id is consistent 
    obstacles_msg.obstacles = C_CURR;
    match_pub_->publish(obstacles_msg);

    RCLCPP_INFO(this->get_logger(), "done");

    // instead of saving C_CURR, we make a prediction as to where the next matched cluster will be
    // and save that predicition as obs_msg_prev_.

    // naive bayes filter for prediction:
    // via markov assumption, we only need xt-1 for xt
    // i should define a global parameter that maps
    // cluster_id : prev pc, prev box, prev state (xt-1)
  // compute velocity via dx/dt
    marker_pub_->publish(obb_markers);  
    obs_msg_prev_ = C_CURR;
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

  std::optional<visualization_msgs::msg::Marker> generateOBB(int cid, std::shared_ptr<open3d::geometry::PointCloud> cloud, float z_min, float z_max) 
  {
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

  void writeClusterIds(
      const std::vector<sensor_msgs::msg::PointCloud2>& C_PREV,
      std::vector<sensor_msgs::msg::PointCloud2>& C_CURR,
      std::vector<int> matchings, std::vector<uint8_t> inactive_ids,
      std::vector<std::vector<Transform>> T,
      std::unordered_map<int32_t, int32_t> predefined_matches
    ) {

      sensor_msgs::msg::PointCloud2 bev_points;
      bev_points.header.frame_id = "rslidar";
      bev_points.height = 1;

      size_t total_points = 0;
      for (const auto &pts : C_CURR) {
        total_points += pts.width * pts.height;
      }
      
      sensor_msgs::PointCloud2Modifier modifier(bev_points);
      modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
      modifier.resize(total_points);

      sensor_msgs::PointCloud2Iterator<float> out_x(bev_points, "x");
      sensor_msgs::PointCloud2Iterator<float> out_y(bev_points, "y");
      sensor_msgs::PointCloud2Iterator<float> out_z(bev_points, "z");
      sensor_msgs::PointCloud2Iterator<uint8_t> out_r(bev_points, "r");
      sensor_msgs::PointCloud2Iterator<uint8_t> out_g(bev_points, "g");
      sensor_msgs::PointCloud2Iterator<uint8_t> out_b(bev_points, "b");

      RCLCPP_INFO(this->get_logger(), "C_prev count: %d v. C_curr count: %d\n", C_PREV.size(), C_CURR.size());
      for (int i = 0; i < matchings.size(); ++i) {
        // job[j_prev] = i_curr;
          int prev_idx = i;
          int curr_idx = matchings[i];

          if (prev_idx >= 0 && prev_idx < C_PREV.size()) {
            RCLCPP_INFO(this->get_logger(), "match prev_idx [%d], %d -- curr_idx [%d]", prev_idx, getClusterId(C_PREV[prev_idx]), curr_idx);
          }
      }

      RCLCPP_INFO(this->get_logger(), "\n");

      std::vector<int> curr_to_prev(C_CURR.size(), -1);
      for (int prev_idx = 0; prev_idx < matchings.size(); ++prev_idx) {
          int curr_idx = matchings[prev_idx];

          if (curr_idx >= 0 && curr_idx < C_CURR.size()) {
            curr_to_prev[curr_idx] = prev_idx;
          }
      }

      for (int curr_idx = 0; curr_idx < C_CURR.size(); ++curr_idx) {
          // RCLCPP_INFO(this->get_logger(), "BEFORE CLUSTER_ID (%d)", curr_idx);
          // == LOGIC FOR SETTING CLUSTER_ID ==
          int prev_idx = curr_to_prev[curr_idx];
          uint32_t cluster_id;
          if (predefined_matches.find(curr_idx) != predefined_matches.end()) {
            cluster_id = getClusterId(C_PREV[predefined_matches[curr_idx]]);
          } else if (prev_idx >= 0 && prev_idx < C_PREV.size()) {
            cluster_id = getClusterId(C_PREV[prev_idx]);
          } else {
            // if c_curr --- invalid c_prev: then this c_curr is not matched to anything: assign new unique cluster_id
            // this new unique cluster_id is either an inactive id in inactive_ids, or max_active_id 
            if (!inactive_ids.empty()) {
              cluster_id = inactive_ids.back();
              inactive_ids.pop_back();
            } else {
              cluster_id = max_active_id++;
            }
          }
    
          setClusterId(C_CURR[curr_idx], cluster_id);
          // RCLCPP_INFO(this->get_logger(), "RESULT CLUSTER_ID (%d)", cluster_id);

          std::vector<cv::Point2f> cv_points;
          cv_points.reserve(C_CURR[curr_idx].width * C_CURR[curr_idx].height);

          sensor_msgs::PointCloud2ConstIterator<float> in_x(C_CURR[curr_idx], "x");
          sensor_msgs::PointCloud2ConstIterator<float> in_y(C_CURR[curr_idx], "y");
          sensor_msgs::PointCloud2ConstIterator<float> in_z(C_CURR[curr_idx], "z");

          for (; in_x != in_x.end(); ++in_x, ++in_y, ++in_z) {
            cv_points.emplace_back(*in_x, *in_y);

            std_msgs::msg::ColorRGBA color = getColorFromId(cluster_id);

            *out_x = *in_x;
            *out_y = *in_y;
            *out_z = *in_z;
            *out_r = color.r * 255;
            *out_g = color.g * 255;
            *out_b = color.b * 255;
            ++out_x; ++out_y; ++out_z;
            ++out_r; ++out_g; ++out_b;
          }
      } 

      bev_points.width = static_cast<uint32_t>(bev_points.data.size() / bev_points.point_step);
      bev_points.row_step = bev_points.point_step * bev_points.width;

      bev_pub_->publish(bev_points);
  }

  uint32_t getClusterId(const sensor_msgs::msg::PointCloud2& cloud)
  {
      for (const auto& field : cloud.fields) {
          if (field.name == "id") {
              const uint8_t* data_ptr = &cloud.data[field.offset];
              return *reinterpret_cast<const uint32_t*>(data_ptr);
          }
      }
      throw std::runtime_error("id field not found");
  }

  void setClusterId(sensor_msgs::msg::PointCloud2& cloud, uint32_t cluster_id)
  {
      for (auto& field : cloud.fields) {
          if (field.name == "id") {
              for (size_t i = 0; i < cloud.width * cloud.height; ++i) {
                  uint8_t* data_ptr = &cloud.data[i * cloud.point_step + field.offset];
                  *reinterpret_cast<uint32_t*>(data_ptr) = cluster_id;
              }
              return;
          }
      }
      throw std::runtime_error("id field not found");
  }

  std::vector<int> hungarian_assignment(std::vector<std::vector<float>> mat,
                  std::unordered_map<int32_t, int32_t> predefined_matches)  {
    // i is the cluster id of C_PREV, job[i] is the cluster id of C_CURR
    // find perfect matching from C_PREV to C_CURR that minimizes total assignment cost
    // job[j_prev] = i_curr;
    const int J = mat.size(); // cost_matrix.rows() is C_CURR idx
    assert(J > 0);
    const int W = mat[0].size(); // cost_matrix.cols() is C_PREV idx

    std::vector<int> job(W + 1, -1);
    std::vector<float> ys(J, 0.0f);
    std::vector<float> yt(W + 1, 0.0f);
    std::vector<float> answers;
    const float inf = std::numeric_limits<float>::max();
    const float eps = static_cast<float>(1e-9);

    for (int jCur = 0; jCur < J; ++jCur) {  // assign jCur-th job
      int wCur = W;
      job[wCur] = jCur;

      std::vector<float> minTo(W + 1, inf);
      std::vector<int> prev(W + 1, -1);  // previous worker on alternating path
      std::vector<bool> inZ(W + 1, false);     // whether worker is in Z
      
      while (job[wCur] != -1) {    // runs at most jCur + 1 times
        inZ[wCur] = true;
        const int j = job[wCur];
        float delta = inf;
        int wNext = -1;

        for (int w = 0; w < W; ++w) {
            if (!inZ[w]) {
              float new_cost = mat[j][w] - ys[j] - yt[w];
              if (new_cost < minTo[w] - eps) {
                minTo[w] = new_cost;
                prev[w] = wCur;
              }
              if (minTo[w] < delta - eps) {
                delta = minTo[w];
                wNext = w;
              }
            }
        }

        if (wNext == -1) {
            RCLCPP_ERROR(this->get_logger(), "No valid worker found!");
            break;
        }
        // delta will always be nonnegative,
        // except possibly during the first time this loop runs
        // if any entries of C[jCur] are negative
        for (int w = 0; w <= W; ++w) {
            if (inZ[w]) {
                if (job[w] != -1) ys[job[w]] += delta;
                yt[w] -= delta;
            } else {
                minTo[w] -= delta;
            }
        }
        wCur = wNext;
      }
      // update assignments along alternating path
      while (wCur != W) {
          int w = prev[wCur];
          job[wCur] = job[w];
          wCur = w;
      }

      answers.push_back(-yt[W]);
    }

    job.pop_back();
    return job;
  }


  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr prev_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr curr_pub_;
  rclcpp::Publisher<cev_msgs::msg::Obstacles>::SharedPtr match_pub_;
  sensor_msgs::msg::PointCloud2::SharedPtr msg_prev_;
  std::vector<sensor_msgs::msg::PointCloud2> obs_msg_prev_;
  std::unordered_map<int32_t, std::vector<PointXYZCluster>> C_prev_;
  rclcpp::Subscription<cev_msgs::msg::Obstacles>::SharedPtr obs_sub_;
  // rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr bev_pub_;
  int max_active_id = 0;
  // maps cluster_id to a state change
  std::map<uint8_t, Transform> transform;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  // float dt;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ObstacleTracker>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}