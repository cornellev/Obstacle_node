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

class ObstacleTracker : public rclcpp::Node {
public:
  ObstacleTracker()
  : Node("obstacle_tracker")
  {
    bev_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/bev_obstacles", 10);
    match_pub_ = this->create_publisher<cev_msgs::msg::Obstacles>("rslidar_matches", 10);
    prev_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/prev", 10);
    curr_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/curr", 10);
    // nearest neighbor association method -> MHT
    pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/rslidar_clusters", 10,
      std::bind(&ObstacleTracker::pcCallback, this, std::placeholders::_1));

    obs_sub_ = this->create_subscription<cev_msgs::msg::Obstacles>(
      "/rslidar_obstacles", 10,
      std::bind(&ObstacleTracker::obsCallback, this, std::placeholders::_1));

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

    RCLCPP_INFO(this->get_logger(), "C_prev count: %d v. C_curr count: %d", C_prev_.size(), C.size());
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

  struct Box {
      float cx, cy;
      float length, width;
      float yaw;  // radians
  };

  inline float wrapAngle(float angle) {
      while (angle > M_PI) angle -= 2.0f * M_PI;
      while (angle < -M_PI) angle += 2.0f * M_PI;
      return angle;
  }

  float boxLoss(const Box& a, const Box& b) {
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

    return w_d * D_pos + w_s * D_size + w_y * D_yaw;
  }

  void obsCallback(const cev_msgs::msg::Obstacles::SharedPtr msg) {
    // initialize the bipartite graph
    std::vector<sensor_msgs::msg::PointCloud2> C_PREV = obs_msg_prev_;
    std::vector<sensor_msgs::msg::PointCloud2> C_CURR = msg->obstacles;
    cev_msgs::msg::Obstacles obstacles_msg;
    obstacles_msg.obstacles.reserve(C_CURR.size());

    // or maybe a mapping from (c_prev, c) -> edge weight, we'll see
    // hungarian algorithm takes in cost (adjacency) matrix where C_CURR is row
    // C_PREV is col
    std::vector<Edge> E;
    int max_size = std::max(C_CURR.size(), C_PREV.size());
    std::vector<std::vector<float>> C(max_size, std::vector<float>(max_size, std::numeric_limits<float>::max()));
    // Eigen::MatrixXf cost_matrix(C_CURR.size(), C_PREV.size());

    // bipartite graph construction:
    // C_PREV (t-1), C_CURR (t): sets of clusters at time t-1 and t, C_PREV \intersect C_CURR = \emptyset
    // E: set of edges with elements (c_prev, c) s.t. c_prev \in C_PREV and c \in C_CURR
    // set cluster_ids of c \in C_CURR to be the bipartite matched cluster_ids of c_prev \in C_PREV:
    // i.e. if (c_prev, c) is a match in max bipartite match, then set cluster_id of c_prev to be cluster_id of c
    if (max_size == 0) return;

    auto start = std::chrono::high_resolution_clock::now();
    std::mutex mtx;
    std::vector<std::future<void>> futures;

    for (int i = 0; i < C_CURR.size(); ++i) {
        sensor_msgs::msg::PointCloud2 c = C_CURR[i];
        if (c.width * c.height == 0) continue;
        
        icp::PointCloud<icp::ThreeD> icp_c = pc_to_icp_pc(c);

        std::vector<cv::Point2f> cv_points_curr;
        cv_points_curr.reserve(c.width * c.height);

        sensor_msgs::PointCloud2ConstIterator<float> in_x(c, "x");
        sensor_msgs::PointCloud2ConstIterator<float> in_y(c, "y");
        sensor_msgs::PointCloud2ConstIterator<float> in_z(c, "z");

        for (; in_x != in_x.end(); ++in_x, ++in_y, ++in_z) {
          cv_points_curr.emplace_back(*in_x, *in_y);
        }
        cv::RotatedRect rect = cv::minAreaRect(cv_points_curr);

        Box box_curr;
        box_curr.cx = rect.center.x;
        box_curr.cy = rect.center.y;
        box_curr.width = rect.size.width;
        box_curr.length = rect.size.height;
        float angle_curr = rect.angle;
        box_curr.yaw = angle_curr * M_PI / 180.0f;
        
        for (int j = 0; j < C_PREV.size(); ++j) {

          sensor_msgs::msg::PointCloud2 c_prev = C_PREV[j];

          std::vector<cv::Point2f> cv_points_prev;
          cv_points_prev.reserve(c.width * c.height);

          sensor_msgs::PointCloud2ConstIterator<float> in_x(c, "x");
          sensor_msgs::PointCloud2ConstIterator<float> in_y(c, "y");
          sensor_msgs::PointCloud2ConstIterator<float> in_z(c, "z");

          for (; in_x != in_x.end(); ++in_x, ++in_y, ++in_z) {
            cv_points_prev.emplace_back(*in_x, *in_y);
          }
          cv::RotatedRect rect = cv::minAreaRect(cv_points_prev);

          Box box_prev;
          box_prev.cx = rect.center.x;
          box_prev.cy = rect.center.y;
          box_prev.width = rect.size.width;
          box_prev.length = rect.size.height;
          float angle_prev = rect.angle;
          box_prev.yaw = angle_prev * M_PI / 180.0f;

          Edge edge;
          edge.edge = std::make_tuple(j, i);
          // edge.weight = result.cost;
          edge.weight = boxLoss(box_curr, box_prev);

          E.push_back(edge);
          C[i][j] = edge.weight;

            // futures.push_back(std::async(std::launch::async, [&, i, j, icp_c]() {
            //     sensor_msgs::msg::PointCloud2 c_prev = C_PREV[j];

            //     if (c_prev.width * c_prev.height == 0) return;
            //     icp::PointCloud<icp::ThreeD> icp_c_prev = pc_to_icp_pc(c_prev);
                
            //     std::unique_ptr<icp::ICP3> icp = icp::ICP3::from_method("vanilla", icp::Config()).value();
            //     icp::ICPDriver driver(std::move(icp));

            //     driver.set_max_iterations(1);
            //     driver.set_transform_tolerance(0.1 * M_PI / 180, 0.1);
                
            //     auto result = driver.converge(icp_c_prev, icp_c, icp::RBTransform3::Identity());
                
            //     Edge edge;
            //     edge.edge = std::make_tuple(j, i);
            //     edge.weight = result.cost;
                
            //     // Need mutex protection for shared data structures
            //     {
            //         std::lock_guard<std::mutex> lock(mtx);
            //         E.push_back(edge);
            //         C[i][j] = result.cost;
            //     }
            // }));
        }
    }

    // Wait for all tasks to complete
    // for (auto& f : futures) {
    //     f.get();
    // }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    RCLCPP_INFO(this->get_logger(), "RAN FOR: %ld ms", duration.count());

    std::vector<int> matchings = hungarian_assignment(C);
    // auto markers = outputMatching(C_PREV, C_CURR, matchings);
    writeClusterIds(C_PREV, C_CURR, matchings);

    // given cluster_id of past C_PREV, output C_CURR s.t. the cluster_id is consistent 
    obstacles_msg.obstacles = C_CURR;
    match_pub_->publish(obstacles_msg);

    RCLCPP_INFO(this->get_logger(), "done");
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

  void writeClusterIds(
      const std::vector<sensor_msgs::msg::PointCloud2>& C_PREV,
      std::vector<sensor_msgs::msg::PointCloud2>& C_CURR,
      std::vector<int> matchings) {

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

      RCLCPP_INFO(this->get_logger(), "C_prev count: %d v. C_curr count: %d", C_PREV.size(), C_CURR.size());
      for (int i = 0; i < matchings.size(); ++i) {
          int prev_idx = i;
          int curr_idx = matchings[i];
          
          // if prev_idx or curr_idx is -1, then this is an invalid match
          if (curr_idx == -1 || curr_idx >= C_CURR.size() || prev_idx == -1 || prev_idx >= C_PREV.size()) {
              continue;
          }

          // after setting cluster_id, check the new cluster_id of c_curr
          uint32_t cluster_id = getClusterId(C_PREV[prev_idx]);
          setClusterId(C_CURR[curr_idx], cluster_id);

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

  std::vector<int> hungarian_assignment(std::vector<std::vector<float>> mat)  {
    // i is the cluster id of C_PREV, job[i] is the cluster id of C_CURR
    // find perfect matching from C_PREV to C_CURR that minimizes total assignment cost
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
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr bev_pub_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ObstacleTracker>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
