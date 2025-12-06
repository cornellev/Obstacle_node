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
    match_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("rslidar_matches", 10);
    prev_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/prev", 10);
    curr_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/curr", 10);
    // nearest neighbor association method -> MHT
    pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/rslidar_clusters", 10,
      std::bind(&ObstacleTracker::pcCallback, this, std::placeholders::_1));

    obs_sub_ = this->create_subscription<cev_msgs::msg::Obstacles>(
      "/rslidar_obstacles", 10,
      std::bind(&ObstacleTracker::obsCallback, this, std::placeholders::_1));

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

  void obsCallback(const cev_msgs::msg::Obstacles::SharedPtr msg) {
    // initialize the bipartite graph
    std::vector<sensor_msgs::msg::PointCloud2> C_PREV = obs_msg_prev_;
    std::vector<sensor_msgs::msg::PointCloud2> C_CURR = msg->obstacles;

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
    // std::vector<std::future<void>> futures;

    // for (int i = 0; i < C_CURR.size(); ++i) {
    //     sensor_msgs::msg::PointCloud2 c = C_CURR[i];
    //     if (c.width * c.height == 0) continue;

    //     auto start = std::chrono::high_resolution_clock::now();
    //     icp::PointCloud<icp::ThreeD> icp_c = pc_to_icp_pc(c);

    //     for (int j = 0; j < C_PREV.size(); ++j)
    //     {
    //         sensor_msgs::msg::PointCloud2 c_prev = C_PREV[j];
    //         icp::PointCloud<icp::ThreeD> icp_c_prev = pc_to_icp_pc(c_prev);

    //         std::unique_ptr<icp::ICP3> icp = icp::ICP3::from_method("vanilla", icp::Config()).value();
    //         icp::ICPDriver driver(std::move(icp));

    //         driver.set_max_iterations(1);
    //         driver.set_transform_tolerance(0.1 * M_PI / 180, 0.1);
    //         auto result = driver.converge(icp_c_prev, icp_c, icp::RBTransform3::Identity());

    //         // check whether translation between the two clusters are within max_radius (which we can define dynamically by past cluster's velocity)
    //         Edge edge;
    //         edge.edge = std::make_tuple(j, i);
    //         edge.weight = result.cost;

    //         E.push_back(edge);
    //         C[i][j] = result.cost;
    //         // cost_matrix(i, j) = result.cost;
    //     }
    // }

    // for (auto &f : futures) f.get();

    std::mutex mtx;
    std::vector<std::future<void>> futures;

    for (int i = 0; i < C_CURR.size(); ++i) {
        sensor_msgs::msg::PointCloud2 c = C_CURR[i];
        if (c.width * c.height == 0) continue;
        
        icp::PointCloud<icp::ThreeD> icp_c = pc_to_icp_pc(c);
        
        for (int j = 0; j < C_PREV.size(); ++j) {
            futures.push_back(std::async(std::launch::async, [&, i, j, icp_c]() {
                sensor_msgs::msg::PointCloud2 c_prev = C_PREV[j];

                if (c_prev.width * c_prev.height == 0) return;
                icp::PointCloud<icp::ThreeD> icp_c_prev = pc_to_icp_pc(c_prev);
                
                std::unique_ptr<icp::ICP3> icp = icp::ICP3::from_method("vanilla", icp::Config()).value();
                icp::ICPDriver driver(std::move(icp));

                driver.set_max_iterations(1);
                driver.set_transform_tolerance(0.1 * M_PI / 180, 0.1);
                
                auto result = driver.converge(icp_c_prev, icp_c, icp::RBTransform3::Identity());
                
                Edge edge;
                edge.edge = std::make_tuple(j, i);
                edge.weight = result.cost;
                
                // Need mutex protection for shared data structures
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    E.push_back(edge);
                    C[i][j] = result.cost;
                }
            }));
        }
    }

    // Wait for all tasks to complete
    for (auto& f : futures) {
        f.get();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    RCLCPP_INFO(this->get_logger(), "RAN FOR: %ld ms", duration.count());

    std::vector<int> matchings = hungarian_assignment(C);
    auto markers = outputMatching(C_PREV, C_CURR, matchings);
    // outputMatching(C_PREV, C_CURR, matchings);

    match_pub_->publish(markers);

    RCLCPP_INFO(this->get_logger(), "done");
    obs_msg_prev_ = msg->obstacles;
  }

  visualization_msgs::msg::MarkerArray outputMatching(
      const std::vector<sensor_msgs::msg::PointCloud2>& C_PREV,
      const std::vector<sensor_msgs::msg::PointCloud2>& C_CURR,
      std::vector<int> matchings) {
      
      visualization_msgs::msg::MarkerArray marker_array;
      
      RCLCPP_INFO(this->get_logger(), "C_prev count: %d v. C_curr count: %d", C_PREV.size(), C_CURR.size());
      for (int i = 0; i < matchings.size(); ++i) {
          int prev_idx = i;
          int curr_idx = matchings[i];
          
          // if prev_idx or curr_idx is -1, then this is an invalid match
          if (curr_idx == -1 || curr_idx >= C_CURR.size() || prev_idx == -1 || prev_idx >= C_PREV.size()) {
              continue;
          }

          RCLCPP_INFO(this->get_logger(), "C_PREV [%d] -- C_CURR [%d]\n", prev_idx, curr_idx);
          
          Eigen::Vector4f centroid_prev, centroid_curr;
          
          // Convert to PCL and compute centroids
          pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_prev(new pcl::PointCloud<pcl::PointXYZ>);
          pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_curr(new pcl::PointCloud<pcl::PointXYZ>);
          
          pcl::fromROSMsg(C_PREV[prev_idx], *cloud_prev);
          pcl::fromROSMsg(C_CURR[curr_idx], *cloud_curr);
          
          pcl::compute3DCentroid(*cloud_prev, centroid_prev);
          pcl::compute3DCentroid(*cloud_curr, centroid_curr);

          RCLCPP_INFO(this->get_logger(), "C_PREV: (%f, %f, %f) -- C_CURR: (%f, %f, %f)\n",
          centroid_prev[0], centroid_prev[1], centroid_prev[2], centroid_curr[0], centroid_curr[1], centroid_curr[2]);
          
          // Create line marker
          visualization_msgs::msg::Marker line;
          line.header.frame_id = "rslidar";
          line.header.stamp = this->now();
          line.ns = "cluster_matches";
          line.id = i;
          line.type = visualization_msgs::msg::Marker::ARROW;
          line.action = visualization_msgs::msg::Marker::ADD;
          
          // Start point (previous cluster)
          geometry_msgs::msg::Point p1;
          p1.x = centroid_prev[0];
          p1.y = centroid_prev[1];
          p1.z = centroid_prev[2];
          
          // End point (current cluster)
          geometry_msgs::msg::Point p2;
          p2.x = centroid_curr[0];
          p2.y = centroid_curr[1];
          p2.z = centroid_curr[2];
          
          line.points.push_back(p1);
          line.points.push_back(p2);
          
          // Style
          line.scale.x = 0.05;
          line.scale.y = 0.1;
          line.scale.z = 0.1;
          
          // Color (green for matched)
          line.color.r = 0.0;
          line.color.g = 1.0;
          line.color.b = 0.0;
          line.color.a = 0.8;
          
          line.lifetime = rclcpp::Duration::from_seconds(2.0);
          
          marker_array.markers.push_back(line);
      }
      
      return marker_array;
  }

  // void sanityCheckHungarian() {
  //     RCLCPP_INFO(this->get_logger(), "STARTING SANITY CHECK\n");
  //     std::vector<std::vector<float>> costs{{8.0, 5.0, 9.0}, {4.0, 2.0, 4.0}, {7.0, 3.0, 8.0}};
  //     //          A   B   C
  //     // clean   8.0 5.0 9.0
  //     // sweep   4.0 2.0 4.0
  //     //  wash   7.0 3.0 8.0
  //     assert((hungarian_assignment(costs) == std::vector<float>{5.0, 9.0, 15.0}));
  //     RCLCPP_INFO(this->get_logger(), "Sanity check passed\n");
  // }

  void multi_hypothesis_tracking(std::unordered_map<int32_t, std::vector<PointXYZCluster>> C_prev, 
                                 std::unordered_map<int32_t, std::vector<PointXYZCluster>> C) {
    // takes in ??? vector<clusters> or the entire point cloud idrk
    // tbh for both MHT and max bipartite matching, might need to consider all
    // possible matches
    // or like icp_cost_bipartite_matching is a way to generate all possible 
    // association hypothesis -> run SHT -> reduce number of hypotheses

    // cases for matching clusters:
    // 1) old obstacle being tracked gradually / suddenly disappear
    //      -> there are no obstacles associated w/ the current frame
    //      -> no cluster in C that matches old obstacle in C_prev_: {T_jN}
    // 2) new obstacle suddenly / gradually enters the LiDAR range
    //      -> previous frame is not associated with the obstacle
    //      -> no cluster in C_prev_ that matches new obstacle in C: {Z_jN}
    // 3) obstacle in current frame is assocaited with previously tracked obstacle
    //      -> denote as {Y(T_j, Z_j)}
  }

  // void hungarian_assignment(Eigen::MatrixXf mat, float max_cost)  {
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

  void bertsekas_auction() {
    // for parallelization approximate max bipartite matching
    // problems: we have non-integral bipartite graph.
    // multiplicative auction algo for (1-e)-approximation of max bipartite:
    // one-sided vertex deletion (delete c_prev given corr obstacle left the frame)
    // other-sided vertex insertion (insert c given corr new obstacle appear in frame)
    // could scale up the edge weights by 10^6 to get integer weights -> run the algo

    // allow unmatched bidders: can assign matches s.t. (c_prev, null): c_prev's
    // obstacle disappeared
    // or (null, c): c's obstacle newly appeared in frame
    // WLOG: if there are no c_prev's remaining or if all remaining c_prev's have cost > threshold
    // then (null, c)
  }

  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr prev_pub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr curr_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr match_pub_;
  sensor_msgs::msg::PointCloud2::SharedPtr msg_prev_;
  std::vector<sensor_msgs::msg::PointCloud2> obs_msg_prev_;
  std::unordered_map<int32_t, std::vector<PointXYZCluster>> C_prev_;
  rclcpp::Subscription<cev_msgs::msg::Obstacles>::SharedPtr obs_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ObstacleTracker>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
