// src/occupancy_grid.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <cev_msgs/msg/obstacles.hpp>
#include <obstacle/msg/obstacle_array.hpp>

#include <unordered_map>
#include <vector>
#include <cstdint>
#include <opencv2/opencv.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

using obstacle::msg::ObstacleArray;

constexpr double deg2rad(double deg) {
    return deg * M_PI / 180.0;
};

struct PointXYZCluster {
  float x;
  float y;
  float z;
  int32_t cluster_id;
};

 struct BinInfo {
  BinInfo() = default;
  BinInfo(const double _range, const double _wx, const double _wy)
  : range(_range), wx(_wx), wy(_wy)
  {
  }
  double range;
  double wx;
  double wy;
};

enum class CellState {
  OCCUPIED,
  UNKNOWN,
  FREE
};

class OccupancyGridNode : public rclcpp::Node {
public:
  OccupancyGridNode()
  : Node("occupancy_grid")
  {
    obs_sub_ = this->create_subscription<cev_msgs::msg::Obstacles>(
      "/rslidar_matches", 10,
      std::bind(&OccupancyGridNode::obstaclesCallback, this, std::placeholders::_1));

    pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "/rslidar_matches_points", 10,
      std::bind(&OccupancyGridNode::pcCallback, this, std::placeholders::_1));

    bev_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/bev_obstacles", 10);

    grid_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/occupancy_grid", 10);

    RCLCPP_INFO(this->get_logger(), "OccupancyGridNode started - waiting for PointCloud2 on 'input_points'");
  }

private:
  void obstaclesCallback(const cev_msgs::msg::Obstacles::SharedPtr msg)
  {
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

    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    std::unordered_map<int32_t, std::vector<PointXYZCluster>> clusters;
    size_t total_points = 0;
    for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_id) {
      PointXYZCluster p;
      p.x = *iter_x;
      p.y = *iter_y;
      p.z = *iter_z;
      p.cluster_id = *iter_id;
      clusters[p.cluster_id].push_back(p);

      min_x = std::min(min_x, p.x);
      min_y = std::min(min_y, p.y);
      max_x = std::max(max_x, p.x);
      max_y = std::max(max_y, p.y);

      ++total_points;
    }

    const float padding = 1.0f;
    min_x -= padding;
    min_y -= padding;
    max_x += padding;
    max_y += padding;

    float cell_size = 0.05f;
    float angle_increment = deg2rad(0.4);
    // 26 fov has horizontal angle:  0.4°
    int width = static_cast<int>((max_x - min_x) / cell_size);
    int height = static_cast<int>((max_y - min_y) / cell_size);

    grid_msg_.header.frame_id = "rslidar";
    grid_msg_.info.resolution = cell_size;
    grid_msg_.info.width = width;
    grid_msg_.info.height = height;
    grid_msg_.info.origin.position.x = min_x;
    grid_msg_.info.origin.position.y = min_y;
    grid_msg_.info.origin.position.z = 0.0;
    grid_msg_.info.origin.orientation.x = 0.0;
    grid_msg_.info.origin.orientation.y = 0.0;
    grid_msg_.info.origin.orientation.z = 0.0;
    grid_msg_.info.origin.orientation.w = 1.0;

    grid_msg_.data.assign(width * height, static_cast<int8_t>(CellState::UNKNOWN));

    // z-axis filtering
    const float z_min_allowed = 0.0f;   
    const float z_max_allowed = 2.5f;   

    std::unordered_map<int32_t, std::vector<PointXYZCluster>> filtered_clusters;
    size_t total_kept_points = 0;

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
        total_kept_points += filtered_clusters[cid].size();
      } 
    }

    // === BEV projection + OBB ===
    ObstacleArray out;
    out.header = msg->header;

    sensor_msgs::msg::PointCloud2 bev_points;
    bev_points.header = out.header;
    bev_points.header.frame_id = "rslidar";
    bev_points.height = 1;
    
    sensor_msgs::PointCloud2Modifier modifier(bev_points);
    modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");
    modifier.resize(total_kept_points);

    sensor_msgs::PointCloud2Iterator<float> out_x(bev_points, "x");
    sensor_msgs::PointCloud2Iterator<float> out_y(bev_points, "y");
    sensor_msgs::PointCloud2Iterator<float> out_z(bev_points, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> out_r(bev_points, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> out_g(bev_points, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> out_b(bev_points, "b");

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
        std_msgs::msg::ColorRGBA color = getColorFromId(cid);

        *out_x = p.x;
        *out_y = p.y;
        *out_z = 0.0f; 
        *out_r = color.r * 255;
        *out_g = color.g * 255;
        *out_b = color.b * 255;
        ++out_x; ++out_y; ++out_z;
        ++out_r; ++out_g; ++out_b;
      }
    }

    bev_points.width = static_cast<uint32_t>(bev_points.data.size() / bev_points.point_step);
    bev_points.row_step = bev_points.point_step * bev_points.width;

    // bev_points done -> get occupancy grid:
    sensor_msgs::msg::PointCloud2 map_scan;
    map_scan.header = out.header;
    map_scan.header.frame_id = "rslidar";
    map_scan.height = 1;

    sensor_msgs::PointCloud2Modifier map_modifier(map_scan);
    map_modifier.setPointCloud2Fields(
      4,
      "x", 1, sensor_msgs::msg::PointField::FLOAT32,
      "y", 1, sensor_msgs::msg::PointField::FLOAT32,
      "z", 1, sensor_msgs::msg::PointField::FLOAT32,
      "id", 1, sensor_msgs::msg::PointField::UINT32
    );

    map_modifier.resize(total_kept_points);

    worldToMap(filtered_clusters, map_scan);
    Obstacle2OccupancyGrid(bev_points, map_scan, angle_increment);

    bev_pub_->publish(bev_points);
    grid_pub_->publish(grid_msg_);
  }

  // need ground segmentor before obstacle seg
  void Obstacle2OccupancyGrid(
    sensor_msgs::msg::PointCloud2 &world_scan,
    sensor_msgs::msg::PointCloud2 &map_scan,
    float angle_increment
  ) {
      // add step that transforms point cloud 
      std::vector<std::vector<BinInfo>> obstacle_angle_bins;
      std::vector<signed char> grid_points_(grid_msg_.info.width * grid_msg_.info.height, 50);
      // 360 degrees
      constexpr double min_angle = deg2rad(-180.0);
      constexpr double max_angle = deg2rad(180.0);
      const size_t angle_bin_size = ((max_angle - min_angle) / angle_increment) + size_t(1);
      obstacle_angle_bins.resize(angle_bin_size);

      // add obstacle points to their corresponding angle bins -> 1 bin per ray
      RCLCPP_INFO(this->get_logger(), "world_scan %zu and map_scan %zu", world_scan.data.size(), map_scan.data.size());

      for (sensor_msgs::PointCloud2ConstIterator<float> iter_x(world_scan, "x"), iter_y(world_scan, "y"), 
        iter_wx(map_scan, "x"), iter_wy(map_scan, "y");
        iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_wx, ++iter_wy) 
        {
          const double angle = atan2(*iter_y, *iter_x);
          int angle_bin_idx = (angle - min_angle) / angle_increment;
          // BinInfo(l2-dist of obstacle point from origin, x, y)
          obstacle_angle_bins.at(angle_bin_idx)
            .push_back(BinInfo(std::hypot(*iter_y, *iter_x), *iter_wx, *iter_wy));
      }
      
      double ox = std::floor(grid_msg_.info.origin.position.x / grid_msg_.info.resolution);
      double oy = std::floor(grid_msg_.info.origin.position.y / grid_msg_.info.resolution);
      // sort by distance 
      for (auto & obstacle_angle_bin : obstacle_angle_bins) {
        std::sort(obstacle_angle_bin.begin(), obstacle_angle_bin.end(),
        [](auto a, auto b) { return a.range < b.range; });
      }

        // initialize cells to the final point with freespace
      for (size_t bin_idx = 0; bin_idx < obstacle_angle_bins.size(); ++bin_idx) {
        // iterate through all angle_bins, find the farthest point of each bin -> last element
        auto & obstacle_angle_bin = obstacle_angle_bins.at(bin_idx);

        BinInfo end_distance;
        if (obstacle_angle_bin.empty()) {
          continue;
        } else {
          end_distance = obstacle_angle_bin.back(); // furthest away point
        }
        // from origin to farthest point are all FREE initially
        rayTrace(ox, oy, end_distance.wx, end_distance.wy, CellState::FREE);

        // later implement method that takes into account blindspot behind obstacles -> UNKNOWN
        
        // fill in obstacle points as OCCUPIED
        fillOccupied(obstacle_angle_bins, 0.0);
      }
  }

  void setCellValue(double x, double y, CellState state) {
    int xi = static_cast<int>(x);
    int yi = static_cast<int>(y);
    if (xi < 0 || yi < 0 ||
        xi >= static_cast<int>(grid_msg_.info.width) ||
        yi >= static_cast<int>(grid_msg_.info.height))
        return;

    size_t index = yi * grid_msg_.info.width + xi;
    switch (state) {
      case CellState::OCCUPIED: grid_msg_.data[index] = 97; break;
      case CellState::UNKNOWN:  grid_msg_.data[index] = 50; break;
      case CellState::FREE:     grid_msg_.data[index] = 3;  break;
    }
  }

  void fillOccupied(std::vector<std::vector<BinInfo>> obstacle_angle_bins, double distance_margin) {
    for (size_t bin_idx = 0; bin_idx < obstacle_angle_bins.size(); ++bin_idx) {
      auto & obstacle_angle_bin = obstacle_angle_bins.at(bin_idx);
      for (size_t dist_idx = 0; dist_idx < obstacle_angle_bin.size(); ++dist_idx) {
        const auto & source = obstacle_angle_bin.at(dist_idx);
        setCellValue(source.wx, source.wy, CellState::OCCUPIED);

        if (dist_idx + 1 == obstacle_angle_bin.size()) {
          continue;
        }

        auto next_dist = std::abs(
          obstacle_angle_bin.at(dist_idx + 1).range -
          obstacle_angle_bin.at(dist_idx).range);
        // the distance_margin should be
        //     obstacle height |\
        //                     | \ -> angle is vertical fov
        //                     ----
        //obstacle dist from og    intersect w/ ground
        if (next_dist <= distance_margin) {
          const auto & source = obstacle_angle_bin.at(dist_idx);
          const auto & target = obstacle_angle_bin.at(dist_idx + 1);
          rayTrace(source.wx, source.wy, target.wx, target.wy, CellState::OCCUPIED);
          continue;
        }
      }
    }
  }

  void worldToMap(std::unordered_map<int32_t, std::vector<PointXYZCluster>> &world_scan, sensor_msgs::msg::PointCloud2 &map_scan) {
    sensor_msgs::PointCloud2Iterator<float> mx(map_scan, "x");
    sensor_msgs::PointCloud2Iterator<float> my(map_scan, "y");
    sensor_msgs::PointCloud2Iterator<float> mz(map_scan, "z");
    sensor_msgs::PointCloud2Iterator<int32_t> mid(map_scan, "id");

    for (const auto &kv : world_scan) {
      int cid = kv.first;
      const auto &pts = kv.second;

      for (const auto &p : pts) {
        *mx = std::floor(fabs((p.x - (grid_msg_.info.origin.position.x)) / grid_msg_.info.resolution));
        *my = std::floor(fabs((p.y - (grid_msg_.info.origin.position.y)) / grid_msg_.info.resolution));
        *mz = 0.0;
        *mid = cid;

        ++mx; ++my; ++mz; ++mid;
      }
    }

    map_scan.width = static_cast<uint32_t>(map_scan.data.size() / map_scan.point_step);
    map_scan.row_step = map_scan.point_step * map_scan.width;

  }

  // implement Bresenham's line algo for ray tracing
  void rayTrace(double ox, double oy, double tx, double ty, CellState state) {
      // ray defined by (ox, oy) + (tx - ox, ty - oy) * t;
      int x = static_cast<int>(ox);
      int y = static_cast<int>(oy);
      int x_end = static_cast<int>(tx);
      int y_end = static_cast<int>(ty);

      int dx = std::abs(x_end - x);
      int dy = std::abs(y_end - y);
      int sx = (x < x_end) ? 1 : -1;
      int sy = (y < y_end) ? 1 : -1;
      int err = dx - dy;

      while (true) {
        setCellValue(x, y, state);
        if (x == x_end && y == y_end) break;
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 < dx)  { err += dx; y += sy; }
      }
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

  rclcpp::Subscription<cev_msgs::msg::Obstacles>::SharedPtr obs_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr bev_pub_;
  rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr grid_pub_;
  nav_msgs::msg::OccupancyGrid grid_msg_;
};

int main(int argc, char ** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OccupancyGridNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
