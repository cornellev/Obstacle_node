# Obstacle_node
Hi all, I'm going to start yapping again!

## Angle Bins and Ray Tracing Via Bresenham's Line Algo

![Occupancy Grid 1](demos/occ.png)
![Occupancy Grid 2](demos/occupancygrid.png)

## Multi-Hypothesis Tracking
```
cd Obstacle_node
git submodule add https://github.com/cornellev/icp.git lib/cev_icp
git submodule update --init --recursive
cd lib/cev_icp
sudo make install LIB_INSTALL=/usr/local/lib HEADER_INSTALL=/usr/local/include
```

# Important Links
[Rasterize a point cloud](https://r-lidar.github.io/lasR/reference/rasterize.html)

[Dynamic Obstacle Detection and Tracking Based on 3D Lidar](https://www.jstage.jst.go.jp/article/jaciii/22/5/22_602/_pdf)