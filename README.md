# Autonomous Exploration using 2D LiDAR with Dubins-RRT* Planning and Sampling-Based Goal Selection

This ROS package implements an autonomous exploration system using **RRT\*** with **Dubins path constraints** and a **Pure Pursuit controller** for a mobile robot equipped with a 2D LiDAR. It is tested in the **Stonefish simulator** and supports **real-time exploration**, goal sampling, path planning, and trajectory execution.

---

## Objectives

- Perform autonomous exploration in unknown environments.
- Sample informative goals using **information gain**.
- Plan feasible, smooth paths using **RRT\*** and **Dubins curves**.
- Track paths using a **Pure Pursuit controller**.
- Integrate with simulated or real TurtleBot via ROS.

---
## System Architecture

The diagram below shows the data flow between the main components of the exploration system.

![System Architecture](planning%20flow%20chart.png)


##  How to Run

### 1. Launch Simulation and All Nodes
```bash
roslaunch lidar_based_exploration stonefish.launch
```

This launches:
- TurtleBot robot model (via `turtlebot_basic.launch`)
- Octomap server and point cloud conversion
- Online planner and sampling-based goal selector
- RQT GUI and RViz visualization

---

### 2. Node Overview

| Node | Description |
|------|-------------|
| `laser_scan_to_point_cloud_node.py` | Converts `/rplidar` scan to a usable point cloud |
| `sampling_based_exploration_node.py` | Samples goals based on information gain |
| `turtlebot_online_path_planning_node.py` | Plans with RRT* and executes with Pure Pursuit |
| `octomap_server_node` | Builds occupancy map from LiDAR |
| `rqt_gui` | GUI interface  |



## Authors

**Gebrecherkos Gebrezgabhier**  
**Pravin Oli**  
Supervised by: *Narcis Palomeras, Sebastian, Martha*  
University of Girona – Intelligent Field Robotics Systems
