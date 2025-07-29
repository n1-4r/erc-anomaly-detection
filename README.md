# erc-anomaly-detection
# 🔍 ROS2 Real-Time Anomaly Detection using DINOv2 (ViT-S/14)

This ROS2 package performs **real-time anomaly detection** from a camera feed using [Meta's DINOv2](https://github.com/facebookresearch/dinov2) Vision Transformer (ViT-S/14), running on incoming image topics.

It uses zero-shot learning to generate **semantic heatmaps** that highlight unusual or unexpected regions in the robot’s environment — no retraining needed.

---

## 🚀 Features

- Uses **DINOv2 (ViT-S/14)** for patch-level vision feature extraction  
- Computes anomaly scores based on **self-similarity distances**  
- Draws bounding boxes on anomalous areas  
- Saves anomalous frames with annotated metadata  
- ROS2-native using `rclpy` and `cv_bridge`  
- Runs with or without GPU  

---

## 🧠 How It Works

### 1. Patch Feature Embedding  
The input image is split into patches, and each patch is passed through DINOv2 to extract feature vectors.

### 2. Anomaly Scoring via Similarity  
Patch-wise cosine similarities are computed. Unusual patches (low similarity to others) are flagged and visualized as a heatmap. Bounding boxes are drawn over anomalous regions.

---

## 📦 Installation

### Prerequisites

- ROS2 Humble (or newer)  
- Python 3.8+  
- CUDA (optional, for GPU acceleration)  
- A ROS2-compatible camera publishing to `/front_cam/color/image_raw`  

---

### Clone and Build

```bash
cd ~/ros2_ws/src
git clone https://github.com/yourusername/dino_anomaly_ros.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
