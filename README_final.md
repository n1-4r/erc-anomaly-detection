# erc-anomaly-detection
# 🔍 ROS2 Real-Time Anomaly Detection & Processing Suite

This ROS2 package provides a **modular anomaly detection pipeline** for the European Rover Challenge (ERC), combining state-of-the-art vision models with efficient data handling.  

It contains **three ROS2 nodes**:  
1. **DINOv2 Anomaly Detection** – real-time anomaly detection using Meta’s DINOv2.  
2. **Mars Object Detection & Description** – AI-powered semantic description of detected anomalies using Ollama vision models.  
3. **Anomaly De-Duplication** – ensures anomalies are stored only once by removing duplicates via feature matching.  

---

## 🚀 Features

- **DINOv2 (ViT-S/14) Anomaly Detection**
  - Patch-level feature extraction from images  
  - Self-similarity–based anomaly scoring  
  - Heatmap + bounding box visualization  
  - GPU acceleration (CUDA optional)  

- **Mars Object Detection & Description**
  - Uses Ollama (qwen2.5vl:7b-q8_0) for semantic description  
  - 50-word textual description of anomalies  
  - Determines if object is typical in a Mars environment  
  - Outputs results into structured CSV logs  

- **Anomaly De-Duplication**
  - Uses ORB feature extraction + BFMatcher  
  - Compares new anomalies against stored ones  
  - Stores only **unique anomalies** (saves space, avoids redundancy)  

---

## 🧠 How It Works

### 1. Anomaly Detection (DINOv2)
- Camera feed is split into patches → DINOv2 extracts feature embeddings.  
- Cosine similarity scores detect unusual patches.  
- Bounding boxes + heatmaps visualize anomalies.  

### 2. Object Description (Ollama)
- Anomaly images are processed by Ollama’s vision model.  
- Generates short, structured descriptions.  
- Results saved into CSV for downstream use.  

### 3. De-Duplication
- ORB descriptors extracted from anomalies.  
- BFMatcher compares against stored database.  
- New anomalies saved only if not already seen.  

---

## 📦 Installation

### Prerequisites
- **ROS2** (Humble recommended)  
- **Python 3.8+**  
- **CUDA** (optional, for GPU acceleration)  
- **Ollama with qwen2.5vl:7b-q8_0 model** (for Mars description node)  

---

### Clone and Build
```bash
cd ~/ros2_ws/src
git clone https://github.com/n1-4r/erc-anomaly-detection.git
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

---

### Python Dependencies
```bash
# Core
pip install numpy pillow

# OpenCV (image handling)
pip install "opencv-python<2.0" opencv-contrib-python

# PyTorch + DINOv2
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# ROS2 CV bridge
pip install cv_bridge

# Ollama interaction
pip install watchdog
```
dinov2 is installed automatically through torch.hub.

---

### Running the nodes

```bash
ros2 run anomaly_detection anomaly_detection_front
ros2 run anomaly_detection anomaly_deduplication_node
ollama run qwen2.5vl:7b-q8_0
ros2 run anomaly_detection image_monitor_node
ros2 run anomaly_detection pdf_generator_node
