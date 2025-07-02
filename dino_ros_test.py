"""
sudo apt update && sudo apt install -y \
  ros-humble-rclpy \
  ros-humble-sensor-msgs \
  ros-humble-cv-bridge \
  python3-opencv \
  python3-pip && \
pip3 install --upgrade pip && \
pip3 install torch torchvision matplotlib pillow
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import torch
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image as PILImage
import math
import time

class DINOAnomalyNode(Node):
    def __init__(self):
        super().__init__('dino_anomaly_node')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=True, trust_repo=True
        ).eval().to(self.device)

        self.skip_rate = 6
        self.frame_count = 0

        self.get_logger().info("DINOv2 anomaly node initialized.")

    def preprocess(self, img):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        return transform(img).unsqueeze(0)

    def patch_anomaly_scores(self, patch_feats):
        feats = F.normalize(patch_feats, p=2, dim=1)
        dist = 1.0 - feats @ feats.T
        return dist.mean(dim=1)

    def draw_boxes(self, frame, scores, thr_sigma=1.3):
        N = scores.numel()
        side = int(math.sqrt(N))
        heat = scores.view(side, side).cpu().numpy()
        thr = heat.mean() + thr_sigma * heat.std()
        mask = (heat > thr).astype(np.uint8)

        H, W = frame.shape[:2]
        mask_big = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        heat_big = cv2.resize(heat, (W, H), interpolation=cv2.INTER_NEAREST)

        kernel = np.ones((7, 7), np.uint8)
        mask_big = cv2.morphologyEx(mask_big, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_size = 15000
        min_box_score = 0.6

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h < min_size:
                continue

            avg_score = heat_big[y:y + h, x:x + w].mean()
            if avg_score < min_box_score:
                continue

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        return frame

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.skip_rate != 0:
            return

        start_time = time.time()

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = self.preprocess(pil_img).to(self.device)

        with torch.no_grad():
            out = self.model.forward_features(x)
            patch_tokens = out.get("x_norm_patchtokens", out["x_norm_patch_tokens"])
        patches = patch_tokens.squeeze(0)
        scores = self.patch_anomaly_scores(patches)

        frame = self.draw_boxes(frame, scores)
        elapsed = time.time() - start_time
        self.get_logger().info(f"Processed in {elapsed:.2f} sec")

        cv2.imshow("DINOv2 Anomaly Detection", frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DINOAnomalyNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
