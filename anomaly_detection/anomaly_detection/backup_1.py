import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image as PILImage
import numpy as np
import cv2
import math
import time
import os


def load_dino_model(device):
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        pretrained=True,
        trust_repo=True,
    )
    model.eval().to(device)
    return model


def preprocess_pil(img):
    _mean = [0.485, 0.456, 0.406]
    _std = [0.229, 0.224, 0.225]

    t = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])
    return t(img).unsqueeze(0)


def patch_anomaly_scores(patch_feats: torch.Tensor) -> torch.Tensor:
    feats = F.normalize(patch_feats, p=2, dim=1)
    dist = 1.0 - feats @ feats.T
    return dist.mean(dim=1)


def show_boxes_on_frame(self, node, frame: np.ndarray, scores: torch.Tensor, thr_sigma: float = 1.2 ):
    N = scores.numel()
    side = int(math.sqrt(N))
    if side * side != N:
        raise ValueError("Anomaly scores length is not a perfect square")

    save_dir = "/home/smartnihar6/ros2_ws/src/testrun_folder/detected_frames" ##################

    heat = scores.view(side, side).cpu().numpy()
    thr = heat.mean() + thr_sigma * heat.std()
    mask = (heat > thr).astype(np.uint8)

    H, W = frame.shape[:2]
    mask_big = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    heat_big = cv2.resize(heat, (W, H), interpolation=cv2.INTER_NEAREST)

    kernel = np.ones((7, 7), np.uint8)
    mask_big = cv2.morphologyEx(mask_big, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_size = 25000
    min_box_score = 0.6


    os.makedirs(save_dir, exist_ok=True)

    for c in contours:

        x,y,w,h = cv2.boundingRect(c)
        if w * h < min_size:
            continue
        avg_score = heat_big[y:y + h, x:x + w].mean()
        if avg_score < min_box_score:
            continue
        
        print(f"[DEBUG] Current working directory: {os.getcwd()}")

        print("Anomaly saved.")

        filename = os.path.join(save_dir, f"{node.anomaly_frame_idx:04d}_a={w*h}_x={x+(w/2)}_y={y+(h/2)}_.png")
        
        #filename = os.path.join(save_dir, f"image_x={x+(w/2)}_y={y+(h/2)}_.png")
        node.anomaly_frame_idx += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.imwrite(filename, frame)        
        
        

    cv2.imshow("DINOv2 Anomaly Detection", frame)
    cv2.waitKey(1)


class DINOAnomalyNode(Node):
    def __init__(self):
        super().__init__('dino_anomaly_node')
        self.bridge = CvBridge()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = load_dino_model(self.device)
        self.subscription = self.create_subscription(
            Image,
            'front_cam/color/image_raw',  # You can remap this later
            self.image_callback,
            10)
        self.anomaly_frame_idx = 0
        self.frame_count = 0
        self.skip_rate = 8
        print("[INFO] [DINOv2] Anomaly node initialized.")

    def image_callback(self, msg):
        self.frame_count += 1
        if self.frame_count % self.skip_rate != 0:
            return

        start_time = time.time()
        clear_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        blur_ksize = 65
        frame = cv2.GaussianBlur(clear_frame, (blur_ksize, blur_ksize), 0)

        img_pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = preprocess_pil(img_pil).to(self.device)

        with torch.no_grad():
            out = self.model.forward_features(x)
            patch_tokens = (
                out["x_norm_patchtokens"]
                if "x_norm_patchtokens" in out
                else out["x_norm_patch_tokens"]
            )
        patches = patch_tokens.squeeze(0)
        scores = patch_anomaly_scores(patches)

        thr = scores.mean() + scores.std()
        #print(f"Threshold: {thr:.3f}  —  {(scores > thr).sum().item()} anomalous patches of {scores.numel()}")
        #print(f"Time per frame: {time.time() - start_time:.2f} seconds")

        show_boxes_on_frame(self, self, clear_frame, scores)


def main(args=None):
    rclpy.init(args=args)
    node = DINOAnomalyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
