import os
import json
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import tf_transformations

SCALE = 1.0

class AnomalyDeduplicator(Node):
    def __init__(self):
        super().__init__('anomaly_deduplicator')
        self.image_dir = '/home/smartnihar6/ros2_ws/src/testrun_folder/detected_frames'
        self.output_dir = '/home/smartnihar6/ros2_ws/src/testrun_folder/filtered_frames'
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure filtered_frames exists

        self.orb = cv2.ORB_create(500)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.declared_anomalies = []
        self.seen_files = set()

        self.current_pose = [0.0, 0.0, 0.0]
        self.current_yaw = 0.0

        # Removed the create_subscription() line that caused the error

        self.timer = self.create_timer(2.0, self.process_new_images)

    def image_quality(self, gray, keypoints):
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        return 0.7 * sharpness + 0.3 * len(keypoints)

    def find_duplicate_index(self, new_des, new_area=None, match_threshold=30, area_ratio_threshold=0.5):
        for i, anomaly in enumerate(self.declared_anomalies):
            if anomaly.get('des') is not None and new_des is not None:
                matches = self.bf.match(anomaly['des'], new_des)
                if len(matches) >= match_threshold:
                    prev_area = anomaly.get('area', 0)
                    # Check area ratio — skip duplicate if areas differ too much
                    if new_area is not None and prev_area > 0:
                        ratio = min(new_area, prev_area) / max(new_area, prev_area)
                        if ratio < area_ratio_threshold:
                            continue  # treat as different object
                    return True, i
        return False, -1

    def update_anomaly(self, idx, des, quality, area, idno, image_file, full_path):
        self.declared_anomalies[idx].update({
            'des': des,
            'quality': quality,
            'area': area,
            'idno': idno,
            'filename': image_file
        })
        dst_path = os.path.join(self.output_dir, image_file)
        cv2.imwrite(dst_path, cv2.imread(full_path))
        self.get_logger().info(
            f"Replaced representative for anomaly #{idx} with higher-quality, larger-area image: {image_file}"
        )
        self.save_metadata()

    def process_new_images(self):
        # Wait until detected_frames exists
        if not os.path.exists(self.image_dir):
            self.get_logger().warn(f"Waiting for directory to be created: {self.image_dir}")
            return

        for image_file in sorted(os.listdir(self.image_dir)):
            if not image_file.endswith(('.png', '.jpg', '.jpeg')):
                continue

            full_path = os.path.join(self.image_dir, image_file)

            try:
                idno, area, rel_x, rel_y = self.extract_data_from_filename(image_file)
                z = 0.0  # Placeholder since depth estimation is removed

                cos_yaw = np.cos(self.current_yaw)
                sin_yaw = np.sin(self.current_yaw)
                global_dx = rel_x * cos_yaw - rel_y * sin_yaw
                global_dy = rel_x * sin_yaw + rel_y * cos_yaw

                x = self.current_pose[0] + global_dx
                y = self.current_pose[1] + global_dy

                image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                kp, des = self.orb.detectAndCompute(image, None)
                quality = self.image_quality(image, kp)

                is_dup, idx = self.find_duplicate_index(des, new_area=area)

                if is_dup:
                    prev_area = self.declared_anomalies[idx].get('area', 0)
                    if area > prev_area:
                        self.delete_nearby_idnos(idno)
                        self.update_anomaly(idx, des, quality, area, idno, image_file, full_path)
                    else:
                        self.get_logger().info(
                            f"Duplicate (lower quality/area) discarded: {image_file} at ({x:.2f}, {y:.2f}, {z:.2f})"
                        )
                    continue

                self.declared_anomalies.append({
                    'des': des,
                    'quality': quality,
                    'area': area,
                    'idno': idno,
                    'filename': image_file
                })
                cv2.imwrite(os.path.join(self.output_dir, image_file), cv2.imread(full_path))
                self.get_logger().info(f"New anomaly saved: {image_file} at ({x:.2f}, {y:.2f}, {z:.2f})")
                self.save_metadata()

            except Exception as e:
                self.get_logger().error(f"Error processing {image_file}: {e}")

    def delete_nearby_idnos(self, idno, delta=10):
        files = os.listdir(self.output_dir)
        for fname in files:
            if not fname.endswith(('.png', '.jpg', '.jpeg')):
                continue
            try:
                fid = int(fname.split("_")[0])
                if abs(fid - idno) <= delta:
                    os.remove(os.path.join(self.output_dir, fname))
                    self.get_logger().info(f"Deleted old nearby image due to better anomaly: {fname}")
            except Exception as e:
                self.get_logger().warning(f"Could not parse ID from filename {fname}: {e}")

    def extract_data_from_filename(self, filename):
        base = os.path.basename(filename)
        idno = int(base.split("=")[1].split("_")[0])
        area = float(base.split("a=")[1].split("_")[0])
        x = float(base.split("x=")[1].split("_")[0])
        y = float(base.split("y=")[1].split("_")[0])
        return idno, area, x, y

    def is_duplicate(self, new_des, match_threshold=30):
        for anomaly in self.declared_anomalies:
            if anomaly['des'] is not None and new_des is not None:
                matches = self.bf.match(anomaly['des'], new_des)
                if len(matches) >= match_threshold:
                    return True
        return False

    def save_metadata(self):
        metadata = [
            {k: v for k, v in d.items() if k in ['x', 'y', 'z', 'area', 'idno']}
            for d in self.declared_anomalies
        ]
        with open(os.path.join(self.output_dir, "anomalies_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDeduplicator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
