import math

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time

# ---------- utilities -------------------------------------------------------
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
    """
    Resize input image directly to 224×224.
    Normalizes using ImageNet statistics.
    """
    _mean = [0.485, 0.456, 0.406]
    _std  = [0.229, 0.224, 0.225]

    t = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # <-- Resize to exactly 224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=_mean, std=_std),
        ]
    )
    return t(img).unsqueeze(0)  # [1,3,224,224]



def patch_anomaly_scores(patch_feats: torch.Tensor) -> torch.Tensor:
    feats = F.normalize(patch_feats, p=2, dim=1)
    dist  = 1.0 - feats @ feats.T          # cosine distance
    return dist.mean(dim=1)                # [N_patches]


def visualise(scores: torch.Tensor):
    n = scores.numel()
    g = int(math.sqrt(n))                  # 14 for 224×224
    heat = scores.view(g, g).cpu().numpy()
    plt.imshow(heat, cmap="hot")
    plt.title("heatmap")
    plt.axis("off")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def show_boxes_on_frame(frame: np.ndarray, scores: torch.Tensor, thr_sigma: float = 1.3):
    """
    Draws bounding boxes around high-anomaly regions on a live webcam frame.

    Parameters:
    - frame: np.ndarray (BGR image from OpenCV)
    - scores: 1D torch.Tensor of patch anomaly scores
    - thr_sigma: controls sensitivity (higher = stricter)
    """
    N = scores.numel()
    side = int(math.sqrt(N))  # typically 14 for 224x224
    if side * side != N:
        raise ValueError("Anomaly scores length is not a perfect square")

    heat = scores.view(side, side).cpu().numpy()
    thr = heat.mean() + thr_sigma * heat.std()
    mask = (heat > thr).astype(np.uint8)

    H, W = frame.shape[:2]
    mask_big = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    heat_big = cv2.resize(heat, (W, H), interpolation=cv2.INTER_NEAREST)

    kernel = np.ones((7, 7), np.uint8)
    mask_big = cv2.morphologyEx(mask_big, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_size = 15000         # pixels
    min_box_score = 0.6      # average anomaly score inside box

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h < min_size:
            continue

        avg_score = heat_big[y:y + h, x:x + w].mean()
        if avg_score < min_box_score:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Show the processed frame
    cv2.imshow("DINOv2 Anomaly Detection", frame)





# ---------- main ------------------------------------------------------------
def main():
    print("Starting webcam anomaly detection...")
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_dino_model(device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    frame_count = 0
    skip_rate = 6  # Process every 6th frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % skip_rate != 0:
            continue

        start_time = time.time()

        img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        x = preprocess_pil(img_pil).to(device)

        with torch.no_grad():
            out = model.forward_features(x)
            patch_tokens = (
                out["x_norm_patchtokens"]
                if "x_norm_patchtokens" in out
                else out["x_norm_patch_tokens"]
            )
        patches = patch_tokens.squeeze(0)
        scores = patch_anomaly_scores(patches)

        thr = scores.mean() + scores.std()
        print(
            f"Threshold: {thr:.3f}  —  "
            f"{(scores > thr).sum().item()} anomalous patches of {scores.numel()}"
        )

        elapsed = time.time() - start_time
        print(f"Time per frame: {elapsed:.2f} seconds")

        # Modified version of boxes_from_scores to take raw frame, not path
        show_boxes_on_frame(frame, scores)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

