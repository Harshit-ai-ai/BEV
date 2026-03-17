import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.projection import get_bev
from models.backbone import UNet

# ---------------------------
# Load image
# ---------------------------
image = cv2.imread("/home/harshit/Documents/BEV(MAHE MOBILITY)/bev-occupancy/data/test.jpg")

if image is None:
    print("Image not found!")
    exit()

# ---------------------------
# BEV transform
# ---------------------------
bev = get_bev(image)

# ---------------------------
# Preprocess
# ---------------------------
bev_resized = cv2.resize(bev, (128, 128))
x = torch.tensor(bev_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

# ---------------------------
# Load model (IMPORTANT: BEFORE inference)
# ---------------------------
model = UNet()
model.load_state_dict(torch.load("model.pth"))
model.eval()

# ---------------------------
# Inference
# ---------------------------
with torch.no_grad():
    pred = model(x)

# ---------------------------
# Convert output
# ---------------------------
occupancy_map = pred.squeeze().cpu().numpy()

# Optional: binary map
occupancy_map = (occupancy_map > 0.5).astype("uint8") * 255

# ---------------------------
# Visualization
# ---------------------------
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
bev_rgb = cv2.cvtColor(bev, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(image_rgb)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("BEV")
plt.imshow(bev_rgb)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Occupancy")
plt.imshow(occupancy_map, cmap='gray')
plt.axis("off")

plt.show()