import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import glob
import numpy as np

from models.backbone import UNet
from utils.projection import get_bev

# ---------------------------
# Model
# ---------------------------
model = UNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

# ---------------------------
# Load dataset (images folder)
# ---------------------------
image_paths = glob.glob("data/v1.0-mini/samples/CAM_BACK/**/*", recursive=True)
image_paths=[p for p in image_paths if p.endswidth((".jpg", ".png"))]
random.shuffle(image_paths)

if len(image_paths) == 0:
    print("❌ No images found in data/ folder")
    exit()

print(f"✅ Found {len(image_paths)} images")

# ---------------------------
# Training loop
# ---------------------------
EPOCHS = 50

for epoch in range(EPOCHS):
    total_loss = 0

    for path in image_paths:
        # Load image
        image = cv2.imread(path)

        if image is None:
            continue

        # ---------------------------
        # BEV transform
        # ---------------------------
        bev = get_bev(image)
        bev = cv2.resize(bev, (128, 128))

        # ---------------------------
        # Input tensor
        # ---------------------------
        x = torch.tensor(bev).permute(2, 0, 1).float().unsqueeze(0) / 255.0

        # ---------------------------
        # Create BETTER fake labels (edges)
        # ---------------------------
        gray = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        gt = edges / 255.0
        gt = torch.tensor(gt).float().unsqueeze(0).unsqueeze(0)

        # Resize GT to match model output
        gt = torch.nn.functional.interpolate(gt, size=(128, 128))

        # ---------------------------
        # Forward pass
        # ---------------------------
        pred = model(x)

        # ---------------------------
        # Loss
        # ---------------------------
        loss = loss_fn(pred, gt)

        # ---------------------------
        # Backprop
        # ---------------------------
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(image_paths)
    print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

# ---------------------------
# Save model
# ---------------------------
torch.save(model.state_dict(), "model.pth")
print("✅ Model saved as model.pth")
