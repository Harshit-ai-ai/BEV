# BEV

BEV Occupancy Prediction using UNet

This project implements a **Bird’s Eye View (BEV) Occupancy Prediction model** using a UNet-based architecture. It processes front camera images and predicts occupancy maps for autonomous driving scenarios.

---
Overview

- Uses **nuScenes v1.0-mini dataset**
- Converts camera images into BEV representation
- Trains a **UNet model** for occupancy prediction
- Outputs binary occupancy maps

---
Model Architecture

- Backbone: **UNet**
- Input: Front camera images (`CAM_FRONT`)
- Output: BEV occupancy grid
- Activation: **Sigmoid** (for binary classification)

---
Project Structure

bev-occupancy/
│
├── data/
│ └── v1.0-mini/
│ ├── samples/
│ │ └── CAM_FRONT/
│ ├── maps/
│ └── sweeps/
│
├── datasets/
│ └── nuscenes_loader.py
│
├── models/
│ ├── backbone.py
│ ├── bev_transform.py
│ └── occupancy_head.py
│
├── utils/
│ ├── geometry.py
│ └── projection.py
│
├── train.py
├── infer.py
└── README.md


---


