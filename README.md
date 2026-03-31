BEV Occupancy Prediction using UNet

Project Overview

This project focuses on predicting **Bird’s Eye View (BEV) occupancy maps** from monocular camera input using deep learning.  

In autonomous driving systems, understanding the spatial layout of the environment is crucial. This project transforms front-view camera images into a **top-down occupancy representation**, enabling better scene understanding for navigation and planning.

The system is built using a **UNet-based architecture** trained on the **nuScenes v1.0-mini dataset**.

---

Model Architecture

The model uses a **UNet (encoder-decoder) architecture**:

- **Encoder (Downsampling path)**  
  Extracts hierarchical spatial features from input images using convolutional layers.

- **Decoder (Upsampling path)**  
  Reconstructs spatial resolution using transpose convolutions and skip connections.

- **Skip Connections**  
  Preserve fine-grained spatial details from encoder to decoder.

- **Output Layer**  
  - 1-channel BEV occupancy map  
  - Activation: **Sigmoid** (binary occupancy prediction)

---

Dataset Used

**Dataset:** nuScenes v1.0-mini  

- Source: https://www.nuscenes.org/
- Input: Front camera images (`CAM_FRONT`)
- Structure:

---
Dataset used-https://drive.google.com/file/d/1o0JGyAuVkdbSohdXR88HwqHT5ILU0xHz/view?usp=drive_link

data/v1.0-mini/
├── samples/
│ └── CAM_FRONT/
├── maps/
└── sweeps/

The dataset provides real-world driving scenarios with diverse environments.

---

Setup & Installation

### 1. Clone Repository
```bash
git clone https://github.com/Harshit-ai-ai/BEV
cd bev-occupancy

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

pip install torch torchvision numpy opencv-python

How to Run the Code
Step 1: Ensure Dataset Path

Place dataset inside:

data/v1.0-mini/
Step 2: Train the Model
python train.py

Expected output:

Epoch 1/100, Loss: ...
...
Epoch 100/100, Loss: ...
Model saved as model.pth
Step 3: Run Inference
python infer.py

This will:

Load trained model

Generate BEV occupancy predictions

Example Outputs / Results
✔ Training Output (Sample)
Epoch 92/100, Loss: 0.0403
Epoch 95/100, Loss: 0.0340
Epoch 100/100, Loss: 0.0308
Model saved as model.pth
✔ Observations

Loss consistently decreases, indicating successful learning

Model is able to generate structured occupancy maps

Works on real-world driving scenes from nuScenes dataset

Common Issues & Fixes
No images found

Use recursive loading:

import glob
image_paths = glob.glob("data/**/*.jpg", recursive=True)
ModuleNotFoundError: models

Run from project root:

cd bev-occupancy
python train.py
Future Improvements

Multi-camera BEV fusion

Temporal modeling using sequential frames

Transformer-based BEV models

Real-time optimization for deployment
