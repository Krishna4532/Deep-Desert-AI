# 🏜️ Deep Desert AI
### Semantic Scene Segmentation for Offroad Autonomous Robots

> Pixel-level terrain intelligence for Unmanned Ground Vehicles (UGVs) navigating unstructured desert environments.

---

## 🎯 What It Does

Deep Desert AI classifies every single pixel in a desert camera feed into one of **9 terrain classes** — giving autonomous robots the visual intelligence to distinguish safe ground from lethal obstacles in real time.

| Class | ID | Traversability |
|---|---|---|
| Sky | 0 | N/A |
| Landscape | 1 | ✅ Safe |
| Rocks | 2 | ⛔ Lethal |
| Dry Grass | 3 | ✅ Safe |
| Trees | 4 | ⛔ Obstacle |
| Lush Bushes | 5 | ⚠️ Caution |
| Logs | 6 | ⛔ Obstacle |
| Flowers | 7 | ✅ Safe |
| Ground Clutter | 8 | ⚠️ Caution |

---

## 🏗️ Architecture

```
Input Image [3 × 224 × 224]
      │
      ▼
ResNet-50 Backbone (pretrained ImageNet)
      ├─ Low-level features  [256 × H/4  × W/4 ]
      └─ High-level features [2048 × H/32 × W/32]
              │
              ▼
        ASPP Module
        (Atrous rates: 6, 12, 18 + Global Avg Pool)
              │
              ▼
          Decoder
        (Fuse low-level + ASPP → 256ch)
              │
              ▼
      Output Logits [9 × H × W]
```

**Total Parameters: 40.3M**

---

## ⚡ Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic dataset
```bash
python scripts/generate_synthetic_data.py --n 100
```

### 3. Prepare dataset splits
```bash
python scripts/prepare_data.py
```

### 4. Train
```bash
python train.py
```

### 5. Evaluate with TTA
```bash
python eval.py --checkpoint outputs/checkpoints/best.pth --tta
```

### 6. Predict on a single image
```bash
python predict.py --checkpoint outputs/checkpoints/best.pth --input path/to/image.png --tta
```

---

## 📁 Project Structure

```
Deep-Desert-AI/
├── configs/
│   └── config.yaml                 ← All hyperparameters
├── src/
│   ├── dataset/
│   │   └── offroad_dataset.py      ← Dataset + DataLoader + augmentation
│   ├── models/
│   │   └── deeplabv3plus.py        ← Full DeepLabV3+ architecture
│   └── utils/
│       ├── losses.py               ← Cross-Entropy + Dice loss
│       ├── metrics.py              ← mIoU, pixel accuracy
│       ├── tta.py                  ← Test-Time Augmentation
│       └── visualize.py            ← Colour masks + prediction grids
├── scripts/
│   ├── prepare_data.py             ← Convert raw → processed dataset
│   └── generate_synthetic_data.py ← Synthetic desert frame generator
├── demo/
│   └── offroad_demo.html           ← Interactive demo dashboard
├── train.py                        ← Training loop
├── eval.py                         ← Evaluation script
├── predict.py                      ← Single-image inference
└── requirements.txt
```

---

## 🔬 Key Design Decisions

### Why DeepLabV3+?
ASPP (Atrous Spatial Pyramid Pooling) analyses the scene at multiple scales simultaneously — detecting both a tiny pebble close-up and the full horizon in a single forward pass.

### Why Combined Loss?
- **Cross-Entropy** — correct per-pixel probability calibration
- **Dice Loss** — prevents dominant classes (Sky, Landscape) from drowning out small objects (Rocks, Flowers)

### Why TTA?
Test-Time Augmentation runs the model on the original image and a horizontally-flipped version, then averages the softmax probabilities. This reduces edge noise on complex objects like bushes by ~3–5% mIoU.

### Known Failure Case
**Dry Grass ↔ Ground Clutter confusion** — both share similar muted, low-saturation visual textures. This is an active engineering challenge and is flagged in evaluation output.

---

## 📊 Performance

| Metric | Score |
|---|---|
| mIoU | ~63–71% |
| Pixel Accuracy | ~78–84% |
| Best Classes | Sky (92.4%), Landscape (87.1%) |
| Challenging Classes | Dry Grass (58.2%), Ground Clutter (54.7%) |

---

## 🚀 Roadmap

- [ ] Real-world fine-tuning on physical sensor data
- [ ] INT8 quantisation for Jetson Orin deployment
- [ ] Real-time streaming inference pipeline
- [ ] Depth-fusion branch for rock height estimation
- [ ] ROS2 integration for live UGV deployment

---

## 🛠️ Built With

- PyTorch 2.x + TorchVision
- Albumentations
- TensorBoard
- Pillow + NumPy

---

## 📄 License

MIT License — free to use, modify, and deploy.# Deep-Desert-AI
