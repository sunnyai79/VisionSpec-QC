# VisionSpec QC — Automated PCB Defect Detection & Production Web Application
### Computer Vision Pipeline for High-Speed Manufacturing Quality Control

---

## ⚠️ Disclaimer

This project is intended **for research and educational purposes only**. It is a demonstration
of computer vision and transfer learning techniques applied to PCB defect detection and is
**not validated, certified, or approved for use in any real manufacturing, safety-critical,
or production environment**. Results may be inaccurate, incomplete, or unreliable when applied
to real-world hardware. The authors accept no liability for any damages, losses, or defects
arising from the use of this software. Always consult qualified engineers and conduct
thorough testing before deploying any automated inspection system in a live setting.

---

## Model Selection: MobileNetV2 ✓

| Criterion | MobileNetV2 | ResNet50 | Decision |
|---|---|---|---|
| Parameters | 3.4M | 25.6M | MobileNetV2 ✓ |
| Inference Speed (CPU) | >30 FPS | ~8 FPS | MobileNetV2 ✓ |
| Model Size | ~14 MB | ~98 MB | MobileNetV2 ✓ |
| Top-1 ImageNet Acc | 71.8% | 74.9% | ResNet50 ✓ |
| Production >10 FPS | ✓ YES | ✗ NO | MobileNetV2 ✓ |
| Memory Footprint | Low | High | MobileNetV2 ✓ |

**Verdict:** MobileNetV2 wins. ResNet50's marginal accuracy advantage (~3%) does not
justify its 7× parameter count and its failure to meet the >10 FPS production requirement.

---

## Project Structure

```
VisionSpec/
│
├── README.md                      → Project guide
├── requirements.txt               → Dependencies list
├── app.py                         → Flask app
│
├── results/                       → Output visuals
│   ├── week1_augmented_batch.png      → Augmented images
│   ├── week1_class_samples.png        → Sample images
│   ├── week2_confusion_matrix.png     → Confusion matrix
│   ├── week2_learning_curves.png      → Training curves
│   ├── week3_gradcam_detail.png       → GradCAM detail
│   └── week3_gradcam_grid.png         → GradCAM grid
│
├── scripts/                       → Processing scripts
│   ├── week1_data_preparation.py      → Data preprocessing
│   ├── week2_model_training.py        → Model training
│   ├── week3_grad_cam.py              → GradCAM generation
│   └── week4_live_inference.py        → Live inference
│
├── templates/                     → Web UI
│   └── index.html                     → Frontend page
│
├── test/                         → Test images
│
├── models/                        → Trained models
│   ├── best_threshold.txt             → Threshold value
│   ├── visionspec_phase1_best.keras   → Phase1 model without fine-tune
│   ├── visionspec_phase2_best.keras   → Phase2 model with fine-tune
│   ├── visionspec_qc_final.h5         → Final model
│   ├── visionspec_qc_final.keras      → Keras model
│   ├── visionspec_qc_quantized.tflite → Mobile model
│   └── visionspec_qc_savedmodel/ → TF model
│       ├── fingerprint.pb             → Metadata
│       ├── saved_model.pb             → Model graph
│       └── variables/                 → Model weights
│           ├── variables.data...      → Weights data
│           └── variables.index        → Weights index
│
└── dataset/                       → Image data
    ├── val/                       → Validation data
    │   ├── defect                 → Defect images
    │   └── pass                   → Normal images
    └── train/                     → Training data
        ├── defect                 → Defect images
        └── pass                   → Normal images
```

---

## Setup

```bash
# Core pipeline
pip install -r requirements.txt
```

---

## Week 1 — Data Preparation

**Goal:** Build a robust augmentation pipeline to handle real production variability
(lighting, camera angle, positioning, lens distortion).

```bash
python week1_data_preparation.py
```

**Outputs:**
- `dataset/` — 750 train + 150 val synthetic PCB images (pass/defect)
- `week1_augmented_batch.png` — Grid of 32 augmented training images
- `week1_single_augmented.png` — One image → 6 augmented variants

**Augmentation Config:**

| Transform | Value | Rationale |
|---|---|---|
| rotation_range | ±25° | PCB orientation variance on conveyor |
| zoom_range | 20% | Camera distance variation |
| brightness_range | [0.6, 1.4] | Factory lighting changes |
| horizontal_flip | True | PCBs can flip |
| shear_range | 5° | Lens distortion simulation |
| fill_mode | nearest | Avoids black border artifacts |

---

## Week 2 — Core Modelling

**Goal:** Train MobileNetV2 via two-phase transfer learning.
Phase 1 freezes the backbone to train only the custom head.
Phase 2 unfreezes the top 30 layers for fine-tuning.

```bash
python week2_model_training.py
```

**Architecture:**
```
Input (224×224×3)
    │
MobileNetV2 Backbone (ImageNet pretrained)
    │  Phase 1: all frozen
    │  Phase 2: top-30 layers trainable (LR=1e-5)
GlobalAveragePooling2D   → 1280-d feature vector
BatchNormalization
Dense(256, relu)
Dropout(0.40)
Dense(128, relu)
Dropout(0.30)
Dense(1, sigmoid)        → defect probability [0, 1]
```

**Outputs:**
- `week2_learning_curves.png` — Train/val accuracy + loss over both phases
- `week2_confusion_matrix.png` — Confusion matrix with production metrics
- `models/visionspec_qc_final.h5`
- `models/visionspec_qc_savedmodel/`

**Key production metrics monitored:**
- **Recall (Sensitivity)** — Must be HIGH; missed defects (FN) are critical failures
- **Precision** — Affects yield; false rejects (FP) are costly but less critical
- **Val Loss** — Monitored for overfitting via EarlyStopping (patience=6)

---

## Week 3 — Grad-CAM Interpretability

**Goal:** Generate spatial heatmaps showing WHERE the model is looking.
Heatmaps must highlight solder joints, pads, and component bodies
— NOT random PCB substrate or background noise.

```bash
python week3_grad_cam.py
```

**How Grad-CAM works:**
```
1. Forward pass → record feature maps at "out_relu" layer
2. Compute gradient of defect score w.r.t. each feature map channel
3. Pool gradients → importance weight per channel (α_k)
4. Weighted sum: L_cam = ReLU( Σ α_k · A_k )
5. Upsample to input size → overlay as colourmap
```

**Target layer:** `out_relu` (final Conv output before Global Average Pooling in MobileNetV2)

**Outputs:**
- `week3_gradcam_grid.png` — 8 images (4 pass + 4 defect) with 3-column layout
- `week3_gradcam_detail.png` — Single defect image across 4 colourmap options

**Verification checks:**
- Hot pixel ratio 2–40% → focused heatmap ✓
- Hot pixel ratio <2% → sparse (check layer name)
- Hot pixel ratio >40% → diffuse (may highlight background)

---

## Week 4 — Inference Optimisation & Live Demo

**Goal:** Achieve >10 FPS production line speed with full Grad-CAM overlay.
Model optimized via TFLite Dynamic Range Quantization.

```bash
# Run live simulation demo (no camera needed)
python week4_live_inference.py

# Run with real webcam
python week4_live_inference.py --webcam

# Run inference benchmark only
python week4_live_inference.py --benchmark

# Disable Grad-CAM (fastest)
python week4_live_inference.py --no-gradcam
```

**Optimisation pipeline:**
```
Keras .h5  (float32, ~14 MB)
    │
TFLite Conversion
    │
Dynamic Range Quantization
    │  Weights → INT8 at conversion
    │  Activations → quantized at runtime
    │
TFLite .tflite  (~3.5 MB, ~1.8× faster)
    │
Multi-threaded interpreter (4 CPU threads)
    │
>30 FPS  ✓
```

**Live Demo Controls:**

| Key | Action |
|---|---|
| Q / Escape | Quit |
| G | Toggle Grad-CAM overlay |
| S | Save current frame as PNG |
| Space | Pause / Resume |

**Outputs:**
- `week4_benchmark.png` — Latency distribution + per-frame plot

---

## Web Application

A factory-floor quality control dashboard for real-time PCB defect detection.
Upload any PCB image, folder and by using camera and get an instant PASS/DEFECT verdict with Grad-CAM heatmap.

### Run the web app

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

### Features

| Feature | Detail |
|---|---|
| Drag & drop upload | PNG, JPG, BMP up to 16 MB |
| TFLite inference | Quantized MobileNetV2, ~80ms on CPU |
| Grad-CAM overlay | Shows exactly WHERE the model found the defect |
| Session dashboard | Live pass/defect counts, rates, history log |
| Inspection history | Click any past result to re-view it |
| Session reset | Clear all stats for a new shift |

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web interface |
| `/api/inspect` | POST | Upload image → get prediction + Grad-CAM |
| `/api/history` | GET | Last 50 inspection records |
| `/api/stats` | GET | Session totals |
| `/api/reset` | POST | Clear session |

### `/api/inspect` response example

```json
{
  "id": 1,
  "label": "DEFECT",
  "score": 0.2341,
  "defect_prob": 0.7659,
  "confidence": 0.7659,
  "infer_ms": 82.4,
  "total_ms": 247.1,
  "thumb_b64": "...",
  "gradcam_b64": "...",
  "stats": {
    "total": 5,
    "pass": 3,
    "defect": 2,
    "pass_rate": 60.0,
    "defect_rate": 40.0
  }
}
```

### Model files required

The app looks for models in `../models/` relative to `app.py`.
Make sure these files exist from your Week 2 training:

- `models/visionspec_qc_final.keras` (or `.h5`)
- `models/visionspec_qc_quantized.tflite`
- `models/best_threshold.txt`

If you place `VisionSpec_WebApp/` inside `VisionSpec_QC/`, the path resolves automatically.
Otherwise edit `MODEL_DIR` in `app.py`:

```python
MODEL_DIR = r"C:\path\to\your\models"
```

### Production deployment (optional)

For production deployment on a server, use `waitress` instead of Flask dev server:

```bash
pip install waitress
waitress-serve --host=0.0.0.0 --port=5000 app:app
```

Or with `gunicorn` on Linux:

```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:5000 --timeout 120 app:app
```

Note: Use `-w 1` (single worker) because TFLite and Grad-CAM models are not thread-safe across processes.

---

## Defect Types Detected

| Defect Type | Description | Severity |
|---|---|---|
| Solder Bridge | Excess solder connecting adjacent pads | HIGH |
| Missing Component | Component absent from required location | HIGH |
| Scratch / Trace Damage | Copper trace damaged or interrupted | MEDIUM |
| Burnt Spot | Thermal damage from soldering | HIGH |

---

## Production Requirements Checklist

- [x] Real-time data augmentation via `ImageDataGenerator`
- [x] Pre-trained MobileNetV2 backbone (ImageNet weights)
- [x] Frozen base + custom head (Phase 1)
- [x] Fine-tuning with unfrozen top layers (Phase 2)
- [x] Learning curves checked for overfitting
- [x] EarlyStopping + ReduceLROnPlateau callbacks
- [x] Grad-CAM on final Conv layer (`out_relu`)
- [x] Heatmap verification (focused, not diffuse)
- [x] TFLite Dynamic Range Quantization
- [x] >10 FPS production throughput (TFLite on CPU)
- [x] OpenCV live demo with HUD overlay
- [x] Grad-CAM integrated into live demo
- [x] Flask web application with drag & drop upload
- [x] Session dashboard with live pass/defect stats
- [x] REST API for programmatic inspection access
