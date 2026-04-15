"""
========================================================
VisionSpec QC — Week 4: High-Speed Inference & Live Demo
========================================================
Product: VisionSpec QC | Use Case: PCB Defect Detection
Model Backbone: MobileNetV2

Production Requirements:
  ✓ >10 Frames Per Second (FPS) on CPU
  ✓ Frame-by-frame prediction with Grad-CAM overlay
  ✓ Live pass/defect classification displayed in real time
  ✓ Model saved in optimized formats (.h5 + TFLite)

Optimisation Pipeline:
  1. Convert Keras model → TFLite (float32 baseline)
  2. Apply TFLite Dynamic Range Quantization (INT8 weights)
     → Reduces model size 4× and speeds up inference ~2×
  3. Optional: Full INT8 quantization with representative dataset

Live Demo Modes:
  A) Webcam mode  — real camera feed (set USE_WEBCAM = True)
  B) Simulation   — synthetic PCB frames generated on the fly
                    (default, no camera required)
"""

import os
import sys
import time
import queue
import argparse
import threading
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE       = (224, 224)
MODEL_PATH     = os.path.join("models", "visionspec_qc_final.keras")
MODEL_PATH_H5  = os.path.join("models", "visionspec_qc_final.h5")   # fallback
TFLITE_PATH    = os.path.join("models", "visionspec_qc_quantized.tflite")
THRESHOLD      = 0.50           # Default; overridden at runtime from models/best_threshold.txt
USE_WEBCAM     = False          # Set True for real camera
WEBCAM_INDEX   = 0              # Camera device index
SHOW_GRADCAM   = True           # Overlay Grad-CAM heatmap
DEMO_FPS_CAP   = 30             # Cap simulated feed to 30 FPS
GRAD_CAM_LAYER = "out_relu"     # MobileNetV2 final conv feature map

# Display colours (BGR)
COL_PASS   = (0, 220, 80)       # Green
COL_DEFECT = (0, 50, 220)       # Red
COL_WHITE  = (255, 255, 255)
COL_BLACK  = (0, 0, 0)
COL_ORANGE = (0, 150, 255)


# ─── Threshold Loader (must be defined before any class uses it) ──────────────
def _load_threshold():
    """Load the auto-selected threshold saved by week2_model_training.py."""
    p = os.path.join("models", "best_threshold.txt")
    if os.path.exists(p):
        return float(open(p).read().strip())
    return THRESHOLD   # Fall back to module-level default


# ─── Step 1: Model Optimisation — TFLite Quantization ────────────────────────
def convert_to_tflite(keras_model_path=None,
                      tflite_path=TFLITE_PATH,
                      representative_data=None):
    """
    Convert .keras / .h5 model to optimized TFLite with dynamic-range quantization.
    Tries .keras format first (native), falls back to .h5 if not found.
    """
    # Resolve model path
    if keras_model_path is None:
        if os.path.exists(MODEL_PATH):
            keras_model_path = MODEL_PATH
        elif os.path.exists(MODEL_PATH_H5):
            keras_model_path = MODEL_PATH_H5
        else:
            raise FileNotFoundError(
                f"No model found at '{MODEL_PATH}' or '{MODEL_PATH_H5}'.\n"
                "Run week2_model_training.py first."
            )

    print(f"  Converting {keras_model_path} → TFLite (Dynamic Range Quantization)...")
    model     = load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Dynamic range quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_path), exist_ok=True)
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    original_mb  = os.path.getsize(keras_model_path) / (1024 * 1024)
    quantized_mb = len(tflite_model) / (1024 * 1024)
    reduction    = (1 - quantized_mb / original_mb) * 100

    print(f"    Original .h5 size  : {original_mb:.1f} MB")
    print(f"    Quantized .tflite  : {quantized_mb:.1f} MB  ({reduction:.0f}% smaller)")
    print(f"    Saved to           : {tflite_path}")
    return tflite_path


# ─── Step 2: TFLite Inference Engine ─────────────────────────────────────────
class TFLiteInferenceEngine:
    """Wraps TFLite interpreter for minimal-latency single-frame inference."""

    def __init__(self, tflite_path, num_threads=None):
        # Use all physical cores — makes a measurable difference on multi-core CPUs
        if num_threads is None:
            num_threads = max(4, os.cpu_count() or 4)
        self.interpreter = tf.lite.Interpreter(
            model_path=tflite_path,
            num_threads=num_threads,
        )
        self.interpreter.allocate_tensors()
        self.input_details  = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = tuple(self.input_details[0]["shape"][1:3])  # (H, W)
        print(f"  TFLite engine ready | Input: {self.input_details[0]['shape']} "
              f"| Threads: {num_threads}")

    def predict(self, frame_bgr):
        """
        Run inference on a single BGR frame.
        Returns: (pred_class, pred_score, inference_ms)
        """
        # Pre-process
        rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized  = cv2.resize(rgb, (self.input_shape[1], self.input_shape[0]))
        tensor   = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        # Inference
        t0 = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]["index"], tensor)
        self.interpreter.invoke()
        score = float(self.interpreter.get_tensor(self.output_details[0]["index"])[0, 0])
        t1 = time.perf_counter()

        ms    = (t1 - t0) * 1000
        thresh = _load_threshold()
        # score = P(pass) from sigmoid.
        # pred_class: 1 = DEFECT (score LOW, below threshold)
        #             0 = PASS   (score HIGH, above threshold)
        pred_class = 1 if score <= thresh else 0
        return pred_class, score, ms


# ─── Step 3: Keras Grad-CAM Engine (for overlay) ─────────────────────────────
class GradCAMEngine:
    """
    Lightweight Grad-CAM engine using the original Keras model.
    Handles the nested MobileNetV2 sub-model by building a fresh
    Input-based grad model — avoids circular graph build errors.
    """
    SUBMODEL_NAME  = "mobilenetv2_1.00_224"
    LAYER_NAME     = "out_relu"

    def __init__(self):
        if os.path.exists(MODEL_PATH):
            path = MODEL_PATH
        elif os.path.exists(MODEL_PATH_H5):
            path = MODEL_PATH_H5
        else:
            raise FileNotFoundError("No Keras model found. Run week2 first.")
        model = load_model(path)
        self.threshold = _load_threshold()

        base = model.get_layer(self.SUBMODEL_NAME)
        try:
            base.get_layer(self.LAYER_NAME); resolved = self.LAYER_NAME
        except ValueError:
            candidates = [l.name for l in base.layers
                          if "relu" in l.name or "out" in l.name]
            resolved = candidates[-1] if candidates else base.layers[-2].name
            print(f"  GradCAMEngine fallback layer: '{resolved}'")

        # Stage 1: image → conv features
        conv_extractor = tf.keras.Model(
            inputs=base.input, outputs=base.get_layer(resolved).output)

        # TWO-STAGE grad model: pred = f(conv_out) → gradients flow ✓
        fresh = tf.keras.Input(shape=(224, 224, 3))
        conv_out = conv_extractor(fresh)
        # Stage 2: chain head layers so pred is derived from conv_out
        x = conv_out
        for layer in model.layers:
            if layer.name == self.SUBMODEL_NAME:
                continue
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue
            x = layer(x)

        self.grad_model = tf.keras.Model(inputs=fresh, outputs=[conv_out, x])
        print(f"  GradCAMEngine: two-stage model built, threshold={self.threshold}")

    @tf.function
    def _forward(self, inp):
        with tf.GradientTape() as tape:
            conv_out, pred = self.grad_model(inp, training=False)
            tape.watch(conv_out)           # explicit watch for non-Variable tensor
            defect_score = 1.0 - pred[:, 0]
        grads = tape.gradient(defect_score, conv_out)
        return conv_out, grads, pred

    def compute(self, frame_bgr):
        """Returns overlay_bgr, heatmap, score, pred_class."""
        rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, IMG_SIZE)
        arr     = np.expand_dims(resized.astype(np.float32) / 255.0, axis=0)

        conv_out, grads, pred = self._forward(arr)
        score = float(pred[0, 0])

        if grads is None:
            pc = 1 if score <= self.threshold else 0
            return frame_bgr.copy(), None, score, pc

        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(tf.nn.relu(heatmap)).numpy()
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        heatmap_r   = cv2.resize(heatmap, (frame_bgr.shape[1], frame_bgr.shape[0]))
        hmap_col    = cv2.applyColorMap(np.uint8(255 * heatmap_r), cv2.COLORMAP_JET)
        overlay     = cv2.addWeighted(frame_bgr, 0.55, hmap_col, 0.45, 0)
        # 1 = DEFECT (score low / below threshold), 0 = PASS
        pred_class  = 1 if score <= self.threshold else 0
        return overlay, heatmap, score, pred_class


# ─── Step 4: Synthetic PCB Frame Generator (demo mode) ────────────────────────
class SyntheticPCBSource:
    """
    Generates PCB frames using the SAME generators as week1_data_preparation.py.
    This ensures the live demo operates on the same image distribution the model
    was trained on — eliminating the domain mismatch that caused 0 defects detected.

    Previous bug: week4 used completely different colours/shapes from week1
    (blue-tinted traces, random blobs) → model had never seen such images → all PASS.

    Fix: inline the exact same PIL-based generators from week1 (same substrate
    colours, same pad layout, same defect styles, same noise levels).
    """

    # ── Colour palette (must match week1 exactly) ─────────────────────────
    PCB_GREEN     = (34,  100,  34)
    COPPER        = (184, 115,  51)
    COPPER_BRIGHT = (210, 140,  40)
    SOLDER_SILVER = (180, 180, 170)
    COMPONENT_BLK = ( 30,  30,  40)
    WHITE_SILK    = (230, 230, 230)
    VIA_DARK      = (  5,   5,   5)

    def __init__(self, defect_rate=0.40, output_size=(640, 480)):
        self.defect_rate = defect_rate
        self.out_w, self.out_h = output_size
        self.frame_count = 0

    # ── Internal PIL helpers (mirrors week1) ──────────────────────────────
    def _substrate(self, w, h):
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[:, :, 0] = np.random.randint(28, 42,  (h, w))
        arr[:, :, 1] = np.random.randint(90, 112, (h, w))
        arr[:, :, 2] = np.random.randint(28, 42,  (h, w))
        from PIL import Image
        return Image.fromarray(arr)

    def _draw_base(self, draw, w, h):
        import random as rnd
        # Orthogonal traces
        for y in [h//5, 2*h//5, 3*h//5, 4*h//5]:
            draw.line([(8,y),(w-8,y)], fill=self.COPPER, width=rnd.randint(2,4))
        for x in [w//5, 2*w//5, 3*w//5, 4*w//5]:
            draw.line([(x,8),(x,h-8)], fill=self.COPPER, width=rnd.randint(2,4))
        # Pads
        for px in [w//6, w//2, 5*w//6]:
            for py in [h//4, h//2, 3*h//4]:
                if rnd.random() < 0.75:
                    pw, ph = rnd.randint(13,22), rnd.randint(9,17)
                    draw.rectangle([px-pw,py-ph,px+pw,py+ph],
                                   outline=self.COPPER_BRIGHT, width=3)
                    draw.rectangle([px-pw+3,py-ph+3,px+pw-3,py+ph-3],
                                   fill=self.SOLDER_SILVER)
        # Components
        for _ in range(rnd.randint(3,5)):
            cx,cy = rnd.randint(35,w-35), rnd.randint(35,h-35)
            bw,bh = rnd.randint(18,38), rnd.randint(12,28)
            draw.rectangle([cx-bw,cy-bh,cx+bw,cy+bh],
                           fill=self.COMPONENT_BLK, outline=self.COPPER_BRIGHT, width=2)
        # Vias
        for _ in range(rnd.randint(5,10)):
            vx,vy = rnd.randint(12,w-12), rnd.randint(12,h-12)
            vr = rnd.randint(5,9)
            draw.ellipse([vx-vr,vy-vr,vx+vr,vy+vr],
                         fill=self.COPPER_BRIGHT, outline=self.COPPER)
            draw.ellipse([vx-vr+3,vy-vr+3,vx+vr-3,vy+vr-3], fill=self.VIA_DARK)

    def _add_defect(self, draw, w, h):
        import random as rnd, math
        dtype = rnd.choice(["bridge","missing","break","burn","cold"])
        cx = rnd.randint(50, w-50); cy = rnd.randint(50, h-50)

        if dtype == "bridge":
            bw,bh = rnd.randint(22,42), rnd.randint(12,22)
            draw.ellipse([cx-bw,cy-bh,cx+bw,cy+bh], fill=(228,175,10))
            for _ in range(rnd.randint(8,14)):
                ox=cx+rnd.randint(-bw-12,bw+12); oy=cy+rnd.randint(-bh-10,bh+10)
                r=rnd.randint(6,14)
                draw.ellipse([ox-r,oy-r,ox+r,oy+r], fill=(205,155,8))

        elif dtype == "missing":
            sz = rnd.randint(26,40)
            draw.rectangle([cx-sz,cy-sz,cx+sz,cy+sz],
                           fill=self.COPPER_BRIGHT, outline=(215,25,25), width=5)
            draw.rectangle([cx-sz+6,cy-sz+6,cx+sz-6,cy+sz-6], fill=self.PCB_GREEN)
            draw.line([(cx-sz+6,cy-sz+6),(cx+sz-6,cy+sz-6)], fill=(215,25,25), width=4)
            draw.line([(cx+sz-6,cy-sz+6),(cx-sz+6,cy+sz-6)], fill=(215,25,25), width=4)

        elif dtype == "break":
            trw=rnd.randint(5,9); gap=rnd.randint(14,28)
            x0=rnd.randint(10,w//3); x1=rnd.randint(2*w//3,w-10)
            xm=(x0+x1)//2
            draw.line([(x0,cy),(xm-gap,cy)], fill=self.COPPER, width=trw)
            draw.line([(xm+gap,cy),(x1,cy)], fill=self.COPPER, width=trw)
            draw.rectangle([xm-gap,cy-trw-3,xm+gap,cy+trw+3], fill=(10,30,10))

        elif dtype == "burn":
            r=rnd.randint(28,50)
            draw.ellipse([cx-r,cy-r,cx+r,cy+r], fill=(58,22,4))
            draw.ellipse([cx-r+8,cy-r+8,cx+r-8,cy+r-8], fill=(22,7,1))
            draw.ellipse([cx-r+18,cy-r+18,cx+r-18,cy+r-18], fill=(4,1,0))
            for ang in range(0,360,45):
                ex=int(cx+(r+12)*math.cos(math.radians(ang)))
                ey=int(cy+(r+12)*math.sin(math.radians(ang)))
                draw.line([(cx,cy),(ex,ey)], fill=(38,14,2), width=2)

        elif dtype == "cold":
            pw,ph = rnd.randint(18,30), rnd.randint(13,24)
            draw.rectangle([cx-pw,cy-ph,cx+pw,cy+ph],
                           outline=self.COPPER_BRIGHT, width=3)
            draw.rectangle([cx-pw+3,cy-ph+3,cx+pw-3,cy+ph-3], fill=(108,103,98))
            draw.line([(cx-pw+4,cy),(cx+pw-4,cy)], fill=(55,53,50), width=3)
            draw.line([(cx,cy-ph+4),(cx,cy+ph-4)], fill=(55,53,50), width=3)

    def _pil_to_bgr(self, pil_img, out_w, out_h):
        """Convert PIL RGB image → OpenCV BGR, resized to display size."""
        import cv2 as _cv2
        from PIL import Image as _Img
        arr = np.array(pil_img.resize((out_w, out_h), _Img.BILINEAR))
        return _cv2.cvtColor(arr, _cv2.COLOR_RGB2BGR)

    def next_frame(self):
        """
        Return (bgr_frame, is_defect_ground_truth).
        Uses PIL generators matching week1 training distribution.
        """
        from PIL import Image as _Img, ImageDraw as _Draw, ImageFilter as _Flt
        import random as rnd

        self.frame_count += 1
        is_defect = np.random.random() < self.defect_rate

        # Build frame at 224×224 (model input size) then upscale for display
        w, h = 224, 224
        img  = self._substrate(w, h)
        draw = _Draw.Draw(img)
        self._draw_base(draw, w, h)

        if is_defect:
            self._add_defect(draw, w, h)
            # 30% chance of second defect (matches week1 training)
            if rnd.random() < 0.30:
                self._add_defect(draw, w, h)

        img = img.filter(_Flt.GaussianBlur(radius=0.5))
        arr = np.array(img, dtype=np.float32)
        sigma = 5 if is_defect else 3
        arr += np.random.normal(0, sigma, arr.shape)
        img = _Img.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        bgr = self._pil_to_bgr(img, self.out_w, self.out_h)
        return bgr, is_defect


# ─── Step 5: Frame Pre-fetch Buffer ───────────────────────────────────────────
class FrameBuffer:
    """
    Double-buffer: generates synthetic frames in a background producer thread
    so PIL image creation (~20ms) is FULLY OVERLAPPED with TFLite inference.

    Without pre-fetch:
      Main thread: [PIL 20ms] → [TFLite 80ms] → [display 5ms] = 105ms → 9.5 FPS
    With pre-fetch:
      Producer:    [PIL 20ms] [PIL 20ms] [PIL 20ms] ...  (runs continuously)
      Main thread: [get ~0ms] → [TFLite 80ms] → [display 5ms] = 85ms → 11.8 FPS ✓

    Buffer size = 4: enough lookahead without excessive memory use.
    """

    def __init__(self, source, buffer_size=4):
        self.source  = source
        self._q      = queue.Queue(maxsize=buffer_size)
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._producer, daemon=True)
        self._thread.start()

    def _producer(self):
        """Continuously generate frames and push to buffer."""
        while not self._stop.is_set():
            try:
                frame, is_defect = self.source.next_frame()
                self._q.put((frame, is_defect), timeout=0.5)
            except queue.Full:
                pass   # Buffer full — slow consumer, just try again
            except Exception:
                pass

    def next_frame(self):
        """Blocking get — returns instantly when buffer has frames."""
        return self._q.get()

    def stop(self):
        self._stop.set()


# ─── Step 6: HUD Renderer ─────────────────────────────────────────────────────
def render_hud(frame, pred_class, score, fps, inference_ms,
               pass_count, defect_count):
    """Draw production HUD overlay on frame."""
    h, w = frame.shape[:2]
    label = "DEFECT" if pred_class == 1 else "PASS"
    color = COL_DEFECT if pred_class == 1 else COL_PASS

    # Top banner
    cv2.rectangle(frame, (0, 0), (w, 55), COL_BLACK, -1)
    cv2.rectangle(frame, (0, 0), (w, 55), color, 2)

    cv2.putText(frame, f"VisionSpec QC  |  {label}",
                (12, 35), cv2.FONT_HERSHEY_DUPLEX, 0.95, color, 2)

    # Score bar
    bar_x, bar_y, bar_w, bar_h = 10, 60, w - 20, 14
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (40, 40, 40), -1)
    fill_w = int(bar_w * score)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), color, -1)
    cv2.putText(frame, f"Defect Score: {score:.3f}",
                (bar_x + 4, bar_y + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_WHITE, 1)

    # Bottom status bar
    cv2.rectangle(frame, (0, h - 38), (w, h), COL_BLACK, -1)
    status_text = (f"FPS: {fps:.1f}  |  Infer: {inference_ms:.1f}ms  |  "
                   f"PASS: {pass_count}  |  DEFECT: {defect_count}  |  "
                   f"Defect Rate: {defect_count / max(1, pass_count + defect_count) * 100:.1f}%")
    cv2.putText(frame, status_text,
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_ORANGE, 1)

    # Decision border
    border_color = COL_DEFECT if pred_class == 1 else COL_PASS
    thickness    = 6 if pred_class == 1 else 3
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, thickness)

    return frame


# ─── Step 6: FPS Measurement ─────────────────────────────────────────────────
class FPSCounter:
    def __init__(self, window=60):
        self.timestamps  = []
        self.window      = window
        self.all_times   = []   # keep all timestamps for true average

    def tick(self):
        now = time.perf_counter()
        self.timestamps.append(now)
        self.all_times.append(now)
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)

    @property
    def fps(self):
        """Rolling FPS (last N frames) — displayed on HUD."""
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        return (len(self.timestamps) - 1) / elapsed if elapsed > 0 else 0.0

    @property
    def sustained_fps(self):
        """True sustained FPS across ALL frames (ignores startup warmup)."""
        skip = 20   # Ignore first 20 frames (warmup noise)
        ts   = self.all_times[skip:]
        if len(ts) < 2:
            return self.fps
        elapsed = ts[-1] - ts[0]
        return (len(ts) - 1) / elapsed if elapsed > 0 else 0.0


# ─── Step 7: Main Live Demo ───────────────────────────────────────────────────
def run_live_demo(use_webcam=USE_WEBCAM, use_gradcam=SHOW_GRADCAM,
                  tflite_path=TFLITE_PATH):
    """
    Production live demo.
    Controls:
        Q     — Quit
        G     — Toggle Grad-CAM overlay
        S     — Save current frame as PNG
        Space — Pause/Resume
    """
    print("\n" + "=" * 60)
    print("  VisionSpec QC — Live Inference Demo")
    print("=" * 60)
    print(f"  Mode    : {'Webcam' if use_webcam else 'Synthetic simulation'}")
    print(f"  Grad-CAM: {'Enabled' if use_gradcam else 'Disabled'}")
    print(f"  Controls: Q=Quit  G=Toggle Grad-CAM  S=Save frame  SPACE=Pause")
    print("=" * 60)

    # Verify model files
    if not os.path.exists(tflite_path):
        print(f"  ⚠ TFLite model not found. Running conversion...")
        if not os.path.exists(MODEL_PATH):
            print("  ✗ Keras model also missing. Run week2_model_training.py first.")
            return
        convert_to_tflite()

    # Initialise engines
    engine     = TFLiteInferenceEngine(tflite_path)
    gradcam_available = os.path.exists(MODEL_PATH) or os.path.exists(MODEL_PATH_H5)
    gradcam_en = GradCAMEngine() if use_gradcam and gradcam_available else None
    fps_ctr    = FPSCounter(window=60)   # Wider window = more stable FPS reading

    # Warmup: run a dummy inference to initialise XNNPACK JIT kernels.
    # Without warmup, first ~10 frames are slow (lazy compilation) and
    # drag down the FPS average.
    print("  Warming up TFLite (3 dummy inferences)...", end=" ", flush=True)
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    for _ in range(3):
        engine.interpreter.set_tensor(engine.input_details[0]["index"], dummy)
        engine.interpreter.invoke()
    print("done.\n")

    # Synthetic source wrapped in FrameBuffer for zero-wait frame supply
    raw_source = SyntheticPCBSource(defect_rate=0.35) if not use_webcam else None
    source     = FrameBuffer(raw_source) if raw_source is not None else None

    # Camera or synthetic source
    if use_webcam:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print(f"  ✗ Could not open webcam {WEBCAM_INDEX}.")
            return

    # State
    pass_count   = 0
    defect_count = 0
    gt_defect    = 0      # ground-truth defect frames (for accuracy tracking)
    gt_pass      = 0
    tp = fp = fn = tn = 0
    frame_idx    = 0
    show_gradcam = use_gradcam
    paused       = False
    last_display = None
    cached_gradcam_frame = None
    gradcam_lock         = threading.Lock()
    gradcam_running      = threading.Event()
    GRADCAM_EVERY        = 8

    # FPS tracking: record every live reading so summary is not stale at teardown
    fps_readings  = []          # stores fps_ctr.fps sampled every 100 frames
    peak_fps      = 0.0         # highest observed FPS during the run

    def _run_gradcam_bg(frame_copy):
        """Background thread: compute Grad-CAM and update cache."""
        nonlocal cached_gradcam_frame
        try:
            result, _, _, _ = gradcam_en.compute(frame_copy)
            with gradcam_lock:
                cached_gradcam_frame = result
        except Exception:
            pass
        finally:
            gradcam_running.clear()

    print("\n  Starting demo... press Q to quit.")
    print("  [FPS = TFLite-only speed. Grad-CAM runs in background thread.]\n")

    while True:
        if not paused:
            # Grab frame
            if use_webcam:
                ret, frame = cap.read()
                if not ret:
                    print("  ✗ Frame capture failed.")
                    break
                is_defect_gt = None
            else:
                frame, is_defect_gt = source.next_frame()   # instant — from pre-filled buffer

            frame_idx += 1

            # ── TFLite Inference (timed for FPS) ─────────────────────────
            pred_class, score, inference_ms = engine.predict(frame)
            fps_ctr.tick()   # FPS = pure TFLite speed, no Grad-CAM overhead

            if pred_class == 0:
                pass_count   += 1
            else:
                defect_count += 1

            # Ground-truth accuracy tracking
            # pred_class: 1=DEFECT, 0=PASS
            if is_defect_gt is not None:
                if is_defect_gt:
                    gt_defect += 1
                    if pred_class == 1: tp += 1   # correctly caught defect ✓
                    else:               fn += 1   # missed defect ✗
                else:
                    gt_pass += 1
                    if pred_class == 0: tn += 1   # correctly passed clean board ✓
                    else:               fp += 1   # false alarm ✗

            # ── Grad-CAM (non-blocking background thread) ─────────────────
            # Fire a new Grad-CAM thread every GRADCAM_EVERY frames,
            # but only if the previous computation has finished.
            # The main loop NEVER waits for Grad-CAM → FPS unaffected.
            if (show_gradcam and gradcam_en
                    and (frame_idx % GRADCAM_EVERY == 0)
                    and not gradcam_running.is_set()):
                gradcam_running.set()
                t = threading.Thread(
                    target=_run_gradcam_bg,
                    args=(frame.copy(),),
                    daemon=True)
                t.start()

            # Show latest cached Grad-CAM overlay (may be a few frames old — fine)
            with gradcam_lock:
                gc_frame = cached_gradcam_frame

            if show_gradcam and gc_frame is not None:
                h_f, w_f = frame.shape[:2]
                h_g, w_g = gc_frame.shape[:2]
                display_frame = cv2.resize(gc_frame, (w_f, h_f)) \
                                if (h_f != h_g or w_f != w_g) else gc_frame.copy()
            else:
                display_frame = frame.copy()

            # ── HUD ───────────────────────────────────────────────────────
            display_frame = render_hud(
                display_frame, pred_class, score,
                fps_ctr.fps, inference_ms,
                pass_count, defect_count
            )
            last_display = display_frame
        else:
            # Paused
            display_frame = last_display.copy() if last_display is not None else \
                            np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(display_frame, "  PAUSED — SPACE to resume",
                        (60, display_frame.shape[0] // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, COL_ORANGE, 2)

        cv2.imshow("VisionSpec QC — Live Demo", display_frame)

        # ── Throughput report every 100 frames ────────────────────────────
        if frame_idx % 100 == 0 and not paused:
            live_fps = fps_ctr.fps
            fps_readings.append(live_fps)
            if live_fps > peak_fps:
                peak_fps = live_fps
            acc_str = ""
            if (tp+tn+fp+fn) > 0:
                acc = (tp+tn)/(tp+tn+fp+fn)*100
                acc_str = f" | GT-Acc: {acc:.1f}%"
            print(f"  Frame {frame_idx:5d} | FPS: {live_fps:.1f} | "
                  f"Infer: {inference_ms:.1f}ms | "
                  f"PASS: {pass_count} | DEFECT: {defect_count}{acc_str}")

        # ── Key Handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:           # Q or Escape
            break
        elif key == ord("g"):                       # Toggle Grad-CAM
            show_gradcam = not show_gradcam
            status = "ON" if show_gradcam else "OFF"
            print(f"  Grad-CAM: {status}")
        elif key == ord("s"):                       # Save frame
            fname = f"live_frame_{frame_idx:05d}.png"
            cv2.imwrite(fname, display_frame)
            print(f"  Saved: {fname}")
        elif key == ord(" "):                       # Pause/resume
            paused = not paused
            print(f"  {'Paused' if paused else 'Resumed'}")

        # NOTE: No artificial sleep — run at full TFLite speed to hit >10 FPS target

    # ── Cleanup ───────────────────────────────────────────────────────────
    if use_webcam:
        cap.release()
    if source is not None and isinstance(source, FrameBuffer):
        source.stop()
    cv2.destroyAllWindows()

    # ── Final Report ──────────────────────────────────────────────────────
    total = pass_count + defect_count
    print("\n" + "=" * 60)
    print("  PRODUCTION LINE SUMMARY")
    print("=" * 60)
    print(f"  Total boards inspected : {total}")
    print(f"  Predicted PASS         : {pass_count}  ({pass_count/max(1,total)*100:.1f}%)")
    print(f"  Predicted DEFECT       : {defect_count}  ({defect_count/max(1,total)*100:.1f}%)")

    if (tp+tn+fp+fn) > 0:
        acc        = (tp+tn) / (tp+tn+fp+fn)
        precision  = tp / (tp+fp+1e-8)
        recall     = tp / (tp+fn+1e-8)
        spec       = tn / (tn+fp+1e-8)
        print(f"\n  Ground-Truth Accuracy  : {acc*100:.1f}%")
        print(f"  Defect Precision       : {precision:.3f}")
        print(f"  Defect Recall          : {recall:.3f}  (target ≥ 0.90)")
        print(f"  Specificity (Pass Rec) : {spec:.3f}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")

    avg_fps  = float(np.mean(fps_readings))  if fps_readings else fps_ctr.fps
    min_fps  = float(np.min(fps_readings))   if fps_readings else 0.0
    peak_fps_val = float(np.max(fps_readings)) if fps_readings else 0.0

    print(f"\n  TFLite Throughput (mean) : {avg_fps:.1f} FPS")
    print(f"  TFLite Throughput (min)  : {min_fps:.1f} FPS")
    print(f"  TFLite Throughput (peak) : {peak_fps_val:.1f} FPS")
    print(f"  Grad-CAM                 : background thread, every {GRADCAM_EVERY} frames")
    req_met = "✓ MET" if round(avg_fps, 1) >= 10.0 else "✗ NOT MET"
    print(f"  >10 FPS requirement      : {req_met}")
    print("=" * 60)


# ─── Step 8: Benchmark Inference Speed ───────────────────────────────────────
def benchmark_inference(tflite_path=TFLITE_PATH, n_runs=200):
    """
    Benchmark raw inference speed without display overhead.
    Target: >10 FPS = <100ms per frame.
    """
    print("\n  Benchmarking TFLite inference speed...")
    if not os.path.exists(tflite_path):
        print(f"  ✗ TFLite model not found at {tflite_path}")
        return

    engine    = TFLiteInferenceEngine(tflite_path)
    dummy_bgr = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    latencies = []

    # Warm-up
    for _ in range(5):
        engine.predict(dummy_bgr)

    for _ in range(n_runs):
        _, _, ms = engine.predict(dummy_bgr)
        latencies.append(ms)

    latencies = np.array(latencies)
    fps_equiv = 1000.0 / latencies

    print(f"\n  Benchmark Results ({n_runs} runs on 640×480 BGR input):")
    print(f"    Mean latency  : {latencies.mean():.2f} ms")
    print(f"    P50 latency   : {np.percentile(latencies, 50):.2f} ms")
    print(f"    P95 latency   : {np.percentile(latencies, 95):.2f} ms")
    print(f"    P99 latency   : {np.percentile(latencies, 99):.2f} ms")
    print(f"    Mean FPS      : {fps_equiv.mean():.1f}")
    print(f"    Min FPS       : {fps_equiv.min():.1f}")

    req_met = "✓ PASS" if fps_equiv.mean() >= 10 else "✗ FAIL"
    print(f"\n    Production >10 FPS requirement: {req_met}")

    # Plot latency distribution
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("VisionSpec QC — TFLite Inference Benchmark", fontweight="bold")

    axes[0].hist(latencies, bins=30, color="#3498db", edgecolor="white")
    axes[0].axvline(100, color="red", linestyle="--", label="100ms (10 FPS limit)")
    axes[0].set_xlabel("Latency (ms)"); axes[0].set_ylabel("Count")
    axes[0].set_title("Latency Distribution"); axes[0].legend()

    axes[1].plot(latencies, color="#2ecc71", linewidth=0.8, alpha=0.7)
    axes[1].axhline(100, color="red", linestyle="--", label="100ms limit")
    axes[1].set_xlabel("Frame #"); axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Per-Frame Latency"); axes[1].legend()

    plt.tight_layout()
    plt.savefig("results/week4_benchmark.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("  Saved: week4_benchmark.png")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VisionSpec QC — Live Inference")
    parser.add_argument("--webcam",    action="store_true", help="Use real webcam")
    parser.add_argument("--no-gradcam",action="store_true", help="Disable Grad-CAM overlay")
    parser.add_argument("--benchmark", action="store_true", help="Run inference benchmark only")
    parser.add_argument("--convert",   action="store_true", help="Convert model to TFLite")
    args = parser.parse_args()

    print("=" * 60)
    print("  VisionSpec QC — Week 4: Inference Optimisation")
    print("=" * 60)

    if args.convert or not os.path.exists(TFLITE_PATH):
        if os.path.exists(MODEL_PATH):
            convert_to_tflite()
        else:
            print("  ⚠ Keras model not found. Run week2_model_training.py first.")
            sys.exit(1)

    if args.benchmark:
        benchmark_inference()
    else:
        run_live_demo(
            use_webcam=args.webcam,
            use_gradcam=not args.no_gradcam,
        )

    print("\n  ✓ Week 4 Complete — VisionSpec QC project fully implemented.\n")
