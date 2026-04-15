"""
VisionSpec QC — Production Web Application
Flask backend: image upload → TFLite inference → Grad-CAM → JSON response
"""

import os, io, time, base64, json, math, threading
from datetime import datetime
from collections import deque

import numpy as np
import cv2
from PIL import Image
from flask import Flask, request, jsonify, render_template, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model

# ── Resolve all paths relative to this file — works from any working directory ─
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR  = os.path.join(BASE_DIR, "templates")
MODEL_DIR     = os.path.join(BASE_DIR, "..", "models")

# If models folder not found one level up, try same folder (flat layout)
if not os.path.isdir(MODEL_DIR):
    MODEL_DIR = os.path.join(BASE_DIR, "models")

KERAS_PATH    = os.path.join(MODEL_DIR, "visionspec_qc_final.keras")
H5_PATH       = os.path.join(MODEL_DIR, "visionspec_qc_final.h5")
TFLITE_PATH   = os.path.join(MODEL_DIR, "visionspec_qc_quantized.tflite")
THRESH_PATH   = os.path.join(MODEL_DIR, "best_threshold.txt")
SUBMODEL_NAME = "mobilenetv2_1.00_224"
GRAD_CAM_LAYER= "out_relu"
IMG_SIZE      = (224, 224)
MAX_HISTORY   = 50          # keep last N inspections in memory

app = Flask(__name__, template_folder=TEMPLATE_DIR)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB upload limit

# ── Global state ──────────────────────────────────────────────────────────────
_tflite_engine  = None
_grad_cam_model = None
_threshold      = 0.50
_history        = deque(maxlen=MAX_HISTORY)
_stats          = {"total": 0, "pass": 0, "defect": 0, "session_start": datetime.now().isoformat()}
_lock           = threading.Lock()


# ── Model loading ─────────────────────────────────────────────────────────────
def _load_threshold():
    if os.path.exists(THRESH_PATH):
        return float(open(THRESH_PATH).read().strip())
    return 0.50


def _load_tflite():
    if not os.path.exists(TFLITE_PATH):
        raise FileNotFoundError(f"TFLite model not found: {TFLITE_PATH}")
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH, num_threads=4)
    interp.allocate_tensors()
    return interp


def _load_grad_cam(keras_path):
    model = load_model(keras_path)
    base  = model.get_layer(SUBMODEL_NAME)

    # Resolve layer with fallback
    try:
        base.get_layer(GRAD_CAM_LAYER)
        resolved = GRAD_CAM_LAYER
    except ValueError:
        candidates = [l.name for l in base.layers if "relu" in l.name or "out" in l.name]
        resolved   = candidates[-1] if candidates else base.layers[-2].name

    conv_extractor = tf.keras.Model(inputs=base.input, outputs=base.get_layer(resolved).output)
    fresh = tf.keras.Input(shape=IMG_SIZE + (3,))
    conv_out = conv_extractor(fresh)
    x = conv_out
    for layer in model.layers:
        if layer.name == SUBMODEL_NAME or isinstance(layer, tf.keras.layers.InputLayer):
            continue
        x = layer(x)
    return tf.keras.Model(inputs=fresh, outputs=[conv_out, x])


def init_models():
    global _tflite_engine, _grad_cam_model, _threshold
    _threshold = _load_threshold()
    _tflite_engine = _load_tflite()

    # Warmup
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    inp   = _tflite_engine.get_input_details()[0]["index"]
    _tflite_engine.set_tensor(inp, dummy)
    _tflite_engine.invoke()

    keras_path = KERAS_PATH if os.path.exists(KERAS_PATH) else H5_PATH
    if os.path.exists(keras_path):
        _grad_cam_model = _load_grad_cam(keras_path)
    print(f"  Models loaded | threshold={_threshold:.3f}")


# ── Inference helpers ─────────────────────────────────────────────────────────
def _preprocess(pil_img):
    """PIL RGB → (1, 224, 224, 3) float32 tensor."""
    resized = pil_img.resize(IMG_SIZE, Image.BILINEAR)
    arr     = np.array(resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0), np.array(resized)


def _tflite_predict(tensor):
    inp_idx = _tflite_engine.get_input_details()[0]["index"]
    out_idx = _tflite_engine.get_output_details()[0]["index"]
    _tflite_engine.set_tensor(inp_idx, tensor)
    _tflite_engine.invoke()
    score = float(_tflite_engine.get_tensor(out_idx)[0, 0])
    pred  = 1 if score <= _threshold else 0      # 1=DEFECT, 0=PASS
    return pred, score


def _compute_gradcam(tensor, frame_224):
    """Returns Grad-CAM overlay as JPEG base64 string."""
    if _grad_cam_model is None:
        return None

    with tf.GradientTape() as tape:
        conv_out, pred = _grad_cam_model(tensor, training=False)
        tape.watch(conv_out)
        defect_score = 1.0 - pred[:, 0]

    grads   = tape.gradient(defect_score, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_out[0] @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(tf.nn.relu(heatmap)).numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap_r = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))
    hmap_col  = cv2.applyColorMap(np.uint8(255 * heatmap_r), cv2.COLORMAP_JET)
    frame_bgr = cv2.cvtColor(frame_224, cv2.COLOR_RGB2BGR)
    overlay   = cv2.addWeighted(frame_bgr, 0.55, hmap_col, 0.45, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    _, buf = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf).decode()


def _img_to_base64(pil_img, fmt="JPEG"):
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt, quality=88)
    return base64.b64encode(buf.getvalue()).decode()


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", thresh=_threshold)


@app.route("/api/inspect", methods=["POST"])
def inspect():
    """Main inference endpoint. Accepts image file, returns JSON result."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        pil_img = Image.open(file.stream).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image file"}), 400

    t0 = time.perf_counter()
    tensor, frame_224 = _preprocess(pil_img)

    pred_class, score = _tflite_predict(tensor)
    infer_ms = (time.perf_counter() - t0) * 1000

    # Grad-CAM (may take ~150ms; run synchronously for web use)
    gradcam_b64 = _compute_gradcam(tensor, frame_224)
    total_ms = (time.perf_counter() - t0) * 1000

    label       = "DEFECT" if pred_class == 1 else "PASS"
    defect_prob = 1.0 - score          # human-readable: probability of being defective
    confidence  = defect_prob if pred_class == 1 else score

    # Thumbnail of original
    thumb = pil_img.copy()
    thumb.thumbnail((224, 224))
    thumb_b64 = _img_to_base64(thumb)

    # Update session stats
    ts = datetime.now().isoformat()
    with _lock:
        _stats["total"]  += 1
        _stats["defect" if pred_class == 1 else "pass"] += 1
        record = {
            "id":           _stats["total"],
            "timestamp":    ts,
            "filename":     file.filename,
            "label":        label,
            "score":        round(score, 4),
            "defect_prob":  round(defect_prob, 4),
            "confidence":   round(confidence, 4),
            "infer_ms":     round(infer_ms, 1),
            "total_ms":     round(total_ms, 1),
            "thumb_b64":    thumb_b64,
            "gradcam_b64":  gradcam_b64,
        }
        _history.appendleft(record)

    return jsonify({
        **record,
        "stats": {
            "total":       _stats["total"],
            "pass":        _stats["pass"],
            "defect":      _stats["defect"],
            "pass_rate":   round(_stats["pass"]  / max(1, _stats["total"]) * 100, 1),
            "defect_rate": round(_stats["defect"]/ max(1, _stats["total"]) * 100, 1),
        }
    })


@app.route("/api/history")
def history():
    """Return last N inspection records (without base64 images for speed)."""
    with _lock:
        slim = [{k: v for k, v in r.items() if k not in ("gradcam_b64", "thumb_b64")}
                for r in _history]
    return jsonify({"records": slim, "stats": dict(_stats)})


@app.route("/api/stats")
def stats():
    with _lock:
        return jsonify(dict(_stats))


@app.route("/api/reset", methods=["POST"])
def reset():
    with _lock:
        _stats.update({"total": 0, "pass": 0, "defect": 0,
                        "session_start": datetime.now().isoformat()})
        _history.clear()
    return jsonify({"ok": True})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  VisionSpec QC — Production Web Server")
    print("=" * 55)
    print(f"  App directory  : {BASE_DIR}")
    print(f"  Templates dir  : {TEMPLATE_DIR}")
    print(f"  Models dir     : {MODEL_DIR}")
    print(f"  TFLite model   : {'✓ found' if os.path.exists(TFLITE_PATH) else '✗ MISSING — run week4 first'}")
    print(f"  Keras model    : {'✓ found' if os.path.exists(KERAS_PATH) or os.path.exists(H5_PATH) else '✗ MISSING — run week2 first'}")
    print(f"  Threshold file : {'✓ found' if os.path.exists(THRESH_PATH) else '✗ MISSING — using default 0.50'}")
    init_models()
    print("  Open http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
