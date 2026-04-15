"""
========================================================
VisionSpec QC — Week 3: Interpretability via Grad-CAM
========================================================
Product: VisionSpec QC | Use Case: PCB Defect Detection
Model Backbone: MobileNetV2

Grad-CAM (Gradient-weighted Class Activation Mapping):
  Computes the gradient of the defect class score with respect
  to the final convolutional feature map. Channels with large
  positive gradients contribute most to the prediction.
  These are weighted-averaged to produce a spatial heatmap.

  Formula:
    α_k^c   = (1/Z) ΣΣ (∂y^c / ∂A^k_ij)   ← importance weight per channel
    L^c_cam = ReLU( Σ_k α_k^c · A^k )       ← class activation map

  For MobileNetV2, we hook into the last Conv layer before GAP:
    "out_relu" — the output of the final inverted residual block.

Verification goal:
    Heatmaps must highlight soldering joints, pads, or defect
    regions — NOT background PCB substrate or random noise.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE       = (224, 224)
MODEL_PATH     = os.path.join("models", "visionspec_qc_final.keras")
MODEL_PATH_H5  = os.path.join("models", "visionspec_qc_final.h5")   # fallback
DATASET_DIR    = "dataset"
GRAD_CAM_LAYER = "out_relu"    # Last Conv feature map in MobileNetV2
SUBMODEL_NAME  = "mobilenetv2_1.00_224"   # Nested backbone name
ALPHA          = 0.55
THRESHOLD      = 0.50          # Default; overridden at runtime from best_threshold.txt
CLASS_NAMES    = {0: "DEFECT", 1: "PASS"}  # score>threshold → PASS(1), else DEFECT(0)


def load_threshold():
    """Load the auto-selected threshold saved by week2."""
    thresh_path = os.path.join("models", "best_threshold.txt")
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            t = float(f.read().strip())
        print(f"  Loaded threshold from week2: {t}")
        return t
    print(f"  Threshold file not found — using default {THRESHOLD}")
    return THRESHOLD


# ─── Step 1: Load Model ───────────────────────────────────────────────────────
def load_visionspec_model(model_path=None):
    """Load the trained VisionSpec QC model. Prefers .keras, falls back to .h5."""
    if model_path is None:
        if os.path.exists(MODEL_PATH):
            model_path = MODEL_PATH
        elif os.path.exists(MODEL_PATH_H5):
            model_path = MODEL_PATH_H5
        else:
            raise FileNotFoundError(
                f"Model not found at '{MODEL_PATH}' or '{MODEL_PATH_H5}'.\n"
                "Please run week2_model_training.py first."
            )
    model = load_model(model_path)
    print(f"  Model loaded: {model_path}")
    print(f"  Model input shape: {model.input_shape}")
    return model


# ─── Step 2: Identify Grad-CAM Target Layer ───────────────────────────────────
def find_conv_layers(model):
    """
    List Conv/DepthwiseConv layers inside the nested MobileNetV2 sub-model.
    The top-level model only exposes wrapper layers; real Conv layers live
    inside the sub-model.  We must drill in to find them.
    """
    try:
        base = model.get_layer(SUBMODEL_NAME)
        conv_layers = [
            layer.name for layer in base.layers
            if isinstance(layer, (tf.keras.layers.Conv2D,
                                   tf.keras.layers.DepthwiseConv2D,
                                   tf.keras.layers.Activation))
            and hasattr(layer, "output")
        ]
        # Filter to activation layers that follow Conv (these are the ReLU outputs)
        relu_layers = [n for n in conv_layers if "relu" in n or "out" in n]
        print(f"\n  Conv/Activation layers inside '{SUBMODEL_NAME}' "
              f"({len(conv_layers)} total):")
        for name in relu_layers[-10:]:
            print(f"    {name}")
        if relu_layers:
            recommended = relu_layers[-1]
            print(f"\n  Recommended Grad-CAM layer: '{recommended}'")
            print(f"  Configured layer          : '{GRAD_CAM_LAYER}'")
            if recommended != GRAD_CAM_LAYER:
                print(f"  ⚠ Mismatch — update GRAD_CAM_LAYER = '{recommended}'")
        return conv_layers
    except ValueError:
        print(f"  ⚠ Sub-model '{SUBMODEL_NAME}' not found.")
        print(f"    Top-level layers: {[l.name for l in model.layers]}")
        return []


def resolve_grad_cam_layer(model, preferred=GRAD_CAM_LAYER):
    """
    Resolve the actual layer object from nested MobileNetV2 sub-model.
    Falls back gracefully if the preferred name doesn't exist.
    """
    base = model.get_layer(SUBMODEL_NAME)
    try:
        layer = base.get_layer(preferred)
        print(f"  Grad-CAM target layer: '{preferred}' (inside '{SUBMODEL_NAME}')")
        return preferred
    except ValueError:
        # Auto-detect: find last activation/conv layer in base model
        candidates = [
            l.name for l in base.layers
            if isinstance(l, (tf.keras.layers.Conv2D,
                               tf.keras.layers.DepthwiseConv2D,
                               tf.keras.layers.Activation))
            and ("relu" in l.name or "out" in l.name or "conv" in l.name)
        ]
        if candidates:
            fallback = candidates[-1]
            print(f"  ⚠ Layer '{preferred}' not found — using fallback: '{fallback}'")
            return fallback
        raise ValueError(
            f"No suitable Grad-CAM layer found in '{SUBMODEL_NAME}'.\n"
            f"Available layers: {[l.name for l in base.layers]}"
        )


# ─── Step 3: Core Grad-CAM Implementation ────────────────────────────────────
def build_grad_model(model, layer_name=None):
    """
    Build the Grad-CAM model correctly for a nested MobileNetV2 structure.

    THE BUG FIXED HERE:
      Previous version computed conv_out and pred from INDEPENDENT paths:
        conv_out = conv_extractor(fresh_inp)   ← branch A
        pred     = model(fresh_inp)            ← branch B (no dep. on conv_out)
      → tape.gradient(pred, conv_out) = None (no computational path)

    CORRECT DESIGN — two-stage chain:
      Stage 1: fresh_inp → [conv_extractor] → conv_out   (watched by tape)
      Stage 2: conv_out  → [tail layers]    → pred        (derived from conv_out)
      → tape.gradient(pred, conv_out) flows correctly ✓

    The tail layers (GAP, BN, Dense...) are the top-level model layers that
    come AFTER the MobileNetV2 sub-model. We apply them sequentially to
    conv_out so the computation graph connects conv_out → pred.
    """
    if layer_name is None:
        layer_name = resolve_grad_cam_layer(model)

    base = model.get_layer(SUBMODEL_NAME)

    # Stage 1: image → target conv layer feature map
    conv_extractor = tf.keras.Model(
        inputs  = base.input,
        outputs = base.get_layer(layer_name).output,
        name    = "conv_extractor",
    )

    # Stage 2: identify tail layers (everything after the sub-model)
    # These are the head layers in the top-level model
    SUBMODEL_LAYER_NAMES = {SUBMODEL_NAME}
    tail_layers = [
        layer for layer in model.layers
        if layer.name not in SUBMODEL_LAYER_NAMES
        and not isinstance(layer, tf.keras.layers.InputLayer)
    ]

    # Build the two-stage model with a fresh input
    fresh_inp = tf.keras.Input(shape=IMG_SIZE + (3,), name="gradcam_input")

    # Stage 1: image → conv features
    conv_out = conv_extractor(fresh_inp)   # (None, 7, 7, 1280)

    # Stage 2: conv features → prediction (chain through head layers)
    # NOTE: conv_out feeds into tail layers so pred IS a function of conv_out
    x = conv_out
    for layer in tail_layers:
        x = layer(x)

    grad_model = tf.keras.Model(
        inputs  = fresh_inp,
        outputs = [conv_out, x],    # pred = f(conv_out) ← gradient flows ✓
        name    = "grad_cam_model",
    )
    print(f"  Grad-CAM model built: input→conv_out{conv_out.shape}→pred{x.shape}")
    return grad_model, layer_name


def compute_gradcam(model, img_array, layer_name=None, threshold=None,
                    _cache={}):
    """
    Compute Grad-CAM heatmap for a single image.

    KEY FIX — why grads were None before:
      The old grad model computed conv_out and pred from INDEPENDENT paths
      off the same input. tape.gradient(pred, conv_out) = None because
      pred did not flow through conv_out in the graph.

      Now pred IS computed FROM conv_out (two-stage chain), so gradients
      flow correctly: d(pred)/d(conv_out) > 0 wherever defect activations
      contribute positively.

    tape.watch(conv_out) added as an extra safety measure — GradientTape
    only auto-watches tf.Variable; explicit watch ensures non-variable
    intermediate tensors are tracked.
    """
    if threshold is None:
        threshold = load_threshold()

    cache_key = id(model)
    if cache_key not in _cache:
        gm, rl = build_grad_model(model, layer_name)
        _cache[cache_key] = (gm, rl)
    grad_model, _ = _cache[cache_key]

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)   # Explicit watch — safety net for non-Variable tensors
        # Defect direction: 1 - P(pass). High where model is uncertain or leaning defect.
        defect_score = 1.0 - predictions[:, 0]

    grads = tape.gradient(defect_score, conv_outputs)   # (1, h, w, C)

    if grads is None:
        raise RuntimeError(
            "Grad-CAM gradient is None — the computational graph between "
            "pred and conv_out is broken. Check build_grad_model()."
        )

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))   # (C,)
    conv_outputs = conv_outputs[0]                          # (h, w, C)
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]   # (h, w, 1)
    heatmap      = tf.squeeze(tf.nn.relu(heatmap)).numpy()        # (h, w)

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    pred_score = float(predictions[0, 0])
    pred_class = int(pred_score > threshold)
    return heatmap, pred_score, pred_class


# ─── Step 4: Overlay Heatmap on Image ────────────────────────────────────────
def overlay_gradcam(original_img_array, heatmap, alpha=ALPHA, colormap=cv2.COLORMAP_JET):
    """
    Resize heatmap to match input image and overlay as a colour map.

    Args:
        original_img_array: float32 array (224, 224, 3) in [0, 1].
        heatmap            : 2D float32 array (h, w) in [0, 1].
        alpha              : Heatmap opacity (0=invisible, 1=opaque).
        colormap           : OpenCV colormap constant.

    Returns:
        superimposed: uint8 RGB image (224, 224, 3).
    """
    # Resize heatmap to input size
    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE[1], IMG_SIZE[0]))

    # Convert to uint8 and apply colour map
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb     = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Overlay on original image
    original_uint8  = np.uint8(255 * original_img_array)
    superimposed    = cv2.addWeighted(original_uint8, 1 - alpha,
                                      heatmap_rgb,    alpha, 0)
    return superimposed


# ─── Step 5: Batch Visualisation & Verification ───────────────────────────────
def visualize_gradcam_grid(model, n_pass=4, n_defect=4,
                           save_path="week3_gradcam_grid.png"):
    """
    Visualise Grad-CAM for a mix of pass/defect images.
    Verification check: heatmaps must highlight component/pad regions,
    not random background substrate.

    Layout per image: Original | Heatmap | Overlay
    """
    pass_dir   = os.path.join(DATASET_DIR, "val", "pass")
    defect_dir = os.path.join(DATASET_DIR, "val", "defect")

    pass_files   = sorted(os.listdir(pass_dir))[:n_pass]
    defect_files = sorted(os.listdir(defect_dir))[:n_defect]

    all_files  = [(f, pass_dir,   "pass")   for f in pass_files]
    all_files += [(f, defect_dir, "defect") for f in defect_files]

    n_images = len(all_files)
    fig, axes = plt.subplots(n_images, 3, figsize=(12, n_images * 3.2))
    fig.suptitle("VisionSpec QC — Grad-CAM Heatmap Verification\n"
                 "Left: Original | Centre: Grad-CAM Heatmap | Right: Overlay\n"
                 "⚑ Heatmap should highlight PCB pads / defect regions, not substrate",
                 fontsize=12, fontweight="bold", y=1.01)

    for row_idx, (fname, fdir, true_label) in enumerate(all_files):
        img_path  = os.path.join(fdir, fname)
        raw       = load_img(img_path, target_size=IMG_SIZE)
        img_arr   = img_to_array(raw) / 255.0
        input_arr = np.expand_dims(img_arr, axis=0)

        heatmap, pred_score, pred_class = compute_gradcam(model, input_arr)
        overlay = overlay_gradcam(img_arr, heatmap)

        pred_label = CLASS_NAMES[pred_class]
        correct    = (pred_label.lower() == true_label)
        result_str = "✓ Correct" if correct else "✗ Wrong"
        title_color= "#2ecc71" if correct else "#e74c3c"

        # Original
        axes[row_idx, 0].imshow(img_arr)
        axes[row_idx, 0].set_title(
            f"True: {true_label.upper()}\n{result_str} | Score: {pred_score:.3f}",
            fontsize=9, color=title_color, fontweight="bold"
        )
        axes[row_idx, 0].axis("off")

        # Heatmap only
        axes[row_idx, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[row_idx, 1].set_title("Grad-CAM Heatmap\n(Red=High Importance)",
                                    fontsize=9)
        axes[row_idx, 1].axis("off")

        # Overlay
        axes[row_idx, 2].imshow(overlay)
        axes[row_idx, 2].set_title("Overlay\n(Verify: hot regions = defect zone)",
                                    fontsize=9)
        axes[row_idx, 2].axis("off")

        # Verification flag
        hot_ratio = float(np.mean(heatmap > 0.5))
        if hot_ratio < 0.02:
            print(f"  ⚠ Row {row_idx}: Heatmap seems sparse — check layer name.")
        elif hot_ratio > 0.40:
            print(f"  ⚠ Row {row_idx}: Heatmap very diffuse — may be highlighting background.")
        else:
            print(f"  ✓ Row {row_idx}: Heatmap focused ({hot_ratio:.1%} hot pixels) [{fname}]")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Saved: {save_path}")


# ─── Step 6: Deep-Dive — Single Image with Multiple Colormaps ─────────────────
def visualize_single_gradcam(model, img_path, save_path="week3_gradcam_detail.png"):
    """
    Full deep-dive for a single defect image:
    Show 4 colormaps to pick the clearest for the production dashboard.
    """
    raw      = load_img(img_path, target_size=IMG_SIZE)
    img_arr  = img_to_array(raw) / 255.0
    inp      = np.expand_dims(img_arr, axis=0)

    heatmap, score, pred_class = compute_gradcam(model, inp)

    colormaps = [
        ("JET",      cv2.COLORMAP_JET),
        ("HOT",      cv2.COLORMAP_HOT),
        ("INFERNO",  cv2.COLORMAP_INFERNO),
        ("RAINBOW",  cv2.COLORMAP_RAINBOW),
    ]

    fig, axes = plt.subplots(1, len(colormaps) + 1, figsize=(18, 4))
    fig.suptitle(
        f"VisionSpec QC — Grad-CAM Deep-Dive\n"
        f"Prediction: {CLASS_NAMES[pred_class]} (score={score:.4f}) | "
        f"File: {os.path.basename(img_path)}",
        fontsize=12, fontweight="bold"
    )

    axes[0].imshow(img_arr)
    axes[0].set_title("Original Image", fontweight="bold")
    axes[0].axis("off")

    for i, (cmap_name, cmap_id) in enumerate(colormaps):
        overlay = overlay_gradcam(img_arr, heatmap, colormap=cmap_id)
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(f"Colormap: {cmap_name}", fontsize=10)
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")
    return heatmap, score, pred_class


# ─── Step 7: Export Grad-CAM Function for Week 4 ─────────────────────────────
def gradcam_on_frame(model, bgr_frame, _cache={}):
    """
    Production-ready Grad-CAM for a single BGR frame (from OpenCV).
    Grad model is built once and cached for speed.
    Returns: overlay_bgr (for cv2.imshow), pred_score, pred_class.
    """
    key = id(model)
    if key not in _cache:
        _cache[key], _ = build_grad_model(model)

    grad_model = _cache[key]

    rgb       = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    resized   = cv2.resize(rgb, IMG_SIZE)
    img_arr   = resized.astype(np.float32) / 255.0
    inp       = np.expand_dims(img_arr, axis=0)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(inp, training=False)
        defect_score = 1.0 - predictions[:, 0]

    grads        = tape.gradient(defect_score, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap      = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(tf.nn.relu(heatmap)).numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    overlay_rgb = overlay_gradcam(img_arr, heatmap, alpha=0.50)
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    score       = float(predictions[0, 0])
    pred_class  = int(score > THRESHOLD)
    return overlay_bgr, score, pred_class


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  VisionSpec QC — Week 3: Grad-CAM Interpretability")
    print("=" * 60)

    model = load_visionspec_model()

    # Inspect and auto-resolve the correct layer
    find_conv_layers(model)
    resolved_layer = resolve_grad_cam_layer(model)

    # Pre-build and cache the grad model
    print(f"\n  Building Grad-CAM model with layer: '{resolved_layer}'...")
    grad_model, _ = build_grad_model(model, resolved_layer)
    print(f"  Grad-CAM model output shapes: "
          f"{[o.shape for o in grad_model.outputs]}")

    print("\n  Generating Grad-CAM grid for pass + defect samples...")
    visualize_gradcam_grid(model, n_pass=4, n_defect=4)

    # Deep-dive on first defect image
    first_defect = os.path.join(DATASET_DIR, "val", "defect",
                                os.listdir(os.path.join(DATASET_DIR, "val", "defect"))[0])
    print(f"\n  Deep-dive Grad-CAM on: {first_defect}")
    visualize_single_gradcam(model, first_defect)

    print("\n  ✓ Week 3 Complete — Grad-CAM heatmaps verified.")
    print("    Ensure heatmaps highlight pads/components, not substrate.")
    print("    Proceed to week4_live_inference.py\n")
