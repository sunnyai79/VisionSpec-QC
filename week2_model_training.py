"""
===========================================================
VisionSpec QC — Week 2: Core Modelling (Transfer Learning)
===========================================================
Product: VisionSpec QC | Use Case: PCB Defect Detection
Model Backbone: MobileNetV2 (pretrained on ImageNet)

Architecture Decision — Why MobileNetV2 over ResNet50:
┌──────────────────┬──────────────────┬──────────────────┐
│ Metric           │ MobileNetV2      │ ResNet50         │
├──────────────────┼──────────────────┼──────────────────┤
│ Parameters       │ 3.4M             │ 25.6M            │
│ Inference Speed  │ >30 FPS (CPU)    │ ~8 FPS (CPU)     │
│ Top-1 Accuracy   │ 71.8%            │ 74.9%            │
│ Model Size       │ ~14 MB           │ ~98 MB           │
│ Memory (RAM)     │ Low              │ High             │
│ Production Fit   │ ✓ Excellent      │ ✗ Too heavy      │
└──────────────────┴──────────────────┴──────────────────┘
MobileNetV2 meets the >10 FPS production requirement while
remaining highly accurate for binary defect classification.

Strategy:
  Phase 1 — Feature Extraction: Freeze all MobileNetV2 layers,
            train only the custom classification head.
  Phase 2 — Fine-tuning: Unfreeze top N layers and train end-to-end
            with a very small learning rate.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─── Configuration ────────────────────────────────────────────────────────────
IMG_SIZE       = (224, 224)
BATCH_SIZE     = 32
EPOCHS_PHASE1  = 25      # More epochs — larger dataset needs more passes
EPOCHS_PHASE2  = 20      # Fine-tuning
LR_PHASE1      = 1e-3
LR_PHASE2      = 2e-5
UNFREEZE_LAYERS= 20
DROPOUT_RATE   = 0.35    # Reduced from 0.40 — less aggressive regularisation
DATASET_DIR    = "dataset"
TRAIN_DIR      = os.path.join(DATASET_DIR, "train")
VAL_DIR        = os.path.join(DATASET_DIR, "val")
MODEL_DIR      = "models"
SEED           = 42
os.makedirs(MODEL_DIR, exist_ok=True)


# ─── Step 1: Data Generators ──────────────────────────────────────────────────
def build_generators():
    """Rebuild generators matching week1 augmentation config."""
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        brightness_range=[0.70, 1.30],
        shear_range=4.0,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=True, seed=SEED,
    )
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=False,
    )
    return train_gen, val_gen


# ─── Step 2: Build MobileNetV2 Transfer Learning Model ───────────────────────
def build_model(trainable_base=False):
    """
    Construct the VisionSpec QC model.

    Architecture:
        Input (224,224,3)
            │
        MobileNetV2 Backbone (ImageNet pretrained)
            │  ← frozen during Phase 1
            │  ← top-30 layers unfrozen during Phase 2
        GlobalAveragePooling2D     (1280-d feature vector)
            │
        BatchNormalization
            │
        Dense(256, activation='relu')
            │
        Dropout(0.40)
            │
        Dense(128, activation='relu')
            │
        Dropout(0.30)
            │
        Dense(1, activation='sigmoid')   ← binary: Pass / Defect
    """
    # Load MobileNetV2 backbone — exclude top classification layers
    base = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
        alpha=1.0,            # Full-width model (vs. 0.75 for even faster inference)
    )
    base.trainable = trainable_base

    # Custom classification head
    inputs  = Input(shape=IMG_SIZE + (3,), name="pcb_input")
    x       = base(inputs, training=trainable_base)
    x       = GlobalAveragePooling2D(name="gap")(x)
    x       = BatchNormalization(name="bn_head")(x)
    x       = Dense(256, activation="relu", name="fc1")(x)
    x       = Dropout(DROPOUT_RATE, name="drop1")(x)
    x       = Dense(128, activation="relu", name="fc2")(x)
    x       = Dropout(0.30, name="drop2")(x)
    outputs = Dense(1, activation="sigmoid", name="defect_score")(x)

    model = Model(inputs, outputs, name="VisionSpec_QC_MobileNetV2")
    return model, base


def print_model_summary(model, base):
    model.summary()
    total      = model.count_params()
    trainable  = sum([tf.size(w).numpy() for w in model.trainable_weights])
    frozen     = total - trainable
    print(f"\n  Total params    : {total:,}")
    print(f"  Trainable params: {trainable:,}")
    print(f"  Frozen params   : {frozen:,}\n")


# ─── Step 3: Callbacks ────────────────────────────────────────────────────────
def get_callbacks(phase_name):
    # Phase 2 fine-tuning needs more patience — val_loss fluctuates more
    es_patience   = 8  if phase_name == "phase2" else 6
    rlr_patience  = 4  if phase_name == "phase2" else 3
    rlr_factor    = 0.5 if phase_name == "phase2" else 0.3
    ckpt_ext      = ".keras"   # Native Keras format — avoids legacy .h5 warning

    return [
        EarlyStopping(
            monitor="val_loss",
            patience=es_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=rlr_factor,
            patience=rlr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=os.path.join(MODEL_DIR, f"visionspec_{phase_name}_best{ckpt_ext}"),
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]


# ─── Step 4: Plot Learning Curves ────────────────────────────────────────────
def plot_learning_curves(h1, h2=None, save_path="results/week2_learning_curves.png"):
    """
    Plot accuracy + loss curves for both training phases.
    Healthy curves: val_loss decreasing and converging to train_loss.
    Overfitting signal: val_loss rising while train_loss falls.
    """
    fig = plt.figure(figsize=(16, 6))
    fig.suptitle("VisionSpec QC — MobileNetV2 Learning Curves",
                 fontsize=14, fontweight="bold")

    # Concatenate history if two phases
    if h2:
        acc     = h1.history["accuracy"]     + h2.history["accuracy"]
        val_acc = h1.history["val_accuracy"] + h2.history["val_accuracy"]
        loss    = h1.history["loss"]         + h2.history["loss"]
        val_loss= h1.history["val_loss"]     + h2.history["val_loss"]
        split   = len(h1.history["accuracy"])
    else:
        acc, val_acc = h1.history["accuracy"], h1.history["val_accuracy"]
        loss, val_loss = h1.history["loss"],   h1.history["val_loss"]
        split = None

    epochs = range(1, len(acc) + 1)

    # --- Accuracy ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(epochs, acc,     "b-o", markersize=4, label="Train Acc",      linewidth=2)
    ax1.plot(epochs, val_acc, "r-o", markersize=4, label="Val Acc",        linewidth=2)
    if split:
        ax1.axvline(x=split, color="gray", linestyle="--", alpha=0.6, label="Fine-tune start")
    ax1.set_title("Accuracy", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax1.set_ylim([0, 1.05])

    # --- Loss ---
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(epochs, loss,     "b-o", markersize=4, label="Train Loss",    linewidth=2)
    ax2.plot(epochs, val_loss, "r-o", markersize=4, label="Val Loss",      linewidth=2)
    if split:
        ax2.axvline(x=split, color="gray", linestyle="--", alpha=0.6, label="Fine-tune start")
    ax2.set_title("Loss (BCELoss)", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


# ─── Step 5: Evaluation & Confusion Matrix ───────────────────────────────────
def evaluate_model(model, val_gen, threshold=None, save_path="results/week2_confusion_matrix.png"):
    """
    Full evaluation with correct defect-class metrics and auto-threshold selection.
    Auto-threshold: scan 0.25→0.80, find threshold where recall>=0.90 AND F1 is highest.
    """
    val_gen.reset()
    y_pred_prob = model.predict(val_gen, verbose=1)
    y_true      = val_gen.classes
    class_names = list(val_gen.class_indices.keys())   # ['defect', 'pass']
    di = val_gen.class_indices['defect']   # 0
    pi = val_gen.class_indices['pass']     # 1

    # ── Auto-select optimal threshold ─────────────────────────────────────
    scan = np.arange(0.25, 0.82, 0.01)
    best_thresh, best_f1 = 0.50, 0.0
    records = []
    for t in scan:
        yp = (y_pred_prob > t).astype(int).flatten()
        c  = confusion_matrix(y_true, yp)
        if c.shape == (2, 2):
            dtp = c[di,di]; dfn = c[di,pi]; dfp = c[pi,di]; dtn = c[pi,pi]
            r = dtp/(dtp+dfn+1e-8); p = dtp/(dtp+dfp+1e-8)
            f = 2*p*r/(p+r+1e-8);  sp = dtn/(dtn+dfp+1e-8)
        else:
            r=p=f=sp=0.0
        records.append((t, r, p, f, sp))
        if r >= 0.90 and f > best_f1:
            best_thresh, best_f1 = t, f

    if threshold is not None:
        best_thresh = threshold
        print(f"\n  Threshold (manual): {best_thresh:.2f}")
    else:
        print(f"\n  Auto-selected threshold: {best_thresh:.2f}  "
              f"(recall≥0.90, best F1={best_f1:.3f})")

    y_pred = (y_pred_prob > best_thresh).astype(int).flatten()
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("VisionSpec QC — Evaluation (MobileNetV2)", fontweight="bold")

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=axes[0])
    axes[0].set_title(f"Confusion Matrix (threshold={best_thresh:.2f})")
    axes[0].set_ylabel("True Label"); axes[0].set_xlabel("Predicted Label")

    ts   = [r[0] for r in records]
    recs = [r[1] for r in records]; precs = [r[2] for r in records]
    f1s  = [r[3] for r in records]; sps   = [r[4] for r in records]
    axes[1].plot(ts, recs,  "r-", lw=2, label="Defect Recall")
    axes[1].plot(ts, precs, "b-", lw=2, label="Defect Precision")
    axes[1].plot(ts, f1s,   "g-", lw=2, label="F1 Score")
    axes[1].plot(ts, sps,   "m--",lw=1.5, label="Specificity (Pass Rec)")
    axes[1].axvline(best_thresh, color="orange", lw=2, linestyle="--",
                    label=f"Chosen={best_thresh:.2f}")
    axes[1].axhline(0.90, color="red", linestyle=":", alpha=0.6, label="90% target")
    axes[1].set_title("Threshold Sensitivity Curve")
    axes[1].set_xlabel("Threshold"); axes[1].set_ylabel("Score")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3); axes[1].set_ylim([0,1.05])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")

    defect_tp = cm[di,di]; defect_fn = cm[di,pi]
    defect_fp = cm[pi,di]; defect_tn = cm[pi,pi]
    precision   = defect_tp/(defect_tp+defect_fp+1e-8)
    recall      = defect_tp/(defect_tp+defect_fn+1e-8)
    f1          = 2*precision*recall/(precision+recall+1e-8)
    specificity = defect_tn/(defect_tn+defect_fp+1e-8)

    print(f"\n  Production Metrics (threshold={best_thresh:.2f}):")
    print(f"    Defect TP (caught)     : {defect_tp} / 150")
    print(f"    Defect FN (missed!)    : {defect_fn}  ← defects passed as good board")
    print(f"    Defect FP (false alarm): {defect_fp}  ← good boards wrongly rejected")
    print(f"    Defect TN (pass ok)    : {defect_tn} / 150")
    print(f"    Defect Precision       : {precision:.4f}")
    print(f"    Defect Recall (Sens.)  : {recall:.4f}  ← CRITICAL: target ≥ 0.90")
    print(f"    Specificity (Pass Rec) : {specificity:.4f}")
    print(f"    F1 Score               : {f1:.4f}")
    req = "✓ MET" if recall >= 0.90 else f"✗ NOT MET (gap: {0.90-recall:.3f})"
    print(f"    90% Recall requirement : {req}")

    # Save threshold for week3/week4
    thresh_path = os.path.join(MODEL_DIR, "best_threshold.txt")
    with open(thresh_path, "w") as fh:
        fh.write(str(round(float(best_thresh), 4)))
    print(f"  Best threshold saved → {thresh_path}")


# ─── Step 6: Save Final Model ─────────────────────────────────────────────────
def save_model(model):
    """
    Save in three formats for Week 4 deployment.

    Format       | API                   | Use case
    -------------|------------------------|---------------------------
    .keras       | model.save()           | Recommended native format
    .h5          | model.save()           | Legacy compatibility
    SavedModel/  | model.export()         | TFLite/TFServing conversion
    """
    keras_path = os.path.join(MODEL_DIR, "visionspec_qc_final.keras")
    h5_path    = os.path.join(MODEL_DIR, "visionspec_qc_final.h5")
    sm_path    = os.path.join(MODEL_DIR, "visionspec_qc_savedmodel")

    # Native Keras format (recommended)
    model.save(keras_path)
    keras_size = os.path.getsize(keras_path) / (1024 * 1024)
    print(f"\n  Saved .keras model  : {keras_path}  ({keras_size:.1f} MB)")

    # Legacy .h5 (for TFLite converter compatibility in Week 4)
    model.save(h5_path)
    h5_size = os.path.getsize(h5_path) / (1024 * 1024)
    print(f"  Saved .h5 model     : {h5_path}  ({h5_size:.1f} MB)")

    # SavedModel format — suppress verbose TensorSpec dump with stdout redirect
    import io, contextlib
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model.export(sm_path)
        print(f"  Saved SavedModel    : {sm_path}/")
    except AttributeError:
        import tensorflow as _tf
        _tf.saved_model.save(model, sm_path)
        print(f"  Saved SavedModel    : {sm_path}/  (via tf.saved_model)")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  VisionSpec QC — Week 2: MobileNetV2 Transfer Learning")
    print("=" * 60)

    # Check GPU
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n  GPUs available: {len(gpus)} {'(GPU training enabled)' if gpus else '(CPU mode)'}")

    train_gen, val_gen = build_generators()

    # ── Class weights: mild boost for defects ──────────────────────────────
    # class_indices = {'defect': 0, 'pass': 1}
    # Previous: 1.5 → recall=0.96 but precision=0.64, FP=40/75 (too many false alarms)
    # Fix: reduce to 1.2 — dataset now has more visually distinct images so
    #      the model no longer needs a heavy penalty to find defects.
    #      1.2 keeps recall safely above 0.90 while recovering precision.
    CLASS_WEIGHT = {0: 1.2, 1: 1.0}
    print(f"\n  Class weights: {CLASS_WEIGHT}  "
          f"(mild 1.2× penalty — larger dataset carries the recall load)")

    # ── Phase 1: Feature Extraction (frozen backbone) ──────────────────────
    print("\n  ── Phase 1: Feature Extraction (Frozen MobileNetV2 Backbone) ──")
    model, base = build_model(trainable_base=False)
    model.compile(
        optimizer=Adam(learning_rate=LR_PHASE1),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    print_model_summary(model, base)

    history1 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE1,
        validation_data=val_gen,
        class_weight=CLASS_WEIGHT,
        callbacks=get_callbacks("phase1"),
        verbose=1,
    )

    # ── Phase 2: Fine-tuning (top N layers of backbone unfrozen) ───────────
    print(f"\n  ── Phase 2: Fine-Tuning (Unfreeze top {UNFREEZE_LAYERS} backbone layers) ──")

    # Reload the best Phase 1 checkpoint to guarantee we start from the best weights
    best_p1 = os.path.join(MODEL_DIR, "visionspec_phase1_best.keras")
    if os.path.exists(best_p1):
        from tensorflow.keras.models import load_model as _load
        model = _load(best_p1)
        # Re-acquire the base model reference from the loaded model
        base  = model.get_layer("mobilenetv2_1.00_224")
        print(f"  Loaded best Phase 1 weights from: {best_p1}")
    else:
        print("  Phase 1 checkpoint not found — continuing from current weights.")

    base.trainable = True
    # Freeze all except the last UNFREEZE_LAYERS
    for layer in base.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False

    # Freeze BatchNormalization layers in the backbone — critical for fine-tuning
    # stability on small datasets; prevents running stats from being corrupted
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LR_PHASE2),    # Very small LR!
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"),
                 tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )

    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"  Trainable params after unfreeze: {trainable_count:,}")
    print(f"  BatchNorm layers in backbone : FROZEN (prevents stat corruption)")

    history2 = model.fit(
        train_gen,
        epochs=EPOCHS_PHASE2,
        validation_data=val_gen,
        class_weight=CLASS_WEIGHT,
        callbacks=get_callbacks("phase2"),
        verbose=1,
    )

    # ── Results ────────────────────────────────────────────────────────────
    plot_learning_curves(history1, history2)
    evaluate_model(model, val_gen)   # threshold=None → auto-selected
    save_model(model)

    print("\n  ✓ Week 2 Complete — Model trained and saved.")
    print("    Proceed to week3_grad_cam.py\n")
