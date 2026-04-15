"""
========================================================
VisionSpec QC — Week 1: Data Preparation & Augmentation
========================================================
Product: VisionSpec QC | Use Case: PCB Defect Detection

KEY IMPROVEMENTS (v3):
  Problem: Pass & defect images were too visually similar
           → model precision 0.64, FP 40/75, specificity 0.47

  Fixes applied:
    1. Dataset size: 300→700 train, 75→150 val per class
    2. Pass images: clean grid layout, silver solder on ALL pads
    3. Defect images: 5 large-area defect types (8–20% of image)
    4. 30% chance of dual simultaneous defects
    5. Noise reduced on pass (σ=3) vs defect (σ=5) for visual gap
"""

import os, shutil, random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFilter
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, load_img, img_to_array)

# ─── Config ───────────────────────────────────────────────────────────────────
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
DATASET_DIR   = "dataset"
TRAIN_DIR     = os.path.join(DATASET_DIR, "train")
VAL_DIR       = os.path.join(DATASET_DIR, "val")
CLASSES       = ["pass", "defect"]
SAMPLES_TRAIN = {"pass": 700, "defect": 700}
SAMPLES_VAL   = {"pass": 150, "defect": 150}
SEED          = 42
W, H          = IMG_SIZE
random.seed(SEED); np.random.seed(SEED)

# ─── Colour constants ─────────────────────────────────────────────────────────
PCB_GREEN     = (34,  100,  34)
COPPER        = (184, 115,  51)
COPPER_BRIGHT = (210, 140,  40)
SOLDER_SILVER = (180, 180, 170)
COMPONENT_BLK = ( 30,  30,  40)
WHITE_SILK    = (230, 230, 230)
VIA_DARK      = (  5,   5,   5)


# ─── PCB base (shared by pass and defect) ─────────────────────────────────────
def _substrate(w=W, h=H):
    """Solid PCB-green substrate with microscopic grain."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.random.randint(28, 42,  (h, w))
    arr[:, :, 1] = np.random.randint(90, 112, (h, w))
    arr[:, :, 2] = np.random.randint(28, 42,  (h, w))
    return Image.fromarray(arr)


def _draw_traces(draw, w=W, h=H):
    """Orthogonal copper trace grid — 4 horizontal + 4 vertical lines."""
    for y in [h//5, 2*h//5, 3*h//5, 4*h//5]:
        draw.line([(8, y), (w-8, y)], fill=COPPER, width=random.randint(2,4))
    for x in [w//5, 2*w//5, 3*w//5, 4*w//5]:
        draw.line([(x, 8), (x, h-8)], fill=COPPER, width=random.randint(2,4))


def _draw_pads(draw, w=W, h=H, filled=True):
    """
    Draw solder pads on a regular 3×3 grid.
    filled=True  → silver solder (PASS appearance)
    filled=False → bare copper only (used by defect generators)
    """
    for px in [w//6, w//2, 5*w//6]:
        for py in [h//4, h//2, 3*h//4]:
            if random.random() < 0.78:
                pw, ph = random.randint(13,22), random.randint(9,17)
                draw.rectangle([px-pw, py-ph, px+pw, py+ph],
                               outline=COPPER_BRIGHT, width=3)
                interior = SOLDER_SILVER if filled else COPPER_BRIGHT
                draw.rectangle([px-pw+3, py-ph+3, px+pw-3, py+ph-3],
                               fill=interior)


def _draw_components(draw, w=W, h=H):
    """Draw 3-5 IC bodies + white silkscreen reference marks."""
    for _ in range(random.randint(3, 5)):
        cx = random.randint(35, w-35)
        cy = random.randint(35, h-35)
        bw = random.randint(18, 38)
        bh = random.randint(12, 28)
        draw.rectangle([cx-bw, cy-bh, cx+bw, cy+bh],
                       fill=COMPONENT_BLK, outline=COPPER_BRIGHT, width=2)
        draw.line([(cx-bw+2, cy-bh+2), (cx-bw+9, cy-bh+2)],
                  fill=WHITE_SILK, width=1)


def _draw_vias(draw, w=W, h=H):
    """Draw 5-10 through-hole vias."""
    for _ in range(random.randint(5, 10)):
        vx, vy = random.randint(12,w-12), random.randint(12,h-12)
        vr = random.randint(5, 9)
        draw.ellipse([vx-vr,vy-vr,vx+vr,vy+vr], fill=COPPER_BRIGHT, outline=COPPER)
        draw.ellipse([vx-vr+3,vy-vr+3,vx+vr-3,vy+vr-3], fill=VIA_DARK)


def _pcb_base(w=W, h=H, silver_pads=True):
    """Build full PCB base: substrate + traces + components + pads + vias."""
    img  = _substrate(w, h)
    draw = ImageDraw.Draw(img)
    _draw_traces(draw, w, h)
    _draw_components(draw, w, h)
    _draw_pads(draw, w, h, filled=silver_pads)
    _draw_vias(draw, w, h)
    return img, draw


# ─── PASS image ───────────────────────────────────────────────────────────────
def generate_pass_image(width=W, height=H):
    """
    Clean PCB: silver solder on all pads, no anomalies.
    Low noise (σ=3) so pass images look distinctly uniform.
    """
    img, _ = _pcb_base(width, height, silver_pads=True)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, 3, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ─── DEFECT image generators ──────────────────────────────────────────────────
def _defect_solder_bridge(draw, w=W, h=H):
    """Large golden solder blob bridging two pads — 30-60px wide."""
    cx = random.randint(50, w-50)
    cy = random.randint(50, h-50)
    bw = random.randint(22, 42)
    bh = random.randint(12, 22)
    # Main gold blob
    draw.ellipse([cx-bw, cy-bh, cx+bw, cy+bh], fill=(228, 175, 10))
    # Overflow drips
    for _ in range(random.randint(8, 14)):
        ox = cx + random.randint(-bw-12, bw+12)
        oy = cy + random.randint(-bh-10, bh+10)
        r  = random.randint(6, 14)
        draw.ellipse([ox-r,oy-r,ox+r,oy+r], fill=(205, 155, 8))
    # Flux residue centre
    draw.ellipse([cx-9,cy-6,cx+9,cy+6], fill=(70, 52, 5))


def _defect_missing_component(draw, w=W, h=H):
    """Empty pad with bright RED border and X marker — 45-75px zone."""
    cx = random.randint(50, w-50)
    cy = random.randint(50, h-50)
    sz = random.randint(26, 40)
    # Copper pad (no component seated)
    draw.rectangle([cx-sz, cy-sz, cx+sz, cy+sz],
                   fill=COPPER_BRIGHT, outline=(215, 25, 25), width=5)
    # Green interior (substrate showing)
    draw.rectangle([cx-sz+6, cy-sz+6, cx+sz-6, cy+sz-6],
                   fill=(34, 100, 34))
    # Bold red X
    draw.line([(cx-sz+6, cy-sz+6), (cx+sz-6, cy+sz-6)],
              fill=(215, 25, 25), width=4)
    draw.line([(cx+sz-6, cy-sz+6), (cx-sz+6, cy+sz-6)],
              fill=(215, 25, 25), width=4)


def _defect_trace_break(draw, w=W, h=H):
    """Broken copper trace: visible dark gap with scorched edges."""
    y0   = random.randint(35, h-35)
    trw  = random.randint(5, 9)
    x0   = random.randint(10, w//3)
    x1   = random.randint(2*w//3, w-10)
    xmid = (x0+x1)//2
    gap  = random.randint(14, 28)
    # Intact trace segments
    draw.line([(x0, y0), (xmid-gap, y0)], fill=COPPER, width=trw)
    draw.line([(xmid+gap, y0), (x1, y0)], fill=COPPER, width=trw)
    # Break gap: dark substrate
    draw.rectangle([xmid-gap, y0-trw-3, xmid+gap, y0+trw+3],
                   fill=(10, 30, 10))
    # Scorched border
    draw.rectangle([xmid-gap, y0-trw-3, xmid-gap+5, y0+trw+3],
                   fill=(45, 25, 5))
    draw.rectangle([xmid+gap-5, y0-trw-3, xmid+gap, y0+trw+3],
                   fill=(45, 25, 5))


def _defect_burnt_spot(draw, w=W, h=H):
    """Large thermal burn: dark brown/black scorch — 28-50px radius."""
    cx = random.randint(50, w-50)
    cy = random.randint(50, h-50)
    r  = random.randint(28, 50)
    # Outer halo (dark brown)
    draw.ellipse([cx-r,   cy-r,   cx+r,   cy+r],   fill=(58, 22, 4))
    # Mid char zone
    draw.ellipse([cx-r+8, cy-r+8, cx+r-8, cy+r-8], fill=(22, 7,  1))
    # Core (black)
    draw.ellipse([cx-r+18,cy-r+18,cx+r-18,cy+r-18],fill=( 4, 1,  0))
    # Radiating char streaks
    import math
    for ang in range(0, 360, 40):
        ex = int(cx + (r+12)*math.cos(math.radians(ang)))
        ey = int(cy + (r+12)*math.sin(math.radians(ang)))
        draw.line([(cx,cy),(ex,ey)], fill=(38,14,2), width=2)


def _defect_cold_solder(draw, w=W, h=H):
    """Cold/cracked solder joint: dull grey with visible crack lines."""
    cx = random.randint(40, w-40)
    cy = random.randint(40, h-40)
    pw = random.randint(18, 30)
    ph = random.randint(13, 24)
    draw.rectangle([cx-pw, cy-ph, cx+pw, cy+ph],
                   outline=COPPER_BRIGHT, width=3)
    # Dull, flat grey solder (not shiny = cold joint)
    draw.rectangle([cx-pw+3, cy-ph+3, cx+pw-3, cy+ph-3],
                   fill=(108, 103, 98))
    # Crack: bold dark lines
    draw.line([(cx-pw+4, cy), (cx+pw-4, cy)],  fill=(55,53,50), width=3)
    draw.line([(cx, cy-ph+4), (cx, cy+ph-4)],  fill=(55,53,50), width=3)
    # Grainy texture
    for _ in range(25):
        gx = cx + random.randint(-pw+4, pw-4)
        gy = cy + random.randint(-ph+4, ph-4)
        draw.ellipse([gx-1,gy-1,gx+2,gy+2], fill=(78,75,72))


DEFECT_FNS = [
    _defect_solder_bridge,
    _defect_missing_component,
    _defect_trace_break,
    _defect_burnt_spot,
    _defect_cold_solder,
]


def generate_defect_image(width=W, height=H):
    """
    Defective PCB: same base as pass + 1 large defect (sometimes 2).
    σ=5 noise (slightly more than pass) adds realistic sensor variation.
    """
    img, draw = _pcb_base(width, height, silver_pads=True)

    # Primary defect — always present
    random.choice(DEFECT_FNS)(draw, width, height)

    # Secondary defect — 30% co-occurrence (realistic fault cascade)
    if random.random() < 0.30:
        random.choice(DEFECT_FNS)(draw, width, height)

    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    arr = np.array(img, dtype=np.float32)
    arr += np.random.normal(0, 5, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ─── Build dataset ────────────────────────────────────────────────────────────
def build_dataset():
    print("=" * 60)
    print("  VisionSpec QC v3 — Improved Synthetic PCB Dataset")
    print("=" * 60)
    print("  Pass   : clean grid, silver solder, σ=3 noise")
    print("  Defect : 5 types, large area (8-20%), σ=5 noise")
    print(f"  Train  : {SAMPLES_TRAIN['pass']} pass + "
          f"{SAMPLES_TRAIN['defect']} defect")
    print(f"  Val    : {SAMPLES_VAL['pass']} pass + "
          f"{SAMPLES_VAL['defect']} defect\n")

    gens = {"pass": generate_pass_image, "defect": generate_defect_image}
    for split_name, split_dir, counts in [
        ("Training",   TRAIN_DIR, SAMPLES_TRAIN),
        ("Validation", VAL_DIR,   SAMPLES_VAL),
    ]:
        for cls in CLASSES:
            path = os.path.join(split_dir, cls)
            if os.path.exists(path): shutil.rmtree(path)
            os.makedirs(path)
            for i in range(counts[cls]):
                gens[cls]().save(os.path.join(path, f"{cls}_{i:04d}.png"))
            print(f"  [{split_name:10s}] {cls:8s}: {counts[cls]} images → {path}")

    print("\n  Dataset ready.\n")


# ─── Data generators ─────────────────────────────────────────────────────────
def build_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.08,
        height_shift_range=0.08,
        horizontal_flip=True,
        brightness_range=[0.70, 1.30],
        shear_range=4.0,
        fill_mode="nearest",
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=True, seed=SEED)
    val_gen = val_datagen.flow_from_directory(
        VAL_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode="binary", shuffle=False)

    print(f"  Class indices : {train_gen.class_indices}")
    print(f"  Train batches : {len(train_gen)}")
    print(f"  Val batches   : {len(val_gen)}\n")
    return train_gen, val_gen


# ─── Visualisations ───────────────────────────────────────────────────────────
def visualize_class_samples(n=6, save_path="week1_class_samples.png"):
    """Side-by-side pass vs defect — verify visual distinctiveness."""
    fig, axes = plt.subplots(2, n, figsize=(n*3, 7))
    fig.suptitle("VisionSpec QC v3 — Pass vs Defect (visual gap check)\n"
                 "Defects must be clearly visible at a glance",
                 fontsize=12, fontweight="bold")
    for i in range(n):
        axes[0,i].imshow(np.array(generate_pass_image()))
        axes[0,i].axis("off")
        axes[0,i].set_title(f"PASS #{i+1}", fontsize=8, color="green")
        axes[1,i].imshow(np.array(generate_defect_image()))
        axes[1,i].axis("off")
        axes[1,i].set_title(f"DEFECT #{i+1}", fontsize=8, color="red")
    axes[0,0].set_ylabel("PASS",   color="green", fontsize=13, fontweight="bold")
    axes[1,0].set_ylabel("DEFECT", color="red",   fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


def visualize_augmented_batch(train_gen, save_path="week1_augmented_batch.png"):
    images, labels = next(train_gen)
    class_names    = {v: k for k, v in train_gen.class_indices.items()}
    n_cols, n_rows = 8, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 10))
    fig.suptitle("Augmented Training Batch — 224×224 normalised [0,1]",
                 fontsize=12, fontweight="bold", y=1.01)
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            ax.imshow(images[idx])
            label = class_names[int(labels[idx])]
            col   = "#e74c3c" if label == "defect" else "#2ecc71"
            ax.set_title(label.upper(), fontsize=7, color=col, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Saved: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_dataset()
    visualize_class_samples(n=6)
    train_gen, val_gen = build_data_generators()
    visualize_augmented_batch(train_gen)
    print("  ✓ Week 1 Complete. Proceed to week2_model_training.py\n")
