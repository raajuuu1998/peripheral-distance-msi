"""
Attention Map Visualization
============================
Generates attention map figures comparing baseline TransMIL
against TransMIL + Peripheral Distance (PD) for representative
MSI-H and MSS slides from TCGA-COAD.

Produces two figures:
    - Figure 1: COAD MSI-H slide — attention shift toward invasive margin
    - Figure 2: COAD MSS slide — diffuse attention, no peripheral concentration

Usage:
    python attention_maps.py
"""

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from peripheral_distance import peripheral_distance
from transmil_pd import TransMIL_PD

# ================================================================
# PATHS — Update to match your Google Drive structure
# ================================================================
EMB_DIR_COAD  = "/content/drive/MyDrive/TCGA_COAD/embeddings_uni2h_coad"
CSV_COAD      = "/content/drive/MyDrive/TCGA_COAD/coad_labels_combined.csv"
BASELINE_DIR  = "/content/drive/MyDrive/TCGA_COAD/saved_models_137/uni2h_transmil_baseline_v2"
PD_DIR        = "/content/drive/MyDrive/TCGA_COAD/saved_models_137/uni2h_transmil_pd_v4"
SAVE_DIR      = "/content/drive/MyDrive/TCGA_COAD/"

# ================================================================
# CONFIG
# ================================================================
DROP_SLIDES = {"TCGA-AA-3678", "TCGA-AA-3833"}
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# BASELINE MODEL (no spatial prior, feat_dim=1536)
# ================================================================
class TransMIL_Base(TransMIL_PD):
    """Baseline TransMIL without peripheral distance prior."""
    def __init__(self, feat_dim=1536, **kwargs):
        super().__init__(feat_dim=feat_dim, **kwargs)


# ================================================================
# LOAD BEST FOLD MODEL
# ================================================================
def load_best_model(model_dir: str, model_class, input_dim: int):
    best_auc  = 0.0
    best_fold = 1
    for fold in range(1, 6):
        path = os.path.join(model_dir, f"fold{fold}.pth")
        if not os.path.exists(path):
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if ckpt["metrics_msi"]["auc"] > best_auc:
            best_auc  = ckpt["metrics_msi"]["auc"]
            best_fold = fold
    ckpt  = torch.load(os.path.join(model_dir, f"fold{best_fold}.pth"),
                       map_location=DEVICE, weights_only=False)
    model = model_class(input_dim).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ================================================================
# COMPUTE ATTENTION WEIGHTS FOR A SLIDE
# ================================================================
def get_attention(model, filepath: str, use_pd: bool = False):
    """
    Run model forward pass and return attention weights + coordinates.

    Args:
        model:    Trained TransMIL model.
        filepath: Path to slide .pt embedding file.
        use_pd:   If True, append peripheral distance before inference.

    Returns:
        attn:   Attention weights as numpy array (N,).
        coords: Tile coordinates as numpy array (N, 2).
        W, H:   Slide dimensions in pixels.
    """
    data   = torch.load(filepath, map_location="cpu", weights_only=False)
    feats  = data["features"].float()
    coords = data["coords"].float()
    W      = float(data["slide_dims"][0])
    H      = float(data["slide_dims"][1])

    coords_norm = coords / torch.tensor([W, H])

    if use_pd:
        # Append peripheral distance for PD model
        inputs = torch.cat(
            [feats, peripheral_distance(coords_norm)], dim=1
        ).to(DEVICE)
    else:
        inputs = feats.to(DEVICE)

    with torch.no_grad():
        _, _, attn_weights = model(inputs)

    return attn_weights.cpu().numpy(), coords.numpy(), W, H


# ================================================================
# PLOT ATTENTION MAP ON SLIDE COORDINATES
# ================================================================
def plot_attention(ax, attn, coords, W, H, title: str):
    """
    Scatter plot of tiles colored by normalized attention weight.

    High-attention tiles (top 10%) are rendered larger for visibility.
    """
    # Normalize attention to [0, 1]
    attn_norm = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    # Normalize coordinates and flip y-axis for display
    cx = coords[:, 0] / W
    cy = 1 - coords[:, 1] / H

    # All tiles — small markers
    ax.scatter(cx, cy, c=attn_norm, cmap="hot",
               s=2, alpha=0.8, vmin=0, vmax=1)

    # Top 10% attention tiles — larger markers for emphasis
    top_mask = attn_norm >= np.percentile(attn_norm, 90)
    ax.scatter(cx[top_mask], cy[top_mask],
               c=attn_norm[top_mask], cmap="hot",
               s=8, alpha=1.0, vmin=0, vmax=1)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def make_colorbar(fig):
    """Add a shared colorbar to the figure."""
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = ScalarMappable(cmap="hot", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Attention weight (normalized)", fontsize=10)


# ================================================================
# MAIN
# ================================================================
print(f"Device : {DEVICE}")
print("Loading models ...")
baseline_model = load_best_model(BASELINE_DIR, TransMIL_Base, 1536)
pd_model       = load_best_model(PD_DIR,       TransMIL_PD,   1537)
print("Models loaded.")

# Load COAD slide metadata
df_csv  = pd.read_csv(CSV_COAD)
records = []
for f in sorted(os.listdir(EMB_DIR_COAD)):
    if not f.endswith(".pt"):
        continue
    pid = "-".join(f.split("-")[:3])
    if pid in DROP_SLIDES:
        continue
    row = df_csv[df_csv["Patient ID"] == pid]
    if len(row) == 0:
        continue
    data = torch.load(os.path.join(EMB_DIR_COAD, f),
                      map_location="cpu", weights_only=False)
    records.append({
        "filename":  f,
        "patient_id": pid,
        "label_msi": 1 if row["MSI Status"].values[0] == "MSI-H" else 0,
        "n_tiles":   data["features"].shape[0],
    })
df_coad = pd.DataFrame(records)

# Select representative slides with most tiles for clearest visualization
msi_slide = (df_coad[df_coad["label_msi"] == 1]
             .sort_values("n_tiles", ascending=False).iloc[0])
mss_slide = (df_coad[df_coad["label_msi"] == 0]
             .sort_values("n_tiles", ascending=False).iloc[0])

print(f"\nMSI-H slide : {msi_slide['patient_id']}  ({msi_slide['n_tiles']} tiles)")
print(f"MSS slide   : {mss_slide['patient_id']}  ({mss_slide['n_tiles']} tiles)")

# Compute attention maps
print("\nComputing attention maps ...")
path_msi = os.path.join(EMB_DIR_COAD, msi_slide["filename"])
path_mss = os.path.join(EMB_DIR_COAD, mss_slide["filename"])

attn_msi_base, coords_msi, W_msi, H_msi = get_attention(baseline_model, path_msi, False)
attn_msi_pd,   _,          _,     _      = get_attention(pd_model,       path_msi, True)
attn_mss_base, coords_mss, W_mss, H_mss = get_attention(baseline_model, path_mss, False)
attn_mss_pd,   _,          _,     _      = get_attention(pd_model,       path_mss, True)

# ── FIGURE 1: MSI-H ────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 2, figsize=(12, 6))

plot_attention(axes1[0], attn_msi_base, coords_msi, W_msi, H_msi,
               f"MSI-H — TransMIL (No Spatial Prior)\n{msi_slide['patient_id']}")
plot_attention(axes1[1], attn_msi_pd,   coords_msi, W_msi, H_msi,
               f"MSI-H — TransMIL + Peripheral Distance (PD)\n{msi_slide['patient_id']}")

axes1[0].set_ylabel("COAD MSI-H", fontsize=12, fontweight="bold")
make_colorbar(fig1)
fig1.suptitle("Attention Maps (MSI-H): Effect of Peripheral Distance Prior",
              fontsize=12)
plt.tight_layout(rect=[0, 0, 0.91, 0.95])

out1 = os.path.join(SAVE_DIR, "attn_fig1_coad_msih.pdf")
fig1.savefig(out1, dpi=150)
print(f"\nFigure 1 saved: {out1}")

# ── FIGURE 2: MSS ───────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))

plot_attention(axes2[0], attn_mss_base, coords_mss, W_mss, H_mss,
               f"MSS — TransMIL (No Spatial Prior)\n{mss_slide['patient_id']}")
plot_attention(axes2[1], attn_mss_pd,   coords_mss, W_mss, H_mss,
               f"MSS — TransMIL + Peripheral Distance (PD)\n{mss_slide['patient_id']}")

axes2[0].set_ylabel("COAD MSS", fontsize=12, fontweight="bold")
make_colorbar(fig2)
fig2.suptitle("Attention Maps (MSS): Effect of Peripheral Distance Prior",
              fontsize=12)
plt.tight_layout(rect=[0, 0, 0.91, 0.95])

out2 = os.path.join(SAVE_DIR, "attn_fig2_coad_mss.pdf")
fig2.savefig(out2, dpi=150)
print(f"Figure 2 saved: {out2}")

plt.show()
print("\nDone.")
