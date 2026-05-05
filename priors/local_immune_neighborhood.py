"""
Local Immune Neighborhood (LIN) Encoding
=========================================
Encodes the local immune context of each tile by computing the
lymphocyte-to-tumor ratio in its spatial neighborhood.

A per-tile tissue class probability vector is obtained from a linear
probe trained on NCT-CRC-HE-100K (9 tissue classes). The LYM/TUM
log-ratio within a normalized spatial radius r=0.10 is appended as
a scalar prior to each tile's feature vector.

Biological motivation:
    At the invasive front of MSI-H tumors, lymphocytes cluster in
    immediate proximity to tumor nests. A tile surrounded by high
    LYM relative to TUM in its neighborhood provides local structural
    evidence of this immune clustering pattern, distinct from what
    any single tile's appearance encodes alone.

Reference:
    Dasari Naga Raju, "Biological Spatial Priors Regularize Foundation
    Model Representations for Cross-Site MSI Generalization in
    Colorectal Cancer" arXiv:2605.02660, 2026. (Table 2, Rows 3 & 6)
"""

import torch
import torch.nn as nn
import numpy as np


def compute_lin_scores(
    features:   torch.Tensor,
    coords:     torch.Tensor,
    slide_dims: torch.Tensor,
    probe:      nn.Module,
    device:     str   = "cpu",
    radius:     float = 0.10,
    lym_idx:    int   = 3,
    tum_idx:    int   = 5,
    epsilon:    float = 1e-6,
) -> torch.Tensor:
    """
    Compute Local Immune Neighborhood scores for all tiles in a slide.

    Args:
        features:   Tile features (N, feat_dim) — UNI2-h 1536-dim.
        coords:     Tile pixel coordinates (N, 2).
        slide_dims: Slide width and height (2,) in pixels.
        probe:      Linear probe trained on NCT-CRC-HE-100K (Linear(1536, 9)).
        device:     Torch device.
        radius:     Normalized spatial radius for neighborhood. Default: 0.10.
        lym_idx:    Index of LYM class in NCT-CRC-HE-100K. Default: 3.
        tum_idx:    Index of TUM class in NCT-CRC-HE-100K. Default: 5.
        epsilon:    Numerical stability constant. Default: 1e-6.

    Returns:
        lin_scores: Tensor of shape (N, 1) — LYM/TUM log-ratio per tile.

    Note:
        NCT-CRC-HE-100K class order (fixed):
        0=ADI, 1=BACK, 2=DEB, 3=LYM, 4=MUC, 5=MUS, 6=NORM, 7=STR, 8=TUM
        LYM_IDX=3, TUM_IDX=8. Verify against your probe before use.
    """
    W = float(slide_dims[0])
    H = float(slide_dims[1])

    # Normalize coordinates to [0, 1]
    coords_norm = coords.float() / torch.tensor([W, H])  # (N, 2)

    # Get per-tile tissue class probabilities from the NCT probe
    features = features.float().to(device)
    with torch.no_grad():
        logits = probe(features)
        probs  = torch.softmax(logits, dim=-1).cpu()  # (N, 9)

    lym_probs = probs[:, lym_idx].numpy()  # LYM probability per tile
    tum_probs = probs[:, tum_idx].numpy()  # TUM probability per tile
    coords_np = coords_norm.numpy()

    lin_scores = np.zeros(len(coords_np), dtype=np.float32)

    # For each tile, compute mean LYM and TUM in its spatial neighborhood
    for i in range(len(coords_np)):
        cx, cy = coords_np[i]

        # Find all tiles within normalized radius r
        dx   = coords_np[:, 0] - cx
        dy   = coords_np[:, 1] - cy
        dist = np.sqrt(dx**2 + dy**2)
        mask = dist <= radius

        # LYM/TUM log-ratio for neighborhood
        mean_lym = lym_probs[mask].mean()
        mean_tum = tum_probs[mask].mean()
        lin_scores[i] = np.log(
            (mean_lym + epsilon) / (mean_tum + epsilon)
        )

    return torch.tensor(lin_scores).unsqueeze(1)  # (N, 1)


def load_nct_probe(probe_path: str, feat_dim: int = 1536,
                   n_classes: int = 9) -> nn.Linear:
    """
    Load the NCT-CRC-HE-100K linear probe from a checkpoint.

    Args:
        probe_path: Path to probe checkpoint (.pth file).
        feat_dim:   Input feature dimension. Default: 1536 (UNI2-h).
        n_classes:  Number of tissue classes. Default: 9 (NCT-CRC-HE-100K).

    Returns:
        probe: Loaded and eval-mode linear probe.
    """
    probe_state = torch.load(probe_path, map_location="cpu")
    probe       = nn.Linear(feat_dim, n_classes)
    probe.load_state_dict(probe_state["model_state"])
    probe.eval()
    return probe
