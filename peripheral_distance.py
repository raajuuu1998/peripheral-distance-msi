"""
Peripheral Distance Encoding
=============================
Encodes each tile's proximity to the slide boundary as a scalar prior.
Tiles at the boundary receive a score of 1.0; tiles at the center receive 0.0.

Biological motivation:
    MSI-H colorectal tumors show dense lymphocytic infiltration at the tumor
    invasive margin (Crohn's-like reaction). This pattern is spatially conserved
    across institutions and scanners. Peripheral distance encodes tile proximity
    to the slide boundary as a lightweight geometric proxy for margin proximity,
    without requiring tissue segmentation or tile-level annotations.

Reference:
    Dasari Naga Raju, "Biological Spatial Priors Regularize Foundation Model
    Representations for Cross-Site MSI Generalization in Colorectal Cancer"
    arXiv:2605.02660, 2026.
"""

import torch


def peripheral_distance(coords_norm: torch.Tensor) -> torch.Tensor:
    """
    Compute peripheral distance score for each tile.

    Args:
        coords_norm: Tensor of shape (N, 2) with normalized tile coordinates
                     in range [0, 1], where each row is (x_norm, y_norm).
                     Normalize as: coords / torch.tensor([slide_W, slide_H])

    Returns:
        Tensor of shape (N, 1) with peripheral distance scores in [0, 1].
        Score = 1.0 at the slide boundary, 0.0 at the center.

    Example:
        >>> coords = data["coords"].float()
        >>> W, H = float(data["slide_dims"][0]), float(data["slide_dims"][1])
        >>> coords_norm = coords / torch.tensor([W, H])
        >>> pd_scores = peripheral_distance(coords_norm)   # (N, 1)
        >>> enriched = torch.cat([features, pd_scores], dim=1)  # (N, feat_dim+1)
    """
    cx = coords_norm[:, 0]  # normalized x coordinate
    cy = coords_norm[:, 1]  # normalized y coordinate

    # Distance from each tile to its nearest boundary
    # min over left/right and top/bottom distances
    dist = torch.min(
        torch.min(cx, 1 - cx),
        torch.min(cy, 1 - cy)
    )

    # Invert: boundary tiles (dist=0) get score 1.0, center tiles get score 0.0
    periph_score = (1.0 - dist * 2).clamp(0, 1)

    return periph_score.unsqueeze(1)  # (N, 1)
