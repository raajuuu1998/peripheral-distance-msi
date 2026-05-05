"""
TransMIL — Transformer-Based Multiple Instance Learning (Baseline)
===================================================================
Inter-tile self-attention MIL aggregator. A learnable CLS token is
prepended to the bag of tile features, and two multi-head self-attention
layers allow each tile to attend to all others before the CLS output
is used for classification.

This is the baseline configuration without any spatial prior.
For the version with Peripheral Distance encoding, see transmil_pd.py.

Reference:
    Shao et al., "TransMIL: Transformer Based Correlated Multiple
    Instance Learning for Whole Slide Image Classification"
    NeurIPS 2021.

    Dasari Naga Raju, "Biological Spatial Priors Regularize Foundation
    Model Representations for Cross-Site MSI Generalization in
    Colorectal Cancer" arXiv:2605.02660, 2026. (Table 1, Rows 3 & 6)
"""

import torch
import torch.nn as nn


class TransLayer(nn.Module):
    """Single transformer layer with pre-norm and feed-forward network."""

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.attn  = nn.MultiheadAttention(hidden_dim, n_heads,
                                            dropout=dropout, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class TransMIL_MultiTask(nn.Module):
    """
    TransMIL baseline with MSI and hypermutation heads.

    Args:
        feat_dim:   Input tile feature dimension.
                    UNI2-h: 1536 | Virchow2: 2560
        hidden_dim: Transformer hidden dimension. Default: 512.
        n_heads:    Number of attention heads. Default: 8.
        n_layers:   Number of transformer layers. Default: 2.
        dropout:    Dropout rate. Default: 0.25.
    """

    def __init__(self,
                 feat_dim:   int   = 1536,
                 hidden_dim: int   = 512,
                 n_heads:    int   = 8,
                 n_layers:   int   = 2,
                 dropout:    float = 0.25):
        super().__init__()

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Learnable CLS token for slide-level aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer layers
        self.layers = nn.ModuleList([
            TransLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Task-specific classification heads
        self.head_msi   = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.head_hyper = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tile features (N, feat_dim)
        Returns:
            logit_msi:    MSI prediction logit (scalar)
            logit_hyper:  Hypermutation prediction logit (scalar)
            attn_weights: Attention weights (N, 1) for visualization
        """
        # Project and prepend CLS token
        x   = self.input_proj(x).unsqueeze(0)        # (1, N, hidden_dim)
        cls = self.cls_token.expand(1, -1, -1)        # (1, 1, hidden_dim)
        x   = torch.cat([cls, x], dim=1)              # (1, N+1, hidden_dim)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        cls_out      = x[:, 0, :]    # CLS output: (1, hidden_dim)
        patch_tokens = x[:, 1:, :]   # Tile representations: (1, N, hidden_dim)

        # Attention weights for visualization
        attn_weights = torch.softmax(
            torch.matmul(cls_out.unsqueeze(1), patch_tokens.transpose(1, 2))
            / (cls_out.shape[-1] ** 0.5),
            dim=-1
        ).squeeze(0).squeeze(0).unsqueeze(-1)  # (N, 1)

        logit_msi   = self.head_msi(cls_out).squeeze()
        logit_hyper = self.head_hyper(cls_out).squeeze()

        return logit_msi, logit_hyper, attn_weights


def subsample(features: torch.Tensor, max_tiles: int) -> torch.Tensor:
    """Randomly subsample tiles if slide exceeds max_tiles."""
    if features.shape[0] > max_tiles:
        idx = torch.randperm(features.shape[0])[:max_tiles]
        return features[idx]
    return features
