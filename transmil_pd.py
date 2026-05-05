"""
TransMIL + Peripheral Distance (PD) Model
==========================================
TransMIL aggregator with peripheral distance spatial prior injected
before self-attention. The prior is appended to tile features before
the input linear projection, so all transformer layers process spatial
context alongside semantic features.

Architecture:
    - Input: tile features (feat_dim) + peripheral distance scalar (1)
    - Input projection: Linear(feat_dim, hidden_dim)
    - CLS token prepended to tile sequence
    - N transformer layers with multi-head self-attention
    - CLS token used for slide-level classification
    - Three prediction heads: MSI, MSS (complement of MSI), Hypermutation

Reference:
    Dasari Naga Raju, "Biological Spatial Priors Regularize Foundation Model
    Representations for Cross-Site MSI Generalization in Colorectal Cancer"
    arXiv:2605.02660, 2026.
"""

import torch
import torch.nn as nn


class TransLayer(nn.Module):
    """Single transformer layer with pre-norm and feed-forward network."""

    def __init__(self, hidden_dim: int, n_heads: int = 8, dropout: float = 0.25):
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
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        a, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + a
        # Feed-forward with residual
        return x + self.ff(self.norm2(x))


class TransMIL_PD(nn.Module):
    """
    TransMIL with Peripheral Distance spatial prior.

    The peripheral distance scalar is appended to each tile's foundation
    model feature vector before the input projection. This means all
    transformer self-attention layers integrate spatial context with
    semantic features at every attention step.

    Args:
        feat_dim:   Input feature dimension. For UNI2-h: 1537 (1536 + 1 PD scalar).
        hidden_dim: Transformer hidden dimension. Default: 512.
        n_heads:    Number of attention heads. Default: 8.
        n_layers:   Number of transformer layers. Default: 2.
        dropout:    Dropout rate. Default: 0.25.
    """

    def __init__(self,
                 feat_dim:   int   = 1537,
                 hidden_dim: int   = 512,
                 n_heads:    int   = 8,
                 n_layers:   int   = 2,
                 dropout:    float = 0.25):
        super().__init__()

        # Input projection: map enriched tile features to hidden dim
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Learnable CLS token for slide-level aggregation
        self.cls = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer layers
        self.layers = nn.ModuleList([
            TransLayer(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

        # Prediction heads: MSI, Hypermutation
        # MSS is the logical complement of MSI but trained as a separate head
        self.head_msi   = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.head_hyper = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tile feature tensor of shape (N, feat_dim).
               feat_dim should include the peripheral distance scalar appended.

        Returns:
            logit_msi:   MSI prediction logit (scalar).
            logit_hyper: Hypermutation prediction logit (scalar).
            attn_weights: Attention weights of shape (N, 1) for visualization.
        """
        # Project tiles and prepend CLS token
        x   = self.proj(x).unsqueeze(0)              # (1, N, hidden_dim)
        cls = self.cls.expand(1, -1, -1)              # (1, 1, hidden_dim)
        x   = torch.cat([cls, x], dim=1)              # (1, N+1, hidden_dim)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        # CLS token attends over tile representations
        co = x[:, 0, :]     # CLS token output: (1, hidden_dim)
        pt = x[:, 1:, :]    # Tile representations: (1, N, hidden_dim)

        # Compute attention weights for slide aggregation
        attn = torch.softmax(
            torch.matmul(co.unsqueeze(1), pt.transpose(1, 2))
            / (co.shape[-1] ** 0.5),
            dim=-1
        )  # (1, 1, N)

        # Weighted aggregation
        z = torch.matmul(attn, pt).squeeze(1)         # (1, hidden_dim)

        # Predictions
        logit_msi   = self.head_msi(z).squeeze()
        logit_hyper = self.head_hyper(z).squeeze()

        # Return attention weights for visualization (N, 1)
        attn_weights = attn.squeeze(0).squeeze(0).unsqueeze(-1)

        return logit_msi, logit_hyper, attn_weights
