"""
ABMIL — Attention-Based Multiple Instance Learning
====================================================
Gated attention pooling for slide-level MSI and hypermutation
prediction. Learns a weighted average of tile features where
attention weights reflect each tile's relevance.

Reference:
    Ilse et al., "Attention-Based Deep Multiple Instance Learning"
    ICML 2018.

    Dasari Naga Raju, "Biological Spatial Priors Regularize Foundation
    Model Representations for Cross-Site MSI Generalization in
    Colorectal Cancer" arXiv:2605.02660, 2026. (Table 1, Rows 1 & 4)
"""

import torch
import torch.nn as nn


class GatedAttention(nn.Module):
    """
    Gated attention mechanism.
    Two learned linear projections whose element-wise product
    defines the attention logits — more expressive than single-pathway attention.
    """

    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.V = nn.Linear(feat_dim, hidden_dim)
        self.U = nn.Linear(feat_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tile features (N, feat_dim)
        Returns:
            z:    Aggregated slide representation (1, feat_dim)
            attn: Attention weights (N, 1)
        """
        # Gated attention: tanh branch * sigmoid gate
        attn = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x)))
        attn = torch.softmax(attn, dim=0)           # (N, 1)
        z    = (attn * x).sum(dim=0, keepdim=True)  # (1, feat_dim)
        return z, attn


class ABMIL_MultiTask(nn.Module):
    """
    ABMIL with two classification heads: MSI and hypermutation.

    Args:
        feat_dim:   Input tile feature dimension.
                    UNI2-h: 1536 | Virchow2: 2560
        hidden_dim: Attention hidden dimension. Default: 256.
        dropout:    Dropout rate. Default: 0.25.
    """

    def __init__(self,
                 feat_dim:   int   = 1536,
                 hidden_dim: int   = 256,
                 dropout:    float = 0.25):
        super().__init__()

        # Project tile features to 512-dim shared representation
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated attention pooling
        self.attention = GatedAttention(512, hidden_dim)

        # Task-specific classification heads
        self.head_msi   = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))
        self.head_hyper = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tile features (N, feat_dim)
        Returns:
            logit_msi:   MSI prediction logit (scalar)
            logit_hyper: Hypermutation prediction logit (scalar)
            attn:        Attention weights (N, 1)
        """
        x           = self.projector(x)
        z, attn     = self.attention(x)
        logit_msi   = self.head_msi(z).squeeze()
        logit_hyper = self.head_hyper(z).squeeze()
        return logit_msi, logit_hyper, attn
