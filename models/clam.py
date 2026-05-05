"""
CLAM-SB — Clustering-constrained Attention MIL (Single Branch)
===============================================================
Extends ABMIL with instance-level clustering constraints.
Top-k and bottom-k attended tiles receive pseudo-labels during
training via instance classifier heads, providing weakly supervised
spatial guidance beyond standard attention pooling.

Loss = 0.7 * bag_loss + 0.3 * instance_loss

Reference:
    Lu et al., "Data-Efficient and Weakly Supervised Computational
    Pathology on Whole-Slide Images" Nature Biomedical Engineering 2021.

    Dasari Naga Raju, "Biological Spatial Priors Regularize Foundation
    Model Representations for Cross-Site MSI Generalization in
    Colorectal Cancer" arXiv:2605.02660, 2026. (Table 1, Rows 2 & 5)
"""

import torch
import torch.nn as nn


class GatedAttention(nn.Module):
    """Gated attention mechanism — same as ABMIL."""

    def __init__(self, feat_dim: int, hidden_dim: int):
        super().__init__()
        self.V = nn.Linear(feat_dim, hidden_dim)
        self.U = nn.Linear(feat_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        attn = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x)))
        attn = torch.softmax(attn, dim=0)           # (N, 1)
        z    = (attn * x).sum(dim=0, keepdim=True)  # (1, feat_dim)
        return z, attn


class CLAM_SB_MultiTask(nn.Module):
    """
    CLAM Single-Branch with MSI and hypermutation heads.

    Args:
        feat_dim:   Input tile feature dimension.
                    UNI2-h: 1536 | Virchow2: 2560
        hidden_dim: Attention hidden dimension. Default: 256.
        dropout:    Dropout rate. Default: 0.25.
        n_inst:     Number of top/bottom instances for instance loss. Default: 8.
    """

    def __init__(self,
                 feat_dim:   int   = 1536,
                 hidden_dim: int   = 256,
                 dropout:    float = 0.25,
                 n_inst:     int   = 8):
        super().__init__()
        self.n_inst = n_inst

        # Shared feature projection
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Shared gated attention
        self.attention = GatedAttention(512, hidden_dim)

        # Instance classifiers — one per task (CLAM key contribution)
        # Top-k tiles get bag label, bottom-k tiles get opposite label
        self.inst_classifier_msi   = nn.Linear(512, 2)
        self.inst_classifier_hyper = nn.Linear(512, 2)

        # Bag-level classification heads
        self.head_msi   = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))
        self.head_hyper = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, 1))

    def forward(self, x: torch.Tensor,
                label_msi=None, label_hyper=None, training: bool = False):
        """
        Args:
            x:           Tile features (N, feat_dim)
            label_msi:   MSI bag label — required during training
            label_hyper: Hypermutation bag label — required during training
            training:    Whether to compute instance loss

        Returns:
            logit_msi:   MSI prediction logit (scalar)
            logit_hyper: Hypermutation prediction logit (scalar)
            attn:        Attention weights (N, 1)
            inst_loss:   Instance clustering loss (0.0 during inference)
        """
        x          = self.projector(x)
        z, attn    = self.attention(x)

        logit_msi   = self.head_msi(z).squeeze()
        logit_hyper = self.head_hyper(z).squeeze()

        inst_loss = torch.tensor(0.0, device=x.device)
        if training and label_msi is not None and label_hyper is not None:
            inst_loss = self._instance_loss(x, attn, label_msi, label_hyper)

        return logit_msi, logit_hyper, attn, inst_loss

    def _instance_loss(self, x, attn, label_msi, label_hyper):
        """
        Compute instance-level clustering loss.
        Top-k tiles predicted as bag class, bottom-k as opposite class.
        """
        attn_scores = attn.squeeze()
        n_inst      = min(self.n_inst, len(attn_scores))

        # Select top-k and bottom-k tiles by attention score
        top_ids    = torch.topk(attn_scores, n_inst).indices
        bottom_ids = torch.topk(attn_scores, n_inst, largest=False).indices

        loss = torch.tensor(0.0, device=x.device)

        for inst_clf, label in [
            (self.inst_classifier_msi,   label_msi),
            (self.inst_classifier_hyper, label_hyper),
        ]:
            lbl = label.long()

            # Top instances → assigned bag label
            top_logits  = inst_clf(x[top_ids])
            top_labels  = torch.full((n_inst,), lbl.item(),
                                     dtype=torch.long, device=x.device)
            loss += nn.CrossEntropyLoss()(top_logits, top_labels)

            # Bottom instances → assigned opposite label
            bottom_logits = inst_clf(x[bottom_ids])
            bottom_labels = torch.full((n_inst,), 1 - lbl.item(),
                                       dtype=torch.long, device=x.device)
            loss += nn.CrossEntropyLoss()(bottom_logits, bottom_labels)

        return loss / 4  # average over 4 instance loss terms
