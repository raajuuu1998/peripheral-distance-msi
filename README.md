# Biological Spatial Priors for Cross-Site MSI Generalization

<p align="center">
  <a href="https://arxiv.org/abs/2605.02660"><img src="https://img.shields.io/badge/arXiv-2605.02660-b31b1b.svg" alt="arXiv"/></a>
  <a href="https://drive.google.com/drive/folders/1Yo6VLX7CuSvStGcXWWIPGweCXAMdDM_e?usp=sharing"><img src="https://img.shields.io/badge/Google%20Drive-Data%20%26%20Models-4285F4?logo=googledrive" alt="Google Drive"/></a>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License"/>
</p>

---

## Overview

This repository contains the full implementation for:

> **Biological Spatial Priors Regularize Foundation Model Representations for Cross-Site MSI Generalization in Colorectal Cancer**
> Dasari Naga Raju — arXiv 2026

The core contribution is **peripheral distance encoding** — a biologically grounded spatial prior that encodes each tile's proximity to the slide boundary as a scalar injected into TransMIL before self-attention.

MSI-H colorectal tumors show a well-known spatial pattern: dense lymphocytic infiltration at the tumor invasive margin, known as the Crohn's-like reaction. This biology is conserved across institutions and scanners. Encoding it geometrically guides foundation model representations toward site-invariant features without any architectural changes or target-domain retraining.

---

## Pipeline

![Pipeline](X.png)

*Tile features extracted by UNI2-h or Virchow2 are augmented with the peripheral distance scalar and aggregated using TransMIL for simultaneous MSI, MSS, and hypermutation prediction.*

---

## Results

### Table 1 — Baseline Configurations (TCGA-COAD → TCGA-READ)

| Encoder | Aggregator | COAD MSI AUC | Hyper AUC | READ MSS Spec |
|---------|-----------|:------------:|:---------:|:-------------:|
| UNI2-h | ABMIL | 0.948 ± 0.028 | 0.903 ± 0.049 | 0.796 |
| UNI2-h | CLAM-SB | 0.935 ± 0.042 | 0.897 ± 0.054 | 0.837 |
| UNI2-h | TransMIL | 0.957 ± 0.013 | 0.902 ± 0.075 | 0.939 |
| Virchow2 | ABMIL | 0.934 ± 0.044 | 0.881 ± 0.045 | 0.408 |
| Virchow2 | CLAM-SB | 0.929 ± 0.053 | 0.868 ± 0.065 | 0.939 |
| Virchow2 | TransMIL | 0.915 ± 0.037 | 0.865 ± 0.079 | 0.878 |
| Kather et al. [4] | — | 0.840 | — | — |

### Table 2 — With Biological Spatial Priors

| Encoder | Configuration | COAD MSI AUC | Hyper AUC | READ MSS Spec |
|---------|--------------|:------------:|:---------:|:-------------:|
| UNI2-h | TransMIL | 0.957 ± 0.013 | 0.902 ± 0.075 | 0.939 |
| UNI2-h | **TransMIL + PD** | **0.959 ± 0.012** | 0.808 ± 0.121 | **1.000** |
| UNI2-h | TransMIL + LIN | 0.953 ± 0.022 | 0.881 ± 0.068 | 0.939 |
| Virchow2 | TransMIL | 0.915 ± 0.037 | 0.865 ± 0.079 | 0.878 |
| Virchow2 | TransMIL + PD | 0.941 ± 0.036 | 0.863 ± 0.079 | 0.959 |
| Virchow2 | TransMIL + LIN | 0.905 ± 0.047 | 0.876 ± 0.045 | 0.959 |

> **PD = Peripheral Distance encoding | LIN = Local Immune Neighborhood encoding**
> Training: TCGA-COAD (137 slides, 5-fold CV) | External validation: TCGA-READ (50 slides, no target-domain retraining)

---

## Attention Maps

The attention maps show how peripheral distance encoding shifts model focus toward the tumor invasive margin in MSI-H slides, while suppressing diffuse central attention in MSS slides.

### MSI-H Slide (TCGA-A6-2672)

![Attention MSI-H](X2.png)

*Left: baseline TransMIL — attention distributed broadly across tissue interior. Right: TransMIL + PD — attention concentrated toward the slide boundary, consistent with the Crohn's-like lymphocytic reaction at the invasive margin.*

### MSS Slide (TCGA-A6-2677)

![Attention MSS](X3.png)

*Both configurations produce diffuse attention with no peripheral concentration, consistent with the absence of peritumoral immune infiltrate in microsatellite stable tumors.*

---

## Repository Structure

```
peripheral-distance-msi/
│
├── priors/
│   ├── peripheral_distance.py        # PD encoding — core contribution
│   └── local_immune_neighborhood.py  # LIN encoding
│
├── models/
│   ├── abmil.py                      # ABMIL (Table 1, Rows 1 & 4)
│   ├── clam.py                       # CLAM-SB (Table 1, Rows 2 & 5)
│   └── transmil.py                   # TransMIL baseline (Table 1, Rows 3 & 6)
│
├── baselines/
│   └── train_baselines.py            # All 6 baseline configurations
│
├── figures/
│   ├── pipeline.png                  # Architecture diagram
│   ├── attn_msih.png                 # Attention map — MSI-H slide
│   └── attn_mss.png                  # Attention map — MSS slide
│
├── train.py                          # TransMIL + PD training (main result)
├── evaluate_read.py                  # Cross-site evaluation on TCGA-READ
├── attention_maps.py                 # Attention map visualization
├── peripheral_distance.py            # PD encoding (standalone)
├── transmil_pd.py                    # TransMIL + PD model (standalone)
└── requirements.txt
```

---

## Data & Pretrained Models

All embeddings, labels, and trained model checkpoints are available on Google Drive:

**[Google Drive — Data & Models](https://drive.google.com/drive/folders/1Yo6VLX7CuSvStGcXWWIPGweCXAMdDM_e?usp=sharing)**

```
TCGA_COAD/
├── raw_slides_coad/                      # TCGA-COAD .svs WSI files
├── raw_slides_read/                      # TCGA-READ .svs WSI files
│
├── embeddings_uni2h_coad/                # UNI2-h features — TCGA-COAD (137 slides)
├── embeddings_uni2h_read/                # UNI2-h features — TCGA-READ (50 slides)
├── embeddings_virchow2_coad/             # Virchow2 features — TCGA-COAD
├── embeddings_virchow2_read/             # Virchow2 features — TCGA-READ
├── embeddings_uni2h_til_coad/            # UNI2-h + LYM scores — TCGA-COAD
│
├── coad_labels_combined.csv              # TCGA-COAD labels (MSI, hypermutation)
├── read_labels.csv                       # TCGA-READ labels
├── nct_linear_probe.pth                  # NCT-CRC-HE-100K linear probe (for LIN)
│
└── saved_models_137/
    ├── uni2h_abmil_baseline/             # UNI2-h + ABMIL checkpoints
    ├── uni2h_clam_baseline/              # UNI2-h + CLAM-SB checkpoints
    ├── uni2h_transmil_baseline_v2/       # UNI2-h + TransMIL baseline checkpoints
    ├── virchow2_abmil_baseline/          # Virchow2 + ABMIL checkpoints
    ├── virchow2_clam_baseline/           # Virchow2 + CLAM-SB checkpoints
    ├── virchow2_transmil_baseline/       # Virchow2 + TransMIL baseline checkpoints
    ├── uni2h_transmil_pd_v4/             # UNI2-h + TransMIL + PD (main result)
    ├── uni2h_transmil_lin/               # UNI2-h + TransMIL + LIN
    ├── virchow2_transmil_pd/             # Virchow2 + TransMIL + PD
    └── virchow2_transmil_lin/            # Virchow2 + TransMIL + LIN
```

Each `.pt` embedding file contains:
```python
{
    "features":   torch.Tensor,  # (N, feat_dim) — UNI2-h: 1536-dim | Virchow2: 2560-dim
    "coords":     torch.Tensor,  # (N, 2) — tile pixel coordinates (x, y)
    "slide_dims": torch.Tensor,  # (2,) — slide width and height in pixels
}
```

---

## Installation & Usage

```bash
git clone https://github.com/raajuuu1998/peripheral-distance-msi.git
cd peripheral-distance-msi
pip install -r requirements.txt
```

Update the Google Drive path constants at the top of each script, then:

```bash
# Train all 6 baselines (Table 1)
python baselines/train_baselines.py --encoder uni2h --aggregator abmil
python baselines/train_baselines.py --encoder uni2h --aggregator clam
python baselines/train_baselines.py --encoder uni2h --aggregator transmil
python baselines/train_baselines.py --encoder virchow2 --aggregator abmil
python baselines/train_baselines.py --encoder virchow2 --aggregator clam
python baselines/train_baselines.py --encoder virchow2 --aggregator transmil

# Train main result — TransMIL + PD (Table 2)
python train.py

# Cross-site evaluation on TCGA-READ
python evaluate_read.py

# Generate attention map figures
python attention_maps.py
```

---

## Citation

```bibtex
@article{raju2026biological,
  title   = {Biological Spatial Priors Regularize Foundation Model Representations
             for Cross-Site MSI Generalization in Colorectal Cancer},
  author  = {Dasari Naga Raju},
  journal = {arXiv preprint arXiv:2605.02660},
  year    = {2026}
}
```

---

## Acknowledgements

- [UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h) — MahmoodLab, Harvard
- [Virchow2](https://huggingface.co/paige-ai/Virchow2) — Paige AI
- [TransMIL](https://github.com/szc19990412/TransMIL) — Shao et al., NeurIPS 2021
- [CLAM](https://github.com/mahmoodlab/CLAM) — Lu et al., Nature Biomedical Engineering 2021
- TCGA-COAD and TCGA-READ data from [GDC Data Portal](https://portal.gdc.cancer.gov/)
