"""
External Cross-Site Evaluation — TCGA-READ
==========================================
Evaluates the best COAD-trained TransMIL+PD model on TCGA-READ
without any target-domain retraining. TCGA-READ (rectal cancer)
serves as the external site with a different scanner and staining
protocol than TCGA-COAD (colon cancer).

Primary metric: MSS specificity — clinically important because
false positives route MSS patients toward immunotherapy that will
not benefit them.

Usage:
    python evaluate_read.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix

from peripheral_distance import peripheral_distance
from transmil_pd import TransMIL_PD

# ================================================================
# PATHS — Update to match your Google Drive structure
# ================================================================
READ_EMB_DIR  = "/content/drive/MyDrive/TCGA_COAD/embeddings_uni2h_read"
READ_CSV_PATH = "/content/drive/MyDrive/TCGA_COAD/read_labels.csv"
SAVE_DIR      = "/content/drive/MyDrive/TCGA_COAD/saved_models_137/uni2h_transmil_pd_v4"

# ================================================================
# CONFIG
# ================================================================
INPUT_DIM   = 1537   # UNI2-h 1536 + 1 PD scalar
DROP_SLIDES = {"TCGA-AA-3678", "TCGA-AA-3833"}
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# LOAD BEST FOLD MODEL
# Select the fold checkpoint with highest COAD MSI AUC
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
    print(f"Loaded fold {best_fold} | COAD MSI AUC = {best_auc:.4f}")
    return model


# ================================================================
# METRICS
# ================================================================
def print_metrics(probs, labels, task: str):
    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs >= 0.5).astype(int)

    # AUC only defined when both classes present
    auc = (roc_auc_score(labels, probs)
           if len(np.unique(labels)) > 1 else float("nan"))

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    print(f"\n{task}:")
    if not np.isnan(auc):
        print(f"  AUC         : {auc:.4f}")
    else:
        print(f"  AUC         : N/A (only one class present)")
    print(f"  Specificity : {specificity:.4f}  ← key cross-site metric")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")


# ================================================================
# MAIN
# ================================================================
print(f"Device : {DEVICE}")
print("Loading model ...")
model = load_best_model(SAVE_DIR, TransMIL_PD, INPUT_DIM)

df_read = pd.read_csv(READ_CSV_PATH)
files   = sorted([f for f in os.listdir(READ_EMB_DIR) if f.endswith(".pt")])

probs_msi   = []
probs_hyper = []
true_msi    = []
true_hyper  = []
results     = []

print(f"\nRunning inference on TCGA-READ ...")
for fname in files:
    pid = "-".join(fname.split("-")[:3])
    if pid in DROP_SLIDES:
        continue
    row = df_read[df_read["Patient ID"] == pid]
    if len(row) == 0:
        continue

    lbl_msi   = 1 if row["MSI Status"].values[0]  == "MSI-H" else 0
    lbl_hyper = 1 if row["Hypermutated"].values[0] == "Hyper"  else 0

    # Load embeddings and compute peripheral distance
    data   = torch.load(os.path.join(READ_EMB_DIR, fname),
                        map_location="cpu", weights_only=False)
    feats  = data["features"].float()
    coords = data["coords"].float()
    W      = float(data["slide_dims"][0])
    H      = float(data["slide_dims"][1])

    coords_norm = coords / torch.tensor([W, H])

    # Append peripheral distance and move to device
    enriched = torch.cat(
        [feats, peripheral_distance(coords_norm)], dim=1
    ).to(DEVICE)

    with torch.no_grad():
        logit_msi, logit_hyper, _ = model(enriched)

    p_msi   = torch.sigmoid(logit_msi.view(1)).cpu().item()
    p_hyper = torch.sigmoid(logit_hyper.view(1)).cpu().item()

    probs_msi.append(p_msi)
    probs_hyper.append(p_hyper)
    true_msi.append(lbl_msi)
    true_hyper.append(lbl_hyper)
    results.append({
        "patient_id": pid,
        "true_msi":   lbl_msi,
        "prob_msi":   round(p_msi, 4),
        "prob_hyper": round(p_hyper, 4),
    })

print(f"\n{'='*60}")
print(f"READ EXTERNAL VALIDATION — UNI2-h + TransMIL + PD")
print(f"Slides evaluated : {len(results)}")
print(f"{'='*60}")
print_metrics(probs_msi,   true_msi,   "MSI Task")
print_metrics(probs_hyper, true_hyper, "Hypermutation Task")

# Save per-slide predictions
out_path = os.path.join(SAVE_DIR, "read_predictions.csv")
pd.DataFrame(results).to_csv(out_path, index=False)
print(f"\nPer-slide predictions saved to: {out_path}")
