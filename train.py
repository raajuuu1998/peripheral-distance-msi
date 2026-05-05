"""
Training Script — UNI2-h + TransMIL + Peripheral Distance (PD)
===============================================================
5-fold cross-validation on TCGA-COAD for simultaneous MSI, MSS,
and hypermutation prediction. No tile subsampling — all tiles used
during training to minimize fold variance.

Embeddings and labels are expected to be stored on Google Drive.
Update the path constants below before running.

Usage:
    python train.py

Requirements:
    pip install torch scikit-learn pandas numpy
"""

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score,
                              balanced_accuracy_score, confusion_matrix)

from peripheral_distance import peripheral_distance
from transmil_pd import TransMIL_PD

# ================================================================
# PATHS — Update these to match your Google Drive structure
# ================================================================
EMB_DIR   = "/content/drive/MyDrive/TCGA_COAD/embeddings_uni2h_coad"
CSV_PATH  = "/content/drive/MyDrive/TCGA_COAD/coad_labels_combined.csv"
SAVE_DIR  = "/content/drive/MyDrive/TCGA_COAD/saved_models_137/uni2h_transmil_pd_v4"

# ================================================================
# HYPERPARAMETERS
# ================================================================
FEAT_DIM      = 1536        # UNI2-h feature dimension
PERIPH_DIM    = 1           # peripheral distance scalar
INPUT_DIM     = FEAT_DIM + PERIPH_DIM   # 1537
HIDDEN_DIM    = 512
N_HEADS       = 8
N_LAYERS      = 2
DROPOUT       = 0.25
EPOCHS        = 30
LR_MAX        = 1e-4
LR_MIN        = 1e-5
WARMUP_EPOCHS = 3
WEIGHT_DECAY  = 1e-5
MSI_WEIGHT    = 8.0         # class weight for MSI positive class
HYPER_WEIGHT  = 4.07        # class weight for hypermutation positive class
MAX_PATIENCE  = 15          # early stopping patience
N_FOLDS       = 5
SEED          = 42

# Slides excluded due to corrupted embeddings
DROP_SLIDES = {"TCGA-AA-3678", "TCGA-AA-3833"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Device   : {DEVICE}")
print(f"Encoder  : UNI2-h  feat_dim={FEAT_DIM}")
print(f"Model    : TransMIL + PD (ALL tiles, no subsampling)")
print(f"Epochs   : {EPOCHS}  LR={LR_MIN}→{LR_MAX}")


# ================================================================
# LOAD LABELS
# ================================================================
df_csv  = pd.read_csv(CSV_PATH)
records = []
for f in sorted(os.listdir(EMB_DIR)):
    if not f.endswith(".pt"):
        continue
    pid = "-".join(f.split("-")[:3])
    if pid in DROP_SLIDES:
        continue
    row = df_csv[df_csv["Patient ID"] == pid]
    if len(row) == 0:
        continue
    records.append({
        "filename":    f,
        "patient_id":  pid,
        "label_msi":   1 if row["MSI Status"].values[0]  == "MSI-H" else 0,
        "label_hyper": 1 if row["Hypermutated"].values[0] == "Hyper"  else 0,
    })
df = pd.DataFrame(records)
print(f"\nDataset : {len(df)} slides | "
      f"MSI-H={df['label_msi'].sum()} | "
      f"MSS={(df['label_msi']==0).sum()}")


# ================================================================
# PRELOAD ALL TILES + PERIPHERAL DISTANCE INTO MEMORY
# Appends the PD scalar to each tile's feature vector upfront
# so there is no per-batch overhead during training.
# ================================================================
print("\nPreloading all tiles + peripheral distance ...")
embed_cache = {}
tile_counts = []

for f in df["filename"]:
    data   = torch.load(os.path.join(EMB_DIR, f),
                        map_location="cpu", weights_only=False)
    feats  = data["features"].float()
    coords = data["coords"].float()
    W      = float(data["slide_dims"][0])
    H      = float(data["slide_dims"][1])

    # Normalize coordinates to [0, 1]
    coords_norm = coords / torch.tensor([W, H])

    # Append peripheral distance scalar to tile features
    embed_cache[f] = torch.cat(
        [feats, peripheral_distance(coords_norm)], dim=1
    )
    tile_counts.append(feats.shape[0])

print(f"Done — {len(embed_cache)} slides | input_dim={INPUT_DIM}")
print(f"Tiles per slide: "
      f"min={min(tile_counts)}  "
      f"max={max(tile_counts)}  "
      f"mean={int(np.mean(tile_counts))}")


# ================================================================
# DATASET — NO TILE SUBSAMPLING
# ================================================================
class MILDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Return all tiles — no subsampling
        return (
            embed_cache[row["filename"]],
            torch.tensor(row["label_msi"],   dtype=torch.float32),
            torch.tensor(row["label_hyper"], dtype=torch.float32),
        )


def collate_fn(batch):
    features, labels_msi, labels_hyper = zip(*batch)
    return list(features), torch.stack(labels_msi), torch.stack(labels_hyper)


# ================================================================
# LEARNING RATE SCHEDULE — Warmup + Cosine Annealing
# ================================================================
def get_lr(epoch, warmup, lr_max, lr_min, total):
    if epoch <= warmup:
        return lr_min + (lr_max - lr_min) * (epoch / warmup)
    progress = (epoch - warmup) / (total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * progress))


# ================================================================
# METRICS
# ================================================================
def get_metrics(probs, labels):
    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs >= 0.5).astype(int)
    auc    = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc":     auc,
        "f1":      f1_score(labels, preds, zero_division=0),
        "bal_acc": balanced_accuracy_score(labels, preds),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ================================================================
# TRAIN / EVALUATE
# ================================================================
def train_epoch(model, loader, optimizer, crit_msi, crit_hyper):
    model.train()
    total_loss = 0
    for features_list, labels_msi, labels_hyper in loader:
        for feats, lm, lh in zip(features_list, labels_msi, labels_hyper):
            feats = feats.to(DEVICE)
            lm    = lm.to(DEVICE)
            lh    = lh.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logit_msi, logit_hyper, _ = model(feats)
                loss = (crit_msi(logit_msi.view(1), lm.view(1)) +
                        crit_hyper(logit_hyper.view(1), lh.view(1)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    probs_msi, probs_hyper = [], []
    true_msi,  true_hyper  = [], []
    with torch.no_grad():
        for features_list, labels_msi, labels_hyper in loader:
            for feats, lm, lh in zip(features_list, labels_msi, labels_hyper):
                feats = feats.to(DEVICE)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logit_msi, logit_hyper, _ = model(feats)
                probs_msi.append(
                    torch.sigmoid(logit_msi.view(1)).cpu().float().item())
                probs_hyper.append(
                    torch.sigmoid(logit_hyper.view(1)).cpu().float().item())
                true_msi.append(lm.item())
                true_hyper.append(lh.item())
    return get_metrics(probs_msi, true_msi), get_metrics(probs_hyper, true_hyper)


# ================================================================
# 5-FOLD CROSS-VALIDATION
# ================================================================
w_msi   = torch.tensor([MSI_WEIGHT],   dtype=torch.float32).to(DEVICE)
w_hyper = torch.tensor([HYPER_WEIGHT], dtype=torch.float32).to(DEVICE)
crit_msi   = nn.BCEWithLogitsLoss(pos_weight=w_msi)
crit_hyper = nn.BCEWithLogitsLoss(pos_weight=w_hyper)

skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
all_msi_auc  = []
all_hyp_auc  = []
fold_results = []
t0           = time.time()

for fold, (train_idx, val_idx) in enumerate(
        skf.split(df, df["label_msi"].values), 1):

    print(f"\n{'='*70}\nFOLD {fold}/{N_FOLDS}")
    train_df = df.iloc[train_idx]
    val_df   = df.iloc[val_idx]
    print(f"  Train: {len(train_df)} | MSI-H={train_df['label_msi'].sum()}  "
          f"Val: {len(val_df)} | MSI-H={val_df['label_msi'].sum()}")

    train_loader = DataLoader(MILDataset(train_df), batch_size=1,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(MILDataset(val_df),   batch_size=1,
                              shuffle=False, collate_fn=collate_fn)

    model     = TransMIL_PD(INPUT_DIM, HIDDEN_DIM, N_HEADS, N_LAYERS, DROPOUT).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR_MIN, weight_decay=WEIGHT_DECAY)

    best_msi_auc = 0.0
    best_epoch   = 0
    best_state   = None
    best_msi_m   = None
    best_hyp_m   = None
    patience     = 0

    for epoch in range(1, EPOCHS + 1):
        # Update learning rate
        lr = get_lr(epoch, WARMUP_EPOCHS, LR_MAX, LR_MIN, EPOCHS)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        loss = train_epoch(model, train_loader, optimizer, crit_msi, crit_hyper)
        msi_m, hyp_m = evaluate(model, val_loader)

        print(f"  Ep {epoch:02d}/{EPOCHS} | lr={lr:.1e} | loss={loss:.4f} | "
              f"MSI={msi_m['auc']:.3f} F1={msi_m['f1']:.3f} | "
              f"Hyper={hyp_m['auc']:.3f}", end="\r")

        # Save best checkpoint based on MSI AUC
        if msi_m["auc"] > best_msi_auc:
            best_msi_auc = msi_m["auc"]
            best_epoch   = epoch
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            best_msi_m   = msi_m
            best_hyp_m   = hyp_m
            patience     = 0
        else:
            patience += 1

        if patience >= MAX_PATIENCE:
            print(f"\n  Early stop at epoch {epoch}")
            break

    print()

    # Save fold checkpoint
    torch.save({
        "model_state":   best_state,
        "fold":          fold,
        "best_epoch":    best_epoch,
        "metrics_msi":   best_msi_m,
        "metrics_hyper": best_hyp_m,
        "input_dim":     INPUT_DIM,
        "encoder":       "uni2h",
        "mil":           "transmil_pd_v4",
    }, os.path.join(SAVE_DIR, f"fold{fold}.pth"))

    print(f"  Best epoch : {best_epoch}")
    print(f"  MSI   → AUC={best_msi_m['auc']:.3f}  "
          f"F1={best_msi_m['f1']:.3f}  "
          f"BalAcc={best_msi_m['bal_acc']:.3f}  "
          f"TP={best_msi_m['tp']}  FN={best_msi_m['fn']}")
    print(f"  Hyper → AUC={best_hyp_m['auc']:.3f}  "
          f"F1={best_hyp_m['f1']:.3f}")

    all_msi_auc.append(best_msi_m["auc"])
    all_hyp_auc.append(best_hyp_m["auc"])
    fold_results.append({
        "fold":       fold,
        "msi_auc":    best_msi_m["auc"],
        "hyper_auc":  best_hyp_m["auc"],
        "msi_f1":     best_msi_m["f1"],
        "best_epoch": best_epoch,
    })
    torch.cuda.empty_cache()

# Save fold results summary
results_df = pd.DataFrame(fold_results)
results_df.to_csv(os.path.join(SAVE_DIR, "fold_results.csv"), index=False)

print(f"\n{'='*70}")
print(f"FINAL — UNI2-h + TransMIL + PD  (5-Fold CV, ALL tiles)")
print(f"{'='*70}")
print(f"MSI AUC  : {np.mean(all_msi_auc):.3f} ± {np.std(all_msi_auc):.3f}")
print(f"           folds → {' | '.join([f'{x:.3f}' for x in all_msi_auc])}")
print(f"Hyper AUC: {np.mean(all_hyp_auc):.3f} ± {np.std(all_hyp_auc):.3f}")
print(f"Time     : {(time.time()-t0)/60:.2f} min")
print(f"Saved to : {SAVE_DIR}")
print("="*70)
