"""
Baseline Training Script — All 6 Configurations (Table 1)
==========================================================
Trains all six foundation model × MIL aggregator combinations
for simultaneous MSI, MSS, and hypermutation prediction.

Configurations (Table 1):
    Row 1: UNI2-h   + ABMIL     → COAD AUC=0.948 | READ Spec=0.796
    Row 2: UNI2-h   + CLAM-SB   → COAD AUC=0.935 | READ Spec=0.837
    Row 3: UNI2-h   + TransMIL  → COAD AUC=0.957 | READ Spec=0.939
    Row 4: Virchow2 + ABMIL     → COAD AUC=0.934 | READ Spec=0.408
    Row 5: Virchow2 + CLAM-SB   → COAD AUC=0.929 | READ Spec=0.939
    Row 6: Virchow2 + TransMIL  → COAD AUC=0.915 | READ Spec=0.878

Usage:
    # Train a specific configuration
    python baselines/train_baselines.py --encoder uni2h --aggregator abmil
    python baselines/train_baselines.py --encoder uni2h --aggregator clam
    python baselines/train_baselines.py --encoder uni2h --aggregator transmil
    python baselines/train_baselines.py --encoder virchow2 --aggregator abmil
    python baselines/train_baselines.py --encoder virchow2 --aggregator clam
    python baselines/train_baselines.py --encoder virchow2 --aggregator transmil
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              balanced_accuracy_score, confusion_matrix)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.abmil import ABMIL_MultiTask
from models.clam import CLAM_SB_MultiTask
from models.transmil import TransMIL_MultiTask, subsample

# ================================================================
# ARGUMENT PARSING
# ================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--encoder",    type=str, required=True,
                    choices=["uni2h", "virchow2"],
                    help="Foundation model encoder")
parser.add_argument("--aggregator", type=str, required=True,
                    choices=["abmil", "clam", "transmil"],
                    help="MIL aggregator")
args = parser.parse_args()

ENCODER    = args.encoder
AGGREGATOR = args.aggregator

# ================================================================
# PATHS — Update to match your Google Drive structure
# ================================================================
EMB_DIRS = {
    "uni2h":    "/content/drive/MyDrive/TCGA_COAD/embeddings_uni2h_coad",
    "virchow2": "/content/drive/MyDrive/TCGA_COAD/embeddings_virchow2_coad",
}
CSV_PATH = "/content/drive/MyDrive/TCGA_COAD/coad_labels_combined.csv"
SAVE_BASE = "/content/drive/MyDrive/TCGA_COAD/saved_models_137"

EMB_DIR  = EMB_DIRS[ENCODER]
SAVE_DIR = os.path.join(SAVE_BASE, f"{ENCODER}_{AGGREGATOR}_baseline")
os.makedirs(SAVE_DIR, exist_ok=True)

# ================================================================
# HYPERPARAMETERS
# ================================================================
FEAT_DIMS = {"uni2h": 1536, "virchow2": 2560}
FEAT_DIM  = FEAT_DIMS[ENCODER]

HIDDEN_DIM    = 256 if AGGREGATOR in ["abmil", "clam"] else 512
DROPOUT       = 0.25
EPOCHS        = 20
LR            = 2e-4
WEIGHT_DECAY  = 1e-5
MSI_WEIGHT    = 8.0
HYPER_WEIGHT  = 4.07
MAX_PATIENCE  = 10
MAX_TILES_TRANS = 2000   # tile subsampling for TransMIL baselines
BAG_WEIGHT    = 0.7      # CLAM: weight for bag loss vs instance loss
N_FOLDS       = 5
SEED          = 42
DROP_SLIDES   = {"TCGA-AA-3678", "TCGA-AA-3833"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Encoder    : {ENCODER}  feat_dim={FEAT_DIM}")
print(f"Aggregator : {AGGREGATOR}")
print(f"Device     : {DEVICE}")
print(f"Save dir   : {SAVE_DIR}")


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
# PRELOAD EMBEDDINGS
# ================================================================
print("\nPreloading embeddings ...")
embed_cache = {}
for f in df["filename"]:
    data = torch.load(os.path.join(EMB_DIR, f),
                      map_location="cpu", weights_only=False)
    embed_cache[f] = data["features"].float()
print(f"Done — {len(embed_cache)} slides")


# ================================================================
# DATASET
# ================================================================
class MILDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return (
            embed_cache[row["filename"]],
            torch.tensor(row["label_msi"],   dtype=torch.float32),
            torch.tensor(row["label_hyper"], dtype=torch.float32),
        )


def collate_fn(batch):
    features, labels_msi, labels_hyper = zip(*batch)
    return list(features), torch.stack(labels_msi), torch.stack(labels_hyper)


# ================================================================
# METRICS
# ================================================================
def get_metrics(probs, labels):
    probs  = np.array(probs)
    labels = np.array(labels)
    preds  = (probs >= 0.5).astype(int)
    auc     = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auc":     auc,
        "f1":      f1_score(labels, preds, zero_division=0),
        "acc":     accuracy_score(labels, preds),
        "bal_acc": balanced_accuracy_score(labels, preds),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


# ================================================================
# TRAIN / EVAL (aggregator-specific)
# ================================================================
def train_epoch(model, loader, optimizer, crit_msi, crit_hyper):
    model.train()
    total_loss = 0

    for features_list, labels_msi, labels_hyper in loader:
        for feats, lm, lh in zip(features_list, labels_msi, labels_hyper):

            # Tile subsampling for TransMIL
            if AGGREGATOR == "transmil":
                feats = subsample(feats, MAX_TILES_TRANS)

            feats = feats.to(DEVICE)
            lm    = lm.to(DEVICE)
            lh    = lh.to(DEVICE)

            optimizer.zero_grad()

            if AGGREGATOR == "clam":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16) \
                        if AGGREGATOR == "transmil" else torch.no_grad.__class__():
                    logit_msi, logit_hyper, _, inst_loss = model(
                        feats, lm, lh, training=True)
                bag_loss = (crit_msi(logit_msi.view(1), lm.view(1)) +
                            crit_hyper(logit_hyper.view(1), lh.view(1)))
                loss = BAG_WEIGHT * bag_loss + (1 - BAG_WEIGHT) * inst_loss
            elif AGGREGATOR == "transmil":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logit_msi, logit_hyper, _ = model(feats)
                    loss = (crit_msi(logit_msi.view(1), lm.view(1)) +
                            crit_hyper(logit_hyper.view(1), lh.view(1)))
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            else:  # abmil
                logit_msi, logit_hyper, _ = model(feats)
                loss = (crit_msi(logit_msi.view(1), lm.view(1)) +
                        crit_hyper(logit_hyper.view(1), lh.view(1)))

            loss.backward()
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
                if AGGREGATOR == "transmil":
                    feats = subsample(feats, MAX_TILES_TRANS)
                feats = feats.to(DEVICE)

                if AGGREGATOR == "clam":
                    logit_msi, logit_hyper, _, _ = model(feats)
                else:
                    logit_msi, logit_hyper, _ = model(feats)

                probs_msi.append(torch.sigmoid(logit_msi.view(1)).cpu().item())
                probs_hyper.append(torch.sigmoid(logit_hyper.view(1)).cpu().item())
                true_msi.append(lm.item())
                true_hyper.append(lh.item())

    return get_metrics(probs_msi, true_msi), get_metrics(probs_hyper, true_hyper)


# ================================================================
# MODEL FACTORY
# ================================================================
def build_model():
    if AGGREGATOR == "abmil":
        return ABMIL_MultiTask(FEAT_DIM, HIDDEN_DIM, DROPOUT).to(DEVICE)
    elif AGGREGATOR == "clam":
        return CLAM_SB_MultiTask(FEAT_DIM, HIDDEN_DIM, DROPOUT).to(DEVICE)
    elif AGGREGATOR == "transmil":
        return TransMIL_MultiTask(FEAT_DIM, 512, 8, 2, DROPOUT).to(DEVICE)


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

    train_loader = DataLoader(MILDataset(train_df), batch_size=1,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(MILDataset(val_df),   batch_size=1,
                              shuffle=False, collate_fn=collate_fn)

    model     = build_model()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=LR, weight_decay=WEIGHT_DECAY)

    best_msi_auc = 0.0
    best_epoch   = 0
    best_state   = None
    best_msi_m   = None
    best_hyp_m   = None
    patience     = 0

    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_loader, optimizer, crit_msi, crit_hyper)
        msi_m, hyp_m = evaluate(model, val_loader)

        print(f"  Ep {epoch:02d}/{EPOCHS} | loss={loss:.4f} | "
              f"MSI={msi_m['auc']:.3f} F1={msi_m['f1']:.3f} | "
              f"Hyper={hyp_m['auc']:.3f}", end="\r")

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

    torch.save({
        "model_state":   best_state,
        "fold":          fold,
        "best_epoch":    best_epoch,
        "metrics_msi":   best_msi_m,
        "metrics_hyper": best_hyp_m,
        "encoder":       ENCODER,
        "aggregator":    AGGREGATOR,
        "feat_dim":      FEAT_DIM,
    }, os.path.join(SAVE_DIR, f"fold{fold}.pth"))

    print(f"  Best epoch : {best_epoch}")
    print(f"  MSI   → AUC={best_msi_m['auc']:.3f}  F1={best_msi_m['f1']:.3f}")
    print(f"  Hyper → AUC={best_hyp_m['auc']:.3f}")

    all_msi_auc.append(best_msi_m["auc"])
    all_hyp_auc.append(best_hyp_m["auc"])
    fold_results.append({
        "fold":       fold,
        "msi_auc":    best_msi_m["auc"],
        "hyper_auc":  best_hyp_m["auc"],
        "best_epoch": best_epoch,
    })
    torch.cuda.empty_cache()

pd.DataFrame(fold_results).to_csv(
    os.path.join(SAVE_DIR, "fold_results.csv"), index=False)

print(f"\n{'='*70}")
print(f"FINAL — {ENCODER} + {AGGREGATOR}  (5-Fold CV)")
print(f"{'='*70}")
print(f"MSI AUC  : {np.mean(all_msi_auc):.3f} ± {np.std(all_msi_auc):.3f}")
print(f"           folds → {' | '.join([f'{x:.3f}' for x in all_msi_auc])}")
print(f"Hyper AUC: {np.mean(all_hyp_auc):.3f} ± {np.std(all_hyp_auc):.3f}")
print(f"Time     : {(time.time()-t0)/60:.2f} min")
print(f"Saved to : {SAVE_DIR}")
