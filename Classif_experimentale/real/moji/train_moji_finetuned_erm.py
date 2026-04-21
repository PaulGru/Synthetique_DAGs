#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path as _Path
# Ajoute la racine du projet + le dossier shared/ au chemin Python
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

"""
train_moji_finetuned_erm.py
===========================
Modèle 3 : Fine-tuning ERM complet (la baseline forte).

Fine-tune l'intégralité de DistilBERT + tête linéaire avec ERM classique
sur le dataset Moji biaisé (environnements E1 + E2).

Ce script est la première étape d'une série de 2 :
  Modèle 3 (ce script) : ERM full fine-tune → adapte le backbone au domaine
  Modèle 5 (irm+ft)    : gèle ce backbone, réentraîne la tête avec la pénalité IRM

Hypothèse : même s'il encode le raccourci dialecte→sentiment, ce backbone
sera un meilleur point de départ pour IRM que les embeddings génériques
de DistilBERT pré-entraîné sur WikiBooks.

Sauvegardes dans --out_dir :
    best_backbone.pt       — state_dict du DistilBERT (bert.*)
    best_head.pt           — state_dict de la tête linéaire
    best_full_model.pt     — full state_dict (backbone + tête)
    results.json           — métriques finales
    history.json           — courbes d'entraînement par epoch

Usage :
    uv run train_moji_finetuned_erm.py --device auto
    uv run train_moji_finetuned_erm.py --device auto --epochs 3 --lr 2e-5
    uv run train_moji_finetuned_erm.py --device auto --unfreeze_last_n 2
"""


import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

# Réutilisation des utilitaires du script principal
from run_moji_erm_vs_irm import (
    load_moji,
    _get_group_indices,
    build_train_envs,
    build_val_ind,
    build_test_ood,
    print_split_stats,
)
from utils_irm import resolve_device


# =============================================================================
# 1. Dataset PyTorch — textes bruts, tokenisation dynamique par batch
# =============================================================================

class MojiTextDataset(Dataset):
    """
    Dataset minimal stockant les textes + labels bruts.
    La tokenisation (padding dynamique) se fait dans le collate_fn
    pour minimiser le padding inutile entre batches.
    """

    def __init__(
        self,
        texts: List[str],
        labels: np.ndarray,
        attributes: np.ndarray,
    ):
        self.texts = texts
        self.labels = labels.astype(np.float32)
        self.attributes = attributes.astype(np.int64)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        return self.texts[idx], float(self.labels[idx]), int(self.attributes[idx])


def make_collate_fn(tokenizer, max_length: int):
    """Retourne un collate_fn qui tokenise dynamiquement un batch de textes."""
    def collate(batch):
        texts, labels, attrs = zip(*batch)
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return (
            enc["input_ids"],
            enc["attention_mask"],
            torch.tensor(labels, dtype=torch.float32),
            torch.tensor(attrs, dtype=torch.long),
        )
    return collate


# =============================================================================
# 3. Modèle : DistilBERT + tête linéaire
# =============================================================================

class FineTunedDistilBERT(nn.Module):
    """
    DistilBERT (ou autre backbone HuggingFace) avec mean pooling + tête binaire.

    Méthodes auxiliaires :
      encode()             — embedding sans la tête (pour IRM)
      get_backbone_state() — state_dict du transformer (pour réutilisation)
      get_head_state()     — state_dict de la tête linéaire
    """

    def __init__(self, model_name: str = "distilbert-base-uncased"):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        d_model = self.bert.config.hidden_size        # 768 pour DistilBERT
        self.classifier = nn.Linear(d_model, 1)

    @staticmethod
    def _mean_pool(
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._mean_pool(out.last_hidden_state, attention_mask)
        return self.classifier(pooled).squeeze(-1)   # (B,)

    @torch.no_grad()
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Retourne uniquement le pooling (embedding), sans la tête."""
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self._mean_pool(out.last_hidden_state, attention_mask)

    def get_backbone_state(self) -> dict:
        return self.bert.state_dict()

    def get_head_state(self) -> dict:
        return self.classifier.state_dict()


# =============================================================================
# 4. Évaluation sur DataLoader
# =============================================================================

@torch.no_grad()
def evaluate_loader(
    model: FineTunedDistilBERT,
    loader: DataLoader,
    device: str,
    label: str = "",
) -> Dict:
    """
    Évalue le modèle fine-tuné sur un DataLoader et retourne les mêmes
    métriques que full_evaluation() dans run_moji_erm_vs_irm.py.
    """
    from sklearn.metrics import f1_score

    model.eval()
    all_logits, all_labels, all_attrs = [], [], []
    loss_total, n_total = 0.0, 0
    criterion = nn.BCEWithLogitsLoss()
    use_fp16 = str(device).startswith("cuda")

    for input_ids, attention_mask, labels, attrs in loader:
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_d       = labels.to(device)

        if use_fp16:
            with torch.autocast(device_type="cuda"):
                logits = model(input_ids, attention_mask)
        else:
            logits = model(input_ids, attention_mask)

        loss_total += criterion(logits.float(), labels_d).item() * len(labels)
        n_total    += len(labels)
        all_logits.append(logits.cpu().float())
        all_labels.append(labels)
        all_attrs.append(attrs)

    logits_np = torch.cat(all_logits).numpy()
    y_np      = torch.cat(all_labels).numpy().astype(np.int64)
    A_np      = torch.cat(all_attrs).numpy().astype(np.int64)
    y_pred    = (torch.sigmoid(torch.from_numpy(logits_np)) >= 0.5).numpy().astype(np.int64)

    acc       = float((y_pred == y_np).mean())
    macro_f1  = float(f1_score(y_np, y_pred, average="macro", zero_division=0))
    loss      = loss_total / n_total

    mask_aae = A_np == 1
    mask_sae = A_np == 0

    def _tpr(a_mask):
        pos = (y_np == 1) & a_mask
        return float(y_pred[pos].mean()) if pos.sum() > 0 else float("nan")

    def _fpr(a_mask):
        neg = (y_np == 0) & a_mask
        return float(y_pred[neg].mean()) if neg.sum() > 0 else float("nan")

    def _acc_group(mask):
        return float((y_pred[mask] == y_np[mask]).mean()) if mask.sum() > 0 else float("nan")

    tpr_aae, tpr_sae = _tpr(mask_aae), _tpr(mask_sae)
    fpr_aae, fpr_sae = _fpr(mask_aae), _fpr(mask_sae)

    fnr_pos_aae = 1.0 - tpr_aae if not np.isnan(tpr_aae) else float("nan")
    fpr_neg_sae = fpr_sae
    eod_tpr     = abs(tpr_sae - tpr_aae) if not (np.isnan(tpr_sae) or np.isnan(tpr_aae)) else float("nan")
    eod_fpr     = abs(fpr_sae - fpr_aae) if not (np.isnan(fpr_sae) or np.isnan(fpr_aae)) else float("nan")

    acc_groups = {
        "(Y=0,A=0)": _acc_group((y_np == 0) & mask_sae),
        "(Y=0,A=1)": _acc_group((y_np == 0) & mask_aae),
        "(Y=1,A=0)": _acc_group((y_np == 1) & mask_sae),
        "(Y=1,A=1)": _acc_group((y_np == 1) & mask_aae),
    }
    valid_accs     = [v for v in acc_groups.values() if not np.isnan(v)]
    worst_group_acc = float(min(valid_accs))  if valid_accs else float("nan")
    avg_group_acc   = float(np.mean(valid_accs)) if valid_accs else float("nan")

    res = {
        "accuracy":        acc,
        "macro_f1":        macro_f1,
        "loss":            loss,
        "fnr_pos_aae":     fnr_pos_aae,
        "fpr_neg_sae":     fpr_neg_sae,
        "eod_tpr":         eod_tpr,
        "eod_fpr":         eod_fpr,
        "worst_group_acc": worst_group_acc,
        "avg_group_acc":   avg_group_acc,
        "acc_groups":      acc_groups,
    }
    if label:
        print(
            f"  [{label}] Acc={acc:.4f}  MacroF1={macro_f1:.4f}  Loss={loss:.4f}\n"
            f"           FNR(AAE)={fnr_pos_aae:.4f}  FPR(SAE)={fpr_neg_sae:.4f}\n"
            f"           EOD_TPR={eod_tpr:.4f}  EOD_FPR={eod_fpr:.4f}\n"
            f"           Worst-Group={worst_group_acc:.4f}  Avg-Group={avg_group_acc:.4f}"
        )
    return res


# =============================================================================
# 5. Boucle d'entraînement ERM (fine-tuning complet)
# =============================================================================

def train_finetuned_erm(
    model: FineTunedDistilBERT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
    warmup_fraction: float,
    weight_decay: float,
    device: str,
    out_dir: str,
) -> dict:
    """
    Fine-tuning ERM : minimise la BCE sur E1+E2 (distribution biaisée).

    Optimiseur :
      - AdamW avec weight decay sur tous les paramètres sauf bias/LayerNorm
      - LR de la tête × 10 par rapport au backbone (standard en transfer learning)
      - Schedule linéaire avec warmup (6 % des steps total par défaut)

    Sélection du meilleur modèle : val_ind accuracy globale.
    Sauvegarde dans out_dir : best_backbone.pt, best_head.pt, best_full_model.pt.

    Note : la val_ind est biaisée (80/20) — c'est intentionnel. On veut que
    le Modèle 3 soit un bon modèle ERM standard, pas un modèle équitable.
    IRM post-finetuning (Modèle 5) corrigera l'équité.
    """
    from transformers import get_linear_schedule_with_warmup

    use_fp16 = str(device).startswith("cuda")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # ── Groupes de paramètres pour AdamW ──
    # bias et LayerNorm sont exclus du weight decay (pratique standard BERT)
    no_decay = {"bias", "LayerNorm.weight"}
    backbone_wd, backbone_nd, head_params = [], [], []
    for name, param in model.bert.named_parameters():
        if any(nd in name for nd in no_decay):
            backbone_nd.append(param)
        else:
            backbone_wd.append(param)
    head_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_wd, "lr": lr,      "weight_decay": weight_decay},
        {"params": backbone_nd, "lr": lr,      "weight_decay": 0.0},
        {"params": head_params, "lr": lr * 10, "weight_decay": weight_decay},
    ])

    total_steps   = epochs * len(train_loader)
    warmup_steps  = int(total_steps * warmup_fraction)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_acc": [], "val_macro_f1": [], "val_worst_group": [],
        "test_acc": [], "test_macro_f1": [], "test_worst_group": [],
    }
    best_val_acc = -1.0
    best_epoch   = -1

    for epoch in range(1, epochs + 1):
        # ────────── Train ──────────
        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0

        for step, (input_ids, attention_mask, labels, _) in enumerate(train_loader, 1):
            input_ids      = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels_d       = labels.to(device)

            optimizer.zero_grad()

            if use_fp16:
                with torch.autocast(device_type="cuda"):
                    logits = model(input_ids, attention_mask)
                    loss   = criterion(logits.float(), labels_d)
            else:
                logits = model(input_ids, attention_mask)
                loss   = criterion(logits, labels_d)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits.float()) >= 0.5).float()
                running_correct += (preds == labels_d.float()).sum().item()
            running_loss += loss.item() * len(labels)
            running_n    += len(labels)

            if step % 200 == 0:
                print(
                    f"    Epoch {epoch}/{epochs}  step {step}/{len(train_loader)}"
                    f"  loss={running_loss/running_n:.4f}"
                    f"  acc={running_correct/running_n:.4f}"
                )

        train_loss = running_loss / running_n
        train_acc  = running_correct / running_n

        # ────────── Eval ──────────
        val_res  = evaluate_loader(model, val_loader,  device)
        test_res = evaluate_loader(model, test_loader, device)

        print(
            f"\n  [Epoch {epoch}/{epochs}]"
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}\n"
            f"    val   acc={val_res['accuracy']:.4f}"
            f"  f1={val_res['macro_f1']:.4f}"
            f"  worst={val_res['worst_group_acc']:.4f}\n"
            f"    test  acc={test_res['accuracy']:.4f}"
            f"  f1={test_res['macro_f1']:.4f}"
            f"  worst={test_res['worst_group_acc']:.4f}"
        )

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_res["accuracy"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["val_worst_group"].append(val_res["worst_group_acc"])
        history["test_acc"].append(test_res["accuracy"])
        history["test_macro_f1"].append(test_res["macro_f1"])
        history["test_worst_group"].append(test_res["worst_group_acc"])

        # ────────── Checkpoint ──────────
        if val_res["accuracy"] > best_val_acc:
            best_val_acc = val_res["accuracy"]
            best_epoch   = epoch
            torch.save(
                model.bert.state_dict(),
                os.path.join(out_dir, "best_backbone.pt"),
            )
            torch.save(
                model.classifier.state_dict(),
                os.path.join(out_dir, "best_head.pt"),
            )
            torch.save(
                model.state_dict(),
                os.path.join(out_dir, "best_full_model.pt"),
            )
            print(f"    → Best model sauvegardé (epoch {epoch}, val_acc={best_val_acc:.4f})")

    print(f"\n  Best model : epoch {best_epoch}  val_acc={best_val_acc:.4f}")
    return history


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Modèle 3 — Fine-tuning ERM complet de DistilBERT sur Moji"
    )
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--device",         type=str,   default="auto")

    # Dataset
    parser.add_argument("--max_per_group",  type=int,   default=5000,
        help="Taille max par groupe minoritaire dans chaque env (E1/E2)")
    parser.add_argument("--n_per_group_test", type=int, default=7500,
        help="Exemples par groupe dans le Test OOD équilibré")
    parser.add_argument(
        "--sae_ratio", type=float, default=2.0,
        help="Ratio N_SAE / N_AAE dans chaque environnement d'entraînement (défaut 2.0).",
    )

    # Backbone
    parser.add_argument("--bert_model",     type=str,   default="distilbert-base-uncased")
    parser.add_argument("--max_length",     type=int,   default=128)
    parser.add_argument(
        "--unfreeze_last_n", type=int, default=0,
        help=(
            "0 = fine-tune toutes les couches (comportement par défaut).\n"
            "N > 0 = gèle toutes les couches sauf les N dernières couches "
            "transformer. Ex : --unfreeze_last_n 2 pour Modèle 3 partiel."
        ),
    )

    # Entraînement
    parser.add_argument("--epochs",          type=int,   default=3)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=2e-5)
    parser.add_argument("--weight_decay",    type=float, default=0.01)
    parser.add_argument("--warmup_fraction", type=float, default=0.06)

    # Sortie
    parser.add_argument("--out_dir", type=str,
                        default=str(_Path(__file__).parent / "logs" / "finetuned_erm"))

    args   = parser.parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ─────────────────────────────────────────────────────────────────────
    # Étape 1 : Chargement
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 1 : Chargement du dataset LabHC/moji")
    print("=" * 70)
    ds = load_moji()

    # ─────────────────────────────────────────────────────────────────────
    # Étape 2 : Splits
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 2 : Construction des splits")
    print("=" * 70)

    idx_e1, idx_e2 = build_train_envs(
        ds, max_per_group=args.max_per_group, seed=args.seed, sae_ratio=args.sae_ratio,
    )
    idx_val_ind  = build_val_ind(ds, seed=args.seed + 1)
    idx_test_ood = build_test_ood(ds, n_per_group=args.n_per_group_test, seed=args.seed)
    all_split = ds["all"]
    Y_all = np.asarray(all_split.Y)
    A_all = np.asarray(all_split.A)
    print_split_stats("E1 — biais fort    (90/10)", idx_e1, Y_all, A_all)
    print_split_stats("E2 — biais modéré (70/30)", idx_e2, Y_all, A_all)
    print_split_stats("Val InD — biaisée 80/20", idx_val_ind, Y_all, A_all)
    print_split_stats(f"Test OOD — équilibré ({args.n_per_group_test}/groupe)", idx_test_ood, Y_all, A_all)

    # ─────────────────────────────────────────────────────────────────────
    # Étape 3 : Tokenizer & DataLoaders
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 3 : Assemblage des DataLoaders")
    print("=" * 70)

    from transformers import AutoTokenizer
    tokenizer  = AutoTokenizer.from_pretrained(args.bert_model)
    collate_fn = make_collate_fn(tokenizer, max_length=args.max_length)

    all_split   = ds["all"]

    # E1 et E2 sont disjoints par construction → pas de doublons à dédupliquer
    all_train_idx = np.concatenate([idx_e1, idx_e2])

    def _texts_labels_attrs(split, indices):
        texts  = [split.comment_text[int(i)] for i in indices]
        labels = np.asarray(split.Y)[indices]
        attrs  = np.asarray(split.A)[indices]
        return texts, labels, attrs

    train_texts, train_labels, train_attrs = _texts_labels_attrs(all_split, all_train_idx)
    val_texts,   val_labels,   val_attrs   = _texts_labels_attrs(all_split, idx_val_ind)
    test_texts,  test_labels,  test_attrs  = _texts_labels_attrs(all_split, idx_test_ood)

    print(f"  Données train : {len(train_texts):,}")
    print(f"  Données val   : {len(val_texts):,}")
    print(f"  Données test  : {len(test_texts):,}")

    use_pin = str(device).startswith("cuda")
    train_loader = DataLoader(
        MojiTextDataset(train_texts, train_labels, train_attrs),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=use_pin,
    )
    val_loader = DataLoader(
        MojiTextDataset(val_texts, val_labels, val_attrs),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=use_pin,
    )
    test_loader = DataLoader(
        MojiTextDataset(test_texts, test_labels, test_attrs),
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=use_pin,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 4 : Modèle
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 4 : Initialisation du modèle")
    print("=" * 70)

    model = FineTunedDistilBERT(model_name=args.bert_model)

    # Fine-tuning partiel : gèle tout sauf les N dernières couches transformer
    if args.unfreeze_last_n > 0:
        # Gèle d'abord tout le backbone
        for param in model.bert.parameters():
            param.requires_grad = False
        # Dégèle les N dernières couches
        # Pour DistilBERT : model.bert.transformer.layer (6 couches)
        # Pour BERT / RoBERTa : model.bert.encoder.layer
        transformer_block = getattr(
            model.bert, "transformer", getattr(model.bert, "encoder", None)
        )
        if transformer_block is not None and hasattr(transformer_block, "layer"):
            layers = transformer_block.layer
            for layer in layers[-args.unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            print(
                f"  Fine-tuning partiel : "
                f"{args.unfreeze_last_n} dernière(s) couche(s) sur {len(layers)}"
            )
        else:
            print("  AVERTISSEMENT : --unfreeze_last_n ignoré (architecture non reconnue)")
    else:
        print("  Fine-tuning complet (toutes les couches)")

    n_total     = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Paramètres total       : {n_total:,}")
    print(f"  Paramètres entraînables: {n_trainable:,}  ({100*n_trainable/n_total:.1f} %)")

    # ─────────────────────────────────────────────────────────────────────
    # Étape 5 : Fine-tuning ERM
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 5 : Fine-tuning ERM")
    print(f"  epochs={args.epochs}  batch={args.batch_size}  lr={args.lr}")
    print("=" * 70)

    history = train_finetuned_erm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=args.epochs,
        lr=args.lr,
        warmup_fraction=args.warmup_fraction,
        weight_decay=args.weight_decay,
        device=device,
        out_dir=args.out_dir,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 6 : Évaluation finale (meilleur modèle)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 6 : Évaluation finale (best model)")
    print("=" * 70)

    best_path = os.path.join(args.out_dir, "best_full_model.pt")
    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=True)
    )
    model = model.to(device)

    print("\n--- Modèle 3 : ERM Fine-tuned DistilBERT ---")
    results_val  = evaluate_loader(model, val_loader,  device, "Val InD")
    results_test = evaluate_loader(model, test_loader, device, "Test OOD")

    # ─────────────────────────────────────────────────────────────────────
    # Résumé
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — Modèle 3 : ERM Fine-tuned DistilBERT")
    print("=" * 70)

    print(f"\n  {'Metric':<35} {'Test OOD':>10}  {'Val InD':>10}")
    print("  " + "-" * 60)
    for metric, label_str in [
        ("accuracy",        "Accuracy"),
        ("macro_f1",        "Macro-F1"),
        ("loss",            "Loss"),
        ("fnr_pos_aae",     "FNR  (Y=1, A=1)  AAE positifs"),
        ("fpr_neg_sae",     "FPR  (Y=0, A=0)  SAE négatifs"),
        ("eod_tpr",         "EOD TPR  |TPR_SAE − TPR_AAE|"),
        ("eod_fpr",         "EOD FPR  |FPR_SAE − FPR_AAE|"),
        ("worst_group_acc", "Worst-Group Accuracy"),
        ("avg_group_acc",   "Avg-Group Accuracy"),
    ]:
        print(
            f"  {label_str:<35}"
            f" {results_test[metric]:>10.4f}"
            f"  {results_val[metric]:>10.4f}"
        )
    print(f"\n  {'Groupe':}")
    for gk, glabel in [
        ("(Y=0,A=0)", "    Neg SAE  (Y=0, A=0)"),
        ("(Y=0,A=1)", "    Neg AAE  (Y=0, A=1)"),
        ("(Y=1,A=0)", "    Pos SAE  (Y=1, A=0)"),
        ("(Y=1,A=1)", "    Pos AAE  (Y=1, A=1)"),
    ]:
        print(
            f"  {glabel:<35}"
            f" {results_test['acc_groups'][gk]:>10.4f}"
            f"  {results_val['acc_groups'][gk]:>10.4f}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Sauvegarde
    # ─────────────────────────────────────────────────────────────────────
    results = {
        "model":   "finetuned_erm",
        "val_ind": results_val,
        "test_ood": results_test,
    }
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n  Résultats  → {args.out_dir}/results.json")
    print(f"  Historique → {args.out_dir}/history.json")
    print(f"  Backbone   → {args.out_dir}/best_backbone.pt")
    print(f"  Tête       → {args.out_dir}/best_head.pt")
    print(f"  Full       → {args.out_dir}/best_full_model.pt")
    print("\nDone!")


if __name__ == "__main__":
    main()
