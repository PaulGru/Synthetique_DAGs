#!/usr/bin/env python3
"""
run_multinli_simple.py
======================
Fine-tune les dernières couches de BERT avec ERM vs IRMv1 sur MultiNLI
(3 genres = 3 environnements) et évalue sur :

    - MNLI validation_matched (ID, filtré sur les 3 genres)
    - MNLI validation_mismatched (OOD lexicale / domaine)

Usage :
    uv run real/multinli/run_multinli_simple.py --device auto
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
for _p in [str(_ROOT), str(_ROOT / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from torch.utils.data import DataLoader, Dataset

from utils_irm import resolve_device

# =============================================================================
# 1. Chargement des datasets
# =============================================================================

TRAIN_GENRES = ["fiction", "government"]

def load_all_datasets() -> dict:
    """Charge MultiNLI (HuggingFace)."""
    from datasets import load_dataset

    print("Chargement de MultiNLI …")
    mnli = load_dataset("nyu-mll/multi_nli")
    print(f"  train: {len(mnli['train']):,}  "
          f"val_matched: {len(mnli['validation_matched']):,}  "
          f"val_mismatched: {len(mnli['validation_mismatched']):,}")

    return mnli

# =============================================================================
# 2. Dataset & DataLoader
# =============================================================================

class _NLIPairDataset(Dataset):
    """Dataset NLI."""

    def __init__(self, premises, hypotheses, labels, tokenizer, max_length):
        self.premises   = premises
        self.hypotheses = hypotheses
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            "premise": self.premises[idx],
            "hypothesis": self.hypotheses[idx],
            "label": self.labels[idx],
        }
        return item

def _make_collate(tokenizer):
    """Collate avec padding dynamique."""
    def _collate(batch):
        padded = tokenizer(
            [item["premise"] for item in batch],
            [item["hypothesis"] for item in batch],
            padding=True,
            truncation=True,
            max_length=tokenizer.model_max_length,
            return_tensors="pt"
        )
        padded["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return padded
    return _collate

def _extract_mnli(split_ds, genre: str | None = None):
    """Extrait (premises, hypotheses, labels) depuis un split MNLI."""
    labels_col   = split_ds["label"]
    premises_col = split_ds["premise"]
    hyp_col      = split_ds["hypothesis"]
    genres_col   = split_ds["genre"] if "genre" in split_ds.column_names else None

    premises, hypotheses, labels = [], [], []
    for idx, label in enumerate(labels_col):
        if label == -1:
            continue
        if genre is not None and genres_col is not None and genres_col[idx] != genre:
            continue
        premises.append(premises_col[idx])
        hypotheses.append(hyp_col[idx])
        labels.append(label)
    return premises, hypotheses, labels

def build_env_loaders(
    mnli, tokenizer, max_length: int, batch_size: int,
) -> Dict[str, DataLoader]:
    """Construit un DataLoader par genre MNLI (3 envs)."""
    envs = {}
    collate = _make_collate(tokenizer)

    n_workers = min(8, os.cpu_count() or 1)
    for genre in TRAIN_GENRES:
        p, h, l = _extract_mnli(mnli["train"], genre=genre)
        ds = _NLIPairDataset(p, h, l, tokenizer, max_length)
        envs[genre] = DataLoader(ds, batch_size=batch_size, shuffle=True,
                                 num_workers=n_workers, collate_fn=collate,
                                 pin_memory=True, drop_last=False,
                                 persistent_workers=n_workers > 0)
        print(f"  Env {genre:12s} : {len(ds):,} paires")

    return envs

def build_eval_loaders(
    mnli, tokenizer, max_length: int, batch_size: int,
) -> Dict[str, DataLoader]:
    """
    Construit les loaders d'évaluation standard (val_matched filtré, val_mismatched).
    """
    collate_std = _make_collate(tokenizer)
    evals = {}
    n_workers = min(8, os.cpu_count() or 1)

    # Val matched global (ID) : on ne garde que les 3 genres utilisés
    p_matched, h_matched, l_matched = [], [], []
    for genre in TRAIN_GENRES:
        pg, hg, lg = _extract_mnli(mnli["validation_matched"], genre=genre)
        p_matched.extend(pg)
        h_matched.extend(hg)
        l_matched.extend(lg)
        
        ds_g = _NLIPairDataset(pg, hg, lg, tokenizer, max_length)
        evals[f"val_{genre}"] = DataLoader(ds_g, batch_size=batch_size * 2, shuffle=False,
                                           num_workers=n_workers, collate_fn=collate_std,
                                           pin_memory=True)
        
    ds_matched = _NLIPairDataset(p_matched, h_matched, l_matched, tokenizer, max_length)
    evals["val_matched"] = DataLoader(ds_matched, batch_size=batch_size * 2, shuffle=False,
                                      num_workers=n_workers, collate_fn=collate_std,
                                      pin_memory=True)
    
    print(f"  Val Matched (filtré) : {len(ds_matched):,} paires")

    # Val mismatched (OOD)
    p_mismatched, h_mismatched, l_mismatched = _extract_mnli(mnli["validation_mismatched"])
    ds_mismatched = _NLIPairDataset(p_mismatched, h_mismatched, l_mismatched, tokenizer, max_length)
    evals["val_mismatched"] = DataLoader(ds_mismatched, batch_size=batch_size * 2, shuffle=False,
                                         num_workers=n_workers, collate_fn=collate_std,
                                         pin_memory=True)
    print(f"  Val Mismatched (OOD) : {len(ds_mismatched):,} paires")

    return evals

# =============================================================================
# 3. Modèle BERT + tête
# =============================================================================

class BertNLIModel(nn.Module):
    """BERT/DistilBERT + dropout + tête linéaire pour NLI 3-classes."""

    def __init__(self, backbone: nn.Module, hidden_size: int, n_classes: int = 3):
        super().__init__()
        self.backbone   = backbone
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        out = self.backbone(**kw)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))

def _get_layer_modules(backbone) -> list:
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return list(backbone.encoder.layer)
    if hasattr(backbone, "transformer") and hasattr(backbone.transformer, "layer"):
        return list(backbone.transformer.layer)
    raise ValueError("Architecture non reconnue.")

def freeze_backbone_except_last_n(backbone: nn.Module, n_unfrozen: int = 2):
    for p in backbone.parameters():
        p.requires_grad = False
    layers = _get_layer_modules(backbone)
    n_layers = len(layers)
    for i in range(max(0, n_layers - n_unfrozen), n_layers):
        for p in layers[i].parameters():
            p.requires_grad = True
    n_total = sum(p.numel() for p in backbone.parameters())
    n_train = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"  Backbone : {n_layers} couches, {n_unfrozen} dégelées")
    print(f"  Params   : {n_train:,} / {n_total:,} entraînables ({100*n_train/n_total:.1f}%)")

# =============================================================================
# 4. Évaluation
# =============================================================================

@torch.no_grad()
def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Accuracy 3-classes sur un DataLoader MNLI."""
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        labels_b = batch["labels"].to(device)
        logits = model(input_ids, attention_mask, token_type_ids)
        correct += (logits.argmax(-1) == labels_b).sum().item()
        total   += len(labels_b)
    return correct / total if total > 0 else 0.0

def evaluate_all(
    model: nn.Module,
    eval_loaders: Dict[str, DataLoader],
    device: torch.device,
) -> dict:
    results = {}
    for name, loader in eval_loaders.items():
        results[name] = evaluate_loader(model, loader, device)
    return results

# =============================================================================
# 5. Fine-tuning ERM
# =============================================================================

def finetune_erm(
    model: BertNLIModel,
    env_loaders: Dict[str, DataLoader],
    eval_loaders: Dict[str, DataLoader],
    epochs: int,
    lr_bert: float,
    lr_head: float,
    device: torch.device,
    eval_every_steps: int = 500,
) -> Tuple[dict, list]:
    """Fine-tune ERM : cross-entropie poolée sur tous les envs."""
    from transformers import get_linear_schedule_with_warmup

    bert_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": lr_bert, "weight_decay": 0.01},
        {"params": head_params, "lr": lr_head, "weight_decay": 0.01},
    ])

    steps_per_epoch = max(len(loader) for loader in env_loaders.values())
    total_steps     = epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )

    loss_fn = nn.CrossEntropyLoss()
    use_autocast = "cuda" in str(device)
    scaler = torch.amp.GradScaler(enabled=use_autocast)
    history = []
    log_steps = 100

    print(f"\n  Fine-tuning ERM — {epochs} epochs, {total_steps:,} steps total")
    print(f"  lr_bert={lr_bert}, lr_head={lr_head}")

    global_step = 0
    step_loss = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        env_iters  = {name: iter(loader) for name, loader in env_loaders.items()}
        env_names  = list(env_loaders.keys())

        for _ in range(steps_per_epoch):
            batch_loss = torch.tensor(0.0, device=device)
            n_samples  = 0

            for name in env_names:
                try:
                    batch = next(env_iters[name])
                except StopIteration:
                    env_iters[name] = iter(env_loaders[name])
                    batch = next(env_iters[name])

                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                labels_b = batch["labels"].to(device)

                with torch.amp.autocast("cuda", enabled=use_autocast):
                    logits = model(input_ids, attention_mask, token_type_ids)
                    loss_e = loss_fn(logits, labels_b)
                batch_loss = batch_loss + loss_e
                correct    += (logits.detach().argmax(-1) == labels_b).sum().item()
                n_samples  += len(labels_b)

            batch_loss = batch_loss / len(env_names)
            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss  += batch_loss.item()
            step_loss = batch_loss.item()
            total       += n_samples
            global_step += 1

            if global_step % log_steps == 0:
                avg_loss = total_loss / (global_step - epoch * steps_per_epoch)
                print(f"    [ERM] step {global_step:,}/{total_steps:,}  loss={step_loss:.4f} (avg: {avg_loss:.4f})")

            if eval_every_steps and global_step % eval_every_steps == 0:
                res = evaluate_all(model, eval_loaders, device)
                history.append({"step": global_step, "epoch": epoch + 1, "loss": step_loss, **res})
                step_in_epoch = global_step - epoch * steps_per_epoch
                print(f"    [ERM] step {global_step:,}  "
                      f"loss={total_loss/max(1, step_in_epoch):.4f}  "
                      f"val_m={res.get('val_matched',0):.4f}  "
                      f"val_mm={res.get('val_mismatched',0):.4f}")
                model.train()

        epoch_loss = total_loss / steps_per_epoch
        epoch_acc  = correct / total if total > 0 else 0
        print(f"  → Epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f}  train_acc={epoch_acc:.4f}")

    return evaluate_all(model, eval_loaders, device), history

# =============================================================================
# 6. Fine-tuning IRM
# =============================================================================

def finetune_irm(
    model: BertNLIModel,
    env_loaders: Dict[str, DataLoader],
    eval_loaders: Dict[str, DataLoader],
    epochs: int,
    lr_bert: float,
    lr_head: float,
    irm_lambda: float,
    warmup_fraction: float,
    device: torch.device,
    eval_every_steps: int = 500,
) -> Tuple[dict, list]:
    """Fine-tune IRMv1 : 1 mini-batch par env, loss ERM + λ·pénalité."""
    from transformers import get_linear_schedule_with_warmup

    bert_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": bert_params, "lr": lr_bert, "weight_decay": 0.01},
        {"params": head_params, "lr": lr_head, "weight_decay": 0.01},
    ])

    steps_per_epoch = max(len(loader) for loader in env_loaders.values())
    total_steps     = epochs * steps_per_epoch
    warmup_steps    = int(warmup_fraction * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )

    loss_fn = nn.CrossEntropyLoss()
    use_autocast = "cuda" in str(device)
    scaler = torch.amp.GradScaler(enabled=use_autocast)
    history = []
    log_steps = 100
    E = len(env_loaders)

    print(f"\n  Fine-tuning IRM — {epochs} epochs, {total_steps:,} steps total")
    print(f"  lr_bert={lr_bert}, lr_head={lr_head}, λ_max={irm_lambda}")
    print(f"  warmup: {warmup_steps:,} steps ({warmup_fraction*100:.0f}%)")

    global_step = 0
    step_loss = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, total_penalty = 0.0, 0.0
        correct, total = 0, 0
        env_iters = {name: iter(loader) for name, loader in env_loaders.items()}
        env_names = list(env_loaders.keys())

        for _ in range(steps_per_epoch):
            emp_risk  = torch.tensor(0.0, device=device)
            penalties = []

            for name in env_names:
                try:
                    batch = next(env_iters[name])
                except StopIteration:
                    env_iters[name] = iter(env_loaders[name])
                    batch = next(env_iters[name])

                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                labels_b = batch["labels"].to(device)

                with torch.amp.autocast("cuda", enabled=use_autocast):
                    logits = model(input_ids, attention_mask, token_type_ids)
                    loss_e = loss_fn(logits, labels_b)

                emp_risk = emp_risk + loss_e

                scale = torch.tensor(1.0, device=device, requires_grad=True)
                loss_scaled = loss_fn(logits * scale, labels_b)
                grad_scale  = grad(loss_scaled, [scale], create_graph=True)[0]
                penalties.append(grad_scale ** 2)

                correct += (logits.detach().argmax(-1) == labels_b).sum().item()
                total   += len(labels_b)

            emp_risk = emp_risk / E
            penalty  = torch.stack(penalties).mean()

            lambda_t = irm_lambda * min(1.0, global_step / max(1, warmup_steps))
            objective = emp_risk + lambda_t * penalty
            if lambda_t > 1.0:
                objective = objective / lambda_t

            optimizer.zero_grad()
            scaler.scale(objective).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss    += emp_risk.item()
            total_penalty += penalty.item()
            step_loss = emp_risk.item()
            global_step   += 1

            if global_step % log_steps == 0:
                avg_loss = total_loss / (global_step - epoch * steps_per_epoch)
                avg_penalty = total_penalty / (global_step - epoch * steps_per_epoch)
                print(f"    [IRM] step {global_step:,}/{total_steps:,}  loss={step_loss:.4f} (avg: {avg_loss:.4f})  pen={avg_penalty:.4f}  λ={lambda_t:.1f}")

            if eval_every_steps and global_step % eval_every_steps == 0:
                res = evaluate_all(model, eval_loaders, device)
                step_in_epoch = global_step - epoch * steps_per_epoch
                history.append({"step": global_step, "epoch": epoch + 1,
                                 "loss": step_loss, "penalty": total_penalty / max(1, step_in_epoch), **res})
                print(f"    [IRM] step {global_step:,}  "
                      f"loss={total_loss/max(1, step_in_epoch):.4f}  "
                      f"pen={total_penalty/max(1, step_in_epoch):.4f}  "
                      f"λ={lambda_t:.1f}  "
                      f"val_m={res.get('val_matched',0):.4f}  "
                      f"val_mm={res.get('val_mismatched',0):.4f}")
                model.train()

        epoch_loss = total_loss / steps_per_epoch
        epoch_acc  = correct / total if total > 0 else 0
        print(f"  → Epoch {epoch+1}/{epochs}  loss={epoch_loss:.4f}  train_acc={epoch_acc:.4f}")

    return evaluate_all(model, eval_loaders, device), history

# =============================================================================
# 7. Visualisation
# =============================================================================

def plot_comparison(results_erm: dict, results_irm: dict, out_dir: str):
    """Bar chart ERM vs IRM sur val_matched et val_mismatched."""
    sets = ["val_matched", "val_mismatched"]
    labels = ["Val Matched\n(ID)", "Val Mismatch\n(OOD)"]
    x = np.arange(len(sets))
    width = 0.35

    erm_accs = [results_erm.get(s, 0) for s in sets]
    irm_accs = [results_irm.get(s, 0) for s in sets]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width / 2, erm_accs, width, label="ERM fine-tune", color="#e74c3c", edgecolor="black")
    ax.bar(x + width / 2, irm_accs, width, label="IRM fine-tune", color="#2ecc71", edgecolor="black")

    for i, (e, r) in enumerate(zip(erm_accs, irm_accs)):
        ax.text(i - width / 2, e + 0.005, f"{e:.3f}", ha="center", fontsize=10)
        ax.text(i + width / 2, r + 0.005, f"{r:.3f}", ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy")
    ax.set_title("BERT fine-tuné ERM vs IRM (3 genres)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "simple_comparison.png"), dpi=150)
    plt.close()
    print(f"  Plot sauvegardé dans {out_dir}/simple_comparison.png")

def plot_training_history(hist_erm: list, hist_irm: list, out_dir: str):
    """Courbes d'accuracy pendant le fine-tuning."""
    if not hist_erm or not hist_irm:
        return

    metrics = [
        ("val_matched",    "Val Matched (ID)"),
        ("val_mismatched", "Val Mismatched (OOD)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(12, 5))

    for ax, (key, title) in zip(axes, metrics):
        steps_erm = [h["step"] for h in hist_erm if key in h]
        vals_erm  = [h[key]   for h in hist_erm if key in h]
        steps_irm = [h["step"] for h in hist_irm if key in h]
        vals_irm  = [h[key]   for h in hist_irm if key in h]

        ax.plot(steps_erm, vals_erm, label="ERM FT", alpha=0.85, color="C0")
        ax.plot(steps_irm, vals_irm, label="IRM FT", alpha=0.85, color="C1")
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "simple_curves.png"), dpi=150)
    plt.close()
    print(f"  Courbes sauvegardées dans {out_dir}/simple_curves.png")

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT ERM vs IRM sur MNLI (3 genres)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--bert_model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256)

    # Fine-tuning
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size par env.")
    parser.add_argument("--lr_bert", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=1e-3)
    parser.add_argument("--n_unfrozen_layers", type=int, default=1,
                        help="Nombre de couches Transformer à dégeler.")

    # IRM
    parser.add_argument("--irm_lambda", type=float, default=10.0)
    parser.add_argument("--warmup_fraction", type=float, default=0.1)

    # Évaluation
    parser.add_argument("--eval_every", type=int, default=500)

    # Output
    parser.add_argument("--out_dir", type=str,
                        default=str(_Path(__file__).parent / "plots_simple"))

    args = parser.parse_args()
    device = torch.device(resolve_device(args.device))
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    n_classes = 3

    print("\n" + "=" * 70)
    print("ÉTAPE 1 : Chargement des datasets")
    print("=" * 70)
    mnli = load_all_datasets()

    print("\n" + "=" * 70)
    print("ÉTAPE 2 : Construction des DataLoaders")
    print("=" * 70)
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    print(f"\nEnvironnements d'entraînement (MNLI, {len(TRAIN_GENRES)} genres) :")
    env_loaders = build_env_loaders(mnli, tokenizer, args.max_length, args.batch_size)

    print("\nSets d'évaluation :")
    eval_loaders = build_eval_loaders(
        mnli, tokenizer, args.max_length, batch_size=args.batch_size * 4,
    )
    for name, loader in eval_loaders.items():
        if not name.startswith("val_") or name in ("val_matched", "val_mismatched"):
            print(f"  {name:20s} : {len(loader.dataset):,} paires")

    print("\n" + "=" * 70)
    print("ÉTAPE 3 : Fine-tuning ERM (baseline)")
    print("=" * 70)

    torch.manual_seed(args.seed)
    backbone_erm = AutoModel.from_pretrained(args.bert_model)
    model_erm    = BertNLIModel(backbone_erm, backbone_erm.config.hidden_size, n_classes).to(device)
    freeze_backbone_except_last_n(model_erm.backbone, args.n_unfrozen_layers)

    results_erm, hist_erm = finetune_erm(
        model_erm, env_loaders, eval_loaders,
        epochs=args.epochs,
        lr_bert=args.lr_bert,
        lr_head=args.lr_head,
        device=device,
        eval_every_steps=args.eval_every,
    )
    
    print("\n" + "=" * 70)
    print("ÉTAPE 4 : Fine-tuning IRM")
    print("=" * 70)

    torch.manual_seed(args.seed)
    backbone_irm = AutoModel.from_pretrained(args.bert_model)
    model_irm    = BertNLIModel(backbone_irm, backbone_irm.config.hidden_size, n_classes).to(device)
    freeze_backbone_except_last_n(model_irm.backbone, args.n_unfrozen_layers)

    results_irm, hist_irm = finetune_irm(
        model_irm, env_loaders, eval_loaders,
        epochs=args.epochs,
        lr_bert=args.lr_bert,
        lr_head=args.lr_head,
        irm_lambda=args.irm_lambda,
        warmup_fraction=args.warmup_fraction,
        device=device,
        eval_every_steps=args.eval_every,
    )

    print("\n" + "=" * 70)
    print("ÉTAPE 5 : Résultats")
    print("=" * 70)

    header_keys   = ["val_matched", "val_mismatched"]
    header_labels = ["Val ID", "Mismatch"]

    print(f"\n  {'':12s}  " + "  ".join(f"{h:>10s}" for h in header_labels))
    print(f"  {'':12s}  " + "  ".join("─" * 10 for _ in header_labels))
    for name, res in [("ERM FT", results_erm), ("IRM FT", results_irm)]:
        vals = [f"{res.get(k, 0):10.4f}" for k in header_keys]
        print(f"  {name:12s}  " + "  ".join(vals))

    print("\n  Détail par genre (val_matched) :")
    for g in TRAIN_GENRES:
        k = f"val_{g}"
        print(f"    {g:12s}  ERM={results_erm.get(k,0):.4f}  IRM={results_irm.get(k,0):.4f}")

    all_results = {"erm_ft": results_erm, "irm_ft": results_irm}
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Résultats sauvegardés dans {args.out_dir}/results.json")

    print("\n" + "=" * 70)
    print("ÉTAPE 6 : Plots")
    print("=" * 70)

    plot_comparison(results_erm, results_irm, args.out_dir)
    plot_training_history(hist_erm, hist_irm, args.out_dir)

    print("\n✓ Terminé.")


if __name__ == "__main__":
    main()
