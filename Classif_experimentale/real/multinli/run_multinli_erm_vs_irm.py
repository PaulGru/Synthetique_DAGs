#!/usr/bin/env python3
"""
run_multinli_erm_vs_irm.py
==========================
Compare ERM vs IRM (IRMv1) sur MultiNLI + SNLI avec le **genre/dataset**
comme environnement.

MultiNLI (Williams et al., 2018) est un dataset de Natural Language Inference
à 3 classes (entailment, neutral, contradiction) couvrant plusieurs genres
littéraires. SNLI (Bowman et al., 2015) couvre un seul genre (image captions).

Environnements d'entraînement (6) :
    MNLI : fiction, government, slate, telephone, travel
    SNLI : snli (captions d'images)

Évaluation :
    - validation_matched      (ID)  : mêmes genres que le train MNLI
    - validation_mismatched   (OOD) : genres différents du train MNLI
    - ANLI R1 / R2 / R3       (OOD) : Adversarial NLI

Pipeline :
    1. Chargement de MultiNLI + SNLI + ANLI depuis Hugging Face
    2. Embeddings BERT gelé (CLS pooling, paires correctement tokenisées)
    3. Construction de 6 envs train (5 genres MNLI + 1 SNLI)
    4. Val ID  = validation_matched (par genre → 5 Envs)
    5. Test OOD = val_mismatched + ANLI R1/R2/R3 test
    6. Entraînement ERM et IRM
    7. Évaluation + plots

Usage :
    uv run real/multinli/run_multinli_erm_vs_irm.py
    uv run real/multinli/run_multinli_erm_vs_irm.py --device auto --irm_lambda 500
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[2]
for _p in [str(_ROOT), str(_ROOT / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import hashlib
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from data_synth import Env
from models_training import train_erm, train_irm, compute_accuracy
from utils_irm import resolve_device, evaluate_env

# =============================================================================
# 0. Cache d'embeddings
# =============================================================================

_DEFAULT_CACHE_DIR = str(_Path(__file__).parent / ".embed_cache")


def _cache_key(dataset: str, model_name: str, max_length: int, pooling: str) -> str:
    """Clé déterministe pour un jeu (dataset, modèle, params)."""
    tag = f"{dataset}_{model_name}_{max_length}_{pooling}"
    h = hashlib.md5(tag.encode()).hexdigest()[:10]
    safe = tag.replace("/", "_")
    return f"{safe}_{h}"


def _load_cache(cache_dir: str, key: str):
    """Charge un .npz depuis le cache. Retourne None si absent."""
    path = os.path.join(cache_dir, f"{key}.npz")
    if os.path.isfile(path):
        print(f"  ✓ Cache trouvé : {path}")
        return dict(np.load(path, allow_pickle=True))
    return None


def _save_cache(cache_dir: str, key: str, **arrays):
    """Sauvegarde un ensemble d'arrays numpy dans un .npz."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.npz")
    np.savez(path, **arrays)
    size_mb = os.path.getsize(path) / 1e6
    print(f"  ✓ Cache sauvegardé : {path} ({size_mb:.1f} MB)")


# =============================================================================
# 1. Chargement du dataset MultiNLI
# =============================================================================

TRAIN_GENRES = ["fiction", "government", "slate", "telephone", "travel"]


def load_multinli() -> dict:
    """Charge les 3 splits de MultiNLI depuis Hugging Face."""
    from datasets import load_dataset

    print("Chargement de MultiNLI depuis Hugging Face …")
    ds = load_dataset("nyu-mll/multi_nli")
    print(f"  train              : {len(ds['train']):,} exemples")
    print(f"  validation_matched : {len(ds['validation_matched']):,} exemples")
    print(f"  validation_mismatched: {len(ds['validation_mismatched']):,} exemples")
    return ds


def load_snli() -> dict:
    """Charge SNLI depuis Hugging Face."""
    from datasets import load_dataset

    print("Chargement de SNLI depuis Hugging Face …")
    ds = load_dataset("stanfordnlp/snli")
    print(f"  train : {len(ds['train']):,} exemples")
    print(f"  test  : {len(ds['test']):,} exemples")
    return ds


# =============================================================================
# 2. Embeddings BERT gelé
# =============================================================================

def embed_texts(
    premises: List[str],
    hypotheses: List[str],
    model_name: str = "bert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    batch_size: int = 64,
    pooling: str = "cls",
    loaded_model=None,
    loaded_tokenizer=None,
) -> np.ndarray:
    """
    Embeddings BERT gelé pour des paires (premise, hypothesis).

    Le tokenizer reçoit les deux phrases séparément pour générer les
    token_type_ids corrects (segment A / segment B), ce qu'un simple
    f"{p} [SEP] {h}" ne permet pas.

    pooling='cls' est recommandé pour les tâches sentence-pair.

    Si loaded_model et loaded_tokenizer sont fournis (ex. backbone fine-tuné),
    ils sont utilisés directement — aucune requête HuggingFace.

    Returns : np.ndarray (N, hidden_dim).
    """
    from transformers import AutoTokenizer, AutoModel

    assert len(premises) == len(hypotheses)
    if loaded_tokenizer is not None and loaded_model is not None:
        tokenizer = loaded_tokenizer
        model     = loaded_model
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model     = AutoModel.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model = model.to(device)
    use_autocast = "cuda" in str(device)

    all_emb = []
    n = len(premises)
    for i in range(0, n, batch_size):
        batch_p = premises[i : i + batch_size]
        batch_h = hypotheses[i : i + batch_size]
        # Passer premise et hypothesis comme text / text_pair : le tokenizer
        # insère automatiquement [SEP] et remplit token_type_ids (0 = segment A,
        # 1 = segment B), indispensable pour que BERT traite la paire correctement.
        enc = tokenizer(
            batch_p,
            batch_h,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        token_type_ids = enc.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        fwd_kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            fwd_kwargs["token_type_ids"] = token_type_ids
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_autocast):
                out = model(**fwd_kwargs)

        hidden = out.last_hidden_state  # (B, seq, D)
        if pooling == "cls":
            emb = hidden[:, 0, :]
        else:  # mean
            mask = attention_mask.unsqueeze(-1).float()
            emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

        all_emb.append(emb.cpu().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"  Embedded {i + len(batch_p):,}/{n:,}")

    return np.concatenate(all_emb, axis=0).astype(np.float32)


# =============================================================================
# 2b. Fine-tuning du backbone BERT/DistilBERT (optionnel)
# =============================================================================

class _NLIDataset(torch.utils.data.Dataset):
    """Dataset NLI qui tokenise les paires (premise, hypothesis) à la volée."""

    def __init__(self, premises, hypotheses, labels, tokenizer, max_length):
        self.premises   = premises
        self.hypotheses = hypotheses
        self.labels     = labels
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.premises[idx],
            self.hypotheses[idx],
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        item = {
            "input_ids":      enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "label": self.labels[idx],
        }
        if "token_type_ids" in enc:
            item["token_type_ids"] = enc["token_type_ids"]
        return item


def finetune_bert(
    premises: List[str],
    hypotheses: List[str],
    labels: List[int],
    model_name: str = "distilbert-base-uncased",
    n_classes: int = 3,
    epochs: int = 2,
    lr: float = 2e-5,
    batch_size: int = 32,
    max_length: int = 256,
    device: str = "cpu",
    seed: int = 42,
) -> Tuple[nn.Module, object]:
    """
    Fine-tune un backbone BERT/DistilBERT en 2 étapes :
      1. Entraînement standard (ERM) sur les paires NLI d'entraînement avec
         une tête de classification linéaire.
      2. Suppression de la tête → seul le backbone (gelé) est retourné.

    Le backbone fine-tuné est ensuite utilisé par embed_texts pour
    extraire des représentations CLS enrichies sur lesquelles ERM et IRM
    entraîneront une régression logistique.

    Returns
    -------
    backbone  : nn.Module          (mode eval, requires_grad=False)
    tokenizer : PreTrainedTokenizer
    """
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

    torch.manual_seed(seed)
    device_t = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    backbone  = AutoModel.from_pretrained(model_name)

    class _BertNLI(nn.Module):
        def __init__(self, backbone, hidden_size, n_classes):
            super().__init__()
            self.backbone   = backbone
            self.dropout    = nn.Dropout(0.1)
            self.classifier = nn.Linear(hidden_size, n_classes)

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            kw = dict(input_ids=input_ids, attention_mask=attention_mask)
            if token_type_ids is not None:
                kw["token_type_ids"] = token_type_ids
            out = self.backbone(**kw)
            return self.classifier(self.dropout(out.last_hidden_state[:, 0, :]))

    model = _BertNLI(backbone, backbone.config.hidden_size, n_classes).to(device_t)

    def _collate(batch):
        labels_b = [item.pop("label") for item in batch]
        padded   = tokenizer.pad(batch, return_tensors="pt")
        padded["labels"] = torch.tensor(labels_b, dtype=torch.long)
        return padded

    dataset = _NLIDataset(premises, hypotheses, labels, tokenizer, max_length)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=_collate,
        pin_memory=("cuda" in str(device)),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps,
    )
    loss_fn      = nn.CrossEntropyLoss()
    use_autocast = "cuda" in str(device)

    print(f"  Modèle   : {model_name}")
    print(f"  Données  : {len(dataset):,} paires  |  batch={batch_size}")
    print(f"  Epochs   : {epochs}  |  lr={lr}  |  steps/epoch={len(loader):,}")

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for step, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(device_t)
            attention_mask = batch["attention_mask"].to(device_t)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device_t)
            labels_b = batch["labels"].to(device_t)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_autocast):
                logits = model(input_ids, attention_mask, token_type_ids)
                loss   = loss_fn(logits, labels_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            correct    += (logits.detach().argmax(-1) == labels_b).sum().item()
            total      += len(labels_b)

            if (step + 1) % 500 == 0:
                print(f"    Epoch {epoch+1}/{epochs}  step {step+1:,}/{len(loader):,}"
                      f"  loss={total_loss/(step+1):.4f}  acc={correct/total:.4f}")

        print(f"  → Epoch {epoch+1}/{epochs} terminée"
              f"  loss={total_loss/len(loader):.4f}  acc={correct/len(dataset):.4f}")

    # Retourner uniquement le backbone gelé (tête de classification supprimée)
    ft_backbone = model.backbone
    ft_backbone.eval()
    for p in ft_backbone.parameters():
        p.requires_grad = False

    return ft_backbone, tokenizer


# =============================================================================
# 3. Construction des environnements
# =============================================================================

def build_envs_multinli(
    ds: dict,
    bert_model: str = "bert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    batch_size: int = 64,
    pooling: str = "cls",
    seed: int = 42,
    loaded_model=None,
    loaded_tokenizer=None,
    cache_dir: str | None = None,
) -> Tuple[List[Env], List[Env], Env]:
    """
    Construit les environnements d'entraînement, validation ID et test OOD
    (val_mismatched) pour MultiNLI.

    Returns
    -------
    train_envs     : List[Env]  — 5 Envs (1 par genre d'entraînement)
    val_envs       : List[Env]  — 5 Envs (validation_matched, 1 par genre) = ID
    mismatched_env : Env        — validation_mismatched (OOD, tous genres confondus)
    """
    # --- Extraction des textes / labels / métadonnées ---
    all_premises = []
    all_hypotheses = []
    all_labels = []
    all_splits = []
    all_genres = []

    for split_name, hf_key in [
        ("train", "train"),
        ("val_matched", "validation_matched"),
        ("val_mismatched", "validation_mismatched"),
    ]:
        split_ds = ds[hf_key]
        labels_col   = split_ds["label"]
        premises_col = split_ds["premise"]
        hyp_col      = split_ds["hypothesis"]
        genres_col   = split_ds["genre"]
        for idx, label in enumerate(labels_col):
            if label == -1:
                continue
            all_premises.append(premises_col[idx])
            all_hypotheses.append(hyp_col[idx])
            all_labels.append(label)
            all_splits.append(split_name)
            all_genres.append(genres_col[idx])

    all_labels = np.array(all_labels, dtype=np.int64)
    all_splits = np.array(all_splits)
    all_genres = np.array(all_genres)

    # --- Embeddings (avec cache si backbone gelé) ---
    use_cache = cache_dir is not None and loaded_model is None
    cache_hit = False

    if use_cache:
        key = _cache_key("mnli_full", bert_model, max_length, pooling)
        cached = _load_cache(cache_dir, key)
        if cached is not None:
            all_embeddings = cached["embeddings"]
            cache_hit = True

    if not cache_hit:
        print(f"\nEmbedding de {len(all_premises):,} paires MNLI "
              f"(train + val_matched + val_mismatched) …")
        print(f"  pooling={pooling}, max_length={max_length}, batch_size={batch_size}")
        all_embeddings = embed_texts(
            all_premises,
            all_hypotheses,
            model_name=bert_model,
            max_length=max_length,
            device=device,
            batch_size=batch_size,
            pooling=pooling,
            loaded_model=loaded_model,
            loaded_tokenizer=loaded_tokenizer,
        )
        if use_cache:
            _save_cache(cache_dir, key, embeddings=all_embeddings)

    print("\nConstruction des environnements d'entraînement :")
    train_envs = []
    for genre in TRAIN_GENRES:
        mask = (all_splits == "train") & (all_genres == genre)
        X = torch.from_numpy(all_embeddings[mask]).float()
        y = torch.from_numpy(all_labels[mask]).long()
        env = Env(X=X, y=y, meta={"genre": genre, "split": "train"})
        train_envs.append(env)
        class_counts = np.bincount(all_labels[mask], minlength=3)
        print(f"  {genre:12s} : {mask.sum():,} exemples  "
              f"(E={class_counts[0]}, N={class_counts[1]}, C={class_counts[2]})")

    print("\nConstruction des environnements val_matched (ID) :")
    val_envs = []
    for genre in TRAIN_GENRES:
        mask = (all_splits == "val_matched") & (all_genres == genre)
        X = torch.from_numpy(all_embeddings[mask]).float()
        y = torch.from_numpy(all_labels[mask]).long()
        env = Env(X=X, y=y, meta={"genre": genre, "split": "val_matched"})
        val_envs.append(env)
        class_counts = np.bincount(all_labels[mask], minlength=3)
        print(f"  {genre:12s} : {mask.sum():,} exemples  "
              f"(E={class_counts[0]}, N={class_counts[1]}, C={class_counts[2]})")

    # --- Val mismatched (OOD) ---
    print("\nConstruction de l'env val_mismatched (OOD) :")
    mask_mm = all_splits == "val_mismatched"
    X_mm = torch.from_numpy(all_embeddings[mask_mm]).float()
    y_mm = torch.from_numpy(all_labels[mask_mm]).long()
    mismatched_env = Env(X=X_mm, y=y_mm, meta={"split": "val_mismatched"})
    cc = np.bincount(all_labels[mask_mm], minlength=3)
    print(f"  val_mismatched : {mask_mm.sum():,} exemples  "
          f"(E={cc[0]}, N={cc[1]}, C={cc[2]})")

    return train_envs, val_envs, mismatched_env


# =============================================================================
# 3b. Chargement et construction de l'Env SNLI (train)
# =============================================================================

def build_snli_env(
    snli_ds: dict,
    bert_model: str = "bert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    batch_size: int = 64,
    pooling: str = "cls",
    loaded_model=None,
    loaded_tokenizer=None,
    cache_dir: str | None = None,
) -> Env:
    """
    Construit un unique Env d'entraînement à partir du split train de SNLI.

    Returns
    -------
    Env  — environnement "snli" (train)
    """
    split = snli_ds["train"]
    labels_col   = split["label"]
    premises_col = split["premise"]
    hyp_col      = split["hypothesis"]

    premises, hypotheses, labels = [], [], []
    for idx, label in enumerate(labels_col):
        if label == -1:
            continue
        premises.append(premises_col[idx])
        hypotheses.append(hyp_col[idx])
        labels.append(label)

    labels_np = np.array(labels, dtype=np.int64)

    # --- Embeddings (avec cache si backbone gelé) ---
    use_cache = cache_dir is not None and loaded_model is None
    cache_hit = False

    if use_cache:
        key = _cache_key("snli_train", bert_model, max_length, pooling)
        cached = _load_cache(cache_dir, key)
        if cached is not None:
            embeddings = cached["embeddings"]
            cache_hit = True

    if not cache_hit:
        print(f"\nEmbedding de {len(premises):,} paires SNLI (train) …")
        embeddings = embed_texts(
            premises,
            hypotheses,
            model_name=bert_model,
            max_length=max_length,
            device=device,
            batch_size=batch_size,
            pooling=pooling,
            loaded_model=loaded_model,
            loaded_tokenizer=loaded_tokenizer,
        )
        if use_cache:
            _save_cache(cache_dir, key, embeddings=embeddings)

    X = torch.from_numpy(embeddings).float()
    y = torch.from_numpy(labels_np).long()
    cc = np.bincount(labels_np, minlength=3)
    print(f"  snli (train) : {len(labels_np):,} exemples  "
          f"(E={cc[0]}, N={cc[1]}, C={cc[2]})")
    return Env(X=X, y=y, meta={"genre": "snli", "split": "train"})


# =============================================================================
# 3c. Chargement et construction des Envs ANLI (OOD)
# =============================================================================

ANLI_ROUNDS = ["r1", "r2", "r3"]


def load_anli() -> dict:
    """Charge ANLI depuis Hugging Face."""
    from datasets import load_dataset

    print("Chargement de ANLI depuis Hugging Face …")
    ds = load_dataset("facebook/anli")
    for r in ANLI_ROUNDS:
        test_key = f"test_{r}"
        print(f"  {test_key:12s} : {len(ds[test_key]):,} exemples")
    return ds


def build_anli_envs(
    anli_ds: dict,
    bert_model: str = "bert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    batch_size: int = 64,
    pooling: str = "cls",
    loaded_model=None,
    loaded_tokenizer=None,
    cache_dir: str | None = None,
) -> Dict[str, Env]:
    """
    Construit un Env par round ANLI (test_r1, test_r2, test_r3) + un combiné.

    Returns
    -------
    dict  {"r1": Env, "r2": Env, "r3": Env, "all": Env}
    """
    all_premises = []
    all_hypotheses = []
    all_labels = []
    all_rounds = []

    for r in ANLI_ROUNDS:
        split = anli_ds[f"test_{r}"]
        premises_col = split["premise"]
        hyp_col      = split["hypothesis"]
        labels_col   = split["label"]
        for idx, label in enumerate(labels_col):
            if label == -1:
                continue
            all_premises.append(premises_col[idx])
            all_hypotheses.append(hyp_col[idx])
            all_labels.append(label)
            all_rounds.append(r)

    all_labels = np.array(all_labels, dtype=np.int64)
    all_rounds = np.array(all_rounds)

    # --- Embeddings (avec cache si backbone gelé) ---
    use_cache = cache_dir is not None and loaded_model is None
    cache_hit = False

    if use_cache:
        key = _cache_key("anli", bert_model, max_length, pooling)
        cached = _load_cache(cache_dir, key)
        if cached is not None:
            all_embeddings = cached["embeddings"]
            cache_hit = True

    if not cache_hit:
        print(f"\nEmbedding de {len(all_premises):,} paires ANLI (test R1+R2+R3) …")
        all_embeddings = embed_texts(
            all_premises,
            all_hypotheses,
            model_name=bert_model,
            max_length=max_length,
            device=device,
            batch_size=batch_size,
            pooling=pooling,
            loaded_model=loaded_model,
            loaded_tokenizer=loaded_tokenizer,
        )
        if use_cache:
            _save_cache(cache_dir, key, embeddings=all_embeddings)

    envs = {}
    print("\nConstruction des Envs ANLI :")
    for r in ANLI_ROUNDS:
        mask = all_rounds == r
        X = torch.from_numpy(all_embeddings[mask]).float()
        y = torch.from_numpy(all_labels[mask]).long()
        envs[r] = Env(X=X, y=y, meta={"round": r, "split": "test"})
        cc = np.bincount(all_labels[mask], minlength=3)
        print(f"  {r} : {mask.sum():,}  (E={cc[0]}, N={cc[1]}, C={cc[2]})")

    # Combiné
    X_all = torch.from_numpy(all_embeddings).float()
    y_all = torch.from_numpy(all_labels).long()
    envs["all"] = Env(X=X_all, y=y_all, meta={"round": "all", "split": "test"})
    cc = np.bincount(all_labels, minlength=3)
    print(f"  all: {len(all_labels):,}  (E={cc[0]}, N={cc[1]}, C={cc[2]})")

    return envs


# =============================================================================
# 4. Visualisation
# =============================================================================

def _ema_smooth(values: list, alpha: float = 0.05) -> list:
    if not values:
        return values
    smoothed = [values[0]]
    for v in values[1:]:
        smoothed.append(alpha * v + (1 - alpha) * smoothed[-1])
    return smoothed


def plot_training_curves(
    hist_erm: dict,
    hist_irm: dict,
    out_dir: str,
    smooth_alpha: float = 0.05,
):
    """Courbes d'entraînement ERM vs IRM (loss, train acc, val ID acc, test OOD acc)."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # --- Loss ---
    ax = axes[0]
    for hist, color, name in [(hist_erm, "C0", "ERM"), (hist_irm, "C1", "IRM")]:
        steps = hist["step"]
        raw = hist["loss"]
        smooth = _ema_smooth(raw, alpha=smooth_alpha)
        ax.plot(steps, raw, color=color, alpha=0.15, linewidth=0.8)
        ax.plot(steps, smooth, color=color, alpha=0.9, linewidth=1.8, label=name)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (EMA)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Train Acc ---
    ax = axes[1]
    ax.plot(hist_erm["step"], hist_erm["train_acc"], label="ERM train", alpha=0.8)
    ax.plot(hist_irm["step"], hist_irm["train_acc"], label="IRM train", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Train Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Val Acc (ID = matched) ---
    ax = axes[2]
    ax.plot(hist_erm["step"], hist_erm["val_acc"], label="ERM val (matched)", alpha=0.8)
    ax.plot(hist_irm["step"], hist_irm["val_acc"], label="IRM val (matched)", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation Matched (ID)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Test Acc (OOD = ANLI) ---
    ax = axes[3]
    ax.plot(hist_erm["step"], hist_erm["test_acc"], label="ERM test (ANLI)", alpha=0.8)
    ax.plot(hist_irm["step"], hist_irm["test_acc"], label="IRM test (ANLI)", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Accuracy")
    ax.set_title("Test OOD (ANLI)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  Courbes sauvegardées dans {out_dir}/training_curves.png")


def plot_accuracy_comparison(results: dict, out_dir: str):
    """Bar chart : ERM vs IRM sur val_matched (ID) + val_mismatched + ANLI (OOD)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    sets = ["val_matched", "val_mismatched", "anli_r1", "anli_r2", "anli_r3", "anli_all"]
    labels = ["Val Matched\n(ID)", "Val Mismatched\n(OOD)", "ANLI R1", "ANLI R2", "ANLI R3", "ANLI All"]
    x = np.arange(len(sets))
    width = 0.3

    erm_accs = [results["erm"][s] for s in sets]
    irm_accs = [results["irm"][s] for s in sets]

    ax.bar(x - width / 2, erm_accs, width, label="ERM", color="#e74c3c", edgecolor="black")
    ax.bar(x + width / 2, irm_accs, width, label="IRM", color="#2ecc71", edgecolor="black")

    for i, (e, r) in enumerate(zip(erm_accs, irm_accs)):
        ax.text(i - width / 2, e + 0.005, f"{e:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, r + 0.005, f"{r:.3f}", ha="center", fontsize=9)

    # Séparateur ID / OOD
    ax.axvline(x=0.5, color="gray", ls="--", alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy")
    ax.set_title("ERM vs IRM — MNLI+SNLI train / OOD (genre = environnement)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print(f"  Accuracy plot sauvegardé dans {out_dir}/accuracy_comparison.png")


def plot_per_genre_accuracy(
    model_erm: nn.Module,
    model_irm: nn.Module,
    val_envs: List[Env],
    mismatched_env: Env,
    anli_envs: Dict[str, Env],
    device: str,
    out_dir: str,
):
    """Bar chart : accuracy par genre (val matched ID) + val_mismatched + ANLI R1/R2/R3 (OOD)."""
    genres = [e.meta["genre"] for e in val_envs]

    erm_accs = [evaluate_env(model_erm, e, device=device) for e in val_envs]
    irm_accs = [evaluate_env(model_irm, e, device=device) for e in val_envs]

    # Val mismatched
    erm_accs.append(evaluate_env(model_erm, mismatched_env, device=device))
    irm_accs.append(evaluate_env(model_irm, mismatched_env, device=device))

    # ANLI rounds
    for r in ANLI_ROUNDS:
        erm_accs.append(evaluate_env(model_erm, anli_envs[r], device=device))
        irm_accs.append(evaluate_env(model_irm, anli_envs[r], device=device))

    all_labels = genres + ["Mismatched", "ANLI R1", "ANLI R2", "ANLI R3"]

    x = np.arange(len(all_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))
    bars_erm = ax.bar(x - width / 2, erm_accs, width, label="ERM", color="#e74c3c",
                      edgecolor="black", alpha=0.85)
    bars_irm = ax.bar(x + width / 2, irm_accs, width, label="IRM", color="#2ecc71",
                       edgecolor="black", alpha=0.85)

    for bar, v in zip(bars_erm, erm_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=8)
    for bar, v in zip(bars_irm, irm_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005, f"{v:.3f}",
                ha="center", fontsize=8)

    # Séparateur ID / OOD
    ax.axvline(x=len(genres) - 0.5, color="gray", ls="--", alpha=0.5)
    ax.text(len(genres) - 0.75, 0.98, "ID ←", ha="right", fontsize=9, color="gray",
            transform=ax.get_xaxis_transform())
    ax.text(len(genres) - 0.25, 0.98, "→ OOD", ha="left", fontsize=9, color="gray",
            transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=15, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy par genre / round — ERM vs IRM (MNLI+SNLI → OOD)")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "per_genre_accuracy.png"), dpi=150)
    plt.close()
    print(f"  Per-genre plot sauvegardé dans {out_dir}/per_genre_accuracy.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MultiNLI — ERM vs IRM (genre = environnement)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Max tokens par paire premise+hypothesis. 256 recommandé pour MultiNLI.")
    parser.add_argument("--embed_batch", type=int, default=64,
                        help="Batch size pour l'embedding BERT. Augmenter (128/256) accélère l'étape.")
    parser.add_argument("--pooling", type=str, default="cls", choices=["mean", "cls"],
                        help="cls (recommandé pour sentence-pair) ou mean.")

    # Training
    parser.add_argument("--erm_steps", type=int, default=25_000)
    parser.add_argument("--erm_lr", type=float, default=1e-3)
    parser.add_argument("--irm_steps", type=int, default=25_000)
    parser.add_argument("--irm_lr", type=float, default=1e-3)
    parser.add_argument("--irm_lambda", type=float, default=500.0)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--use_mlp", action="store_true", default=True,
                        help="Utiliser SmallMLP (défaut) au lieu de LogisticReg. "
                             "Recommandé pour NLI : la frontière de décision n'est "
                             "pas linéaire dans l'espace BERT gelé.")
    parser.add_argument("--no_mlp", dest="use_mlp", action="store_false",
                        help="Forcer LogisticReg (désactive --use_mlp).")
    parser.add_argument("--mlp_hidden", type=int, default=512,
                        help="Taille de la première couche cachée du SmallMLP.")

    # Fine-tuning du backbone (optionnel)
    parser.add_argument("--finetune_epochs", type=int, default=0,
                        help="Epochs de fine-tuning du backbone avant embedding. "
                             "0 = désactivé (BERT gelé). 2-3 recommandé pour MultiNLI.")
    parser.add_argument("--finetune_lr", type=float, default=2e-5,
                        help="Learning rate AdamW pour le fine-tuning du backbone.")
    parser.add_argument("--finetune_batch", type=int, default=32,
                        help="Batch size pour le fine-tuning du backbone.")

    # Output
    parser.add_argument("--out_dir", type=str,
                        default=str(_Path(__file__).parent / "plots"))
    parser.add_argument("--cache_dir", type=str,
                        default=_DEFAULT_CACHE_DIR,
                        help="Répertoire de cache pour les embeddings gelés. "
                             "Utiliser --cache_dir '' pour désactiver.")

    args = parser.parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)
    cache_dir = args.cache_dir if args.cache_dir else None

    n_classes = 3  # entailment, neutral, contradiction

    # ─────────────────────────────────────────────────────────────────────
    # Étape 1 : Chargement des datasets
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 1 : Chargement de MultiNLI + SNLI + ANLI")
    print("=" * 70)
    ds = load_multinli()
    snli_ds = load_snli()
    anli_ds = load_anli()

    # ─────────────────────────────────────────────────────────────────────
    # Étape 2a : Fine-tuning du backbone (optionnel)
    # ─────────────────────────────────────────────────────────────────────
    ft_model, ft_tokenizer = None, None
    if args.finetune_epochs > 0:
        print("\n" + "=" * 70)
        print(f"ÉTAPE 2a : Fine-tuning {args.bert_model} ({args.finetune_epochs} epoch(s))")
        print("=" * 70)
        train_split     = ds["train"]
        ft_labels_raw   = train_split["label"]
        ft_premises_raw = train_split["premise"]
        ft_hyps_raw     = train_split["hypothesis"]
        ft_premises = [ft_premises_raw[i] for i, l in enumerate(ft_labels_raw) if l != -1]
        ft_hyps     = [ft_hyps_raw[i]     for i, l in enumerate(ft_labels_raw) if l != -1]
        ft_labels   = [l for l in ft_labels_raw if l != -1]
        ft_model, ft_tokenizer = finetune_bert(
            premises=ft_premises,
            hypotheses=ft_hyps,
            labels=ft_labels,
            model_name=args.bert_model,
            n_classes=n_classes,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            batch_size=args.finetune_batch,
            max_length=args.max_length,
            device=device,
            seed=args.seed,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 2b : Embeddings MNLI + construction des environnements
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    _ft_label = f"fine-tuné {args.finetune_epochs} epoch(s)" if args.finetune_epochs > 0 else "gelé"
    print(f"ÉTAPE 2b : Embeddings [{_ft_label}] MNLI + SNLI + ANLI")
    print("=" * 70)
    train_envs, val_envs, mismatched_env = build_envs_multinli(
        ds,
        bert_model=args.bert_model,
        max_length=args.max_length,
        device=device,
        batch_size=args.embed_batch,
        pooling=args.pooling,
        seed=args.seed,
        loaded_model=ft_model,
        loaded_tokenizer=ft_tokenizer,
        cache_dir=cache_dir,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 2c : Embeddings SNLI (6e env d'entraînement)
    # ─────────────────────────────────────────────────────────────────────
    snli_env = build_snli_env(
        snli_ds,
        bert_model=args.bert_model,
        max_length=args.max_length,
        device=device,
        batch_size=args.embed_batch,
        pooling=args.pooling,
        loaded_model=ft_model,
        loaded_tokenizer=ft_tokenizer,
        cache_dir=cache_dir,
    )
    train_envs.append(snli_env)

    # ─────────────────────────────────────────────────────────────────────
    # Étape 2d : Embeddings ANLI (OOD)
    # ─────────────────────────────────────────────────────────────────────
    anli_envs = build_anli_envs(
        anli_ds,
        bert_model=args.bert_model,
        max_length=args.max_length,
        device=device,
        batch_size=args.embed_batch,
        pooling=args.pooling,
        loaded_model=ft_model,
        loaded_tokenizer=ft_tokenizer,
        cache_dir=cache_dir,
    )
    # test_env pour les courbes d'entraînement = ANLI combiné
    test_env = anli_envs["all"]

    # ─────────────────────────────────────────────────────────────────────
    # Étape 3 : Entraînement ERM
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 3 : Entraînement ERM")
    print("=" * 70)
    erm_model, erm_hist = train_erm(
        envs=train_envs,
        val_envs=val_envs,
        test_env=test_env,
        steps=args.erm_steps,
        lr=args.erm_lr,
        batch=args.batch,
        seed=args.seed,
        device=device,
        eval_every=args.eval_every,
        dataset_name="multinli",
        n_classes=n_classes,
        use_mlp=args.use_mlp,
        mlp_hidden=args.mlp_hidden,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 4 : Entraînement IRM
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 4 : Entraînement IRM")
    print("=" * 70)
    irm_model, irm_hist = train_irm(
        envs=train_envs,
        val_envs=val_envs,
        test_env=test_env,
        steps=args.irm_steps,
        lr=args.irm_lr,
        batch=args.batch,
        irm_lambda=args.irm_lambda,
        seed=args.seed,
        device=device,
        eval_every=args.eval_every,
        dataset_name="multinli",
        n_classes=n_classes,
        use_mlp=args.use_mlp,
        mlp_hidden=args.mlp_hidden,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 5 : Évaluation finale
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 5 : Évaluation finale")
    print("=" * 70)

    erm_val_matched = compute_accuracy(erm_model, val_envs, device=device)
    irm_val_matched = compute_accuracy(irm_model, val_envs, device=device)

    results = {
        "erm": {"val_matched": erm_val_matched},
        "irm": {"val_matched": irm_val_matched},
    }

    # Val mismatched (OOD)
    results["erm"]["val_mismatched"] = evaluate_env(erm_model, mismatched_env, device=device)
    results["irm"]["val_mismatched"] = evaluate_env(irm_model, mismatched_env, device=device)

    # ANLI par round
    for r in ANLI_ROUNDS:
        erm_r = evaluate_env(erm_model, anli_envs[r], device=device)
        irm_r = evaluate_env(irm_model, anli_envs[r], device=device)
        results["erm"][f"anli_{r}"] = erm_r
        results["irm"][f"anli_{r}"] = irm_r
    results["erm"]["anli_all"] = evaluate_env(erm_model, anli_envs["all"], device=device)
    results["irm"]["anli_all"] = evaluate_env(irm_model, anli_envs["all"], device=device)

    print(f"\n  {'':12s}  {'Val ID':>8s}  {'Mismatch':>8s}  {'ANLI R1':>8s}  {'ANLI R2':>8s}  {'ANLI R3':>8s}  {'ANLI All':>8s}")
    print(f"  {'':12s}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")
    for method in ["erm", "irm"]:
        r = results[method]
        print(f"  {method.upper():12s}  {r['val_matched']:8.4f}  {r['val_mismatched']:8.4f}  {r['anli_r1']:8.4f}  "
              f"{r['anli_r2']:8.4f}  {r['anli_r3']:8.4f}  {r['anli_all']:8.4f}")

    # Per-genre details (ID)
    print("\n  Détail par genre (val_matched) :")
    for e in val_envs:
        g = e.meta["genre"]
        acc_erm = evaluate_env(erm_model, e, device=device)
        acc_irm = evaluate_env(irm_model, e, device=device)
        print(f"    {g:12s}  ERM={acc_erm:.4f}  IRM={acc_irm:.4f}")

    # Save results
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Résultats sauvegardés dans {args.out_dir}/results.json")

    # ─────────────────────────────────────────────────────────────────────
    # Étape 6 : Plots
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 6 : Visualisation")
    print("=" * 70)

    plot_training_curves(erm_hist, irm_hist, args.out_dir)
    plot_accuracy_comparison(results, args.out_dir)
    plot_per_genre_accuracy(
        erm_model, irm_model, val_envs, mismatched_env, anli_envs,
        device=device, out_dir=args.out_dir,
    )

    print("\n✓ Terminé.")


if __name__ == "__main__":
    main()
