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
eval_moji_irm_ft.py
=======================
Modèle 5 (IRM + Fine-tuned backbone).

Utilise le backbone fine-tuné produit par le Modèle 3
(train_moji_finetuned_erm.py). Ce backbone est gelé dès le départ ;
seule la tête de classification est (ré)entraînée.

  Modèle 5 — IRM + Fine-tuned backbone
    - Backbone fine-tuné (Modèle 3) gelé
    - Entraîne une NOUVELLE tête LogReg avec la PÉNALITÉ IRM sur E1+E2

Pipeline :
    1. Charger le backbone Modèle 3 (best_backbone.pt), geler
    2. Calculer les embeddings FT pour E1, E2, val_ind, test_ood
    3. Modèle 5 : train_irm sur E1 + E2 (2 envs biaisés)
    4. Évaluation + comparaison avec Modèle 3 si results.json dispo

Usage :
    uv run eval_moji_irm_ft.py --device auto \\
        --backbone_dir logs_finetuned_erm

    uv run eval_moji_irm_ft.py --device auto \\
        --backbone_dir logs_finetuned_erm \\
        --irm_lambda 150 --out_dir logs_irm_ft
"""


import argparse
import json
import os
from typing import List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# ── Utilitaires du Modèle 3 ──────────────────────────────────────────────────
from train_moji_finetuned_erm import (
    FineTunedDistilBERT,
    MojiTextDataset,
    make_collate_fn,
)

# ── Utilitaires du script principal (Modèles 1/2) ────────────────────────────
from run_moji_erm_vs_irm import (
    load_moji,
    build_train_envs,
    build_val_ind,
    build_test_ood,
    make_env,
    full_evaluation,
    _get_group_indices,
    _C,
    _annotate_bar,
    print_split_stats,
)

# ── Entraînement IRM (mêmes que Modèles 1/2, sur les nouvelles embed.) ──
from models_training import train_irm
from utils_irm import resolve_device


# =============================================================================
# 1. Embedding avec le backbone fine-tuné (gelé)
# =============================================================================

def embed_with_finetuned_backbone(
    texts: List[str],
    model: FineTunedDistilBERT,
    tokenizer,
    max_length: int,
    device: str,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Calcule les embeddings via le backbone fine-tuné (mean pool, sans tête).
    Le modèle doit être en eval() et ses paramètres gelés avant l'appel.

    Returns : np.ndarray (N, 768), float32.
    """
    model.eval()
    use_fp16 = str(device).startswith("cuda")
    all_emb = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            if use_fp16:
                with torch.autocast(device_type="cuda"):
                    emb = model.encode(input_ids, attention_mask)
            else:
                emb = model.encode(input_ids, attention_mask)

        all_emb.append(emb.cpu().float().numpy())
        if (i // batch_size) % 50 == 0:
            print(f"    FT-embed {i + len(batch):,}/{len(texts):,}")

    return np.concatenate(all_emb, axis=0).astype(np.float32)


# =============================================================================
# 2. MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Modèle 5 (IRM+FT) — backbone Modèle 3 gelé"
    )
    parser.add_argument("--seed",    type=int, default=42)
    parser.add_argument("--device",  type=str, default="auto")

    # ── Répertoire du Modèle 3 ──
    parser.add_argument(
        "--backbone_dir", type=str,
        default=str(_Path(__file__).parent / "logs" / "finetuned_erm"),
        help="Répertoire contenant best_backbone.pt "
             "(produit par train_moji_finetuned_erm.py)",
    )

    # ── Dataset ──
    parser.add_argument("--max_per_group",    type=int, default=5000)
    parser.add_argument("--n_per_group_test", type=int, default=7500)
    parser.add_argument(
        "--sae_ratio", type=float, default=2.0,
        help="Ratio N_SAE / N_AAE dans chaque environnement d'entraînement (défaut 2.0).",
    )
    parser.add_argument("--bert_model",       type=str, default="distilbert-base-uncased")
    parser.add_argument("--max_length",       type=int, default=128)
    parser.add_argument("--embed_batch",      type=int, default=256)

    # ── Modèle 5 (IRM + FT) ──
    parser.add_argument("--irm_steps",  type=int,   default=20_000)
    parser.add_argument("--irm_lr",     type=float, default=5e-4)
    parser.add_argument("--irm_lambda", type=float, default=100.0)

    # ── Commun entraînement tête ──
    parser.add_argument("--batch",      type=int, default=256)
    parser.add_argument("--eval_every", type=int, default=200)

    # ── Sortie ──
    parser.add_argument("--out_dir", type=str,
                        default=str(_Path(__file__).parent / "logs" / "irm_ft"))

    args   = parser.parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ─────────────────────────────────────────────────────────────────────
    # Étape 1 : Chargement du dataset
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 1 : Chargement du dataset LabHC/moji")
    print("=" * 70)
    ds = load_moji()

    # ─────────────────────────────────────────────────────────────────────
    # Étape 2 : Splits (même seed que Modèle 3 → indices identiques)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 2 : Construction des splits")
    print("=" * 70)

    idx_e1, idx_e2  = build_train_envs(ds, max_per_group=args.max_per_group, seed=args.seed, sae_ratio=args.sae_ratio)
    idx_val_ind      = build_val_ind(ds, seed=args.seed + 1)
    idx_test_ood     = build_test_ood(ds, n_per_group=args.n_per_group_test, seed=args.seed)

    all_split = ds["all"]
    Y_all = np.asarray(all_split.Y)
    A_all = np.asarray(all_split.A)
    print_split_stats("E1 — biais fort    (90/10)", idx_e1, Y_all, A_all)
    print_split_stats("E2 — biais modéré (70/30)", idx_e2, Y_all, A_all)
    print_split_stats("Val InD — biaisée 80/20", idx_val_ind, Y_all, A_all)
    print_split_stats(f"Test OOD — équilibré ({args.n_per_group_test}/groupe)", idx_test_ood, Y_all, A_all)

    # ─────────────────────────────────────────────────────────────────────
    # Étape 3 : Chargement du backbone Modèle 3
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 3 : Chargement du backbone Modèle 3 (gelé)")
    print("=" * 70)

    # Stratégie de chargement : best_backbone.pt en priorité, sinon extraction
    # depuis best_full_model.pt (backbone + tête, clés préfixées par "bert.")
    backbone_path  = os.path.join(args.backbone_dir, "best_backbone.pt")
    full_model_path = os.path.join(args.backbone_dir, "best_full_model.pt")

    if not os.path.exists(backbone_path) and not os.path.exists(full_model_path):
        raise FileNotFoundError(
            f"Aucun checkpoint trouvé dans {args.backbone_dir}. "
            "Lancez d'abord train_moji_finetuned_erm.py."
        )

    ft_model = FineTunedDistilBERT(model_name=args.bert_model)

    loaded = False
    if os.path.exists(backbone_path):
        try:
            ft_model.bert.load_state_dict(
                torch.load(backbone_path, map_location="cpu", weights_only=True)
            )
            print(f"  Backbone chargé depuis : {backbone_path}")
            loaded = True
        except Exception as e:
            print(f"  AVERTISSEMENT : best_backbone.pt illisible ({e})")
            print(f"  → Fallback sur best_full_model.pt")

    if not loaded:
        full_state = torch.load(full_model_path, map_location="cpu", weights_only=True)
        # Les clés backbone sont préfixées par "bert." dans le full model
        backbone_state = {
            k[len("bert."):]: v
            for k, v in full_state.items()
            if k.startswith("bert.")
        }
        ft_model.bert.load_state_dict(backbone_state)
        print(f"  Backbone extrait depuis : {full_model_path}")

    # Gel complet du backbone
    for param in ft_model.bert.parameters():
        param.requires_grad = False
    ft_model.eval()
    ft_model = ft_model.to(device)
    print(f"  Paramètres gelés : {sum(p.numel() for p in ft_model.bert.parameters()):,}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)

    # ─────────────────────────────────────────────────────────────────────
    # Étape 4 : Embeddings via backbone Modèle 3 (gelé)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 4 : Calcul des embeddings (backbone FT gelé)")
    print("=" * 70)

    def _texts(split, indices):
        return [split.comment_text[int(i)] for i in indices]

    all_train_idx = np.unique(np.concatenate([idx_e1, idx_e2]))
    train_pos = {int(orig): pos for pos, orig in enumerate(all_train_idx)}

    print(f"\n  [Train E1+E2]  {len(all_train_idx):,} textes …")
    emb_train = embed_with_finetuned_backbone(
        _texts(all_split, all_train_idx), ft_model, tokenizer,
        args.max_length, device, args.embed_batch,
    )

    val_ind_pos = {int(orig): pos for pos, orig in enumerate(idx_val_ind)}
    print(f"\n  [Val InD]  {len(idx_val_ind):,} textes …")
    emb_val_ind = embed_with_finetuned_backbone(
        _texts(all_split, idx_val_ind), ft_model, tokenizer,
        args.max_length, device, args.embed_batch,
    )

    all_test_idx = np.unique(idx_test_ood)
    test_pos = {int(orig): pos for pos, orig in enumerate(all_test_idx)}
    print(f"\n  [Test OOD]  {len(all_test_idx):,} textes …")
    emb_test = embed_with_finetuned_backbone(
        _texts(all_split, all_test_idx), ft_model, tokenizer,
        args.max_length, device, args.embed_batch,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 5 : Assemblage des Env
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 5 : Assemblage des Env")
    print("=" * 70)

    def _make_env(split, indices, emb_all, pos_map):
        pos  = np.array([pos_map[int(i)] for i in indices])
        emb  = emb_all[pos]
        Y    = np.asarray(split.Y)[indices].astype(np.float32)
        A    = np.asarray(split.A)[indices].astype(np.int64)
        return make_env(emb, Y, identities=A)

    env_e1        = _make_env(all_split, idx_e1,      emb_train,  train_pos)
    env_e2        = _make_env(all_split, idx_e2,      emb_train,  train_pos)
    env_val_ind   = _make_env(all_split, idx_val_ind, emb_val_ind, val_ind_pos)
    env_test_ood  = _make_env(all_split, idx_test_ood, emb_test,   test_pos)

    print(f"  E1        : {env_e1.X.shape}  Y=1: {env_e1.y.mean():.2%}")
    print(f"  E2        : {env_e2.X.shape}  Y=1: {env_e2.y.mean():.2%}")
    print(f"  Val InD   : {env_val_ind.X.shape}  Y=1: {env_val_ind.y.mean():.2%}")
    print(f"  Test OOD  : {env_test_ood.X.shape}  Y=1: {env_test_ood.y.mean():.2%}")

    # ─────────────────────────────────────────────────────────────────────
    # Étape 6 : Modèle 5 — IRM + Fine-tuned backbone
    # Tête LogReg entraînée par IRM sur E1+E2 (embeddings FT)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 6 : Modèle 5 — IRM + FT backbone (tête IRM sur E1+E2)")
    print(f"  steps={args.irm_steps}  lr={args.irm_lr}  lambda={args.irm_lambda}")
    print("=" * 70)

    model_irm_ft, hist_irm_ft = train_irm(
        envs=[env_e1, env_e2],       # E1+E2 avec embeddings du backbone FT
        steps=args.irm_steps,
        lr=args.irm_lr,
        batch=args.batch,
        irm_lambda=args.irm_lambda,
        seed=args.seed,
        device=device,
        eval_every=args.eval_every,
        val_envs=[env_val_ind],
        test_env=env_test_ood,
        dataset_name="moji_irm_ft",
        n_classes=2,
    )

    # ─────────────────────────────────────────────────────────────────────
    # Étape 7 : Évaluation finale
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ÉTAPE 7 : Évaluation finale")
    print("=" * 70)

    results = {
        "irm_ft": {},
    }

    print("\n--- Modèle 5 : IRM + FT backbone ---")
    results["irm_ft"]["val_ind"]  = full_evaluation(model_irm_ft, env_val_ind,  device, "Val InD")
    results["irm_ft"]["test_ood"] = full_evaluation(model_irm_ft, env_test_ood, device, "Test OOD")

    # ─────────────────────────────────────────────────────────────────────
    # Résumé — comparaison IRM+FT vs Modèle 3 (si dispo)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY — Test OOD (équilibré, 7500/groupe)")
    print("=" * 70)

    # Chargement optionnel des résultats du Modèle 3
    model3_path = os.path.join(args.backbone_dir, "results.json")
    m3 = None
    if os.path.exists(model3_path):
        with open(model3_path) as f:
            m3_raw = json.load(f)
        m3 = m3_raw.get("test_ood", m3_raw)
        print(f"  (Modèle 3 chargé depuis {model3_path})")

    header = f"  {'Metric':<35}"
    if m3:
        header += f" {'M3 ERM-FT':>12}"
    header += f" {'M5 IRM+FT':>12}"
    print(header)
    print("  " + "-" * (35 + 12 * (2 if m3 else 1) + 2 * (2 if m3 else 1)))

    for metric, label_str in [
        ("accuracy",        "Accuracy"),
        ("macro_f1",        "Macro-F1"),
        ("loss",            "Loss"),
        ("fnr_pos_aae",     "FNR  (Y=1,A=1)  AAE positifs"),
        ("fpr_neg_sae",     "FPR  (Y=0,A=0)  SAE négatifs"),
        ("eod_tpr",         "EOD TPR  |TPR_SAE − TPR_AAE|"),
        ("eod_fpr",         "EOD FPR  |FPR_SAE − FPR_AAE|"),
        ("worst_group_acc", "Worst-Group Accuracy"),
        ("avg_group_acc",   "Avg-Group Accuracy"),
    ]:
        row = f"  {label_str:<35}"
        if m3:
            row += f" {m3.get(metric, float('nan')):>12.4f}"
        row += f" {results['irm_ft']['test_ood'][metric]:>12.4f}"
        print(row)

    print(f"\n  {'Groupe (test OOD)':<35}")
    for gk, glabel in [
        ("(Y=0,A=0)", "    Neg SAE  (Y=0, A=0)"),
        ("(Y=0,A=1)", "    Neg AAE  (Y=0, A=1)"),
        ("(Y=1,A=0)", "    Pos SAE  (Y=1, A=0)"),
        ("(Y=1,A=1)", "    Pos AAE  (Y=1, A=1)"),
    ]:
        row = f"  {glabel:<35}"
        if m3:
            row += f" {m3.get('acc_groups', {}).get(gk, float('nan')):>12.4f}"
        row += f" {results['irm_ft']['test_ood']['acc_groups'][gk]:>12.4f}"
        print(row)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY — Val InD (biaisé 80/20)")
    print("=" * 70)

    m3_val = None
    if m3_raw:
        m3_val = m3_raw.get("val_ind", None)

    for metric, label_str in [
        ("accuracy",        "Accuracy"),
        ("macro_f1",        "Macro-F1"),
        ("eod_tpr",         "EOD TPR  |TPR_SAE − TPR_AAE|"),
        ("worst_group_acc", "Worst-Group Accuracy"),
        ("avg_group_acc",   "Avg-Group Accuracy"),
    ]:
        row = f"  {label_str:<35}"
        if m3_val:
            row += f" {m3_val.get(metric, float('nan')):>12.4f}"
        row += f" {results['irm_ft']['val_ind'][metric]:>12.4f}"
        print(row)

    # ─────────────────────────────────────────────────────────────────────
    # Sauvegarde
    # ─────────────────────────────────────────────────────────────────────
    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(args.out_dir, "history_irm_ft.json"), "w") as f:
        json.dump(hist_irm_ft, f, indent=2)

    # Sauvegarde de la tête (le backbone est partagé avec logs_finetuned_erm)
    torch.save(model_irm_ft.state_dict(), os.path.join(args.out_dir, "head_irm_ft.pt"))

    print(f"\n  Résultats      → {args.out_dir}/results.json")
    print(f"  Tête IRM+FT    → {args.out_dir}/head_irm_ft.pt")
    print(f"  Historique IRM → {args.out_dir}/history_irm_ft.json")
    print("\nDone!")


if __name__ == "__main__":
    main()
