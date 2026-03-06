"""
main_bios.py — ERM vs IRM sur Bias in Bios.

Évaluation sur deux jeux de test :
  - test_id  : split "test" du dataset (in-distribution)
  - test_ood : split "dev" filtré aux exemples de genre minoritaire (OOD)
"""

from __future__ import annotations

import argparse
import os

from data_bios import build_envs_bios, N_CLASSES
from models_bios import train_erm, train_irm
from utils_bios import (
    evaluate_by_gender,
    evaluate_full_report,
    print_full_summary,
    plot_results,
    plot_confusion_matrix,
    plot_f1_per_class_comparison,
    resolve_device,
)


def parse_args():
    p = argparse.ArgumentParser(description="ERM vs IRM — Bias in Bios")
    p.add_argument("--n_train_envs", type=int,   default=2)
    p.add_argument("--max_samples",  type=int,   default=None)
    p.add_argument("--seed",         type=int,   default=42)
    # BERT
    p.add_argument("--bert_model",      type=str,   default="bert-base-uncased")
    p.add_argument("--max_length",      type=int,   default=128)
    p.add_argument("--bert_batch_size", type=int,   default=64)
    p.add_argument("--pooling",         type=str,   default="mean", choices=["mean", "cls"])
    # Modèle
    p.add_argument("--model_kind",  type=str,   default="logreg", choices=["logreg", "mlp"])
    p.add_argument("--mlp_hidden",  type=int,   default=256)
    p.add_argument("--mlp_layers",  type=int,   default=1)
    p.add_argument("--mlp_dropout", type=float, default=0.1)
    # ERM
    p.add_argument("--erm_steps", type=int,   default=10000)
    p.add_argument("--erm_lr",    type=float, default=1e-3)
    p.add_argument("--erm_batch", type=int,   default=256)
    p.add_argument("--skip_erm",  action="store_true")
    # IRM
    p.add_argument("--irm_steps",  type=int,   default=10000)
    p.add_argument("--irm_lr",     type=float, default=1e-3)
    p.add_argument("--irm_batch",  type=int,   default=256)
    p.add_argument("--irm_lambda", type=float, default=100.0)
    p.add_argument("--skip_irm",   action="store_true")
    # Infra
    p.add_argument("--device",     type=str, default="auto")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--outdir",     type=str, default="plots")
    return p.parse_args()


def _compare_table(tag: str, erm_rep, irm_rep):
    if not (erm_rep and irm_rep):
        return
    print("\n" + "=" * 60)
    print(f"COMPARAISON ERM vs IRM ({tag})")
    print(f"  {'':20}  {'ERM':>8}  {'IRM':>8}  {'ΔIRM':>8}")
    print(f"  {'-'*52}")
    for metric in ["accuracy", "f1_macro", "f1_micro"]:
        e = erm_rep[metric]
        i = irm_rep[metric]
        print(f"  {metric:<20}  {e:>8.4f}  {i:>8.4f}  {i-e:>+8.4f}")
    print("=" * 60)


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = resolve_device(args.device)

    print("=" * 60)
    print("Bias in Bios — ERM vs IRM")
    print(f"  Classes     : {N_CLASSES} professions déséquilibrées")
    print(f"  Train envs  : {args.n_train_envs}")
    print(f"  Test OOD    : dev filtré aux genres minoritaires")
    print(f"  Modèle      : {args.model_kind}  |  Device : {device}")
    print("=" * 60)

    # ── 1. Environnements ────────────────────────────────────────────────────
    print("\n[1/3] Construction des environnements...")
    train_envs, val_env, test_id_env, test_ood_env = build_envs_bios(
        n_train_envs=args.n_train_envs,
        seed=args.seed,
        bert_model=args.bert_model,
        max_length=args.max_length,
        device=device,
        pooling=args.pooling,
        bert_batch_size=args.bert_batch_size,
        max_samples=args.max_samples,
    )

    # ── 2. ERM ───────────────────────────────────────────────────────────────
    erm_model = None
    erm_history = {"step": [], "train_acc": [], "val_acc": [], "test_acc": [],
                   "test_f1": [], "gender_gap": [], "gender_gap_f1": [], "loss": []}
    if not args.skip_erm:
        print("\n[2/3] Entraînement ERM...")
        erm_model, erm_history = train_erm(
            envs=train_envs, steps=args.erm_steps, lr=args.erm_lr,
            batch=args.erm_batch, seed=args.seed, device=device,
            eval_every=args.eval_every, val_envs=[val_env],
            test_env=test_ood_env,          # on monitore sur l'OOD pendant le train
            model_kind=args.model_kind, mlp_hidden=args.mlp_hidden,
            mlp_layers=args.mlp_layers, mlp_dropout=args.mlp_dropout,
        )

    # ── 3. IRM ───────────────────────────────────────────────────────────────
    irm_model = None
    irm_history = {"step": [], "train_acc": [], "val_acc": [], "test_acc": [],
                   "test_f1": [], "gender_gap": [], "gender_gap_f1": [], "loss": []}
    if not args.skip_irm:
        print("\n[3/3] Entraînement IRM...")
        irm_model, irm_history = train_irm(
            envs=train_envs, steps=args.irm_steps, lr=args.irm_lr,
            batch=args.irm_batch, irm_lambda=args.irm_lambda, seed=args.seed,
            device=device, eval_every=args.eval_every, val_envs=[val_env],
            test_env=test_ood_env,          # on monitore sur l'OOD pendant le train
            model_kind=args.model_kind,
            mlp_hidden=args.mlp_hidden, mlp_layers=args.mlp_layers,
            mlp_dropout=args.mlp_dropout,
        )

    # ── 4. Résumés finaux ────────────────────────────────────────────────────
    for tag, env in [("Test ID (in-distribution)", test_id_env),
                     ("Test OOD (genres minoritaires)", test_ood_env)]:
        print(f"\n{'#'*60}")
        print(f"# {tag}")
        print(f"{'#'*60}")

        erm_rep, irm_rep = None, None
        if erm_model:
            erm_rep = evaluate_full_report(erm_model, env, device)
            erm_gs  = evaluate_by_gender(erm_model, env, device)
            print_full_summary("ERM", erm_rep, erm_gs)
        if irm_model:
            irm_rep = evaluate_full_report(irm_model, env, device)
            irm_gs  = evaluate_by_gender(irm_model, env, device)
            print_full_summary("IRM", irm_rep, irm_gs)

        _compare_table(tag, erm_rep, irm_rep)

    # ── 5. Visualisations ────────────────────────────────────────────────────
    if erm_history["step"] or irm_history["step"]:
        plot_results(erm_history, irm_history, outdir=args.outdir)

    # Plots sur le test OOD
    plot_f1_per_class_comparison(erm_model, irm_model, test_ood_env, device, args.outdir)
    if erm_model:
        plot_confusion_matrix(erm_model, test_ood_env, "ERM_ood", device, args.outdir)
    if irm_model:
        plot_confusion_matrix(irm_model, test_ood_env, "IRM_ood", device, args.outdir)

    print(f"\n✅ Expérience terminée. Plots dans : {args.outdir}/")


if __name__ == "__main__":
    main()
