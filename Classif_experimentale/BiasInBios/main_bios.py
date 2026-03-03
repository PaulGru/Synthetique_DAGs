"""
main_bios.py — Point d'entrée pour l'expérience ERM vs IRM sur Bias in Bios.

Usage :
    uv run main_bios.py --train_rho 0.8 0.6 --test_rho 0.2 \
        --erm_steps 10000 --irm_steps 10000 --irm_lambda 500.0 \
        --model_kind logreg --device auto --outdir plots/
"""

from __future__ import annotations

import argparse
import os

from data_bios import build_envs_bios, N_CLASSES, PROFESSIONS
from models_bios import train_erm, train_irm
from utils_bios import (
    evaluate_multiclass,
    evaluate_by_gender,
    plot_results,
    plot_per_class_accuracy,
    resolve_device,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Expérience ERM vs IRM — Bias in Bios (multi-classes)"
    )
    # Environnements
    p.add_argument(
        "--train_rho", type=float, nargs="+", default=[0.8, 0.6],
        help="Force de corrélation genre-profession par env train. Ex: 0.8 0.6",
    )
    p.add_argument(
        "--test_rho", type=float, default=0.2,
        help="Force de corrélation pour le test OOD (< train_rho pour inverser).",
    )
    p.add_argument(
        "--max_samples", type=int, default=None,
        help="Limiter le nb d'exemples chargés (utile pour tests rapides).",
    )
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)

    # BERT
    p.add_argument("--bert_model", type=str, default="bert-base-uncased")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--bert_batch_size", type=int, default=64)
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])

    # Modèle
    p.add_argument(
        "--model_kind", type=str, default="logreg", choices=["logreg", "mlp"],
        help="logreg : régression logistique | mlp : petit MLP",
    )
    p.add_argument("--mlp_hidden", type=int, default=256)
    p.add_argument("--mlp_layers", type=int, default=1)
    p.add_argument("--mlp_dropout", type=float, default=0.1)

    # ERM
    p.add_argument("--erm_steps", type=int, default=10000)
    p.add_argument("--erm_lr", type=float, default=1e-3)
    p.add_argument("--erm_batch", type=int, default=256)
    p.add_argument("--skip_erm", action="store_true", help="Ne pas entraîner ERM")

    # IRM
    p.add_argument("--irm_steps", type=int, default=10000)
    p.add_argument("--irm_lr", type=float, default=1e-3)
    p.add_argument("--irm_batch", type=int, default=256)
    p.add_argument("--irm_lambda", type=float, default=500.0)
    p.add_argument("--skip_irm", action="store_true", help="Ne pas entraîner IRM")

    # Infra
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--outdir", type=str, default="plots")

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    device = resolve_device(args.device)

    print("=" * 60)
    print("Bias in Bios — ERM vs IRM")
    print(f"  Classes          : {N_CLASSES} professions")
    print(f"  Train rho        : {args.train_rho}")
    print(f"  Test rho (OOD)   : {args.test_rho}")
    print(f"  Modèle           : {args.model_kind}")
    print(f"  Device           : {device}")
    print("=" * 60)

    # ── 1. Construction des environnements ───────────────────────────────────
    print("\n[1/3] Construction des environnements...")
    train_envs, val_envs, test_env = build_envs_bios(
        train_rho=args.train_rho,
        test_rho=args.test_rho,
        seed=args.seed,
        val_frac=args.val_frac,
        bert_model=args.bert_model,
        max_length=args.max_length,
        device=device,
        pooling=args.pooling,
        bert_batch_size=args.bert_batch_size,
        max_samples=args.max_samples,
    )

    # ── 2. ERM ───────────────────────────────────────────────────────────────
    erm_model, erm_history = None, {"step": [], "train_acc": [], "val_acc": [],
                                     "test_acc": [], "gender_gap": [], "loss": []}
    if not args.skip_erm:
        print("\n[2/3] Entraînement ERM...")
        erm_model, erm_history = train_erm(
            envs=train_envs,
            steps=args.erm_steps,
            lr=args.erm_lr,
            batch=args.erm_batch,
            seed=args.seed,
            device=device,
            eval_every=args.eval_every,
            val_envs=val_envs,
            test_env=test_env,
            model_kind=args.model_kind,
            mlp_hidden=args.mlp_hidden,
            mlp_layers=args.mlp_layers,
            mlp_dropout=args.mlp_dropout,
        )
        erm_te = evaluate_multiclass(erm_model, test_env, device)
        erm_gg = evaluate_by_gender(erm_model, test_env, device)
        print(f"\n  ERM — Test OOD acc : {erm_te:.3f}")
        print(f"  ERM — Gender gap (OOD) : "
              f"acc_M={erm_gg['acc_male']:.3f}, acc_F={erm_gg['acc_female']:.3f}, "
              f"gap={erm_gg['gap']:.3f}")

    # ── 3. IRM ───────────────────────────────────────────────────────────────
    irm_model, irm_history = None, {"step": [], "train_acc": [], "val_acc": [],
                                     "test_acc": [], "gender_gap": [], "loss": []}
    if not args.skip_irm:
        print("\n[3/3] Entraînement IRM...")
        irm_model, irm_history = train_irm(
            envs=train_envs,
            steps=args.irm_steps,
            lr=args.irm_lr,
            batch=args.irm_batch,
            irm_lambda=args.irm_lambda,
            seed=args.seed,
            device=device,
            eval_every=args.eval_every,
            val_envs=val_envs,
            test_env=test_env,
            model_kind=args.model_kind,
            mlp_hidden=args.mlp_hidden,
            mlp_layers=args.mlp_layers,
            mlp_dropout=args.mlp_dropout,
        )
        irm_te = evaluate_multiclass(irm_model, test_env, device)
        irm_gg = evaluate_by_gender(irm_model, test_env, device)
        print(f"\n  IRM — Test OOD acc : {irm_te:.3f}")
        print(f"  IRM — Gender gap (OOD) : "
              f"acc_M={irm_gg['acc_male']:.3f}, acc_F={irm_gg['acc_female']:.3f}, "
              f"gap={irm_gg['gap']:.3f}")

    # ── 4. Résumé final ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RÉSUMÉ FINAL")
    if erm_model and irm_model:
        erm_te = evaluate_multiclass(erm_model, test_env, device)
        irm_te = evaluate_multiclass(irm_model, test_env, device)
        print(f"  ERM Test OOD : {erm_te:.3f}")
        print(f"  IRM Test OOD : {irm_te:.3f}")
        print(f"  Gain IRM     : {irm_te - erm_te:+.3f}")

    # ── 5. Visualisations ────────────────────────────────────────────────────
    if erm_history["step"] or irm_history["step"]:
        plot_results(erm_history, irm_history, outdir=args.outdir)
    if erm_model:
        plot_per_class_accuracy(erm_model, test_env, "ERM", device, args.outdir)
    if irm_model:
        plot_per_class_accuracy(irm_model, test_env, "IRM", device, args.outdir)

    print(f"\n✅ Expérience terminée. Plots dans : {args.outdir}/")


if __name__ == "__main__":
    main()
