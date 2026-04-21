#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
toy_train_eval_grid.py
Lance une grille d'expériences sur le dataset jouet (toy_letters.py) :
  - Génère les environnements d'entraînement (m=2 par défaut) avec un 'gap'
    contrôlant la force du raccourci (p_align) par environnement.
  - Entraîne et évalue deux variantes de modèles :
      * eLM / ERM  (mode="ilm")   -> train_file = .../train_erm.txt
      * IRM-Games  (mode="game")  -> train_file = .../envs/
  - Val OOD (anti-alignée) via val.txt
Compatibilité : prévu pour ton script run_invariant_mlm.py

Exemple d'usage :
  python toy_train_eval_grid.py \
    --gaps 0.10,0.30 \
    --lrs 1e-4,5e-5 \
    --seeds 0,1 \
    --out_root runs_toy

Astuce :
  - Assure-toi que run_invariant_mlm.py infère bien num_labels=2 (ou force-le).
  - Si tu utilises W&B, exporte WANDB_PROJECT/WANDB_ENTITY avant.
"""

import os
import sys
import shlex
import argparse
import subprocess
from pathlib import Path

# ---------- utils ----------
def run(cmd_list, env=None):
    print("[CMD]", " ".join(shlex.quote(x) for x in cmd_list))
    subprocess.run(cmd_list, check=True, env=env)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    # grille
    ap.add_argument("--gaps", type=str, default="0.10,0.30", help="Liste des gaps séparés par des virgules.")
    ap.add_argument("--lrs", type=str, default="1e-4,5e-5", help="Liste des learning rates.")
    ap.add_argument("--seeds", type=str, default="0,1", help="Liste des seeds.")
    # data gen
    ap.add_argument("--m", type=int, default=2, help="Nombre d'environnements d'entraînement.")
    ap.add_argument("--color_p_mean", type=float, default=0.15, help="Mean des coloring probabilities.")
    ap.add_argument("--etas", type=str, default="0.25,0.25", help="Bruit de label par env (taille m).")
    ap.add_argument("--val_eta", type=float, default=0.0, help="Bruit de label val.")
    ap.add_argument("--n_train_per_env", type=int, default=20000)
    ap.add_argument("--n_val", type=int, default=5000)
    ap.add_argument("--val_color_p", type=float, default=1.0, help="1.0 => p_align_val=0.0 (anti-aligné).")
    # training commun
    ap.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased")
    ap.add_argument("--nb_steps", type=int, default=5000)
    ap.add_argument("--eval_steps", type=int, default=250)
    ap.add_argument("--max_seq_length", type=int, default=64)
    ap.add_argument("--per_device_train_batch_size", type=int, default=64)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=64)
    ap.add_argument("--fp16", action="store_true", default=True)
    # invariant (IRM-Games) spécifiques
    ap.add_argument("--K", type=int, default=1, help="--head_updates_per_encoder_update")
    ap.add_argument("--freeze_phi", action="store_true", default=True, help="F-IRM si True")
    # sorties
    ap.add_argument("--out_root", type=str, default="runs_toy", help="Racine des sorties (runs + data).")
    ap.add_argument("--data_root", type=str, default="data_toy", help="Racine des datasets générés.")
    # options diverses
    ap.add_argument("--skip_erm", action="store_true", help="Ne pas lancer ERM.")
    ap.add_argument("--skip_invariant", action="store_true", help="Ne pas lancer IRM-Games.")
    ap.add_argument("--overwrite_output_dir", action="store_true", default=True)
    ap.add_argument("--local_rank", type=int, default=-1)
    args = ap.parse_args()

    gaps  = [float(x) for x in args.gaps.split(",") if x.strip() != ""]
    lrs   = [x.strip() for x in args.lrs.split(",") if x.strip() != ""]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]

    out_root  = Path(args.out_root)
    data_root = Path(args.data_root)
    ensure_dir(out_root)
    ensure_dir(data_root)

    # boucle principale : gap × seed × lr
    for gap in gaps:
        # ---- 1) Générer dataset si absent ----
        data_dir = data_root / f"gap_{int(gap*100):03d}"
        env_dir  = data_dir / "envs"
        val_dir  = data_dir / "val_test"

        if not (env_dir.exists() and (data_dir / "train_erm.txt").exists() and (val_dir / "val.txt").exists()):
            ensure_dir(data_dir)
            cmd_data = [
                sys.executable, "toy_letters.py",
                "--gap", f"{gap:.2f}",
                "--color_p_mean", f"{args.color_p_mean:.4f}",
                "--m", str(args.m),
                "--etas", args.etas,
                "--val_eta", f"{args.val_eta:.4f}",
                "--n_train_per_env", str(args.n_train_per_env),
                "--n_val", str(args.n_val),
                "--val_color_p", f"{args.val_color_p:.4f}",
                "--out_dir", str(data_dir),
                "--seed", "0",   # même données pour tous les runs; change si besoin
            ]
            run(cmd_data)
        else:
            print(f"[DATA] Skipping generation (exists): {data_dir}")

        # chemins communs
        train_erm_txt = str(data_dir / "train_erm.txt")
        train_envs    = str(env_dir)  # dossier
        val_txt       = str(val_dir / "val.txt")

        for seed in seeds:
            for lr in lrs:
                # ---- 2) ERM (eLM) ----
                if not args.skip_erm:
                    out_dir_erm = out_root / f"elm_gap{int(gap*100):03d}_lr{lr.replace('.', '')}_seed{seed}"
                    ensure_dir(out_dir_erm)
                    cmd_erm = [
                        sys.executable, "run_invariant_mlm.py",
                        "--mode", "ilm",
                        "--model_name_or_path", args.model_name_or_path,
                        "--train_file", train_erm_txt,
                        "--validation_file", val_txt,
                        "--do_train", "--do_eval",
                        "--nb_steps", str(args.nb_steps),
                        "--learning_rate", str(lr),
                        "--max_seq_length", str(args.max_seq_length),
                        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
                        "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
                        "--output_dir", str(out_dir_erm),
                        "--run_name", f"elm_gap{gap:.2f}_lr{lr}_seed{seed}",
                        "--evaluation_strategy", "steps",
                        "--eval_steps", str(args.eval_steps),
                        "--save_strategy", "no",
                        "--seed", str(seed),
                        "--local_rank", str(args.local_rank),
                    ]
                    if args.fp16:
                        cmd_erm += ["--fp16", "--half_precision_backend", "cuda_amp"]
                    if args.overwrite_output_dir:
                        cmd_erm += ["--overwrite_output_dir"]
                    run(cmd_erm)

                if not args.skip_erm:
                    out_dir_erm = out_root / f"ilm_gap{int(gap*100):03d}_lr{lr.replace('.', '')}_seed{seed}"
                    ensure_dir(out_dir_erm)
                    cmd_erm = [
                        sys.executable, "run_invariant_mlm.py",
                        "--mode", "ilm",
                        "--model_name_or_path", args.model_name_or_path,
                        "--train_file", train_envs,
                        "--validation_file", val_txt,
                        "--do_train", "--do_eval",
                        "--nb_steps", str(args.nb_steps),
                        "--learning_rate", str(lr),
                        "--max_seq_length", str(args.max_seq_length),
                        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
                        "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
                        "--output_dir", str(out_dir_erm),
                        "--run_name", f"elm_gap{gap:.2f}_lr{lr}_seed{seed}",
                        "--evaluation_strategy", "steps",
                        "--eval_steps", str(args.eval_steps),
                        "--save_strategy", "no",
                        "--seed", str(seed),
                        "--local_rank", str(args.local_rank),
                    ]
                    if args.fp16:
                        cmd_erm += ["--fp16", "--half_precision_backend", "cuda_amp"]
                    if args.overwrite_output_dir:
                        cmd_erm += ["--overwrite_output_dir"]
                    run(cmd_erm)

                # ---- 3) IRM-Games (invariant) ----
                if not args.skip_invariant:
                    out_dir_ilmg = out_root / f"ilmg_gap{int(gap*100):03d}_lr{lr.replace('.', '')}_seed{seed}_K{args.K}{'_freeze' if args.freeze_phi else ''}"
                    ensure_dir(out_dir_ilmg)
                    cmd_ilmg = [
                        sys.executable, "run_invariant_mlm.py",
                        "--mode", "game",
                        "--model_name_or_path", args.model_name_or_path,
                        "--train_file", train_envs,              # dossier envs/
                        "--validation_file", val_txt,
                        "--do_train", "--do_eval",
                        "--nb_steps", str(args.nb_steps),
                        "--learning_rate", str(lr),
                        "--max_seq_length", str(args.max_seq_length),
                        "--per_device_train_batch_size", str(args.per_device_train_batch_size),
                        "--per_device_eval_batch_size", str(args.per_device_eval_batch_size),
                        "--output_dir", str(out_dir_ilmg),
                        "--run_name", f"ilmg_gap{gap:.2f}_lr{lr}_seed{seed}_K{args.K}{'_freeze' if args.freeze_phi else ''}",
                        "--evaluation_strategy", "steps",
                        "--eval_steps", str(args.eval_steps),
                        "--save_strategy", "no",
                        "--seed", str(seed),
                        "--local_rank", str(args.local_rank),
                        "--head_updates_per_encoder_update", str(args.K),
                    ]
                    if args.freeze_phi:
                        cmd_ilmg += ["--freeze_phi"]
                    if args.fp16:
                        cmd_ilmg += ["--fp16", "--half_precision_backend", "cuda_amp"]
                    if args.overwrite_output_dir:
                        cmd_ilmg += ["--overwrite_output_dir"]
                    run(cmd_ilmg)

    print("\n[OK] Grid finished.")

if __name__ == "__main__":
    main()
