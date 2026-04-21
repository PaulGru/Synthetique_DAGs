#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toy Letters Dataset (setup inspiré de "IRM — when it works and when it doesn't"):

Entrée = préfixe spurious (<c>/<d>) + une lettre x ∈ {a, b}
Label causal y = 1 si x='a', 0 si x='b'

Entrainement (m environnements):
 - Chaque env e a une probabilité p_align,e que le préfixe soit aligné au label (sinon anti-aligné).
 - On ajoute un bruit de label eta_e (flip y_true -> 1-y_true avec proba eta_e) pour rendre le raccourci
   plus fiable que la règle causale (on vise 1-eta_e < p_align,e).

Validation OoD:
 - p_align,val fixé via val_color_p (p_align,val = 1 - val_color_p).
   Par défaut val_color_p=1.0 => p_align,val=0.0, donc inversion/anti-alignement pur du raccourci.
 - Pas (ou peu) de bruit sur la validation (val_eta=0.0 par défaut).

Ce script produit:
  out_dir/
    envs/env_1.txt, env_2.txt, ...   (texte \t label)
    train_erm.txt                     (concat des envs, pour ERM)
    val_test/val.txt                  (validation OoD)

Exemple d’appel (2 envs, p_align ≈ {0.9, 0.8}, val anti-alignée, bruit 0.25 au train) :
  python toy_letters.py \
    --gap 0.10 --color_p_mean 0.15 --m 2 \
    --etas 0.25,0.25 --val_eta 0.0 \
    --n_train_per_env 20000 --n_val 5000 \
    --val_color_p 1.0 \
    --out_dir donnees_gap_010 --seed 0
"""

import os
import sys
import random
import argparse
from pathlib import Path
from typing import Tuple, List

def clip01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

def linspace(a: float, b: float, m: int) -> List[float]:
    if m <= 1:
        return [a]
    step = (b - a) / float(m - 1)
    return [a + j * step for j in range(m)]

def make_example(y_true: int, p_align: float, eta: float, rng: random.Random) -> Tuple[str, int, bool]:
    """
    Crée un exemple texte + label (avec bruit) + indicateur 'aligned' (True si le préfixe correspond au label final).
    - y_true : label causal (1 si 'a', 0 si 'b')
    - p_align: proba que le préfixe soit aligné au label (sinon anti-aligné)
    - eta    : proba de flip du label (bruit)
    """
    # 1) Flip de label (bruit)
    y = 1 - y_true if rng.random() < eta else y_true
    x = 'a' if y_true == 1 else 'b'

    # 2) Préfixe spurious aligné ou non avec le label y
    aligned = (rng.random() < p_align)
    spur_aligned = '<d>' if y == 1 else '<c>'   # aligné au label
    spur_anti    = '<c>' if y == 1 else '<d>'   # anti-aligné
    spur = spur_aligned if aligned else spur_anti

    text = f"{spur} {x}"
    return text, y, aligned

def main():
    p = argparse.ArgumentParser()
    # Paramétrage des environnements (p_color -> p_align = 1 - p_color)
    p.add_argument("--gap", type=float, default=0.10,
                   help="Amplitude sur les 'coloring probabilities'. p_align = 1 - p_color.")
    p.add_argument("--color_p_mean", type=float, default=0.15,
                   help="Moyenne des 'coloring probabilities' (ex: 0.15 => p_align moyen ≈ 0.85).")
    p.add_argument("--m", type=int, default=2, help="Nombre d'environnements d'entraînement (>=2).")

    # Bruit de label (train/val)
    p.add_argument("--etas", type=str, default="0.25,0.25",
                   help="Liste des eta_e (flip label) pour chaque environnement de train, ex: '0.25,0.25'")
    p.add_argument("--val_eta", type=float, default=0.0, help="Bruit de label pour la validation (OoD).")

    # Validation OoD
    p.add_argument("--val_color_p", type=float, default=1.0,
                   help="Coloring prob de validation (p_align_val = 1 - val_color_p). 1.0 => p_align_val=0.0 (anti-aligné).")

    # Tailles
    p.add_argument("--n_train_per_env", type=int, default=20000,
                   help="Nombre d'exemples par environnement d'entraînement.")
    p.add_argument("--n_val", type=int, default=5000,
                   help="Taille du set de validation OoD.")

    # I/O & seed
    p.add_argument("--out_dir", type=str, required=True, help="Dossier de sortie.")
    p.add_argument("--seed", type=int, default=0, help="Graine aléatoire.")
    args = p.parse_args()

    rng = random.Random(args.seed)

    # --- Dossiers de sortie
    out_dir = Path(args.out_dir)
    env_dir = out_dir / "envs"
    val_dir = out_dir / "val_test"
    env_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # --- Paramètres des environnements
    m = max(2, int(args.m))
    etas = [float(x) for x in args.etas.split(",")]
    if len(etas) != m:
        print(f"[toy][ERREUR] --etas doit avoir {m} valeurs (actuel: {len(etas)}).", file=sys.stderr)
        sys.exit(1)

    # offsets sur p_color dans [mean - gap/2, mean + gap/2], puis clip dans [0,1]
    offs = linspace(-args.gap / 2.0, +args.gap / 2.0, m)
    p_colors = [clip01(args.color_p_mean + off) for off in offs]
    p_aligns = [1.0 - pc for pc in p_colors]

    # Validation: p_align_val = 1 - val_color_p
    p_align_val = 1.0 - float(args.val_color_p)

    print(f"[toy] m={m}")
    print(f"[toy] p_color (train): {p_colors}")
    print(f"[toy] p_align (train): {p_aligns}")
    print(f"[toy] p_align (val):   {p_align_val}")
    print(f"[toy] etas (train):    {etas}")
    print(f"[toy] val_eta:         {args.val_eta}")

    # --- Génération des environnements d'entraînement, équilibrés 50/50 sur y_true
    train_erm_lines: List[Tuple[str, int]] = []
    for i, (p_align, eta) in enumerate(zip(p_aligns, etas), start=1):
        lines: List[Tuple[str, int]] = []
        n = int(args.n_train_per_env)
        # équilibre exact (si n impair, on mettra un exemple de plus pour y_true=0)
        n0 = n // 2
        n1 = n - n0

        aligned_count = 0
        for _ in range(n0):
            text, y, aligned = make_example(0, p_align, eta, rng)
            lines.append((text, y))
            aligned_count += int(aligned)
        for _ in range(n1):
            text, y, aligned = make_example(1, p_align, eta, rng)
            lines.append((text, y))
            aligned_count += int(aligned)

        rng.shuffle(lines)
        out_path = env_dir / f"env_{i}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            for text, y in lines:
                f.write(f"{text}\t{y}\n")

        train_erm_lines.extend(lines)

        # Statistiques rapides par environnement
        align_ratio = aligned_count / float(n) if n > 0 else 0.0
        print(f"[toy][env_{i}] n={n} | approx align_ratio={align_ratio:.3f} (théorique: {p_align:.3f})")

    # --- Concaténation ERM
    rng.shuffle(train_erm_lines)
    with open(out_dir / "train_erm.txt", "w", encoding="utf-8") as f:
        for text, y in train_erm_lines:
            f.write(f"{text}\t{y}\n")

    # --- Validation OoD (par défaut anti-alignée et sans bruit)
    val_lines: List[Tuple[str, int]] = []
    n_val = int(args.n_val)
    n0 = n_val // 2
    n1 = n_val - n0

    aligned_count_val = 0
    for _ in range(n0):
        text, y, aligned = make_example(0, p_align_val, float(args.val_eta), rng)
        val_lines.append((text, y))
        aligned_count_val += int(aligned)
    for _ in range(n1):
        text, y, aligned = make_example(1, p_align_val, float(args.val_eta), rng)
        val_lines.append((text, y))
        aligned_count_val += int(aligned)

    rng.shuffle(val_lines)
    with open(val_dir / "val.txt", "w", encoding="utf-8") as f:
        for text, y in val_lines:
            f.write(f"{text}\t{y}\n")

    align_ratio_val = aligned_count_val / float(n_val) if n_val > 0 else 0.0
    print(f"[toy][val] n={n_val} | approx align_ratio={align_ratio_val:.3f} (théorique: {p_align_val:.3f})")

    # --- Récap chemins
    print(f"[toy] OK → {out_dir}/envs/env_1.txt,...  {out_dir}/val_test/val.txt  {out_dir}/train_erm.txt")

if __name__ == "__main__":
    main()
