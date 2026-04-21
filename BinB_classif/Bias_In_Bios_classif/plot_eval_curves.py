#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan des runs, agrégation par learning rate (moyenne sur seeds) et plots par métrique.
Prérequis: pandas, matplotlib
"""
import argparse, re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# --- Config par défaut ---
TARGET_LRS = {1e-3, 5e-4, 1e-4}  # on garde seulement ces LR
CSV_REL_PATH = Path("csv_logs/eval.csv")  # chemin des eval.csv dans chaque run

LR_PATTERNS = [
    re.compile(r"lr(?P<val>[0-9]*\.?[0-9]+(?:e-?\d+)?)", re.IGNORECASE),  # lr0.001, lr1e-3, lr5e-4
]

def parse_lr_from_path(p: Path):
    s = str(p)
    for pat in LR_PATTERNS:
        m = pat.search(s)
        if m:
            txt = m.group("val")
            try:
                return float(txt)
            except ValueError:
                # essayer des variantes (ex: 1e-03)
                try:
                    return float(txt.replace("E", "e"))
                except Exception:
                    pass
    return None

def discover_runs(base_dir: Path):
    """Retourne liste des chemins vers csv_logs/eval.csv + lr extrait."""
    paths = []
    for eval_csv in base_dir.rglob(str(CSV_REL_PATH)):
        lr = parse_lr_from_path(eval_csv)
        if lr is not None:
            paths.append((eval_csv, lr))
    return paths

def load_eval(eval_csv: Path, lr: float):
    df = pd.read_csv(eval_csv)
    # Normaliser noms attendus
    # Colonnes minimales: step, accuracy, f1_macro, f1_head, f1_tail, gap_head_tail, loss (optionnel)
    # Si 'loss' n'existe pas, on ignore.
    keep = ["step","accuracy","f1_macro","f1_head","f1_tail","gap_head_tail","loss"]
    for k in keep:
        if k not in df.columns:
            # certaines colonnes peuvent manquer (ex: loss)
            pass
    df = df[[c for c in keep if c in df.columns]].copy()
    df["lr"] = lr
    return df

def aggregate_by_lr(df_all: pd.DataFrame, metrics):
    """
    Agrège par (lr, step): mean & std. Retourne dict metric -> (mean_df, std_df).
    mean_df a les colonnes: ['lr','step', metric]
    """
    agg = {}
    for metric in metrics:
        if metric not in df_all.columns:
            continue
        grp = df_all.groupby(["lr","step"], as_index=False)[metric].agg(["mean","std"]).reset_index()
        mean_df = grp[["lr","step","mean"]].rename(columns={"mean": metric})
        std_df  = grp[["lr","step","std"]].rename(columns={"std": metric+"_std"})
        agg[metric] = (mean_df, std_df)
    return agg

def plot_metric(agg_mean: pd.DataFrame, metric: str, outdir: Path, title_suffix=""):
    """
    Un plot par métrique, une courbe par lr. Pas de couleurs explicitement fixées (matplotlib par défaut).
    """
    plt.figure()
    for lr, sub in agg_mean.groupby("lr"):
        # tri par step
        sub = sub.sort_values("step")
        plt.plot(sub["step"], sub[metric], label=f"lr={lr:g}")
    plt.xlabel("step")
    plt.ylabel(metric)
    ttl = f"{metric} vs step"
    if title_suffix:
        ttl += f" — {title_suffix}"
    plt.title(ttl)
    plt.legend()
    out = outdir / f"{metric}.png"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Dossier racine contenant tous les runs (sera scanné récursivement).")
    parser.add_argument("--metrics", nargs="+",
                        default=["accuracy","f1_macro","f1_head","f1_tail","gap_head_tail"],
                        help="Métriques à tracer.")
    parser.add_argument("--outdir", type=str, default="plots",
                        help="Dossier de sortie pour figures et CSV d'agrégats.")
    args = parser.parse_args()

    base = Path(args.base_dir)
    outdir = Path(args.outdir)
    runs = discover_runs(base)

    if not runs:
        raise SystemExit(f"Aucun fichier {CSV_REL_PATH} trouvé sous {base.resolve()}")

    frames = []
    for csv_path, lr in runs:
        if lr in TARGET_LRS:
            try:
                frames.append(load_eval(csv_path, lr))
            except Exception as e:
                print(f"[WARN] lecture échouée pour {csv_path}: {e}")

    if not frames:
        raise SystemExit(f"Aucun eval.csv correspondant aux LR {sorted(TARGET_LRS)} trouvé.")

    df_all = pd.concat(frames, ignore_index=True)
    # garder uniquement steps & métriques demandées dispos
    available_metrics = [m for m in args.metrics if m in df_all.columns]
    if not available_metrics:
        raise SystemExit(f"Aucune des métriques demandées {args.metrics} n'est présente dans les CSV.")

    agg = aggregate_by_lr(df_all, available_metrics)

    # Sauvegarde CSV agrégés + plots
    for metric in available_metrics:
        mean_df, std_df = agg[metric]
        # join pour écrire un seul CSV par metric
        merged = pd.merge(mean_df, std_df, on=["lr","step"], how="left")
        csv_out = outdir / f"summary_{metric}.csv"
        outdir.mkdir(parents=True, exist_ok=True)
        merged.sort_values(["lr","step"]).to_csv(csv_out, index=False)
        fig_path = plot_metric(mean_df, metric, outdir)
        print(f"[OK] {metric}: figure -> {fig_path}, résumé -> {csv_out}")

    print("\nTerminé.")

if __name__ == "__main__":
    main()
