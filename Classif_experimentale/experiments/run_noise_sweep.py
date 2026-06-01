#!/usr/bin/env python3
"""
run_noise_sweep_nlp.py
======================
Sweeps the noise level (label_flip for SAC/selection datasets, gamma for
conf_varying_proxy) while keeping the environment gap fixed.

This measures how IRM's OOD advantage changes as the causal signal degrades.
  - SAC / selection  : noise param = label_flip in [0, noise_max]
  - conf_varying_proxy: noise param = gamma     in [0, noise_max]

Usage:
    uv run nlp_synthetic/run_noise_sweep_nlp.py --dataset nlp_agnews_semi_anti_causal
    uv run nlp_synthetic/run_noise_sweep_nlp.py --dataset nlp_amazon_conf_varying_proxy --noise_step 0.05
    uv run nlp_synthetic/run_noise_sweep_nlp.py --dataset nlp_imdb_genres_size_selection --seeds 0 1 2
"""

import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in [str(_ROOT), str(_ROOT / 'irm'), str(_ROOT / 'nlp')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os
import json
import argparse
from datetime import datetime

import torch
import numpy as np
import matplotlib.pyplot as plt

from data import (
    build_envs_ag_news_semi_anti_causal,
    build_envs_ag_news_size_selection,
    build_envs_ag_news_conf_varying_proxy,
    build_envs_imdb_genres_size_selection,
    build_envs_imdb_genres_semi_anti_causal,
    build_envs_imdb_genres_conf_varying_proxy,
    build_envs_amazon_semi_anti_causal,
    build_envs_amazon_sentiment_selection,
    build_envs_amazon_conf_varying_proxy,
)
from training import train_erm, train_irm
from evaluation import resolve_device


# =============================================================================
# Per-dataset defaults
# =============================================================================
# For SAC / selection : noise = label_flip  (p1, p2 = train env params, fixed)
# For conf_varying_proxy: noise = gamma     (p1, p2 = a_train values, fixed)
# =============================================================================
NLP_NOISE_DEFAULTS = {
    # ── AG News ──────────────────────────────────────────────────────────────
    'nlp_agnews_semi_anti_causal': {
        'p1':           0.9,
        'p2':           0.7,
        'p_test':       0.0,
        'noise_param':  'label_flip',
        'noise_max':    0.25,
        'noise_step':   0.05,
        'param_label':  'p_correct',
        'x_label':      'Label flip rate',
        'max_length':   256,
        'n_classes':    4,
    },
    'nlp_agnews_size_selection': {
        'p1':           0.9,
        'p2':           0.7,
        'p_test':       0.0,
        'noise_param':  'label_flip',
        'noise_max':    0.25,
        'noise_step':   0.05,
        'param_label':  'p_select',
        'x_label':      'Label flip rate',
        'max_length':   256,
        'n_classes':    4,
    },
    'nlp_agnews_conf_varying_proxy': {
        'p1':           0.01,
        'p2':           0.10,
        'p_test':       1.0,
        'noise_param':  'gamma',
        'noise_max':    1.0,
        'noise_step':   0.1,
        'param_label':  'a_train',
        'x_label':      'Gamma (force C→Y)',
        'max_length':   256,
        'n_classes':    4,
    },
    # ── IMDB Genres ──────────────────────────────────────────────────────────
    'nlp_imdb_genres_size_selection': {
        'p1':           0.9,
        'p2':           0.7,
        'p_test':       0.0,
        'noise_param':  'label_flip',
        'noise_max':    0.25,
        'noise_step':   0.05,
        'param_label':  'p_select',
        'x_label':      'Label flip rate',
        'max_length':   256,
        'n_classes':    2,
    },
    'nlp_imdb_genres_semi_anti_causal': {
        'p1':           0.9,
        'p2':           0.7,
        'p_test':       0.0,
        'noise_param':  'label_flip',
        'noise_max':    0.25,
        'noise_step':   0.05,
        'param_label':  'p_correct',
        'x_label':      'Label flip rate',
        'max_length':   256,
        'n_classes':    2,
    },
    'nlp_imdb_genres_conf_varying_proxy': {
        'p1':           0.01,
        'p2':           0.10,
        'p_test':       1.0,
        'noise_param':  'gamma',
        'noise_max':    1.0,
        'noise_step':   0.1,
        'param_label':  'a_train',
        'x_label':      'Gamma (force C→Y)',
        'max_length':   256,
        'n_classes':    2,
    },
    # ── Amazon ───────────────────────────────────────────────────────────────
    'nlp_amazon_semi_anti_causal': {
        'p1':           0.9,
        'p2':           0.7,
        'p_test':       0.0,
        'noise_param':  'label_flip',
        'noise_max':    0.25,
        'noise_step':   0.05,
        'param_label':  'p_correct',
        'x_label':      'Label flip rate',
        'max_length':   512,
        'n_classes':    2,
    },
    'nlp_amazon_sentiment_selection': {
        'p1':           0.9,
        'p2':           0.7,
        'p_test':       0.0,
        'noise_param':  'label_flip',
        'noise_max':    0.25,
        'noise_step':   0.05,
        'param_label':  'p_select',
        'x_label':      'Label flip rate',
        'max_length':   512,
        'n_classes':    2,
    },
    'nlp_amazon_conf_varying_proxy': {
        'p1':           0.01,
        'p2':           0.10,
        'p_test':       1.0,
        'noise_param':  'gamma',
        'noise_max':    1.0,
        'noise_step':   0.1,
        'param_label':  'a_train',
        'x_label':      'Gamma (force C→Y)',
        'max_length':   512,
        'n_classes':    2,
    },
}

_SLUG_MAP = {
    'nlp_agnews_semi_anti_causal':        'causal_agnews_sac',
    'nlp_agnews_size_selection':           'causal_agnews_size_selection',
    'nlp_agnews_conf_varying_proxy':       'causal_agnews_conf_proxy',
    'nlp_imdb_genres_size_selection':      'ac_imdb_genres_size_selection',
    'nlp_imdb_genres_semi_anti_causal':    'ac_imdb_genres_sac',
    'nlp_imdb_genres_conf_varying_proxy':  'ac_imdb_genres_conf_proxy',
    'nlp_amazon_semi_anti_causal':         'causal_amazon_sac',
    'nlp_amazon_sentiment_selection':       'causal_amazon_sentiment_selection',
    'nlp_amazon_conf_varying_proxy':        'causal_amazon_conf_proxy',
}


# =============================================================================
# Parser
# =============================================================================
def make_noise_sweep_nlp_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Noise sweep for NLP datasets (AG News, IMDB Genres, Amazon)"
    )

    p.add_argument('--dataset', required=True, choices=list(NLP_NOISE_DEFAULTS.keys()))

    # ── Noise sweep params ──
    p.add_argument('--p1',          type=float, default=None,
                   help='Paramètre env 1 (fixé). Défaut selon dataset.')
    p.add_argument('--p2',          type=float, default=None,
                   help='Paramètre env 2 (fixé). Défaut selon dataset.')
    p.add_argument('--p_test',      type=float, default=None,
                   help='Paramètre test OOD (fixé). Défaut selon dataset.')
    p.add_argument('--noise_max',   type=float, default=None,
                   help='Valeur maximale du bruit. Défaut selon dataset.')
    p.add_argument('--noise_step',  type=float, default=None,
                   help='Pas du balayage. Défaut selon dataset.')

    # ── BERT ──
    p.add_argument('--nlp_bert_model',       type=str,   default='distilbert-base-uncased')
    p.add_argument('--nlp_max_length',        type=int,   default=None)
    p.add_argument('--nlp_pooling',           type=str,   default='mean', choices=['mean', 'cls'])
    p.add_argument('--finetune_bert_layers',  type=int,   default=0)

    # ── NLP data params ──
    p.add_argument('--nlp_conf_p_c_flip',         type=float, default=0.5)
    p.add_argument('--nlp_amazon_n_target',        type=int,   default=100_000)
    p.add_argument('--nlp_size_threshold_method',  type=str,   default='quartile',
                   choices=['quartile', 'median', 'soft'])

    # ── AG News class distribution ──
    p.add_argument('--nlp_agnews_class_dist_train', type=float, nargs='+', default=None)
    p.add_argument('--nlp_agnews_class_dist_test',  type=float, nargs='+', default=None)

    # ── Training ──
    p.add_argument('--erm_steps',   type=int,   default=25_000)
    p.add_argument('--erm_lr',      type=float, default=1e-4)
    p.add_argument('--irm_steps',   type=int,   default=25_000)
    p.add_argument('--irm_lr',      type=float, default=1e-4)
    p.add_argument('--irm_lambda',  type=float, default=100.0)
    p.add_argument('--eval_every',  type=int,   default=20)
    p.add_argument('--seed',        type=int,   default=1)
    p.add_argument('--seeds',       type=int,   nargs='+', default=None,
                   help='List of seeds (e.g. 0 1 2). Overrides --seed.')
    p.add_argument('--device',      type=str,   default='auto')

    # ── Output ──
    p.add_argument('--out_dir', type=str, default=None)

    # ── Plot only ──
    p.add_argument('--plot_only', action='store_true',
                   help='Relit noise_sweep_results.json depuis --out_dir et retrace uniquement le plot.')

    return p


# =============================================================================
# Plot
# =============================================================================

def _plot_noise_sweep(out_dir: str, defs: dict, results: list, seeds: list,
                     noise_step: float) -> None:
    n_vals   = [r['noise']          for r in results]
    erm_mean = [r['erm_test_mean']  for r in results]
    erm_std  = [r['erm_test_std']   for r in results]
    irm_mean = [r['irm_test_mean']  for r in results]
    irm_std  = [r['irm_test_std']   for r in results]
    n_seeds  = len(seeds)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: OOD accuracy
    ax = axes[0]
    ax.plot(n_vals, erm_mean, 'o-', color='orange',    linewidth=2.5, markersize=7, label='ERM (OOD)')
    ax.fill_between(n_vals,
                    [m - s for m, s in zip(erm_mean, erm_std)],
                    [m + s for m, s in zip(erm_mean, erm_std)],
                    alpha=0.18, color='orange')
    ax.plot(n_vals, irm_mean, 's-', color='steelblue', linewidth=2.5, markersize=7, label='IRM (OOD)')
    ax.fill_between(n_vals,
                    [m - s for m, s in zip(irm_mean, irm_std)],
                    [m + s for m, s in zip(irm_mean, irm_std)],
                    alpha=0.18, color='steelblue')
    ax.set_xlabel(defs['x_label'], fontsize=18)
    ax.set_ylabel('OOD Test Accuracy', fontsize=18)
    ax.set_title('Final OOD accuracy', fontsize=18)
    ax.set_xticks(n_vals)
    ax.set_xticklabels([f'{x:.3f}' for x in n_vals], rotation=45, ha='right', fontsize=22)
    ax.tick_params(axis='y', labelsize=22)
    all_acc = erm_mean + irm_mean
    margin  = max(0.02, (max(all_acc) - min(all_acc)) * 0.15)
    ax.set_ylim(max(0.0, min(all_acc) - margin), min(1.02, max(all_acc) + margin))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    for xi, ey, iy in zip(n_vals, erm_mean, irm_mean):
        ax.annotate(f'{ey:.3f}', (xi, ey), textcoords='offset points',
                    xytext=(0, 8),  ha='center', fontsize=7, color='orange')
        ax.annotate(f'{iy:.3f}', (xi, iy), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=7, color='steelblue')

    # Right: IRM advantage
    ax2 = axes[1]
    delta      = [i - e for i, e in zip(irm_mean, erm_mean)]
    delta_std  = [np.sqrt(si**2 + se**2) for si, se in zip(irm_std, erm_std)]
    bar_colors = ['steelblue' if d >= 0 else 'tomato' for d in delta]
    bars = ax2.bar(n_vals, delta, width=noise_step * 0.7,
                   color=bar_colors, alpha=0.8, edgecolor='white')
    ax2.errorbar(n_vals, delta, yerr=delta_std, fmt='none',
                 color='black', capsize=4, linewidth=1.2)
    ax2.axhline(0, color='black', linewidth=1.2)
    ax2.set_xlabel(defs['x_label'], fontsize=18)
    ax2.set_ylabel('IRM advantage (IRM − ERM)', fontsize=18)
    ax2.set_title('IRM gain over ERM', fontsize=18)
    ax2.set_xticks(n_vals)
    ax2.set_xticklabels([f'{x:.3f}' for x in n_vals], rotation=45, ha='right', fontsize=22)
    ax2.tick_params(axis='y', labelsize=22)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, d in zip(bars, delta):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 d + (0.003 if d >= 0 else -0.008),
                 f'{d:+.3f}', ha='center',
                 va='bottom' if d >= 0 else 'top', fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'noise_sweep.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.close()


# =============================================================================
# Build environments for a given dataset + noise value
# =============================================================================
def _build_envs(args, noise: float):
    p1   = args.p1
    p2   = args.p2
    test = args.p_test
    bert = args.nlp_bert_model
    mxlen = args.nlp_max_length
    pool  = args.nlp_pooling
    dev   = args.device_str
    ft    = args.finetune_bert_layers

    # AG News class distribution parsing (flat list → per-env lists)
    agnews_cd_train = None
    agnews_cd_test  = args.nlp_agnews_class_dist_test
    if args.nlp_agnews_class_dist_train is not None:
        flat = args.nlp_agnews_class_dist_train
        if len(flat) % 4 != 0:
            raise ValueError(f"--nlp_agnews_class_dist_train needs a multiple of 4 floats, got {len(flat)}")
        agnews_cd_train = [flat[k:k+4] for k in range(0, len(flat), 4)]

    if args.dataset == 'nlp_agnews_semi_anti_causal':
        return build_envs_ag_news_semi_anti_causal(
            train_p_correct=[p1, p2], test_p_correct=test,
            seed=args.seed, label_flip=noise,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            class_dist_train=agnews_cd_train, class_dist_test=agnews_cd_test,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_agnews_size_selection':
        return build_envs_ag_news_size_selection(
            train_p_select=[p1, p2],
            seed=args.seed, label_flip=noise,
            threshold_method=args.nlp_size_threshold_method,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            class_dist_train=agnews_cd_train, class_dist_test=agnews_cd_test,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_agnews_conf_varying_proxy':
        return build_envs_ag_news_conf_varying_proxy(
            a_train=[p1, p2], a_test=test,
            seed=args.seed, p_c_flip=args.nlp_conf_p_c_flip, gamma=noise,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_imdb_genres_size_selection':
        return build_envs_imdb_genres_size_selection(
            train_p_select=[p1, p2],
            seed=args.seed, label_flip=noise,
            threshold_method=args.nlp_size_threshold_method,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_imdb_genres_semi_anti_causal':
        return build_envs_imdb_genres_semi_anti_causal(
            train_p_correct=[p1, p2], test_p_correct=test,
            seed=args.seed, label_flip=noise,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_imdb_genres_conf_varying_proxy':
        return build_envs_imdb_genres_conf_varying_proxy(
            a_train=[p1, p2], a_test=test,
            seed=args.seed, p_c_flip=args.nlp_conf_p_c_flip, gamma=noise,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_amazon_semi_anti_causal':
        return build_envs_amazon_semi_anti_causal(
            train_p_correct=[p1, p2], test_p_correct=test,
            seed=args.seed, label_flip=noise,
            n_target=args.nlp_amazon_n_target,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_amazon_sentiment_selection':
        return build_envs_amazon_sentiment_selection(
            train_p_select=[p1, p2],
            seed=args.seed, label_flip=noise,
            n_target=args.nlp_amazon_n_target,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_amazon_conf_varying_proxy':
        return build_envs_amazon_conf_varying_proxy(
            a_train=[p1, p2], a_test=test,
            seed=args.seed, p_c_flip=args.nlp_conf_p_c_flip, gamma=noise,
            n_target=args.nlp_amazon_n_target,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    raise ValueError(f"Unsupported dataset for noise sweep: {args.dataset}")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    args = make_noise_sweep_nlp_parser().parse_args()
    defs = NLP_NOISE_DEFAULTS[args.dataset]

    # Apply dataset defaults where not overridden
    if args.p1             is None: args.p1             = defs['p1']
    if args.p2             is None: args.p2             = defs['p2']
    if args.p_test         is None: args.p_test         = defs['p_test']
    if args.noise_max      is None: args.noise_max      = defs['noise_max']
    if args.noise_step     is None: args.noise_step     = defs['noise_step']
    if args.nlp_max_length is None: args.nlp_max_length = defs['max_length']

    n_classes   = defs['n_classes']
    noise_param = defs['noise_param']

    if args.out_dir is None:
        if args.plot_only:
            raise ValueError(
                "--plot_only nécessite --out_dir pointant vers un dossier existant "
                "contenant noise_sweep_results.json\n"
                "Exemple : --out_dir nlp_synthetic/plots/noise_sweep/causal_amazon_sac/20260520_152427"
            )
        _slug = _SLUG_MAP.get(args.dataset, args.dataset)
        _ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.out_dir = str(_Path(__file__).parent / 'plots' / 'noise_sweep' / _slug / _ts)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Mode plot_only : relit le JSON et retrace sans entraîner ─────────────
    if args.plot_only:
        json_path = os.path.join(args.out_dir, 'noise_sweep_results.json')
        try:
            with open(json_path) as f:
                saved = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"--plot_only : fichier introuvable : {json_path}\n"
                f"Vérifie que --out_dir pointe vers un dossier contenant noise_sweep_results.json"
            ) from None
        except json.JSONDecodeError as e:
            raise ValueError(
                f"--plot_only : fichier JSON invalide ou incomplet : {json_path}\n"
                f"Détail : {e}"
            ) from None
        _plot_noise_sweep(
            out_dir=args.out_dir,
            defs=defs,
            results=saved['aggregated'],
            seeds=saved['seeds'],
            noise_step=args.noise_step,
        )
        import sys as _sys; _sys.exit(0)

    device = resolve_device(args.device)
    args.device_str = str(device)

    n_steps = round(args.noise_max / args.noise_step)
    noises  = [round(i * args.noise_step, 10) for i in range(n_steps + 1)]

    print(f"{'='*62}")
    print(f"NLP Noise sweep – {args.dataset}")
    print(f"  noise param  : {noise_param}")
    print(f"  [{defs['param_label']}] fixed at p1={args.p1}, p2={args.p2}, test={args.p_test}")
    print(f"  noise values = {noises}")
    print(f"  n_classes    = {n_classes}")
    print(f"  BERT         = {args.nlp_bert_model}  |  max_length = {args.nlp_max_length}")
    print(f"  ERM steps    = {args.erm_steps}  lr = {args.erm_lr}")
    print(f"  IRM steps    = {args.irm_steps}  lr = {args.irm_lr}  λ = {args.irm_lambda}")
    print(f"{'='*62}\n")

    seeds = args.seeds if args.seeds is not None else [args.seed]
    raw_results: dict = {}

    for seed in seeds:
        print(f"\n{'#'*62}")
        print(f"# Seed {seed}")
        print(f"{'#'*62}")
        seed_results = []
        args.seed = seed

        for noise in noises:
            print(f"\n-- {noise_param} = {noise:.4f} --")

            train_envs, val_envs, test_env = _build_envs(args, noise)

            # ── ERM ──
            print("  [ERM]", end=" ", flush=True)
            _, erm_hist = train_erm(
                envs=train_envs, val_envs=val_envs, test_env=test_env,
                steps=args.erm_steps, lr=args.erm_lr, batch=512,
                seed=seed, device=device, eval_every=args.eval_every,
                dataset_name=args.dataset, n_classes=n_classes,
            )
            erm_test_final = erm_hist['test_acc'][-1] if erm_hist['test_acc'] else float('nan')
            erm_test_best  = max(erm_hist['test_acc']) if erm_hist['test_acc'] else float('nan')
            print(f"test_OOD={erm_test_final:.3f}  (best={erm_test_best:.3f})")

            # ── IRM ──
            print("  [IRM]", end=" ", flush=True)
            _, irm_hist = train_irm(
                envs=train_envs, val_envs=val_envs, test_env=test_env,
                steps=args.irm_steps, lr=args.irm_lr, batch=512,
                irm_lambda=args.irm_lambda,
                seed=seed, device=device, eval_every=args.eval_every,
                dataset_name=args.dataset, n_classes=n_classes,
            )
            irm_test_final = irm_hist['test_acc'][-1] if irm_hist['test_acc'] else float('nan')
            irm_test_best  = max(irm_hist['test_acc']) if irm_hist['test_acc'] else float('nan')
            print(f"test_OOD={irm_test_final:.3f}  (best={irm_test_best:.3f})")

            seed_results.append({
                'noise':          noise,
                'seed':           seed,
                'erm_test_final': erm_test_final,
                'erm_test_best':  erm_test_best,
                'irm_test_final': irm_test_final,
                'irm_test_best':  irm_test_best,
            })

            del _
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()

        raw_results[seed] = seed_results

    # Aggregate across seeds
    results = []
    for i, noise in enumerate(noises):
        erm_vals = [raw_results[s][i]['erm_test_final'] for s in seeds]
        irm_vals = [raw_results[s][i]['irm_test_final'] for s in seeds]
        results.append({
            'noise':           noise,
            'erm_test_mean':   float(np.mean(erm_vals)),
            'erm_test_std':    float(np.std(erm_vals, ddof=0)),
            'irm_test_mean':   float(np.mean(irm_vals)),
            'irm_test_std':    float(np.std(irm_vals, ddof=0)),
        })

    # Save JSON
    json_path = os.path.join(args.out_dir, 'noise_sweep_results.json')
    with open(json_path, 'w') as f:
        json.dump({'seeds': seeds, 'noise_param': noise_param,
                   'aggregated': results, 'raw': raw_results}, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # Console summary
    n_seeds = len(seeds)
    print(f"\n{'='*72}")
    print(f"Seeds : {seeds}  |  noise param : {noise_param}")
    print(f"{'Noise':>7} | {'ERM mean':>9} | {'ERM std':>7} | {'IRM mean':>9} | {'IRM std':>7} | {'Delta':>8}")
    print(f"{'─'*7}-+-{'─'*9}-+-{'─'*7}-+-{'─'*9}-+-{'─'*7}-+-{'─'*8}")
    for r in results:
        delta = r['irm_test_mean'] - r['erm_test_mean']
        print(f"  {r['noise']:.4f} | {r['erm_test_mean']:>9.3f} | {r['erm_test_std']:>7.3f} | "
              f"{r['irm_test_mean']:>9.3f} | {r['irm_test_std']:>7.3f} | {delta:>+8.3f}")
    print(f"{'='*72}")

    # Summary plot
    _plot_noise_sweep(args.out_dir, defs, results, seeds, args.noise_step)

    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
