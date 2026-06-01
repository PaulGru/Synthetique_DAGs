#!/usr/bin/env python3
"""
run_gap_sweep_nlp.py
====================
Sweeps the environment gap (difference in spurious-correlation strength between
train environments) for NLP datasets. At gap=0 both envs are identical;
larger gaps give IRM a stronger invariance signal.

Usage:
    uv run nlp_synthetic/run_gap_sweep_nlp.py --dataset nlp_agnews_semi_anti_causal
    uv run nlp_synthetic/run_gap_sweep_nlp.py --dataset nlp_agnews_conf_varying_proxy --gap_step 0.02
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
# Defaults par dataset
# =============================================================================
NLP_DATASET_DEFAULTS = {
    # ── AG News ──────────────────────────────────────────────────────────────
    'nlp_agnews_semi_anti_causal': {
        'gap_center':   0.85,
        'gap_max':      0.20,
        'gap_step':     0.04,
        'gap_test':     0.0,
        'param_label':  'p_correct',
        'x_label':      'Gap Δp_correct = p₂ − p₁',
        'max_length':   256,
        'label_flip':   0.25,
        'n_classes':    4,
    },
    'nlp_agnews_size_selection': {
        'gap_center':   0.85,
        'gap_max':      0.20,
        'gap_step':     0.04,
        'gap_test':     0.0,
        'param_label':  'p_select',
        'x_label':      'Gap Δp_select = p₂ − p₁',
        'max_length':   256,
        'label_flip':   0.25,
        'n_classes':    4,
    },
    'nlp_agnews_conf_varying_proxy': {
        'gap_center':   0.05,
        'gap_max':      0.18,
        'gap_step':     0.03,
        'gap_test':     1.0,
        'param_label':  'a_train',
        'x_label':      'Gap Δa = a₂ − a₁',
        'max_length':   256,
        'label_flip':   0.0,
        'n_classes':    4,
    },
    # ── IMDB Genres ──────────────────────────────────────────────────────────
    'nlp_imdb_genres_size_selection': {
        'gap_center':   0.85,
        'gap_max':      0.20,
        'gap_step':     0.04,
        'gap_test':     0.0,
        'param_label':  'p_select',
        'x_label':      'Gap Δp_select = p₂ − p₁',
        'max_length':   256,
        'label_flip':   0.25,
        'n_classes':    2,
    },
    'nlp_imdb_genres_semi_anti_causal': {
        'gap_center':   0.85,
        'gap_max':      0.20,
        'gap_step':     0.04,
        'gap_test':     0.0,
        'param_label':  'p_correct',
        'x_label':      'Gap Δp_correct = p₂ − p₁',
        'max_length':   256,
        'label_flip':   0.25,
        'n_classes':    2,
    },
    'nlp_imdb_genres_conf_varying_proxy': {
        'gap_center':   0.05,
        'gap_max':      0.18,
        'gap_step':     0.03,
        'gap_test':     1.0,
        'param_label':  'a_train',
        'x_label':      'Gap Δa = a₂ − a₁',
        'max_length':   256,
        'label_flip':   0.0,
        'n_classes':    2,
    },
    # ── Amazon ───────────────────────────────────────────────────────────────
    'nlp_amazon_semi_anti_causal': {
        'gap_center':   0.85,
        'gap_max':      0.20,
        'gap_step':     0.04,
        'gap_test':     0.0,
        'param_label':  'p_correct',
        'x_label':      'Gap Δp_correct = p₂ − p₁',
        'max_length':   512,
        'label_flip':   0.25,
        'n_classes':    2,
    },
    'nlp_amazon_sentiment_selection': {
        'gap_center':   0.85,
        'gap_max':      0.20,
        'gap_step':     0.04,
        'gap_test':     0.0,
        'param_label':  'p_select',
        'x_label':      'Gap Δp_select = p₂ − p₁',
        'max_length':   512,
        'label_flip':   0.25,
        'n_classes':    2,
    },
    'nlp_amazon_conf_varying_proxy': {
        'gap_center':   0.05,
        'gap_max':      0.18,
        'gap_step':     0.03,
        'gap_test':     1.0,
        'param_label':  'a_train',
        'x_label':      'Gap Δa = a₂ − a₁',
        'max_length':   512,
        'label_flip':   0.0,
        'n_classes':    2,
    },
}

# =============================================================================
# Parser
# =============================================================================
def make_gap_sweep_nlp_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Gap sweep for NLP datasets (AG News, IMDB Genres, Amazon)"
    )

    p.add_argument('--dataset', required=True, choices=list(NLP_DATASET_DEFAULTS.keys()))

    # ── Gap sweep params ──
    p.add_argument('--gap_center', type=float, default=None)
    p.add_argument('--gap_max',    type=float, default=None)
    p.add_argument('--gap_step',   type=float, default=None)
    p.add_argument('--gap_test',   type=float, default=None)

    # ── BERT ──
    p.add_argument('--nlp_bert_model',  type=str, default='distilbert-base-uncased')
    p.add_argument('--nlp_max_length',  type=int, default=None)
    p.add_argument('--nlp_pooling',     type=str, default='mean', choices=['mean', 'cls'])
    p.add_argument('--finetune_bert_layers', type=int, default=0)

    # ── NLP data params ──
    p.add_argument('--nlp_label_flip', type=float, default=None)
    p.add_argument('--nlp_conf_p_c_flip', type=float, default=0.25)
    p.add_argument('--nlp_conf_gamma', type=float, default=0.5)
    p.add_argument('--nlp_amazon_n_target', type=int, default=100_000)
    p.add_argument('--nlp_size_threshold_method', type=str, default='quartile',
                   choices=['quartile', 'median', 'soft'])

    # ── AG News class distribution ──
    p.add_argument('--nlp_agnews_class_dist_train', type=float, nargs='+', default=None)
    p.add_argument('--nlp_agnews_class_dist_test',  type=float, nargs='+', default=None)

    # ── Training ──
    p.add_argument('--erm_steps',  type=int,   default=25_000)
    p.add_argument('--erm_lr',     type=float, default=1e-4)
    p.add_argument('--irm_steps',  type=int,   default=25_000)
    p.add_argument('--irm_lr',     type=float, default=1e-4)
    p.add_argument('--irm_lambda', type=float, default=100.0)
    p.add_argument('--eval_every', type=int,   default=20)
    p.add_argument('--seed',       type=int,   default=1)
    p.add_argument('--seeds',      type=int,   nargs='+', default=None,
                   help='List of seeds to run (e.g. 0 1 2). Overrides --seed.')
    p.add_argument('--device',     type=str,   default='auto')

    # ── Output ──
    p.add_argument('--out_dir', type=str, default=None)

    return p


# =============================================================================
# Construction des envs selon dataset + gap
# =============================================================================
def _build_envs(args, p1: float, p2: float):
    """Build (train_envs, val_envs, test_env) for a given dataset and gap."""
    g_test = args.gap_test
    bert   = args.nlp_bert_model
    mxlen  = args.nlp_max_length
    pool   = args.nlp_pooling
    dev    = args.device_str
    flip   = args.nlp_label_flip
    ft     = args.finetune_bert_layers

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
            train_p_correct=[p1, p2],
            test_p_correct=g_test,
            seed=args.seed,
            label_flip=flip,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            class_dist_train=agnews_cd_train,
            class_dist_test=agnews_cd_test,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_agnews_size_selection':
        return build_envs_ag_news_size_selection(
            train_p_select=[p1, p2],
            seed=args.seed,
            label_flip=flip,
            threshold_method=args.nlp_size_threshold_method,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            class_dist_train=agnews_cd_train,
            class_dist_test=agnews_cd_test,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_agnews_conf_varying_proxy':
        return build_envs_ag_news_conf_varying_proxy(
            a_train=[p1, p2],
            a_test=g_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            gamma=args.nlp_conf_gamma,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_imdb_genres_size_selection':
        return build_envs_imdb_genres_size_selection(
            train_p_select=[p1, p2],
            seed=args.seed,
            threshold_method=args.nlp_size_threshold_method,
            label_flip=flip,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_imdb_genres_semi_anti_causal':
        return build_envs_imdb_genres_semi_anti_causal(
            train_p_correct=[p1, p2],
            test_p_correct=g_test,
            seed=args.seed,
            label_flip=flip,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_imdb_genres_conf_varying_proxy':
        return build_envs_imdb_genres_conf_varying_proxy(
            a_train=[p1, p2],
            a_test=g_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            gamma=args.nlp_conf_gamma,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_amazon_semi_anti_causal':
        return build_envs_amazon_semi_anti_causal(
            train_p_correct=[p1, p2],
            test_p_correct=g_test,
            seed=args.seed,
            label_flip=flip,
            n_target=args.nlp_amazon_n_target,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_amazon_sentiment_selection':
        return build_envs_amazon_sentiment_selection(
            train_p_select=[p1, p2],
            seed=args.seed,
            label_flip=flip,
            n_target=args.nlp_amazon_n_target,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    elif args.dataset == 'nlp_amazon_conf_varying_proxy':
        return build_envs_amazon_conf_varying_proxy(
            a_train=[p1, p2],
            a_test=g_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            gamma=args.nlp_conf_gamma,
            n_target=args.nlp_amazon_n_target,
            bert_model=bert, max_length=mxlen, device=dev, pooling=pool,
            finetune_bert_layers=ft,
        )

    raise ValueError(f"Unsupported dataset: {args.dataset}")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    args = make_gap_sweep_nlp_parser().parse_args()
    defs = NLP_DATASET_DEFAULTS[args.dataset]

    # Apply dataset defaults where not overridden
    if args.gap_center     is None: args.gap_center     = defs['gap_center']
    if args.gap_max        is None: args.gap_max         = defs['gap_max']
    if args.gap_step       is None: args.gap_step        = defs['gap_step']
    if args.gap_test       is None: args.gap_test        = defs['gap_test']
    if args.nlp_max_length is None: args.nlp_max_length  = defs['max_length']
    if args.nlp_label_flip is None: args.nlp_label_flip  = defs['label_flip']

    n_classes = defs['n_classes']

    if args.out_dir is None:
        from datetime import datetime
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
        _slug = _SLUG_MAP.get(args.dataset, args.dataset)
        _ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.out_dir = str(_Path(__file__).parent / 'plots' / 'gap_sweep' / _slug / _ts)

    os.makedirs(args.out_dir, exist_ok=True)

    device = resolve_device(args.device)
    args.device_str = str(device)

    n_steps = round(args.gap_max / args.gap_step)
    gaps    = [round(i * args.gap_step, 10) for i in range(n_steps + 1)]

    print(f"{'='*62}")
    print(f"NLP Gap sweep – {args.dataset}")
    print(f"  param     : {defs['param_label']}")
    print(f"  center    = {args.gap_center}  |  test_OOD = {args.gap_test}")
    print(f"  gaps      = {gaps}")
    print(f"  n_classes = {n_classes}")
    print(f"  BERT      = {args.nlp_bert_model}  |  max_length = {args.nlp_max_length}")
    print(f"  ERM steps = {args.erm_steps}  lr = {args.erm_lr}")
    print(f"  IRM steps = {args.irm_steps}  lr = {args.irm_lr}  λ = {args.irm_lambda}")
    print(f"{'='*62}\n")

    seeds = args.seeds if args.seeds is not None else [args.seed]
    # raw_results[seed] = list of per-gap dicts
    raw_results: dict = {}

    for seed in seeds:
        print(f"\n{'#'*62}")
        print(f"# Seed {seed}")
        print(f"{'#'*62}")
        seed_results = []

        for gap in gaps:
            p1 = max(0.0, min(1.0, round(args.gap_center - gap / 2.0, 10)))
            p2 = max(0.0, min(1.0, round(args.gap_center + gap / 2.0, 10)))
            print(f"\n-- gap={gap:.4f}  [{defs['param_label']}]=[{p1:.4f}, {p2:.4f}] --")

            train_envs, val_envs, test_env = _build_envs(args, p1, p2)

            # ERM
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

            # IRM
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
                'gap':            gap,
                'p1':             p1,
                'p2':             p2,
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
    for i, gap in enumerate(gaps):
        erm_vals = [raw_results[s][i]['erm_test_final'] for s in seeds]
        irm_vals = [raw_results[s][i]['irm_test_final'] for s in seeds]
        p1 = raw_results[seeds[0]][i]['p1']
        p2 = raw_results[seeds[0]][i]['p2']
        results.append({
            'gap':             gap,
            'p1':              p1,
            'p2':              p2,
            'erm_test_mean':   float(np.mean(erm_vals)),
            'erm_test_std':    float(np.std(erm_vals, ddof=0)),
            'irm_test_mean':   float(np.mean(irm_vals)),
            'irm_test_std':    float(np.std(irm_vals, ddof=0)),
            'erm_test_final':  float(np.mean(erm_vals)),
            'irm_test_final':  float(np.mean(irm_vals)),
        })

    # Save JSON
    json_path = os.path.join(args.out_dir, 'gap_sweep_results.json')
    with open(json_path, 'w') as f:
        json.dump({'seeds': seeds, 'aggregated': results, 'raw': raw_results}, f, indent=2)
    print(f"\nResults saved: {json_path}")

    # Console summary
    param_lbl = defs['param_label']
    n_seeds   = len(seeds)
    print(f"\n{'='*72}")
    print(f"Seeds : {seeds}")
    print(f"{'Gap':>6} | {param_lbl+'₁':>10} | {param_lbl+'₂':>10} | {'ERM mean':>9} | {'ERM std':>7} | {'IRM mean':>9} | {'IRM std':>7} | {'Δ mean':>8}")
    print(f"{'─'*6}-+-{'─'*10}-+-{'─'*10}-+-{'─'*9}-+-{'─'*7}-+-{'─'*9}-+-{'─'*7}-+-{'─'*8}")
    for r in results:
        delta = r['irm_test_mean'] - r['erm_test_mean']
        print(f"  {r['gap']:.3f} | {r['p1']:>10.4f} | {r['p2']:>10.4f} | "
              f"{r['erm_test_mean']:>9.3f} | {r['erm_test_std']:>7.3f} | "
              f"{r['irm_test_mean']:>9.3f} | {r['irm_test_std']:>7.3f} | {delta:>+8.3f}")
    print(f"{'='*72}")

    # Summary plot
    g        = [r['gap']           for r in results]
    erm_mean = [r['erm_test_mean']  for r in results]
    erm_std  = [r['erm_test_std']   for r in results]
    irm_mean = [r['irm_test_mean']  for r in results]
    irm_std  = [r['irm_test_std']   for r in results]
    n_seeds  = len(seeds)
    seed_lbl = f'(mean ± std, {n_seeds} seeds)'

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Gap Sweep – {args.dataset}\n{seed_lbl}', fontsize=14)

    ax = axes[0]
    ax.plot(g, erm_mean, 'o-', color='orange', linewidth=2.5, markersize=7, label='ERM (OOD)')
    ax.fill_between(g,
                    [m - s for m, s in zip(erm_mean, erm_std)],
                    [m + s for m, s in zip(erm_mean, erm_std)],
                    alpha=0.18, color='orange')
    ax.plot(g, irm_mean, 's-', color='steelblue', linewidth=2.5, markersize=7, label='IRM (OOD)')
    ax.fill_between(g,
                    [m - s for m, s in zip(irm_mean, irm_std)],
                    [m + s for m, s in zip(irm_mean, irm_std)],
                    alpha=0.18, color='steelblue')
    ax.set_xlabel(defs['x_label'])
    ax.set_ylabel('Test OOD accuracy')
    ax.set_title('Final OOD accuracy')
    ax.set_xticks(g)
    ax.set_xticklabels([f'{x:.3f}' for x in g], rotation=45, ha='right')
    all_acc = erm_mean + irm_mean
    margin = max(0.02, (max(all_acc) - min(all_acc)) * 0.15)
    ax.set_ylim(max(0.0, min(all_acc) - margin), min(1.02, max(all_acc) + margin))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    for xi, ey, iy in zip(g, erm_mean, irm_mean):
        ax.annotate(f'{ey:.3f}', (xi, ey), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=8, color='orange')
        ax.annotate(f'{iy:.3f}', (xi, iy), textcoords='offset points',
                    xytext=(0, -14), ha='center', fontsize=8, color='steelblue')

    ax2 = axes[1]
    delta     = [i - e for i, e in zip(irm_mean, erm_mean)]
    delta_std = [np.sqrt(si**2 + se**2) for si, se in zip(irm_std, erm_std)]
    bar_colors = ['steelblue' if d >= 0 else 'tomato' for d in delta]
    bars = ax2.bar(g, delta, width=args.gap_step * 0.7, color=bar_colors, alpha=0.8, edgecolor='white')
    ax2.errorbar(g, delta, yerr=delta_std, fmt='none', color='black', capsize=4, linewidth=1.2)
    ax2.axhline(0, color='black', linewidth=1.2)
    ax2.set_xlabel(defs['x_label'])
    ax2.set_ylabel('IRM advantage (IRM − ERM)')
    ax2.set_title('IRM gain over ERM')
    ax2.set_xticks(g)
    ax2.set_xticklabels([f'{x:.3f}' for x in g], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, d in zip(bars, delta):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 d + (0.003 if d >= 0 else -0.008),
                 f'{d:+.3f}', ha='center',
                 va='bottom' if d >= 0 else 'top', fontsize=8)

    fig_path = os.path.join(args.out_dir, 'gap_sweep.png')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved: {fig_path}")
