#!/usr/bin/env python3
"""
run_grand_test.py
=================
Full grid: 3 datasets x 3 mechanisms x 4 noise levels x 2 spurious-correlation
strengths = 72 runs total.

Output structure:
    plots/grand_test/{run_name}/
        {dataset_slug}/
            noise{N}__{corr_tag}/
                results.json   <- ERM / IRM metrics (final + best, per seed)
                config.json    <- exact run parameters

Grid:
  +---------------+----------------------------+------------------------------+
  | Mechanism     | Noise levels               | Spurious correlations        |
  +---------------+----------------------------+------------------------------+
  | SAC           | label_flip in              | p=[1.0,0.8]  p=[0.9,0.7]    |
  | Selection     |   {0, 0.1, 0.2, 0.25}      |                              |
  +---------------+----------------------------+------------------------------+
  | Confounder    | gamma in                   | a=[0.01,0.1] a=[0.05,0.15]  |
  |               |   {0, 0.1, 0.2, 0.25}      |                              |
  +---------------+----------------------------+------------------------------+

Usage:
    # Run everything
    uv run nlp_synthetic/run_grand_test.py --device cuda:0

    # Resume an interrupted run (skips cells where results.json already exists)
    uv run nlp_synthetic/run_grand_test.py --device cuda:0 --resume

    # Filter a subset of datasets
    uv run nlp_synthetic/run_grand_test.py --datasets nlp_agnews_semi_anti_causal nlp_imdb_genres_conf_varying_proxy

    # Multiple seeds
    uv run nlp_synthetic/run_grand_test.py --seeds 0 1 2

    # Custom run name (output folder)
    uv run nlp_synthetic/run_grand_test.py --run_name my_grand_test
"""

from pathlib import Path, Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]

import os
import json
import argparse
from datetime import datetime

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data.nlp_datasets import (
    build_envs_ag_news_semi_anti_causal,
    build_envs_ag_news_size_selection,
    build_envs_ag_news_conf_varying_proxy,
    build_envs_imdb_genres_semi_anti_causal,
    build_envs_imdb_genres_size_selection,
    build_envs_imdb_genres_conf_varying_proxy,
    build_envs_amazon_semi_anti_causal,
    build_envs_amazon_sentiment_selection,
    build_envs_amazon_conf_varying_proxy,
)
from core.training import train_erm, train_irm
from core.evaluation import resolve_device


# =============================================================================
# Experiment grid
# =============================================================================

# 4 noise levels
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.25]

# Spurious correlation strengths per mechanism
CORR_SAC_SELECTION = [(0.9, 0.7)]
CORR_CONF          = [(0.01, 0.1)]

# IRM penalty coefficients explored per run
IRM_LAMBDAS = [50, 75, 100, 125]

# Fixed configuration per dataset
_DATASET_CFG = {
    # ── AG News (4 classes) ──────────────────────────────────────────────────
    'nlp_agnews_semi_anti_causal': {
        'slug':       'agnews_sac',
        'mechanism':  'sac',
        'p_test':     0.0,
        'n_classes':  4,
        'max_length': 256,
        'agnews_class_dist_train': [[0.25]*4, [0.25]*4],
        'agnews_class_dist_test':  [0.25]*4,
    },
    'nlp_agnews_size_selection': {
        'slug':       'agnews_size_selection',
        'mechanism':  'selection',
        'p_test':     0.0,
        'n_classes':  4,
        'max_length': 256,
        'agnews_class_dist_train': [[0.25]*4, [0.25]*4],
        'agnews_class_dist_test':  [0.25]*4,
    },
    'nlp_agnews_conf_varying_proxy': {
        'slug':       'agnews_conf',
        'mechanism':  'conf',
        'p_test':     1.0,
        'n_classes':  4,
        'max_length': 256,
        'p_c_flip':   0.5,
    },
    # ── IMDB Genres (2 classes) ──────────────────────────────────────────────
    'nlp_imdb_genres_semi_anti_causal': {
        'slug':       'imdb_sac',
        'mechanism':  'sac',
        'p_test':     0.0,
        'n_classes':  2,
        'max_length': 256,
    },
    'nlp_imdb_genres_size_selection': {
        'slug':       'imdb_size_selection',
        'mechanism':  'selection',
        'p_test':     0.0,
        'n_classes':  2,
        'max_length': 256,
    },
    'nlp_imdb_genres_conf_varying_proxy': {
        'slug':       'imdb_conf',
        'mechanism':  'conf',
        'p_test':     1.0,
        'n_classes':  2,
        'max_length': 256,
        'p_c_flip':   0.5,
    },
    # ── Amazon Books (2 classes) ─────────────────────────────────────────────
    'nlp_amazon_semi_anti_causal': {
        'slug':       'amazon_sac',
        'mechanism':  'sac',
        'p_test':     0.0,
        'n_classes':  2,
        'max_length': 512,
        'amazon_n_target': 100_000,
    },
    'nlp_amazon_sentiment_selection': {
        'slug':       'amazon_sentiment_selection',
        'mechanism':  'selection',
        'p_test':     0.0,
        'n_classes':  2,
        'max_length': 512,
        'amazon_n_target': 100_000,
    },
    'nlp_amazon_conf_varying_proxy': {
        'slug':       'amazon_conf',
        'mechanism':  'conf',
        'p_test':     1.0,
        'n_classes':  2,
        'max_length': 512,
        'p_c_flip':   0.5,
        'amazon_n_target': 100_000,
    },
}

ALL_DATASETS = list(_DATASET_CFG.keys())


# =============================================================================
# Environment construction
# =============================================================================

def _build_envs(dataset: str, p1: float, p2: float, noise: float,
                seed: int, cfg: dict, device_str: str,
                bert_model: str, pooling: str):
    """Build (train_envs, val_envs, test_env) for a single grid cell."""
    kwargs_common = dict(
        seed=seed,
        bert_model=bert_model,
        max_length=cfg['max_length'],
        device=device_str,
        pooling=pooling,
    )

    if dataset == 'nlp_agnews_semi_anti_causal':
        return build_envs_ag_news_semi_anti_causal(
            train_p_correct=[p1, p2], test_p_correct=cfg['p_test'],
            label_flip=noise,
            class_dist_train=cfg.get('agnews_class_dist_train'),
            class_dist_test=cfg.get('agnews_class_dist_test'),
            **kwargs_common,
        )

    elif dataset == 'nlp_agnews_size_selection':
        return build_envs_ag_news_size_selection(
            train_p_select=[p1, p2], label_flip=noise,
            class_dist_train=cfg.get('agnews_class_dist_train'),
            class_dist_test=cfg.get('agnews_class_dist_test'),
            **kwargs_common,
        )

    elif dataset == 'nlp_agnews_conf_varying_proxy':
        return build_envs_ag_news_conf_varying_proxy(
            a_train=[p1, p2], a_test=cfg['p_test'],
            p_c_flip=cfg.get('p_c_flip', 0.5), gamma=noise,
            **kwargs_common,
        )

    elif dataset == 'nlp_imdb_genres_semi_anti_causal':
        return build_envs_imdb_genres_semi_anti_causal(
            train_p_correct=[p1, p2], test_p_correct=cfg['p_test'],
            label_flip=noise,
            **kwargs_common,
        )

    elif dataset == 'nlp_imdb_genres_size_selection':
        return build_envs_imdb_genres_size_selection(
            train_p_select=[p1, p2], label_flip=noise,
            **kwargs_common,
        )

    elif dataset == 'nlp_imdb_genres_conf_varying_proxy':
        return build_envs_imdb_genres_conf_varying_proxy(
            a_train=[p1, p2], a_test=cfg['p_test'],
            p_c_flip=cfg.get('p_c_flip', 0.5), gamma=noise,
            **kwargs_common,
        )

    elif dataset == 'nlp_amazon_semi_anti_causal':
        return build_envs_amazon_semi_anti_causal(
            train_p_correct=[p1, p2], test_p_correct=cfg['p_test'],
            label_flip=noise,
            n_target=cfg.get('amazon_n_target', 100_000),
            **kwargs_common,
        )

    elif dataset == 'nlp_amazon_sentiment_selection':
        return build_envs_amazon_sentiment_selection(
            train_p_select=[p1, p2], label_flip=noise,
            n_target=cfg.get('amazon_n_target', 100_000),
            **kwargs_common,
        )

    elif dataset == 'nlp_amazon_conf_varying_proxy':
        return build_envs_amazon_conf_varying_proxy(
            a_train=[p1, p2], a_test=cfg['p_test'],
            p_c_flip=cfg.get('p_c_flip', 0.5), gamma=noise,
            n_target=cfg.get('amazon_n_target', 100_000),
            **kwargs_common,
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


# =============================================================================
# Folder naming
# =============================================================================

def _corr_tag(mechanism: str, p1: float, p2: float) -> str:
    """Human-readable tag for the spurious-correlation level."""
    if mechanism == 'conf':
        return f"a{p1:.3f}_{p2:.3f}".replace('.', 'p')
    else:
        return f"p{p1:.2f}_{p2:.2f}".replace('.', 'p')


def _run_dir(root: Path, dataset_slug: str, noise: float,
             mechanism: str, p1: float, p2: float) -> Path:
    noise_tag = f"noise{noise:.3f}".replace('.', 'p')
    corr_tag  = _corr_tag(mechanism, p1, p2)
    return root / dataset_slug / f"{noise_tag}__{corr_tag}"


# =============================================================================
# Training curve visualisation for a grid cell
# =============================================================================

_MODEL_COLORS = {
    'erm':     '#1f77b4',
    'irm_50':  '#ff7f0e',
    'irm_75':  '#2ca02c',
    'irm_100': '#d62728',
    'irm_125': '#9467bd',
}
_MODEL_LABELS = {
    'erm':     'ERM',
    'irm_50':  'IRM λ=50',
    'irm_75':  'IRM λ=75',
    'irm_100': 'IRM λ=100',
    'irm_125': 'IRM λ=125',
}


def _plot_cell_curves(histories: dict, cell_dir: Path, title: str = '') -> None:
    """
    Plot and save loss + OOD accuracy curves for a single run.

    ``histories`` is a dict  {model_key: hist}  where hist has keys
    ``step``, ``loss``, ``test_acc``, ``val_acc``.
    """
    model_keys = list(histories.keys())

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    for key in model_keys:
        h     = histories[key]
        steps = h['step']
        color = _MODEL_COLORS.get(key, None)
        label = _MODEL_LABELS.get(key, key)
        axes[0].plot(steps, h['loss'],     color=color, label=label)
        axes[1].plot(steps, h['val_acc'],  color=color, label=label)
        axes[2].plot(steps, h['test_acc'], color=color, label=label)

    axes[0].set_title('Loss')
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('ID Accuracy (val)')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title('OOD Accuracy (test)')
    axes[2].set_xlabel('Step')
    axes[2].set_ylabel('Accuracy')
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(cell_dir / 'training_curves.png', dpi=120, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# Single run: ERM + IRM on one grid cell
# =============================================================================

def run_cell(
    dataset: str, p1: float, p2: float, noise: float,
    seeds: list, cfg: dict, cell_dir: Path,
    device, bert_model: str, pooling: str,
    erm_steps: int, erm_lr: float,
    irm_steps: int, irm_lr: float, irm_lambdas: list,
    eval_every: int,
):
    mechanism  = cfg['mechanism']
    n_classes  = cfg['n_classes']
    device_str = str(device)

    # Run configuration
    config = {
        'dataset':    dataset,
        'mechanism':  mechanism,
        'p1':         p1,
        'p2':         p2,
        'noise':      noise,
        'noise_param': 'gamma' if mechanism == 'conf' else 'label_flip',
        'seeds':      seeds,
        'erm_steps':  erm_steps,  'erm_lr':  erm_lr,
        'irm_steps':  irm_steps,  'irm_lr':  irm_lr,
        'irm_lambdas': irm_lambdas,
        'eval_every': eval_every,
        'bert_model': bert_model, 'pooling':  pooling,
        'max_length': cfg['max_length'],
        'device':     device_str,
    }
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / 'config.json').write_text(json.dumps(config, indent=2))

    # Helper: extract the relevant keys from a history dict
    def _trim_hist(h):
        return {
            'step':     h['step'],
            'loss':     h['loss'],
            'val_acc':  h['val_acc'],
            'test_acc': h['test_acc'],
        }

    per_seed = []
    # Curves averaged over seeds (for plotting): accumulated per model_key, then averaged
    all_histories: dict[str, list[dict]] = {}

    for seed in seeds:
        print(f"    seed {seed} …", end=" ", flush=True)

        train_envs, val_envs, test_env = _build_envs(
            dataset, p1, p2, noise, seed, cfg, device_str, bert_model, pooling
        )

        seed_row: dict = {'seed': seed}

        # ── ERM ──────────────────────────────────────────────────────────────
        _, erm_hist = train_erm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=erm_steps, lr=erm_lr, batch=512,
            seed=seed, device=device, eval_every=eval_every,
            dataset_name=dataset, n_classes=n_classes,
        )
        erm_final = erm_hist['test_acc'][-1] if erm_hist['test_acc'] else float('nan')
        erm_best  = max(erm_hist['test_acc'])  if erm_hist['test_acc'] else float('nan')
        seed_row['erm'] = {
            'final':   erm_final,
            'best':    erm_best,
            'history': _trim_hist(erm_hist),
        }
        all_histories.setdefault('erm', []).append(_trim_hist(erm_hist))
        del _

        # ── IRM (multiple lambdas) ────────────────────────────────────────────
        irm_parts = []
        for lam in irm_lambdas:
            key = f'irm_{int(lam)}'
            _, irm_hist = train_irm(
                envs=train_envs, val_envs=val_envs, test_env=test_env,
                steps=irm_steps, lr=irm_lr, batch=512,
                irm_lambda=float(lam),
                seed=seed, device=device, eval_every=eval_every,
                dataset_name=dataset, n_classes=n_classes,
            )
            irm_final = irm_hist['test_acc'][-1] if irm_hist['test_acc'] else float('nan')
            irm_best  = max(irm_hist['test_acc'])  if irm_hist['test_acc'] else float('nan')
            seed_row[key] = {
                'final':   irm_final,
                'best':    irm_best,
                'history': _trim_hist(irm_hist),
            }
            all_histories.setdefault(key, []).append(_trim_hist(irm_hist))
            irm_parts.append(f"IRM{int(lam)}={irm_final:.3f}")
            del _

        print(f"ERM={erm_final:.3f}  " + "  ".join(irm_parts))
        per_seed.append(seed_row)

        del train_envs, val_envs, test_env, erm_hist, irm_hist
        if device_str.startswith('cuda'):
            torch.cuda.empty_cache()

    # ── Multi-seed aggregation ────────────────────────────────────────────────
    def _agg_scalar(model_key, metric):
        vals = [r[model_key][metric] for r in per_seed
                if not np.isnan(r[model_key][metric])]
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                'values': vals} if vals else {'mean': float('nan'), 'std': 0.0, 'values': []}

    def _mean_curves(histories_list):
        """Average curves over multiple seeds (common step grid assumed)."""
        steps = histories_list[0]['step']
        return {
            'step':     steps,
            'loss':     [float(np.mean([h['loss'][i]     for h in histories_list])) for i in range(len(steps))],
            'val_acc':  [float(np.mean([h['val_acc'][i]  for h in histories_list])) for i in range(len(steps))],
            'test_acc': [float(np.mean([h['test_acc'][i] for h in histories_list])) for i in range(len(steps))],
        }

    model_keys = ['erm'] + [f'irm_{int(l)}' for l in irm_lambdas]

    summary = {}
    mean_histories = {}
    for key in model_keys:
        summary[key] = {
            'final': _agg_scalar(key, 'final'),
            'best':  _agg_scalar(key, 'best'),
        }
        if all_histories.get(key):
            mean_histories[key] = _mean_curves(all_histories[key])

    # For compatibility with plot_grand_test.py (erm_final, irm_final keys),
    # expose the best IRM (highest final_mean across lambdas)
    best_irm_key = max(
        (k for k in model_keys if k != 'erm'),
        key=lambda k: summary[k]['final']['mean'],
        default=model_keys[1] if len(model_keys) > 1 else 'irm_50',
    )
    summary['erm_final']   = summary['erm']['final']
    summary['erm_best']    = summary['erm']['best']
    summary['irm_final']   = summary[best_irm_key]['final']
    summary['irm_best']    = summary[best_irm_key]['best']
    delta_vals_final = [
        summary[best_irm_key]['final']['values'][i] - summary['erm']['final']['values'][i]
        for i in range(len(summary['erm']['final']['values']))
    ]
    delta_vals_best  = [
        summary[best_irm_key]['best']['values'][i] - summary['erm']['best']['values'][i]
        for i in range(len(summary['erm']['best']['values']))
    ]
    def _agg_list(vals):
        return {'mean': float(np.mean(vals)), 'std': float(np.std(vals)), 'values': vals} \
               if vals else {'mean': float('nan'), 'std': 0.0, 'values': []}
    summary['delta_final'] = _agg_list(delta_vals_final)
    summary['delta_best']  = _agg_list(delta_vals_best)

    results = {
        'config':        config,
        'per_seed':      per_seed,
        'summary':       summary,
        'mean_histories': mean_histories,
    }
    (cell_dir / 'results.json').write_text(json.dumps(results, indent=2))

    # ── Training curves ───────────────────────────────────────────────────────
    mech   = cfg['mechanism']
    noise_str = f"noise={noise:.3f}"
    corr_str  = _corr_tag(mech, p1, p2)
    title = f"{cfg['slug']}  {noise_str}  corr={corr_str}"
    _plot_cell_curves(mean_histories, cell_dir, title=title)

    return results


# =============================================================================
# CLI
# =============================================================================

def make_parser():
    p = argparse.ArgumentParser(
        description="Full experiment grid: 3 datasets x 3 mechanisms x 4 noise levels x 2 correlations"
    )
    p.add_argument('--run_name',  type=str, default=None,
                   help="Root folder name (default: timestamp YYYYmmdd_HHMMSS).")
    p.add_argument('--out_root',  type=str, default=None,
                   help="Root output path (default: plots/grand_test/).")
    p.add_argument('--datasets',  type=str, nargs='+', default=None,
                   choices=ALL_DATASETS,
                   help="Subset of datasets to run (default: all).")
    p.add_argument('--resume',    action='store_true',
                   help='Skip cells where results.json already exists.')

    p.add_argument('--seeds',     type=int, nargs='+', default=[0],
                   help="Seeds to run (default: 0).")

    p.add_argument('--erm_steps', type=int,   default=35_000)
    p.add_argument('--erm_lr',    type=float, default=1e-4)
    p.add_argument('--irm_steps', type=int,   default=35_000)
    p.add_argument('--irm_lr',    type=float, default=1e-4)
    p.add_argument('--irm_lambdas', type=float, nargs='+', default=IRM_LAMBDAS,
                   help="IRM penalty coefficients to compare (default: 50 75 100 125).")

    p.add_argument('--n_parts',     type=int,  default=1,
                   help="Number of grid partitions (for parallelisation).")
    p.add_argument('--part',        type=int,  default=0,
                   help="0-indexed partition to run (default: 0).")

    p.add_argument('--eval_every',  type=int,  default=500)
    p.add_argument('--device',      type=str,  default='auto')
    p.add_argument('--bert_model',  type=str,  default='distilbert-base-uncased')
    p.add_argument('--pooling',     type=str,  default='mean', choices=['mean', 'cls'])

    p.add_argument('--plot_only', action='store_true',
                   help='Regenerate training_curves.png from existing results.json without rerunning training.')
    return p


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    args    = make_parser().parse_args()
    device  = resolve_device(args.device)

    run_name = args.run_name or datetime.now().strftime('%Y%m%d_%H%M%S')
    out_root = Path(args.out_root) if args.out_root else \
               _ROOT / 'results' / 'grand_test' / run_name
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Results -> {out_root}\n")

    datasets = args.datasets or ALL_DATASETS

    # Build the full list of (dataset, p1, p2, noise) cells
    grid = []
    for ds in datasets:
        cfg  = _DATASET_CFG[ds]
        mech = cfg['mechanism']
        corr_levels = CORR_CONF if mech == 'conf' else CORR_SAC_SELECTION
        for p1, p2 in corr_levels:
            for noise in NOISE_LEVELS:
                grid.append((ds, p1, p2, noise))

    # Slice the grid for parallelisation
    if args.n_parts > 1:
        grid = grid[args.part::args.n_parts]

    total   = len(grid)
    done    = 0
    skipped = 0

    part_info = f"  [part {args.part+1}/{args.n_parts}]" if args.n_parts > 1 else ""
    print(f"Grid: {len(datasets)} datasets x {len(NOISE_LEVELS)} noise levels x 2 correlations "
          f"= {total} cells{part_info}  |  seeds={args.seeds}\n")

    # Save global config
    global_cfg = {
        'run_name':   run_name,
        'datasets':   datasets,
        'noise_levels': NOISE_LEVELS,
        'corr_sac_selection': CORR_SAC_SELECTION,
        'corr_conf':  CORR_CONF,
        'seeds':      args.seeds,
        'erm_steps':  args.erm_steps, 'erm_lr':  args.erm_lr,
        'irm_steps':  args.irm_steps, 'irm_lr':  args.irm_lr,
        'irm_lambdas': args.irm_lambdas,
        'eval_every': args.eval_every,
        'device':     str(device),
        'bert_model': args.bert_model,
        'pooling':    args.pooling,
        'started_at': datetime.now().isoformat(),
    }
    (out_root / 'grand_test_config.json').write_text(json.dumps(global_cfg, indent=2))

    # ── Plot-only mode: regenerate PNGs from existing results.json ─────────
    if args.plot_only:
        replotted = 0
        missing   = 0
        for ds, p1, p2, noise in grid:
            cfg      = _DATASET_CFG[ds]
            cell_dir = _run_dir(out_root, cfg['slug'], noise, cfg['mechanism'], p1, p2)
            json_path = cell_dir / 'results.json'
            if not json_path.exists():
                print(f"  SKIP (no results.json): {cell_dir}")
                missing += 1
                continue
            data = json.loads(json_path.read_text())
            mean_histories = data.get('mean_histories', {})
            if not mean_histories:
                print(f"  SKIP (no mean_histories): {cell_dir}")
                missing += 1
                continue
            mech      = cfg['mechanism']
            title     = f"{cfg['slug']}  noise={noise:.3f}  corr={_corr_tag(mech, p1, p2)}"
            _plot_cell_curves(mean_histories, cell_dir, title=title)
            print(f"  Replotted: {cell_dir}")
            replotted += 1
        print(f"\nDone: {replotted} plots regenerated, {missing} skipped (missing JSON).")
        raise SystemExit(0)

    all_summaries = []

    for i, (ds, p1, p2, noise) in enumerate(grid, 1):
        cfg  = _DATASET_CFG[ds]
        mech = cfg['mechanism']
        slug = cfg['slug']

        cell_dir = _run_dir(out_root, slug, noise, mech, p1, p2)
        noise_tag = f"noise={noise:.3f}"
        corr_tag  = _corr_tag(mech, p1, p2)

        print(f"[{i:>3}/{total}] {slug:<30}  {noise_tag}  corr={corr_tag}", end="")

        if args.resume and (cell_dir / 'results.json').exists():
            print("  -> SKIP (results.json exists)")
            skipped += 1
            continue

        print()

        results = run_cell(
            dataset=ds, p1=p1, p2=p2, noise=noise,
            seeds=args.seeds, cfg=cfg,
            cell_dir=cell_dir,
            device=device,
            bert_model=args.bert_model,
            pooling=args.pooling,
            erm_steps=args.erm_steps, erm_lr=args.erm_lr,
            irm_steps=args.irm_steps, irm_lr=args.irm_lr,
            irm_lambdas=args.irm_lambdas,
            eval_every=args.eval_every,
        )

        s = results['summary']
        all_summaries.append({
            'dataset':  ds,
            'slug':     slug,
            'mechanism': mech,
            'p1': p1, 'p2': p2,
            'noise': noise,
            'erm_final_mean': s['erm_final']['mean'],
            'irm_final_mean': s['irm_final']['mean'],
            'delta_final_mean': s['delta_final']['mean'],
        })
        done += 1

    # Save global summary
    (out_root / 'summary_all.json').write_text(
        json.dumps(all_summaries, indent=2)
    )

    print(f"\n{'='*60}")
    print(f"Done: {done} runs completed, {skipped} skipped.")
    print(f"Results in: {out_root}")
    print(f"{'='*60}\n")

    # Compact summary table
    if all_summaries:
        print(f"{'Dataset':<30} {'noise':>7} {'corr':>14} {'ERM':>7} {'IRM*':>7} {'Δ':>7}")
        print(f"  (* best IRM lambda across {args.irm_lambdas})")
        print('-' * 75)
        for r in all_summaries:
            ctag = _corr_tag(r['mechanism'], r['p1'], r['p2'])
            print(f"{r['slug']:<30} {r['noise']:>7.3f} {ctag:>14} "
                  f"{r['erm_final_mean']:>7.3f} {r['irm_final_mean']:>7.3f} "
                  f"{r['delta_final_mean']:>+7.3f}")
