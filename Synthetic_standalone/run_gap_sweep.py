#!/usr/bin/env python3
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

"""
run_gap_sweep.py
================
Balayage du GAP entre les deux environnements de train.
Généralise run_gap_sweep_sac.py à tous les datasets synthétiques.

Le "gap" est la différence entre le paramètre d'environnement des deux envs
de train. À gap=0, les deux envs sont identiques (IRM ne peut pas distinguer
le signal causal du signal spurieux). À gap élevé, la différence entre les
envs est grande et IRM peut exploiter l'incohérence.

Paramètre variant par dataset :
  synthetic_semi_anti_causal          : p_spur   (center=0.2,  test=1.0)
  synthetic_selection                 : alpha    (center=0.85, test=0.0)
  synthetic_confounding_varying_proxy : a        (center=0.06, test=0.99)

Usage :
    uv run run_gap_sweep.py --dataset synthetic_selection
    uv run run_gap_sweep.py --dataset synthetic_confounding_varying_proxy --gap_step 0.02
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from args_synthetic import make_gap_sweep_parser, DATASET_DEFAULTS
from data_synth import (
    build_envs_semi_anti_causal,
    build_envs_selection,
    build_envs_confounding_varying_proxy,
    # Anti-causal variants
    build_envs_anti_causal_semi_anti_causal,
    build_envs_anti_causal_selection,
    build_envs_anti_causal_confounding_varying_proxy,
)
from models_training import train_erm, train_irm
from utils_irm import resolve_device

# ─────────────────────────────────────────────────────────────────────────────
# Defaults par dataset
# ─────────────────────────────────────────────────────────────────────────────
# DATASET_DEFAULTS is defined in args_synthetic.py

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
args = make_gap_sweep_parser().parse_args()

# Appliquer les defaults
defs = DATASET_DEFAULTS[args.dataset]
if args.gap_center is None: args.gap_center = defs['gap_center']
if args.gap_test   is None: args.gap_test   = defs['gap_test']
if args.gap_max    is None: args.gap_max    = defs['gap_max']
if args.out_dir is None:
    from datetime import datetime
    _SLUG_MAP = {
        'synthetic_semi_anti_causal':              'causal_semi_anti_causal',
        'synthetic_selection':                      'causal_selection',
        'synthetic_confounding_varying_proxy':      'causal_conf_varying_proxy',
        'synthetic_ac_semi_anti_causal':            'ac_semi_anti_causal',
        'synthetic_ac_selection':                   'ac_selection',
        'synthetic_ac_confounding_varying_proxy':   'ac_conf_varying_proxy',
    }
    _slug = _SLUG_MAP.get(args.dataset, args.dataset.replace('synthetic_', ''))
    _ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.out_dir = str(_Path(__file__).parent / 'plots' / 'gap_sweep' / _slug / _ts)

os.makedirs(args.out_dir, exist_ok=True)
device = resolve_device(args.device)

n_steps = round(args.gap_max / args.gap_step)
gaps = [round(i * args.gap_step, 10) for i in range(n_steps + 1)]

print(f"{'='*60}")
print(f"Gap sweep – {args.dataset}")
print(f"  param     = {defs['param_label']}")
print(f"  center    = {args.gap_center}  |  test_OOD = {args.gap_test}")
print(f"  Gaps      = {gaps}")
print(f"  N/env     = {args.n:,}  |  N_test = {args.n_test:,}")
print(f"  ERM steps = {args.erm_steps}  |  IRM steps = {args.irm_steps}")
print(f"  IRM λ     = {args.irm_lambda}")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Construction des environnements selon le dataset
# ─────────────────────────────────────────────────────────────────────────────
def _build(gap):
    p1 = max(0.0, min(1.0, round(args.gap_center - gap / 2.0, 10)))
    p2 = max(0.0, min(1.0, round(args.gap_center + gap / 2.0, 10)))
    g_test = args.gap_test

    if args.dataset == 'synthetic_semi_anti_causal':
        envs = build_envs_semi_anti_causal(
            n=args.n, train_p_spurs=[p1, p2], test_p_spur=g_test,
            seed=args.seed, val_frac=args.val_frac, label_flip=args.label_flip,
            n_test=args.n_test, dim_z=args.dim_z, dim_y=args.dim_y,
        )
    elif args.dataset == 'synthetic_selection':
        envs = build_envs_selection(
            n=args.n, train_alphas=[p1, p2], test_alpha=g_test,
            seed=args.seed, val_frac=args.val_frac, label_flip=args.label_flip,
            n_test=args.n_test, dim_z=args.dim_z, dim_y=args.dim_y,
        )
    elif args.dataset == 'synthetic_confounding_varying_proxy':
        envs = build_envs_confounding_varying_proxy(
            n=args.n, a_train=[p1, p2], a_test=g_test,
            gamma=args.conf_gamma,
            seed=args.seed, val_frac=args.val_frac, n_test=args.n_test,
            dim_z=args.dim_z, dim_y=args.dim_y,
        )    # ── Anti-causal variants ──
    elif args.dataset == 'synthetic_ac_semi_anti_causal':
        envs = build_envs_anti_causal_semi_anti_causal(
            n=args.n, train_p_spurs=[p1, p2], test_p_spur=g_test,
            seed=args.seed, val_frac=args.val_frac, label_flip=args.label_flip,
            n_test=args.n_test, dim_z=args.dim_z, dim_y=args.dim_y,
        )
    elif args.dataset == 'synthetic_ac_selection':
        envs = build_envs_anti_causal_selection(
            n=args.n, train_alphas=[p1, p2], test_alpha=g_test,
            seed=args.seed, val_frac=args.val_frac, label_flip=args.label_flip,
            n_test=args.n_test, dim_z=args.dim_z, dim_y=args.dim_y,
        )
    elif args.dataset == 'synthetic_ac_confounding_varying_proxy':
        envs = build_envs_anti_causal_confounding_varying_proxy(
            n=args.n, a_train=[p1, p2], a_test=g_test,
            gamma=args.conf_gamma,
            seed=args.seed, val_frac=args.val_frac, n_test=args.n_test,
            dim_z=args.dim_z, dim_y=args.dim_y,
            label_flip=args.label_flip,
        )
    return envs, (p1, p2)

# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────
dataset_name = args.dataset
results = []

for gap in gaps:
    (train_envs, val_envs, test_env), (p1, p2) = _build(gap)

    print(f"\n── Gap = {gap:.3f}  →  [{defs['param_label']}] = [{p1:.4f}, {p2:.4f}] ──")

    # ---- ERM ----
    print("  [ERM]", end=" ", flush=True)
    _, erm_hist = train_erm(
        envs=train_envs, val_envs=val_envs, test_env=test_env,
        steps=args.erm_steps, lr=args.erm_lr, batch=512,
        seed=args.seed, device=device, eval_every=args.eval_every,
        dataset_name=dataset_name,
    )
    erm_test_final = erm_hist['test_acc'][-1] if erm_hist['test_acc'] else float('nan')
    erm_test_best  = max(erm_hist['test_acc']) if erm_hist['test_acc'] else float('nan')
    print(f"test_OOD={erm_test_final:.3f}  (best={erm_test_best:.3f})")

    # ---- IRM ----
    print("  [IRM]", end=" ", flush=True)
    _, irm_hist = train_irm(
        envs=train_envs, val_envs=val_envs, test_env=test_env,
        steps=args.irm_steps, lr=args.irm_lr, batch=512,
        irm_lambda=args.irm_lambda,
        seed=args.seed, device=device, eval_every=args.eval_every,
        dataset_name=dataset_name,
    )
    irm_test_final = irm_hist['test_acc'][-1] if irm_hist['test_acc'] else float('nan')
    irm_test_best  = max(irm_hist['test_acc']) if irm_hist['test_acc'] else float('nan')
    print(f"test_OOD={irm_test_final:.3f}  (best={irm_test_best:.3f})")

    results.append({
        'gap':            gap,
        'p1':             p1,
        'p2':             p2,
        'erm_test_final': erm_test_final,
        'erm_test_best':  erm_test_best,
        'irm_test_final': irm_test_final,
        'irm_test_best':  irm_test_best,
    })

    # Free GPU memory between gap iterations
    del _
    if device == 'cuda':
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde JSON
# ─────────────────────────────────────────────────────────────────────────────
json_path = os.path.join(args.out_dir, 'gap_sweep_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nRésultats bruts sauvegardés : {json_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Résumé console
# ─────────────────────────────────────────────────────────────────────────────
param_lbl = defs['param_label']
print(f"\n{'='*60}")
print(f"{'Gap':>6} | {param_lbl+'₁':>10} | {param_lbl+'₂':>10} | {'ERM OOD':>8} | {'IRM OOD':>8} | {'Δ(IRM-ERM)':>11}")
print(f"{'─'*6}-+-{'─'*10}-+-{'─'*10}-+-{'─'*8}-+-{'─'*8}-+-{'─'*11}")
for r in results:
    delta = r['irm_test_final'] - r['erm_test_final']
    print(f"  {r['gap']:.3f} | {r['p1']:>10.4f} | {r['p2']:>10.4f} | "
          f"{r['erm_test_final']:>8.3f} | {r['irm_test_final']:>8.3f} | {delta:>+11.3f}")
print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────────────────────
# Graphe récapitulatif
# ─────────────────────────────────────────────────────────────────────────────
g        = [r['gap']            for r in results]
erm_acc  = [r['erm_test_final'] for r in results]
irm_acc  = [r['irm_test_final'] for r in results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))


# ── Axe gauche : précision OOD finale ──
ax = axes[0]
ax.plot(g, erm_acc, 'o-', color='orange',    linewidth=2.5, markersize=7, label='ERM (Final OOD)')
ax.plot(g, irm_acc, 's-', color='steelblue', linewidth=2.5, markersize=7, label='IRM (Final OOD)')
ax.fill_between(g, erm_acc, irm_acc, alpha=0.12, color='steelblue', label='IRM Advantage')
ax.set_xlabel(defs['x_label'], fontsize=11)
ax.set_ylabel('OOD Test Accuracy', fontsize=11)
ax.set_title('Final Accuracy', fontsize=11)
ax.set_xlim(-0.005, args.gap_max + 0.005)
_all_acc = erm_acc + irm_acc
_margin  = max(0.02, (max(_all_acc) - min(_all_acc)) * 0.15)
ax.set_ylim(max(0.0, min(_all_acc) - _margin), min(1.02, max(_all_acc) + _margin))
ax.set_xticks(g)
ax.set_xticklabels([f'{x:.3f}' for x in g], rotation=45, ha='right')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
for xi, ye, yi in zip(g, erm_acc, irm_acc):
    ax.annotate(f'{ye:.2f}', (xi, ye), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=7, color='orange')
    ax.annotate(f'{yi:.2f}', (xi, yi), textcoords='offset points',
                xytext=(0, -14), ha='center', fontsize=7, color='steelblue')

# ── Axe droit : avantage IRM ──
ax2 = axes[1]
delta  = [i - e for i, e in zip(irm_acc, erm_acc)]
colors = ['steelblue' if d >= 0 else 'tomato' for d in delta]
bars   = ax2.bar(g, delta, width=args.gap_step * 0.7, color=colors, alpha=0.8, edgecolor='white')
ax2.axhline(0, color='black', linewidth=1.2)
ax2.set_xlabel(defs['x_label'], fontsize=11)
ax2.set_ylabel('IRM Advantage (IRM − ERM)', fontsize=11)
ax2.set_title('IRM Gain over ERM', fontsize=11)
ax2.set_xticks(g)
ax2.set_xticklabels([f'{x:.3f}' for x in g], rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')
for bar, d in zip(bars, delta):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             d + (0.003 if d >= 0 else -0.008),
             f'{d:+.3f}', ha='center',
             va='bottom' if d >= 0 else 'top', fontsize=8)

plt.tight_layout()
plot_path = os.path.join(args.out_dir, 'gap_sweep.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Graphe sauvegardé : {plot_path}")
plt.close()

# Ensure full GPU memory release before process exits (so the next sweep can start cleanly)
if device == 'cuda':
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
