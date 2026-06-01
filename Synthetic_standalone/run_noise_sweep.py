#!/usr/bin/env python3
import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

"""
run_noise_sweep.py
==================
Balayage du bruit (label_flip) pour les datasets semi_anti_causal et selection.

Le gap entre les deux environnements de train est fixé (paramètres des
expériences principales), et on fait varier le taux de retournement
d'étiquettes de 0 à noise_max.  Ceci permet de mesurer à quel point IRM
bénéficie d'un signal causal propre vs bruité, à gap fixe.

Paramètre variant :
  label_flip ∈ [0, noise_max]  (par défaut 0 → 0.25 par pas de 0.025)

Datasets supportés :
  synthetic_semi_anti_causal          : p_spur   fixé à [0.1, 0.2], test=1.0
  synthetic_selection                 : alpha    fixé à [0.9, 0.8], test=0.0
  synthetic_ac_semi_anti_causal       : idem, anti-causal
  synthetic_ac_selection              : idem, anti-causal

Usage :
    uv run run_noise_sweep.py --dataset synthetic_semi_anti_causal
    uv run run_noise_sweep.py --dataset synthetic_selection --noise_step 0.05
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt

from args_synthetic import make_noise_sweep_parser, NOISE_SWEEP_DEFAULTS
from data_synth import (
    build_envs_semi_anti_causal,
    build_envs_selection,
    build_envs_anti_causal_semi_anti_causal,
    build_envs_anti_causal_selection,
    build_envs_confounding_varying_proxy,
    build_envs_anti_causal_confounding_varying_proxy,
)
from models_training import train_erm, train_irm
from utils_irm import resolve_device

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
args = make_noise_sweep_parser().parse_args()

# Appliquer les defaults par dataset
defs = NOISE_SWEEP_DEFAULTS[args.dataset]
if args.p1         is None: args.p1         = defs['p1']
if args.p2         is None: args.p2         = defs['p2']
if args.p_test     is None: args.p_test     = defs['p_test']
if args.noise_max  is None: args.noise_max  = defs['noise_max']
if args.noise_step is None: args.noise_step = defs.get('noise_step', 0.025)

if args.out_dir is None:
    from datetime import datetime
    _SLUG_MAP = {
        'synthetic_semi_anti_causal':   'causal_semi_anti_causal',
        'synthetic_selection':           'causal_selection',
        'synthetic_ac_semi_anti_causal': 'ac_semi_anti_causal',
        'synthetic_ac_selection':        'ac_selection',
        'synthetic_confounding_varying_proxy': 'causal_conf_varying_proxy',
        'synthetic_ac_confounding_varying_proxy': 'ac_conf_varying_proxy',
    }
    _slug = _SLUG_MAP.get(args.dataset, args.dataset.replace('synthetic_', ''))
    _ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.out_dir = str(_Path(__file__).parent / 'plots' / 'noise_sweep' / _slug / _ts)

os.makedirs(args.out_dir, exist_ok=True)
device = resolve_device(args.device)

n_steps = round(args.noise_max / args.noise_step)
noises  = [round(i * args.noise_step, 10) for i in range(n_steps + 1)]

print(f"{'='*60}")
print(f"Noise sweep – {args.dataset}")
print(f"  [{defs['param_label']}]  p1={args.p1}, p2={args.p2}, test={args.p_test}")
print(f"  Bruit values = {noises}")
print(f"  N/env        = {args.n:,}  |  N_test = {args.n_test:,}")
print(f"  ERM steps    = {args.erm_steps}  |  IRM steps = {args.irm_steps}")
print(f"  IRM λ        = {args.irm_lambda}")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Construction des environnements selon le dataset
# ─────────────────────────────────────────────────────────────────────────────
def _build(noise):
    common = dict(
        n=args.n, seed=args.seed, val_frac=args.val_frac,
        n_test=args.n_test, dim_z=args.dim_z, dim_y=args.dim_y,
        label_flip=noise,
    )
    if args.dataset == 'synthetic_semi_anti_causal':
        return build_envs_semi_anti_causal(
            train_p_spurs=[args.p1, args.p2], test_p_spur=args.p_test, **common,
        )
    elif args.dataset == 'synthetic_selection':
        return build_envs_selection(
            train_alphas=[args.p1, args.p2], test_alpha=args.p_test, **common,
        )
    elif args.dataset == 'synthetic_ac_semi_anti_causal':
        return build_envs_anti_causal_semi_anti_causal(
            train_p_spurs=[args.p1, args.p2], test_p_spur=args.p_test, **common,
        )
    elif args.dataset == 'synthetic_ac_selection':
        return build_envs_anti_causal_selection(
            train_alphas=[args.p1, args.p2], test_alpha=args.p_test, **common,
        )
    elif args.dataset == 'synthetic_confounding_varying_proxy':
        return build_envs_confounding_varying_proxy(
            n=args.n, a_train=[args.p1, args.p2], a_test=args.p_test,
            gamma=noise, seed=args.seed, val_frac=args.val_frac,
            n_test=args.n_test, dim_z=args.dim_z, dim_y=args.dim_y,
        )
    elif args.dataset == 'synthetic_ac_confounding_varying_proxy':
        return build_envs_anti_causal_confounding_varying_proxy(
            a_train=[args.p1, args.p2], a_test=args.p_test, **common
        )
    else:
        raise ValueError(f"Dataset non supporté pour noise sweep : {args.dataset}")

# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────
dataset_name = args.dataset
results = []

for noise in noises:
    train_envs, val_envs, test_env = _build(noise)

    print(f"\n── Noise Rate (gamma/flip) = {noise:.4f} ──")

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
        'noise':          noise,
        'erm_test_final': erm_test_final,
        'erm_test_best':  erm_test_best,
        'irm_test_final': irm_test_final,
        'irm_test_best':  irm_test_best,
    })

    del _
    if device == 'cuda':
        torch.cuda.empty_cache()

# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde JSON
# ─────────────────────────────────────────────────────────────────────────────
json_path = os.path.join(args.out_dir, 'noise_sweep_results.json')
with open(json_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nRésultats bruts sauvegardés : {json_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Résumé console
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'Noise':>7} | {'ERM OOD':>8} | {'IRM OOD':>8} | {'Δ(IRM-ERM)':>11}")
print(f"{'─'*7}-+-{'─'*8}-+-{'─'*8}-+-{'─'*11}")
for r in results:
    delta = r['irm_test_final'] - r['erm_test_final']
    print(f"  {r['noise']:.4f} | {r['erm_test_final']:>8.3f} | {r['irm_test_final']:>8.3f} | {delta:>+11.3f}")
print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────────────────────
# Graphe récapitulatif
# ─────────────────────────────────────────────────────────────────────────────
n_vals  = [r['noise']           for r in results]
erm_acc = [r['erm_test_final']  for r in results]
irm_acc = [r['irm_test_final']  for r in results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))


# ── Axe gauche : précision OOD finale ──
ax = axes[0]
ax.plot(n_vals, erm_acc, 'o-', color='orange',    linewidth=2.5, markersize=7, label='ERM (Final OOD)')
ax.plot(n_vals, irm_acc, 's-', color='steelblue', linewidth=2.5, markersize=7, label='IRM (Final OOD)')
ax.fill_between(n_vals, erm_acc, irm_acc, alpha=0.12, color='steelblue', label='IRM Advantage')
ax.set_xlabel('Noise Rate', fontsize=11)
ax.set_ylabel('OOD Test Accuracy', fontsize=11)
ax.set_title('Final Accuracy', fontsize=11)
ax.set_xlim(-0.005, args.noise_max + 0.005)
_all_acc = erm_acc + irm_acc
_margin  = max(0.02, (max(_all_acc) - min(_all_acc)) * 0.15)
ax.set_ylim(max(0.0, min(_all_acc) - _margin), min(1.02, max(_all_acc) + _margin))
ax.set_xticks(n_vals)
ax.set_xticklabels([f'{x:.3f}' for x in n_vals], rotation=45, ha='right')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
for xi, ye, yi in zip(n_vals, erm_acc, irm_acc):
    ax.annotate(f'{ye:.2f}', (xi, ye), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=7, color='orange')
    ax.annotate(f'{yi:.2f}', (xi, yi), textcoords='offset points',
                xytext=(0, -14), ha='center', fontsize=7, color='steelblue')

# ── Axe droit : avantage IRM ──
ax2 = axes[1]
delta  = [i - e for i, e in zip(irm_acc, erm_acc)]
colors = ['steelblue' if d >= 0 else 'tomato' for d in delta]
bars   = ax2.bar(n_vals, delta, width=args.noise_step * 0.7, color=colors, alpha=0.8, edgecolor='white')
ax2.axhline(0, color='black', linewidth=1.2)
ax2.set_xlabel('Noise Rate', fontsize=11)
ax2.set_ylabel('IRM Advantage (IRM − ERM)', fontsize=11)
ax2.set_title('IRM Gain over ERM', fontsize=11)
ax2.set_xticks(n_vals)
ax2.set_xticklabels([f'{x:.3f}' for x in n_vals], rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')
for bar, d in zip(bars, delta):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             d + (0.003 if d >= 0 else -0.008),
             f'{d:+.3f}', ha='center',
             va='bottom' if d >= 0 else 'top', fontsize=8)

plt.tight_layout()
plot_path = os.path.join(args.out_dir, 'noise_sweep.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"Graphe sauvegardé : {plot_path}")
plt.close()

# Ensure full GPU memory release before process exits
if device == 'cuda':
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
