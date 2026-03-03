"""
run_gap_sweep_sac.py
====================
Balayage du GAP entre les deux environnements de train pour l'expérience
semi-anti-causal.

Paramètre variant : ps_train = [0.2 - gap/2,  0.2 + gap/2]
Gaps testés       : 0.00, 0.05, 0.10, ..., 0.40

Pour chaque gap, ERM et IRM sont entraînés et la précision OOD finale
est enregistrée. Un unique graphe récapitulatif est produit.

Usage :
    uv run run_gap_sweep_sac.py                      # gaps 0→0.4 pas 0.05
    uv run run_gap_sweep_sac.py --gap_step 0.025     # pas plus fin
    uv run run_gap_sweep_sac.py --n 50000            # dataset plus petit
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import torch

from data_synth import build_envs_semi_anti_causal
from models_training import train_erm, train_irm
from utils_irm import resolve_device

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser(description="Gap sweep – semi-anti-causal")
p.add_argument('--n',           type=int,   default=200_000, help='N par env train')
p.add_argument('--n_test',      type=int,   default=10_000,  help='N test OOD')
p.add_argument('--val_frac',    type=float, default=0.1)
p.add_argument('--p_center',    type=float, default=0.2,     help='Centre du gap (p moyen)')
p.add_argument('--p_test_ood',  type=float, default=1.0,     help='p spurieux test OOD')
p.add_argument('--label_flip',  type=float, default=0.25)
p.add_argument('--gap_step',    type=float, default=0.05,    help='Pas du gap')
p.add_argument('--gap_max',     type=float, default=0.40,    help='Gap maximum')
p.add_argument('--erm_steps',   type=int,   default=25_000)
p.add_argument('--erm_lr',      type=float, default=5e-3)
p.add_argument('--irm_steps',   type=int,   default=25_000)
p.add_argument('--irm_lr',      type=float, default=5e-3)
p.add_argument('--irm_lambda',  type=float, default=750.0)
p.add_argument('--seed',        type=int,   default=1)
p.add_argument('--device',      type=str,   default='auto')
p.add_argument('--eval_every',  type=int,   default=500)
p.add_argument('--out_dir',     type=str,   default='plot',  help='Dossier de sortie')
p.add_argument('--save_json',   action='store_true', default=True,
               help='Sauvegarde les résultats bruts en JSON')
args = p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(args.out_dir, exist_ok=True)
device = resolve_device(args.device)

# Gaps à tester : 0.00, 0.05, ..., 0.40
n_steps = round(args.gap_max / args.gap_step)
gaps = [round(i * args.gap_step, 10) for i in range(n_steps + 1)]

print(f"{'='*60}")
print(f"Gap sweep – semi-anti-causal")
print(f"  p_center  = {args.p_center}")
print(f"  p_test    = {args.p_test_ood}")
print(f"  Gaps      = {gaps}")
print(f"  N/env     = {args.n:,}  |  N_test = {args.n_test:,}")
print(f"  ERM steps = {args.erm_steps}  |  IRM steps = {args.irm_steps}")
print(f"  IRM λ     = {args.irm_lambda}")
print(f"{'='*60}\n")

# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────
results = []

for gap in gaps:
    p1 = round(args.p_center - gap / 2.0, 10)
    p2 = round(args.p_center + gap / 2.0, 10)

    # Clamp pour rester dans [0, 1]
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))

    print(f"\n── Gap = {gap:.2f}  →  ps_train = [{p1:.3f}, {p2:.3f}] ──")

    # ---- Génération des environnements ----
    train_envs, val_envs, test_env = build_envs_semi_anti_causal(
        n=args.n,
        train_p_spurs=[p1, p2],
        test_p_spur=args.p_test_ood,
        seed=args.seed,
        val_frac=args.val_frac,
        label_flip=args.label_flip,
        n_test=args.n_test,
    )

    # ---- ERM ----
    print("  [ERM]", end=" ", flush=True)
    _, erm_hist = train_erm(
        envs=train_envs,
        val_envs=val_envs,
        test_env=test_env,
        steps=args.erm_steps,
        lr=args.erm_lr,
        batch=512,
        seed=args.seed,
        device=device,
        eval_every=args.eval_every,
        model_kind='logreg',
        dataset_name='synthetic_semi_anti_causal',
    )
    erm_test_final  = erm_hist['test_acc'][-1]  if erm_hist['test_acc']  else float('nan')
    erm_val_final   = erm_hist['val_acc'][-1]   if erm_hist['val_acc']   else float('nan')
    erm_test_best   = max(erm_hist['test_acc'])  if erm_hist['test_acc']  else float('nan')
    print(f"test_OOD={erm_test_final:.3f}  (best={erm_test_best:.3f})")

    # ---- IRM ----
    print("  [IRM]", end=" ", flush=True)
    _, irm_hist = train_irm(
        envs=train_envs,
        val_envs=val_envs,
        test_env=test_env,
        steps=args.irm_steps,
        lr=args.irm_lr,
        batch=512,
        irm_lambda=args.irm_lambda,
        seed=args.seed,
        device=device,
        eval_every=args.eval_every,
        model_kind='logreg',
        dataset_name='synthetic_semi_anti_causal',
    )
    irm_test_final  = irm_hist['test_acc'][-1]  if irm_hist['test_acc']  else float('nan')
    irm_val_final   = irm_hist['val_acc'][-1]   if irm_hist['val_acc']   else float('nan')
    irm_test_best   = max(irm_hist['test_acc'])  if irm_hist['test_acc']  else float('nan')
    print(f"test_OOD={irm_test_final:.3f}  (best={irm_test_best:.3f})")

    results.append({
        'gap':            gap,
        'p1':             p1,
        'p2':             p2,
        'erm_test_final': erm_test_final,
        'erm_val_final':  erm_val_final,
        'erm_test_best':  erm_test_best,
        'irm_test_final': irm_test_final,
        'irm_val_final':  irm_val_final,
        'irm_test_best':  irm_test_best,
    })

# ─────────────────────────────────────────────────────────────────────────────
# Sauvegarde JSON
# ─────────────────────────────────────────────────────────────────────────────
if args.save_json:
    json_path = os.path.join(args.out_dir, 'gap_sweep_sac_results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRésultats bruts sauvegardés : {json_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Résumé console
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"{'Gap':>6} | {'ps_env1':>8} | {'ps_env2':>8} | {'ERM OOD':>8} | {'IRM OOD':>8} | {'Δ(IRM-ERM)':>11}")
print(f"{'─'*6}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*8}-+-{'─'*11}")
for r in results:
    delta = r['irm_test_final'] - r['erm_test_final']
    print(f"  {r['gap']:.2f} | {r['p1']:>8.3f} | {r['p2']:>8.3f} | "
          f"{r['erm_test_final']:>8.3f} | {r['irm_test_final']:>8.3f} | {delta:>+11.3f}")
print(f"{'='*60}")

# ─────────────────────────────────────────────────────────────────────────────
# Graphe récapitulatif
# ─────────────────────────────────────────────────────────────────────────────
g       = [r['gap']            for r in results]
erm_acc = [r['erm_test_final'] for r in results]
irm_acc = [r['irm_test_final'] for r in results]
erm_best= [r['erm_test_best']  for r in results]
irm_best= [r['irm_test_best']  for r in results]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Semi-Anti-Causal – Impact du gap entre environnements\n"
    f"(p_center={args.p_center}, p_test_OOD={args.p_test_ood}, "
    f"N={args.n:,}, λ_IRM={args.irm_lambda})",
    fontsize=12, fontweight='bold'
)

# ── Axe gauche : précision OOD finale ──
ax = axes[0]
ax.plot(g, erm_acc, 'o-', color='orange', linewidth=2.5, markersize=7, label='ERM (OOD final)')
ax.plot(g, irm_acc, 's-', color='steelblue', linewidth=2.5, markersize=7, label='IRM (OOD final)')
ax.axhline(0.75, color='gray', linestyle=':', linewidth=1.2, label='Chance label-flip=0.25')
ax.fill_between(g, erm_acc, irm_acc, alpha=0.12, color='steelblue', label='Avantage IRM')
ax.set_xlabel('Gap entre environnements  Δp = p₂ − p₁', fontsize=11)
ax.set_ylabel('Précision test OOD', fontsize=11)
ax.set_title('Précision finale', fontsize=11)
ax.set_xlim(-0.01, args.gap_max + 0.01)
ax.set_ylim(0.45, 1.02)
ax.set_xticks(g)
ax.set_xticklabels([f'{x:.2f}' for x in g], rotation=45, ha='right')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Annoter les valeurs
for xi, ye, yi in zip(g, erm_acc, irm_acc):
    ax.annotate(f'{ye:.2f}', (xi, ye), textcoords='offset points',
                xytext=(0, 8), ha='center', fontsize=7, color='orange')
    ax.annotate(f'{yi:.2f}', (xi, yi), textcoords='offset points',
                xytext=(0, -14), ha='center', fontsize=7, color='steelblue')

# ── Axe droit : avantage IRM = Δ(IRM - ERM) ──
ax2 = axes[1]
delta = [i - e for i, e in zip(irm_acc, erm_acc)]
colors = ['steelblue' if d >= 0 else 'tomato' for d in delta]
bars = ax2.bar(g, delta, width=args.gap_step * 0.7, color=colors, alpha=0.8, edgecolor='white')
ax2.axhline(0, color='black', linewidth=1.2)
ax2.set_xlabel('Gap entre environnements  Δp = p₂ − p₁', fontsize=11)
ax2.set_ylabel('Avantage IRM  (IRM − ERM)', fontsize=11)
ax2.set_title('Gain IRM par rapport à ERM', fontsize=11)
ax2.set_xticks(g)
ax2.set_xticklabels([f'{x:.2f}' for x in g], rotation=45, ha='right')
ax2.grid(True, alpha=0.3, axis='y')
for bar, d in zip(bars, delta):
    ax2.text(bar.get_x() + bar.get_width() / 2, d + (0.003 if d >= 0 else -0.008),
             f'{d:+.3f}', ha='center', va='bottom' if d >= 0 else 'top', fontsize=8)

plt.tight_layout()
plot_path = os.path.join(args.out_dir, 'gap_sweep_sac.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"\nGraphe sauvegardé : {plot_path}")
plt.close()
