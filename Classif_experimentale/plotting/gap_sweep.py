#!/usr/bin/env python3
"""
Generate combined gap-sweep plots — one per mechanism, overlaying IMDB Genres
and Amazon Books curves on the same axes. Reads the latest JSON results from
plots/gap_sweep/.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOTS_DIR = Path(__file__).resolve().parents[1] / "results" / "gap_sweep"

# ── Mechanisms grouped by type ──────────────────────────────────────────────
MECHANISMS = {
    "sac": {
        "title": "Semi-Anti-Causal — Gap Sweep",
        "x_label": "Gap \u0394p_correct = p\u2082 \u2212 p\u2081",
        "datasets": [
            ("ac_imdb_genres_sac",  "IMDB Genres"),
            ("causal_amazon_sac",   "Amazon Books"),
        ],
    },
    "size_selection": {
        "title": "Selection Bias — Gap Sweep",
        "x_label": "Gap \u0394p_select = p\u2082 \u2212 p\u2081",
        "datasets": [
            ("ac_imdb_genres_size_selection",    "IMDB Genres"),
            ("causal_amazon_sentiment_selection", "Amazon Books"),
        ],
    },
    "conf": {
        "title": "Confounding — Gap Sweep",
        "x_label": "Gap \u0394a = a\u2082 \u2212 a\u2081",
        "datasets": [
            ("ac_imdb_genres_conf_proxy",  "IMDB Genres"),
            ("causal_amazon_conf_proxy",   "Amazon Books"),
        ],
    },
}

# ── Colours and line styles ────────────────────────────────────────────────────────────
DATASET_COLORS = {
    "IMDB Genres":  "#2471A3",   # blue
    "Amazon Books": "#E67E22",   # orange
}
ERM_STYLE = dict(linestyle="-",  marker="o", linewidth=2.2, markersize=6)
IRM_STYLE = dict(linestyle="--", marker="s", linewidth=2.2, markersize=6)


def load_latest(slug: str):
    folder = PLOTS_DIR / slug
    if not folder.exists():
        return None
    runs = sorted(folder.iterdir())
    if not runs:
        return None
    json_file = runs[-1] / "gap_sweep_results.json"
    if not json_file.exists():
        return None
    with open(json_file) as f:
        return json.load(f)


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).resolve().parents[1] / "results" / "gap_sweep_combined" / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    for mech_key, mech in MECHANISMS.items():
        fig, ax = plt.subplots(figsize=(5.5, 6.5))
        ax.set_xlabel(mech["x_label"], fontsize=22)
        ax.set_ylabel("Test OOD accuracy", fontsize=22)

        all_acc = []
        g_ref = None

        for slug, label in mech["datasets"]:
            data = load_latest(slug)
            if data is None:
                print(f"  [!] Results not found: {slug}")
                continue

            agg = data["aggregated"]
            g         = [r["gap"]           for r in agg]
            erm_mean  = [r["erm_test_mean"]  for r in agg]
            erm_std   = [r["erm_test_std"]   for r in agg]
            irm_mean  = [r["irm_test_mean"]  for r in agg]
            irm_std   = [r["irm_test_std"]   for r in agg]

            if g_ref is None:
                g_ref = g

            color = DATASET_COLORS[label]

            # ERM
            ax.plot(g, erm_mean, color=color, **ERM_STYLE,
                    label=f"{label} — ERM")
            ax.fill_between(
                g,
                [m - s for m, s in zip(erm_mean, erm_std)],
                [m + s for m, s in zip(erm_mean, erm_std)],
                alpha=0.12, color=color,
            )

            # IRM
            ax.plot(g, irm_mean, color=color, **IRM_STYLE,
                    label=f"{label} — IRM")
            ax.fill_between(
                g,
                [m - s for m, s in zip(irm_mean, irm_std)],
                [m + s for m, s in zip(irm_mean, irm_std)],
                alpha=0.12, color=color,
            )

            all_acc.extend(erm_mean + irm_mean)

        if g_ref is not None:
            ax.set_xticks(g_ref)
            ax.set_xticklabels([f"{x:.3f}" for x in g_ref],
                               rotation=45, ha="right", fontsize=20)
        ax.tick_params(axis="y", labelsize=20)

        if all_acc:
            margin = max(0.02, (max(all_acc) - min(all_acc)) * 0.15)
            ax.set_ylim(
                max(0.0, min(all_acc) - margin),
                min(1.02, max(all_acc) + margin),
            )

        ax.legend(fontsize=13, loc="best")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out_path = out_dir / f"gap_sweep_{mech_key}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    print(f"\nDone -- output dir: {out_dir}")


if __name__ == "__main__":
    main()
