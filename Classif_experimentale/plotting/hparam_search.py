"""
plot_hparam_search.py
=====================
Visualise the hyperparameter search results from a grand_test run.

Produces three figures for the paper appendix:
  1. lambda_sensitivity.pdf  – OOD accuracy vs λ, one line per dataset/mechanism,
                               aggregated over noise levels and correlation levels.
  2. lambda_heatmap.pdf      – Heat-map  λ × noise  for each dataset × mechanism.
  3. hparam_table.tex        – LaTeX table: mean ± std per λ, per dataset.

Usage
-----
    python nlp/plot_hparam_search.py
    python nlp/plot_hparam_search.py --run_name myrun
    python nlp/plot_hparam_search.py --run_name myrun --out_dir plots/hparam
"""
import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   12,
    "axes.labelsize":   15,
    "xtick.labelsize":  14,
    "ytick.labelsize":  14,
    "legend.fontsize":  12,
})

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT_GRAND_TEST = Path(__file__).resolve().parents[1] / "results" / "grand_test"

LAMBDAS = [50, 75, 100, 125]

DATASET_LABELS = {
    "agnews":  "AG News",
    "imdb":    "IMDB Genres",
    "amazon":  "Amazon Books",
}
MECH_LABELS = {
    "sac":       "Label-Generated",
    "selection": "Selection",
    "conf":      "Confounder",
}

# Correlation tags to include (one per mechanism type)
# Matches the grids defined in run_grand_test.py
KEEP_CORR_SAC_SELECTION = {"p0p90_0p70"}
KEEP_CORR_CONF          = {"a0p010_0p100"}

# Palette: one colour per λ
LAMBDA_COLORS = {
    50:  "#1f77b4",
    75:  "#2ca02c",
    100: "#d62728",
    125: "#9467bd",
}
LAMBDA_MARKERS = {50: "o", 75: "s", 100: "^", 125: "D"}

ERM_COLOR = "#e15759"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_cell_name(name: str):
    m = re.match(r"noise(\d+p\d+)__(.+)", name)
    if not m:
        return None, None
    noise = float(m.group(1).replace("p", "."))
    corr_tag = m.group(2)
    return noise, corr_tag


def _infer_dataset_mech(slug: str):
    """Return (dataset, mech) from a slug like 'agnews_sac' or 'imdb_size_selection'."""
    for mech in ("sac", "conf"):
        if slug.endswith(f"_{mech}"):
            return slug[: -len(f"_{mech}")], mech
    for suffix, mech in [("_size_selection", "selection"),
                          ("_sentiment_selection", "selection"),
                          ("_selection", "selection")]:
        if slug.endswith(suffix):
            return slug[: -len(suffix)], mech
    return slug, "unknown"


def load_run(run_dir: Path):
    """
    Returns a nested dict:
        data[dataset][mech][corr_tag][noise][lambda_key] = {final, best}
        data[dataset][mech][corr_tag][noise]['erm'] = {final, best}
    """
    data = {}
    for ds_dir in sorted(run_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset, mech = _infer_dataset_mech(ds_dir.name)

        for cell_dir in sorted(ds_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            rfile = cell_dir / "results.json"
            if not rfile.exists():
                continue
            noise, corr_tag = _parse_cell_name(cell_dir.name)
            if noise is None:
                continue
            # Filter: only keep the canonical correlation levels
            if mech in ("sac", "selection") and corr_tag not in KEEP_CORR_SAC_SELECTION:
                continue
            if mech == "conf" and corr_tag not in KEEP_CORR_CONF:
                continue

            with open(rfile) as f:
                res = json.load(f)
            s = res["summary"]

            cell = {"erm": {"final": s["erm"]["final"]["mean"],
                            "best":  s["erm"]["best"]["mean"]}}
            for lam in LAMBDAS:
                key = f"irm_{lam}"
                if key in s:
                    cell[key] = {
                        "final": s[key]["final"]["mean"],
                        "best":  s[key]["best"]["mean"],
                    }
            (data
             .setdefault(dataset, {})
             .setdefault(mech, {})
             .setdefault(corr_tag, {})[noise]) = cell
    return data


def _corr_label(corr_tag: str) -> str:
    """Convert e.g. 'p1p00_0p80' → 'ρ=(1.00, 0.80)'."""
    tag = corr_tag.lstrip("pa")
    parts = tag.split("_")
    if len(parts) == 2:
        try:
            p1 = float(parts[0].replace("p", "."))
            p2 = float(parts[1].replace("p", "."))
            return f"\u03c1=({p1:.2f}, {p2:.2f})"
        except ValueError:
            pass
    return corr_tag


# ---------------------------------------------------------------------------
# Figure 1 – Lambda sensitivity (line plot)
# ---------------------------------------------------------------------------

def plot_lambda_sensitivity(data: dict, out_dir: Path, metric: str = "final"):
    """
    One subplot per (dataset, mech, corr_tag). Each line = one λ value.
    X-axis = noise level; Y-axis = OOD accuracy.
    ERM shown as dashed red baseline.
    """
    datasets = sorted(data.keys())
    mechs    = sorted({m for d in data.values() for m in d})

    # Panels ordered as (dataset, mech, corr_tag)
    panels = [
        (ds, mech, ct)
        for ds in datasets
        for mech in mechs
        for ct in sorted(data.get(ds, {}).get(mech, {}).keys())
        if data[ds][mech][ct]
    ]
    if not panels:
        return

    # Build a consistent (mech, corr_tag) column order
    mech_ct_pairs = sorted({(mech, ct) for _, mech, ct in panels})
    ncols = len(mech_ct_pairs)
    nrows = len(datasets)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.8 * ncols, 3.8 * nrows),
                             squeeze=False)
    # no suptitle — caption carries the information

    panel_set = set(panels)
    for ri, dataset in enumerate(datasets):
        for ci, (mech, ct) in enumerate(mech_ct_pairs):
            ax = axes[ri][ci]
            if (dataset, mech, ct) not in panel_set:
                ax.set_visible(False)
                continue

            ct_data = data[dataset][mech][ct]
            all_noises = sorted(ct_data.keys())
            xs = np.array(all_noises)

            # ERM baseline
            erm_arr = np.array([
                ct_data[n].get("erm", {}).get(metric, float("nan"))
                for n in all_noises
            ])
            ax.plot(xs, erm_arr, color=ERM_COLOR, ls="--", marker="o",
                    linewidth=2, markersize=6, label="ERM")

            # One line per lambda
            for lam in LAMBDAS:
                key = f"irm_{lam}"
                irm_arr = np.array([
                    ct_data[n].get(key, {}).get(metric, float("nan"))
                    for n in all_noises
                ])
                ax.plot(xs, irm_arr,
                        color=LAMBDA_COLORS[lam],
                        marker=LAMBDA_MARKERS[lam],
                        linewidth=2, markersize=6,
                        label=f"IRM λ={lam}")

            n_classes = 4 if dataset == "agnews" else 2
            ax.axhline(1 / n_classes, color="gray", ls=":", linewidth=1, label="Chance")

            ax.set_title(
                f"{DATASET_LABELS.get(dataset, dataset)}\n"
                f"{MECH_LABELS.get(mech, mech)} — {_corr_label(ct)}",
                fontsize=12,
            )
            ax.set_xlabel("Label noise", fontsize=15)
            ax.set_ylabel(f"OOD accuracy ({metric})", fontsize=15)
            ax.set_xticks(all_noises)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=11, loc="lower left")
            ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"lambda_sensitivity_{metric}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 – Heat-map: λ × noise per (dataset, mech)
# ---------------------------------------------------------------------------

def plot_lambda_heatmap(data: dict, out_dir: Path, metric: str = "final"):
    """
    One heat-map per (dataset, mech, corr_tag).
    Rows = λ values; Columns = noise levels.
    """
    datasets = sorted(data.keys())
    mechs    = sorted({m for d in data.values() for m in d})

    panels = [
        (ds, mech, ct)
        for ds in datasets
        for mech in mechs
        for ct in sorted(data.get(ds, {}).get(mech, {}).keys())
        if data[ds][mech][ct]
    ]
    if not panels:
        return

    n_panels = len(panels)
    ncols = min(4, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.2 * nrows),
                             squeeze=False)
    # no suptitle — caption carries the information

    for idx, (dataset, mech, ct) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        ct_data = data[dataset][mech][ct]

        all_noises = sorted(ct_data.keys())
        row_keys   = ["erm"] + [f"irm_{lam}" for lam in LAMBDAS]
        row_labels = ["ERM"] + [f"IRM λ={lam}" for lam in LAMBDAS]

        # Build matrix (rows=models, cols=noise)
        matrix = np.full((len(row_keys), len(all_noises)), float("nan"))
        for ci, noise in enumerate(all_noises):
            for ri, key in enumerate(row_keys):
                v = ct_data[noise].get(key, {}).get(metric, float("nan"))
                matrix[ri, ci] = v

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn",
                       vmin=0.0, vmax=1.0)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

        # Annotate cells
        for ri in range(matrix.shape[0]):
            for ci in range(matrix.shape[1]):
                val = matrix[ri, ci]
                if not np.isnan(val):
                    txt_col = "black" if 0.35 < val < 0.85 else "white"
                    ax.text(ci, ri, f"{val:.2f}", ha="center", va="center",
                            fontsize=9, color=txt_col, fontweight="bold")

        ax.set_xticks(range(len(all_noises)))
        ax.set_xticklabels([f"{n:.2f}" for n in all_noises], fontsize=14)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=14)
        ax.set_xlabel("Label noise", fontsize=15)
        ax.set_title(
            f"{DATASET_LABELS.get(dataset, dataset)} — {MECH_LABELS.get(mech, mech)}\n"
            f"{_corr_label(ct)}",
            fontsize=12,
        )

        # Highlight the best IRM row (per column)
        for ci in range(matrix.shape[1]):
            irm_rows = list(range(1, matrix.shape[0]))  # skip ERM row (0)
            best_ri = max(irm_rows, key=lambda ri: matrix[ri, ci] if not np.isnan(matrix[ri, ci]) else -1)
            rect = plt.Rectangle((ci - 0.5, best_ri - 0.5), 1, 1,
                                  linewidth=2, edgecolor="gold", facecolor="none")
            ax.add_patch(rect)

    # Hide unused axes
    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"lambda_heatmap_{metric}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 – Aggregated bar chart (λ comparison, averaged over everything)
# ---------------------------------------------------------------------------

def plot_lambda_aggregate(data: dict, out_dir: Path, metric: str = "final"):
    """
    Single figure: grouped bar chart.
    X-axis = dataset; groups of bars = one per λ + ERM.
    Y = mean OOD accuracy averaged over all noise levels and corr levels.
    """
    datasets = sorted(data.keys())
    all_mechs = sorted({m for d in data.values() for m in d})

    row_keys   = ["erm"] + [f"irm_{lam}" for lam in LAMBDAS]
    row_labels = ["ERM"] + [f"λ={lam}" for lam in LAMBDAS]
    colors     = [ERM_COLOR] + [LAMBDA_COLORS[lam] for lam in LAMBDAS]

    # Build: agg_vals[dataset][mech][model_key] = mean over all (noise, corr)
    agg: dict = {}
    for ds in datasets:
        agg[ds] = {}
        for mech in all_mechs:
            mech_data = data.get(ds, {}).get(mech, {})
            if not mech_data:
                continue
            agg[ds][mech] = {}
            for key in row_keys:
                vals = [
                    mech_data[ct][noise][key][metric]
                    for ct in mech_data
                    for noise in mech_data[ct]
                    if key in mech_data[ct][noise]
                ]
                agg[ds][mech][key] = (np.nanmean(vals) if vals else float("nan"),
                                      np.nanstd(vals)  if vals else 0.0)

    panels = [(ds, mech) for ds in datasets for mech in all_mechs if agg.get(ds, {}).get(mech)]
    if not panels:
        return

    n_panels = len(panels)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(5.5 * ncols, 4.0 * nrows),
                             squeeze=False)
    # no suptitle — caption carries the information

    x = np.arange(len(row_keys))
    bar_w = 0.6

    for idx, (ds, mech) in enumerate(panels):
        ax = axes[idx // ncols][idx % ncols]
        means = [agg[ds][mech].get(k, (float("nan"), 0))[0] for k in row_keys]
        stds  = [agg[ds][mech].get(k, (float("nan"), 0))[1] for k in row_keys]

        bars = ax.bar(x, means, bar_w, yerr=stds, capsize=4,
                      color=colors, edgecolor="white", linewidth=0.8,
                      error_kw={"elinewidth": 1.5, "ecolor": "black"})

        # Mark best IRM
        irm_means = [(means[i], i) for i in range(1, len(means)) if not np.isnan(means[i])]
        if irm_means:
            best_val, best_i = max(irm_means)
            ax.bar(x[best_i], means[best_i], bar_w,
                   color=colors[best_i], edgecolor="gold", linewidth=2.5,
                   label="Best λ")
            ax.annotate("★", xy=(x[best_i], means[best_i] + stds[best_i] + 0.01),
                        ha="center", fontsize=15, color="goldenrod")

        n_classes = 4 if ds == "agnews" else 2
        ax.axhline(1 / n_classes, color="gray", ls=":", linewidth=1.2, label="Chance")

        ax.set_xticks(x)
        ax.set_xticklabels(row_labels, fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"Mean OOD accuracy ({metric})", fontsize=15)
        ax.set_title(f"{DATASET_LABELS.get(ds, ds)} — {MECH_LABELS.get(mech, mech)}",
                     fontsize=12)
        ax.grid(True, axis="y", alpha=0.35)
        ax.legend(fontsize=11)

    for idx in range(n_panels, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = out_dir / f"lambda_aggregate_{metric}.{ext}"
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"Saved: {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table – LaTeX
# ---------------------------------------------------------------------------

def make_latex_table(data: dict, out_dir: Path, metric: str = "final") -> str:
    """
    One row per (dataset × mechanism × λ/ERM),
    columns = noise levels.
    Values: mean OOD accuracy (averaged over correlation levels).
    """
    datasets  = sorted(data.keys())
    all_mechs = sorted({m for d in data.values() for m in d})
    all_noises = sorted({
        n
        for ds in data.values()
        for mech in ds.values()
        for corr in mech.values()
        for n in corr
    })

    row_keys   = ["erm"] + [f"irm_{lam}" for lam in LAMBDAS]
    row_labels = ["ERM"] + [f"IRM $\\lambda$={lam}" for lam in LAMBDAS]

    col_header = " & ".join([f"noise={n:.2f}" for n in all_noises])
    ncols = len(all_noises)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\small",
        r"\caption{Hyperparameter search: mean OOD accuracy (" + metric + r") per $\lambda$, "
        r"averaged over spurious-correlation levels. \textbf{Bold} = best IRM per column.}",
        r"\label{tab:hparam_search}",
        r"\begin{tabular}{lll" + "c" * ncols + "}",
        r"\toprule",
        r"Dataset & Mechanism & Model & " + col_header + r" \\",
        r"\midrule",
    ]

    for ds in datasets:
        ds_label = DATASET_LABELS.get(ds, ds)
        for mech in all_mechs:
            mech_data = data.get(ds, {}).get(mech, {})
            if not mech_data:
                continue
            mech_label = MECH_LABELS.get(mech, mech)

            # Compute means per (key, noise)
            table: dict[str, dict[float, float]] = {}
            for key in row_keys:
                table[key] = {}
                for noise in all_noises:
                    vals = [mech_data[ct][noise][key][metric]
                            for ct in mech_data
                            if noise in mech_data[ct] and key in mech_data[ct][noise]]
                    table[key][noise] = np.nanmean(vals) if vals else float("nan")

            # Determine best IRM per noise column
            best_irm: dict[float, str] = {}
            for noise in all_noises:
                irm_vals = {k: table[k][noise] for k in row_keys[1:]
                            if not np.isnan(table[k][noise])}
                if irm_vals:
                    best_irm[noise] = max(irm_vals, key=irm_vals.get)

            first_row = True
            for ki, (key, label) in enumerate(zip(row_keys, row_labels)):
                cells = []
                for noise in all_noises:
                    v = table[key].get(noise, float("nan"))
                    if np.isnan(v):
                        cells.append("—")
                    else:
                        txt = f"{v:.3f}"
                        if key != "erm" and best_irm.get(noise) == key:
                            txt = r"\textbf{" + txt + "}"
                        cells.append(txt)

                ds_str   = ds_label if first_row else ""
                mech_str = mech_label if first_row else ""
                first_row = False

                # Separator before ERM row
                if ki == 0 and lines[-1] != r"\midrule":
                    lines.append(r"\midrule")
                row = f"{ds_str} & {mech_str} & {label} & " + " & ".join(cells) + r" \\"
                lines.append(row)

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    tex = "\n".join(lines)

    out_path = out_dir / f"hparam_table_{metric}.tex"
    out_path.write_text(tex)
    print(f"Saved: {out_path}")
    return tex


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def make_parser():
    p = argparse.ArgumentParser(description="Plot hyperparameter search results")
    p.add_argument("--run_name", type=str, default="myrun",
                   help="Subfolder name under plots/grand_test/ (default: myrun)")
    p.add_argument("--run_dir", type=str, default=None,
                   help="Explicit path to the run directory (overrides --run_name)")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory (default: same as run_dir)")
    p.add_argument("--metric", type=str, default="final", choices=["final", "best"],
                   help="Which metric to use: 'final' (last step) or 'best' (peak).")
    return p


def main():
    args = make_parser().parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = ROOT_GRAND_TEST / args.run_name

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {run_dir}")
    data = load_run(run_dir)
    if not data:
        print("No results found — nothing to plot.")
        return

    n_cells = sum(
        1
        for ds in data.values()
        for mech in ds.values()
        for corr in mech.values()
        for _ in corr
    )
    print(f"Loaded {n_cells} cells across {len(data)} datasets.")

    metric = args.metric
    print(f"\n--- Figure 1: Lambda sensitivity line plots ({metric}) ---")
    plot_lambda_sensitivity(data, out_dir, metric=metric)

    print(f"\n--- Figure 2: Lambda × noise heat-maps ({metric}) ---")
    plot_lambda_heatmap(data, out_dir, metric=metric)

    print(f"\n--- Figure 3: Aggregated bar chart ({metric}) ---")
    plot_lambda_aggregate(data, out_dir, metric=metric)

    print(f"\n--- Table: LaTeX ({metric}) ---")
    make_latex_table(data, out_dir, metric=metric)

    print("\nDone.")


if __name__ == "__main__":
    main()
