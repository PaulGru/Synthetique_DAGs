"""
plot_grand_test.py
==================
Visualise grand test results: OOD accuracy (ERM vs IRM) as a function of
noise level, faceted by dataset x mechanism, with one sub-panel per
correlation level.

Usage
-----
    python nlp_synthetic/plot_grand_test.py
    python nlp_synthetic/plot_grand_test.py --run_name grand_20260521
    python nlp_synthetic/plot_grand_test.py --run_name grand_20260521 --metric best
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1] / "results" / "grand_test"

MECH_LABELS = {
    "sac":       "SAC (token injection)",
    "selection": "Size selection",
    "conf":      "Confounding",
}
DATASET_LABELS = {
    "agnews":  "AG News (4-class)",
    "imdb":    "IMDB Genres (binary)",
    "amazon":  "Amazon Books (binary)",
}

ERM_COLOR = "#e15759"   # red
IRM_COLOR = "#4e79a7"   # blue
FINAL_ALPHA = 1.0
BEST_ALPHA  = 0.45
FINAL_LS    = "-"
BEST_LS     = "--"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_cell_name(name: str):
    """
    Parse a cell directory name like 'noise0p100__p1p00_0p80' or
    'noise0p200__a0p010_0p100'.

    Returns (noise_float, corr_tag_str, corr_vals_tuple).
    """
    m = re.match(r"noise(\d+p\d+)__(.+)", name)
    if not m:
        return None
    noise_str, corr_str = m.group(1), m.group(2)
    noise = float(noise_str.replace("p", "."))
    return noise, corr_str


def _load_run(run_dir: Path, metric: str = "final"):
    """
    Walk a grand_test run directory and return a nested dict:
        data[dataset_slug][mech][corr_tag][noise] = {erm, irm, delta}
    where erm/irm are the mean across seeds.
    """
    data = {}
    for ds_dir in sorted(run_dir.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_slug = ds_dir.name  # e.g. 'agnews_sac', 'imdb_conf'

        # Infer dataset and mechanism from slug
        parts = ds_slug.rsplit("_", maxsplit=1)
        if len(parts) == 2 and parts[1] in ("sac", "conf"):
            dataset, mech = parts
        elif ds_slug.endswith("_size_selection"):
            dataset = ds_slug[: -len("_size_selection")]
            mech = "selection"
        elif ds_slug.endswith("_sentiment_selection"):
            dataset = ds_slug[: -len("_sentiment_selection")]
            mech = "selection"
        else:
            # fallback: whole slug as dataset
            dataset, mech = ds_slug, "unknown"

        for cell_dir in sorted(ds_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            rfile = cell_dir / "results.json"
            if not rfile.exists():
                continue

            parsed = _parse_cell_name(cell_dir.name)
            if parsed is None:
                continue
            noise, corr_tag = parsed

            with open(rfile) as f:
                res = json.load(f)

            s = res["summary"]
            erm_key = f"erm_{metric}"
            irm_key = f"irm_{metric}"
            erm_val = s[erm_key]["mean"] if erm_key in s else float("nan")
            irm_val = s[irm_key]["mean"] if irm_key in s else float("nan")

            data.setdefault(dataset, {}).setdefault(mech, {}).setdefault(corr_tag, {})[noise] = {
                "erm": erm_val,
                "irm": irm_val,
                "delta": irm_val - erm_val,
                "n_seeds": len(res["per_seed"]),
                "erm_std": s[erm_key]["std"] if erm_key in s else 0.0,
                "irm_std": s[irm_key]["std"] if irm_key in s else 0.0,
            }

    return data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_run(run_dir: Path, metric: str = "final", out_dir: Path | None = None):
    data = _load_run(run_dir, metric=metric)
    if not data:
        print(f"No results found in {run_dir}")
        return

    if out_dir is None:
        out_dir = run_dir

    datasets = sorted(data.keys())
    mechs    = sorted({m for d in data.values() for m in d})

    n_rows = len(datasets)
    n_cols = len(mechs)

    if n_rows == 0 or n_cols == 0:
        print("Nothing to plot.")
        return

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    fig.suptitle(
        f"Grand Test — OOD accuracy ({metric}) vs label noise\n"
        f"Run: {run_dir.name}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for row_i, dataset in enumerate(datasets):
        for col_j, mech in enumerate(mechs):
            ax = axes[row_i][col_j]
            mech_data = data.get(dataset, {}).get(mech, {})

            if not mech_data:
                ax.set_visible(False)
                continue

            corr_tags = sorted(mech_data.keys())

            # Map corr_tag -> readable label
            # e.g. "p1p00_0p80" -> "p=(1.00, 0.80)" ; "a0p010_0p100" -> "a=(0.01, 0.10)"
            def _corr_label(tag):
                nums = re.findall(r"\d+p\d+", tag)
                vals = [float(n.replace("p", ".")) for n in nums]
                if tag.startswith("p"):
                    return "corr=({})".format(", ".join(f"{v:.2f}" for v in vals))
                else:
                    return "a=({})".format(", ".join(f"{v:.3f}" for v in vals))

            # Use different line styles per corr level
            corr_styles = ["-", "--", ":", "-."]

            for ci, corr_tag in enumerate(corr_tags):
                cell_data = mech_data[corr_tag]
                noises = sorted(cell_data.keys())
                xs = np.array(noises)

                erm_means = np.array([cell_data[n]["erm"]   for n in noises])
                irm_means = np.array([cell_data[n]["irm"]   for n in noises])
                erm_stds  = np.array([cell_data[n]["erm_std"] for n in noises])
                irm_stds  = np.array([cell_data[n]["irm_std"] for n in noises])

                ls = corr_styles[ci % len(corr_styles)]
                clabel = _corr_label(corr_tag)

                ax.plot(xs, erm_means, color=ERM_COLOR, ls=ls, marker="o",
                        linewidth=2, markersize=6,
                        label=f"ERM  {clabel}")
                ax.fill_between(xs, erm_means - erm_stds, erm_means + erm_stds,
                                color=ERM_COLOR, alpha=0.12)

                ax.plot(xs, irm_means, color=IRM_COLOR, ls=ls, marker="s",
                        linewidth=2, markersize=6,
                        label=f"IRM  {clabel}")
                ax.fill_between(xs, irm_means - irm_stds, irm_means + irm_stds,
                                color=IRM_COLOR, alpha=0.12)

            # Chance level
            n_classes = 4 if dataset == "agnews" else 2
            chance = 1.0 / n_classes
            ax.axhline(chance, color="gray", ls=":", linewidth=1.2, label=f"Chance ({chance:.0%})")

            ax.set_xlabel("Label noise", fontsize=11)
            ax.set_ylabel(f"OOD accuracy ({metric})", fontsize=11)
            ds_label   = DATASET_LABELS.get(dataset, dataset)
            mech_label = MECH_LABELS.get(mech, mech)
            ax.set_title(f"{ds_label}\n{mech_label}", fontsize=12)
            ax.set_xticks(sorted({n for cd in mech_data.values() for n in cd}))
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8, loc="lower left")
            ax.grid(True, alpha=0.35)

    fig.tight_layout()
    out_path = out_dir / f"grand_test_accuracy_{metric}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    out_png = out_dir / f"grand_test_accuracy_{metric}.png"
    fig.savefig(out_png, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {out_png}")

    # ------------------------------------------------------------------
    # Also plot IRM - ERM delta
    # ------------------------------------------------------------------
    fig2, axes2 = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )
    fig2.suptitle(
        f"Grand Test — Δ accuracy (IRM − ERM, {metric}) vs label noise\n"
        f"Run: {run_dir.name}",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for row_i, dataset in enumerate(datasets):
        for col_j, mech in enumerate(mechs):
            ax = axes2[row_i][col_j]
            mech_data = data.get(dataset, {}).get(mech, {})

            if not mech_data:
                ax.set_visible(False)
                continue

            corr_tags  = sorted(mech_data.keys())
            corr_styles = ["-", "--", ":", "-."]
            # collect all noises for xlim
            all_noises = sorted({n for cd in mech_data.values() for n in cd})

            for ci, corr_tag in enumerate(corr_tags):
                cell_data = mech_data[corr_tag]
                noises = sorted(cell_data.keys())
                xs = np.array(noises)
                deltas = np.array([cell_data[n]["delta"] for n in noises])

                ls = corr_styles[ci % len(corr_styles)]

                def _corr_label(tag):
                    nums = re.findall(r"\d+p\d+", tag)
                    vals = [float(n.replace("p", ".")) for n in nums]
                    if tag.startswith("p"):
                        return "corr=({})".format(", ".join(f"{v:.2f}" for v in vals))
                    else:
                        return "a=({})".format(", ".join(f"{v:.3f}" for v in vals))

                ax.plot(xs, deltas, color="#59a14f", ls=ls, marker="D",
                        linewidth=2, markersize=6, label=_corr_label(corr_tag))

            ax.axhline(0, color="black", ls="-", linewidth=0.8)
            ax.set_xlabel("Label noise", fontsize=11)
            ax.set_ylabel(f"IRM − ERM ({metric})", fontsize=11)
            ds_label   = DATASET_LABELS.get(dataset, dataset)
            mech_label = MECH_LABELS.get(mech, mech)
            ax.set_title(f"{ds_label}\n{mech_label}", fontsize=12)
            ax.set_xticks(all_noises)
            ax.legend(fontsize=9, loc="upper right")
            ax.grid(True, alpha=0.35)

    fig2.tight_layout()
    out_delta_pdf = out_dir / f"grand_test_delta_{metric}.pdf"
    fig2.savefig(out_delta_pdf, bbox_inches="tight")
    out_delta_png = out_dir / f"grand_test_delta_{metric}.png"
    fig2.savefig(out_delta_png, bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"Saved: {out_delta_pdf}")
    print(f"Saved: {out_delta_png}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot grand test results.")
    parser.add_argument("--run_name", default=None,
                        help="Name of the grand test run directory (default: latest).")
    parser.add_argument("--metric", choices=["final", "best"], default="final",
                        help="Which metric to plot: 'final' (last step) or 'best' (best val).")
    args = parser.parse_args()

    if args.run_name:
        run_dir = ROOT / args.run_name
    else:
        # Use most recent run
        runs = sorted(ROOT.iterdir()) if ROOT.exists() else []
        runs = [r for r in runs if r.is_dir()]
        if not runs:
            print(f"No runs found in {ROOT}")
            return
        run_dir = runs[-1]
        print(f"Using run: {run_dir.name}")

    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}")
        return

    # Plot both metrics
    for metric in ("final", "best"):
        plot_run(run_dir, metric=metric, out_dir=run_dir)


if __name__ == "__main__":
    main()
