"""
Aggregate per-seed result JSON files produced by main.py.

Usage:
    uv run nlp/aggregate_seeds.py
    uv run nlp/aggregate_seeds.py --datasets nlp_agnews_semi_anti_causal nlp_agnews_size_selection
"""
import sys
import json
import glob
import argparse
import numpy as np
from pathlib import Path

_ROOT     = Path(__file__).resolve().parents[1]
PLOTS_DIR = _ROOT / 'nlp' / 'plots'

SLUG_MAP = {
    'nlp_agnews_semi_anti_causal':             'causal_agnews_sac',
    'nlp_agnews_size_selection':               'causal_agnews_size_selection',
    'nlp_agnews_conf_varying_proxy':           'causal_agnews_conf_varying_proxy',
    'nlp_imdb_genres_semi_anti_causal':        'ac_imdb_genres_semi_anti_causal',
    'nlp_imdb_genres_size_selection':          'ac_imdb_genres_size_selection',
    'nlp_imdb_genres_conf_varying_proxy':      'ac_imdb_genres_conf_varying_proxy',
    'nlp_amazon_semi_anti_causal':             'causal_amazon_sac',
    'nlp_amazon_conf_varying_proxy':           'causal_amazon_conf_varying_proxy',
    'nlp_amazon_sentiment_selection':          'causal_amazon_sentiment_selection',
}

METRICS = [
    ('erm_final_test_acc', 'ERM  OOD acc (final)'),
    ('erm_best_val_acc',   'ERM  val acc (best) '),
    ('irm_final_test_acc', 'IRM  OOD acc (final)'),
    ('irm_best_val_acc',   'IRM  val acc (best) '),
]


def aggregate_dataset(dataset: str) -> dict | None:
    slug     = SLUG_MAP.get(dataset, dataset.replace('nlp_', ''))
    plot_dir = PLOTS_DIR / slug
    files    = sorted(glob.glob(str(plot_dir / 'results_seed*.json')))
    if not files:
        print(f'  [warning] no results_seed*.json found in {plot_dir}')
        return None

    results = [json.loads(Path(f).read_text()) for f in files]
    seeds   = [r['seed'] for r in results]
    summary: dict = {'dataset': dataset, 'seeds': seeds}
    for key, _ in METRICS:
        vals = [r[key] for r in results if key in r]
        summary[f'{key}_mean'] = float(np.mean(vals))
        summary[f'{key}_std']  = float(np.std(vals))
    return summary


def print_summary(dataset: str, s: dict) -> None:
    print(f"\n{'='*60}")
    print(f"Dataset : {dataset}")
    print(f"Seeds   : {s['seeds']}")
    print(f"{'-'*60}")
    for key, label in METRICS:
        mean = s[f'{key}_mean']
        std  = s[f'{key}_std']
        print(f"  {label}: {mean:.4f} ± {std:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aggregate per-seed results')
    parser.add_argument(
        '--datasets', nargs='+',
        default=list(SLUG_MAP.keys()),
        help='Dataset names to aggregate (default: all)',
    )
    args = parser.parse_args()

    for ds in args.datasets:
        s = aggregate_dataset(ds)
        if s is None:
            continue
        print_summary(ds, s)
        slug    = SLUG_MAP.get(ds, ds.replace('nlp_', ''))
        out     = PLOTS_DIR / slug / 'results_aggregated.json'
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(s, indent=2))
        print(f"  → Saved to {out}")

    print()
