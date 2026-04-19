"""
Argparse parser for NLP experiments (SMS Spam & AG News).

Usage:
    from args_nlp import make_nlp_parser
    args = make_nlp_parser().parse_args()
"""

import sys
from pathlib import Path as _Path

# Make base_parser() available (shared training args)
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT / 'synthetic') not in sys.path:
    sys.path.insert(0, str(_ROOT / 'synthetic'))

import argparse
from args_synthetic import base_parser


def make_nlp_parser() -> argparse.ArgumentParser:
    """Full parser for main_nlp.py (all NLP datasets)."""
    p = argparse.ArgumentParser(
        description="NLP experiments – SMS Spam, AG News & SST-2",
        parents=[base_parser()],
    )

    p.add_argument('--dataset', required=True, choices=[
        # ── SMS Spam (causal : X → Y) ──
        'nlp_sms_spam',
        'nlp_sms_spam_size_selection',
        'nlp_sms_spam_conf_varying_proxy',
        'nlp_sms_spam_conf_varying_gamma',
        'nlp_sms_spam_conf_varying_pc',
        # ── AG News (causal : X → Y) ──
        'nlp_agnews_semi_anti_causal',
        'nlp_agnews_source_selection',
        'nlp_agnews_conf_varying_proxy',
        # ── SST-2 (anti-causal : Y → X) ──
        'nlp_sst2_semi_anti_causal',
        'nlp_sst2_selection',
        'nlp_sst2_genre_selection',
        'nlp_sst2_conf_varying_proxy',
    ])

    # ---- BERT config ----
    p.add_argument('--nlp_bert_model', type=str, default='bert-base-uncased')
    p.add_argument('--nlp_max_length', type=int, default=128,
                   help='Max token length for BERT (use 256 for AG News)')
    p.add_argument('--nlp_pooling',    type=str, default='mean',
                   choices=['mean', 'cls'])

    # ---- Semi anti-causal ----
    p.add_argument('--nlp_p_correct_train', type=float, nargs='+',
                   default=[0.99, 0.9],
                   help='P(correct spurious token) per train env')
    p.add_argument('--nlp_p_correct_test', type=float, default=0.0,
                   help='P(correct spurious token) for OOD test env')
    p.add_argument('--nlp_label_flip', type=float, default=0.0,
                   help='Symmetric label flip rate')

    # ---- Selection bias ----
    p.add_argument('--nlp_selection_p_train', type=float, nargs='+',
                   default=[0.9, 0.8],
                   help='Keep-probability per train env (selection datasets)')
    p.add_argument('--nlp_sst2_ood_strategy', type=str, default='cross_label',
                   choices=['cross_label', 'atypical'],
                   help='OOD strategy for sst2_selection: '
                        'cross_label=mots forts contredisent le label (adversarial), '
                        'atypical=reviews sans aucun marqueur lexical')

    # ---- Size selection ----
    p.add_argument('--nlp_size_threshold_method', type=str, default='quartile',
                   choices=['quartile', 'median', 'soft'])

    # ---- AG News source selection ----
    p.add_argument('--nlp_n_ood_per_class', type=int, default=250,
                   help='Max OOD examples per class (agnews_source_selection)')

    # ---- Confounding variants ----
    p.add_argument('--nlp_conf_a_train', type=float, nargs='+', default=[0.01, 0.11],
                   help='Proxy noise a_e per train env (conf_varying_proxy)')
    p.add_argument('--nlp_conf_a_test', type=float, default=0.99,
                   help='Proxy noise a_e for OOD test (conf_varying_proxy)')
    p.add_argument('--nlp_conf_a', type=float, default=0.0,
                   help='Fixed proxy noise a (conf_varying_gamma / conf_varying_pc)')
    p.add_argument('--nlp_conf_gamma_train', type=float, nargs='+', default=[0.8, 0.5],
                   help='C→Y flip prob per train env (conf_varying_gamma)')
    p.add_argument('--nlp_conf_gamma_test', type=float, default=0.0,
                   help='C→Y flip prob for OOD test (conf_varying_gamma)')
    p.add_argument('--nlp_conf_gamma', type=float, default=0.5,
                   help='Fixed C→Y flip prob (conf_varying_proxy / conf_varying_pc)')
    p.add_argument('--nlp_conf_pc_train', type=float, nargs='+', default=[0.8, 0.5],
                   help='Prevalence of C per train env (conf_varying_pc)')
    p.add_argument('--nlp_conf_pc_test', type=float, default=0.1,
                   help='Prevalence of C for OOD test (conf_varying_pc)')

    # ---- Output ----
    p.add_argument('--plot_dir', type=str, default=None,
                   help='Plot output directory. Default: nlp_synthetic/plots/<dataset>/')

    return p
