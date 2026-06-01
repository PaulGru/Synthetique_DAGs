"""
Argparse parser for NLP experiments (AG News, IMDB Genres, Amazon Books).
"""

import argparse


def base_parser() -> argparse.ArgumentParser:
    """Return a parent parser with all arguments shared between main.py and sweeps.

    Use add_help=False so the child parser can add its own -h.
    """
    p = argparse.ArgumentParser(add_help=False)

    # ---- Data dimensions ----
    p.add_argument('--n',          type=int,   default=200_000, help='Samples per train env')
    p.add_argument('--n_test',     type=int,   default=10_000,  help='Test OOD size')
    p.add_argument('--val_frac',   type=float, default=0.1,     help='Validation fraction')
    p.add_argument('--dim_z',      type=int,   default=5,       help='Causal feature dimension')
    p.add_argument('--dim_y',      type=int,   default=5,       help='Spurious feature dimension')

    # ---- Label noise ----
    p.add_argument('--label_flip', type=float, default=0.25,
                   help='Symmetric label flip rate (SAC / Selection)')

    # ---- Confounding shared params ----
    p.add_argument('--conf_gamma',    type=float, default=1.0,
                   help='confounding_varying_proxy: causal weight γ (fixed)')

    # ---- Anti-causal class imbalance ----
    p.add_argument('--ac_py_train', type=float, nargs='+', default=None,
                   help='P(Y*=1) per training env for ac_semi_anti_causal / ac_selection.')
    p.add_argument('--ac_proxy_pc_train', type=float, nargs='+', default=None,
                   help='P(C=1) per training env for synthetic_ac_confounding_varying_proxy.')
    p.add_argument('--ac_proxy_pc_test', type=float, default=0.35,
                   help='P(C=1) for the test env in synthetic_ac_confounding_varying_proxy.')

    # ---- Training ----
    p.add_argument('--erm_steps',  type=int,   default=25_000)
    p.add_argument('--erm_lr',     type=float, default=5e-3)
    p.add_argument('--erm_batch',  type=int,   default=512)
    p.add_argument('--irm_steps',  type=int,   default=25_000)
    p.add_argument('--irm_lr',     type=float, default=5e-3)
    p.add_argument('--irm_lambda', type=float, default=200.0)
    p.add_argument('--irm_batch',  type=int,   default=512)
    p.add_argument('--seed',       type=int,  default=1)
    p.add_argument('--device',     type=str,  default='auto')
    p.add_argument('--eval_every', type=int,  default=100)

    return p


def make_nlp_parser() -> argparse.ArgumentParser:
    """Full parser for main_nlp.py (all NLP datasets)."""
    p = argparse.ArgumentParser(
        description="NLP experiments – AG News, IMDB Genres & Amazon Books",
        parents=[base_parser()],
    )

    p.add_argument('--dataset', required=True, choices=[
        # AG News (causal)
        'nlp_agnews_semi_anti_causal',
        'nlp_agnews_size_selection',
        'nlp_agnews_conf_varying_proxy',
        # IMDB Genres (anti-causal)
        'nlp_imdb_genres_semi_anti_causal',
        'nlp_imdb_genres_size_selection',
        'nlp_imdb_genres_conf_varying_proxy',
        # Amazon Books (causal)
        'nlp_amazon_semi_anti_causal',
        'nlp_amazon_sentiment_selection',
        'nlp_amazon_conf_varying_proxy',
    ])

    # ---- BERT config ----
    p.add_argument('--nlp_bert_model', type=str, default='distilbert-base-uncased')
    p.add_argument('--nlp_max_length', type=int, default=128,
                   help='Max token length for BERT (use 256 for AG News)')
    p.add_argument('--nlp_pooling',    type=str, default='mean',
                   choices=['mean', 'cls'])
    p.add_argument('--finetune_bert_layers', type=int, default=0,
                   help='Number of final BERT layers to fine-tune (0 = frozen embeddings)')

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

    # ---- Size selection ----
    p.add_argument('--nlp_size_threshold_method', type=str, default='quartile',
                   choices=['quartile', 'median', 'soft'])

    # ---- AG News source selection ----
    p.add_argument('--nlp_n_ood_per_class', type=int, default=250,
                   help='Max OOD examples per class (agnews_size_selection)')

    # ---- AG News class imbalance (4 classes) ----
    p.add_argument('--nlp_agnews_class_dist_train', type=float, nargs='+', default=None,
                   help='Target class distribution per train env for AG News (4 floats per env, '
                        'flattened). Ex for 2 envs: 0.1 0.1 0.4 0.4 0.2 0.2 0.3 0.3. '
                        'None = balanced (no resampling).')
    p.add_argument('--nlp_agnews_class_dist_test', type=float, nargs='+', default=None,
                   help='Target class distribution for AG News test set (4 floats). '
                        'Ex: 0.4 0.4 0.1 0.1. None = no resampling.')

    # ---- Confounding variants ----
    p.add_argument('--nlp_conf_a_train', type=float, nargs='+', default=[0.01, 0.11],
                   help='Proxy noise a_e per train env (conf_varying_proxy)')
    p.add_argument('--nlp_conf_a_test', type=float, default=0.99,
                   help='Proxy noise a_e for OOD test (conf_varying_proxy)')
    p.add_argument('--nlp_conf_a', type=float, default=0.0,
                   help='Fixed proxy noise a (conf_varying_proxy)')
    p.add_argument('--nlp_conf_gamma', type=float, default=0.5,
                   help='Fixed C→Y alignment strength (conf_varying_proxy)')
    p.add_argument('--nlp_conf_p_c_flip', type=float, default=0.25,
                   help='Prevalence of the binary confounder (confounding datasets)')

    # ---- IMDB class imbalance ----
    p.add_argument('--nlp_imdb_class_ratio_train', type=float, nargs='+', default=None,
                   help='Fraction of positives per train env for IMDB datasets '
                        '(ex: 0.2 0.8). None = balanced (no resampling).')
    p.add_argument('--nlp_imdb_class_ratio_test', type=float, default=None,
                   help='Fraction of positives in test set for IMDB datasets '
                        '(ex: 0.5). None = no resampling.')

    # ---- Amazon Books ----
    p.add_argument('--nlp_amazon_n_target', type=int, default=100_000,
                   help='Number of Amazon Books reviews to load (balanced, default 100k)')

    # ---- Amazon class imbalance ----
    p.add_argument('--nlp_amazon_class_ratio_train', type=float, nargs='+', default=None,
                   help='Fraction of positives per train env for Amazon datasets '
                        '(ex: 0.2 0.3). None = balanced (no resampling).')
    p.add_argument('--nlp_amazon_class_ratio_test', type=float, default=None,
                   help='Fraction of positives in test set for Amazon datasets '
                        '(ex: 0.9). None = no resampling.')

    # ---- Output ----
    p.add_argument('--plot_dir', type=str, default=None,
                   help='Plot output directory. Default: nlp_synthetic/plots/<dataset>/')

    return p
