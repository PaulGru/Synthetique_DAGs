import sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in [str(_ROOT), str(_ROOT / 'shared'), str(_ROOT / 'nlp_synthetic'), str(_ROOT / 'synthetic')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os
import torch
from args_nlp import make_nlp_parser
from data_nlp import (
    build_envs_nlp_semi_anti_causal,
    build_envs_nlp_size_selection,
    build_envs_nlp_conf_varying_proxy,
    build_envs_nlp_conf_varying_gamma,
    build_envs_nlp_conf_varying_pc,
    build_envs_ag_news_semi_anti_causal,
    AG_NEWS_WRONG_CLASS,
    build_envs_ag_news_source_selection,
    build_envs_ag_news_keyword_selection,
    build_envs_ag_news_conf_varying_proxy,
    # SST-2 (anti-causal : Y → X)
    build_envs_sst2_semi_anti_causal,
    build_envs_sst2_selection,
    build_envs_sst2_size_selection,
    build_envs_sst2_conf_varying_proxy,
    # IMDB (anti-causal : Y → X, textes longs)
    build_envs_imdb_conf_varying_proxy,
    build_envs_imdb_semi_anti_causal,
    build_envs_imdb_selection,
    build_envs_imdb_size_selection,
    build_envs_imdb_br_selection,
    build_envs_imdb_genres_size_selection,
    build_envs_imdb_genres_semi_anti_causal,
    build_envs_imdb_genres_conf_varying_proxy,
    # Amazon Books (anti-causal : X → Y)
    build_envs_amazon_semi_anti_causal,
    build_envs_amazon_size_selection,
    build_envs_amazon_conf_varying_proxy,
    build_envs_amazon_rating_natural,
    build_envs_amazon_keyword_selection,
    build_envs_amazon_sentiment_selection,
    # Amazon Reviews Polarity 2013 (textes courts, catégorie comme signal trompeur)
    build_envs_amazon_category_selection,
)
from models_training import train_erm, train_irm, train_ibirm
from utils_irm import resolve_device
from visualization_synth import (
    plot_accuracy_curves,
    plot_loss_curves,
    plot_summary_panel,
    plot_results_table,
)
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    args = make_nlp_parser().parse_args()

    device     = resolve_device(args.device)
    device_str = str(device)

    # Dossier de plots – nommage harmonisé (préfixe causal_ / ac_)
    _SLUG_MAP = {
        'nlp_sms_spam':                       'causal_sms_spam_sac',
        'nlp_sms_spam_size_selection':        'causal_sms_spam_size_selection',
        'nlp_sms_spam_conf_varying_proxy':    'causal_sms_spam_conf_varying_proxy',
        'nlp_sms_spam_conf_varying_gamma':    'causal_sms_spam_conf_varying_gamma',
        'nlp_sms_spam_conf_varying_pc':       'causal_sms_spam_conf_varying_pc',
        'nlp_agnews_semi_anti_causal':        'causal_agnews_sac',
        'nlp_agnews_semi_anti_causal_fixed_wrong': 'causal_agnews_sac_fixed_wrong',
        'nlp_agnews_source_selection':        'causal_agnews_source_selection',
        'nlp_agnews_keyword_selection':       'causal_agnews_keyword_selection',
        'nlp_agnews_conf_varying_proxy':      'causal_agnews_conf_varying_proxy',
        'nlp_sst2_semi_anti_causal':          'ac_sst2_sac',
        'nlp_sst2_selection':                 'ac_sst2_selection',
        'nlp_sst2_size_selection':            'ac_sst2_size_selection',
        'nlp_sst2_conf_varying_proxy':        'ac_sst2_conf_varying_proxy',
        'nlp_imdb_conf_varying_proxy':         'ac_imdb_conf_varying_proxy',
        'nlp_imdb_semi_anti_causal':            'ac_imdb_sac',
        'nlp_imdb_selection':                   'ac_imdb_selection',
        'nlp_imdb_size_selection':              'ac_imdb_size_selection',
        'nlp_imdb_br_selection':                'ac_imdb_br_selection',
        'nlp_imdb_genres_size_selection':        'ac_imdb_genres_size_selection',
        'nlp_imdb_genres_semi_anti_causal':      'ac_imdb_genres_semi_anti_causal',
        'nlp_imdb_genres_conf_varying_proxy':    'ac_imdb_genres_conf_varying_proxy',
        'nlp_amazon_semi_anti_causal':          'causal_amazon_sac',
        'nlp_amazon_size_selection':            'causal_amazon_size_selection',
        'nlp_amazon_conf_varying_proxy':        'causal_amazon_conf_varying_proxy',
        'nlp_amazon_rating_natural':            'causal_amazon_rating_natural',
        'nlp_amazon_keyword_selection':          'causal_amazon_keyword_selection',
        'nlp_amazon_sentiment_selection':        'causal_amazon_sentiment_selection',
        'nlp_amazon_category_selection':         'ac_amazon_category_selection',
    }
    slug     = _SLUG_MAP.get(args.dataset, args.dataset.replace('nlp_', ''))
    plot_dir = args.plot_dir if args.plot_dir else os.path.join(
        str(_ROOT / 'nlp_synthetic' / 'plots'), slug
    )
    os.makedirs(plot_dir, exist_ok=True)

    n_classes = 4 if args.dataset.startswith('nlp_agnews') else 2

    # ── Parse AG News class distribution (flat list → per-env lists) ──────
    agnews_class_dist_train = None
    agnews_class_dist_test  = args.nlp_agnews_class_dist_test
    if args.nlp_agnews_class_dist_train is not None:
        flat = args.nlp_agnews_class_dist_train
        if len(flat) % 4 != 0:
            raise ValueError(f"--nlp_agnews_class_dist_train must have a multiple of 4 floats, got {len(flat)}")
        agnews_class_dist_train = [flat[k:k+4] for k in range(0, len(flat), 4)]

    # ── Data ─────────────────────────────────────────────────────────────────
    if args.dataset == 'nlp_sms_spam':
        train_envs, val_envs, test_env = build_envs_nlp_semi_anti_causal(
            n=0,  # ignored by the function
            train_p_correct=list(args.nlp_p_correct_train),
            test_p_correct=args.nlp_p_correct_test,
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_sms_spam_size_selection':
        train_envs, val_envs, test_env = build_envs_nlp_size_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            threshold_method=args.nlp_size_threshold_method,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_sms_spam_conf_varying_proxy':
        train_envs, val_envs, test_env = build_envs_nlp_conf_varying_proxy(
            a_train=list(args.nlp_conf_a_train),
            a_test=args.nlp_conf_a_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            gamma=args.nlp_conf_gamma,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_sms_spam_conf_varying_gamma':
        train_envs, val_envs, test_env = build_envs_nlp_conf_varying_gamma(
            gamma_train=list(args.nlp_conf_gamma_train),
            gamma_test=args.nlp_conf_gamma_test,
            seed=args.seed,
            a=args.nlp_conf_a,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_sms_spam_conf_varying_pc':
        train_envs, val_envs, test_env = build_envs_nlp_conf_varying_pc(
            pc_train=list(args.nlp_conf_pc_train),
            pc_test=args.nlp_conf_pc_test,
            seed=args.seed,
            a=args.nlp_conf_a,
            gamma=args.nlp_conf_gamma,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_agnews_semi_anti_causal':
        train_envs, val_envs, test_env = build_envs_ag_news_semi_anti_causal(
            train_p_correct=list(args.nlp_p_correct_train),
            test_p_correct=args.nlp_p_correct_test,
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            class_dist_train=agnews_class_dist_train,
            class_dist_test=agnews_class_dist_test,
        )

    elif args.dataset == 'nlp_agnews_semi_anti_causal_fixed_wrong':
        train_envs, val_envs, test_env = build_envs_ag_news_semi_anti_causal(
            train_p_correct=list(args.nlp_p_correct_train),
            test_p_correct=args.nlp_p_correct_test,
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            class_dist_train=agnews_class_dist_train,
            class_dist_test=agnews_class_dist_test,
            wrong_class_map=AG_NEWS_WRONG_CLASS,
        )

    elif args.dataset == 'nlp_agnews_source_selection':
        train_envs, val_envs, test_env = build_envs_ag_news_source_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            n_ood_per_class=args.nlp_n_ood_per_class,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            class_dist_train=agnews_class_dist_train,
            class_dist_test=agnews_class_dist_test,
        )

    elif args.dataset == 'nlp_agnews_keyword_selection':
        train_envs, val_envs, test_env = build_envs_ag_news_keyword_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            threshold_method=args.nlp_size_threshold_method,
            label_flip=args.nlp_label_flip,
            n_ood_per_class=args.nlp_n_ood_per_class,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            class_dist_train=agnews_class_dist_train,
            class_dist_test=agnews_class_dist_test,
        )

    elif args.dataset == 'nlp_agnews_conf_varying_proxy':
        train_envs, val_envs, test_env = build_envs_ag_news_conf_varying_proxy(
            a_train=list(args.nlp_conf_a_train),
            a_test=args.nlp_conf_a_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            gamma=args.nlp_conf_gamma,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_sst2_semi_anti_causal':
        train_envs, val_envs, test_env = build_envs_sst2_semi_anti_causal(
            train_p_correct=list(args.nlp_p_correct_train),
            test_p_correct=args.nlp_p_correct_test,
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_sst2_selection':
        train_envs, val_envs, test_env = build_envs_sst2_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            ood_strategy=args.nlp_sst2_ood_strategy,
        )

    elif args.dataset == 'nlp_sst2_size_selection':
        train_envs, val_envs, test_env = build_envs_sst2_size_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            threshold_method=args.nlp_size_threshold_method,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_sst2_conf_varying_proxy':
        train_envs, val_envs, test_env = build_envs_sst2_conf_varying_proxy(
            a_train=list(args.nlp_conf_a_train),
            a_test=args.nlp_conf_a_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            gamma=args.nlp_conf_gamma,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_imdb_conf_varying_proxy':
        train_envs, val_envs, test_env = build_envs_imdb_conf_varying_proxy(
            a_train=list(args.nlp_conf_a_train),
            a_test=args.nlp_conf_a_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            gamma=args.nlp_conf_gamma,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            class_ratio_train=args.nlp_imdb_class_ratio_train,
            class_ratio_test=args.nlp_imdb_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_imdb_semi_anti_causal':
        train_envs, val_envs, test_env = build_envs_imdb_semi_anti_causal(
            train_p_correct=list(args.nlp_p_correct_train),
            test_p_correct=args.nlp_p_correct_test,
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            class_ratio_train=args.nlp_imdb_class_ratio_train,
            class_ratio_test=args.nlp_imdb_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_imdb_selection':
        train_envs, val_envs, test_env = build_envs_imdb_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            ood_strategy=args.nlp_sst2_ood_strategy,
            class_ratio_train=args.nlp_imdb_class_ratio_train,
            class_ratio_test=args.nlp_imdb_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_imdb_size_selection':
        train_envs, val_envs, test_env = build_envs_imdb_size_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            threshold_method=args.nlp_size_threshold_method,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            class_ratio_train=args.nlp_imdb_class_ratio_train,
            class_ratio_test=args.nlp_imdb_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_imdb_br_selection':
        train_envs, val_envs, test_env = build_envs_imdb_br_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            class_ratio_train=args.nlp_imdb_class_ratio_train,
            class_ratio_test=args.nlp_imdb_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
            max_length_chars=args.nlp_max_length_chars,
        )

    elif args.dataset == 'nlp_imdb_genres_size_selection':
        train_envs, val_envs, test_env = build_envs_imdb_genres_size_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            threshold_method=args.nlp_size_threshold_method,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            class_ratio_train=args.nlp_imdb_class_ratio_train,
            class_ratio_test=args.nlp_imdb_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_imdb_genres_semi_anti_causal':
        train_envs, val_envs, test_env = build_envs_imdb_genres_semi_anti_causal(
            train_p_correct=list(args.nlp_p_correct_train),
            test_p_correct=args.nlp_p_correct_test,
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            class_ratio_train=args.nlp_imdb_class_ratio_train,
            class_ratio_test=args.nlp_imdb_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_imdb_genres_conf_varying_proxy':
        train_envs, val_envs, test_env = build_envs_imdb_genres_conf_varying_proxy(
            a_train=list(args.nlp_conf_a_train),
            a_test=args.nlp_conf_a_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            gamma=args.nlp_conf_gamma,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            class_ratio_train=args.nlp_imdb_class_ratio_train,
            class_ratio_test=args.nlp_imdb_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_amazon_semi_anti_causal':
        train_envs, val_envs, test_env = build_envs_amazon_semi_anti_causal(
            train_p_correct=list(args.nlp_p_correct_train),
            test_p_correct=args.nlp_p_correct_test,
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            n_target=args.nlp_amazon_n_target,
            class_ratio_train=args.nlp_amazon_class_ratio_train,
            class_ratio_test=args.nlp_amazon_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_amazon_size_selection':
        train_envs, val_envs, test_env = build_envs_amazon_size_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            threshold_method=args.nlp_size_threshold_method,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            n_target=args.nlp_amazon_n_target,
            class_ratio_train=args.nlp_amazon_class_ratio_train,
            class_ratio_test=args.nlp_amazon_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_amazon_conf_varying_proxy':
        train_envs, val_envs, test_env = build_envs_amazon_conf_varying_proxy(
            a_train=list(args.nlp_conf_a_train),
            a_test=args.nlp_conf_a_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            gamma=args.nlp_conf_gamma,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            n_target=args.nlp_amazon_n_target,
            class_ratio_train=args.nlp_amazon_class_ratio_train,
            class_ratio_test=args.nlp_amazon_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_amazon_rating_natural':
        train_envs, val_envs, test_env = build_envs_amazon_rating_natural(
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            n_per_group=args.nlp_amazon_n_per_group,
            class_ratio_train=args.nlp_amazon_class_ratio_train,
            class_ratio_test=args.nlp_amazon_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_amazon_keyword_selection':
        train_envs, val_envs, test_env = build_envs_amazon_keyword_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            n_target=args.nlp_amazon_n_target,
            ood_strategy=args.nlp_sst2_ood_strategy,
            class_ratio_train=args.nlp_amazon_class_ratio_train,
            class_ratio_test=args.nlp_amazon_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_amazon_sentiment_selection':
        train_envs, val_envs, test_env = build_envs_amazon_sentiment_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            n_target=args.nlp_amazon_n_target,
            class_ratio_train=args.nlp_amazon_class_ratio_train,
            class_ratio_test=args.nlp_amazon_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    elif args.dataset == 'nlp_amazon_category_selection':
        train_envs, val_envs, test_env = build_envs_amazon_category_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            n_target=args.nlp_amzpol_n_target,
            cat_typical_pos=args.nlp_amzpol_cat_pos,
            cat_typical_neg=args.nlp_amzpol_cat_neg,
            class_ratio_train=args.nlp_amzpol_class_ratio_train,
            class_ratio_test=args.nlp_amzpol_class_ratio_test,
            finetune_bert_layers=args.finetune_bert_layers,
        )

    else:
        raise ValueError(f"Unknown NLP dataset: {args.dataset}")

    # ── Training ──────────────────────────────────────────────────────────────
    erm, erm_hist = train_erm(
        envs=train_envs, val_envs=val_envs, test_env=test_env,
        steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
        seed=args.seed, device=device, eval_every=args.eval_every,
        dataset_name=args.dataset, n_classes=n_classes,
        finetune_bert_layers=args.finetune_bert_layers,
        bert_model_name=args.nlp_bert_model,
        use_mlp=args.use_mlp, mlp_hidden=args.mlp_hidden, mlp_dropout=args.mlp_dropout,
    )

    irm, irm_hist = train_irm(
        envs=train_envs, val_envs=val_envs, test_env=test_env,
        steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
        irm_lambda=args.irm_lambda,
        seed=args.seed, device=device, eval_every=args.eval_every,
        dataset_name=args.dataset, n_classes=n_classes,
        finetune_bert_layers=args.finetune_bert_layers,
        bert_model_name=args.nlp_bert_model,
        use_mlp=args.use_mlp, mlp_hidden=args.mlp_hidden, mlp_dropout=args.mlp_dropout,
    )

    ibirm, ibirm_hist = None, None
    if args.run_ibirm:
        ibirm, ibirm_hist = train_ibirm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.ibirm_steps, lr=args.ibirm_lr, batch=args.ibirm_batch,
            irm_lambda=args.ibirm_lambda, ib_lambda=args.ib_lambda,
            seed=args.seed, device=device, eval_every=args.eval_every,
            dataset_name=args.dataset, n_classes=n_classes,
            finetune_bert_layers=args.finetune_bert_layers,
            bert_model_name=args.nlp_bert_model,
            use_mlp=args.use_mlp, mlp_hidden=args.mlp_hidden, mlp_dropout=args.mlp_dropout,
        )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f'\n--- Saving plots to {plot_dir}/ ---')
    plot_accuracy_curves(
        erm_hist, irm_hist,
        os.path.join(plot_dir, '01_accuracy.png'), args.dataset,
        ibirm_hist=ibirm_hist,
    )
    plot_loss_curves(
        erm_hist, irm_hist,
        os.path.join(plot_dir, '02_loss.png'), args.dataset,
        ibirm_hist=ibirm_hist,
    )
    plot_summary_panel(
        erm_hist, irm_hist,
        os.path.join(plot_dir, '05_summary.png'), args.dataset,
        ibirm_hist=ibirm_hist,
    )
    plot_results_table(
        erm_hist, irm_hist,
        os.path.join(plot_dir, '06_results.png'),
        ibirm_hist=ibirm_hist,
    )
