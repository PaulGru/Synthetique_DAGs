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
    build_envs_ag_news_source_selection,
    build_envs_ag_news_conf_varying_proxy,
    # SST-2 (anti-causal : Y → X)
    build_envs_sst2_semi_anti_causal,
    build_envs_sst2_selection,
    build_envs_sst2_genre_selection,
    build_envs_sst2_conf_varying_proxy,
    # IMDB (anti-causal : Y → X, textes longs)
    build_envs_imdb_conf_varying_proxy,
    build_envs_imdb_semi_anti_causal,
    build_envs_imdb_selection,
    build_envs_imdb_size_selection,
    # Amazon Books (anti-causal : X → Y)
    build_envs_amazon_semi_anti_causal,
    build_envs_amazon_size_selection,
    build_envs_amazon_conf_varying_proxy,
    build_envs_amazon_rating_natural,
    build_envs_amazon_keyword_selection,
    build_envs_amazon_sentiment_selection,
)
from models_training import train_erm, train_irm
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
        'nlp_agnews_source_selection':        'causal_agnews_source_selection',
        'nlp_agnews_conf_varying_proxy':      'causal_agnews_conf_varying_proxy',
        'nlp_sst2_semi_anti_causal':          'ac_sst2_sac',
        'nlp_sst2_selection':                 'ac_sst2_selection',
        'nlp_sst2_genre_selection':           'ac_sst2_genre_selection',
        'nlp_sst2_conf_varying_proxy':        'ac_sst2_conf_varying_proxy',
        'nlp_imdb_conf_varying_proxy':         'ac_imdb_conf_varying_proxy',
        'nlp_imdb_semi_anti_causal':            'ac_imdb_sac',
        'nlp_imdb_selection':                   'ac_imdb_selection',
        'nlp_imdb_size_selection':              'ac_imdb_size_selection',
        'nlp_amazon_semi_anti_causal':          'causal_amazon_sac',
        'nlp_amazon_size_selection':            'causal_amazon_size_selection',
        'nlp_amazon_conf_varying_proxy':        'causal_amazon_conf_varying_proxy',
        'nlp_amazon_rating_natural':            'causal_amazon_rating_natural',
        'nlp_amazon_keyword_selection':          'causal_amazon_keyword_selection',
        'nlp_amazon_sentiment_selection':        'causal_amazon_sentiment_selection',
    }
    slug     = _SLUG_MAP.get(args.dataset, args.dataset.replace('nlp_', ''))
    plot_dir = args.plot_dir if args.plot_dir else os.path.join(
        str(_ROOT / 'nlp_synthetic' / 'plots'), slug
    )
    os.makedirs(plot_dir, exist_ok=True)

    n_classes = 4 if args.dataset.startswith('nlp_agnews') else 2

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
        )

    elif args.dataset == 'nlp_sms_spam_conf_varying_proxy':
        train_envs, val_envs, test_env = build_envs_nlp_conf_varying_proxy(
            a_train=list(args.nlp_conf_a_train),
            a_test=args.nlp_conf_a_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
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
        )

    elif args.dataset == 'nlp_agnews_conf_varying_proxy':
        train_envs, val_envs, test_env = build_envs_ag_news_conf_varying_proxy(
            a_train=list(args.nlp_conf_a_train),
            a_test=args.nlp_conf_a_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
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

    elif args.dataset == 'nlp_sst2_genre_selection':
        train_envs, val_envs, test_env = build_envs_sst2_genre_selection(
            train_p_select=list(args.nlp_selection_p_train),
            seed=args.seed,
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
        )

    elif args.dataset == 'nlp_sst2_conf_varying_proxy':
        train_envs, val_envs, test_env = build_envs_sst2_conf_varying_proxy(
            a_train=list(args.nlp_conf_a_train),
            a_test=args.nlp_conf_a_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
        )

    elif args.dataset == 'nlp_imdb_conf_varying_proxy':
        train_envs, val_envs, test_env = build_envs_imdb_conf_varying_proxy(
            a_train=list(args.nlp_conf_a_train),
            a_test=args.nlp_conf_a_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            class_ratio_train=args.nlp_imdb_class_ratio_train,
            class_ratio_test=args.nlp_imdb_class_ratio_test,
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
        )

    elif args.dataset == 'nlp_amazon_conf_varying_proxy':
        train_envs, val_envs, test_env = build_envs_amazon_conf_varying_proxy(
            a_train=list(args.nlp_conf_a_train),
            a_test=args.nlp_conf_a_test,
            seed=args.seed,
            p_c_flip=args.nlp_conf_p_c_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
            n_target=args.nlp_amazon_n_target,
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
        )

    else:
        raise ValueError(f"Unknown NLP dataset: {args.dataset}")

    # ── Training ──────────────────────────────────────────────────────────────
    erm, erm_hist = train_erm(
        envs=train_envs, val_envs=val_envs, test_env=test_env,
        steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
        seed=args.seed, device=device, eval_every=args.eval_every,
        dataset_name=args.dataset, n_classes=n_classes,
    )

    irm, irm_hist = train_irm(
        envs=train_envs, val_envs=val_envs, test_env=test_env,
        steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
        irm_lambda=args.irm_lambda,
        seed=args.seed, device=device, eval_every=args.eval_every,
        dataset_name=args.dataset, n_classes=n_classes,
    )

    # ── Plots ─────────────────────────────────────────────────────────────────
    print(f'\n--- Saving plots to {plot_dir}/ ---')
    plot_accuracy_curves(
        erm_hist, irm_hist,
        os.path.join(plot_dir, '01_accuracy.png'), args.dataset,
    )
    plot_loss_curves(
        erm_hist, irm_hist,
        os.path.join(plot_dir, '02_loss.png'), args.dataset,
    )
    plot_summary_panel(
        erm_hist, irm_hist,
        os.path.join(plot_dir, '05_summary.png'), args.dataset,
    )
    plot_results_table(
        erm_hist, irm_hist,
        os.path.join(plot_dir, '06_results.png'),
    )
