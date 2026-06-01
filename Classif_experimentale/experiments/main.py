import sys
import json
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parents[1]
for _p in [str(_ROOT), str(_ROOT / 'irm'), str(_ROOT / 'nlp')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import os
import torch
from args import make_nlp_parser
from data import (
    build_envs_ag_news_semi_anti_causal,
    AG_NEWS_WRONG_CLASS,
    build_envs_ag_news_size_selection,
    build_envs_ag_news_conf_varying_proxy,
    build_envs_imdb_genres_size_selection,
    build_envs_imdb_genres_semi_anti_causal,
    build_envs_imdb_genres_conf_varying_proxy,
    build_envs_amazon_semi_anti_causal,
    build_envs_amazon_conf_varying_proxy,
    build_envs_amazon_sentiment_selection,
)
from training import train_erm, train_irm
from evaluation import resolve_device
from plotting import (
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

    # Output directory
    _SLUG_MAP = {
        'nlp_agnews_semi_anti_causal':             'causal_agnews_sac',
        'nlp_agnews_size_selection':               'causal_agnews_size_selection',
        'nlp_agnews_conf_varying_proxy':           'causal_agnews_conf_varying_proxy',
        'nlp_imdb_genres_size_selection':          'ac_imdb_genres_size_selection',
        'nlp_imdb_genres_semi_anti_causal':        'ac_imdb_genres_semi_anti_causal',
        'nlp_imdb_genres_conf_varying_proxy':      'ac_imdb_genres_conf_varying_proxy',
        'nlp_amazon_semi_anti_causal':             'causal_amazon_sac',
        'nlp_amazon_conf_varying_proxy':           'causal_amazon_conf_varying_proxy',
        'nlp_amazon_sentiment_selection':          'causal_amazon_sentiment_selection',
    }
    slug     = _SLUG_MAP.get(args.dataset, args.dataset.replace('nlp_', ''))
    plot_dir = args.plot_dir if args.plot_dir else os.path.join(
        str(_ROOT / 'nlp' / 'plots'), slug
    )
    os.makedirs(plot_dir, exist_ok=True)

    n_classes = 4 if args.dataset.startswith('nlp_agnews') else 2

    # Parse AG News class distribution (flat list → per-env lists)
    agnews_class_dist_train = None
    agnews_class_dist_test  = args.nlp_agnews_class_dist_test
    if args.nlp_agnews_class_dist_train is not None:
        flat = args.nlp_agnews_class_dist_train
        if len(flat) % 4 != 0:
            raise ValueError(f"--nlp_agnews_class_dist_train must have a multiple of 4 floats, got {len(flat)}")
        agnews_class_dist_train = [flat[k:k+4] for k in range(0, len(flat), 4)]

    # Build environments
    if args.dataset == 'nlp_agnews_semi_anti_causal':
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

    elif args.dataset == 'nlp_agnews_size_selection':
        train_envs, val_envs, test_env = build_envs_ag_news_size_selection(
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

    else:
        raise ValueError(f"Unknown NLP dataset: {args.dataset}")

    # Train
    erm, erm_hist = train_erm(
        envs=train_envs, val_envs=val_envs, test_env=test_env,
        steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
        seed=args.seed, device=device, eval_every=args.eval_every,
        dataset_name=args.dataset, n_classes=n_classes,
        finetune_bert_layers=args.finetune_bert_layers,
        bert_model_name=args.nlp_bert_model,
    )

    irm, irm_hist = train_irm(
        envs=train_envs, val_envs=val_envs, test_env=test_env,
        steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
        irm_lambda=args.irm_lambda,
        seed=args.seed, device=device, eval_every=args.eval_every,
        dataset_name=args.dataset, n_classes=n_classes,
        finetune_bert_layers=args.finetune_bert_layers,
        bert_model_name=args.nlp_bert_model,
    )

    # Plots
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

    # Save numerical results for multi-seed aggregation
    metrics = {
        'seed':                 args.seed,
        'dataset':              args.dataset,
        'erm_final_test_acc':   float(erm_hist['test_acc'][-1]),
        'erm_final_val_acc':    float(erm_hist['val_acc'][-1]),
        'erm_best_val_acc':     float(max(erm_hist['val_acc'])),
        'irm_final_test_acc':   float(irm_hist['test_acc'][-1]),
        'irm_final_val_acc':    float(irm_hist['val_acc'][-1]),
        'irm_best_val_acc':     float(max(irm_hist['val_acc'])),
    }
    results_path = os.path.join(plot_dir, f'results_seed{args.seed}.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Metrics saved to {results_path}')
