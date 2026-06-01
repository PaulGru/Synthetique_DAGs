import sys
from pathlib import Path as _Path
# Ajoute shared/ au chemin Python
_ROOT = _Path(__file__).resolve().parent
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

import torch, os
import numpy as np
from args_synthetic import make_main_parser
from data_synth import (
    build_envs_semi_anti_causal,
    build_envs_selection,
    build_envs_confounding_varying_proxy,
    # Anti-causal variants (Y → X)
    build_envs_anti_causal_semi_anti_causal,
    build_envs_anti_causal_selection,
    build_envs_anti_causal_confounding_varying_proxy,
)
from models_training import train_erm, train_irm
from utils_irm import resolve_device
from visualization_synth import generate_all_plots
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    args = make_main_parser().parse_args()

    device = resolve_device(args.device)

    # Dossier de plots – nommage harmonisé (préfixe causal_ / ac_)
    _SLUG_MAP = {
        'synthetic_semi_anti_causal':              'causal_semi_anti_causal',
        'synthetic_selection':                      'causal_selection',
        'synthetic_confounding_varying_proxy':      'causal_conf_varying_proxy',
        'synthetic_ac_semi_anti_causal':            'ac_semi_anti_causal',
        'synthetic_ac_selection':                   'ac_selection',
        'synthetic_ac_confounding_varying_proxy':   'ac_conf_varying_proxy',
    }
    _dataset_slug = _SLUG_MAP.get(args.dataset, args.dataset.replace('synthetic_', ''))
    plot_dir = args.plot_dir if args.plot_dir else str(_Path(__file__).parent / 'plots' / _dataset_slug)
    os.makedirs(plot_dir, exist_ok=True)

    if args.dataset == 'synthetic_semi_anti_causal':
        train_envs, val_envs, test_env = build_envs_semi_anti_causal(
            n=args.n,
            train_p_spurs=list(args.ps_train),
            test_p_spur=float(args.p_test_ood),
            seed=args.seed,
            val_frac=float(args.val_frac),
            label_flip=float(args.label_flip),
            n_test=args.n_test,
            dim_z=args.dim_z,
            dim_y=args.dim_y,
        )

        # ===== ERM =====
        erm, erm_hist = train_erm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,  
            dataset_name=args.dataset
        )
        
        # ===== IRM =====
        irm, irm_hist = train_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,
            irm_lambda=args.irm_lambda,
            dataset_name=args.dataset
        )

        # ===== PLOTS =====
        generate_all_plots(erm_hist, irm_hist, erm, irm, train_envs, test_env, plot_dir, args.dataset)
    
    elif args.dataset == 'synthetic_selection':
        
        train_alphas = list(map(float, args.sel_alpha_train))
        test_alpha   = float(args.sel_alpha_test)

        train_envs, val_envs, test_env = build_envs_selection(
            n=args.n,
            train_alphas=train_alphas,
            test_alpha=test_alpha,
            seed=args.seed,
            val_frac=args.val_frac,
            n_test=args.n_test,
            label_flip=args.sel_label_flip,
            dim_z=args.dim_z,
            dim_y=args.dim_y,
        )

        erm, erm_hist = train_erm(
            envs=train_envs,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device,
            eval_every=args.eval_every, val_envs=val_envs, test_env=test_env,
            dataset_name=args.dataset
        )

        irm, irm_hist = train_irm(
            envs=train_envs,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            irm_lambda=args.irm_lambda,
            seed=args.seed, device=device,
            eval_every=args.eval_every, val_envs=val_envs, test_env=test_env,
            dataset_name=args.dataset
        )

        # ===== PLOTS =====
        generate_all_plots(erm_hist, irm_hist, erm, irm, train_envs, test_env, plot_dir, args.dataset)

    elif args.dataset == 'synthetic_confounding_varying_proxy':
        train_envs, val_envs, test_env = build_envs_confounding_varying_proxy(
            n=args.n,
            a_train=args.conf_a_train,
            a_test=args.conf_a_test,
            gamma=args.conf_gamma,
            seed=args.seed,
            val_frac=args.val_frac,
            n_test=args.n_test,
            dim_z=args.dim_z,
            dim_y=args.dim_y,
        )
        erm, erm_hist = train_erm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,
            dataset_name=args.dataset
        )

        irm, irm_hist = train_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            irm_lambda=args.irm_lambda,
            seed=args.seed, device=device,
            eval_every=args.eval_every,
            dataset_name=args.dataset
        )

        # ===== PLOTS =====
        generate_all_plots(erm_hist, irm_hist, erm, irm, train_envs, test_env, plot_dir, args.dataset)
             
    # =========================================================================
    # ===== CAS ANTI-CAUSAL (Y → X_z) — mêmes 5 perturbations trompeuses =====
    # =========================================================================

    elif args.dataset == 'synthetic_ac_semi_anti_causal':
        train_envs, val_envs, test_env = build_envs_anti_causal_semi_anti_causal(
            n=args.n,
            train_p_spurs=list(args.ps_train),
            test_p_spur=float(args.p_test_ood),
            seed=args.seed,
            val_frac=float(args.val_frac),
            label_flip=float(args.label_flip),
            n_test=args.n_test,
            dim_z=args.dim_z,
            dim_y=args.dim_y,
            p_y_train=list(args.ac_py_train) if args.ac_py_train else None,
        )
        erm, erm_hist = train_erm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,
            dataset_name=args.dataset
        )
        irm, irm_hist = train_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            irm_lambda=args.irm_lambda,
            seed=args.seed, device=device, eval_every=args.eval_every,
            dataset_name=args.dataset
        )
        generate_all_plots(erm_hist, irm_hist, erm, irm, train_envs, test_env, plot_dir, args.dataset)

    elif args.dataset == 'synthetic_ac_selection':
        train_alphas = list(map(float, args.sel_alpha_train))
        test_alpha   = float(args.sel_alpha_test)
        train_envs, val_envs, test_env = build_envs_anti_causal_selection(
            n=args.n,
            train_alphas=train_alphas,
            test_alpha=test_alpha,
            seed=args.seed,
            val_frac=args.val_frac,
            n_test=args.n_test,
            label_flip=args.sel_label_flip,
            dim_z=args.dim_z,
            dim_y=args.dim_y,
            p_y_train=list(args.ac_py_train) if args.ac_py_train else None,
        )
        erm, erm_hist = train_erm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,
            dataset_name=args.dataset
        )
        irm, irm_hist = train_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            irm_lambda=args.irm_lambda,
            seed=args.seed, device=device, eval_every=args.eval_every,
            dataset_name=args.dataset
        )
        generate_all_plots(erm_hist, irm_hist, erm, irm, train_envs, test_env, plot_dir, args.dataset)

    elif args.dataset == 'synthetic_ac_confounding_varying_proxy':
        train_envs, val_envs, test_env = build_envs_anti_causal_confounding_varying_proxy(
            n=args.n,
            a_train=args.conf_a_train,
            a_test=args.conf_a_test,
            gamma=args.conf_gamma,
            seed=args.seed,
            val_frac=args.val_frac,
            n_test=args.n_test,
            dim_z=args.dim_z,
            dim_y=args.dim_y,
            p_c_train=list(args.ac_proxy_pc_train) if args.ac_proxy_pc_train else None,
            p_c_test=args.ac_proxy_pc_test,
            label_flip=args.conf_label_flip,
        )
        erm, erm_hist = train_erm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,
            dataset_name=args.dataset
        )
        irm, irm_hist = train_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            irm_lambda=args.irm_lambda,
            seed=args.seed, device=device, eval_every=args.eval_every,
            dataset_name=args.dataset
        )
        generate_all_plots(erm_hist, irm_hist, erm, irm, train_envs, test_env, plot_dir, args.dataset)
