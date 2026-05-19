"""
Argparse parsers for all synthetic experiments.

Usage:
    from args_synthetic import make_main_parser, make_gap_sweep_parser
    args = make_main_parser().parse_args()
    args = make_gap_sweep_parser().parse_args()
"""

import argparse

# ─────────────────────────────────────────────────────────────────────────────
# Dataset-level defaults for the gap sweep
# ─────────────────────────────────────────────────────────────────────────────
DATASET_DEFAULTS = {
    'synthetic_semi_anti_causal': {
        'gap_center':  0.2,
        'gap_max':     0.20,
        'gap_test':    1.0,
        'param_label': 'p_correct',
        'x_label':     'Gap Δp_correct = p₂ − p₁',
    },
    'synthetic_selection': {
        'gap_center':  0.85,
        'gap_max':     0.20,
        'gap_test':    0.0,
        'param_label': 'p_select',
        'x_label':     'Gap Δp_select = p_select₂ − p_select₁',
    },
    'synthetic_confounding_varying_proxy': {
        'gap_center':  0.06,
        'gap_max':     0.20,
        'gap_test':    0.99,
        'param_label': 'a (flip rate)',
        'x_label':     'Gap Δa = a₂ − a₁',
    },
    'synthetic_confounding_varying_pc': {
        'gap_center':  0.9,
        'gap_max':     0.20,
        'gap_test':    0.1,
        'param_label': 'p_c',
        'x_label':     'Gap Δp_c = p_c₂ − p_c₁',
    },
    # ── Cas ANTI-CAUSAL (Y → X_z) ──
    'synthetic_ac_semi_anti_causal': {
        'gap_center':  0.2,
        'gap_max':     0.20,
        'gap_test':    1.0,
        'param_label': 'p_correct',
        'x_label':     'Gap Δp_correct = p₂ − p₁',
    },
    'synthetic_ac_selection': {
        'gap_center':  0.85,
        'gap_max':     0.20,
        'gap_test':    0.0,
        'param_label': 'p_select',
        'x_label':     'Gap Δp_select = p_select₂ − p_select₁',
    },
    'synthetic_ac_confounding_varying_proxy': {
        'gap_center':  0.06,
        'gap_max':     0.20,
        'gap_test':    0.99,
        'param_label': 'a (flip rate)',
        'x_label':     'Gap Δa = a₂ − a₁',
    },
    'synthetic_ac_confounding_varying_gamma': {
        'gap_center':  2.0,
        'gap_max':     3.0,
        'gap_test':    0.0,
        'param_label': 'gamma',
        'x_label':     'Gap Δγ = γ₂ − γ₁',
    },
    'synthetic_ac_confounding_varying_pc': {
        'gap_center':  0.9,
        'gap_max':     0.20,
        'gap_test':    0.1,
        'param_label': 'p_c',
        'x_label':     'Gap Δp_c = p_c₂ − p_c₁',
    },
}


def base_parser() -> argparse.ArgumentParser:
    """Return a parent parser with all arguments shared between main.py and run_gap_sweep.py.

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
    p.add_argument('--conf_pc_a',     type=float, default=0.0,
                   help='confounding_varying_pc: C→Z link strength (fixed)')
    p.add_argument('--conf_pc_gamma', type=float, default=2.0,
                   help='confounding_varying_pc: confounder weight on Y (fixed)')

    # ---- Anti-causal class imbalance ----
    p.add_argument('--ac_py_train', type=float, nargs='+', default=None,
                   help='P(Y*=1) per training env for ac_semi_anti_causal / ac_selection. '
                        'E.g. --ac_py_train 0.3 0.7 → class 0 majority in env0, class 1 in env1. '
                        'Val/test always use 0.5. Default: None (=0.5 for all envs).')
    p.add_argument('--ac_proxy_pc_train', type=float, nargs='+', default=None,
                   help='P(C=1) per training env for synthetic_ac_confounding_varying_proxy. '
                        'Used to create class imbalance in the anti-causal proxy experiment.')
    p.add_argument('--ac_proxy_pc_test', type=float, default=0.35,
                   help='P(C=1) for the test env in synthetic_ac_confounding_varying_proxy.')
    p.add_argument('--ac_gamma_c_mean_train', type=float, nargs='+', default=None,
                   help='Mean of the Gaussian confounder C per training env for '
                        'synthetic_ac_confounding_varying_gamma.')
    p.add_argument('--ac_gamma_c_mean_test', type=float, default=0.0,
                   help='Mean of the Gaussian confounder C for the test env in '
                        'synthetic_ac_confounding_varying_gamma.')

    # ---- Training ----
    p.add_argument('--erm_steps',  type=int,   default=25_000)
    p.add_argument('--erm_lr',     type=float, default=5e-3)
    p.add_argument('--erm_batch',  type=int,   default=512)
    p.add_argument('--irm_steps',  type=int,   default=25_000)
    p.add_argument('--irm_lr',     type=float, default=5e-3)
    p.add_argument('--irm_lambda', type=float, default=200.0)
    p.add_argument('--irm_batch',  type=int,   default=512)
    # ---- IB-IRM ----
    p.add_argument('--ibirm_steps',  type=int,   default=25_000)
    p.add_argument('--ibirm_lr',     type=float, default=5e-3)
    p.add_argument('--ibirm_lambda', type=float, default=200.0,
                   help='IRM penalty coefficient for IB-IRM')
    p.add_argument('--ibirm_batch',  type=int,   default=512)
    p.add_argument('--ib_lambda',    type=float, default=0.01,
                   help='Information bottleneck coefficient for IB-IRM (logit variance penalty). '
                        'Typical range : 0.001 – 0.1. Independant of ibirm_lambda.')
    p.add_argument('--run_ibirm', action='store_true', default=False,
                   help='Run IB-IRM training in addition to ERM and IRM (disabled by default).')
    # ---- Model architecture ----
    p.add_argument('--use_mlp', action='store_true', default=False,
                   help='Use a small MLP head instead of logistic regression '
                        '(ignored when finetune_bert_layers > 0).')
    p.add_argument('--mlp_hidden',  type=int,   default=256,
                   help='Hidden size of the MLP (only used when --use_mlp).')
    p.add_argument('--mlp_dropout', type=float, default=0.1,
                   help='Dropout rate of the MLP (only used when --use_mlp).')
    p.add_argument('--seed',       type=int,  default=1)
    p.add_argument('--device',     type=str,  default='auto')
    p.add_argument('--eval_every', type=int,  default=100)

    return p


def make_main_parser() -> argparse.ArgumentParser:
    """Full parser for main.py (all 5 synthetic datasets)."""
    p = argparse.ArgumentParser(parents=[base_parser()])

    p.add_argument('--dataset', choices=[
        'synthetic_semi_anti_causal',
        'synthetic_selection',
        'synthetic_confounding_varying_proxy',
        'synthetic_confounding_varying_gamma',
        'synthetic_confounding_varying_pc',
        # ── Anti-causal variants ──
        'synthetic_ac_semi_anti_causal',
        'synthetic_ac_selection',
        'synthetic_ac_confounding_varying_proxy',
        'synthetic_ac_confounding_varying_gamma',
        'synthetic_ac_confounding_varying_pc',
    ], default='synthetic_semi_anti_causal')

    # ---- Semi anti-causal ----
    p.add_argument('--ps_train',      type=float, nargs='+', default=[0.2, 0.1])
    p.add_argument('--p_test_ood',    type=float, default=0.9)
    p.add_argument('--x_shift_train', type=float, nargs='+', default=None,
                   help='Shift of X_z along w_true per training env to create class imbalance. '
                        'P(Y*=1) ≈ Φ(shift/causal_strength). '
                        'E.g. --x_shift_train -1.0 1.0 → ~16%% class 1 in env0, ~84%% in env1. '
                        'Val and test always use shift=0.0. Default: None (=0.0, balanced).')

    # ---- Confounding (proxy) ----
    p.add_argument('--conf_a_train', type=float, nargs='+', default=[3.0, 2.0],
                   help='P(spurious|Y) per train env')
    p.add_argument('--conf_a_test',  type=float, default=0.2,
                   help='P(spurious|Y) for OOD test env')
    p.add_argument('--conf_label_flip', type=float, default=0.25,
                   help='Label flip rate (train only) to weaken causal signal')

    # ---- Selection bias (collider) ----
    p.add_argument('--sel_alpha_train', type=float, nargs='+', default=[0.9, 0.8],
                   help='Keep-probability when Z==Y per train env')
    p.add_argument('--sel_alpha_test',  type=float, default=0.1,
                   help='Keep-probability for OOD test env')
    p.add_argument('--sel_label_flip',  type=float, default=0.25,
                   help='Symmetric label flip rate')
    p.add_argument('--sel_sigma_y',     type=float, default=0.3,
                   help='Noise std for X_y in selection bias')

    # ---- Confounding varying gamma ----
    p.add_argument('--conf_gamma_train', type=float, nargs='+', default=[5.0, 10.0],
                   help='Causal weight γ per train env')
    p.add_argument('--conf_gamma_test',  type=float, default=0.0,
                   help='Causal weight γ for OOD test env')

    # ---- Confounding varying p_c ----
    p.add_argument('--conf_pc_train', type=float, nargs='+', default=[0.2, 0.4],
                   help='Confounder prevalence p_c per train env')
    p.add_argument('--conf_pc_test',  type=float, default=0.9,
                   help='Confounder prevalence p_c for OOD test env')

    # ---- Output ----
    p.add_argument('--plot_dir', type=str, default=None,
                   help='Plot output directory. Default: plots/<dataset>/')

    return p


def make_gap_sweep_parser() -> argparse.ArgumentParser:
    """Full parser for run_gap_sweep.py."""
    p = argparse.ArgumentParser(
        description="Gap sweep – synthetic datasets",
        parents=[base_parser()],
    )

    p.add_argument('--dataset', required=True,
                   choices=list(DATASET_DEFAULTS.keys()))

    p.add_argument('--gap_center', type=float, default=None,
                   help='Centre of the two train envs (dataset default if omitted)')
    p.add_argument('--gap_test',   type=float, default=None,
                   help='Parameter value for OOD test env (dataset default if omitted)')
    p.add_argument('--gap_step',   type=float, default=0.05, help='Sweep step size')
    p.add_argument('--gap_max',    type=float, default=None,
                   help='Maximum gap (dataset default if omitted)')
    p.add_argument('--out_dir',    type=str,   default=None)

    return p


# ─────────────────────────────────────────────────────────────────────────────
# Noise sweep defaults (label_flip sweep, fixed env gap)
# ─────────────────────────────────────────────────────────────────────────────
NOISE_SWEEP_DEFAULTS = {
    'synthetic_semi_anti_causal': {
        'p1':          0.1,
        'p2':          0.2,
        'p_test':      1.0,
        'noise_max':   0.25,
        'noise_step':  0.025,
        'param_label': 'p_correct',
    },
    'synthetic_selection': {
        'p1':          0.9,
        'p2':          0.8,
        'p_test':      0.0,
        'noise_max':   0.25,
        'noise_step':  0.025,
        'param_label': 'p_select',
    },
    'synthetic_ac_semi_anti_causal': {
        'p1':          0.1,
        'p2':          0.2,
        'p_test':      1.0,
        'noise_max':   0.25,
        'noise_step':  0.025,
        'param_label': 'p_correct',
    },
    'synthetic_ac_selection': {
        'p1':          0.9,
        'p2':          0.8,
        'p_test':      0.0,
        'noise_max':   0.25,
        'noise_step':  0.025,
        'param_label': 'p_select',
    },
    'synthetic_confounding_varying_proxy': {
        'p1':          0.06,
        'p2':          0.20,
        'p_test':      0.99,
        'noise_max':   5.0,
        'noise_step':  0.5,
        'param_label': 'a (flip rate)',
    },
    'synthetic_ac_confounding_varying_proxy': {
        'p1':          0.06,
        'p2':          0.20,
        'p_test':      0.99,
        'noise_max':   0.25,
        'noise_step':  0.025,
        'param_label': 'a (flip rate)',
    },
}


def make_noise_sweep_parser() -> argparse.ArgumentParser:
    """Full parser for run_noise_sweep.py.

    Sweeps the label_flip (noise) level from 0 to noise_max while keeping
    the environment gap fixed (at the values used in the main experiments).
    Supported datasets: semi_anti_causal and selection (causal + anti-causal).
    """
    p = argparse.ArgumentParser(
        description="Noise sweep – synthetic datasets (semi_anti_causal and selection)",
        parents=[base_parser()],
    )

    p.add_argument('--dataset', required=True,
                   choices=list(NOISE_SWEEP_DEFAULTS.keys()))

    # Fixed env parameters (dataset defaults applied in the script)
    p.add_argument('--p1',        type=float, default=None,
                   help='Env-1 spurious parameter (dataset default if omitted)')
    p.add_argument('--p2',        type=float, default=None,
                   help='Env-2 spurious parameter (dataset default if omitted)')
    p.add_argument('--p_test',    type=float, default=None,
                   help='OOD test spurious parameter (dataset default if omitted)')

    # Noise sweep range
    p.add_argument('--noise_max',  type=float, default=None,
                   help='Maximum label_flip value to sweep up to (dataset default if omitted)')
    p.add_argument('--noise_step', type=float, default=None,
                   help='Step size for the noise sweep')

    p.add_argument('--out_dir',   type=str,   default=None)

    return p

