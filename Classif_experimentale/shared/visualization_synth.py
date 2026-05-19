"""
Shared plotting utilities for ERM vs IRM synthetic experiments.

Usable from any experiment script (synthetic, semi-synthetic, real …):

    from visualization_synth import generate_all_plots

    generate_all_plots(erm_hist, irm_hist, erm_model, irm_model,
                       train_envs, test_env, plot_dir, dataset_name)

Output files written to plot_dir/:
    01_accuracy.png        – Val (ID) and Test (OOD) accuracy curves
    02_loss.png            – BCE training loss
    03_weight_dynamics.png – Causal / spurious weight norms over training  (logreg only)
    04_final_weights.png   – Per-feature weight bar chart                  (logreg only)
    05_summary.png         – Grouped bar chart + key metrics table
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# ─── Publication-quality style (ACL/EMNLP) ───────────────────────────────────
mpl.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':          11,
    'axes.titlesize':     11,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    9,
    'figure.titlesize':   12,
    'mathtext.fontset':   'stix',
    'lines.linewidth':    1.8,
    'axes.linewidth':     0.8,
    'axes.grid':          True,
    'grid.alpha':         0.3,
    'grid.linewidth':     0.5,
    'grid.color':         '#cccccc',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'legend.framealpha':  0.85,
    'legend.edgecolor':   '#cccccc',
})

# ─── Colour palette (Wong colorblind-safe) ────────────────────────────────────
ERM_C    = '#E69F00'   # orange
IRM_C    = '#0072B2'   # blue
IBIRM_C  = '#009E73'   # green       – IB-IRM
CAU_C    = '#009E73'   # green       – causal features
SPU_C    = '#D55E00'   # vermillion  – spurious features
TRUE_C   = '#CC79A7'   # reddish purple – ground truth direction

# ─── Dataset display names ────────────────────────────────────────────────────
DATASET_LABEL = {
    'synthetic_semi_anti_causal':          'Semi-Anti-Causal',
    'synthetic_selection':                 'Selection Bias',
    'synthetic_confounding_varying_proxy': 'Confounding (Varying Proxy Strength)',
    'synthetic_confounding_varying_gamma': 'Confounding (Varying γ)',
    'synthetic_confounding_varying_pc':    'Confounding (Varying P(C))',
    # NLP datasets
    'nlp_sms_spam':                   'SMS Spam — Semi-Anti-Causal',
    'nlp_sms_spam_size_selection':    'SMS Spam — Size Selection Bias',
    'nlp_agnews_semi_anti_causal':    'AG News — Semi-Anti-Causal',
    'nlp_agnews_source_selection':    'AG News — Source Selection Bias',
}


# ─── Internal helpers ────────────────────────────────────────────────────────

def _save_fig(filename: str) -> None:
    """Save figure as high-res PNG (300 dpi) and vector PDF."""
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    pdf_path = os.path.splitext(filename)[0] + '.pdf'
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f'  Saved: {filename}  +  {os.path.basename(pdf_path)}')


def _subsample(arr, steps, interval=100):
    """Return (steps, arr) keeping every `interval`-th step plus the last."""
    steps, arr = np.array(steps), np.array(arr)
    if len(steps) == 0:
        return steps, arr
    mask = (steps % interval == 0) | (steps == steps[-1])
    return steps[mask], arr[mask]


def _ema(arr, alpha=0.05):
    """Exponential moving average for smoothing noisy training curves."""
    arr = np.array(arr, dtype=float)
    smoothed = np.empty_like(arr)
    smoothed[0] = arr[0]
    for i in range(1, len(arr)):
        smoothed[i] = alpha * arr[i] + (1.0 - alpha) * smoothed[i - 1]
    return smoothed


# ─── Individual plot functions ────────────────────────────────────────────────

def plot_accuracy_curves(erm_hist, irm_hist, filename, dataset_name='', ibirm_hist=None):
    """Val (ID) and Test (OOD) accuracy on a single axes with distinct line styles."""
    label = DATASET_LABEL.get(dataset_name, dataset_name)
    fig, ax = plt.subplots(figsize=(9, 5))


    splits = [
        ('val_acc',  'Validation (ID)', '-',  0.85),
        ('test_acc', 'Test (OOD)',       '--', 0.85),
    ]
    all_vals = []
    for hist_key, split_label, ls, alpha in splits:
        es,  e_acc = _subsample(erm_hist[hist_key], erm_hist['step'])
        is_, i_acc = _subsample(irm_hist[hist_key], irm_hist['step'])
        delta_irm = i_acc[-1] - e_acc[-1]
        ax.plot(es,  e_acc, ls, color=ERM_C, lw=2.0, alpha=alpha,
                label=f'ERM  {split_label}  (final: {e_acc[-1]:.3f})')
        ax.plot(is_, i_acc, ls, color=IRM_C, lw=2.0, alpha=alpha,
                label=f'IRM  {split_label}  (final: {i_acc[-1]:.3f},  Δ={delta_irm:+.3f})')
        all_vals.extend([e_acc, i_acc])
        if ibirm_hist:
            ibs_, ib_acc = _subsample(ibirm_hist[hist_key], ibirm_hist['step'])
            delta_ib = ib_acc[-1] - e_acc[-1]
            ax.plot(ibs_, ib_acc, ls, color=IBIRM_C, lw=2.0, alpha=alpha,
                    label=f'IB-IRM  {split_label}  (final: {ib_acc[-1]:.3f},  Δ={delta_ib:+.3f})')
            all_vals.append(ib_acc)

    ax.set_xlabel('Training step')
    ax.set_ylabel('Accuracy')
    _all = np.concatenate([np.array(v) for v in all_vals])
    margin = max(0.02, (_all.max() - _all.min()) * 0.15)
    ax.set_ylim(max(0.0, _all.min() - margin), min(1.05, _all.max() + margin))
    ax.legend(loc='best')
    plt.tight_layout()
    _save_fig(filename)
    plt.close()


def plot_loss_curves(erm_hist, irm_hist, filename, dataset_name='', ibirm_hist=None):
    """Cross-entropy (ERM vs IRM vs IB-IRM) — seule quantité comparable entre les modèles."""
    label = DATASET_LABEL.get(dataset_name, dataset_name)
    fig, ax = plt.subplots(figsize=(9, 5))


    es, e_loss = _subsample(erm_hist['loss'], erm_hist['step'], interval=50)
    is_, i_loss = _subsample(irm_hist['loss'], irm_hist['step'], interval=50)
    ax.plot(es,  e_loss, '-', color=ERM_C, lw=0.8, alpha=0.2)
    ax.plot(is_, i_loss, '-', color=IRM_C, lw=0.8, alpha=0.2)
    ax.plot(es,  _ema(e_loss), '-', color=ERM_C, lw=2.0, label='ERM')
    ax.plot(is_, _ema(i_loss), '-', color=IRM_C, lw=2.0, label='IRM')
    if ibirm_hist:
        ibs_, ib_loss = _subsample(ibirm_hist['loss'], ibirm_hist['step'], interval=50)
        ax.plot(ibs_, ib_loss, '-', color=IBIRM_C, lw=0.8, alpha=0.2)
        ax.plot(ibs_, _ema(ib_loss), '-', color=IBIRM_C, lw=2.0, label='IB-IRM')
    ax.set_xlabel('Training step')
    ax.set_ylabel('Cross-entropy loss')
    ax.legend(loc='upper right')

    plt.tight_layout()
    _save_fig(filename)
    plt.close()


def plot_feature_weight_dynamics(erm_hist, irm_hist, filename, dataset_name='', ibirm_hist=None):
    """Causal vs spurious weight norms and spurious usage ratio over training.

    Silently skipped when weight tracking is absent (non-logreg models).
    """
    if not erm_hist.get('w_z') or not erm_hist.get('w_y'):
        return
    label = DATASET_LABEL.get(dataset_name, dataset_name)
    fig, ax = plt.subplots(figsize=(9, 5))

    models = [(erm_hist, ERM_C, 'ERM'), (irm_hist, IRM_C, 'IRM')]
    if ibirm_hist:
        models.append((ibirm_hist, IBIRM_C, 'IB-IRM'))
    for hist, color, name in models:
        s,  wz = _subsample(hist['w_z'], hist['step'])
        _,  wy = _subsample(hist['w_y'], hist['step'])
        ax.plot(s, wz, '-',  color=color, lw=2.0, label=f'{name} – Invariant features $\\|w_{{\\mathcal{{Z}}}}\\|$')
        ax.plot(s, wy, '--', color=color, lw=1.5, alpha=0.8,  label=f'{name} – Spurious features $\\|w_{{\\mathcal{{Y}}}}\\|$')
    ax.set_xlabel('Training step')
    ax.set_ylabel('$\\ell_2$ norm of learned weights')
    ax.legend()
    plt.tight_layout()
    _save_fig(filename)
    plt.close()


def plot_final_weight_profile(erm_model, irm_model, train_envs, filename, dataset_name=''):
    """Bar chart of final learned weights per input feature.

    Bars are coloured by causal (teal) vs spurious (coral) block.
    Diamond markers show the rescaled ground-truth causal direction when available.
    Silently skipped for non-logreg models (no `linear` attribute).
    """
    if not hasattr(erm_model, 'linear'):
        return
    from matplotlib.patches import Patch

    label     = DATASET_LABEL.get(dataset_name, dataset_name)
    meta      = train_envs[0].meta or {}
    dim_z     = meta.get('dim_z', 1)
    dim_y     = meta.get('dim_y', 1)
    d_in      = dim_z + dim_y
    w_true_raw = meta.get('w_true')

    w_erm      = erm_model.linear.weight.detach().cpu().numpy()[0]
    w_irm      = irm_model.linear.weight.detach().cpu().numpy()[0]
    feat_idx   = np.arange(d_in)
    bar_colors = [CAU_C if i < dim_z else SPU_C for i in range(d_in)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, w, name in [(axes[0], w_erm, 'ERM'), (axes[1], w_irm, 'IRM')]:
        ax.bar(feat_idx, w, color=bar_colors, edgecolor='white', linewidth=0.6, zorder=2)
        ax.axhline(0, color='black', lw=0.8)
        ax.axvline(dim_z - 0.5, color='grey', lw=1.2, ls='--', alpha=0.6)

        title_extra = ''
        if w_true_raw is not None:
            wt    = np.array(w_true_raw).flatten()[:dim_z]
            scale = np.linalg.norm(w[:dim_z]) / (np.linalg.norm(wt) + 1e-8)
            ax.scatter(np.arange(dim_z), wt * scale,
                       marker='D', s=60, color=TRUE_C, zorder=5)
            cos = float(np.dot(w[:dim_z], wt) /
                        (np.linalg.norm(w[:dim_z]) * np.linalg.norm(wt) + 1e-8))
            title_extra = f'   cos(w_learned, w_true) = {cos:.3f}'

        ax.set_title(f'{name}{title_extra}', fontsize=11)
        ax.set_xlabel('Feature index', fontsize=11)
        ax.set_ylabel('Weight value', fontsize=11)
        ax.grid(True, alpha=0.2, axis='y', zorder=0)

        legend_elems = [
            Patch(facecolor=CAU_C, label=f'Invariant features ($X_{{\\mathcal{{Z}}}}$, dim={dim_z})'),
            Patch(facecolor=SPU_C, label=f'Spurious features ($X_{{\\mathcal{{Y}}}}$, dim={dim_y})'),
        ]
        if w_true_raw is not None:
            legend_elems.append(plt.Line2D(
                [0], [0], marker='D', color='w', markerfacecolor=TRUE_C,
                markersize=8, label='Ground truth invariant direction (rescaled)'))
        ax.legend(handles=legend_elems, fontsize=9)

    plt.tight_layout()
    _save_fig(filename)
    plt.close()


def plot_summary_panel(erm_hist, irm_hist, filename, dataset_name='', ibirm_hist=None):
    """Grouped accuracy bar chart (Train / Val / Test) + key metrics table."""
    label = DATASET_LABEL.get(dataset_name, dataset_name)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    split_keys = [('Val (ID)',    'val_acc'),
                  ('Test (OOD)', 'test_acc')]
    x     = np.arange(len(split_keys))

    # Left: grouped bar chart
    ax = axes[0]
    models_bar = [
        (erm_hist,   ERM_C,   'ERM'),
        (irm_hist,   IRM_C,   'IRM'),
    ]
    if ibirm_hist:
        models_bar.append((ibirm_hist, IBIRM_C, 'IB-IRM'))
    n_models = len(models_bar)
    width = 0.8 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width
    for (hist, color, name), offset in zip(models_bar, offsets):
        vals = [hist[k][-1] if hist[k] else 0.0 for _, k in split_keys]
        bars = ax.bar(x + offset, vals, width, color=color, label=name,
                      edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for lbl, _ in split_keys], fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title('Final accuracy by split', fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')

    # Right: metrics table
    ax2 = axes[1]
    ax2.axis('off')
    rows_data   = []
    rows_labels = []
    for split_label, hist_key in split_keys:
        e_fin = erm_hist[hist_key][-1]  if erm_hist[hist_key]  else float('nan')
        i_fin = irm_hist[hist_key][-1]  if irm_hist[hist_key]  else float('nan')
        row = [f'{e_fin:.4f}', f'{i_fin:.4f}', f'{i_fin - e_fin:+.4f}']
        if ibirm_hist:
            ib_fin = ibirm_hist[hist_key][-1] if ibirm_hist[hist_key] else float('nan')
            row += [f'{ib_fin:.4f}', f'{ib_fin - e_fin:+.4f}']
        rows_data.append(row)
        rows_labels.append(split_label)

    if ibirm_hist:
        col_labels = ['ERM', 'IRM', '\u0394(IRM−ERM)', 'IB-IRM', '\u0394(IB-IRM−ERM)']
    else:
        col_labels = ['ERM', 'IRM', '\u0394 (IRM−ERM)']
    tbl = ax2.table(
        cellText=rows_data, rowLabels=rows_labels,
        colLabels=col_labels, loc='center', cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 2.4)
    for row_idx, row_vals in enumerate(rows_data, start=1):
        # Colour last Δ column
        delta_val = float(row_vals[-1])
        cell = tbl[row_idx, len(col_labels) - 1]
        cell.set_facecolor('#d4edda' if delta_val >  0.005 else
                           '#f8d7da' if delta_val < -0.005 else '#fff3cd')
    ax2.set_title('Key metrics', fontsize=11, pad=20)

    plt.tight_layout()
    _save_fig(filename)
    plt.close()


def plot_results_table(erm_hist, irm_hist, filename, ibirm_hist=None):
    """Standalone metrics table (no title, no bar chart) — for direct inclusion in papers."""
    split_keys = [('Val (ID)',    'val_acc'),
                  ('Test (OOD)', 'test_acc')]

    rows_data   = []
    rows_labels = []
    for split_label, hist_key in split_keys:
        e_fin = erm_hist[hist_key][-1]  if erm_hist[hist_key]  else float('nan')
        i_fin = irm_hist[hist_key][-1]  if irm_hist[hist_key]  else float('nan')
        row = [f'{e_fin:.4f}', f'{i_fin:.4f}', f'{i_fin - e_fin:+.4f}']
        if ibirm_hist:
            ib_fin = ibirm_hist[hist_key][-1] if ibirm_hist[hist_key] else float('nan')
            row += [f'{ib_fin:.4f}', f'{ib_fin - e_fin:+.4f}']
        rows_data.append(row)
        rows_labels.append(split_label)

    if ibirm_hist:
        col_labels = ['ERM', 'IRM', '\u0394(IRM\u2212ERM)', 'IB-IRM', '\u0394(IB-IRM\u2212ERM)']
        fig_w = 8.0
    else:
        col_labels = ['ERM', 'IRM', '\u0394 (IRM\u2212ERM)']
        fig_w = 5.5

    fig, ax = plt.subplots(figsize=(fig_w, 1.6))
    ax.axis('off')

    tbl = ax.table(
        cellText=rows_data, rowLabels=rows_labels,
        colLabels=col_labels, loc='center', cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 2.4)
    for row_idx, row_vals in enumerate(rows_data, start=1):
        delta_val = float(row_vals[-1])
        cell = tbl[row_idx, len(col_labels) - 1]
        cell.set_facecolor('#d4edda' if delta_val >  0.005 else
                           '#f8d7da' if delta_val < -0.005 else '#fff3cd')

    plt.tight_layout()
    _save_fig(filename)
    plt.close()


# ─── Main entry point ─────────────────────────────────────────────────────────

def generate_all_plots(erm_hist, irm_hist, erm_model, irm_model,
                       train_envs, test_env, plot_dir, dataset_name,
                       ibirm_hist=None, ibirm_model=None):
    """Generate all standard diagnostic plots for an ERM vs IRM (vs IB-IRM) experiment.

    Args:
        erm_hist / irm_hist  : history dicts returned by train_erm / train_irm.
        erm_model / irm_model: trained model objects.
        train_envs : list of training environments (used to read meta info).
        test_env   : OOD test environment (currently unused, reserved for future plots).
        plot_dir   : output directory (created if absent).
        dataset_name : key into DATASET_LABEL for the figure titles.
        ibirm_hist   : optional history dict from train_ibirm.
        ibirm_model  : optional trained IB-IRM model.
    """
    os.makedirs(plot_dir, exist_ok=True)
    print(f'\n--- Saving plots to {plot_dir}/ ---')
    plot_accuracy_curves(erm_hist, irm_hist,
                         os.path.join(plot_dir, '01_accuracy.png'), dataset_name,
                         ibirm_hist=ibirm_hist)
    plot_loss_curves(erm_hist, irm_hist,
                     os.path.join(plot_dir, '02_loss.png'), dataset_name,
                     ibirm_hist=ibirm_hist)
    plot_feature_weight_dynamics(erm_hist, irm_hist,
                                 os.path.join(plot_dir, '03_weight_dynamics.png'), dataset_name,
                                 ibirm_hist=ibirm_hist)
    plot_final_weight_profile(erm_model, irm_model, train_envs,
                              os.path.join(plot_dir, '04_final_weights.png'), dataset_name)
    plot_summary_panel(erm_hist, irm_hist,
                       os.path.join(plot_dir, '05_summary.png'), dataset_name,
                       ibirm_hist=ibirm_hist)


