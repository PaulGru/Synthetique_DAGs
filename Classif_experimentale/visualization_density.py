"""
Nouvelle fonction de visualisation basée sur la densité (KDE).
"""
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple
from data_synth import Env

def _ensure_dir(dirname):
    """Crée le répertoire si nécessaire"""
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname)


def visualize_density_with_marginals(
    envs: List[Env],
    env_names: List[str],
    filename: str,
    max_points_per_env: int = 2000,
    figsize: Tuple[int, int] = (12, 10),
    kde_levels: int = 8
):
    """
    Visualise la distribution des données avec des contours de densité KDE
    et des distributions marginales pour X_z et X_y.
    
    Parameters
    ----------
    envs : List[Env]
        Liste des environnements à visualiser
    env_names : List[str]
        Noms des environnements
    filename : str
        Chemin de sauvegarde
    max_points_per_env : int
        Nombre max de points par environnement
    figsize : Tuple[int, int]
        Taille de la figure
    kde_levels : int
        Nombre de niveaux de contours
    """
    from scipy.stats import gaussian_kde
    
    _ensure_dir(os.path.dirname(filename))
    
    # Couleurs pour les environnements
    env_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Créer la figure avec gridspec pour avoir des marges
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 3, hspace=0.05, wspace=0.05,
                          height_ratios=[1, 4, 0.3], 
                          width_ratios=[0.3, 4, 1])
    
    # Axes principaux
    ax_main = fig.add_subplot(gs[1, 1])      # Contours centraux
    ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)    # Marginal X_z
    ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)  # Marginal X_y
    
    # Détection de dimensionnalité
    d_total = envs[0].X.shape[1]
    
    if d_total != 2:
        print(f"⚠️  Cette visualisation nécessite dim=2 (actuel: {d_total})")
        print("   Utilisant PCA pour projeter en 2D...")
        from sklearn.decomposition import PCA
        
        # PCA globale
        all_X = torch.cat([e.X for e in envs], dim=0).cpu().numpy()
        pca = PCA(n_components=2)
        pca.fit(all_X)
        
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    else:
        pca = None
        xlabel = "$X_0$ (1ère dimension)"
        ylabel = "$X_1$ (2ème dimension)"
    
    # Pour chaque environnement
    for env_idx, (env, env_name) in enumerate(zip(envs, env_names)):
        X_np = env.X.cpu().numpy()
        y_np = env.y.cpu().numpy().ravel()
        
        # Sous-échantillonner
        n_samples = min(len(X_np), max_points_per_env)
        indices = np.random.choice(len(X_np), n_samples, replace=False)
        X_sample = X_np[indices]
        y_sample = y_np[indices]
        
        # Appliquer PCA si nécessaire
        if pca is not None:
            X_plot = pca.transform(X_sample)
        else:
            X_plot = X_sample
        
        color = env_colors[env_idx % len(env_colors)]
        
        # ===== CONTOURS DE DENSITÉ (panneau central) =====
        try:
            # Estimation de densité par classe
            for class_val, alpha_val in [(0, 0.3), (1, 0.5)]:
                mask = (y_sample == class_val)
                if np.sum(mask) < 10:
                    continue
                
                X_class = X_plot[mask]
                
                # KDE
                kde = gaussian_kde(X_class.T)
                
                # Grille pour évaluation
                x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
                y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5
                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 100),
                    np.linspace(y_min, y_max, 100)
                )
                
                # Évaluation de la densité
                positions = np.vstack([xx.ravel(), yy.ravel()])
                density = kde(positions).reshape(xx.shape)
                
                # Contours
                linestyle = '-' if class_val == 1 else '--'
                ax_main.contour(xx, yy, density, levels=kde_levels, 
                               colors=color, alpha=alpha_val, 
                               linewidths=1.5, linestyles=linestyle)
        except:
            # Fallback: scatter plot si KDE échoue
            for class_val in [0, 1]:
                mask = (y_sample == class_val)
                if np.sum(mask) > 0:
                    marker = 'o' if class_val == 1 else 'x'
                    ax_main.scatter(X_plot[mask, 0], X_plot[mask, 1],
                                   c=color, marker=marker, s=10, alpha=0.3,
                                   label=f"{env_name} - Y={class_val}")
        
        # ===== DISTRIBUTIONS MARGINALES =====
        # Distribution X_z (axe horizontal, en haut)
        ax_top.hist(X_plot[:, 0], bins=30, color=color, alpha=0.4, 
                    density=True, histtype='stepfilled', edgecolor=color, linewidth=1.5)
        
        # Distribution X_y (axe vertical, à droite)
        ax_right.hist(X_plot[:, 1], bins=30, color=color, alpha=0.4,
                     density=True, orientation='horizontal', 
                     histtype='stepfilled', edgecolor=color, linewidth=1.5)
    
    # ===== STYLE =====
    # Panneau central
    ax_main.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax_main.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    
    # Marges
    ax_top.set_ylabel('Densité', fontsize=10)
    ax_top.set_title('Distribution des Données (Densité KDE)', 
                     fontsize=16, fontweight='bold', pad=20)
    ax_top.grid(True, alpha=0.3, axis='y')
    ax_top.tick_params(labelbottom=False)
    
    ax_right.set_xlabel('Densité', fontsize=10)
    ax_right.grid(True, alpha=0.3, axis='x')
    ax_right.tick_params(labelleft=False)
    
    # Légende (style de ligne)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='Y=1'),
        Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Y=0')
    ]
    # Ajouter les environnements
    for env_idx, env_name in enumerate(env_names):
        color = env_colors[env_idx % len(env_colors)]
        legend_elements.append(
            Line2D([0], [0], color=color, linewidth=3, label=env_name)
        )
    
    ax_main.legend(handles=legend_elements, loc='best', fontsize=10, framealpha=0.9)
    
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Visualisation densité avec marges sauvegardée: {filename}")


def visualize_density_heatmap(
    envs: List[Env],
    env_names: List[str],
    filename: str,
    max_points_per_env: int = 5000,
    figsize: Tuple[int, int] = (14, 5),
    bins: int = 50
):
    """
    Visualise les distributions avec des heatmaps 2D (histogrammes 2D).
    Un subplot par environnement.
    
    Parameters
    ----------
    envs : List[Env]
        Liste des environnements
    env_names : List[str]
        Noms des environnements
    filename : str
        Chemin de sauvegarde
    max_points_per_env : int
        Points max par environnement
    figsize : Tuple[int, int]
        Taille de la figure
    bins : int
        Nombre de bins pour l'histogramme 2D
    """
    _ensure_dir(os.path.dirname(filename))
    
    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=figsize, sharey=True)
    
    if n_envs == 1:
        axes = [axes]
    
    # Détection dimensionnalité
    d_total = envs[0].X.shape[1]
    
    if d_total != 2:
        from sklearn.decomposition import PCA
        all_X = torch.cat([e.X for e in envs], dim=0).cpu().numpy()
        pca = PCA(n_components=2)
        pca.fit(all_X)
        xlabel = f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)"
        ylabel = f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)"
    else:
        pca = None
        xlabel = "$X_0$ (1ère dimension)"
        ylabel = "$X_1$ (2ème dimension)"
    
    # Trouver les limites globales
    all_X_plot = []
    for env in envs:
        X_np = env.X.cpu().numpy()
        if pca is not None:
            X_plot = pca.transform(X_np)
        else:
            X_plot = X_np
        all_X_plot.append(X_plot)
    
    all_X_concat = np.vstack(all_X_plot)
    x_min, x_max = all_X_concat[:, 0].min(), all_X_concat[:, 0].max()
    y_min, y_max = all_X_concat[:, 1].min(), all_X_concat[:, 1].max()
    
    # Marges
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    
    # Pour chaque environnement
    for env_idx, (env, env_name, ax) in enumerate(zip(envs, env_names, axes)):
        X_np = env.X.cpu().numpy()
        y_np = env.y.cpu().numpy().ravel()
        
        # Sous-échantillonner
        n_samples = min(len(X_np), max_points_per_env)
        indices = np.random.choice(len(X_np), n_samples, replace=False)
        X_sample = X_np[indices]
        
        # PCA
        if pca is not None:
            X_plot = pca.transform(X_sample)
        else:
            X_plot = X_sample
        
        # Histogramme 2D
        h = ax.hist2d(X_plot[:, 0], X_plot[:, 1], bins=bins,
                      cmap='viridis', 
                      range=[[x_min - x_margin, x_max + x_margin],
                             [y_min - y_margin, y_max + y_margin]])
        
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        if env_idx == 0:
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
        ax.set_title(env_name, fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Colorbar
        plt.colorbar(h[3], ax=ax, label='Densité')
    
    plt.suptitle('Distribution des Données par Environnement (Heatmap)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Heatmap de densité sauvegardée: {filename}")


def visualize_density_kde_by_env(
    envs: List[Env],
    env_names: List[str],
    filename: str,
    max_points_per_env: int = 2000,
    figsize: Tuple[int, int] = (14, 5),
    kde_levels: int = 8,
    title: str = "Distribution des Données (KDE par Environnement)"
):
    """
    Visualise les distributions KDE séparées par environnement (3 subplots).
    Layout aligné avec les heatmaps, sans marginals.
    
    Parameters
    ----------
    envs : List[Env]
        Liste des environnements
    env_names : List[str]
        Noms des environnements
    filename : str
        Chemin de sauvegarde
    max_points_per_env : int
        Points max par environnement
    figsize : Tuple[int, int]
        Taille de la figure
    kde_levels : int
        Nombre de niveaux de contours
    title : str
        Titre global
    """
    from scipy.stats import gaussian_kde
    
    _ensure_dir(os.path.dirname(filename))
    
    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=figsize, sharey=True)
    
    if n_envs == 1:
        axes = [axes]
    
    # Détection dimensionnalité
    d_total = envs[0].X.shape[1]
    
    if d_total != 2:
        from sklearn.decomposition import PCA
        all_X = torch.cat([e.X for e in envs], dim=0).cpu().numpy()
        pca = PCA(n_components=2)
        pca.fit(all_X)
        xlabel = f"$X_0$ (1ère dimension)"
        ylabel = f"$X_1$ (2ème dimension)"
    else:
        pca = None
        xlabel = "$X_0$ (1ère dimension)"
        ylabel = "$X_1$ (2ème dimension)"
    
    # Trouver les limites globales
    all_X_plot = []
    for env in envs:
        X_np = env.X.cpu().numpy()
        if pca is not None:
            X_plot = pca.transform(X_np)
        else:
            X_plot = X_np
        all_X_plot.append(X_plot)
    
    all_X_concat = np.vstack(all_X_plot)
    x_min, x_max = all_X_concat[:, 0].min(), all_X_concat[:, 0].max()
    y_min, y_max = all_X_concat[:, 1].min(), all_X_concat[:, 1].max()
    
    # Marges
    x_margin = (x_max - x_min) * 0.1
    y_margin = (y_max - y_min) * 0.1
    x_min_plot, x_max_plot =  x_min - x_margin, x_max + x_margin
    y_min_plot, y_max_plot = y_min - y_margin, y_max + y_margin
    
    # Pour chaque environnement
    for env_idx, (env, env_name, ax) in enumerate(zip(envs, env_names, axes)):
        X_np = env.X.cpu().numpy()
        y_np = env.y.cpu().numpy().ravel()
        
        # Sous-échantillonner
        n_samples = min(len(X_np), max_points_per_env)
        indices = np.random.choice(len(X_np), n_samples, replace=False)
        X_sample = X_np[indices]
        y_sample = y_np[indices]
        
        # PCA
        if pca is not None:
            X_plot = pca.transform(X_sample)
        else:
            X_plot = X_sample
        
        # Grille pour KDE
        xx, yy = np.meshgrid(
            np.linspace(x_min_plot, x_max_plot, 100),
            np.linspace(y_min_plot, y_max_plot, 100)
        )
        
        # KDE par classe
        try:
            for class_val in [0, 1]:
                mask = (y_sample == class_val)
                if np.sum(mask) < 10:
                    continue
                
                X_class = X_plot[mask]
                
                # KDE
                kde = gaussian_kde(X_class.T)
                
                # Évaluation de la densité
                positions = np.vstack([xx.ravel(), yy.ravel()])
                density = kde(positions).reshape(xx.shape)
                
                # Contours
                linestyle = '-' if class_val == 1 else '--'
                color = '#e74c3c' if class_val == 0 else '#3498db'
                alpha_val = 0.6 if class_val == 1 else 0.4
                
                ax.contour(xx, yy, density, levels=kde_levels, 
                          colors=color, alpha=alpha_val, 
                          linewidths=1.5, linestyles=linestyle)
                
                # Remplir aussi les contours
                ax.contourf(xx, yy, density, levels=kde_levels, 
                           colors=color, alpha=0.15)
        except Exception as e:
            # Fallback: scatter plot
            for class_val in [0, 1]:
                mask = (y_sample == class_val)
                if np.sum(mask) > 0:
                    marker = 'o' if class_val == 1 else 'x'
                    color = '#3498db' if class_val == 1 else '#e74c3c'
                    ax.scatter(X_plot[mask, 0], X_plot[mask, 1],
                              c=color, marker=marker, s=10, alpha=0.4)
        
        # Limites et labels
        ax.set_xlim(x_min_plot, x_max_plot)
        ax.set_ylim(y_min_plot, y_max_plot)
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        if env_idx == 0:
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
        ax.set_title(env_name, fontsize=14, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Légende globale
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#3498db', linestyle='-', linewidth=2, label='Y=1'),
        Line2D([0], [0], color='#e74c3c', linestyle='--', linewidth=2, label='Y=0')
    ]
    axes[-1].legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Visualisation KDE par environnement sauvegardée: {filename}")
