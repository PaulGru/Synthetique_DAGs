"""
Module de visualisation pour l'analyse IRM (Invariant Risk Minimization).

Ce module fournit des outils pour visualiser :
1. L'échantillonnage initial des points (x, y) dans l'espace original
2. La transformation Φ des points dans l'espace latent après entraînement
3. Comparaisons "Espace Original" vs "Espace Latent Φ"

Auteur: Expert ML & PyTorch
"""

from __future__ import annotations
import os
from typing import List, Optional, Dict, Tuple
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap
import seaborn as sns
from data_synth import Env

# Configuration matplotlib pour garantir de bons rendus
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
sns.set_palette("husl")


def _ensure_dir(directory: str):
    """Crée le dossier s'il n'existe pas."""
    os.makedirs(directory, exist_ok=True)


def _extract_phi_features(model: nn.Module, X: torch.Tensor, device: str = "cpu") -> np.ndarray:
    """
    Extrait la représentation Φ(X) depuis le modèle IRM.
    
    Pour IRMModel : utilise model.get_representation()
    Pour un MLP : extrait la dernière couche cachée (avant le classificateur final)
    Pour LogReg : retourne X directement (pas de transformation)
    
    Parameters
    ----------
    model : nn.Module
        Le modèle entraîné (ERM ou IRM)
    X : torch.Tensor
        Les données d'entrée (N, d_in)
    device : str
        Device pour le calcul
        
    Returns
    -------
    phi : np.ndarray
        La représentation latente Φ(X) de shape (N, d_latent)
    """
    model.eval()
    with torch.no_grad():
        X_dev = X.to(device)
        
        # Cas IRMModel : utiliser la méthode get_representation()
        if hasattr(model, 'get_representation'):
            phi = model.get_representation(X_dev).cpu().numpy()
        # Cas SmallMLP : extraire la représentation avant la dernière couche
        elif hasattr(model, 'net'):
            layers = list(model.net.children())
            # On retire la dernière couche (Linear final) pour obtenir Φ
            phi_net = nn.Sequential(*layers[:-1])
            phi = phi_net(X_dev).cpu().numpy()
        # Cas LogReg : pas de transformation, X est déjà la représentation
        else:
            phi = X_dev.cpu().numpy()
            
    return phi


def visualize_original_space(
    envs: List[Env],
    env_names: List[str],
    filename: str = "plots/original_space.png",
    max_points_per_env: int = 2000,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Visualise la distribution initiale des points (x, y) dans l'espace original.
    
    Cette fonction crée un nuage de points colorés par classe (Y) et stylisés 
    par environnement (marqueurs différents).
    
    Parameters
    ----------
    envs : List[Env]
        Liste des environnements à visualiser
    env_names : List[str]
        Noms des environnements (pour la légende)
    filename : str
        Chemin de sauvegarde du graphique
    max_points_per_env : int
        Nombre max de points à afficher par environnement (pour lisibilité)
    figsize : Tuple[int, int]
        Taille de la figure
    """
    _ensure_dir(os.path.dirname(filename))
    
    # Marqueurs différents pour chaque environnement
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Couleurs pour les classes (bleu pour Y=0, rouge pour Y=1)
    colors_class = {0: '#3498db', 1: '#e74c3c'}  # Bleu et Rouge
    
    fig = plt.figure(figsize=figsize)
    
    # Détecter la dimensionalité
    d_total = envs[0].X.shape[1]
    
    if d_total == 2:
        # Cas 2D : affichage direct (X_z, X_y)
        ax = fig.add_subplot(111)
        
        for env_idx, (env, env_name) in enumerate(zip(envs, env_names)):
            X_np = env.X.cpu().numpy()
            y_np = env.y.cpu().numpy()
            
            # S'assurer que y_np est 1D
            if y_np.ndim > 1:
                y_np = y_np.ravel()
            
            # Sous-échantillonner si nécessaire
            n_samples = min(len(X_np), max_points_per_env)
            indices = np.random.choice(len(X_np), n_samples, replace=False)
            X_sample = X_np[indices]
            y_sample = y_np[indices]
            
            marker = markers[env_idx % len(markers)]
            
            # Séparer par classe
            for class_val in [0, 1]:
                mask = (y_sample == class_val)
                if np.sum(mask) > 0:
                    ax.scatter(
                        X_sample[mask, 0], 
                        X_sample[mask, 1],
                        c=colors_class[class_val],
                        marker=marker,
                        s=50,
                        alpha=0.6,
                        edgecolors='black',
                        linewidths=0.5,
                        label=f"{env_name} - Y={class_val}"
                    )
        
        ax.set_xlabel("$X_z$ (Feature Causale)", fontsize=14, fontweight='bold')
        ax.set_ylabel("$X_y$ (Feature Spurieuse)", fontsize=14, fontweight='bold')
        ax.set_title("Espace Original : Distribution des Points par Environnement et Classe", 
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
    else:
        # Cas haute dimension : PCA ou affichage des 2 premières dimensions
        from sklearn.decomposition import PCA
        
        # Concaténer tous les environnements pour le PCA
        all_X = []
        all_y = []
        all_env_labels = []
        
        for env_idx, env in enumerate(envs):
            X_np = env.X.cpu().numpy()
            y_np = env.y.cpu().numpy().ravel()  # S'assurer que c'est 1D
            
            n_samples = min(len(X_np), max_points_per_env)
            indices = np.random.choice(len(X_np), n_samples, replace=False)
            
            all_X.append(X_np[indices])
            all_y.append(y_np[indices])
            all_env_labels.append(np.full(n_samples, env_idx))
        
        all_X = np.vstack(all_X)
        all_y = np.concatenate(all_y)
        all_env_labels = np.concatenate(all_env_labels)
        
        # Appliquer PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(all_X)
        
        ax = fig.add_subplot(111)
        
        for env_idx, env_name in enumerate(env_names):
            mask_env = (all_env_labels == env_idx)
            marker = markers[env_idx % len(markers)]
            
            for class_val in [0, 1]:
                mask = mask_env & (all_y == class_val)
                if np.sum(mask) > 0:
                    ax.scatter(
                        X_pca[mask, 0], 
                        X_pca[mask, 1],
                        c=colors_class[class_val],
                        marker=marker,
                        s=50,
                        alpha=0.6,
                        edgecolors='black',
                        linewidths=0.5,
                        label=f"{env_name} - Y={class_val}"
                    )
        
        variance_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({variance_explained[0]:.1%} variance)", fontsize=14, fontweight='bold')
        ax.set_ylabel(f"PC2 ({variance_explained[1]:.1%} variance)", fontsize=14, fontweight='bold')
        ax.set_title("Espace Original (PCA) : Distribution des Points par Environnement et Classe", 
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"✅ Visualisation de l'espace original sauvegardée: {filename}")
    plt.close()


def visualize_latent_space(
    model: nn.Module,
    envs: List[Env],
    env_names: List[str],
    filename: str = "plots/latent_space.png",
    max_points_per_env: int = 2000,
    device: str = "cpu",
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Visualise la transformation Φ(X) dans l'espace latent après entraînement.
    
    Cette fonction passe les points de test de chaque environnement à travers Φ(x)
    et affiche les points dans le nouvel espace latent.
    
    Parameters
    ----------
    model : nn.Module
        Le modèle entraîné (ERM ou IRM)
    envs : List[Env]
        Liste des environnements à visualiser
    env_names : List[str]
        Noms des environnements (pour la légende)
    filename : str
        Chemin de sauvegarde du graphique
    max_points_per_env : int
        Nombre max de points à afficher par environnement
    device : str
        Device pour les calculs
    figsize : Tuple[int, int]
        Taille de la figure
    """
    _ensure_dir(os.path.dirname(filename))
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors_class = {0: '#3498db', 1: '#e74c3c'}
    
    fig = plt.figure(figsize=figsize)
    
    # Extraire les représentations latentes
    all_phi = []
    all_y = []
    all_env_labels = []
    
    for env_idx, env in enumerate(envs):
        X_tensor = env.X
        y_np = env.y.cpu().numpy().ravel()  # S'assurer que c'est 1D
        
        # Sous-échantillonner
        n_samples = min(len(X_tensor), max_points_per_env)
        indices = np.random.choice(len(X_tensor), n_samples, replace=False)
        X_sample = X_tensor[indices]
        y_sample = y_np[indices]
        
        # Extraire Φ(X)
        phi = _extract_phi_features(model, X_sample, device)
        
        all_phi.append(phi)
        all_y.append(y_sample)
        all_env_labels.append(np.full(n_samples, env_idx))
    
    all_phi = np.vstack(all_phi)
    all_y = np.concatenate(all_y)
    all_env_labels = np.concatenate(all_env_labels)
    
    d_latent = all_phi.shape[1]
    
    if d_latent == 1:
        # Cas 1D : histogramme ou scatter plot 1D
        ax = fig.add_subplot(111)
        
        for env_idx, env_name in enumerate(env_names):
            mask_env = (all_env_labels == env_idx)
            marker = markers[env_idx % len(markers)]
            
            for class_val in [0, 1]:
                mask = mask_env & (all_y == class_val)
                if np.sum(mask) > 0:
                    # Scatter 1D (x=phi, y=random jitter pour visualisation)
                    jitter = np.random.randn(np.sum(mask)) * 0.02
                    ax.scatter(
                        all_phi[mask, 0],
                        jitter,
                        c=colors_class[class_val],
                        marker=marker,
                        s=50,
                        alpha=0.6,
                        edgecolors='black',
                        linewidths=0.5,
                        label=f"{env_name} - Y={class_val}"
                    )
        
        ax.set_xlabel(r"$\Phi(X)$ (Représentation Latente)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Jitter (pour visualisation)", fontsize=14, fontweight='bold')
        ax.set_title(r"Espace Latent $\Phi$ : Projection après Entraînement", 
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
    elif d_latent == 2:
        # Cas 2D : affichage direct
        ax = fig.add_subplot(111)
        
        for env_idx, env_name in enumerate(env_names):
            mask_env = (all_env_labels == env_idx)
            marker = markers[env_idx % len(markers)]
            
            for class_val in [0, 1]:
                mask = mask_env & (all_y == class_val)
                if np.sum(mask) > 0:
                    ax.scatter(
                        all_phi[mask, 0],
                        all_phi[mask, 1],
                        c=colors_class[class_val],
                        marker=marker,
                        s=50,
                        alpha=0.6,
                        edgecolors='black',
                        linewidths=0.5,
                        label=f"{env_name} - Y={class_val}"
                    )
        
        ax.set_xlabel(r"$\Phi_1(X)$", fontsize=14, fontweight='bold')
        ax.set_ylabel(r"$\Phi_2(X)$", fontsize=14, fontweight='bold')
        ax.set_title(r"Espace Latent $\Phi$ : Projection après Entraînement", 
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
    else:
        # Cas haute dimension : PCA sur l'espace latent
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        phi_pca = pca.fit_transform(all_phi)
        
        ax = fig.add_subplot(111)
        
        for env_idx, env_name in enumerate(env_names):
            mask_env = (all_env_labels == env_idx)
            marker = markers[env_idx % len(markers)]
            
            for class_val in [0, 1]:
                mask = mask_env & (all_y == class_val)
                if np.sum(mask) > 0:
                    ax.scatter(
                        phi_pca[mask, 0],
                        phi_pca[mask, 1],
                        c=colors_class[class_val],
                        marker=marker,
                        s=50,
                        alpha=0.6,
                        edgecolors='black',
                        linewidths=0.5,
                        label=f"{env_name} - Y={class_val}"
                    )
        
        variance_explained = pca.explained_variance_ratio_
        ax.set_xlabel(f"PC1 ({variance_explained[0]:.1%} variance)", fontsize=14, fontweight='bold')
        ax.set_ylabel(f"PC2 ({variance_explained[1]:.1%} variance)", fontsize=14, fontweight='bold')
        ax.set_title(r"Espace Latent $\Phi$ (PCA) : Projection après Entraînement", 
                     fontsize=16, fontweight='bold', pad=20)
        ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"✅ Visualisation de l'espace latent sauvegardée: {filename}")
    plt.close()


def compare_spaces_side_by_side(
    model: nn.Module,
    envs: List[Env],
    env_names: List[str],
    filename: str = "plots/comparison_original_vs_latent.png",
    max_points_per_env: int = 2000,
    device: str = "cpu",
    figsize: Tuple[int, int] = (20, 8)
):
    """
    Génère un graphique comparatif côte à côte : "Espace Original" vs "Espace Latent Φ".
    
    Cette fonction combine les deux visualisations précédentes en un seul graphique
    pour faciliter la comparaison.
    
    Parameters
    ----------
    model : nn.Module
        Le modèle entraîné (ERM ou IRM)
    envs : List[Env]
        Liste des environnements à visualiser
    env_names : List[str]
        Noms des environnements
    filename : str
        Chemin de sauvegarde du graphique
    max_points_per_env : int
        Nombre max de points par environnement
    device : str
        Device pour les calculs
    figsize : Tuple[int, int]
        Taille de la figure
    """
    _ensure_dir(os.path.dirname(filename))
    
    from sklearn.decomposition import PCA
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors_class = {0: '#3498db', 1: '#e74c3c'}
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # ========== PANNEAU GAUCHE : ESPACE ORIGINAL ==========
    all_X = []
    all_y = []
    all_env_labels = []
    
    for env_idx, env in enumerate(envs):
        X_np = env.X.cpu().numpy()
        y_np = env.y.cpu().numpy().ravel()  # S'assurer que c'est 1D
        
        n_samples = min(len(X_np), max_points_per_env)
        indices = np.random.choice(len(X_np), n_samples, replace=False)
        
        all_X.append(X_np[indices])
        all_y.append(y_np[indices])
        all_env_labels.append(np.full(n_samples, env_idx))
    
    all_X = np.vstack(all_X)
    all_y = np.concatenate(all_y)
    all_env_labels = np.concatenate(all_env_labels)
    
    # PCA si nécessaire
    if all_X.shape[1] > 2:
        pca_orig = PCA(n_components=2)
        X_vis = pca_orig.fit_transform(all_X)
        var_exp_orig = pca_orig.explained_variance_ratio_
        xlabel_orig = f"PC1 ({var_exp_orig[0]:.1%})"
        ylabel_orig = f"PC2 ({var_exp_orig[1]:.1%})"
    else:
        X_vis = all_X
        xlabel_orig = "$X_z$ (Causale)"
        ylabel_orig = "$X_y$ (Spurieuse)"
    
    for env_idx, env_name in enumerate(env_names):
        mask_env = (all_env_labels == env_idx)
        marker = markers[env_idx % len(markers)]
        
        for class_val in [0, 1]:
            mask = mask_env & (all_y == class_val)
            if np.sum(mask) > 0:
                ax1.scatter(
                    X_vis[mask, 0],
                    X_vis[mask, 1],
                    c=colors_class[class_val],
                    marker=marker,
                    s=40,
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f"{env_name} - Y={class_val}"
                )
    
    ax1.set_xlabel(xlabel_orig, fontsize=12, fontweight='bold')
    ax1.set_ylabel(ylabel_orig, fontsize=12, fontweight='bold')
    ax1.set_title("Espace Original", fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='best', fontsize=8, framealpha=0.9, ncol=1)
    ax1.grid(True, alpha=0.3)
    
    # ========== PANNEAU DROIT : ESPACE LATENT Φ ==========
    all_phi = []
    all_y_latent = []
    all_env_labels_latent = []
    
    for env_idx, env in enumerate(envs):
        X_tensor = env.X
        y_np = env.y.cpu().numpy().ravel()  # S'assurer que c'est 1D
        
        n_samples = min(len(X_tensor), max_points_per_env)
        indices = np.random.choice(len(X_tensor), n_samples, replace=False)
        X_sample = X_tensor[indices]
        y_sample = y_np[indices]
        
        phi = _extract_phi_features(model, X_sample, device)
        
        all_phi.append(phi)
        all_y_latent.append(y_sample)
        all_env_labels_latent.append(np.full(n_samples, env_idx))
    
    all_phi = np.vstack(all_phi)
    all_y_latent = np.concatenate(all_y_latent)
    all_env_labels_latent = np.concatenate(all_env_labels_latent)
    
    # PCA si nécessaire
    if all_phi.shape[1] > 2:
        pca_latent = PCA(n_components=2)
        phi_vis = pca_latent.fit_transform(all_phi)
        var_exp_latent = pca_latent.explained_variance_ratio_
        xlabel_latent = f"PC1 ({var_exp_latent[0]:.1%})"
        ylabel_latent = f"PC2 ({var_exp_latent[1]:.1%})"
    elif all_phi.shape[1] == 2:
        phi_vis = all_phi
        xlabel_latent = r"$\Phi_1(X)$"
        ylabel_latent = r"$\Phi_2(X)$"
    else:
        # Cas 1D
        phi_vis = np.column_stack([all_phi, np.random.randn(len(all_phi)) * 0.02])
        xlabel_latent = r"$\Phi(X)$"
        ylabel_latent = "Jitter"
    
    for env_idx, env_name in enumerate(env_names):
        mask_env = (all_env_labels_latent == env_idx)
        marker = markers[env_idx % len(markers)]
        
        for class_val in [0, 1]:
            mask = mask_env & (all_y_latent == class_val)
            if np.sum(mask) > 0:
                ax2.scatter(
                    phi_vis[mask, 0],
                    phi_vis[mask, 1],
                    c=colors_class[class_val],
                    marker=marker,
                    s=40,
                    alpha=0.6,
                    edgecolors='black',
                    linewidths=0.5,
                    label=f"{env_name} - Y={class_val}"
                )
    
    ax2.set_xlabel(xlabel_latent, fontsize=12, fontweight='bold')
    ax2.set_ylabel(ylabel_latent, fontsize=12, fontweight='bold')
    ax2.set_title(r"Espace Latent $\Phi$", fontsize=14, fontweight='bold', pad=15)
    ax2.legend(loc='best', fontsize=8, framealpha=0.9, ncol=1)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(r"Comparaison : Espace Original vs Espace Latent $\Phi$", 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"✅ Comparaison Original vs Latent sauvegardée: {filename}")
    plt.close()


def visualize_decision_boundary_1d(
    model: nn.Module,
    env: Env,
    filename: str = "plots/decision_boundary_1d.png",
    device: str = "cpu",
    num_points: int = 500
):
    """
    Visualise la frontière de décision pour un toy model 1D (X_z et X_y scalaires).
    
    Utile pour comprendre comment le modèle sépare les classes dans l'espace 2D.
    
    Parameters
    ----------
    model : nn.Module
        Le modèle entraîné
    env : Env
        L'environnement à visualiser
    filename : str
        Chemin de sauvegarde
    device : str
        Device pour les calculs
    num_points : int
        Résolution de la grille
    """
    _ensure_dir(os.path.dirname(filename))
    
    if env.X.shape[1] != 2:
        print(f"⚠️  La visualisation de frontière 1D nécessite exactement 2 features (X_z, X_y). "
              f"Trouvé: {env.X.shape[1]} features. Visualisation ignorée.")
        return
    
    X_np = env.X.cpu().numpy()
    y_np = env.y.cpu().numpy().ravel()  # S'assurer que c'est 1D
    
    # Limites du graphique
    x_min, x_max = X_np[:, 0].min() - 0.5, X_np[:, 0].max() + 0.5
    y_min, y_max = X_np[:, 1].min() - 0.5, X_np[:, 1].max() + 0.5
    
    # Grille
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, num_points),
                         np.linspace(y_min, y_max, num_points))
    
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(grid).cpu().numpy().ravel()
        probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    
    probs = probs.reshape(xx.shape)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Contour de la probabilité
    contour = ax.contourf(xx, yy, probs, levels=20, cmap='RdYlBu_r', alpha=0.6)
    ax.contour(xx, yy, probs, levels=[0.5], colors='black', linewidths=2)
    
    # Points réels
    colors_class = {0: '#3498db', 1: '#e74c3c'}
    for class_val in [0, 1]:
        mask = (y_np == class_val)
        ax.scatter(X_np[mask, 0], X_np[mask, 1], 
                   c=colors_class[class_val], 
                   s=30, alpha=0.8, edgecolors='black', linewidths=0.5,
                   label=f"Y={class_val}")
    
    plt.colorbar(contour, ax=ax, label="P(Y=1)")
    ax.set_xlabel("$X_z$ (Feature Causale)", fontsize=12, fontweight='bold')
    ax.set_ylabel("$X_y$ (Feature Spurieuse)", fontsize=12, fontweight='bold')
    ax.set_title("Frontière de Décision dans l'Espace Original", fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"✅ Frontière de décision sauvegardée: {filename}")
    plt.close()


def generate_all_visualizations(
    erm_model: nn.Module,
    irm_model: nn.Module,
    train_envs: List[Env],
    test_env: Env,
    output_dir: str = "plots",
    device: str = "cpu",
    max_points: int = 2000
):
    """
    Génère toutes les visualisations pour l'analyse IRM.
    
    Cette fonction est un wrapper pratique qui génère automatiquement :
    - Espace original (tous les environnements)
    - Espace latent ERM
    - Espace latent IRM
    - Comparaisons côte à côte
    
    Parameters
    ----------
    erm_model : nn.Module
        Le modèle ERM entraîné
    irm_model : nn.Module
        Le modèle IRM entraîné
    train_envs : List[Env]
        Environnements de training
    test_env : Env
        Environnement de test OOD
    output_dir : str
        Dossier de sortie
    device : str
        Device pour les calculs
    max_points : int
        Nombre max de points par environnement
    """
    _ensure_dir(output_dir)
    
    all_envs = train_envs + [test_env]
    env_names = [f"Train {i+1}" for i in range(len(train_envs))] + ["Test OOD"]
    
    print("\n" + "="*60)
    print("📊 GÉNÉRATION DES VISUALISATIONS IRM")
    print("="*60)
    
    # 1. Espace original
    print("\n1️⃣  Visualisation de l'espace original...")
    visualize_original_space(
        all_envs, 
        env_names, 
        filename=os.path.join(output_dir, "original_space.png"),
        max_points_per_env=max_points
    )
    
    # 2. Espace latent ERM
    print("\n2️⃣  Visualisation de l'espace latent ERM...")
    visualize_latent_space(
        erm_model,
        all_envs,
        env_names,
        filename=os.path.join(output_dir, "latent_space_erm.png"),
        max_points_per_env=max_points,
        device=device
    )
    
    # 3. Espace latent IRM
    print("\n3️⃣  Visualisation de l'espace latent IRM...")
    visualize_latent_space(
        irm_model,
        all_envs,
        env_names,
        filename=os.path.join(output_dir, "latent_space_irm.png"),
        max_points_per_env=max_points,
        device=device
    )
    
    # 4. Comparaison ERM
    print("\n4️⃣  Comparaison Original vs Latent (ERM)...")
    compare_spaces_side_by_side(
        erm_model,
        all_envs,
        env_names,
        filename=os.path.join(output_dir, "comparison_erm.png"),
        max_points_per_env=max_points,
        device=device
    )
    
    # 5. Comparaison IRM
    print("\n5️⃣  Comparaison Original vs Latent (IRM)...")
    compare_spaces_side_by_side(
        irm_model,
        all_envs,
        env_names,
        filename=os.path.join(output_dir, "comparison_irm.png"),
        max_points_per_env=max_points,
        device=device
    )
    
    # 6. Frontières de décision (si 2D)
    if all_envs[0].X.shape[1] == 2:
        print("\n6️⃣  Frontières de décision...")
        visualize_decision_boundary_1d(
            erm_model,
            test_env,
            filename=os.path.join(output_dir, "decision_boundary_erm.png"),
            device=device
        )
        visualize_decision_boundary_1d(
            irm_model,
            test_env,
            filename=os.path.join(output_dir, "decision_boundary_irm.png"),
            device=device
        )
    
    print("\n" + "="*60)
    print("✅ TOUTES LES VISUALISATIONS ONT ÉTÉ GÉNÉRÉES !")
    print(f"📂 Dossier: {output_dir}/")
    print("="*60 + "\n")
"""
Nouvelle fonction de visualisation basée sur la densité (KDE).
À ajouter dans visualization_irm.py
"""

def visualize_density_with_marginals(
    envs: List[Env],
    env_names: List[str],
    filename: str,
    max_points_per_env: int = 2000,
    figsize: Tuple[int, int] = (12, 10),
    kde_levels: int = 8,
    title: str = None
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
    
    # Axes principaux (NE PAS partager les axes avec les marginales pour éviter les problèmes d'échelle)
    ax_main = fig.add_subplot(gs[1, 1])      # Contours centraux
    ax_top = fig.add_subplot(gs[0, 1])       # Marginal X (PC1) - pas de sharex
    ax_right = fig.add_subplot(gs[1, 2])     # Marginal Y (PC2) - pas de sharey
    
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
        # Détection d'échelle petite (IRM typiquement)
        std_pc2 = X_plot[:, 1].std()
        use_scatter = std_pc2 < 0.15  # Si std très faible, utiliser scatter
        
        if use_scatter:
            # Scatter plot pour distributions très comprimées (ex: IRM)
            if env_idx == 0:
                print(f"ℹ️  Petite échelle détectée (std(PC2)={std_pc2:.3f})")
                print(f"   Utilisant scatter plot au lieu de KDE...")
            
            for class_val in [0, 1]:
                mask = (y_sample == class_val)
                if np.sum(mask) > 0:
                    marker = 'o' if class_val == 1 else 'x'
                    # Points plus gros et plus opaques pour visibilité
                    ax_main.scatter(X_plot[mask, 0], X_plot[mask, 1],
                                   c=color, marker=marker, s=50, alpha=0.6,
                                   edgecolors='white', linewidths=0.5,
                                   label=f"{env_name} - Y={class_val}")
        else:
            # KDE normal pour distributions bien étalées (ex: ERM)
            try:
                for class_val, alpha_val in [(0, 0.3), (1, 0.5)]:
                    mask = (y_sample == class_val)
                    if np.sum(mask) < 10:
                        continue
                    
                    X_class = X_plot[mask]
                    kde = gaussian_kde(X_class.T)
                    
                    # Grille pour évaluation
                    x_min, x_max = X_plot[:, 0].min() - 0.5, X_plot[:, 0].max() + 0.5
                    y_min, y_max = X_plot[:, 1].min() - 0.5, X_plot[:, 1].max() + 0.5
                    xx, yy = np.meshgrid(
                        np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100)
                    )
                    
                    positions = np.vstack([xx.ravel(), yy.ravel()])
                    density = kde(positions).reshape(xx.shape)
                    
                    linestyle = '-' if class_val == 1 else '--'
                    ax_main.contour(xx, yy, density, levels=kde_levels, 
                                   colors=color, alpha=alpha_val, 
                                   linewidths=1.5, linestyles=linestyle)
            except Exception as e:
                # Fallback si KDE échoue
                if env_idx == 0:
                    print(f"⚠️  KDE failed: {e}, using scatter...")
                for class_val in [0, 1]:
                    mask = (y_sample == class_val)
                    if np.sum(mask) > 0:
                        marker = 'o' if class_val == 1 else 'x'
                        ax_main.scatter(X_plot[mask, 0], X_plot[mask, 1],
                                       c=color, marker=marker, s=50, alpha=0.6,
                                       edgecolors='white', linewidths=0.5,
                                       label=f"{env_name} - Y={class_val}")
        
        # ===== DISTRIBUTIONS MARGINALES =====
        # Distribution X (PC1) (axe horizontal, en haut)
        # NOTE: Utiliser density=False pour éviter les échelles de densité extrêmes
        # qui compriment le plot central quand les données sont très concentrées
        ax_top.hist(X_plot[:, 0], bins=30, color=color, alpha=0.4, 
                    density=False, histtype='stepfilled', edgecolor=color, linewidth=1.5)
        
        # Distribution Y (PC2) (axe vertical, à droite)
        ax_right.hist(X_plot[:, 1], bins=30, color=color, alpha=0.4,
                     density=False, orientation='horizontal', 
                     histtype='stepfilled', edgecolor=color, linewidth=1.5)
    
    # ===== STYLE =====
    # Panneau central
    ax_main.set_xlabel(xlabel, fontsize=14, fontweight='bold')
    ax_main.set_ylabel(ylabel, fontsize=14, fontweight='bold')
    ax_main.grid(True, alpha=0.3)
    
    # Marges - Suppression stricte des labels et ticks
    ax_top.set_ylabel("") 
    ax_top.set_yticks([]) # Supprime les ticks explicitement
    
    plot_title = title if title else 'Distribution des Données (Densité KDE)'
    ax_top.set_title(plot_title, fontsize=16, fontweight='bold', pad=20)
    ax_top.grid(True, alpha=0.3, axis='y')
    
    ax_right.set_xlabel("")
    ax_right.set_xticks([]) # Supprime les ticks explicitement
    ax_right.grid(True, alpha=0.3, axis='x')
    
    # Synchroniser manuellement les limites des axes pour les marginales
    xlim = ax_main.get_xlim()
    ylim = ax_main.get_ylim()
    ax_top.set_xlim(xlim)
    ax_right.set_ylim(ylim)
    
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
    bins: int = 50,
    title: str = None
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
    
    plot_title = title if title else 'Distribution des Données par Environnement (Heatmap)'
    plt.suptitle(plot_title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Heatmap de densité sauvegardée: {filename}")

def visualize_density_heatmap_by_class(
    envs: List[Env],
    env_names: List[str],
    filename: str,
    max_points_per_env: int = 5000,
    figsize: Tuple[int, int] = (16, 8),
    bins: int = 50,
    title: str = None
):
    """
    Visualise les distributions avec des heatmaps 2D SÉPARÉES par classe (Y=0 et Y=1).
    Deux heatmaps par environnement pour voir la distinction des labels.
    
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
    title : str
        Titre personnalisé
    """
    _ensure_dir(os.path.dirname(filename))
    
    n_envs = len(envs)
    # 2 rows (Y=0 et Y=1) x n_envs columns
    fig, axes = plt.subplots(2, n_envs, figsize=figsize, sharex=True, sharey=True)
    
    if n_envs == 1:
        axes = axes.reshape(2, 1)
    
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
    for env_idx, (env, env_name) in enumerate(zip(envs, env_names)):
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
        
        # Séparer par classe
        for class_idx, class_val in enumerate([0, 1]):
            ax = axes[class_idx, env_idx]
            mask = (y_sample == class_val)
            X_class = X_plot[mask]
            
            if len(X_class) > 0:
                # Histogramme 2D pour cette classe
                h = ax.hist2d(X_class[:, 0], X_class[:, 1], bins=bins,
                              cmap='viridis', 
                              range=[[x_min - x_margin, x_max + x_margin],
                                     [y_min - y_margin, y_max + y_margin]],
                              vmin=0)  # Même échelle pour comparaison
                
                # Colorbar
                plt.colorbar(h[3], ax=ax, label='Densité')
            
            # Labels
            if env_idx == 0:
                ax.set_ylabel(f"Y={class_val}\n{ylabel}", fontsize=12, fontweight='bold')
            else:
                ax.set_ylabel(f"Y={class_val}", fontsize=10)
            
            if class_idx == 0:
                ax.set_title(env_name, fontsize=14, fontweight='bold', pad=10)
            
            if class_idx == 1:  # Bottom row
                ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
            
            ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
            
            # Annotation du nombre de points
            ax.text(0.02, 0.98, f'n={len(X_class)}', 
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plot_title = title if title else 'Distribution par Environnement et par Classe (Heatmap)'
    plt.suptitle(plot_title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Heatmap par classe sauvegardée: {filename}")

def visualize_decision_boundaries(
    model,
    envs: List[Env],
    env_names: List[str],
    filename: str,
    model_name: str = "Model",
    max_points_per_env: int = 1000,
    figsize: Tuple[int, int] = (16, 5),
    resolution: int = 200
):
    """
    Visualise les frontières de décision d'un modèle sur différents environnements.
    
    Parameters
    ----------
    model : nn.Module
        Modèle PyTorch (ERM ou IRM)
    envs : List[Env]
        Liste des environnements
    env_names : List[str]
        Noms des environnements
    filename : str
        Chemin de sauvegarde
    model_name : str
        Nom du modèle pour le titre
    max_points_per_env : int
        Points max à afficher par environnement
    figsize : Tuple[int, int]
        Taille de la figure
    resolution : int
        Résolution de la grille pour la frontière
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
        print(f"⚠️  Dimensionalité {d_total}D > 2, utilisant PCA pour visualisation")
    else:
        pca = None
        xlabel = "$X_0$"
        ylabel = "$X_1$"
    
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
    
    # Marges raisonnables pour une bonne échelle visuelle
    x_margin = (x_max - x_min) * 0.2  # 20% margin
    y_margin = (y_max - y_min) * 0.2
    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin
    
    # Grille pour la frontière de décision
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution)
    )
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Prédictions sur la grille
    # Si PCA, il faut revenir dans l'espace original
    if pca is not None:
        # Inverse transform PCA
        grid_points_original = pca.inverse_transform(grid_points)
    else:
        grid_points_original = grid_points
    
    # Prédire avec le modèle
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.FloatTensor(grid_points_original).to(next(model.parameters()).device)
        logits = model(grid_tensor)
        if logits.ndim == 1:  # Output 1D -> unsqueeze
            logits = logits.unsqueeze(1)
        if logits.shape[1] == 1:  # Binary classification avec 1 sortie
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
        else:  # 2 sorties
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    
    Z = probs.reshape(xx.shape)
    
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
        
        # Frontière de décision (contour à 0.5)
        contour = ax.contourf(xx, yy, Z, levels=[0, 0.5, 1], 
                              colors=['#ff6b6b', '#4ecdc4'], alpha=0.3)
        
        # Vérifier si la frontière à 0.5 existe dans la zone
        has_boundary = (Z.min() < 0.5) and (Z.max() > 0.5)
        
        if has_boundary:
            # Frontière normale à 0.5
            ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2.5, 
                      linestyles='-', alpha=0.8)
        else:
            # Pas de frontière à 0.5 visible, montrer des iso-contours
            median_prob = np.median(Z)
            levels = [np.percentile(Z, 25), median_prob, np.percentile(Z, 75)]
            
            # Filtrer les niveaux pour ne garder que ceux qui sont strictement croissants
            unique_levels = sorted(list(set(levels)))
            if len(unique_levels) > 1:
                ax.contour(xx, yy, Z, levels=unique_levels, colors='black', linewidths=1.5, 
                          linestyles='--', alpha=0.5)
            
            if env_idx == 0:
                print(f"  ⚠️  Frontière à 0.5 hors zone visible (probs: {Z.min():.3f}-{Z.max():.3f})")
                if len(unique_levels) > 1:
                    print(f"      Affichage de {len(unique_levels)} iso-contours")
                else:
                    print(f"      Prédictions constantes ({unique_levels[0]:.3f}), pas de contours à afficher")
        
        # Points de données
        for class_val in [0, 1]:
            mask = (y_sample == class_val)
            if np.sum(mask) > 0:
                color = '#d62728' if class_val == 0 else '#1f77b4'  # Rouge vs Bleu
                marker = 'x' if class_val == 0 else 'o'
                label = f'Y={class_val}'
                ax.scatter(X_plot[mask, 0], X_plot[mask, 1],
                          c=color, marker=marker, s=40, alpha=0.7,
                          edgecolors='white', linewidths=0.5,
                          label=label, zorder=10)
        
        # Calcul de l'accuracy sur cet environnement
        with torch.no_grad():
            X_env_tensor = torch.FloatTensor(X_np).to(next(model.parameters()).device)
            logits_env = model(X_env_tensor)
            if logits_env.ndim == 1:
                logits_env = logits_env.unsqueeze(1)
            if logits_env.shape[1] == 1:
                preds = (torch.sigmoid(logits_env) > 0.5).float().cpu().numpy().ravel()
            else:
                preds = torch.argmax(logits_env, dim=1).cpu().numpy()
            acc = (preds == y_np).mean()
        
        # Titre avec accuracy
        ax.set_title(f"{env_name}\nAcc: {acc*100:.1f}%", 
                    fontsize=14, fontweight='bold', pad=10)
        
        if env_idx == 0:
            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
        
        ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.suptitle(f'Frontières de Décision - {model_name}', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"✅ Frontières de décision sauvegardées: {filename}")
