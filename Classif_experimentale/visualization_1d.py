"""
Visualisation 1D pour représentations quasi-1D (typique d'IRM en classification binaire)
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from data_synth import Env


def visualize_phi_1d_histogram(
    phi_features: List[np.ndarray],  # [phi_train1, phi_train2, phi_test] - 1D arrays
    Y_labels: List[np.ndarray],      # [Y_train1, Y_train2, Y_test] - label arrays
    env_names: List[str],
    filename: str,
    title: str = "Distribution 1D de Φ(X)"
):
    """
    Visualise la distribution 1D de la composante principale de Φ(X).
    
    Utilisé quand la représentation est quasi-1D (ex: IRM en classif binaire).
    
    Parameters
    ----------
    phi_features : List[np.ndarray]
        Liste des features Φ(X) pour chaque environnement, shape (n_samples,)
    Y_labels : List[np.ndarray]
        Liste des labels Y pour chaque environnement, shape (n_samples,)
    env_names : List[str]
        Noms des environnements
    filename : str
        Chemin du fichier de sortie
    title : str
        Titre du plot
    """
    n_envs = len(phi_features)
    fig, axes = plt.subplots(1, n_envs, figsize=(5 * n_envs, 4), sharey=True)
    
    if n_envs == 1:
        axes = [axes]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Bleu, Rouge, Vert
    
    for idx, (phi, Y, env_name, ax) in enumerate(zip(phi_features, Y_labels, env_names, axes)):
        # Séparer par classe
        phi_y0 = phi[Y == 0]
        phi_y1 = phi[Y == 1]
        
        # Limites globales
        all_vals = np.concatenate([phi_y0, phi_y1])
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = (vmax - vmin) * 0.1
        bins = np.linspace(vmin - margin, vmax + margin, 40)
        
        # Histogrammes
        ax.hist(phi_y0, bins=bins, alpha=0.6, label='Y=0', color='#e74c3c', 
                edgecolor='white', linewidth=0.5)
        ax.hist(phi_y1, bins=bins, alpha=0.6, label='Y=1', color='#3498db',
                edgecolor='white', linewidth=0.5)
        
        # Moyennes
        mean_y0 = phi_y0.mean()
        mean_y1 = phi_y1.mean()
        ax.axvline(mean_y0, color='#c0392b', linestyle='--', linewidth=2, 
                   label=f'μ(Y=0)={mean_y0:.2f}')
        ax.axvline(mean_y1, color='#2980b9', linestyle='--', linewidth=2,
                   label=f'μ(Y=1)={mean_y1:.2f}')
        
        ax.set_xlabel('PC1 (Composante principale)', fontsize=11)
        if idx == 0:
            ax.set_ylabel('Effectif', fontsize=11)
        ax.set_title(env_name, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Histogram 1D sauvegardé: {filename}")
