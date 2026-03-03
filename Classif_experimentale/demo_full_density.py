#!/usr/bin/env python3
"""
Démonstration complète : Distributions originales vs latentes (Φ)
avec visualisations basées sur la densité.

Ce script montre :
1. Distribution originale (X) avec KDE
2. Distribution latente IRM (Φ_IRM(X)) avec KDE
3. Distribution latente ERM (Φ_ERM(X)) avec KDE
4. Comparaisons côte à côte
"""

import torch
import numpy as np
from data_synth import build_envs_semi_anti_causal, Env
from models_training import train_erm, train_irm
from visualization_irm import (
    visualize_density_with_marginals,
    visualize_density_heatmap,
    visualize_density_heatmap_by_class,
    visualize_decision_boundaries,
    _extract_phi_features
)
from visualization_1d import visualize_phi_1d_histogram
from sklearn.decomposition import PCA

def main():
    print("="*70)
    print(" 🧪 ANALYSE COMPLÈTE : DISTRIBUTIONS ORIGINALES vs LATENTES")
    print("="*70)
    print()
    
    # ===== CONFIGURATION =====
    # 🔧 ARCHITECTURE IRM : Architecture fidèle au concept original
    MODEL_KIND = "irm_faithful"  # "irm_faithful", "mlp", ou "logreg"
    PHI_TYPE = "linear"          # "linear" (matrice) ou "mlp" (non-linéaire)
    PHI_REPR_DIM = None             # Dimension de Φ(X) : 2 pour visualisation, None pour d_in
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    output_dir = "plots_full_density"
    
    print("📋 Configuration:")
    print(f"   - Device: {device}")
    print(f"   - Seed: {seed}")
    print(f"   - Architecture: {MODEL_KIND.upper()}")
    if MODEL_KIND == "irm_faithful":
        print(f"   - Phi type: {PHI_TYPE}")
        print(f"   - Phi repr dim: {PHI_REPR_DIM if PHI_REPR_DIM else 'd_in'}")
    print(f"   - Output: {output_dir}/")
    print()
    
    # ===== 1. GÉNÉRATION DES DONNÉES =====
    print("-"*70)
    print("1️⃣  Génération des environnements...")
    print("-"*70)
    
    train_envs, val_envs, test_env = build_envs_semi_anti_causal(
        n=3000,
        train_p_spurs=[0.2, 0.1],
        test_p_spur=0.9,
        seed=seed,
        val_frac=0.1,
        label_flip=0.25,
        n_test=300,
        dim_z=1,
        dim_y=1,
        causal_strength=1.5  # ✨ Séparation modérée avec échelle raisonnable
    )
    
    print(f"✅ {len(train_envs)} train envs + 1 test env")
    print()
    
    # ===== 2. ENTRAÎNEMENT DES MODÈLES =====
    print("-"*70)
    print("2️⃣  Entraînement ERM et IRM...")
    print("-"*70)
    
    # Hyperparamètres conditionnels selon le modèle
    if MODEL_KIND == "irm_faithful":
        # ✅ Architecture fidèle IRM avec Phi séparé
        # IMPORTANT: Utiliser MÊME architecture pour ERM et IRM (comparaison juste!)
        erm_steps, irm_steps = 10000, 10000
        lr = 1e-3
        
        # MÊME ARCHITECTURE pour les deux
        shared_kwargs = {
            "model_kind": "irm_faithful",
            "phi_type": PHI_TYPE,
            "phi_repr_dim": PHI_REPR_DIM,
            "phi_hidden": 64
        }
        erm_kwargs = shared_kwargs.copy()
        irm_kwargs = shared_kwargs.copy()
        
    elif MODEL_KIND == "logreg":
        erm_steps, irm_steps = 5000, 5000
        lr = 1e-2
        erm_kwargs = irm_kwargs = {"model_kind": "logreg"}
    else:  # mlp
        erm_steps, irm_steps = 10000, 10000
        lr = 1e-3
        erm_kwargs = irm_kwargs = {"model_kind": "mlp", "mlp_hidden": 64, "mlp_layers": 2}
    
    print(f"   Training ERM ({erm_kwargs.get('model_kind', MODEL_KIND).upper()})...")
    erm_model, _ = train_erm(
        envs=train_envs,
        steps=erm_steps,
        lr=lr,
        device=device,
        seed=seed,
        **erm_kwargs
    )
    
    print(f"   Training IRM ({MODEL_KIND.upper()})...")
    irm_model, _ = train_irm(
        envs=train_envs,
        steps=irm_steps,
        lr=lr,
        irm_lambda=100.0,
        device=device,
        seed=seed + 1,
        **irm_kwargs
    )
    
    print("✅ Modèles entraînés")
    print()
    
    # ===== 3. EXTRACTION DES FEATURES =====
    print("-"*70)
    if MODEL_KIND == "logreg":
        print("3️⃣  Extraction des représentations Φ(X)...")  # Pour LogReg: Φ(X) = X
    else:
        print("3️⃣  Extraction des représentations internes du MLP...")
    print("-"*70)
    
    def create_latent_envs(model, envs, device):
        """Crée des environnements avec les activations de couches cachées au lieu de X"""
        latent_envs = []
        for env in envs:
            phi_features = _extract_phi_features(model, env.X, device)
            # Créer un nouveau Env avec Φ(X)
            latent_env = Env(
                X=torch.from_numpy(phi_features).float(),
                y=env.y,
                y_true=env.y_true if hasattr(env, 'y_true') else None,
                meta=env.meta if hasattr(env, 'meta') else None
            )
            latent_envs.append(latent_env)
        return latent_envs
    
    erm_latent_train = create_latent_envs(erm_model, train_envs, device)
    erm_latent_test = create_latent_envs(erm_model, [test_env], device)[0]
    
    irm_latent_train = create_latent_envs(irm_model, train_envs, device)
    irm_latent_test = create_latent_envs(irm_model, [test_env], device)[0]
    
    print(f"✅ Features internes extraites (dim = {erm_latent_train[0].X.shape[1]})")
    print()
    
    # ===== 4. VISUALISATIONS =====
    env_names = ["Train 1", "Train 2", "Test OOD"]
    
    print("-"*70)
    print("4️⃣  Visualisation : Espace ORIGINAL (X)...")
    print("-"*70)
    
    visualize_density_with_marginals(
        envs=train_envs + [test_env],
        env_names=env_names,
        filename=f"{output_dir}/01_original_kde.png",
        max_points_per_env=2000,
        kde_levels=10
    )
    
    # visualize_density_heatmap(
    #     envs=train_envs + [test_env],
    #     env_names=env_names,
    #     filename=f"{output_dir}/02_original_heatmap.png",
    #     bins=50,
    #     title="Distribution des Données Originales (Heatmap Totale)"
    # )
    
    visualize_density_heatmap_by_class(
        envs=train_envs + [test_env],
        env_names=env_names,
        filename=f"{output_dir}/02b_original_heatmap_by_class.png",
        bins=50,
        title="Distribution des Données Originales par Classe"
    )
    print()
    
    print("-"*70)
    print("5️⃣  Visualisation : Features Internes ERM (sans contrainte d'invariance)...")
    print("-"*70)
    
    visualize_density_with_marginals(
        envs=erm_latent_train + [erm_latent_test],
        env_names=env_names,
        filename=f"{output_dir}/03_erm_hidden_features_kde.png",
        max_points_per_env=2000,
        kde_levels=10,
        title="Features Internes ERM (MLP sans contrainte d'invariance)"
    )
    
    # visualize_density_heatmap(
    #     envs=erm_latent_train + [erm_latent_test],
    #     env_names=env_names,
    #     filename=f"{output_dir}/04_erm_hidden_features_heatmap.png",
    #     bins=50,
    #     title="Features Internes ERM (Heatmap Totale)"
    # )
    
    visualize_density_heatmap_by_class(
        envs=erm_latent_train + [erm_latent_test],
        env_names=env_names,
        filename=f"{output_dir}/04b_erm_hidden_features_heatmap_by_class.png",
        bins=50,
        title="Features Internes ERM par Classe"
    )
    print()
    
    print("-"*70)
    print("6️⃣  Visualisation : Représentation Invariante IRM Φ(X)...")
    print("-"*70)
    
    # Vérifier si la représentation IRM est quasi-1D
    all_phi = []
    for env in irm_latent_train:
        all_phi.append(env.X.cpu().numpy())
    phi_concat = np.vstack(all_phi)
    
    # Test avec PCA
    pca_test = PCA(n_components=2)
    phi_2d = pca_test.fit_transform(phi_concat)
    pc2_std = phi_2d[:, 1].std()
    is_quasi_1d = pc2_std < 0.05
    
    if is_quasi_1d:
        print(f"ℹ️  Représentation IRM quasi-1D détectée (PC2 std={pc2_std:.4f})")
        print(f"   Utilisation d'histogrammes 1D au lieu de heatmaps 2D...")
        
        # Extraire PC1 pour chaque environnement
        phi_1d_list = []
        Y_list = []
        for env in irm_latent_train + [irm_latent_test]:
            phi_env = env.X.cpu().numpy()
            phi_transformed = pca_test.transform(phi_env)
            phi_1d_list.append(phi_transformed[:, 0])  # PC1 seulement
            Y_list.append(env.y.cpu().numpy().flatten())
        
        # Visualisation 1D
        visualize_phi_1d_histogram(
            phi_features=phi_1d_list,
            Y_labels=Y_list,
            env_names=env_names,
            filename=f"{output_dir}/05_irm_phi_histogram_1d.png",
            title="Distribution 1D de la Représentation Invariante IRM Φ(X)"
        )
    else:
        print(f"ℹ️  Représentation IRM multi-dimensionnelle (PC2 std={pc2_std:.4f})")
        print(f"   Utilisation de visualisations 2D standard...")
        
        # visualize_density_heatmap(
        #     envs=irm_latent_train + [irm_latent_test],
        #     env_names=env_names,
        #     filename=f"{output_dir}/05_irm_representation_phi_heatmap.png",
        #     bins=50,
        #     title="Représentation Invariante IRM Φ(X) (Heatmap Totale)"
        # )
        
        visualize_density_heatmap_by_class(
            envs=irm_latent_train + [irm_latent_test],
            env_names=env_names,
            filename=f"{output_dir}/05b_irm_representation_phi_heatmap_by_class.png",
            bins=50,
            title="Représentation Invariante IRM Φ(X) par Classe"
        )
        
        visualize_density_with_marginals(
            envs=irm_latent_train + [irm_latent_test],
            env_names=env_names,
            filename=f"{output_dir}/06_irm_representation_phi_kde.png",
            max_points_per_env=2000,
            kde_levels=10,
            title="Représentation Invariante IRM Φ(X) - Densité KDE"
        )
    print()
    
    # ===== 7. FRONTIÈRES DE DÉCISION =====
    print("-"*70)
    print("7️⃣  Frontières de décision...")
    print("-"*70)
    
    # Frontières de décision ERM sur données originales
    visualize_decision_boundaries(
        model=erm_model,
        envs=train_envs + [test_env],
        env_names=env_names,
        filename=f"{output_dir}/07_erm_decision_boundaries_original.png",
        model_name="ERM - Données Originales",
        max_points_per_env=500
    )
    
    # Frontières de décision IRM sur données originales
    visualize_decision_boundaries(
        model=irm_model,
        envs=train_envs + [test_env],
        env_names=env_names,
        filename=f"{output_dir}/08_irm_decision_boundaries_original.png",
        model_name="IRM - Données Originales",
        max_points_per_env=500
    )
    print()
    
    # ===== RÉSUMÉ =====
    print("="*70)
    print(" ✅ ANALYSE TERMINÉE !")
    print("="*70)
    print()
    print(f"📂 Visualisations sauvegardées dans: {output_dir}/")
    print()
    print("📊 Fichiers créés:")
    print("   01_original_kde.png                    - X (espace original, KDE)")
    print("   02_original_heatmap.png                - X (espace original, heatmap)")
    print("   03_erm_hidden_features_kde.png         - ERM features internes (KDE)")
    print("   04_erm_hidden_features_heatmap.png     - ERM features internes (heatmap)")
    print("   05_irm_representation_phi_heatmap.png  - IRM Φ(X) invariant (heatmap) ⭐")
    print("   06_irm_representation_phi_kde.png      - IRM Φ(X) invariant (KDE) ⭐")
    print()
    print("🔍 À observer:")
    print("   - Original: Deux distributions gaussiennes séparées (Y=0 vs Y=1)")
    print("   - ERM features: Environnements toujours séparés (exploite corrélations spurieuses)")
    print("   - IRM Φ(X): Environnements ALIGNÉS ! (représentation invariante trouvée) ⭐")
    print()
    print("="*70)


if __name__ == "__main__":
    main()
