
import numpy as np
import torch
import matplotlib.pyplot as plt
import data_nlp
from sklearn.decomposition import PCA
import os

def run_confounding_analysis(
    n_samples=1000, 
    gammas=[5.0, 2.0], 
    seed=42
):
    print(f"Running Confounding Analysis with gammas={gammas}...")
    output_dir = "confounding_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate environments
    # We treat all gammas as "train" environments to visualize the progression
    # The last one can be considered "test" or just part of the spectrum
    train_envs, _, _ = data_nlp.build_envs_nlp_varying_confounder(
        n=n_samples,
        train_gammas=gammas,
        test_gamma=0.0, # Dummy
        seed=seed
    )
    
    results = []
    
    plt.figure(figsize=(20, 5 * ((len(train_envs) + 1) // 4 + 1)))
    
    for i, env in enumerate(train_envs):
        gamma = env.meta["gamma"]
        X = env.X.numpy()
        Y = env.y.numpy().flatten()
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        # Calculate centroids
        mask_0 = (Y == 0)
        mask_1 = (Y == 1)
        
        c0 = np.mean(X[mask_0], axis=0) if np.any(mask_0) else np.zeros(X.shape[1])
        c1 = np.mean(X[mask_1], axis=0) if np.any(mask_1) else np.zeros(X.shape[1])
        
        # Distance
        dist = np.linalg.norm(c0 - c1)
        
        # Check alignment with [WINNER] token ?
        # Ideally we would project on the (Winner - News) axis
        
        results.append({
            "gamma": gamma,
            "dist": dist,
            "n_spam": np.sum(Y)
        })
        
        # Plot
        plt.subplot(1, len(train_envs), i+1)
        plt.scatter(X_pca[mask_0, 0], X_pca[mask_0, 1], c='blue', alpha=0.3, label='Pred HAM', s=10)
        plt.scatter(X_pca[mask_1, 0], X_pca[mask_1, 1], c='red', alpha=0.3, label='Pred SPAM', s=10)
        plt.title(f"Gamma = {gamma}\nDist: {dist:.2f} | Spam%: {np.mean(Y):.1%}")
        if i == 0:
            plt.legend()
            
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confounding_spectrum.png"))
    print(f"Plot saved to {output_dir}/confounding_spectrum.png")
    
    # Text report
    print("\nResults:")
    print(f"{'Gamma':<10} | {'Dist':<10} | {'Spam %':<10}")
    print("-" * 35)
    for r in results:
        print(f"{r['gamma']:<10} | {r['dist']:<10.4f} | {r['n_spam']/n_samples:<10.1%}")

if __name__ == "__main__":
    run_confounding_analysis()
