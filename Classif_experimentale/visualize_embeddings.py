
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import data_nlp
from data_nlp import build_envs_nlp_size_selection, load_sms_spam_dataset, Env, tokenize_and_embed_with_bert
import os

# --- Monkey Patch to speed up execution ---
original_load_dataset = data_nlp.load_sms_spam_dataset

def mocked_load_dataset(seed=42):
    print("⚠️  USING SUBSET OF DATA FOR VISUALIZATION SPEED ⚠️")
    texts, labels = original_load_dataset(seed)
    # INCREASED TO 4000 to have enough samples for 500 per class/env
    n_subset = 4000 
    return texts[:n_subset], labels[:n_subset]

# Apply patch
data_nlp.load_sms_spam_dataset = mocked_load_dataset
# ------------------------------------------

def build_envs_nlp_random_selection(
    n_per_env: int,
    n_control_envs: int,
    seed: int,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> list[tuple[str, Env]]:
    """
    Build control environments with random selection (no bias).
    """
    print("Chargement du dataset SMS Spam (CONTROL - RANDOM)...")
    all_texts, all_labels = load_sms_spam_dataset(seed=seed)
    
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(all_texts))
    rng.shuffle(all_indices)
    
    envs = []
    
    # Create N random environments
    samples_per_split = len(all_texts) // (n_control_envs + 1) # +1 for test
    
    for i in range(n_control_envs + 1):
        start = i * samples_per_split
        end = start + samples_per_split
        indices = all_indices[start:end]
        
        # Sub-sample to get exactly n_per_env total if needed, or just take all
        # Optimization: Only embed n_per_env samples to save time!
        if len(indices) > n_per_env:
            indices = indices[:n_per_env]
            
        batch_texts = [all_texts[j] for j in indices]
        batch_labels = [all_labels[j] for j in indices]
        
        env_name = f"Control_Env_{i}" if i < n_control_envs else "Control_Test"
        print(f"  Embedding {len(batch_texts)} texts for {env_name}...")
        # Embed
        X = tokenize_and_embed_with_bert(batch_texts, bert_model, max_length, device, pooling)
        Y = np.array(batch_labels).reshape(-1, 1).astype(np.float32)
        
        env_name = f"Control_Env_{i}" if i < n_control_envs else "Control_Test"
        env = Env(torch.from_numpy(X), torch.from_numpy(Y), meta={"kind": "random_control"})
        envs.append((env_name, env))
        
    return envs


def compute_centroids_and_distance(X, Y):
    """
    Computes centroids for class 0 and 1, and the distance between them.
    X: (N, D) numpy array
    Y: (N, 1) numpy array
    """
    mask_0 = (Y.flatten() == 0)
    mask_1 = (Y.flatten() == 1)
    
    if not np.any(mask_0) or not np.any(mask_1):
        return None, None, 0.0, 0.0
    
    centroid_0 = np.mean(X[mask_0], axis=0)
    centroid_1 = np.mean(X[mask_1], axis=0)
    
    # Euclidean distance
    euclidean_dist = np.linalg.norm(centroid_0 - centroid_1)
    
    # Cosine similarity (1 = identical direction, 0 = orthogonal, -1 = opposite)
    cos_sim = cosine_similarity(centroid_0.reshape(1, -1), centroid_1.reshape(1, -1))[0][0]
    
    return centroid_0, centroid_1, euclidean_dist, cos_sim

def run_analysis(mode="size", n_samples=500, seed=42):
    print(f"\nrunning RUN ANALYSIS with mode={mode}...")
    
    output_dir = f"embedding_plots_{mode}"
    os.makedirs(output_dir, exist_ok=True)
    
    all_envs = []
    
    if mode == "size":
        # Generate biased environments
        train_envs, val_envs, test_env = build_envs_nlp_size_selection(
            train_p_select=[1.0, 1.0],
            seed=seed,
            threshold_method="quartile",
            val_frac=0.1,
            max_length=128
        )
        for i, env in enumerate(train_envs):
            all_envs.append((f"Train_Env_{i}", env))
        all_envs.append(("Test_OOD", test_env))
        
    elif mode == "random":
        # Generate random control environments
        all_envs = build_envs_nlp_random_selection(
            n_per_env=n_samples,
            n_control_envs=2,
            seed=seed
        )
        
    # --- PRE-CALCULATION: Average Train Separation Vector ---
    train_separations = []
    
    # Identify training environments (exclude Test/OOD)
    train_names = [name for name, _ in all_envs if "Test" not in name]
    
    for name, env in all_envs:
        if name in train_names:
            X = env.X.numpy()
            Y = env.y.numpy()
            c0, c1, _, _ = compute_centroids_and_distance(X, Y)
            if c0 is not None and c1 is not None:
                train_separations.append(c1 - c0) 
            
    avg_train_sep = np.mean(train_separations, axis=0) if train_separations else None


    # Prepare results file
    results_path = os.path.join(output_dir, "results.txt")
    results_file = open(results_path, "w")
    
    header = f"{'Environment':<20} | {'Euc Dist':<12} | {'Cos Sim':<12} | {'Alignment':<12} | {'Samples':<10}"
    print("\n" + "="*95)
    print(f"MODE: {mode.upper()}")
    print(header)
    print("="*95)
    results_file.write(f"MODE: {mode.upper()}\n")
    results_file.write(header + "\n" + "-"*95 + "\n")

    # Store data for plotting
    plot_data = []

    for name, env in all_envs:
        X = env.X.numpy()
        Y = env.y.numpy()
        
        # Sub-sample if requested to have comparable sizes
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X = X[indices]
            Y = Y[indices]
        
        # 1. Compute Distances
        c0, c1, dist, sim = compute_centroids_and_distance(X, Y)
        
        # 2. Compute Alignment (vs Average Train Vector)
        alignment_str = "N/A"
        alignment_val = None
        
        if avg_train_sep is not None and c0 is not None and c1 is not None:
            current_sep = c1 - c0
            alignment_val = cosine_similarity(avg_train_sep.reshape(1, -1), current_sep.reshape(1, -1))[0][0]
            alignment_str = f"{alignment_val:.4f}"
        
        # Print and write
        line = f"{name:<20} | {dist:<12.4f} | {sim:<12.4f} | {alignment_str:<12} | {len(X):<10}"
        print(line)
        results_file.write(line + "\n")
        
        # Collect data for plotting
        if len(X) > 5:
            plot_data.append({
                "name": name,
                "X": X,
                "Y": Y,
                "c0": c0,
                "c1": c1,
                "dist": dist
            })
        
    print("="*95)
    results_file.close()

    # 3. Visualization: 3 plots side-by-side
    if len(plot_data) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Determine strict display name for mode
        mode_display = "Natural" if mode == "random" else "Size Bias"

        for i, data in enumerate(plot_data):
            if i >= 3: break # Safety if more than 3 envs
            
            ax = axes[i]
            X = data["X"]
            Y = data["Y"]
            c0 = data["c0"]
            c1 = data["c1"]
            dist = data["dist"]
            name = data["name"]

            # Compute PCA locally for each plot
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            # Scatter Plot
            ax.scatter(X_pca[Y.flatten()==0, 0], X_pca[Y.flatten()==0, 1], c='blue', alpha=0.5, label='HAM (0)', s=20)
            ax.scatter(X_pca[Y.flatten()==1, 0], X_pca[Y.flatten()==1, 1], c='red', alpha=0.5, label='SPAM (1)', s=20)
            
            if c0 is not None and c1 is not None:
                c_matrix = np.vstack([c0, c1])
                c_pca = pca.transform(c_matrix)
                # Centroids
                ax.scatter(c_pca[0, 0], c_pca[0, 1], c='darkblue', marker='X', s=200, label='Centroid HAM')
                ax.scatter(c_pca[1, 0], c_pca[1, 1], c='darkred', marker='X', s=200, label='Centroid SPAM')
                
                # Dashed line
                ax.plot([c_pca[0, 0], c_pca[1, 0]], [c_pca[0, 1], c_pca[1, 1]], 'k--', linewidth=2, label=f'Separation (Dist: {dist:.2f})')

            # Clean Title
            dataset_type = "TRAIN" if "Train" in name else "TEST (OOD)" if "Test" in name else name
            ax.set_title(f"{dataset_type}\n({mode_display})", fontsize=16, fontweight='bold')
            
            # Move legend to bottom right, empty grid
            ax.legend(loc='lower right', frameon=True, fontsize=10)
            ax.grid(True, alpha=0.15)
            ax.set_xticks([])
            ax.set_yticks([])

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "comparison.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    # Run both experiments
    run_analysis(mode="size", n_samples=500)
    run_analysis(mode="random", n_samples=500)
