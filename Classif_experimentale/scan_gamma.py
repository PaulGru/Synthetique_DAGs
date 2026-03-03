
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from data_nlp import load_sms_spam_dataset

def scan_gamma_correlations(start=0.0, end=2.0, step=0.1, seed=42):
    print("Loading data and model...")
    all_texts, _ = load_sms_spam_dataset(seed=seed)
    base_model = joblib.load('base_spam_model.pkl')
    
    # Pre-calcul et Standardisation des Logits (X_z proxy)
    print("Pré-calcul des logits de base...")
    all_probs = base_model.predict_proba(all_texts)[:, 1]
    epsilon = 1e-6
    all_probs = np.clip(all_probs, epsilon, 1 - epsilon)
    all_raw_logits = np.log(all_probs / (1 - all_probs))
    
    # Standardisation (Mean=0, Std=1)
    logit_mean = np.mean(all_raw_logits)
    logit_std = np.std(all_raw_logits)
    all_base_logits = (all_raw_logits - logit_mean) / logit_std
    
    # X_z "Original" Labels (basé sur le logit brut > 0)
    # C'est la prédiction que ferait le modèle sans confounder
    Y_z = (all_raw_logits > 0).astype(int)
    
    rng = np.random.default_rng(seed)
    
    # On utilise tout le dataset pour la stat
    n = len(all_texts)
    
    # Confounder C (X_y proxy)
    C = rng.choice([-1, 1], size=n)
    
    results = []
    gammas = np.arange(start, end + step/2, step) # Include end
    
    print(f"\nScanning Gamma from {start} to {end} (step {step})...")
    print(f"{'Gamma':<10} | {'Corr(Y, X_y (Token))':<25} | {'Corr(Y, X_z (Semantics))':<25}")
    print("-" * 65)
    
    for gamma in gammas:
        # Calcul du Label Y
        final_logits = all_base_logits + gamma * C
        Y_new = (final_logits > 0).astype(int)
        
        # Corrélations (Agreement %)
        # Corr(Y, C) : A quel point le token injecté (X_y) prédit Y ?
        # Dans notre construction, X_y est un proxy parfait de C (token injecté à 100%).
        agree_Xy = np.mean(Y_new == (C > 0).astype(int))
        
        # Corr(Y, X_z) : A quel point la sémantique originale prédit Y ?
        agree_Xz = np.mean(Y_new == Y_z)
        
        print(f"{gamma:<10.2f} | {agree_Xy:<25.1%} | {agree_Xz:<25.1%}")
        
        results.append({
            "gamma": gamma,
            "correlation_Xy": agree_Xy,
            "correlation_Xz": agree_Xz
        })
        
    df = pd.DataFrame(results)
    df.to_csv("gamma_correlations.csv", index=False)
    print("\nResults saved to gamma_correlations.csv")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df['gamma'], df['correlation_Xy'], label='Correlation(Y, Token X_y)', marker='o')
    plt.plot(df['gamma'], df['correlation_Xz'], label='Correlation(Y, Semantic X_z)', marker='x')
    plt.xlabel('Gamma (Confounder Strength)')
    plt.ylabel('Agreement with Label Y (%)')
    plt.title('Influence of Confounder Strength on Feature Correlations')
    plt.legend()
    plt.grid(True)
    plt.savefig('gamma_correlations_plot.png')
    print("Plot saved to gamma_correlations_plot.png")

    return df

if __name__ == "__main__":
    scan_gamma_correlations()
