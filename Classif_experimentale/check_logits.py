
import numpy as np
import joblib
import pandas as pd
from data_nlp import load_sms_spam_dataset, get_base_logit

def analyze_correlations(gammas=[1.5, 1.2, 1.0, 0.8, 0.5], n=3000, seed=42):
    print("Loading data and model...")
    all_texts, _ = load_sms_spam_dataset(seed=seed)
    base_model = joblib.load('base_spam_model.pkl')
    
    # Pre-calcul et Standardisation
    print("Pré-calcul des logits de base pour tout le dataset...")
    all_probs = base_model.predict_proba(all_texts)[:, 1]
    epsilon = 1e-6
    all_probs = np.clip(all_probs, epsilon, 1 - epsilon)
    all_raw_logits = np.log(all_probs / (1 - all_probs))
    
    logit_mean = np.mean(all_raw_logits)
    logit_std = np.std(all_raw_logits)
    print(f"Stats Logits Brut: Mean={logit_mean:.2f}, Std={logit_std:.2f}")
    
    all_base_logits = (all_raw_logits - logit_mean) / logit_std
    
    rng = np.random.default_rng(seed)
    
    results = []
    
    for gamma in gammas:
        print(f"\n--- Gamma = {gamma} ---")
        indices = rng.choice(len(all_texts), n, replace=True)
        # batch_texts = [all_texts[i] for i in indices]
        
        batch_logits = all_base_logits[indices]
        
        # Confounder
        C = rng.choice([-1, 1], size=n)
        
        # New Logits and Labels
        final_logits = batch_logits + gamma * C
        Y_new = (final_logits > 0).astype(int)
        
        # Original Labels (Proxied by base_logits > 0)
        # However, because we standardized, 0 is now the mean logit.
        # But wait, original decision boundary was 0 on raw logit (prob=0.5).
        # We should use the standardized 0 as the new "center"? 
        # Or check if raw_logit > 0?
        # Let's check agreement with ORIGINAL semantics. 
        # Original semantics -> raw_logits > 0.
        
        raw_logits_batch = all_raw_logits[indices]
        Y_orig = (raw_logits_batch > 0).astype(int)
        
        # Correlations (Fraction of agreement)
        agree_C = np.mean(Y_new == (C > 0).astype(int))
        agree_Orig = np.mean(Y_new == Y_orig)
        
        print(f"Standardized Logits: Mean={np.mean(batch_logits):.2f}, Std={np.std(batch_logits):.2f}")
        print(f"Agreement with Confounder C: {agree_C:.1%}")
        print(f"Agreement with Original Text: {agree_Orig:.1%}")
        
        results.append({
            "gamma": gamma,
            "agree_C": agree_C,
            "agree_Orig": agree_Orig,
        })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    analyze_correlations()
