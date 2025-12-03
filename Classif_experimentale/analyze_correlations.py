import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from data_synth import build_envs_confounding, build_envs_semi_anti_causal, build_envs_selection

def analyze_correlations():
    datasets = ['synthetic_semi_anti_causal', 'synthetic_selection', 'synthetic_confounding']
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")

        if dataset_name == 'synthetic_semi_anti_causal':
            # Parameters
            n = 200000
            n_test = 10000
            val_frac = 0.1
            ps_train = [0.1, 0.2]
            p_test_ood = 0.9
            label_flip = 0.25
            seed = 1

            print("Parameters:")
            print(f"  n: {n}")
            print(f"  ps_train: {ps_train}")
            print(f"  p_test_ood: {p_test_ood}")
            print(f"  label_flip: {label_flip}")
            print("-" * 30)

            train_envs, val_envs, test_env = build_envs_semi_anti_causal(
                n=n,
                train_p_spurs=ps_train,
                test_p_spur=p_test_ood,
                seed=seed,
                val_frac=val_frac,
                label_flip=label_flip,
                n_test=n_test,
            )
            env_names = [f"Train (p={p})" for p in ps_train] + [f"Test (p={p_test_ood})"]

        elif dataset_name == 'synthetic_confounding':
            # Parameters
            n = 200000
            n_test = 10000
            val_frac = 0.1
            conf_a_train = [0.01, 0.11]
            conf_a_test = 0.99
            conf_gamma = 1.2
            seed = 1

            print("Parameters:")
            print(f"  n: {n}")
            print(f"  conf_a_train: {conf_a_train}")
            print(f"  conf_a_test: {conf_a_test}")
            print(f"  conf_gamma: {conf_gamma}")
            print("-" * 30)

            train_envs, val_envs, test_env = build_envs_confounding(
                n=n,
                a_train=conf_a_train,
                a_test=conf_a_test,
                gamma=conf_gamma,
                seed=seed,
                val_frac=val_frac,
                n_test=n_test,
            )
            env_names = [f"Train (a={a})" for a in conf_a_train] + [f"Test (a={conf_a_test})"]

        elif dataset_name == 'synthetic_selection':
            # Parameters
            n = 200000
            n_test = 10000
            val_frac = 0.1
            train_alphas = [0.9, 0.8]
            test_alpha = 0.1
            sel_label_flip = 0.25
            seed = 1

            print("Parameters:")
            print(f"  n: {n}")
            print(f"  train_alphas: {train_alphas}")
            print(f"  test_alpha: {test_alpha}")
            print(f"  sel_label_flip: {sel_label_flip}")
            print("-" * 30)

            train_envs, val_envs, test_env = build_envs_selection(
                n=n,
                train_alphas=train_alphas,
                test_alpha=test_alpha,
                seed=seed,
                val_frac=val_frac,
                n_test=n_test,
                label_flip=sel_label_flip,
            )
            env_names = [f"Train (alpha={a})" for a in train_alphas] + [f"Test (alpha={test_alpha})"]

        all_envs = train_envs + [test_env]

        print(f"{'Environment':<20} | {'Corr(X_z, Y)':<12} | {'Corr(X_y, Y)':<12}")
        print("-" * 80)

        for env, name in zip(all_envs, env_names):
            X = env.X.numpy()
            Y = env.y.numpy().ravel()
            
            # Extract dimensions from metadata if available
            dim_z = env.meta.get('dim_z', 1) if hasattr(env, 'meta') and env.meta else 1
            dim_y = env.meta.get('dim_y', 1) if hasattr(env, 'meta') and env.meta else 1
            
            # X_z are the first dim_z columns, X_y are the next dim_y columns
            X_z = X[:, :dim_z]
            X_y = X[:, dim_z:dim_z+dim_y]
            
            # Pearson Correlation
            # For multivariate features, compute correlation with norm
            if dim_z == 1:
                corr_xz_y = np.corrcoef(X_z.ravel(), Y)[0, 1]
            else:
                # Use L2 norm of X_z vector as a scalar representation
                X_z_norm = np.linalg.norm(X_z, axis=1)
                corr_xz_y = np.corrcoef(X_z_norm, Y)[0, 1]
            
            if dim_y == 1:
                corr_xy_y = np.corrcoef(X_y.ravel(), Y)[0, 1]
            else:
                # Use L2 norm of X_y vector as a scalar representation
                X_y_norm = np.linalg.norm(X_y, axis=1)
                corr_xy_y = np.corrcoef(X_y_norm, Y)[0, 1]

            print(f"{name:<20} | {corr_xz_y:<12.4f} | {corr_xy_y:<12.4f}")

if __name__ == "__main__":
    analyze_correlations()
