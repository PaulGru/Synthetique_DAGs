
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from data_synth import build_envs_confounding, build_envs_semi_anti_causal

def analyze_correlations():
    # Choose dataset to analyze
    # dataset = 'synthetic_confounding'
    dataset = 'synthetic_semi_anti_causal'

    if dataset == 'synthetic_confounding':
        # Parameters from the user's note
        n = 200000
        n_test = 10000
        val_frac = 0.1
        conf_a_train = [0.1, 0.2]
        conf_a_test = 0.9
        conf_w = 1.5
        conf_gamma = 10.0
        conf_label_flip = 0.25
        seed = 1 # Default seed

        print("Generating data (Confounding) with parameters:")
        print(f"  n: {n}")
        print(f"  conf_a_train: {conf_a_train}")
        print(f"  conf_a_test: {conf_a_test}")
        print(f"  conf_w: {conf_w}")
        print(f"  conf_gamma: {conf_gamma}")
        print(f"  conf_label_flip: {conf_label_flip}")
        print("-" * 30)

        train_envs, val_envs, test_env = build_envs_confounding(
            n=n,
            a_train=conf_a_train,
            a_test=conf_a_test,
            w=conf_w,
            gamma=conf_gamma,
            seed=seed,
            val_frac=val_frac,
            n_test=n_test,
            label_flip=conf_label_flip,
        )
        env_names = [f"Train (a={a})" for a in conf_a_train] + [f"Test (a={conf_a_test})"]

    elif dataset == 'synthetic_semi_anti_causal':
        # Parameters
        n = 200000
        n_test = 10000
        val_frac = 0.1
        ps_train = [0.2, 0.1]
        p_test_ood = 0.9
        label_flip = 0.25
        seed = 1

        print("Generating data (Semi-Anti-Causal) with parameters:")
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

    all_envs = train_envs + [test_env]

    print(f"{'Environment':<20} | {'Corr(Z, Y)':<12} | {'Corr(X_z, Y)':<12} | {'Corr(X_y, Y)':<12} | {'Acc(Z->Y)':<12} | {'Acc(X_z->Y)':<12} | {'Acc(X_y->Y)':<12}")
    print("-" * 110)

    for env, name in zip(all_envs, env_names):
        X = env.X.numpy()
        Y = env.y.numpy().ravel()
        Z = env.meta['Z'].numpy().ravel()
        
        # X_z is the first column of X, X_y is the second
        X_z = X[:, 0]
        X_y = X[:, 1]

        # Pearson Correlation
        corr_z_y = np.corrcoef(Z, Y)[0, 1]
        corr_xz_y = np.corrcoef(X_z, Y)[0, 1]
        corr_xy_y = np.corrcoef(X_y, Y)[0, 1]

        # Predictive Power (Logistic Regression)
        # Z -> Y
        clf_z = LogisticRegression(solver='lbfgs')
        clf_z.fit(Z.reshape(-1, 1), Y)
        acc_z = accuracy_score(Y, clf_z.predict(Z.reshape(-1, 1)))

        # X_z -> Y
        clf_xz = LogisticRegression(solver='lbfgs')
        clf_xz.fit(X_z.reshape(-1, 1), Y)
        acc_xz = accuracy_score(Y, clf_xz.predict(X_z.reshape(-1, 1)))

        # X_y -> Y
        clf_xy = LogisticRegression(solver='lbfgs')
        clf_xy.fit(X_y.reshape(-1, 1), Y)
        acc_xy = accuracy_score(Y, clf_xy.predict(X_y.reshape(-1, 1)))

        print(f"{name:<20} | {corr_z_y:<12.4f} | {corr_xz_y:<12.4f} | {corr_xy_y:<12.4f} | {acc_z:<12.4f} | {acc_xz:<12.4f} | {acc_xy:<12.4f}")

if __name__ == "__main__":
    analyze_correlations()
