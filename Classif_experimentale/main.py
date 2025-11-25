import argparse
from data_synth import (
    build_envs_confounding,
    build_envs_selection,
    build_envs_semi_anti_causal,
)
from models_training import train_erm, train_irm
from utils_irm import resolve_device
import matplotlib.pyplot as plt
import os

def plot_history(history, title, filename):
    steps = history['step']
    loss = history['loss']
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    test_acc = history['test_acc']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title)

    # Plot Loss
    ax1.plot(steps, loss, label='Train Loss')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss vs Steps')
    ax1.legend()
    ax1.grid(True)

    # Plot Accuracy
    ax2.plot(steps, train_acc, label='Train (ID) Acc')
    ax2.plot(steps, val_acc, label='Val (ID) Acc')
    ax2.plot(steps, test_acc, label='Test (OOD) Acc')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Steps')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Choix dataset
    p.add_argument('--dataset', choices=['synthetic_semi_anti_causal', 'synthetic_confounding', 'synthetic_selection'], default='synthetic_semi_anti_causal')

    # ---- Paramètres communs aux datasets synthétiques ----
    p.add_argument('--n', type=int, default=200000, help='taille par env train pour toy')
    p.add_argument('--n_test', type=int, default=10000, help='taille test pour toy (défaut = n)')
    p.add_argument('--label_flip', type=float, default=0.25, help='bruit de label global pour toy')
    p.add_argument('--val_frac', type=float, default=0.05, help='fraction validation pour toy')

    # ---- Hyperparams semi anti-causal ----
    p.add_argument('--ps_train', type=float, nargs='+', default=[0.2, 0.1])
    p.add_argument('--p_test_ood', type=float, default=0.9)
        
    # ---- Hyperparams Confounding ----
    p.add_argument('--conf_a_train', type=float, nargs='+', default=[3.0, 2.0],
                help="Liste des a_e pour les environnements de train.")
    p.add_argument('--conf_a_test', type=float, default=0.2,
                help="Valeur de a_e pour l'environnement de test OOD.")
    p.add_argument('--conf_w', type=float, default=1.0,
                help="Poids de la feature causale X_z dans Y.")
    p.add_argument('--conf_label_flip', type=float, default=0.25,
                help="Flip de labels (uniquement en train) pour affaiblir le signal causal.")

    # ---- Hyperparams Selection bias (collider) ----
    p.add_argument('--sel_alpha_train', type=float, nargs='+', default=[ 2.0, -2.0 ],
        help="alphas de sélection pour les environnements d'entraînement (ex: 2.0 -2.0)")
    p.add_argument('--sel_alpha_test', type=float, default=-2.5, help='alpha test OOD')
    p.add_argument('--sel_w', type=float, default=1.0,
        help="poids du signal X dans Y : Y = 1{ w*X > 0 } (pas d'epsilon)")
    p.add_argument('--sel_label_flip', type=float, default=0.25,
        help="taux de flips symétriques sur Y (identique dans tous les envs). Remplace sigma_eps.")

    # Entraînement commun
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--eval_every', type=int, default=10)

    # Type de modèle
    p.add_argument('--model_kind', choices=['logreg', 'mlp'], default='mlp')

    # ERM
    p.add_argument('--erm_steps', type=int, default=1000)
    p.add_argument('--erm_lr', type=float, default=5e-5)
    p.add_argument('--erm_batch', type=int, default=512)

    # IRM
    p.add_argument('--irm_steps', type=int, default=1000)
    p.add_argument('--irm_lr', type=float, default=1e-5)
    p.add_argument('--irm_lambda', type=float, default=7500.0)
    p.add_argument('--irm_batch', type=int, default=512)
    
    args = p.parse_args()

    device = resolve_device(args.device)

    if args.dataset == 'synthetic_semi_anti_causal':
        train_envs, val_envs, test_env = build_envs_semi_anti_causal(
            n=args.n,
            train_p_spurs=list(args.ps_train),
            test_p_spur=float(args.p_test_ood),
            seed=args.seed,
            val_frac=float(args.val_frac),
            label_flip=float(args.label_flip),
            n_test=args.n_test,
        )

        # ===== ERM =====
        erm, erm_hist = train_erm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.erm_steps, lr=args.erm_lr,
            batch=args.erm_batch,
            seed=args.seed, device=device,
            eval_every=args.eval_every, 
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=False
        )
        plot_history(erm_hist, f"ERM - {args.dataset}", "plot_erm.png")
        # ===== IRM =====
        irm, irm_hist = train_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.irm_steps, lr=args.irm_lr, irm_lambda=args.irm_lambda,
            batch=args.irm_batch,
            seed=args.seed, device=device,
            eval_every=args.eval_every,  
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=False
        )
        plot_history(irm_hist, f"IRM - {args.dataset}", "plot_irm.png")
    
    elif args.dataset == 'synthetic_confounding':
        train_envs, val_envs, test_env = build_envs_confounding(
            n=args.n,
            a_train=args.conf_a_train,
            a_test=args.conf_a_test,
            w=args.conf_w,
            seed=args.seed,
            val_frac=args.val_frac,
            n_test=args.n_test,
            label_flip=args.conf_label_flip,
        )
        erm, erm_hist = train_erm(
            envs=train_envs,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device,
            eval_every=args.eval_every, val_envs=val_envs, test_env=test_env,
            model_kind=args.model_kind, mlp_hidden=256, mlp_dropout=0.1, mlp_bn=False
        )
        plot_history(erm_hist, f"ERM - {args.dataset}", "plot_erm.png")

        irm, irm_hist = train_irm(
            envs=train_envs,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            irm_lambda=args.irm_lambda,
            seed=args.seed, device=device,
            eval_every=args.eval_every, val_envs=val_envs, test_env=test_env,
            model_kind=args.model_kind, mlp_hidden=256
        )
        plot_history(irm_hist, f"IRM - {args.dataset}", "plot_irm.png")

    elif args.dataset == 'synthetic_selection':
        
        train_alphas = list(map(float, args.sel_alpha_train))
        test_alpha   = float(args.sel_alpha_test)

        train_envs, val_envs, test_env = build_envs_selection(
            n=args.n,
            train_alphas=train_alphas,
            test_alpha=test_alpha,
            w=args.sel_w,
            seed=args.seed,
            val_frac=args.val_frac,
            n_test=args.n_test,
            label_flip=args.sel_label_flip,
        )

        erm, erm_hist = train_erm(
            envs=train_envs,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device,
            eval_every=args.eval_every, val_envs=val_envs, test_env=test_env,
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=False
        )
        plot_history(erm_hist, f"ERM - {args.dataset}", "plot_erm.png")

        irm, irm_hist = train_irm(
            envs=train_envs,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            irm_lambda=args.irm_lambda,
            seed=args.seed, device=device,
            eval_every=args.eval_every, val_envs=val_envs, test_env=test_env,
            model_kind=args.model_kind, mlp_hidden=256
        )
        plot_history(irm_hist, f"IRM - {args.dataset}", "plot_irm.png")