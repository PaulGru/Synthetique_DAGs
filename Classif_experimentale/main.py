import torch, argparse, os
import numpy as np
from data_synth import build_envs_semi_anti_causal, build_envs_confounding, build_envs_selection, build_envs_confounding_varying_pc
from data_nlp import build_envs_nlp_semi_anti_causal, build_envs_nlp_selection_bias, build_envs_nlp_size_selection
from models_training import train_erm, train_irm, train_ib_irm
from utils_irm import resolve_device, visualize_anti_causal_data, visualize_1d_simple
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def plot_combined_history(erm_hist, irm_hist, filename, ib_irm_hist=None):
    """Combine ERM and IRM accuracy curves on same plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Sous-échantillonner pour avoir un point tous les 100 steps
    def subsample(history, target_interval=100):
        steps = np.array(history['step'])
        val_acc = np.array(history['val_acc'])  # ✅ Val au lieu de Train
        test_acc = np.array(history['test_acc'])
        
        # Trouver les indices correspondant à des multiples de target_interval
        mask = (steps % target_interval == 0) | (steps == steps[-1])
        return steps[mask], val_acc[mask], test_acc[mask]
    
    # ERM - Orange
    erm_steps, erm_val, erm_test = subsample(erm_hist)
    ax.plot(erm_steps, erm_val, '--', color='orange', linewidth=2, 
            label='ERM - Val (ID)', alpha=0.8)
    ax.plot(erm_steps, erm_test, '-', color='orange', linewidth=2.5, 
            label='ERM - Test (OOD)')
    
    # IRM - Bleu
    irm_steps, irm_val, irm_test = subsample(irm_hist)
    ax.plot(irm_steps, irm_val, '--', color='blue', linewidth=2, 
            label='IRM - Val (ID)', alpha=0.8)
    ax.plot(irm_steps, irm_test, '-', color='blue', linewidth=2.5, 
            label='IRM - Test (OOD)')
    
    # IB-IRM - Violet
    if ib_irm_hist is not None:
        ib_steps, ib_val, ib_test = subsample(ib_irm_hist)
        ax.plot(ib_steps, ib_val, '--', color='purple', linewidth=2, 
                label='IB-IRM - Val (ID)', alpha=0.8)
        ax.plot(ib_steps, ib_test, '-', color='purple', linewidth=2.5, 
                label='IB-IRM - Test (OOD)')
    
    ax.set_xlabel('Étapes d\'entraînement', fontsize=12)
    ax.set_ylabel('Précision', fontsize=12)
    ax.set_title('Comparaison ERM vs IRM vs IB-IRM', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot combiné sauvegardé: {filename}")
    plt.close()


def plot_combined_weights(erm_hist, irm_hist, filename, ib_irm_hist=None):
    """Combine ERM and IRM weight evolution on same plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    if not erm_hist.get('w_z') or not erm_hist.get('w_y'):
        print("Pas de poids à afficher (non-logreg).")
        return
    
    # Sous-échantillonner
    def subsample_weights(history, target_interval=100):
        steps = np.array(history['step'])
        w_z = np.array(history['w_z'])
        w_y = np.array(history['w_y'])
        mask = (steps % target_interval == 0) | (steps == steps[-1])
        return steps[mask], w_z[mask], w_y[mask]
    
    # ERM - Orange
    erm_steps, erm_wz, erm_wy = subsample_weights(erm_hist)
    ax.plot(erm_steps, erm_wz, '-', color='orange', linewidth=2.5, 
            label='ERM - Causal', marker='o', markersize=3, markevery=max(1, len(erm_steps)//20))
    ax.plot(erm_steps, erm_wy, '--', color='orange', linewidth=2, 
            label='ERM - Trompeur', alpha=0.7)
    
    # IRM - Bleu
    irm_steps, irm_wz, irm_wy = subsample_weights(irm_hist)
    ax.plot(irm_steps, irm_wz, '-', color='blue', linewidth=2.5, 
            label='IRM - Causal', marker='s', markersize=3, markevery=max(1, len(irm_steps)//20))
    ax.plot(irm_steps, irm_wy, '--', color='blue', linewidth=2, 
            label='IRM - Trompeur', alpha=0.7)
    
    ax.set_xlabel('Étapes d\'entraînement', fontsize=12)
    ax.set_ylabel('Norme des poids', fontsize=12)
    ax.set_title('Évolution des Poids - ERM vs IRM', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot poids combiné sauvegardé: {filename}")
    plt.close()


def plot_combined_loss(erm_hist, irm_hist, filename, ib_irm_hist=None):
    """Combine ERM and IRM loss curves on same plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Sous-échantillonner pour avoir un point tous les 250 steps (loss plus lisse)
    def subsample(history, target_interval=250):
        steps = np.array(history['step'])
        loss = np.array(history['loss'])
        
        # Trouver les indices correspondant à des multiples de target_interval
        mask = (steps % target_interval == 0) | (steps == steps[-1])
        return steps[mask], loss[mask]
    
    # ERM - Orange
    erm_steps, erm_loss = subsample(erm_hist)
    ax.plot(erm_steps, erm_loss, '-', color='orange', linewidth=2.5, 
            label='ERM', alpha=0.9)
    
    # IRM - Bleu
    irm_steps, irm_loss = subsample(irm_hist)
    ax.plot(irm_steps, irm_loss, '-', color='blue', linewidth=2.5, 
            label='IRM', alpha=0.9)

    # IB-IRM - Violet
    if ib_irm_hist is not None:
        ib_steps, ib_loss = subsample(ib_irm_hist)
        ax.plot(ib_steps, ib_loss, '-', color='purple', linewidth=2.5, 
                label='IB-IRM', alpha=0.9)
    
    ax.set_xlabel('Étapes d\'entraînement', fontsize=12)
    ax.set_ylabel('Loss (BCE)', fontsize=12)
    ax.set_title('Évolution de la Loss - ERM vs IRM', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot loss combiné sauvegardé: {filename}")
    plt.close()


def plot_combined_alignment(erm_hist, irm_hist, filename, ib_irm_hist=None):
    """Combine ERM and IRM weight alignment (cosine similarity with w_true) on same plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Sous-échantillonner pour avoir un point tous les 50 steps
    def subsample_alignment(history, target_interval=50):
        if 'align_z' not in history or len(history['align_z']) == 0:
            return np.array([]), np.array([]), np.array([])
        
        steps = np.array(history['step'])
        align_z = np.array(history['align_z'])
        align_y = np.array(history['align_y'])
        
        # Trouver les indices correspondant à des multiples de target_interval
        mask = (steps % target_interval == 0) | (steps == steps[-1])
        return steps[mask], align_z[mask], align_y[mask]
    
    # ERM - Orange
    erm_steps, erm_az, erm_ay = subsample_alignment(erm_hist)
    if len(erm_steps) > 0:
        ax.plot(erm_steps, erm_az, '-', color='orange', linewidth=2.5, 
                label='ERM - Alignement Causal', marker='o', markersize=3, markevery=max(1, len(erm_steps)//20))
        ax.plot(erm_steps, erm_ay, '--', color='orange', linewidth=2, 
                label='ERM - Alignement Trompeur', alpha=0.7)
    
    # IRM - Bleu
    irm_steps, irm_az, irm_ay = subsample_alignment(irm_hist)
    if len(irm_steps) > 0:
        ax.plot(irm_steps, irm_az, '-', color='blue', linewidth=2.5, 
                label='IRM - Alignement Causal', marker='s', markersize=3, markevery=max(1, len(irm_steps)//20))
        ax.plot(irm_steps, irm_ay, '--', color='blue', linewidth=2, 
                label='IRM - Alignement Trompeur', alpha=0.7)
    
    ax.set_xlabel('Étapes d\'entraînement', fontsize=12)
    ax.set_ylabel('Alignement (Cosine Similarity)', fontsize=12)
    ax.set_title('Alignement des Poids avec w_true - ERM vs IRM', fontsize=14, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)  # Ligne de référence à 0
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot alignement combiné sauvegardé: {filename}")
    plt.close()


def plot_combined_distance(erm_hist, irm_hist, filename, ib_irm_hist=None):
    """Combine ERM and IRM weight distance (Euclidean distance between learned and true weights)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Sous-échantillonner pour avoir un point tous les 50 steps
    def subsample_distance(history, target_interval=50):
        if 'dist_z' not in history or len(history['dist_z']) == 0:
            return np.array([]), np.array([]), np.array([])
        
        steps = np.array(history['step'])
        dist_z = np.array(history['dist_z'])
        dist_y = np.array(history['dist_y'])
        
        # Trouver les indices correspondant à des multiples de target_interval
        mask = (steps % target_interval == 0) | (steps == steps[-1])
        return steps[mask], dist_z[mask], dist_y[mask]
    
    # ERM - Orange
    erm_steps, erm_dz, erm_dy = subsample_distance(erm_hist)
    if len(erm_steps) > 0:
        ax.plot(erm_steps, erm_dz, '-', color='orange', linewidth=2.5, 
                label='ERM - Distance Causal', marker='o', markersize=3, markevery=max(1, len(erm_steps)//20))
        ax.plot(erm_steps, erm_dy, '--', color='orange', linewidth=2, 
                label='ERM - Distance Trompeur', alpha=0.7)
    
    # IRM - Bleu
    irm_steps, irm_dz, irm_dy = subsample_distance(irm_hist)
    if len(irm_steps) > 0:
        ax.plot(irm_steps, irm_dz, '-', color='blue', linewidth=2.5, 
                label='IRM - Distance Causal', marker='s', markersize=3, markevery=max(1, len(irm_steps)//20))
        ax.plot(irm_steps, irm_dy, '--', color='blue', linewidth=2, 
                label='IRM - Distance Trompeur', alpha=0.7)
    
    ax.set_xlabel('Étapes d\'entraînement', fontsize=12)
    ax.set_ylabel('Distance Euclidienne (||w - w_true||)', fontsize=12)
    ax.set_title('Distance des Poids par rapport à w_true - ERM vs IRM', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot distance combiné sauvegardé: {filename}")
    plt.close()

def plot_variance_evolution(ib_irm_hist, filename):
    """Affiche l'évolution de Var(Φ) au cours de l'entraînement."""
    if ib_irm_hist is None or 'var_penalty' not in ib_irm_hist:
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    steps = np.array(ib_irm_hist['step'])
    var_penalty = np.array(ib_irm_hist['var_penalty'])
    
    # Subsampling pour lisibilité
    mask = (steps % 100 == 0) | (steps == steps[-1])
    
    ax.plot(steps[mask], var_penalty[mask], '-', color='purple', linewidth=2.5, label='Var(Φ)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Variance Penalty')
    ax.set_title('Évolution de la variance de la représentation (IB)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Plot variance sauvegardé: {filename}")
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Choix dataset
    p.add_argument('--dataset', choices=[
        'synthetic_semi_anti_causal',
        'synthetic_gaussian_shift',
        'synthetic_variance_shift',
        'synthetic_confounding',
        'synthetic_confounding_var_pc', # [NEW]
        'synthetic_selection',
        'nlp_sms_spam',
        'nlp_sms_spam_selection',
        'nlp_sms_spam_size_selection'
    ], default='synthetic_semi_anti_causal')

    # ---- Paramètres communs aux datasets synthétiques ----
    p.add_argument('--n', type=int, default=200000, help='taille par env train pour toy')
    p.add_argument('--n_test', type=int, default=10000, help='taille test pour toy (défaut = n)')
    p.add_argument('--label_flip', type=float, default=0.25, help='bruit de label global pour toy')
    p.add_argument('--val_frac', type=float, default=0.1, help='fraction validation pour toy')
    p.add_argument('--dim_z', type=int, default=350, help='dimension de la feature causale X_z')
    p.add_argument('--dim_y', type=int, default=350, help='dimension de la feature spurieuse X_y')

    # ---- Hyperparams semi anti-causal ----
    p.add_argument('--ps_train', type=float, nargs='+', default=[0.2, 0.1])
    p.add_argument('--p_test_ood', type=float, default=0.9)

    # ---- Hyperparams Gaussian Shift ----
    p.add_argument('--gs_mus_train', type=float, nargs='+', default=[1.0, 2.0])
    p.add_argument('--gs_mu_test', type=float, default=-1.0)
    p.add_argument('--gs_std', type=int, default=0.5)

    # ---- Hyperparams Variance Shift ----
    p.add_argument('--vs_stds_train', type=float, nargs='+', default=[0.1, 1.0])
    p.add_argument('--vs_std_test', type=float, default=3.0)
    p.add_argument('--vs_mu_fixed', type=float, default=1.0)
        
    # ---- Hyperparams Confounding ----
    p.add_argument('--conf_a_train', type=float, nargs='+', default=[3.0, 2.0],
                help="Liste des a_e pour les environnements de train.")
    p.add_argument('--conf_a_test', type=float, default=0.2,
                help="Valeur de a_e pour l'environnement de test OOD.")
    p.add_argument('--conf_gamma', type=float, default=1.2,
                help="Poids de la feature causale X_z dans Y.")
    p.add_argument('--conf_label_flip', type=float, default=0.25,
                help="Flip de labels (uniquement en train) pour affaiblir le signal causal.")
    
    # ---- Hyperparams Confounding Variable Proportion (Var PC) ----
    p.add_argument('--conf_pc_train', type=float, nargs='+', default=[0.1, 0.9],
                help="Liste des p(C) pour les environnements de train (Var PC).")
    p.add_argument('--conf_pc_test', type=float, default=0.5,
                help="p(C) pour l'environnement de test OOD (Var PC).")
    p.add_argument('--gamma_pc', type=float, default=1.0,
                help="Poids de la feature causale X_z dans Y (Var PC).")
    p.add_argument('--conf_a_fixed', type=float, default=0.05,
                help="Valeur de a fixe pour Var PC (bruit C->Z).")

    # ---- Hyperparams Selection bias (collider) ----
    p.add_argument('--sel_alpha_train', type=float, nargs='+', default=[0.9, 0.8],
        help="probabilités de garder si Z==Y pour les envs d'entraînement (ex: 0.9 0.8)")
    p.add_argument('--sel_alpha_test', type=float, default=0.1, help='alpha test OOD (ex: 0.1 pour inverser)')
    p.add_argument('--sel_label_flip', type=float, default=0.25,
        help="taux de flips symétriques sur Y (identique dans tous les envs). Remplace sigma_eps.")
    p.add_argument('--sel_sigma_y', type=float, default=0.3, 
                   help='Noise std for X_y in selection bias')

    # ---- Hyperparams NLP SMS Spam ----
    p.add_argument('--nlp_n_samples', type=int, default=3000,
        help='Nombre d\'exemples par environnement pour NLP')
    p.add_argument('--nlp_p_correct_train', type=float, nargs='+', default=[0.9, 0.8],
        help='Probabilité de corrélation emoji-label pour les envs d\'entraînement (ex: 0.9 0.8 = 90% et 80%)')
    p.add_argument('--nlp_p_correct_test', type=float, default=0.0,
        help='Probabilité de corrélation pour le test OOD (0.0 = inversion totale 100%)')
    p.add_argument('--nlp_label_flip', type=float, default=0.25,
        help='Probabilité de flip symétrique des labels en train (affaiblit signal causal texte→label)')
    p.add_argument('--nlp_selection_p_train', type=float, nargs='+', default=[0.9, 0.8],
        help='Probabilités de sélection pour envs de train (ex: 0.9 0.8 = garder 90% et 80% des typiques)')
    p.add_argument('--nlp_size_threshold_method', type=str, default='quartile',
        choices=['quartile', 'median', 'auto'], help='Méthode de calcul des seuils de taille')
    p.add_argument('--nlp_bert_model', type=str, default='bert-base-uncased',
        help='Modèle BERT à utiliser')
    p.add_argument('--nlp_max_length', type=int, default=128,
        help='Longueur max de séquence pour BERT')
    p.add_argument('--nlp_pooling', type=str, default='mean',
        choices=['mean', 'cls', 'max'], help='Type de pooling pour embeddings BERT')

    # Entraînement commun
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--eval_every', type=int, default=10)

    # Type de modèle
    p.add_argument('--model_kind', choices=['logreg', 'mlp'], default='mlp')

    # ERM
    p.add_argument('--erm_steps', type=int, default=1000)
    p.add_argument('--erm_lr', type=float, default=5e-4)  # ✅ FIX: 20× plus élevé (was 5e-5)
    p.add_argument('--erm_batch', type=int, default=512)

    # IRM
    p.add_argument('--irm_steps', type=int, default=1000)
    p.add_argument('--irm_lr', type=float, default=5e-4)  # ✅ FIX: 50× plus élevé (was 1e-5)
    p.add_argument('--irm_lambda', type=float, default=7500.0)
    p.add_argument('--irm_batch', type=int, default=512)

    # IB-IRM
    p.add_argument('--ib_irm_steps', type=int, default=1000)
    p.add_argument('--ib_irm_lr', type=float, default=5e-4)
    p.add_argument('--ib_irm_lambda', type=float, default=7500.0)
    p.add_argument('--ib_irm_gamma', type=float, default=1e-2, 
                   help='Poids du terme de variance (typiquement 1e-3 à 1e-1)')
    p.add_argument('--ib_irm_batch', type=int, default=512)
    
    args = p.parse_args()

    device = resolve_device(args.device)
    
    # Créer le dossier pour les plots
    plot_dir = "plot"
    os.makedirs(plot_dir, exist_ok=True)

    if args.dataset == 'synthetic_semi_anti_causal':
        train_envs, val_envs, test_env = build_envs_semi_anti_causal(
            n=args.n,
            train_p_spurs=list(args.ps_train),
            test_p_spur=float(args.p_test_ood),
            seed=args.seed,
            val_frac=float(args.val_frac),
            label_flip=float(args.label_flip),
            n_test=args.n_test,
            dim_z=args.dim_z,
            dim_y=args.dim_y,
        )

        # Visualisation des données
        if args.dim_z == 1 and args.dim_y == 1:
            all_visu_envs = train_envs + [test_env]
            env_names = [f"Train {i} (p={e.meta['p_spur']})" for i, e in enumerate(train_envs)] + ["Test (OOD)"]
            
            visualize_1d_simple(all_visu_envs, env_names, filename=os.path.join(plot_dir, "visu_1d.png"))
        else:
            all_visu_envs = train_envs + [test_env]
            env_names = [f"Train Env {i} (p={e.meta['p_spur']})" for i, e in enumerate(train_envs)] + ["Test Env (OOD)"]
    
            visualize_anti_causal_data(all_visu_envs, env_names, filename=os.path.join(plot_dir, "data_distribution.png"))

        # ===== ERM =====
        erm, erm_hist = train_erm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,  
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,  # ✅ FIX: BN activée
            dataset_name=args.dataset
        )
        
        # ===== IRM =====
        irm, irm_hist = train_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,
            irm_lambda=args.irm_lambda, model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )
        
        # ===== IB-IRM =====
        ib_irm, ib_irm_hist = train_ib_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.ib_irm_steps, lr=args.ib_irm_lr, batch=args.ib_irm_batch,
            irm_lambda=args.ib_irm_lambda, ib_gamma=args.ib_irm_gamma,
            seed=args.seed, device=device,
            eval_every=args.eval_every,
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )
        
        # ===== PLOTS COMBINÉS =====
        plot_combined_history(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_accuracy.png"), ib_irm_hist)
        plot_combined_weights(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_weights.png"), ib_irm_hist)
        plot_combined_loss(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_loss.png"), ib_irm_hist)
        plot_combined_alignment(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_alignment.png"), ib_irm_hist)
        plot_combined_distance(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_distance.png"), ib_irm_hist)
        plot_variance_evolution(ib_irm_hist, os.path.join(plot_dir, "variance_evolution.png"))


    elif args.dataset == 'synthetic_gaussian_shift':
        train_envs, val_envs, test_env = build_envs_gaussian_shift(
            n=args.n,
            train_mus=args.gs_mus_train,
            test_mu=args.gs_mu_test,
            std_spur=args.gs_std,
            seed=args.seed,
            val_frac=float(args.val_frac),
            label_flip=float(args.label_flip),
            n_test=args.n_test,
            dim_z=args.dim_z,
            dim_y=args.dim_y,
        )

        # Visualisation des données
        if args.dim_z == 1 and args.dim_y == 1:
            all_visu_envs = train_envs + [test_env]
            env_names = [f"Train {i}" for i, e in enumerate(train_envs)] + ["Test (OOD)"]
            
            visualize_1d_simple(all_visu_envs, env_names, filename=os.path.join(plot_dir, "visu_1d.png"))
        else:
            all_visu_envs = train_envs + [test_env]
            env_names = [f"Train Env {i}" for i, e in enumerate(train_envs)] + ["Test Env (OOD)"]
    
            visualize_anti_causal_data(all_visu_envs, env_names, filename=os.path.join(plot_dir, "data_distribution.png"))

        # ===== ERM =====
        erm, erm_hist = train_erm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,  
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,  # ✅ FIX: BN activée
            dataset_name=args.dataset
        )
        
        # ===== IRM =====
        irm, irm_hist = train_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,
            irm_lambda=args.irm_lambda, model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )
        
        # ===== IB-IRM =====
        ib_irm, ib_irm_hist = train_ib_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.ib_irm_steps, lr=args.ib_irm_lr, batch=args.ib_irm_batch,
            irm_lambda=args.ib_irm_lambda, ib_gamma=args.ib_irm_gamma,
            seed=args.seed, device=device,
            eval_every=args.eval_every,
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )
        
        # ===== PLOTS COMBINÉS =====
        plot_combined_history(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_accuracy.png"), ib_irm_hist)
        plot_combined_weights(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_weights.png"), ib_irm_hist)
        plot_combined_loss(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_loss.png"), ib_irm_hist)
        plot_combined_alignment(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_alignment.png"), ib_irm_hist)
        plot_combined_distance(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_distance.png"), ib_irm_hist)
        plot_variance_evolution(ib_irm_hist, os.path.join(plot_dir, "variance_evolution.png"))
    

    elif args.dataset == 'synthetic_variance_shift':
        train_envs, val_envs, test_env = build_envs_variance_shift(
            n=args.n,
            train_stds=args.vs_stds_train,
            test_std=args.vs_std_test,
            fixed_mu=args.vs_mu_fixed,
            seed=args.seed,
            val_frac=float(args.val_frac),
            label_flip=float(args.label_flip),
            n_test=args.n_test,
            dim_z=args.dim_z,
            dim_y=args.dim_y,
        )

        # Visualisation des données
        if args.dim_z == 1 and args.dim_y == 1:
            all_visu_envs = train_envs + [test_env]
            env_names = [f"Train {i}" for i, e in enumerate(train_envs)] + ["Test (OOD)"]
            
            visualize_1d_simple(all_visu_envs, env_names, filename=os.path.join(plot_dir, "visu_1d.png"))
        else:
            all_visu_envs = train_envs + [test_env]
            env_names = [f"Train Env {i}" for i, e in enumerate(train_envs)] + ["Test Env (OOD)"]
    
            visualize_anti_causal_data(all_visu_envs, env_names, filename=os.path.join(plot_dir, "data_distribution.png"))

        # ===== ERM =====
        erm, erm_hist = train_erm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,  
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,  # ✅ FIX: BN activée
            dataset_name=args.dataset
        )
        
        # ===== IRM =====
        irm, irm_hist = train_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,
            irm_lambda=args.irm_lambda, model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )
        
        # ===== IB-IRM =====
        ib_irm, ib_irm_hist = train_ib_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.ib_irm_steps, lr=args.ib_irm_lr, batch=args.ib_irm_batch,
            irm_lambda=args.ib_irm_lambda, ib_gamma=args.ib_irm_gamma,
            seed=args.seed, device=device,
            eval_every=args.eval_every,
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )
        
        # ===== PLOTS COMBINÉS =====
        plot_combined_history(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_accuracy.png"), ib_irm_hist)
        plot_combined_weights(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_weights.png"), ib_irm_hist)
        plot_combined_loss(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_loss.png"), ib_irm_hist)
        plot_combined_alignment(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_alignment.png"), ib_irm_hist)
        plot_combined_distance(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_distance.png"), ib_irm_hist)
        plot_variance_evolution(ib_irm_hist, os.path.join(plot_dir, "variance_evolution.png"))

    
    elif args.dataset == 'synthetic_confounding':
        train_envs, val_envs, test_env = build_envs_confounding(
            n=args.n,
            a_train=args.conf_a_train,
            a_test=args.conf_a_test,
            gamma=args.conf_gamma,
            seed=args.seed,
            val_frac=args.val_frac,
            n_test=args.n_test,
            dim_z=args.dim_z,
            dim_y=args.dim_y,
        )
        erm, erm_hist = train_erm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,
            model_kind=args.model_kind, mlp_hidden=256, mlp_dropout=0.1, mlp_bn=True,  # ✅ FIX: BN activée
            dataset_name=args.dataset
        )

        irm, irm_hist = train_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            irm_lambda=args.irm_lambda,
            seed=args.seed, device=device,
            eval_every=args.eval_every,
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,  # ✅ FIX: BN activée
            dataset_name=args.dataset
        )
        
        # ===== IB-IRM =====
        ib_irm, ib_irm_hist = train_ib_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.ib_irm_steps, lr=args.ib_irm_lr, batch=args.ib_irm_batch,
            irm_lambda=args.ib_irm_lambda, ib_gamma=args.ib_irm_gamma,
            seed=args.seed, device=device,
            eval_every=args.eval_every,
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )
        
        # ===== PLOTS COMBINÉS =====
        plot_combined_history(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_accuracy.png"), ib_irm_hist)
        plot_combined_weights(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_weights.png"), ib_irm_hist)
        plot_combined_loss(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_loss.png"), ib_irm_hist)
        plot_combined_alignment(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_alignment.png"), ib_irm_hist)
        plot_combined_distance(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_distance.png"), ib_irm_hist)
        plot_variance_evolution(ib_irm_hist, os.path.join(plot_dir, "variance_evolution.png"))

    elif args.dataset == 'synthetic_confounding_var_pc':
        train_envs, val_envs, test_env = build_envs_confounding_varying_pc(
            n=args.n,
            train_p_cs=args.conf_pc_train,
            test_p_c=args.conf_pc_test,
            a=args.conf_a_fixed,
            gamma=args.gamma_pc,
            seed=args.seed,
            val_frac=args.val_frac,
            n_test=args.n_test,
            dim_z=args.dim_z,
            dim_y=args.dim_y,
        )
        
        # Visualisation simplifiée
        if args.dim_z == 1 and args.dim_y == 1:
            all_visu_envs = train_envs + [test_env]
            env_names = [f"Train {i} (pc={e.meta['p_c']})" for i, e in enumerate(train_envs)] + ["Test (OOD)"]
            visualize_1d_simple(all_visu_envs, env_names, filename=os.path.join(plot_dir, "visu_1d.png"))

        erm, erm_hist = train_erm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,
            model_kind=args.model_kind, mlp_hidden=256, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )

        irm, irm_hist = train_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            irm_lambda=args.irm_lambda,
            seed=args.seed, device=device,
            eval_every=args.eval_every,
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )
        
        ib_irm, ib_irm_hist = train_ib_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.ib_irm_steps, lr=args.ib_irm_lr, batch=args.ib_irm_batch,
            irm_lambda=args.ib_irm_lambda, ib_gamma=args.ib_irm_gamma,
            seed=args.seed, device=device,
            eval_every=args.eval_every,
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )
        
        plot_combined_history(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_accuracy.png"), ib_irm_hist)
        plot_combined_weights(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_weights.png"), ib_irm_hist)
        plot_combined_loss(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_loss.png"), ib_irm_hist)
        plot_combined_alignment(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_alignment.png"), ib_irm_hist)
        plot_combined_distance(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_distance.png"), ib_irm_hist)
        plot_variance_evolution(ib_irm_hist, os.path.join(plot_dir, "variance_evolution.png"))

    elif args.dataset == 'synthetic_selection':
        
        train_alphas = list(map(float, args.sel_alpha_train))
        test_alpha   = float(args.sel_alpha_test)

        train_envs, val_envs, test_env = build_envs_selection(
            n=args.n,
            train_alphas=train_alphas,
            test_alpha=test_alpha,
            seed=args.seed,
            val_frac=args.val_frac,
            n_test=args.n_test,
            label_flip=args.sel_label_flip,
            dim_z=args.dim_z,
            dim_y=args.dim_y,
        )

        erm, erm_hist = train_erm(
            envs=train_envs,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device,
            eval_every=args.eval_every, val_envs=val_envs, test_env=test_env,
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,  # ✅ FIX: BN activée
            dataset_name=args.dataset
        )

        irm, irm_hist = train_irm(
            envs=train_envs,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            irm_lambda=args.irm_lambda,
            seed=args.seed, device=device,
            eval_every=args.eval_every, val_envs=val_envs, test_env=test_env,
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,  # ✅ FIX: BN activée
            dataset_name=args.dataset
        )
        
        # ===== IB-IRM =====
        ib_irm, ib_irm_hist = train_ib_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.ib_irm_steps, lr=args.ib_irm_lr, batch=args.ib_irm_batch,
            irm_lambda=args.ib_irm_lambda, ib_gamma=args.ib_irm_gamma,
            seed=args.seed, device=device,
            eval_every=args.eval_every,
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )

        # ===== PLOTS COMBINÉS =====
        plot_combined_history(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_accuracy.png"), ib_irm_hist)
        plot_combined_weights(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_weights.png"), ib_irm_hist)
        plot_combined_loss(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_loss.png"), ib_irm_hist)
        plot_combined_alignment(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_alignment.png"), ib_irm_hist)
        plot_combined_distance(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_distance.png"), ib_irm_hist)
        plot_variance_evolution(ib_irm_hist, os.path.join(plot_dir, "variance_evolution.png"))

    elif args.dataset == 'nlp_sms_spam':
        # Convertir device en string pour BERT
        device_str = str(device) if hasattr(device, '__str__') else 'cpu'
        
        train_envs, val_envs, test_env = build_envs_nlp_semi_anti_causal(
            n=args.nlp_n_samples,
            train_p_correct=args.nlp_p_correct_train,
            test_p_correct=args.nlp_p_correct_test,
            seed=args.seed,
            val_frac=args.val_frac,
            n_test=args.nlp_n_samples,  # Même taille pour test
            label_flip=args.nlp_label_flip,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
        )
        
    
    elif args.dataset == 'nlp_sms_spam_selection':
        # Convertir device en string pour BERT
        device_str = str(device) if hasattr(device, '__str__') else 'cpu'
        
        train_envs, val_envs, test_env = build_envs_nlp_selection_bias(
            train_p_select=args.nlp_selection_p_train,
            seed=args.seed,
            val_frac=args.val_frac,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
        )
    
    elif args.dataset == 'nlp_sms_spam_size_selection':
        # Convertir device en string pour BERT
        device_str = str(device) if hasattr(device, '__str__') else 'cpu'
        
        train_envs, val_envs, test_env = build_envs_nlp_size_selection(
            train_p_select=args.nlp_selection_p_train,
            seed=args.seed,
            threshold_method=args.nlp_size_threshold_method,
            val_frac=args.val_frac,
            bert_model=args.nlp_bert_model,
            max_length=args.nlp_max_length,
            device=device_str,
            pooling=args.nlp_pooling,
        )
        
        # ===== ERM =====
        erm, erm_hist = train_erm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.erm_steps, lr=args.erm_lr, batch=args.erm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )
        
        # ===== IRM =====
        irm, irm_hist = train_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.irm_steps, lr=args.irm_lr, batch=args.irm_batch,
            seed=args.seed, device=device, eval_every=args.eval_every,
            irm_lambda=args.irm_lambda, model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )
        
        # ===== IB-IRM =====
        ib_irm, ib_irm_hist = train_ib_irm(
            envs=train_envs, val_envs=val_envs, test_env=test_env,
            steps=args.ib_irm_steps, lr=args.ib_irm_lr, batch=args.ib_irm_batch,
            irm_lambda=args.ib_irm_lambda, ib_gamma=args.ib_irm_gamma,
            seed=args.seed, device=device,
            eval_every=args.eval_every,
            model_kind=args.model_kind, mlp_hidden=256,
            mlp_layers=1, mlp_dropout=0.1, mlp_bn=True,
            dataset_name=args.dataset
        )
        
        # ===== PLOTS COMBINÉS =====
        plot_combined_history(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_accuracy.png"), ib_irm_hist)
        plot_combined_weights(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_weights.png"), ib_irm_hist)
        plot_combined_loss(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_loss.png"), ib_irm_hist)
        plot_combined_alignment(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_alignment.png"), ib_irm_hist)
        plot_combined_distance(erm_hist, irm_hist, os.path.join(plot_dir, "comparison_distance.png"), ib_irm_hist)
        plot_variance_evolution(ib_irm_hist, os.path.join(plot_dir, "variance_evolution.png"))