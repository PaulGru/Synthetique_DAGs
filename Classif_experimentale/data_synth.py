# Générateurs d'environnements *synthétiques* pour nos expériences IRM :
#   - Covariate shift (pur)
#   - Toy spurious (Y -> C)
#   - Confounding (Z -> {X_s, Y})
#   - Selection / collider (conditionnement sur S=1)


from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional, Dict
import os, csv
import numpy as np
import torch


# =============================================================================
# Conteneur commun d'un environnement
# =============================================================================

@dataclass
class Env:
    """
    Un environnement contenant données, labels et méta-infos.

    Attributes
    ----------
    X : torch.Tensor
        Matrice (N, d) de features. Par convention, d=2 pour les jouets [X_s, C].
    y : torch.Tensor
        Vecteur (N, 1) de labels binaires {0,1} (float32).
    y_true : Optional[torch.Tensor]
        (Optionnel) vérité terrain si on a ajouté du bruit de labels.
    meta : Optional[Dict]
        Dictionnaire libre (kind, paramètres génératifs, split, etc.).
    """
    X: torch.Tensor
    y: torch.Tensor
    y_true: Optional[torch.Tensor] = None
    meta: Optional[Dict] = None


# =============================================================================
# Helpers internes (graine, split, fonctions élémentaires)
# =============================================================================

def _np_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def _split_indices(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retourne (train_idx, val_idx) de tailles ~ (1-val_frac)n et val_frac*n.
    Le split est déterministe via `seed`.
    """
    rng = _np_rng(seed)
    idx = rng.permutation(n)
    k = int(n * val_frac)
    return idx[k:], idx[:k]

def _split_numpy(X: np.ndarray, Y: np.ndarray, val_frac: float, seed: int):
    """Split numpy (X, Y) -> ((X_tr, y_tr), (X_val, y_val))."""
    tr_idx, va_idx = _split_indices(X.shape[0], val_frac, seed)
    return (X[tr_idx], Y[tr_idx]), (X[va_idx], Y[va_idx])


# =============================================================================
# 1) Semi anti-causal : X_z -> Y -> Z -> X_y
# =============================================================================
# Modèle génératif (par environnement e) :
#   1) X_z ~ N(0, 1)                      (feature causale, identique partout)
#   2) Y* = 1{ X_z > 0 }                  (règle causale "propre")
#   3) Flip du label : Y = Y* XOR Bernoulli(label_flip)
#        -> Affaiblit la corrélation causale X_z <-> Y
#   4) Variable binaire de style :
#        Z = Y XOR Bernoulli(p_spur_e)
#        -> Corrélation forte Y <-> Z si p_spur_e << 0.5
#   5) Feature spurieuse continue :
#        X_y = Z + ε_X,  ε_X ~ N(0, sigma_x^2)
#
# Objectif :
#   - corr(Y, Z) > corr(Y, X_z) dans les environnements d'entraînement
#   - en test, on peut augmenter p_spur_e (≈ 0.5 ou > 0.5) pour casser la
#     corrélation spurious, tout en gardant le mécanisme X_z -> Y invariant.


def _generate_semi_anti_causal(
    n: int,
    p_spur: float,
    seed: int,
    label_flip: float = 0.25,
    dim_z: int = 1,
    dim_y: int = 1,
    causal_strength: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère n échantillons du modèle semi anti-causal X_z -> Y -> Z -> X_y.

    Paramètres
    ----------
    n : int
        Nombre d'exemples à générer.
    p_spur : float
        Probabilité de flipper Z après l'avoir copié depuis Y :
          - p_spur = 0.0  -> Z = Y (corrélation maximale)
          - p_spur = 0.5  -> Z ⟂ Y (indépendance)
          - p_spur > 0.5  -> corrélation inversée.
    seed : int
        Graine RNG.
    label_flip : float, optional
        Proba de flip symétrique du label Y :
          - augmente → affaiblit le lien causal X_z -> Y.
          - typiquement 0.25 (comme dans Empirical or Invariant RM).
    dim_z : int, optional
        Dimension de la feature causale X_z (défaut: 1).
    dim_y : int, optional
        Dimension de la feature spurieuse X_y (défaut: 1).
    causal_strength : float, optional
        Facteur multiplicatif pour la variance de X_z.
        Plus grand = meilleure séparation visuelle entre Y=0 et Y=1.
        Défaut: 1.0 (variance standard N(0,1)).

    Renvoie
    -------
    Xc : np.ndarray (n, dim_z + dim_y)
        Features [X_z, X_y].
    Y  : np.ndarray (n, 1), float32 in {0,1}
        Labels (après flip).
    Z  : np.ndarray (n, 1), float32 in {0,1}
        Variable binaire de style (non utilisée comme feature, mais dispo pour analyse).
    """
    # ✅ FIX CRITIQUE: Seed GLOBALE fixe pour w_true et u
    # Garantit que la fonction causale est IDENTIQUE entre tous les environnements
    # C'est l'hypothèse fondamentale d'IRM !
    rng_global = _np_rng(42)  # Seed fixe pour les vecteurs causaux
    rng = _np_rng(seed)       # Seed variable pour le reste (échantillonnage)

    # 1) Vecteur de poids causaux "vrais" : w_true (INVARIANT entre envs)
    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    # 2) Direction spurieuse u (INVARIANTE entre envs)
    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,))).astype(np.float32)
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    # 3) Feature causale : X_z ~ N(0, causal_strength^2 * I_dim_z)
    X_z = rng.normal(0.0, causal_strength, size=(n, dim_z)).astype(np.float32)

    # 4) Label "propre" : Y* = 1{w_true · X_z > 0}
    Y_star = ((X_z @ w_true) > 0).astype(np.float32).reshape(-1, 1)

    # 5) Flip symétrique des labels pour affaiblir le signal causal
    Y = Y_star.copy()
    if label_flip > 0.0:
        mask = rng.uniform(0.0, 1.0, size=(n, 1)) < label_flip
        Y[mask] = 1.0 - Y[mask]

    # 6) Variable de style binaire Z = Y XOR Bernoulli(p_spur)
    Z = Y.copy()
    flips_z = rng.uniform(0.0, 1.0, size=(n, 1)) < p_spur
    Z[flips_z] = 1.0 - Z[flips_z]

    # 7) Feature spurieuse : X_y = u * Z + bruit
    X_y = (Z @ u.reshape(1, -1)) + rng.normal(0.0, 1e-1, size=(n, dim_y)).astype(np.float32)
    # Standardisation de X_y pour avoir une variance ~1 (comme X_z)
    X_y = (X_y - X_y.mean(axis=0)) / (X_y.std(axis=0) + 1e-8)

    # 8) Features finales : [X_z, X_y]
    Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)

    return Xc, Y.astype(np.float32), Z.astype(np.float32), w_true, u


def build_envs_semi_anti_causal(
    n: int,
    train_p_spurs: List[float],
    test_p_spur: float,
    seed: int,
    val_frac: float = 0.2,
    label_flip: float = 0.25,
    n_test: Optional[int] = None,
    dim_z: int = 1,
    dim_y: int = 1,
    causal_strength: float = 1.0,
) -> Tuple[List[Env], List[Env], Env]:
    """
    Construit des environnements semi anti-causaux.

    Paramètres
    ----------
    n : int
        Nombre d'exemples par environnement d'entraînement.
    train_p_spurs : List[float]
        Liste des p_spur_e pour les envs d'entraînement (ex.: [0.1, 0.2]).
        -> fort alignement spurious en train.
    test_p_spur : float
        p_spur_e pour l'env de test (ex.: 0.9 pour corrélation inversée).
    seed : int
        Graine globale.
    val_frac : float, optional
        Fraction de validation dans chaque env d'entraînement.
    label_flip : float, optional
        Proba de flip de label (affecte le signal causal X_z->Y de la même façon
        dans tous les envs).
    n_test : Optional[int], optional
        Nombre d'exemples en test (défaut: n).
    dim_z : int, optional
        Dimension de la feature causale X_z (défaut: 1).
    dim_y : int, optional
        Dimension de la feature spurieuse X_y (défaut: 1).

    Renvoie
    -------
    train_envs : List[Env]
        Env d'entraînement (avec X=[X_z,X_y]).
    val_envs : List[Env]
        Env de validation correspondants.
    test_env : Env
        Env de test OOD (spurious cassé/inversé).
    """
    if n_test is None:
        n_test = n

    train_envs, val_envs = [], []
    for i, p_spur in enumerate(train_p_spurs):
        Xc, Y, Z, w_true, u = make_env_semi_anti_causal(
            n=n,
            p_spur=p_spur,
            seed=seed + i,
            label_flip=label_flip,
            dim_z=dim_z,
            dim_y=dim_y,
            causal_strength=causal_strength,
        )

        (X_tr, y_tr), (X_val, y_val) = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)

        # On splitte Z de la même façon pour avoir le Z aligné avec X_tr
        (Z_tr, _), (_, _) = _split_numpy(Z, Y, val_frac, seed + 1000 + i)

        meta_train = {
            "p_spur": p_spur,
            "label_flip": label_flip,
            "kind": "train",
            "env_id": i,
            "Z": torch.from_numpy(Z_tr),
            "dim_z": dim_z,
            "dim_y": dim_y,
            "w_true": w_true,
            "u": u,
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), meta=meta_train))

        # ======== VALIDATION ========
        Xc_val, Y_val_clean, Z_val, w_true_val, u_val = make_env_semi_anti_causal(
            n=y_val.shape[0],
            p_spur=p_spur,
            seed=seed + 5000 + i,
            label_flip=0.0,
            dim_z=dim_z,
            dim_y=dim_y,
            causal_strength=causal_strength,
        )
        val_envs.append(Env(torch.from_numpy(Xc_val), torch.from_numpy(Y_val_clean),
                            meta={"p_spur": p_spur, "label_flip": 0.0, "kind": "val", "Z": torch.from_numpy(Z_val), "dim_z": dim_z, "dim_y": dim_y, "w_true": w_true_val, "u": u_val}))

    # Environnement de test OOD
    Xc_t, Y_t, Z_t, w_true_t, u_t = make_env_semi_anti_causal(
        n=n_test,
        p_spur=test_p_spur,
        seed=seed + 777,
        label_flip=0.0,
        dim_z=dim_z,
        dim_y=dim_y,
        causal_strength=causal_strength,
    )
    meta_test = {
        "p_spur": test_p_spur,
        "label_flip": 0.0,
        "kind": "test",
        "Z": torch.from_numpy(Z_t),
        "dim_z": dim_z,
        "dim_y": dim_y,
        "w_true": w_true_t,
        "u": u_t,
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), meta=meta_test)

    return train_envs, val_envs, test_env

# =============================================================================
# 3) anticausal avec Gaussian Shift
# =============================================================================

def make_env_gaussian_shift(
    n: int,
    mu_spur: float,     # Le paramètre clé : la "force" et le "sens" de la corrélation
    std_spur: float,    # La variance (bruit autour de la moyenne)
    seed: int,
    label_flip: float = 0.25, # On garde le bruit sur Y pour que la tâche causale ne soit pas triviale
    dim_z: int = 1,
    dim_y: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    rng_global = _np_rng(42)  # Invariant
    rng = _np_rng(seed)       # Variable

    # 1) Vecteurs directeurs invariants
    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,))).astype(np.float32)
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    # 2) Feature Causale (Invariant)
    X_z = rng.normal(0.0, 1.0, size=(n, dim_z)).astype(np.float32)

    # 3) Label Y (basé sur X_z)
    Y_star = ((X_z @ w_true) > 0).astype(np.float32).reshape(-1, 1)
    
    # Flip de label (bruit causal)
    Y = Y_star.copy()
    if label_flip > 0.0:
        mask = rng.uniform(0.0, 1.0, size=(n, 1)) < label_flip
        Y[mask] = 1.0 - Y[mask]

    # =========================================================================
    # 4) Feature Spurieuse (Gaussian Shift)
    # =========================================================================
    
    # On convertit Y {0, 1} en signe {-1, +1}
    Y_sign = (Y * 2.0 - 1.0) 
    
    # Le centre de la gaussienne dépend de l'environnement (mu_spur)
    # Mean = Signe * mu_spur * u
    mean_vec = (Y_sign @ u.reshape(1, -1)) * mu_spur
    
    # Bruit gaussien
    noise = rng.normal(0.0, std_spur, size=(n, dim_y)).astype(np.float32)
    
    X_y = mean_vec + noise
    
    # ⚠️ IMPORTANT : NE PAS STANDARDISER X_y ICI !
    # Si on fait (X - mean) / std, on efface le shift qu'on vient de créer.
    # On veut justement que la moyenne soit différente entre les envs.

    # 5) Concaténation
    Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)

    return Xc, Y.astype(np.float32), w_true, u


def build_envs_gaussian_shift(
    n: int,
    train_mus: List[float],    # Ex: [1.0, 2.0]
    test_mu: float,            # Ex: -1.0 (Inversion)
    std_spur: float,           # Ex: 0.5 (Bruit constant)
    seed: int,
    val_frac: float = 0.2,
    label_flip: float = 0.25,
    n_test: Optional[int] = None,
    dim_z: int = 1,
    dim_y: int = 1,
) -> Tuple[List[Env], List[Env], Env]:

    if n_test is None: n_test = n
    train_envs, val_envs = [], []

    # --- TRAIN ENVS ---
    for i, mu in enumerate(train_mus):
        Xc, Y, w_true, u = make_env_gaussian_shift(
            n=n, mu_spur=mu, std_spur=std_spur, seed=seed + i,
            label_flip=label_flip, dim_z=dim_z, dim_y=dim_y
        )
        
        # Split Train/Val
        (X_tr, y_tr), (X_val_dummy, y_val_dummy) = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)
        
        # Meta-data (utile pour la visu)
        meta = {
            "kind": "gaussian_shift",
            "mu_spur": mu,
            "std_spur": std_spur,
            "env_id": i,
            "w_true": w_true,
            "u": u,
            "dim_z": dim_z
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), meta=meta))
        
        # Val Env (Régénéré proprement sans flip de label si on veut valider la causalité pure, 
        # ou avec flip si on valide la tâche noisy. Ici on garde la cohérence avec le train).
        Xc_v, Y_v, _, _ = make_env_gaussian_shift(
            n=len(y_val_dummy), mu_spur=mu, std_spur=std_spur, seed=seed + 5000 + i,
            label_flip=0.0, dim_z=dim_z, dim_y=dim_y # Pas de flip en validation souvent
        )
        meta_val = meta.copy(); meta_val["split"] = "val"
        val_envs.append(Env(torch.from_numpy(Xc_v), torch.from_numpy(Y_v), meta=meta_val))

    # --- TEST ENV (OOD) ---
    Xc_t, Y_t, w_true_t, u_t = make_env_gaussian_shift(
        n=n_test, mu_spur=test_mu, std_spur=std_spur, seed=seed + 777,
        label_flip=0.0, dim_z=dim_z, dim_y=dim_y
    )
    meta_test = {
        "kind": "gaussian_shift",
        "mu_spur": test_mu,
        "std_spur": std_spur,
        "env_id": "test",
        "w_true": w_true_t,
        "u": u_t,
        "dim_z": dim_z
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), meta=meta_test)

    return train_envs, val_envs, test_env


# =============================================================================
# anti-causal for variance shift
# =============================================================================

def build_envs_variance_shift(
    n: int,
    train_stds: List[float],   # Ce qui varie : le "flou" (Ex: [0.1, 2.0])
    test_std: float,           # Ex: 5.0 (Très bruité) ou très petit
    fixed_mu: float,           # La moyenne reste fixe (Ex: 1.0)
    seed: int,
    val_frac: float = 0.2,
    label_flip: float = 0.25,
    n_test: Optional[int] = None,
    dim_z: int = 1,
    dim_y: int = 1,
) -> Tuple[List[Env], List[Env], Env]:

    if n_test is None: n_test = n
    train_envs, val_envs = [], []

    # --- TRAIN ENVS (Variance Variable) ---
    for i, std in enumerate(train_stds):
        # On utilise la fonction générique gaussian_shift définie précédemment
        # Mais cette fois, mu est fixe, et std change.
        Xc, Y, w_true, u = make_env_gaussian_shift(
            n=n, 
            mu_spur=fixed_mu,      # FIXE (ex: 1.0)
            std_spur=std,          # VARIABLE (ex: 0.1 puis 2.0)
            seed=seed + i,
            label_flip=label_flip, 
            dim_z=dim_z, 
            dim_y=dim_y
        )
        
        (X_tr, y_tr), (X_val_dummy, y_val_dummy) = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)
        
        meta = {
            "kind": "variance_shift",
            "mu_spur": fixed_mu,
            "std_spur": std,       # On loggue la variance
            "env_id": i,
            "w_true": w_true,
            "u": u,
            "dim_z": dim_z
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), meta=meta))
        
        # Validation
        Xc_v, Y_v, _, _ = make_env_gaussian_shift(
            n=len(y_val_dummy), mu_spur=fixed_mu, std_spur=std, seed=seed + 5000 + i,
            label_flip=0.0, dim_z=dim_z, dim_y=dim_y
        )
        meta_val = meta.copy(); meta_val["split"] = "val"
        val_envs.append(Env(torch.from_numpy(Xc_v), torch.from_numpy(Y_v), meta=meta_val))

    # --- TEST ENV (OOD Variance) ---
    Xc_t, Y_t, w_true_t, u_t = make_env_gaussian_shift(
        n=n_test, 
        mu_spur=fixed_mu,     # Toujours la même moyenne
        std_spur=test_std,    # Variance extrême (très petite ou très grande)
        seed=seed + 777,
        label_flip=0.0, 
        dim_z=dim_z, 
        dim_y=dim_y
    )
    meta_test = {
        "kind": "variance_shift",
        "mu_spur": fixed_mu,
        "std_spur": test_std,
        "env_id": "test",
        "w_true": w_true_t,
        "u": u_t,
        "dim_z": dim_z
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), meta=meta_test)

    return train_envs, val_envs, test_env



# =============================================================================
# 2) Confounding
# =============================================================================

def make_env_confounding(
    n: int,
    seed: int,
    a: float,             # intensité du lien C -> Z (varie avec l'env)
    gamma: float = 1.0,   # poids du confondeur C dans Y
    *,
    dim_z: int = 1,
    dim_y: int = 1,
    include_yz: bool = False,  # Active la 3e colonne X^⊥_{Y,Z}
    dim_yz: int = 1,           # Dimension de X^⊥_{Y,Z}
    gamma_yz: float = 0.5,     # Force de l'effet causal de X^⊥_{Y,Z} sur Y
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère un environnement confounded avec Z binaire.

    Graphe causal (cas de base, include_yz=False) :
      C   ~ Ber(0.35)              (confondeur latent)
      Z   = C ⊕ N^e               (proxy spurieux, varie avec l'env)
      X^⊥_Z ~ N(0, I)             (feature CAUSALE PURE, ⟂ C et Z)
      X^⊥_Y = u * Z               (feature SPURIEUSE PURE, parent : Z)
      Y  = sign(w·X^⊥_Z + γ·(2C−1))
      X  = [X^⊥_Z, X^⊥_Y]

    Graphe causal enrichi (include_yz=True) :
      X^⊥_{Y,Z} = α·Z + β·X^⊥_Z + ε   (MIXTE : parents Z ET X^⊥_Z, agit sur Y)
      Y  = sign(w·X^⊥_Z + γ·(2C−1) + γ_yz·v·X^⊥_{Y,Z})
      X  = [X^⊥_Z, X^⊥_Y, X^⊥_{Y,Z}]

    X^⊥_{Y,Z} est à la fois :
      - Corrélé avec Z (corr. spurieuse, change entre envs)
      - Informative sur Y (signal causal via γ_yz)
    """
    rng_global = _np_rng(42)  # Seed fixe pour les vecteurs causaux
    rng = _np_rng(seed)       # Seed variable pour le reste

    # 1) Vecteur de direction spurieuse u (INVARIANT)
    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,)))
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    # 2) Vecteur de poids causaux w_true (INVARIANT)
    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    # 3) Vecteur pour X^⊥_{Y,Z} → Y (INVARIANT)
    v_yz = np.abs(rng_global.normal(0.0, 1.0, size=(dim_yz,)))
    v_yz = v_yz / np.linalg.norm(v_yz) * np.sqrt(dim_yz)

    # 4) Confounder latent C
    C = rng.binomial(1, 0.35, size=(n, 1)).astype(np.float32)

    # 5) Feature causale X^⊥_Z, indépendante de C
    X_z = rng.normal(0.0, 1.0, size=(n, dim_z)).astype(np.float32)

    # 6) Proxy spurieux Z = C XOR N^e
    N_e = rng.binomial(1, a, size=(n, 1))
    Z = np.logical_xor(C.astype(bool), N_e.astype(bool)).astype(np.float32)

    # 7) Feature spurieuse PURE X^⊥_Y = u * Z
    X_y = (Z @ u.reshape(1, -1)).astype(np.float32)
    X_y = (X_y - X_y.mean(axis=0)) / (X_y.std(axis=0) + 1e-8)

    # 8) Label Y (avec ou sans X^⊥_{Y,Z})
    gamma_scaled = gamma * np.sqrt(dim_z)
    logit = (X_z @ w_true).reshape(-1, 1) + gamma_scaled * (2.0 * C - 1.0)

    if include_yz:
        # Feature MIXTE X^⊥_{Y,Z} : parents = Z ET X^⊥_Z
        # alpha contrôle le canal direct Z → X^⊥_{Y,Z}
        # beta contrôle le canal causal X^⊥_Z → X^⊥_{Y,Z}
        alpha_yz = 1.0   # force du lien Z    → X^⊥_{Y,Z}
        beta_yz  = 1.0   # force du lien X^⊥_Z → X^⊥_{Y,Z}
        eps_yz = rng.normal(0.0, 0.1, size=(n, dim_yz)).astype(np.float32)
        # Projection de X_z dans la dim_yz dimensions via v_yz
        X_z_proj = (X_z @ w_true).reshape(-1, 1) * np.ones((1, dim_yz))  # scalaire * dim_yz
        X_yz = alpha_yz * (Z @ np.ones((1, dim_yz))) + beta_yz * X_z_proj + eps_yz
        X_yz = (X_yz - X_yz.mean(axis=0)) / (X_yz.std(axis=0) + 1e-8)

        # X^⊥_{Y,Z} contribue causalement à Y via gamma_yz
        gamma_yz_scaled = gamma_yz * np.sqrt(dim_yz)
        logit = logit + gamma_yz_scaled * (X_yz @ v_yz).reshape(-1, 1)

        Y = (logit > 0.0).astype(np.float32)
        # X = [X^⊥_Z, X^⊥_Y, X^⊥_{Y,Z}]
        Xc = np.concatenate([X_z, X_y, X_yz], axis=1).astype(np.float32)
    else:
        Y = (logit > 0.0).astype(np.float32)
        # X = [X^⊥_Z, X^⊥_Y]
        Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)

    return Xc, Y.astype(np.float32), Z.astype(np.float32), C, w_true, u


def build_envs_confounding(
    n: int,
    a_train: List[float],        # liste des a_e (beta^e) pour les environnements de TRAIN
    a_test: float,               # a_e (beta^e) pour l'environnement de TEST OOD
    gamma: float = 1.0,
    seed: int = 1,
    val_frac: float = 0.2,
    n_test: Optional[int] = None,
    *,
    dim_z: int = 1,
    dim_y: int = 1,
    include_yz: bool = False,
    dim_yz: int = 1,
    gamma_yz: float = 0.5,
) -> Tuple[List[Env], List[Env], Env]:
    """
    Construit un jeu multi-environnements avec confounder de type CF-CMNIST :

      C   ~ Ber(0.25)                      (confondeur)
      X_z ~ N(0, 1)                        (feature causale, ⟂ C)

      Pour chaque env e :
        N^e ~ Ber(a_e)
        Z   = C XOR N^e
        X_y = (2 Z - 1) + ε_X,  ε_X ~ N(0, 0.5)

      Y_base = sign( w X_z + gamma (2C-1) )
      
      - En TRAIN/VAL/TEST : flip aléatoire avec prob. label_flip (si > 0).
        Note: on met souvent label_flip=0 en test pour évaluer la "vraie" fonction.

      X = [X_z, X_y].

    Variation d'environnements :
      - a_e (paramètre de Ber(a_e) pour N^e) contrôle la force du lien
        C -> Z -> X_y, donc la corrélation spurieuse entre X_y et Y.
      - Le mécanisme causal X_z -> Y_base (w) et la loi de C sont identiques.
    """

    if n_test is None:
        n_test = n

    train_envs, val_envs = [], []

    for i, a_e in enumerate(a_train):
        # ===== TRAIN env i =====
        Xc, Y, Z, _C, w_true, u = make_env_confounding(
            n=n,
            seed=seed + i,
            a=a_e,
            gamma=gamma,
            dim_z=dim_z,
            dim_y=dim_y,
            include_yz=include_yz,
            dim_yz=dim_yz,
            gamma_yz=gamma_yz,
        )

        (X_tr, y_tr), (X_val_dummy, y_val_dummy) = _split_numpy(
            Xc, Y, val_frac, seed + 1000 + i
        )
        (Z_tr, _), (_, _) = _split_numpy(Z, Y, val_frac, seed + 1000 + i)

        n_val = y_val_dummy.shape[0]

        meta_train = {
            "kind": "confounding",
            "a": float(a_e),
            "gamma": float(gamma),
            "split": "train",
            "env_id": i,
            "dim_z": dim_z,
            "dim_y": dim_y,
            "include_yz": include_yz,
            "dim_yz": dim_yz if include_yz else 0,
            "gamma_yz": gamma_yz,
            "w_true": w_true,
            "u": u,
            "Z": torch.from_numpy(Z_tr)
        }
        train_envs.append(
            Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), None, meta_train)
        )

        # ===== VAL env i =====
        X_val, Y_val, Z_val, _C_val, w_true_val, u_val = make_env_confounding(
            n=n_val,
            seed=seed + 5000 + i,
            a=a_e,
            gamma=gamma,
            dim_z=dim_z,
            dim_y=dim_y,
            include_yz=include_yz,
            dim_yz=dim_yz,
            gamma_yz=gamma_yz,
        )
        meta_val = {
            **meta_train,
            "split": "val",
            "Z": torch.from_numpy(Z_val)
        }
        val_envs.append(
            Env(torch.from_numpy(X_val), torch.from_numpy(Y_val), None, meta_val)
        )

    # ===== TEST OOD =====
    Xc_t, Y_t, Z_t, _C_t, w_true_t, u_t = make_env_confounding(
        n=n_test,
        seed=seed + 777,
        a=a_test,
        gamma=gamma,
        dim_z=dim_z,
        dim_y=dim_y,
        include_yz=include_yz,
        dim_yz=dim_yz,
        gamma_yz=gamma_yz,
    )
    meta_t = {
        "kind": "confounding",
        "a": float(a_test),
        "gamma": float(gamma),
        "split": "test_ood",
        "env_id": "test",
        "dim_z": dim_z,
        "dim_y": dim_y,
        "include_yz": include_yz,
        "dim_yz": dim_yz if include_yz else 0,
        "gamma_yz": gamma_yz,
        "w_true": w_true_t,
        "u": u_t,
        "Z": torch.from_numpy(Z_t)
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), None, meta_t)

    return train_envs, val_envs, test_env


# =============================================================================
# 3) Selection bias — Causalité brisée par processus de sélection
# =============================================================================
def make_env_selection(
    n: int,
    alpha: float,      # Probabilité de garder un exemple où Z==Y (créer corrélation spurieuse)
    seed: int,
    *,
    label_flip: float = 0.25,
    keep_if_one: bool = True,
    dim_z: int = 1,
    dim_y: int = 1,
    include_yz: bool = False,  # Active la 3e colonne X^⊥_{Y,Z}
    dim_yz: int = 1,           # Dimension de X^⊥_{Y,Z}
    gamma_yz: float = 0.5,     # Force de l'effet causal de X^⊥_{Y,Z} sur Y
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Biais de sélection.

    Graphe causal (cas de base, include_yz=False) :
      Z              ~ Bernoulli(1/2)           (variable de contexte spurieuse)
      X^⊥_Z          ~ N(0, I_dim_z)           (feature CAUSALE PURE)
      Y*             = sign(w·X^⊥_Z)           (latent)
      Y              = 1{Y*>0}                 (avec flip optionnel)
      X^⊥_Y          = u * Z                   (feature SPURIEUSE PURE)
      Sélection : P(garder) = alpha si Z==Y, 1-alpha sinon
      X = [X^⊥_Z, X^⊥_Y]

    Graphe enrichi (include_yz=True) :
      X^⊥_{Y,Z} = α·Z + β·X^⊥_Z + ε          (MIXTE : parents Z ET X^⊥_Z)
      Y* = sign(w·X^⊥_Z + gamma_yz·v·X^⊥_{Y,Z})  (agit sur Y)
      X = [X^⊥_Z, X^⊥_Y, X^⊥_{Y,Z}]

    Paramètres
    ----------
    alpha : float
        alpha = 0.9 → forte corrélation Z==Y (train)
        alpha = 0.5 → pas de biais
        alpha = 0.1 → corrélation inversée (OOD)
    """
    rng_global = _np_rng(42)
    rng = _np_rng(seed)

    # 1) Vecteur de poids causaux (INVARIANT)
    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    # 2) Direction spurieuse (INVARIANTE)
    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,)))
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    # 3) Vecteur pour X^⊥_{Y,Z} → Y (INVARIANT)
    v_yz = np.abs(rng_global.normal(0.0, 1.0, size=(dim_yz,)))
    v_yz = v_yz / np.linalg.norm(v_yz) * np.sqrt(dim_yz)

    kept_Xz, kept_Xy, kept_Xyz, kept_Y, kept_Z = [], [], [], [], []
    kept, total = 0, 0

    while kept < n:
        B = max(2048, n - kept)

        # --- Population de base ---
        Z = rng.binomial(1, 0.5, size=(B, 1)).astype(np.float32)
        Xz = rng.normal(0, 1.0, size=(B, dim_z)).astype(np.float32)  # X^⊥_Z (causal)
        Xy = Z @ u.reshape(1, -1)                                      # X^⊥_Y (spurieux)

        if include_yz:
            # X^⊥_{Y,Z} : parents Z ET X^⊥_Z
            alpha_yz = 1.0
            beta_yz  = 1.0
            eps_yz = rng.normal(0.0, 0.1, size=(B, dim_yz)).astype(np.float32)
            Xz_proj = (Xz @ w_true).reshape(-1, 1) * np.ones((1, dim_yz))
            Xyz = alpha_yz * (Z @ np.ones((1, dim_yz))) + beta_yz * Xz_proj + eps_yz

            # Y dépend de X^⊥_Z ET de X^⊥_{Y,Z}
            gamma_yz_scaled = gamma_yz * np.sqrt(dim_yz)
            logit = (Xz @ w_true).reshape(-1, 1) + gamma_yz_scaled * (Xyz @ v_yz).reshape(-1, 1)
        else:
            logit = (Xz @ w_true).reshape(-1, 1)
            Xyz = None

        Y = (logit > 0.0).astype(np.float32)

        if label_flip and label_flip > 0.0:
            flips = rng.uniform(size=Y.shape) < label_flip
            Y[flips] = 1.0 - Y[flips]

        # --- Sélection basée sur Z==Y ---
        same = (Z == Y).astype(np.float32)
        prob_keep_if_same = float(alpha)
        prob_keep_if_diff = 1.0 - prob_keep_if_same
        S_p = np.where(same == 1.0, prob_keep_if_same, prob_keep_if_diff)
        S_samples = (rng.uniform(size=S_p.shape) < S_p).astype(np.float32)
        mask = (S_samples == (1.0 if keep_if_one else 0.0)).flatten()

        kept_Xz.append(Xz[mask])
        kept_Xy.append(Xy[mask])
        kept_Y.append(Y[mask])
        kept_Z.append(Z[mask])
        if include_yz:
            kept_Xyz.append(Xyz[mask])

        kept += mask.sum()
        total += B

    Xz_k  = np.concatenate(kept_Xz, axis=0)[:n]
    Xy_k  = np.concatenate(kept_Xy, axis=0)[:n]
    Xy_k  = (Xy_k - Xy_k.mean(axis=0)) / (Xy_k.std(axis=0) + 1e-8)
    Y_k   = np.concatenate(kept_Y,  axis=0)[:n]
    Z_k   = np.concatenate(kept_Z,  axis=0)[:n]

    if include_yz:
        Xyz_k = np.concatenate(kept_Xyz, axis=0)[:n]
        Xyz_k = (Xyz_k - Xyz_k.mean(axis=0)) / (Xyz_k.std(axis=0) + 1e-8)
        Xc = np.concatenate([Xz_k, Xy_k, Xyz_k], axis=1).astype(np.float32)
    else:
        Xc = np.concatenate([Xz_k, Xy_k], axis=1).astype(np.float32)

    sel_rate = kept / total if total > 0 else 0.0
    return Xc, Y_k.astype(np.float32), sel_rate, w_true, u


def build_envs_selection(
    n: int,
    train_alphas: List[float],   # liste des ψ^e pour les envs de train
    test_alpha: float,           # ψ^e pour le test OOD
    seed: int = 1,
    val_frac: float = 0.2,
    n_test: Optional[int] = None,
    label_flip: float = 0.25,
    dim_z: int = 1,
    dim_y: int = 1,
    include_yz: bool = False,
    dim_yz: int = 1,
    gamma_yz: float = 0.5,
) -> Tuple[List[Env], List[Env], Env]:

    if n_test is None:
        n_test = n

    train_envs, val_envs = [], []

    for i, psi in enumerate(train_alphas):
        # ===== TRAIN : flip de label autorisé =====
        Xc, Y, rate, w_true, u = make_env_selection(
            n=n,
            alpha=psi,
            seed=seed + i,
            label_flip=label_flip,
            dim_z=dim_z,
            dim_y=dim_y,
            include_yz=include_yz,
            dim_yz=dim_yz,
            gamma_yz=gamma_yz,
        )
        (X_tr, y_tr), (X_val, y_val) = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)

        meta = {
            "kind": "selection",
            "psi": float(psi),
            "label_flip": float(label_flip),
            "sel_rate": rate,
            "split": "train",
            "dim_z": dim_z,
            "dim_y": dim_y,
            "include_yz": include_yz,
            "dim_yz": dim_yz if include_yz else 0,
            "gamma_yz": gamma_yz,
            "w_true": w_true,
            "u": u,
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), None, meta))

        # ===== VAL : pas de flip de label =====
        Xc_val, Y_val, rate_val, w_true_val, u_val = make_env_selection(
            n=n,
            alpha=psi,
            seed=seed + 5000 + i,
            label_flip=0.0,
            dim_z=dim_z,
            dim_y=dim_y,
            include_yz=include_yz,
            dim_yz=dim_yz,
            gamma_yz=gamma_yz,
        )
        meta_val = {
            "kind": "selection",
            "psi": float(psi),
            "label_flip": 0.0,
            "sel_rate": rate_val,
            "split": "val",
            "dim_z": dim_z,
            "dim_y": dim_y,
            "include_yz": include_yz,
            "dim_yz": dim_yz if include_yz else 0,
            "gamma_yz": gamma_yz,
            "w_true": w_true_val,
            "u": u_val,
        }
        val_envs.append(Env(torch.from_numpy(Xc_val), torch.from_numpy(Y_val), None, meta_val))

    # ===== TEST : pas de flip de label =====
    Xc_t, Y_t, rate_t, w_true_t, u_t = make_env_selection(
        n=n_test,
        alpha=test_alpha,
        seed=seed + 777,
        label_flip=0.0,
        dim_z=dim_z,
        dim_y=dim_y,
        include_yz=include_yz,
        dim_yz=dim_yz,
        gamma_yz=gamma_yz,
    )
    meta_t = {
        "kind": "selection",
        "psi": float(test_alpha),
        "label_flip": 0.0,
        "sel_rate": rate_t,
        "split": "test_ood",
        "dim_z": dim_z,
        "dim_y": dim_y,
        "include_yz": include_yz,
        "dim_yz": dim_yz if include_yz else 0,
        "gamma_yz": gamma_yz,
        "w_true": w_true_t,
        "u": u_t,
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), None, meta_t)

    return train_envs, val_envs, test_env


def make_env_confounding_varying_gamma(
    n: int,
    seed: int,
    gamma: float,         # Coefficient d'influence de C sur Y (varie avec l'env)
    *,
    dim_z: int = 1,
    dim_y: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère un environnement où l'influence du confondeur C sur Y varie (gamma).
    
    C ~ N(0, 1) (Confondeur continu Gaussian)
    Z = 1{C > 0} (Proxy binarisé)
    
    X_z ~ N(0, I)
    X_y = u * Z + noise
    
    Y = sign( w*X_z + gamma*C )
    """
    rng_global = _np_rng(42)
    rng = _np_rng(seed)

    # 1) Vecteurs invariants
    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,)))
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    # 2) Confounder C (Continu Gaussien)
    C = rng.normal(0.0, 1.0, size=(n, 1)).astype(np.float32)

    # 3) Z est la version binarisée de C
    Z = (C > 0).astype(np.float32)

    # 4) Feature causale X_z
    X_z = rng.normal(0.0, 1.0, size=(n, dim_z)).astype(np.float32)
    
    # 5) Feature spurieuse X_y
    # X_y dépend de Z (donc du signe de C). 
    X_y = (Z @ u.reshape(1, -1)).astype(np.float32)
    # Bruit sur X_y
    X_y += rng.normal(0.0, 0.1, size=(n, dim_y)).astype(np.float32)
    # Standardisation
    X_y = (X_y - X_y.mean(axis=0)) / (X_y.std(axis=0) + 1e-8)

    # 6) Label Y
    # Y dépend de X_z (causal) et de C (spurious continu) avec poids gamma
    # On scale gamma par sqrt(dim_z) pour garder le ratio SNR constant vs dimension
    gamma_scaled = gamma * np.sqrt(dim_z)
    
    # C est N(0,1), donc pas besoin de (2C-1)
    logit = (X_z @ w_true).reshape(-1, 1) + gamma_scaled * C
    Y = (logit > 0.0).astype(np.float32)

    # 7) Features
    Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)

    return Xc, Y, Z, C, w_true, u


def build_envs_confounding_varying_gamma(
    n: int,
    train_gammas: List[float],
    test_gamma: float,
    seed: int,
    val_frac: float = 0.2,
    n_test: Optional[int] = None,
    dim_z: int = 1,
    dim_y: int = 1,
) -> Tuple[List[Env], List[Env], Env]:
    
    if n_test is None:
        n_test = n
        
    train_envs, val_envs = [], []
    
    # --- TRAIN ---
    for i, g in enumerate(train_gammas):
        Xc, Y, Z, C, w_true, u = make_env_confounding_varying_gamma(
            n=n, seed=seed+i, gamma=g, dim_z=dim_z, dim_y=dim_y
        )
        
        (X_tr, y_tr), (X_val_dummy, y_val_dummy) = _split_numpy(Xc, Y, val_frac, seed+1000+i)
        (Z_tr, _), (_, _) = _split_numpy(Z, Y, val_frac, seed+1000+i)
        
        meta = {
            "kind": "conf_vary_gamma",
            "gamma": g,
            "env_id": i,
            "dim_z": dim_z,
            "dim_y": dim_y,
            "Z": torch.from_numpy(Z_tr),
            "w_true": w_true, "u": u
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), meta=meta))
        
        # Validation
        Xc_v, Y_v, Z_v, _, _, _ = make_env_confounding_varying_gamma(
            n=len(y_val_dummy), seed=seed+5000+i, gamma=g, dim_z=dim_z, dim_y=dim_y
        )
        meta_val = meta.copy(); meta_val["split"] = "val"; meta_val["Z"] = torch.from_numpy(Z_v)
        val_envs.append(Env(torch.from_numpy(Xc_v), torch.from_numpy(Y_v), meta=meta_val))
        
    # --- TEST ---
    Xc_t, Y_t, Z_t, C_t, w_true_t, u_t = make_env_confounding_varying_gamma(
        n=n_test, seed=seed+999, gamma=test_gamma, dim_z=dim_z, dim_y=dim_y
    )
    meta_test = {
        "kind": "conf_vary_gamma",
        "gamma": test_gamma,
        "env_id": "test",
        "dim_z": dim_z, "dim_y": dim_y,
        "Z": torch.from_numpy(Z_t),
        "w_true": w_true_t, "u": u_t
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), meta=meta_test)
    
    return train_envs, val_envs, test_env


# Wrapper pour compatibilité
def make_env_semi_anti_causal(
    n: int,
    p_spur: float,
    seed: int,
    label_flip: float = 0.25,
    dim_z: int = 1,
    dim_y: int = 1,
    causal_strength: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper pour génération semi anti-causale (retourne aussi w_true et u)."""
    return _generate_semi_anti_causal(n, p_spur, seed, label_flip, dim_z, dim_y, causal_strength)



def make_custom_causal_confounding(
    n: int,
    a: float,              # Probabilité de flip (1 - p_e) entre U et Z
    alpha: float = 1.0,    # Poids du confondeur U sur le label Y
    sigma_x: float = 0.1,  # Bruit sur la feature spurieuse (X_y)
    sigma_y: float = 0.1,  # Bruit sur le logit du label (Y)
    seed: int = 42,
    dim_z: int = 1,        # Dimension causale (fixée à 1 pour l'instant dans ce modèle simplifié)
    dim_y: int = 1         # Dimension spurieuse (fixée à 1 pour l'instant)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    
    rng = _np_rng(seed)
    
    # 1. Le Confondeur U ~ Bernoulli(0.5) [Source de corrélation fallacieuse]
    U = rng.binomial(1, 0.5, size=(n, 1)).astype(np.float32)
    
    # 2. Feature Causale X_z ~ N(0, 1) [Signal stable]
    X_z = rng.normal(0, 1, size=(n, dim_z)).astype(np.float32)
    
    # Poids causaux (Ici on simplifie : w_true = 1.0 si dim_z=1, sinon vecteur unitaire)
    if dim_z == 1:
        w_true = np.array([[1.0]], dtype=np.float32)
    else:
        rng_global = _np_rng(42) # Seed fixe pour w_true
        w_true = rng_global.normal(0, 1, size=(dim_z, 1)).astype(np.float32)
        w_true /= np.linalg.norm(w_true)

    # 3. Label Y binaire (Classification)
    # Y = sign(X_z * w + alpha * (2U - 1) + epsilon)
    epsilon_y = rng.normal(0, sigma_y, size=(n, 1)).astype(np.float32)
    
    # Terme causal
    causal_term = X_z @ w_true
    
    # Terme confondeur (2U - 1 pour avoir {-1, 1})
    # alpha contrôle l'intensité de l'incitation pour le modèle
    confounder_term = alpha * (2.0 * U - 1.0)
    
    logit = causal_term + confounder_term + epsilon_y
    Y = (logit > 0).astype(np.float32)
    
    # 4. Attribut parasite Z (Style spurieux, proxy de U)
    # On crée Z en fonction de U avec un taux d'erreur 'a' (flip)
    # Z = U XOR Bernoulli(a)
    flip_mask = rng.binomial(1, a, size=(n, 1)).astype(bool)
    Z = U.copy()
    Z[flip_mask] = 1.0 - Z[flip_mask]
    
    # 5. Feature non-causale X_y = Z + epsilon_x [Raccourci pour le modèle]
    epsilon_x = rng.normal(0, sigma_x, size=(n, dim_y)).astype(np.float32)
    
    # Si dim_y > 1, on projette Z sur dim_y dimensions (simple répétition + bruit)
    if dim_y > 1:
        X_y = np.tile(Z, (1, dim_y)) + epsilon_x
    else:
        X_y = Z + epsilon_x
    
    # Concaténation des features [Causale (dim_z), Spureuse (dim_y)]
    Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)
    
    w_true_flat = w_true.flatten()
    u_vector = np.ones(dim_y, dtype=np.float32) # U est scalaire ici, projeté implicitement

    return Xc, Y, w_true_flat, u_vector


def build_custom_experiment(
    n: int = 5000, 
    seed: int = 0,
    train_a: List[float] = [0.05, 0.15], # Taux d'erreur U -> Z (corrélation forte)
    test_a: float = 0.90,                # Taux d'erreur inversé (corrélation inversée)
    alpha: float = 0.20,                 # Taux de flip U -> Y (1.0 = bruit total, 0.0 = pas d'effet)
    val_frac: float = 0.2,
    dim_z: int = 1,
    dim_y: int = 1
) -> Tuple[List[Env], List[Env], Env]:
    
    train_envs = []
    val_envs = []
    
    # --- TRAIN ENVS ---
    for i, a in enumerate(train_a):
        Xc, Y, w_true, u = make_custom_causal_confounding(n, a=a, alpha=alpha, seed=seed+i, dim_z=dim_z, dim_y=dim_y)
        
        # Split Train/Val
        (X_tr, y_tr), (X_val_dummy, y_val_dummy) = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)
        
        meta = {
            "kind": "custom_confounding",
            "a": a,
            "env_id": i,
            "dim_z": dim_z,
            "dim_y": dim_y,
            "w_true": w_true,
            "u": u
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), meta=meta))
        
        # Validation Env (Regénéré pour être propre)
        Xc_v, Y_v, _, _ = make_custom_causal_confounding(len(y_val_dummy), a=a, alpha=alpha, seed=seed+5000+i, dim_z=dim_z, dim_y=dim_y)
        meta_val = meta.copy(); meta_val["split"] = "val"
        val_envs.append(Env(torch.from_numpy(Xc_v), torch.from_numpy(Y_v), meta=meta_val))

    # --- TEST OOD ---
    Xc_t, Y_t, w_true_t, u_t = make_custom_causal_confounding(n, a=test_a, alpha=alpha, seed=seed+999, dim_z=dim_z, dim_y=dim_y)
    
    meta_test = {
        "kind": "custom_confounding",
        "a": test_a,
        "alpha": alpha,
        "env_id": "test",
        "dim_z": dim_z,
        "dim_y": dim_y,
        "w_true": w_true_t,
        "u": u_t
    }
    
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), meta=meta_test)
    
    return train_envs, val_envs, test_env