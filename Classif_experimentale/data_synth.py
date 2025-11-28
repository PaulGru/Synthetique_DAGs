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


def make_env_semi_anti_causal(
    n: int,
    p_spur: float,
    seed: int,
    label_flip: float = 0.25,
    dim_z: int = 1,
    dim_y: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Jouet semi anti-causal : X_z -> Y -> Z -> X_y, avec Z binaire.

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

    Renvoie
    -------
    Xc : np.ndarray (n, dim_z + dim_y)
        Features [X_z, X_y].
    Y  : np.ndarray (n, 1), float32 in {0,1}
        Labels (après flip).
    Z  : np.ndarray (n, 1), float32 in {0,1}
        Variable binaire de style (non utilisée comme feature, mais dispo pour analyse).
    """
    rng = _np_rng(seed)

    # 1) Feature causale : X_z ~ N(0, I_dim_z)
    X_z = rng.normal(0.0, 1.0, size=(n, dim_z)).astype(np.float32)

    # 2) Vecteur de poids causaux "vrais" : w_true positifs
    #    Pour garantir rétrocompatibilité: tous les poids sont positifs
    #    Normalisé pour que la magnitude soit comparable quelle que soit dim_z
    w_true = np.abs(rng.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    # 3) Label "propre" : Y* = 1{w_true · X_z > 0}
    Y_star = ((X_z @ w_true) > 0).astype(np.float32).reshape(-1, 1)

    # 4) Flip symétrique des labels pour affaiblir le signal causal
    Y = Y_star.copy()
    if label_flip > 0.0:
        mask = rng.uniform(0.0, 1.0, size=(n, 1)) < label_flip
        Y[mask] = 1.0 - Y[mask]

    # 5) Variable de style binaire Z = Y XOR Bernoulli(p_spur)
    Z = Y.copy()
    flips_z = rng.uniform(0.0, 1.0, size=(n, 1)) < p_spur
    Z[flips_z] = 1.0 - Z[flips_z]

    # 6) Feature spurieuse : X_y = u * Z + bruit, où u ~ N(0, 1)
    #    u est un vecteur de "direction spurieuse"
    #    Pour garantir rétrocompatibilité: tous les poids sont positifs (corrélation positive avec Z)
    u = np.abs(rng.normal(0.0, 1.0, size=(dim_y,))).astype(np.float32)
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)
    
    # X_y = outer(Z, u) + noise
    X_y = (Z @ u.reshape(1, -1)) + rng.normal(0.0, 0.1, size=(n, dim_y)).astype(np.float32)

    # 7) Features finales : [X_z, X_y]
    Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)

    return Xc, Y.astype(np.float32), Z.astype(np.float32)


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
        Xc, Y, Z = make_env_semi_anti_causal(
            n=n,
            p_spur=p_spur,
            seed=seed + i,
            label_flip=label_flip,
            dim_z=dim_z,
            dim_y=dim_y,
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
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), meta=meta_train))

        # ======== VALIDATION ========
        Xc_val, Y_val_clean, Z_val = make_env_semi_anti_causal(
            n=y_val.shape[0],
            p_spur=p_spur,
            seed=seed + 5000 + i,
            label_flip=0.0,
            dim_z=dim_z,
            dim_y=dim_y,
        )
        val_envs.append(Env(torch.from_numpy(Xc_val), torch.from_numpy(Y_val_clean),
                            meta={"p_spur": p_spur, "label_flip": 0.0, "kind": "val", "Z": torch.from_numpy(Z_val), "dim_z": dim_z, "dim_y": dim_y}))

    # Environnement de test OOD
    Xc_t, Y_t, Z_t = make_env_semi_anti_causal(
        n=n_test,
        p_spur=test_p_spur,
        seed=seed + 777,
        label_flip=0.0,
        dim_z=dim_z,
        dim_y=dim_y,
    )
    meta_test = {
        "p_spur": test_p_spur,
        "label_flip": 0.0,
        "kind": "test",
        "Z": torch.from_numpy(Z_t),
        "dim_z": dim_z,
        "dim_y": dim_y,
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
    w: float,             # poids de X_z dans Y
    gamma: float = 2.0,   # poids du confondeur C dans Y
    *,
    dim_z: int = 1,
    dim_y: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère un environnement confounded avec Z binaire :

      C   ~ Ber(0.25)                      (confondeur)
      X_z ~ N(0, I_dim_z)                  (feature causale, ⟂ C)

      Z = C \XOR N^e,   N^e ~ Ber(beta^e), beta^e dans {0.1, 0.2, 0.9}

      X_y = u_y * Z + eps,  eps ~ N(0, sigma_y^2)   (feature spurieuse continue)

      Y   = sign( w_true * X_z + gamma * (2C - 1) )
      
      Puis flip aléatoire avec proba label_flip.

      X   = [X_z, X_y]
    """
    rng = _np_rng(seed)

    # 1) Confounder latent C
    C = rng.binomial(1, 0.25, size=(n, 1)).astype(np.float32)

    # 2) Feature causale X_Z^{⊥}, indépendante de C
    X_z = rng.normal(0.0, 1.0, size=(n, dim_z)).astype(np.float32)

    # 3) Bruit d'environnement N^e ~ Ber(a) et variable intermédiaire Z = C XOR N^e
    N_e = rng.binomial(1, a, size=(n, 1))  # {0,1}
    Z = np.logical_xor(C.astype(bool), N_e.astype(bool)).astype(np.float32)

    # 4) Feature spurieuse X_Y^{⊥} = 2 * Z + eps, eps ~ N(0, sigma_y^2)
    u_y = np.abs(rng.normal(0.0, 1.0, size=(dim_y,)))
    u_y = u_y / np.linalg.norm(u_y) * np.sqrt(dim_y)

    eps_y = rng.normal(0.0, 0.3, size=(n, dim_y)).astype(np.float32)
    X_y = (Z @ u_y.reshape(1, -1) + eps_y).astype(np.float32)

    # 5) Label Y = sign( w_true * X_z + gamma * (2C - 1) )
    w_true = np.abs(rng.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * w  # Échelle selon le paramètre w
    
    logit = (X_z @ w_true).reshape(-1, 1) + gamma * (2.0 * C - 1.0)
    Y = (logit > 0.0).astype(np.float32)

    # 7) Entrée modèle : X = [X_Z^{⊥}, X_Y^{⊥}]
    Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)

    # On retourne aussi C (confondeur) pour debug/plots
    return Xc, Y.astype(np.float32), Z.astype(np.float32), C



def build_envs_confounding(
    n: int,
    a_train: List[float],        # liste des a_e (beta^e) pour les environnements de TRAIN
    a_test: float,               # a_e (beta^e) pour l'environnement de TEST OOD
    w: float = 1.0,
    gamma: float = 2.0,
    seed: int = 1,
    val_frac: float = 0.2,
    n_test: Optional[int] = None,
    *,
    dim_z: int = 1,
    dim_y: int = 1,
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
        Xc, Y, Z, _C = make_env_confounding(
            n=n,
            seed=seed + i,
            a=a_e,
            w=w,
            gamma=gamma,
            dim_z=dim_z,
            dim_y=dim_y,
        )

        # On découpe ce jeu en train / val (val sera régénéré sans flip)
        # Note: Z n'est pas splitté ici car on ne l'utilise pas pour l'entraînement standard
        # Mais pour l'analyse, on voudrait le Z correspondant.
        # Pour faire simple, on va stocker le Z complet dans les meta si besoin, 
        # ou mieux : on splitte tout.
        
        (X_tr, y_tr), (X_val_dummy, y_val_dummy) = _split_numpy(
            Xc, Y, val_frac, seed + 1000 + i
        )
        # On splitte Z de la même façon pour avoir le Z aligné avec X_tr
        (Z_tr, _), (_, _) = _split_numpy(Z, Y, val_frac, seed + 1000 + i)

        n_val = y_val_dummy.shape[0]

        meta_train = {
            "kind": "confounding",
            "a": float(a_e),
            "w": float(w),
            "gamma": float(gamma),
            "split": "train",
            "env_id": i,
            "Z": torch.from_numpy(Z_tr) # Stockage de Z pour analyse
        }
        train_envs.append(
            Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), None, meta_train)
        )

        # ===== VAL env i : même a_e, mais sans confounding supplémentaire ni flip de label =====
        X_val, Y_val, Z_val, _C_val = make_env_confounding(
            n=n_val,
            seed=seed + 5000 + i,
            a=a_e,
            w=w,
            gamma=gamma,
            dim_z=dim_z,
            dim_y=dim_y,
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
    Xc_t, Y_t, Z_t, _C_t = make_env_confounding(
        n=n_test,
        seed=seed + 777,
        a=a_test,
        w=w,
        gamma=gamma,
        dim_z=dim_z,
        dim_y=dim_y,
    )
    meta_t = {
        "kind": "confounding",
        "a": float(a_test),
        "w": float(w),
        "gamma": float(gamma),
        "split": "test_ood",
        "env_id": "test",
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
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Biais de sélection à la CS-CMNIST.

    Population de base (commune à tous les envs)
    -------------------------------------------
      Z              ~ Bernoulli(1/2)                 (variable de contexte)
      X_Z^{⊥}        ~ N(0, I_dim_z)
      Y*             = sign( w_true * X_Z^{⊥} )      (latent ∈ {-1,+1})
      Y              = 1{ Y* > 0 } ∈ {0,1}
      (optionnel)    flip(Y) avec prob label_flip
      X_Y^{⊥}        = u * Z

    Biais de sélection (spécifique à l'env e)
    ----------------------------------------
      On garde un point avec probabilité :
         - alpha        si Z == Y (créer corrélation spurieuse)
         - 1 - alpha    si Z ≠ Y
      
      Exemple:
         - alpha = 0.9 → garde 90% des Z==Y, 10% des Z≠Y → forte corrélation
         - alpha = 0.5 → garde 50% partout → pas de biais
         - alpha = 0.1 → garde 10% des Z==Y, 90% des Z≠Y → corrélation inversée (OOD)

      On garde les points sélectionnés (S = 1 si keep_if_one=True).

    Renvoie
    -------
    Xc : ndarray (n, dim_z + dim_y)
    Y  : ndarray (n, 1)  labels {0,1}
    sel_rate : proportion retenue avant tronquage à n
    """
    rng = _np_rng(seed)

    kept_Xz, kept_Xy, kept_Y, kept_Z = [], [], [], []
    kept, total = 0, 0

    while kept < n:
        B = max(2048, n - kept)

        # --- Population de base ---
        Z = rng.binomial(1, 0.5, size=(B, 1)).astype(np.float32)  # contexte binaire

        Xz = rng.normal(0, 1.0, size=(B, dim_z)).astype(np.float32)  # X_Z^{⊥}

        w_true = np.abs(rng.normal(0.0, 1.0, size=(dim_z,)))
        w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)
        Y = ((Xz @ w_true).reshape(-1, 1) > 0.0).astype(np.float32)                                # ∈ {0,1}

        # Flip symétrique des labels (si demandé)
        if label_flip and label_flip > 0.0:
            flips = rng.uniform(size=Y.shape) < label_flip
            Y[flips] = 1.0 - Y[flips]

        u = np.abs(rng.normal(0.0, 1.0, size=(dim_y,)))
        u = u / np.linalg.norm(u) * np.sqrt(dim_y)

        # eps_X = rng.normal(0.0, 0.3, size=(B, dim_y)).astype(np.float32)
        Xy = Z @ u.reshape(1, -1) # + eps_X

        # --- Sélection basée sur Z==Y ---
        # "same" <=> Z == Y
        same = (Z == Y).astype(np.float32)
        
        # alpha = probabilité de garder si Z==Y
        # 1-alpha = probabilité de garder si Z≠Y
        prob_keep_if_same = float(alpha)
        prob_keep_if_diff = 1.0 - prob_keep_if_same
        
        S_p = np.where(same == 1.0, prob_keep_if_same, prob_keep_if_diff)
        
        S_samples = (rng.uniform(size=S_p.shape) < S_p).astype(np.float32)

        mask = (S_samples == (1.0 if keep_if_one else 0.0)).flatten()

        kept_Xz.append(Xz[mask])
        kept_Xy.append(Xy[mask])
        kept_Y.append(Y[mask])
        kept_Z.append(Z[mask])  # Garder Z aussi

        kept += mask.sum()
        total += B

    Xz_k = np.concatenate(kept_Xz, axis=0)[:n]
    Xy_k = np.concatenate(kept_Xy, axis=0)[:n]
    Y_k  = np.concatenate(kept_Y,  axis=0)[:n]
    Z_k  = np.concatenate(kept_Z,  axis=0)[:n]

    Xc = np.concatenate([Xz_k, Xy_k], axis=1).astype(np.float32)
    sel_rate = kept / total if total > 0 else 0.0
    
    # Vérification de la proportion finale Z==Y
    same_final = (Z_k == Y_k).astype(np.float32).mean()
    print(f"[Selection α={alpha:.2f}] Proportion finale Z==Y: {same_final:.2%} (attendu: {alpha:.2%})")

    return Xc, Y_k.astype(np.float32), sel_rate


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
) -> Tuple[List[Env], List[Env], Env]:

    if n_test is None:
        n_test = n

    train_envs, val_envs = [], []

    for i, psi in enumerate(train_alphas):
        # ===== TRAIN : flip de label autorisé =====
        Xc, Y, rate = make_env_selection(
            n=n,
            alpha=psi,
            seed=seed + i,
            label_flip=label_flip,
            dim_z=dim_z,
            dim_y=dim_y,
        )
        (X_tr, y_tr), (X_val, y_val) = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)

        meta = {
            "kind": "selection",
            "psi": float(psi),
            "label_flip": float(label_flip),
            "sel_rate": rate,
            "split": "train",
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), None, meta))

        # ===== VAL : pas de flip de label =====
        Xc_val, Y_val, rate_val = make_env_selection(
            n=n,
            alpha=psi,
            seed=seed + 5000 + i,
            label_flip=0.0,
            dim_z=dim_z,
            dim_y=dim_y,
        )
        meta_val = {
            "kind": "selection",
            "psi": float(psi),
            "label_flip": 0.0,
            "sel_rate": rate_val,
            "split": "val",
        }
        val_envs.append(Env(torch.from_numpy(Xc_val), torch.from_numpy(Y_val), None, meta_val))

    # ===== TEST : pas de flip de label =====
    Xc_t, Y_t, rate_t = make_env_selection(
        n=n_test,
        alpha=test_alpha,
        seed=seed + 777,
        label_flip=0.0,
        dim_z=dim_z,
        dim_y=dim_y,
    )
    meta_t = {
        "kind": "selection",
        "psi": float(test_alpha),
        "label_flip": 0.0,
        "sel_rate": rate_t,
        "split": "test_ood",
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), None, meta_t)

    return train_envs, val_envs, test_env
