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

    Renvoie
    -------
    Xc : np.ndarray (n, 2)
        Features [X_z, X_y].
    Y  : np.ndarray (n, 1), float32 in {0,1}
        Labels (après flip).
    Z  : np.ndarray (n, 1), float32 in {0,1}
        Variable binaire de style (non utilisée comme feature, mais dispo pour analyse).
    """
    rng = _np_rng(seed)

    # 1) Feature causale : X_z ~ N(0,1)
    X_z = rng.normal(0.0, 1.0, size=(n, 1)).astype(np.float32)

    # 2) Label "propre" : Y* = 1{X_z > 0}
    Y_star = (X_z > 0).astype(np.float32)

    # 3) Flip symétrique des labels pour affaiblir le signal causal
    Y = Y_star.copy()
    if label_flip > 0.0:
        mask = rng.uniform(0.0, 1.0, size=(n, 1)) < label_flip
        Y[mask] = 1.0 - Y[mask]

    # 4) Variable de style binaire Z = Y XOR Bernoulli(p_spur)
    Z = Y.copy()
    flips_z = rng.uniform(0.0, 1.0, size=(n, 1)) < p_spur
    Z[flips_z] = 1.0 - Z[flips_z]

    # 5) Feature spurieuse continue X_y = Z
    X_y = Z

    # 6) Features finales : [X_z, X_y]
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
        )

        (X_tr, y_tr), (X_val, y_val) = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)

        meta_train = {
            "p_spur": p_spur,
            "label_flip": label_flip,
            "kind": "train",
            "env_id": i,
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), meta=meta_train))

        # ======== VALIDATION ========
        Xc_val, Y_val_clean, Z_val = make_env_semi_anti_causal(
            n=y_val.shape[0],
            p_spur=p_spur,
            seed=seed + 5000 + i,
            label_flip=0.0,
        )
        val_envs.append(Env(torch.from_numpy(Xc_val), torch.from_numpy(Y_val_clean),
                            meta={"p_spur": p_spur, "label_flip": 0.0, "kind": "val"}))

    # Environnement de test OOD
    Xc_t, Y_t, Z_t = make_env_semi_anti_causal(
        n=n_test,
        p_spur=test_p_spur,
        seed=seed + 777,
        label_flip=0.0,
    )
    meta_test = {
        "p_spur": test_p_spur,
        "label_flip": 0.0,
        "kind": "test",
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
    *,
    label_flip: float = 0.25,  # flip des labels seulement à l'entraînement
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère un environnement confounded avec Z binaire :

      C   ~ Ber(0.25)                      (confondeur)
      X_z ~ N(0, 1)                        (feature causale, ⟂ C)

      Z = C \XOR N^e,   N^e ~ Ber(beta^e), beta^e dans {0.1, 0.2, 0.9}

      X_y = gamma * Z + eps,  eps ~ N(0, sigma_y^2)   (feature spurieuse continue)

      Y   = 1{ w * X_z > 0 } ∈ {0,1}

      Y*  = Y \XOR C en entraînement, pas de C au test ou val

      X   = [X_z, X_y]
    """
    rng = _np_rng(seed)

    # 1) Confounder latent C
    C = rng.binomial(1, 0.25, size=(n, 1)).astype(np.float32)

    # 2) Feature causale X_Z^{⊥}, indépendante de C
    X_z = rng.normal(0.0, 1.0, size=(n, 1)).astype(np.float32)

    # 3) Bruit d'environnement N^e ~ Ber(a) et variable intermédiaire Z = C XOR N^e
    N_e = rng.binomial(1, a, size=(n, 1))  # {0,1}
    Z = np.logical_xor(C.astype(bool), N_e.astype(bool)).astype(np.float32)

    # 4) Feature spurieuse X_Y^{⊥} = gamma * Z + eps, eps ~ N(0, sigma_y^2)
    eps_y = rng.normal(0.0, 0.3, size=(n, 1)).astype(np.float32)
    X_y = (2 * Z + eps_y).astype(np.float32)

    # 5) Label Y = 1{ w * X_z > 0 }
    Y = (w * X_z > 0.0).astype(np.float32)

    # 6) Flip symétrique des labels (train uniquement)
    if label_flip and label_flip > 0.0:
        Y = np.logical_xor(Y.astype(np.int32), C.astype(np.int32)).astype(np.float32)

    # 7) Entrée modèle : X = [X_Z^{⊥}, X_Y^{⊥}]
    Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)

    # On retourne aussi C (confondeur) pour debug/plots
    return Xc, Y.astype(np.float32), C



def build_envs_confounding(
    n: int,
    a_train: List[float],        # liste des a_e (beta^e) pour les environnements de TRAIN
    a_test: float,               # a_e (beta^e) pour l'environnement de TEST OOD
    w: float = 1.0,
    seed: int = 1,
    val_frac: float = 0.2,
    n_test: Optional[int] = None,
    *,
    label_flip: float = 0.25,    # > 0 pour le TRAIN ; en VAL/TEST on mettra 0.0
) -> Tuple[List[Env], List[Env], Env]:
    """
    Construit un jeu multi-environnements avec confounder de type CF-CMNIST :

      C   ~ Ber(0.25)                      (confondeur)
      X_z ~ N(0, 1)                        (feature causale, ⟂ C)

      Pour chaque env e :
        N^e ~ Ber(a_e)
        Z   = C XOR N^e
        X_y = (2 Z - 1) + ε_X,  ε_X ~ N(0, 0.5)

      Y_base = 1{ w X_z > 0 }

      - En TRAIN (label_flip > 0 dans make_env_confounding) :
          Y <- Y_base XOR C, puis flip symétrique avec prob. label_flip.
      - En VAL/TEST (label_flip = 0) :
          Y <- Y_base (pas de confounding supplémentaire par C, pas de flip).

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
        Xc, Y, _C = make_env_confounding(
            n=n,
            seed=seed + i,
            a=a_e,
            w=w,
            label_flip=label_flip,   # TRAIN : confounding + bruit de label
        )

        # On découpe ce jeu en train / val (val sera régénéré sans flip)
        (X_tr, y_tr), (X_val_dummy, y_val_dummy) = _split_numpy(
            Xc, Y, val_frac, seed + 1000 + i
        )
        n_val = y_val_dummy.shape[0]

        meta_train = {
            "kind": "confounding",
            "a": float(a_e),
            "w": float(w),
            "label_flip": float(label_flip),
            "split": "train",
            "env_id": i,
        }
        train_envs.append(
            Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), None, meta_train)
        )

        # ===== VAL env i : même a_e, mais sans confounding supplémentaire ni flip de label =====
        X_val, Y_val, _C_val = make_env_confounding(
            n=n_val,
            seed=seed + 5000 + i,
            a=a_e,
            w=w,
            label_flip=0.0,          # VAL : pas de XOR avec C, pas de flips
        )
        meta_val = {
            **meta_train,
            "label_flip": label_flip,
            "split": "val",
        }
        val_envs.append(
            Env(torch.from_numpy(X_val), torch.from_numpy(Y_val), None, meta_val)
        )

    # ===== TEST OOD =====
    Xc_t, Y_t, _C_t = make_env_confounding(
        n=n_test,
        seed=seed + 777,
        a=a_test,
        w=w,
        label_flip=0.0,              # TEST : pas de XOR avec C, pas de flips
    )
    meta_t = {
        "kind": "confounding",
        "a": float(a_test),
        "w": float(w),
        "label_flip": label_flip,
        "split": "test_ood",
        "env_id": "test",
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), None, meta_t)

    return train_envs, val_envs, test_env



# =============================================================================
# 3) Selection bias — Causalité brisée par processus de sélection
#     Processus génératif :
#       Z              ~ Bernoulli(1/2)                 (variable de contexte)
#       X_Z            ~ N(0, 1)                        (feature causale)
#       ε_Y            ~ N(0, sigma_y^2)
#       Y*             = sign( w * X_Z + ε_Y )          (label latent ∈ {-1,+1})
#       Y              ∈ {0,1} obtenu via Y = 1{Y* > 0}
#       (optionnel) flip(Y) avec prob label_flip        (bruit de label)
#       ε_X            ~ N(0, sigma_x^2)
#       X_Y^{⊥}        = γ * Z + ε_X                    (feature trompeuse)
#
#     Sélection spécifique à l'environnement e :
#       S_e ~ Bernoulli( sigmoid( alpha_e * Z + Y* ) )
#       On ne garde que les points avec S_e = 1 (ou S_e = 0 si keep_if_one=False).
#
#     -> P(Y | X_Z^{⊥}) est invariant dans la population de base,
#        mais la corrélation entre X_Y^{⊥} et Y est induite par le processus de sélection.
# =============================================================================
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def make_env_selection(
    n: int,
    w: float,
    alpha: float,      # joue maintenant le rôle de ψ^e
    seed: int,
    *,
    gamma: float = 1.0,
    label_flip: float = 0.0,
    keep_if_one: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Biais de sélection à la CS-CMNIST.

    Population de base (commune à tous les envs)
    -------------------------------------------
      Z              ~ Bernoulli(1/2)                 (variable de contexte)
      X_Z^{⊥}        ~ N(0, 1)
      ε_Y            ~ N(0, sigma_y^2)
      Y*             = sign( w * X_Z^{⊥} + ε_Y )      (latent ∈ {-1,+1})
      Y              = 1{ Y* > 0 } ∈ {0,1}
      (optionnel)    flip(Y) avec prob label_flip
      ε_X            ~ N(0, sigma_x^2)
      X_Y^{⊥}        = γ * Z + ε_X

    Biais de sélection (spécifique à l'env e)
    ----------------------------------------
      On garde un point avec probabilité :
         - 1 - ψ^e   si Z == Y
         - ψ^e       sinon
      où ψ^e = alpha (dans le code).

      On garde les points sélectionnés (S = 1 si keep_if_one=True, sinon S=0).

    Renvoie
    -------
    Xc : ndarray (n, 2)  avec colonnes [X_Z^{⊥}, X_Y^{⊥}]
    Y  : ndarray (n, 1)  labels {0,1}
    sel_rate : proportion approx. retenue (S=1/0) avant tronquage à n
    """
    rng = _np_rng(seed)

    kept_Xz, kept_Xy, kept_Y = [], [], []
    kept, total = 0, 0

    while kept < n:
        B = max(2048, n - kept)

        # --- Population de base ---
        Z = rng.binomial(1, 0.5, size=(B, 1)).astype(np.float32)  # contexte binaire

        Xz = rng.normal(0, 1.0, size=(B, 1)).astype(np.float32)  # X_Z^{⊥}
        Y_latent = w * Xz

        Y_star_sign = np.where(Y_latent > 0.0, 1.0, -1.0).astype(np.float32)  # ∈ {-1,+1}
        Y = (Y_latent > 0.0).astype(np.float32)                                # ∈ {0,1}

        # Flip symétrique des labels (si demandé)
        if label_flip and label_flip > 0.0:
            flips = rng.uniform(size=Y.shape) < label_flip
            Y[flips] = 1.0 - Y[flips]
            # on recalcule le signe cohérent avec le label observé
            Y_star_sign = 2.0 * Y - 1.0

        eps_X = rng.normal(0.0, 1.0, size=(B, 1)).astype(np.float32)
        Xy = (gamma * Z + eps_X).astype(np.float32)  # X_Y^{⊥}

        # --- Sélection à la CS-CMNIST ---
        # "same" <=> Z == Y
        same = (Z == Y).astype(np.float32)
        # proba de sélection : 1-ψ si même, ψ sinon
        psi = float(alpha)
        S_p = np.where(same == 1.0, 1.0 - psi, psi)
        S = rng.uniform(size=(B, 1)) < S_p

        mask = (S[:, 0] == (1 if keep_if_one else 0))

        kept_Xz.append(Xz[mask])
        kept_Xy.append(Xy[mask])
        kept_Y.append(Y[mask])

        kept += int(mask.sum())
        total += B

    Xz_k = np.concatenate(kept_Xz, axis=0)[:n]
    Xy_k = np.concatenate(kept_Xy, axis=0)[:n]
    Y_k  = np.concatenate(kept_Y,  axis=0)[:n]

    Xc = np.concatenate([Xz_k, Xy_k], axis=1).astype(np.float32)
    sel_rate = float(kept) / float(total)

    return Xc, Y_k.astype(np.float32), sel_rate


def build_envs_selection(
    n: int,
    train_alphas: List[float],   # liste des ψ^e pour les envs de train
    test_alpha: float,           # ψ^e pour le test OOD
    w: float = 1.0,
    gamma: float = 1.0,
    seed: int = 1,
    val_frac: float = 0.2,
    n_test: Optional[int] = None,
    label_flip: float = 0.25,
) -> Tuple[List[Env], List[Env], Env]:

    if n_test is None:
        n_test = n

    train_envs, val_envs = [], []

    for i, psi in enumerate(train_alphas):
        # ===== TRAIN : flip de label autorisé =====
        Xc, Y, rate = make_env_selection(
            n=n,
            w=w,
            alpha=psi,
            seed=seed + i,
            gamma=gamma,
            label_flip=label_flip,
        )
        (X_tr, y_tr), (X_val, y_val) = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)

        meta = {
            "kind": "selection",
            "psi": float(psi),
            "w": float(w),
            "gamma": float(gamma),
            "label_flip": float(label_flip),
            "sel_rate": rate,
            "split": "train",
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), None, meta))

        # ===== VAL : pas de flip de label =====
        Xc_val, Y_val, rate_val = make_env_selection(
            n=n,
            w=w,
            alpha=psi,
            seed=seed + 5000 + i,
            gamma=gamma,
            label_flip=0.0,
        )
        meta_val = {
            "kind": "selection",
            "psi": float(psi),
            "w": float(w),
            "gamma": float(gamma),
            "label_flip": 0.0,
            "sel_rate": rate_val,
            "split": "val",
        }
        val_envs.append(Env(torch.from_numpy(Xc_val), torch.from_numpy(Y_val), None, meta_val))

    # ===== TEST : pas de flip de label =====
    Xc_t, Y_t, rate_t = make_env_selection(
        n=n_test,
        w=w,
        gamma=gamma,
        alpha=test_alpha,
        seed=seed + 777,
        label_flip=0.0,
    )
    meta_t = {
        "kind": "selection",
        "psi": float(test_alpha),
        "w": float(w),
        "gamma": float(gamma),
        "label_flip": 0.0,
        "sel_rate": rate_t,
        "split": "test_ood",
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), None, meta_t)

    return train_envs, val_envs, test_env
