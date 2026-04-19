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
    causal_strength: float = 1.0,
    x_shift: float = 0.0,
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

    # 3) Feature causale : X_z ~ N(x_shift * ŵ_true, causal_strength² * I)
    # Le shift est aligné avec w_true (direction discriminante) pour déséquilibrer P(Y*=1).
    w_hat = w_true / (np.linalg.norm(w_true) + 1e-8)  # direction unitaire
    mu = x_shift * w_hat  # (dim_z,)
    X_z = rng.normal(0.0, causal_strength, size=(n, dim_z)).astype(np.float32) + mu.astype(np.float32)

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
    x_shifts_train: Optional[List[float]] = None,
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
    x_shifts_train : Optional[List[float]], optional
        Décalage de X_z le long de w_true par environnement d'entraînement.
        P(Y*=1) ≈ Φ(x_shift / causal_strength) par env.
        Ex. [-1.0, 1.0] → env0 ≈ 16% classe 1, env1 ≈ 84% classe 1.
        Val et test utilisent toujours x_shift=0 (classes équilibrées).
        Défaut: None (=0.0 pour tous les envs).

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
    if x_shifts_train is None:
        x_shifts_train = [0.0] * len(train_p_spurs)

    train_envs, val_envs = [], []
    for i, p_spur in enumerate(train_p_spurs):
        x_shift = x_shifts_train[i]
        Xc, Y, Z, w_true, u = make_env_semi_anti_causal(
            n=n,
            p_spur=p_spur,
            seed=seed + i,
            label_flip=label_flip,
            dim_z=dim_z,
            dim_y=dim_y,
            causal_strength=causal_strength,
            x_shift=x_shift,
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
            "x_shift": x_shift,
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), meta=meta_train))

        # ======== VALIDATION ========
        Xc_val, Y_val_clean, Z_val, w_true_val, u_val = make_env_semi_anti_causal(
            n=y_val.shape[0],
            p_spur=p_spur,
            seed=seed + 5000 + i,
            label_flip=label_flip,
            dim_z=dim_z,
            dim_y=dim_y,
            causal_strength=causal_strength,
            x_shift=0.0,
        )
        val_envs.append(Env(torch.from_numpy(Xc_val), torch.from_numpy(Y_val_clean),
                            meta={"p_spur": p_spur, "label_flip": label_flip, "kind": "val", "Z": torch.from_numpy(Z_val), "dim_z": dim_z, "dim_y": dim_y, "w_true": w_true_val, "u": u_val, "x_shift": 0.0}))

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
# 2) Confounding
# =============================================================================

def make_env_confounding_varying_proxy(
    n: int,
    seed: int,
    a: float,             # intensité du lien C -> Z (varie avec l'env)
    gamma: float = 1.0,   # poids du confondeur C dans Y
    *,
    dim_z: int = 1,
    dim_y: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère un environnement confounded avec Z binaire.

    Graphe causal :
      C   ~ Ber(0.35)              (confondeur latent)
      Z   = C ⊕ N^e               (proxy spurieux, varie avec l'env)
      X^⊥_Z ~ N(0, I)             (feature CAUSALE PURE, ⟂ C et Z)
      X^⊥_Y = u * Z               (feature SPURIEUSE PURE, parent : Z)
      Y  = sign(w·X^⊥_Z + γ·(2C−1))
      X  = [X^⊥_Z, X^⊥_Y]
    """
    rng_global = _np_rng(42)  # Seed fixe pour les vecteurs causaux
    rng = _np_rng(seed)       # Seed variable pour le reste

    # 1) Vecteur de direction spurieuse u (INVARIANT)
    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,)))
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    # 2) Vecteur de poids causaux w_true (INVARIANT)
    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    # 3) Confounder latent C
    C = rng.binomial(1, 0.35, size=(n, 1)).astype(np.float32)

    # 4) Feature causale X^⊥_Z, indépendante de C
    X_z = rng.normal(0.0, 1.0, size=(n, dim_z)).astype(np.float32)

    # 5) Proxy spurieux Z = C XOR N^e
    N_e = rng.binomial(1, a, size=(n, 1))
    Z = np.logical_xor(C.astype(bool), N_e.astype(bool)).astype(np.float32)

    # 6) Feature spurieuse PURE X^⊥_Y = u * Z
    X_y = (Z @ u.reshape(1, -1)).astype(np.float32)
    X_y = (X_y - X_y.mean(axis=0)) / (X_y.std(axis=0) + 1e-8)

    # 7) Label Y
    gamma_scaled = gamma * np.sqrt(dim_z)
    logit = (X_z @ w_true).reshape(-1, 1) + gamma_scaled * (2.0 * C - 1.0)
    Y = (logit > 0.0).astype(np.float32)

    # X = [X^⊥_Z, X^⊥_Y]
    Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)

    return Xc, Y.astype(np.float32), Z.astype(np.float32), C, w_true, u


def build_envs_confounding_varying_proxy(
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
        Xc, Y, Z, _C, w_true, u = make_env_confounding_varying_proxy(
            n=n,
            seed=seed + i,
            a=a_e,
            gamma=gamma,
            dim_z=dim_z,
            dim_y=dim_y,
        )

        (X_tr, y_tr), (X_val_dummy, y_val_dummy) = _split_numpy(
            Xc, Y, val_frac, seed + 1000 + i
        )
        (Z_tr, _), (_, _) = _split_numpy(Z, Y, val_frac, seed + 1000 + i)

        n_val = y_val_dummy.shape[0]

        meta_train = {
            "kind": "confounding_varying_proxy",
            "a": float(a_e),
            "gamma": float(gamma),
            "split": "train",
            "env_id": i,
            "dim_z": dim_z,
            "dim_y": dim_y,
            "w_true": w_true,
            "u": u,
            "Z": torch.from_numpy(Z_tr)
        }
        train_envs.append(
            Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), None, meta_train)
        )

        # ===== VAL env i =====
        X_val, Y_val, Z_val, _C_val, w_true_val, u_val = make_env_confounding_varying_proxy(
            n=n_val,
            seed=seed + 5000 + i,
            a=a_e,
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
    Xc_t, Y_t, Z_t, _C_t, w_true_t, u_t = make_env_confounding_varying_proxy(
        n=n_test,
        seed=seed + 777,
        a=a_test,
        gamma=gamma,
        dim_z=dim_z,
        dim_y=dim_y,
    )
    meta_t = {
        "kind": "confounding_varying_proxy",
        "a": float(a_test),
        "gamma": float(gamma),
        "split": "test_ood",
        "env_id": "test",
        "dim_z": dim_z,
        "dim_y": dim_y,
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
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Biais de sélection.

    Graphe causal :
      Z              ~ Bernoulli(1/2)           (variable de contexte spurieuse)
      X^⊥_Z          ~ N(0, I_dim_z)           (feature CAUSALE PURE)
      Y*             = sign(w·X^⊥_Z)           (latent)
      Y              = 1{Y*>0}                 (avec flip optionnel)
      X^⊥_Y          = u * Z                   (feature SPURIEUSE PURE)
      Sélection : P(garder) = alpha si Z==Y, 1-alpha sinon
      X = [X^⊥_Z, X^⊥_Y]

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

    kept_Xz, kept_Xy, kept_Y, kept_Z = [], [], [], []
    kept, total = 0, 0

    while kept < n:
        B = max(2048, n - kept)

        # --- Population de base ---
        Z = rng.binomial(1, 0.5, size=(B, 1)).astype(np.float32)
        Xz = rng.normal(0, 1.0, size=(B, dim_z)).astype(np.float32)  # X^⊥_Z (causal)
        Xy = Z @ u.reshape(1, -1)                                      # X^⊥_Y (spurieux)

        logit = (Xz @ w_true).reshape(-1, 1)

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

        kept += mask.sum()
        total += B

    Xz_k  = np.concatenate(kept_Xz, axis=0)[:n]
    Xy_k  = np.concatenate(kept_Xy, axis=0)[:n]
    Xy_k  = (Xy_k - Xy_k.mean(axis=0)) / (Xy_k.std(axis=0) + 1e-8)
    Y_k   = np.concatenate(kept_Y,  axis=0)[:n]
    Z_k   = np.concatenate(kept_Z,  axis=0)[:n]

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
            "w_true": w_true,
            "u": u,
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), None, meta))

        # ===== VAL : même label flip que l'entraînement =====
        Xc_val, Y_val, rate_val, w_true_val, u_val = make_env_selection(
            n=n,
            alpha=psi,
            seed=seed + 5000 + i,
            label_flip=label_flip,
            dim_z=dim_z,
            dim_y=dim_y,
        )
        meta_val = {
            "kind": "selection",
            "psi": float(psi),
            "label_flip": label_flip,
            "sel_rate": rate_val,
            "split": "val",
            "dim_z": dim_z,
            "dim_y": dim_y,
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
    )
    meta_t = {
        "kind": "selection",
        "psi": float(test_alpha),
        "label_flip": 0.0,
        "sel_rate": rate_t,
        "split": "test_ood",
        "dim_z": dim_z,
        "dim_y": dim_y,
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


# =============================================================================
# 5) Confounding avec prévalence variable — C ~ Ber(p_e), p_e varie par env
# =============================================================================
def make_env_confounding_varying_pc(
    n: int,
    seed: int,
    p_c: float,           # Prévalence du confondeur C (varie avec l'env)
    a: float = 0.0,       # Force du lien C -> Z (fixe à travers les envs)
    gamma: float = 1.0,   # Poids du confondeur sur Y (fixe)
    *,
    dim_z: int = 1,
    dim_y: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère un environnement confounded où la PRÉVALENCE de C varie entre envs.

    Graphe causal :
      C   ~ Ber(p_e)               (confondeur latent — p_e VARIE avec l'env)
      Z   = C ⊕ Ber(a)             (proxy de C — lien fixe)
      X^⊥_Z ~ N(0, I)             (feature CAUSALE PURE, ⟂ C et Z)
      X^⊥_Y = u * Z               (feature SPURIEUSE PURE)
      Y  = sign(w·X^⊥_Z + γ·(2C−1))
      X  = [X^⊥_Z, X^⊥_Y]

    Violation d'invariance :
      P(Y|X_Z) dépend de p_e via la marginalisation sur C :
        P(Y=1|X_Z=x) = p_e · 1{w·x+γ>0} + (1-p_e) · 1{w·x-γ>0}
      La violation est subtile quand γ << ||w·X_Z|| et forte quand γ >> ||w·X_Z||.
    """
    rng_global = _np_rng(42)
    rng = _np_rng(seed)

    # 1) Vecteurs invariants
    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,)))
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    # 2) Confounder C ~ Ber(p_e)
    C = rng.binomial(1, p_c, size=(n, 1)).astype(np.float32)

    # 3) Feature causale X_Z (⟂ C)
    X_z = rng.normal(0.0, 1.0, size=(n, dim_z)).astype(np.float32)

    # 4) Proxy spurieux Z = C XOR Ber(a)
    N_e = rng.binomial(1, a, size=(n, 1))
    Z = np.logical_xor(C.astype(bool), N_e.astype(bool)).astype(np.float32)

    # 5) Feature spurieuse X_Y = u * Z
    X_y = (Z @ u.reshape(1, -1)).astype(np.float32)
    X_y = (X_y - X_y.mean(axis=0)) / (X_y.std(axis=0) + 1e-8)

    # 6) Label Y
    gamma_scaled = gamma * np.sqrt(dim_z)
    logit = (X_z @ w_true).reshape(-1, 1) + gamma_scaled * (2.0 * C - 1.0)
    Y = (logit > 0.0).astype(np.float32)

    Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)
    return Xc, Y.astype(np.float32), Z.astype(np.float32), C, w_true, u


def build_envs_confounding_varying_pc(
    n: int,
    pc_train: List[float],       # Prévalences p_e pour les envs de TRAIN
    pc_test: float,              # Prévalence p_e pour le test OOD
    a: float = 0.0,              # Force du lien C -> Z (fixe)
    gamma: float = 1.0,          # Poids du confondeur sur Y (fixe)
    seed: int = 1,
    val_frac: float = 0.2,
    n_test: Optional[int] = None,
    *,
    dim_z: int = 1,
    dim_y: int = 1,
) -> Tuple[List[Env], List[Env], Env]:
    """
    Confounding avec prévalence variable : C ~ Ber(p_e), p_e diffère par environnement.

    Les envs de train ont des prévalences p_e modérées et distinctes (ex: 0.2, 0.4).
    Le test OOD a une prévalence extrême (ex: 0.9) non vue en train.

    Le mécanisme causal (w, gamma) est FIXE ; seule la distribution marginale de C change.
    → Violation d'invariance réelle mais d'intensité contrôlée par gamma.
    """
    if n_test is None:
        n_test = n

    train_envs, val_envs = [], []

    for i, p_e in enumerate(pc_train):
        Xc, Y, Z, _C, w_true, u = make_env_confounding_varying_pc(
            n=n, seed=seed + i, p_c=p_e, a=a, gamma=gamma,
            dim_z=dim_z, dim_y=dim_y,
        )
        (X_tr, y_tr), (_, y_val_dummy) = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)
        (Z_tr, _), (_, _) = _split_numpy(Z, Y, val_frac, seed + 1000 + i)
        n_val = y_val_dummy.shape[0]

        meta_train = {
            "kind": "confounding_varying_pc",
            "p_c": float(p_e),
            "a": float(a),
            "gamma": float(gamma),
            "split": "train",
            "env_id": i,
            "dim_z": dim_z,
            "dim_y": dim_y,
            "w_true": w_true,
            "u": u,
            "Z": torch.from_numpy(Z_tr),
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), None, meta_train))

        # VAL
        X_val, Y_val, Z_val, _C_val, _, _ = make_env_confounding_varying_pc(
            n=n_val, seed=seed + 5000 + i, p_c=p_e, a=a, gamma=gamma,
            dim_z=dim_z, dim_y=dim_y,
        )
        meta_val = {**meta_train, "split": "val", "Z": torch.from_numpy(Z_val)}
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(Y_val), None, meta_val))

    # TEST OOD
    Xc_t, Y_t, Z_t, _C_t, w_true_t, u_t = make_env_confounding_varying_pc(
        n=n_test, seed=seed + 777, p_c=pc_test, a=a, gamma=gamma,
        dim_z=dim_z, dim_y=dim_y,
    )
    meta_t = {
        "kind": "confounding_varying_pc",
        "p_c": float(pc_test),
        "a": float(a),
        "gamma": float(gamma),
        "split": "test_ood",
        "env_id": "test",
        "dim_z": dim_z,
        "dim_y": dim_y,
        "w_true": w_true_t,
        "u": u_t,
        "Z": torch.from_numpy(Z_t),
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), None, meta_t)
    return train_envs, val_envs, test_env


# =============================================================================
# 6) Anti-causal : Y → X_z  (direction inverse du cas causal)
#    + mêmes 4 perturbations trompeuses que ci-dessus
# =============================================================================
#
# Dans le cas ANTI-CAUSAL, Y est échantillonné en premier (prior uniforme),
# puis X_z est généré *depuis* Y : X_z = w_true*(2Y-1)*cs + ε.
# La tâche de classification reste "prédire Y depuis X = [X_z, X_y]".
# La différence conceptuelle est que le mécanisme INVARIANT est P(X_z|Y)
# et non P(Y|X_z).  IRM cherche P(Y|Φ(X)) invariant, ce qui correspond ici
# à inverser ce mécanisme par le théorème de Bayes.  Les 5 perturbations
# trompeuses sont identiques aux cas causaux :
#
#   ac_semi_anti_causal      : Y→X_z, Y→Z→X_y
#   ac_selection             : Y→X_z, sélection sur Z==Y
#   ac_confounding_proxy     : C→Y, C→Z→X_y, Y→X_z
#   ac_confounding_gamma     : idem, gamma varie par env
#   ac_confounding_pc        : idem, P(C=1) varie par env
# =============================================================================

def _generate_anti_causal_semi_anti_causal(
    n: int,
    p_spur: float,
    seed: int,
    label_flip: float = 0.25,
    dim_z: int = 1,
    dim_y: int = 1,
    causal_strength: float = 1.0,
    p_y: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Génère n échantillons du modèle ANTI-CAUSAL semi-anti-causal :

      Y*  ~ Ber(p_y)                  (prior sur le label, p_y varie par env)
      Y   = Y* XOR Ber(label_flip)    (label bruité)
      X_z = w_true*(2Y*-1)*cs + ε    (feature ANTI-CAUSALE : Y → X_z)
      Z   = Y  XOR Ber(p_spur)        (variable spurieuse)
      X_y = u*Z + ε_y                 (feature spurieuse)

    La direction causale est inversée par rapport au cas causal :
    ici c'est Y qui *cause* X_z, pas X_z qui cause Y.

    DAG : Y* → X_z    et    Y → Z → X_y
    """
    rng_global = _np_rng(42)
    rng        = _np_rng(seed)

    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,))).astype(np.float32)
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    # 1) Label vrai Y* (prior potentiellement déséquilibré)
    Y_star = rng.binomial(1, p_y, size=(n, 1)).astype(np.float32)

    # 2) Label bruité (observé à l'entraînement)
    Y = Y_star.copy()
    if label_flip > 0.0:
        mask = rng.uniform(0.0, 1.0, size=(n, 1)) < label_flip
        Y[mask] = 1.0 - Y[mask]

    # 3) Anti-causal : X_z est *généré depuis* Y_star
    shift = causal_strength * (2.0 * Y_star - 1.0) * w_true.reshape(1, -1)  # (n, dim_z)
    X_z   = shift + rng.normal(0.0, 1.0, size=(n, dim_z)).astype(np.float32)

    # 4) Variable spurieuse Z = Y XOR Ber(p_spur)
    Z = Y.copy()
    flips_z = rng.uniform(0.0, 1.0, size=(n, 1)) < p_spur
    Z[flips_z] = 1.0 - Z[flips_z]

    # 5) Feature spurieuse X_y
    X_y = (Z @ u.reshape(1, -1)) + rng.normal(0.0, 1e-1, size=(n, dim_y)).astype(np.float32)
    X_y = (X_y - X_y.mean(axis=0)) / (X_y.std(axis=0) + 1e-8)

    Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)
    return Xc, Y.astype(np.float32), Z.astype(np.float32), w_true, u


def build_envs_anti_causal_semi_anti_causal(
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
    p_y_train: Optional[List[float]] = None,
) -> Tuple[List[Env], List[Env], Env]:
    """Version anti-causale de build_envs_semi_anti_causal.

    p_y_train : prior P(Y*=1) par environnement d'entraînement.
        Si None, utilise 0.5 (équilibré) pour tous les envs.
        Exemple : [0.3, 0.7] → classe 0 majoritaire dans env 0,
                               classe 1 majoritaire dans env 1.
        Le jeu de validation et le test utilisent toujours p_y=0.5.
    """
    if n_test is None:
        n_test = n
    if p_y_train is None:
        p_y_train = [0.5] * len(train_p_spurs)

    train_envs, val_envs = [], []
    for i, p_spur in enumerate(train_p_spurs):
        p_y = p_y_train[i]
        Xc, Y, Z, w_true, u = _generate_anti_causal_semi_anti_causal(
            n=n, p_spur=p_spur, seed=seed + i,
            label_flip=label_flip, dim_z=dim_z, dim_y=dim_y, causal_strength=causal_strength,
            p_y=p_y,
        )
        (X_tr, y_tr), _ = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)
        (Z_tr, _), _    = _split_numpy(Z,  Y, val_frac, seed + 1000 + i)

        meta_train = {
            "kind": "anti_causal_semi_anti_causal", "p_spur": p_spur,
            "label_flip": label_flip, "env_id": i, "split": "train",
            "dim_z": dim_z, "dim_y": dim_y, "w_true": w_true, "u": u,
            "p_y": p_y, "Z": torch.from_numpy(Z_tr),
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), meta=meta_train))

        Xc_v, Y_v, Z_v, w_v, u_v = _generate_anti_causal_semi_anti_causal(
            n=int(n * val_frac), p_spur=p_spur, seed=seed + 5000 + i,
            label_flip=label_flip, dim_z=dim_z, dim_y=dim_y, causal_strength=causal_strength,
            p_y=p_y,
        )
        meta_val = {
            "kind": "anti_causal_semi_anti_causal", "p_spur": p_spur,
            "label_flip": label_flip, "env_id": i, "split": "val",
            "dim_z": dim_z, "dim_y": dim_y, "w_true": w_v, "u": u_v,
            "p_y": p_y, "Z": torch.from_numpy(Z_v),
        }
        val_envs.append(Env(torch.from_numpy(Xc_v), torch.from_numpy(Y_v), meta=meta_val))

    Xc_t, Y_t, Z_t, w_t, u_t = _generate_anti_causal_semi_anti_causal(
        n=n_test, p_spur=test_p_spur, seed=seed + 777,
        label_flip=0.0, dim_z=dim_z, dim_y=dim_y, causal_strength=causal_strength,
        p_y=0.5,
    )
    meta_test = {
        "kind": "anti_causal_semi_anti_causal", "p_spur": test_p_spur,
        "label_flip": 0.0, "env_id": "test", "split": "test",
        "dim_z": dim_z, "dim_y": dim_y, "w_true": w_t, "u": u_t,
        "p_y": 0.5, "Z": torch.from_numpy(Z_t),
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), meta=meta_test)
    return train_envs, val_envs, test_env


# ---------------------------------------------------------------------------
# 7) Anti-causal + confounding varying proxy
# ---------------------------------------------------------------------------
def make_env_anti_causal_confounding_varying_proxy(
    n: int,
    seed: int,
    a: float,
    gamma: float = 1.0,
    *,
    dim_z: int = 1,
    dim_y: int = 1,
    causal_strength: float = 1.0,
    p_c: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Anti-causal + confounding varying proxy.

    DAG :
            C   ~ Ber(p_c)
      Y   = sign(γ_scaled*(2C-1) + ε_Y)   (C → Y)
      X_z = w_true*(2Y-1)*cs + ε_Xz       (Y → X_z, anti-causal)
      Z   = C XOR Ber(a)                   (proxy spurieux, a varie)
      X_y = u*Z                             (feature spurieuse)
    """
    rng_global = _np_rng(42)
    rng        = _np_rng(seed)

    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,)))
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    C = rng.binomial(1, p_c, size=(n, 1)).astype(np.float32)

    gamma_scaled = gamma * np.sqrt(dim_z)
    noise_y = rng.normal(0.0, 1.0, size=(n, 1)).astype(np.float32)
    Y = (gamma_scaled * (2.0 * C - 1.0) + noise_y > 0.0).astype(np.float32)

    shift = causal_strength * (2.0 * Y - 1.0) * w_true.reshape(1, -1)
    X_z   = shift + rng.normal(0.0, 1.0, size=(n, dim_z)).astype(np.float32)

    N_e = rng.binomial(1, a, size=(n, 1))
    Z   = np.logical_xor(C.astype(bool), N_e.astype(bool)).astype(np.float32)

    X_y = (Z @ u.reshape(1, -1)).astype(np.float32)
    X_y = (X_y - X_y.mean(axis=0)) / (X_y.std(axis=0) + 1e-8)

    Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)
    return Xc, Y.astype(np.float32), Z.astype(np.float32), C, w_true, u


def build_envs_anti_causal_confounding_varying_proxy(
    n: int,
    a_train: List[float],
    a_test: float,
    gamma: float = 1.0,
    seed: int = 1,
    val_frac: float = 0.2,
    n_test: Optional[int] = None,
    *,
    dim_z: int = 1,
    dim_y: int = 1,
    causal_strength: float = 1.0,
    p_c_train: Optional[List[float]] = None,
    p_c_test: float = 0.35,
) -> Tuple[List[Env], List[Env], Env]:
    """Version anti-causale de build_envs_confounding_varying_proxy."""
    if n_test is None:
        n_test = n
    if p_c_train is None:
        p_c_train = [0.35] * len(a_train)

    train_envs, val_envs = [], []
    for i, a_e in enumerate(a_train):
        p_c_e = p_c_train[i]
        Xc, Y, Z, _C, w_true, u = make_env_anti_causal_confounding_varying_proxy(
            n=n, seed=seed + i, a=a_e, gamma=gamma, dim_z=dim_z, dim_y=dim_y,
            causal_strength=causal_strength, p_c=p_c_e,
        )
        (X_tr, y_tr), (_, y_vd) = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)
        (Z_tr, _), _             = _split_numpy(Z,  Y, val_frac, seed + 1000 + i)

        meta_train = {
            "kind": "anti_causal_confounding_proxy", "a": float(a_e),
            "gamma": float(gamma), "split": "train", "env_id": i,
            "dim_z": dim_z, "dim_y": dim_y, "w_true": w_true, "u": u,
            "p_c": float(p_c_e), "Z": torch.from_numpy(Z_tr),
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), None, meta_train))

        Xc_v, Y_v, Z_v, _, _, _ = make_env_anti_causal_confounding_varying_proxy(
            n=y_vd.shape[0], seed=seed + 5000 + i, a=a_e, gamma=gamma,
            dim_z=dim_z, dim_y=dim_y, causal_strength=causal_strength, p_c=p_c_e,
        )
        meta_val = {**meta_train, "split": "val", "Z": torch.from_numpy(Z_v)}
        val_envs.append(Env(torch.from_numpy(Xc_v), torch.from_numpy(Y_v), None, meta_val))

    Xc_t, Y_t, Z_t, _, w_t, u_t = make_env_anti_causal_confounding_varying_proxy(
        n=n_test, seed=seed + 777, a=a_test, gamma=gamma,
        dim_z=dim_z, dim_y=dim_y, causal_strength=causal_strength, p_c=p_c_test,
    )
    meta_t = {
        "kind": "anti_causal_confounding_proxy", "a": float(a_test),
        "gamma": float(gamma), "split": "test_ood", "env_id": "test",
        "dim_z": dim_z, "dim_y": dim_y, "w_true": w_t, "u": u_t,
        "p_c": float(p_c_test), "Z": torch.from_numpy(Z_t),
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), None, meta_t)
    return train_envs, val_envs, test_env


# ---------------------------------------------------------------------------
# 8) Anti-causal + confounding varying gamma
# ---------------------------------------------------------------------------
def make_env_anti_causal_confounding_varying_gamma(
    n: int,
    seed: int,
    gamma: float,
    *,
    dim_z: int = 1,
    dim_y: int = 1,
    causal_strength: float = 1.0,
    c_mean: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Anti-causal + confounding varying gamma.

    DAG :
            C   ~ N(c_mean, 1)                    (confondeur Gaussien continu)
      Y   = sign(γ_scaled*C + ε_Y)          (C → Y, γ varie par env)
      X_z = w_true*(2Y-1)*cs + ε_Xz        (Y → X_z, anti-causal)
      Z   = 1{C > 0}                         (proxy binarisé du confondeur)
      X_y = u*Z + bruit                      (feature spurieuse)
    """
    rng_global = _np_rng(42)
    rng        = _np_rng(seed)

    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,)))
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    C = rng.normal(c_mean, 1.0, size=(n, 1)).astype(np.float32)

    gamma_scaled = gamma * np.sqrt(dim_z)
    noise_y = rng.normal(0.0, 1.0, size=(n, 1)).astype(np.float32)
    Y = (gamma_scaled * C + noise_y > 0.0).astype(np.float32)

    shift = causal_strength * (2.0 * Y - 1.0) * w_true.reshape(1, -1)
    X_z   = shift + rng.normal(0.0, 1.0, size=(n, dim_z)).astype(np.float32)

    Z = (C > 0).astype(np.float32)

    X_y = (Z @ u.reshape(1, -1)).astype(np.float32)
    X_y += rng.normal(0.0, 0.1, size=(n, dim_y)).astype(np.float32)
    X_y = (X_y - X_y.mean(axis=0)) / (X_y.std(axis=0) + 1e-8)

    Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)
    return Xc, Y.astype(np.float32), Z.astype(np.float32), C, w_true, u


def build_envs_anti_causal_confounding_varying_gamma(
    n: int,
    train_gammas: List[float],
    test_gamma: float,
    seed: int,
    val_frac: float = 0.2,
    n_test: Optional[int] = None,
    dim_z: int = 1,
    dim_y: int = 1,
    causal_strength: float = 1.0,
    c_mean_train: Optional[List[float]] = None,
    c_mean_test: float = 0.0,
) -> Tuple[List[Env], List[Env], Env]:
    """Version anti-causale de build_envs_confounding_varying_gamma."""
    if n_test is None:
        n_test = n
    if c_mean_train is None:
        c_mean_train = [0.0] * len(train_gammas)

    train_envs, val_envs = [], []
    for i, g in enumerate(train_gammas):
        c_mean_e = c_mean_train[i]
        Xc, Y, Z, C, w_true, u = make_env_anti_causal_confounding_varying_gamma(
            n=n, seed=seed + i, gamma=g, dim_z=dim_z, dim_y=dim_y,
            causal_strength=causal_strength, c_mean=c_mean_e,
        )
        (X_tr, y_tr), (_, y_vd) = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)
        (Z_tr, _), _             = _split_numpy(Z,  Y, val_frac, seed + 1000 + i)

        meta = {
            "kind": "anti_causal_conf_vary_gamma", "gamma": g, "env_id": i,
            "dim_z": dim_z, "dim_y": dim_y, "w_true": w_true, "u": u,
            "c_mean": float(c_mean_e), "Z": torch.from_numpy(Z_tr),
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), meta=meta))

        Xc_v, Y_v, Z_v, _, _, _ = make_env_anti_causal_confounding_varying_gamma(
            n=y_vd.shape[0], seed=seed + 5000 + i, gamma=g,
            dim_z=dim_z, dim_y=dim_y, causal_strength=causal_strength, c_mean=c_mean_e,
        )
        meta_val = meta.copy(); meta_val["split"] = "val"; meta_val["Z"] = torch.from_numpy(Z_v)
        val_envs.append(Env(torch.from_numpy(Xc_v), torch.from_numpy(Y_v), meta=meta_val))

    Xc_t, Y_t, Z_t, _, w_t, u_t = make_env_anti_causal_confounding_varying_gamma(
        n=n_test, seed=seed + 999, gamma=test_gamma,
        dim_z=dim_z, dim_y=dim_y, causal_strength=causal_strength, c_mean=c_mean_test,
    )
    meta_test = {
        "kind": "anti_causal_conf_vary_gamma", "gamma": test_gamma,
        "env_id": "test", "dim_z": dim_z, "dim_y": dim_y,
        "w_true": w_t, "u": u_t, "c_mean": float(c_mean_test), "Z": torch.from_numpy(Z_t),
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), meta=meta_test)
    return train_envs, val_envs, test_env


# ---------------------------------------------------------------------------
# 9) Anti-causal + confounding varying pc
# ---------------------------------------------------------------------------
def make_env_anti_causal_confounding_varying_pc(
    n: int,
    seed: int,
    p_c: float,
    a: float = 0.0,
    gamma: float = 1.0,
    *,
    dim_z: int = 1,
    dim_y: int = 1,
    causal_strength: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Anti-causal + confounding varying pc.

    DAG :
      C   ~ Ber(p_e)                        (confondeur, p_e varie)
      Y   = sign(γ_scaled*(2C-1) + ε_Y)    (C → Y)
      X_z = w_true*(2Y-1)*cs + ε_Xz        (Y → X_z, anti-causal)
      Z   = C XOR Ber(a)                    (proxy spurieux)
      X_y = u*Z                              (feature spurieuse)
    """
    rng_global = _np_rng(42)
    rng        = _np_rng(seed)

    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,)))
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    C = rng.binomial(1, p_c, size=(n, 1)).astype(np.float32)

    gamma_scaled = gamma * np.sqrt(dim_z)
    noise_y = rng.normal(0.0, 1.0, size=(n, 1)).astype(np.float32)
    Y = (gamma_scaled * (2.0 * C - 1.0) + noise_y > 0.0).astype(np.float32)

    shift = causal_strength * (2.0 * Y - 1.0) * w_true.reshape(1, -1)
    X_z   = shift + rng.normal(0.0, 1.0, size=(n, dim_z)).astype(np.float32)

    N_e = rng.binomial(1, a, size=(n, 1))
    Z   = np.logical_xor(C.astype(bool), N_e.astype(bool)).astype(np.float32)

    X_y = (Z @ u.reshape(1, -1)).astype(np.float32)
    X_y = (X_y - X_y.mean(axis=0)) / (X_y.std(axis=0) + 1e-8)

    Xc = np.concatenate([X_z, X_y], axis=1).astype(np.float32)
    return Xc, Y.astype(np.float32), Z.astype(np.float32), C, w_true, u


def build_envs_anti_causal_confounding_varying_pc(
    n: int,
    pc_train: List[float],
    pc_test: float,
    a: float = 0.0,
    gamma: float = 1.0,
    seed: int = 1,
    val_frac: float = 0.2,
    n_test: Optional[int] = None,
    *,
    dim_z: int = 1,
    dim_y: int = 1,
    causal_strength: float = 1.0,
) -> Tuple[List[Env], List[Env], Env]:
    """Version anti-causale de build_envs_confounding_varying_pc."""
    if n_test is None:
        n_test = n

    train_envs, val_envs = [], []
    for i, p_e in enumerate(pc_train):
        Xc, Y, Z, _C, w_true, u = make_env_anti_causal_confounding_varying_pc(
            n=n, seed=seed + i, p_c=p_e, a=a, gamma=gamma,
            dim_z=dim_z, dim_y=dim_y, causal_strength=causal_strength,
        )
        (X_tr, y_tr), (_, y_vd) = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)
        (Z_tr, _), _             = _split_numpy(Z,  Y, val_frac, seed + 1000 + i)

        meta_train = {
            "kind": "anti_causal_confounding_pc", "p_c": float(p_e),
            "a": float(a), "gamma": float(gamma), "split": "train", "env_id": i,
            "dim_z": dim_z, "dim_y": dim_y, "w_true": w_true, "u": u,
            "Z": torch.from_numpy(Z_tr),
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), None, meta_train))

        Xc_v, Y_v, Z_v, _, _, _ = make_env_anti_causal_confounding_varying_pc(
            n=y_vd.shape[0], seed=seed + 5000 + i, p_c=p_e, a=a, gamma=gamma,
            dim_z=dim_z, dim_y=dim_y, causal_strength=causal_strength,
        )
        meta_val = {**meta_train, "split": "val", "Z": torch.from_numpy(Z_v)}
        val_envs.append(Env(torch.from_numpy(Xc_v), torch.from_numpy(Y_v), None, meta_val))

    Xc_t, Y_t, Z_t, _, w_t, u_t = make_env_anti_causal_confounding_varying_pc(
        n=n_test, seed=seed + 777, p_c=pc_test, a=a, gamma=gamma,
        dim_z=dim_z, dim_y=dim_y, causal_strength=causal_strength,
    )
    meta_t = {
        "kind": "anti_causal_confounding_pc", "p_c": float(pc_test),
        "a": float(a), "gamma": float(gamma), "split": "test_ood", "env_id": "test",
        "dim_z": dim_z, "dim_y": dim_y, "w_true": w_t, "u": u_t,
        "Z": torch.from_numpy(Z_t),
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), None, meta_t)
    return train_envs, val_envs, test_env


# ---------------------------------------------------------------------------
# 10) Anti-causal + sélection (collider)
# ---------------------------------------------------------------------------
def make_env_anti_causal_selection(
    n: int,
    alpha: float,
    seed: int,
    *,
    label_flip: float = 0.25,
    dim_z: int = 1,
    dim_y: int = 1,
    causal_strength: float = 1.0,
    p_y: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Anti-causal + biais de sélection (collider).

    DAG :
      Y*  ~ Ber(p_y)
      Y   = Y* XOR Ber(label_flip)
      X_z = w_true*(2Y*-1)*cs + ε    (Y → X_z, anti-causal)
      Z   ~ Ber(0.5)                  (variable spurieuse indépendante)
      X_y = u*Z                        (feature spurieuse)
      Sélection : P(garder) = alpha si Z==Y, sinon 1-alpha

    Variation d'environnement : alpha (même interprétation que le cas causal).
    p_y varie entre envs pour créer un déséquilibre de classe.
    """
    rng_global = _np_rng(42)
    rng        = _np_rng(seed)

    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,)))
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    kept_Xz, kept_Xy, kept_Y, kept_Z = [], [], [], []
    kept, total = 0, 0

    while kept < n:
        B = max(2048, n - kept)

        Y_star = rng.binomial(1, p_y, size=(B, 1)).astype(np.float32)
        Y = Y_star.copy()
        if label_flip > 0.0:
            flips = rng.uniform(size=(B, 1)) < label_flip
            Y[flips] = 1.0 - Y[flips]

        shift  = causal_strength * (2.0 * Y_star - 1.0) * w_true.reshape(1, -1)
        Xz     = shift + rng.normal(0.0, 1.0, size=(B, dim_z)).astype(np.float32)

        Z  = rng.binomial(1, 0.5, size=(B, 1)).astype(np.float32)
        Xy = Z @ u.reshape(1, -1)

        same       = (Z == Y).astype(np.float32)
        S_p        = np.where(same == 1.0, float(alpha), 1.0 - float(alpha))
        mask       = (rng.uniform(size=S_p.shape) < S_p).flatten()

        kept_Xz.append(Xz[mask]);  kept_Xy.append(Xy[mask])
        kept_Y.append(Y[mask]);    kept_Z.append(Z[mask])
        kept  += int(mask.sum());  total += B

    Xz_k = np.concatenate(kept_Xz, axis=0)[:n]
    Xy_k = np.concatenate(kept_Xy, axis=0)[:n]
    Xy_k = (Xy_k - Xy_k.mean(axis=0)) / (Xy_k.std(axis=0) + 1e-8)
    Y_k  = np.concatenate(kept_Y,  axis=0)[:n]
    Z_k  = np.concatenate(kept_Z,  axis=0)[:n]

    Xc      = np.concatenate([Xz_k, Xy_k], axis=1).astype(np.float32)
    sel_rate = kept / total if total > 0 else 0.0
    return Xc, Y_k.astype(np.float32), sel_rate, w_true, u


def build_envs_anti_causal_selection(
    n: int,
    train_alphas: List[float],
    test_alpha: float,
    seed: int = 1,
    val_frac: float = 0.2,
    n_test: Optional[int] = None,
    label_flip: float = 0.25,
    dim_z: int = 1,
    dim_y: int = 1,
    causal_strength: float = 1.0,
    p_y_train: Optional[List[float]] = None,
) -> Tuple[List[Env], List[Env], Env]:
    """Version anti-causale de build_envs_selection.

    p_y_train : prior P(Y*=1) par environnement d'entraînement.
        Si None, utilise 0.5 pour tous les envs.
        Exemple : [0.3, 0.7] → déséquilibre inversé entre les deux envs.
        Validation et test utilisent toujours p_y=0.5.
    """
    if n_test is None:
        n_test = n
    if p_y_train is None:
        p_y_train = [0.5] * len(train_alphas)

    train_envs, val_envs = [], []
    for i, psi in enumerate(train_alphas):
        p_y = p_y_train[i]
        Xc, Y, rate, w_true, u = make_env_anti_causal_selection(
            n=n, alpha=psi, seed=seed + i, label_flip=label_flip,
            dim_z=dim_z, dim_y=dim_y, causal_strength=causal_strength,
            p_y=p_y,
        )
        (X_tr, y_tr), _ = _split_numpy(Xc, Y, val_frac, seed + 1000 + i)

        meta = {
            "kind": "anti_causal_selection", "psi": float(psi),
            "label_flip": float(label_flip), "sel_rate": rate, "split": "train",
            "dim_z": dim_z, "dim_y": dim_y, "w_true": w_true, "u": u,
            "p_y": p_y,
        }
        train_envs.append(Env(torch.from_numpy(X_tr), torch.from_numpy(y_tr), None, meta))

        Xc_v, Y_v, rate_v, _, _ = make_env_anti_causal_selection(
            n=n, alpha=psi, seed=seed + 5000 + i, label_flip=label_flip,
            dim_z=dim_z, dim_y=dim_y, causal_strength=causal_strength,
            p_y=p_y,
        )
        (X_val, y_val), _ = _split_numpy(Xc_v, Y_v, val_frac, seed + 5000 + i)
        meta_val = {**meta, "split": "val", "sel_rate": rate_v, "label_flip": label_flip, "p_y": p_y}
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(y_val), None, meta_val))

    Xc_t, Y_t, rate_t, w_t, u_t = make_env_anti_causal_selection(
        n=n_test, alpha=test_alpha, seed=seed + 777, label_flip=0.0,
        dim_z=dim_z, dim_y=dim_y, causal_strength=causal_strength,
        p_y=0.5,
    )
    meta_t = {
        "kind": "anti_causal_selection", "psi": float(test_alpha),
        "label_flip": 0.0, "sel_rate": rate_t, "split": "test_ood",
        "dim_z": dim_z, "dim_y": dim_y, "w_true": w_t, "u": u_t,
        "p_y": 0.5,
    }
    test_env = Env(torch.from_numpy(Xc_t), torch.from_numpy(Y_t), None, meta_t)
    return train_envs, val_envs, test_env


# Wrapper pour compatibilité
def make_env_semi_anti_causal(
    n: int,
    p_spur: float,
    seed: int,
    label_flip: float = 0.25,
    dim_z: int = 1,
    dim_y: int = 1,
    causal_strength: float = 1.0,
    x_shift: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wrapper pour génération semi anti-causale (retourne aussi w_true et u)."""
    return _generate_semi_anti_causal(n, p_spur, seed, label_flip, dim_z, dim_y, causal_strength, x_shift)
