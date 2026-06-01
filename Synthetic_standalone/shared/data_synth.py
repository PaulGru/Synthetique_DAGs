# Generators for *synthetic* environments used in our IRM experiments:
#   - Covariate shift (pure)
#   - Toy spurious (Y -> C)
#   - Confounding (Z -> {X_s, Y})
#   - Selection / collider (conditioning on S=1)


from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Literal, Optional, Dict
import os, csv
import numpy as np
import torch


# =============================================================================
# Common environment container
# =============================================================================

@dataclass
class Env:
    """
    Container for a single environment: data, labels, and meta-information.

    Attributes
    ----------
    X : torch.Tensor
        Feature matrix (N, d). By convention d=2 for toy data: [X_s, C].
    y : torch.Tensor
        Label vector (N, 1), binary {0,1} (float32).
    y_true : Optional[torch.Tensor]
        (Optional) ground-truth labels before label noise was applied.
    meta : Optional[Dict]
        Free-form dict (kind, generative parameters, split, etc.).
    """
    X: torch.Tensor
    y: torch.Tensor
    y_true: Optional[torch.Tensor] = None
    meta: Optional[Dict] = None


# =============================================================================
# Internal helpers (seeding, splitting, basic functions)
# =============================================================================

def _np_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def _split_indices(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (train_idx, val_idx) of sizes ~(1-val_frac)*n and val_frac*n.
    The split is deterministic given `seed`.
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
# Generative model (per environment e):
#   1) X_z ~ N(0, 1)                      (causal feature, identical across envs)
#   2) Y* = 1{ X_z > 0 }                  (clean causal rule)
#   3) Label flip: Y = Y* XOR Bernoulli(label_flip)
#        -> Weakens the causal correlation X_z <-> Y
#   4) Binary style variable:
#        Z = Y XOR Bernoulli(p_spur_e)
#        -> Strong Y <-> Z correlation when p_spur_e << 0.5
#   5) Continuous spurious feature:
#        X_y = Z + ε_X,  ε_X ~ N(0, sigma_x^2)
#
# Goal:
#   - corr(Y, Z) > corr(Y, X_z) in training environments
#   - at test time, increasing p_spur_e (≈ 0.5 or > 0.5) breaks the spurious
#     correlation while keeping the causal mechanism X_z -> Y invariant.


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
    Generate n samples from the semi anti-causal model X_z -> Y -> Z -> X_y.

    Parameters
    ----------
    n : int
        Number of examples to generate.
    p_spur : float
        Probability of flipping Z after copying it from Y:
          - p_spur = 0.0  -> Z = Y (maximum correlation)
          - p_spur = 0.5  -> Z ⟂ Y (independence)
          - p_spur > 0.5  -> reversed correlation.
    seed : int
        RNG seed.
    label_flip : float, optional
        Symmetric label flip probability:
          - higher → weakens the causal link X_z -> Y.
          - typically 0.25 (as in Empirical or Invariant RM).
    dim_z : int, optional
        Dimension of the causal feature X_z (default: 1).
    dim_y : int, optional
        Dimension of the spurious feature X_y (default: 1).
    causal_strength : float, optional
        Multiplicative factor for the variance of X_z.
        Larger = better visual separation between Y=0 and Y=1.
        Default: 1.0 (standard N(0,1) variance).

    Returns
    -------
    Xc : np.ndarray (n, dim_z + dim_y)
        Features [X_z, X_y].
    Y  : np.ndarray (n, 1), float32 in {0,1}
        Labels (after flipping).
    Z  : np.ndarray (n, 1), float32 in {0,1}
        Binary style variable (not used as feature, but available for analysis).
    """
    # CRITICAL FIX: Fixed GLOBAL seed for w_true and u
    # Guarantees the causal function is IDENTICAL across all environments
    # This is the fundamental IRM assumption!
    rng_global = _np_rng(42)  # Fixed seed for the causal direction vectors
    rng = _np_rng(seed)       # Variable seed for the rest (sampling)

    # 1) True causal weight vector: w_true (INVARIANT across envs)
    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    # 2) Spurious direction u (INVARIANT across envs)
    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,))).astype(np.float32)
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    # 3) Feature causale : X_z ~ N(x_shift * ŵ_true, causal_strength² * I)
    # The shift is aligned with w_true (discriminant direction) to unbalance P(Y*=1).
    w_hat = w_true / (np.linalg.norm(w_true) + 1e-8)  # direction unitaire
    mu = x_shift * w_hat  # (dim_z,)
    X_z = rng.normal(0.0, causal_strength, size=(n, dim_z)).astype(np.float32) + mu.astype(np.float32)

    # 4) "Clean" label: Y* = 1{w_true · X_z > 0}
    Y_star = ((X_z @ w_true) > 0).astype(np.float32).reshape(-1, 1)

    # 5) Symmetric label flip to weaken the causal signal
    Y = Y_star.copy()
    if label_flip > 0.0:
        mask = rng.uniform(0.0, 1.0, size=(n, 1)) < label_flip
        Y[mask] = 1.0 - Y[mask]

    # 6) Binary style variable Z = Y XOR Bernoulli(p_spur)
    Z = Y.copy()
    flips_z = rng.uniform(0.0, 1.0, size=(n, 1)) < p_spur
    Z[flips_z] = 1.0 - Z[flips_z]

    # 7) Spurious feature: X_y = u * Z + noise
    X_y = (Z @ u.reshape(1, -1)) + rng.normal(0.0, 1e-1, size=(n, dim_y)).astype(np.float32)
    # Standardize X_y to have variance ~1 (same as X_z)
    X_y = (X_y - X_y.mean(axis=0)) / (X_y.std(axis=0) + 1e-8)

    # 8) Final features: [X_z, X_y]
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
    Build semi anti-causal environments.

    Parameters
    ----------
    n : int
        Number of examples per training environment.
    train_p_spurs : List[float]
        List of p_spur_e values for training envs (e.g. [0.1, 0.2]).
        -> strong spurious alignment in training.
    test_p_spur : float
        p_spur_e for the test environment (e.g. 0.9 for reversed correlation).
    seed : int
        Global RNG seed.
    val_frac : float, optional
        Validation fraction within each training environment.
    label_flip : float, optional
        Label flip probability (affects the causal signal X_z->Y equally
        across all envs).
    n_test : Optional[int], optional
        Number of test examples (default: n).
    dim_z : int, optional
        Dimension of the causal feature X_z (default: 1).
    dim_y : int, optional
        Dimension of the spurious feature X_y (default: 1).
    x_shifts_train : Optional[List[float]], optional
        Shift of X_z along w_true per training environment.
        P(Y*=1) ≈ Φ(x_shift / causal_strength) per env.
        E.g. [-1.0, 1.0] → env0 ≈ 16% class 1, env1 ≈ 84% class 1.
        Validation and test always use x_shift=0 (balanced classes).
        Default: None (=0.0 for all envs).

    Returns
    -------
    train_envs : List[Env]
        Training environments (with X=[X_z,X_y]).
    val_envs : List[Env]
        Corresponding validation environments.
    test_env : Env
        OOD test environment (spurious correlation broken/reversed).
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

    # OOD test environment
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
    a: float,             # strength of the C -> Z link (varies with env)
    gamma: float = 1.0,   # poids du confondeur C dans Y
    *,
    dim_z: int = 1,
    dim_y: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a confounded environment with binary Z.

    Causal graph:
      C   ~ Ber(0.35)              (latent confounder)
      Z   = C ⊕ N^e               (spurious proxy, varies with env)
      X^⊥_Z ~ N(0, I)             (PURE CAUSAL feature, ⟂ C and Z)
      X^⊥_Y = u * Z               (PURE SPURIOUS feature, parent: Z)
      Y  = sign(w·X^⊥_Z + γ·(2C−1))
      X  = [X^⊥_Z, X^⊥_Y]
    """
    rng_global = _np_rng(42)  # Fixed seed for the causal direction vectors
    rng = _np_rng(seed)       # Variable seed for the rest

    # 1) Spurious direction vector u (INVARIANT)
    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,)))
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    # 2) Causal weight vector w_true (INVARIANT)
    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    # 3) Latent confounder C
    C = rng.binomial(1, 0.35, size=(n, 1)).astype(np.float32)

    # 4) Causal feature X^⊥_Z, independent of C
    X_z = rng.normal(0.0, 1.0, size=(n, dim_z)).astype(np.float32)

    # 5) Spurious proxy Z = C XOR N^e
    N_e = rng.binomial(1, a, size=(n, 1))
    Z = np.logical_xor(C.astype(bool), N_e.astype(bool)).astype(np.float32)

    # 6) Pure spurious feature X^⊥_Y = u * Z
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
    a_train: List[float],        # list of a_e (beta^e) values for TRAIN environments
    a_test: float,               # a_e (beta^e) for the OOD TEST environment
    gamma: float = 1.0,
    seed: int = 1,
    val_frac: float = 0.2,
    n_test: Optional[int] = None,
    *,
    dim_z: int = 1,
    dim_y: int = 1,
) -> Tuple[List[Env], List[Env], Env]:
    """
    Build a multi-environment dataset with CF-CMNIST-style confounding:

      C   ~ Ber(0.25)                      (confounder)
      X_z ~ N(0, 1)                        (causal feature, ⟂ C)

      For each env e:
        N^e ~ Ber(a_e)
        Z   = C XOR N^e
        X_y = (2 Z - 1) + ε_X,  ε_X ~ N(0, 0.5)

      Y_base = sign( w X_z + gamma (2C-1) )

      - In TRAIN/VAL/TEST: random flip with prob. label_flip (if > 0).
        Note: label_flip=0 at test time to evaluate the "true" function.

      X = [X_z, X_y].

    Environment variation:
      - a_e (Ber(a_e) parameter for N^e) controls the strength of the
        C -> Z -> X_y link, hence the spurious correlation between X_y and Y.
      - The causal mechanism X_z -> Y_base (w) and the distribution of C are fixed.
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
# 3) Selection bias — Spurious correlation induced by a selection process
# =============================================================================
def make_env_selection(
    n: int,
    alpha: float,      # Probability of keeping an example where Z==Y (creates spurious correlation)
    seed: int,
    *,
    label_flip: float = 0.25,
    keep_if_one: bool = True,
    dim_z: int = 1,
    dim_y: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Selection bias.

    Causal graph:
      Z              ~ Bernoulli(1/2)           (spurious context variable)
      X^⊥_Z          ~ N(0, I_dim_z)           (PURE CAUSAL feature)
      Y*             = sign(w·X^⊥_Z)           (latent)
      Y              = 1{Y*>0}                 (with optional flip)
      X^⊥_Y          = u * Z                   (PURE SPURIOUS feature)
      Selection: P(keep) = alpha if Z==Y, 1-alpha otherwise
      X = [X^⊥_Z, X^⊥_Y]

    Parameters
    ----------
    alpha : float
        alpha = 0.9 → strong Z==Y correlation (train)
        alpha = 0.5 → no bias
        alpha = 0.1 → reversed correlation (OOD)
    """
    rng_global = _np_rng(42)
    rng = _np_rng(seed)

    # 1) Causal weight vector (INVARIANT)
    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    # 2) Spurious direction (INVARIANT)
    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,)))
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    kept_Xz, kept_Xy, kept_Y, kept_Z = [], [], [], []
    kept, total = 0, 0

    while kept < n:
        B = max(2048, n - kept)

        # --- Base population ---
        Z = rng.binomial(1, 0.5, size=(B, 1)).astype(np.float32)
        Xz = rng.normal(0, 1.0, size=(B, dim_z)).astype(np.float32)  # X^⊥_Z (causal)
        Xy = Z @ u.reshape(1, -1)                                      # X^⊥_Y (spurieux)

        logit = (Xz @ w_true).reshape(-1, 1)

        Y = (logit > 0.0).astype(np.float32)

        if label_flip and label_flip > 0.0:
            flips = rng.uniform(size=Y.shape) < label_flip
            Y[flips] = 1.0 - Y[flips]

        # --- Selection based on Z==Y ---
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
        # ===== TRAIN: label flip enabled =====
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

        # ===== VAL: same label flip as training =====
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

    # ===== TEST: no label flip =====
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


# =============================================================================
# 6) Anti-causal : Y → X_z  (reverse direction compared to the causal case)
#    + same 4 spurious perturbations as above
# =============================================================================
#
# In the ANTI-CAUSAL case, Y is sampled first (uniform prior),
# then X_z is generated *from* Y: X_z = w_true*(2Y-1)*cs + ε.
# The classification task remains "predict Y from X = [X_z, X_y]".
# The conceptual difference is that the INVARIANT mechanism is P(X_z|Y)
# rather than P(Y|X_z). IRM seeks an invariant P(Y|Φ(X)), which here
# corresponds to inverting this mechanism via Bayes' theorem. The 5
# spurious perturbations mirror the causal cases:
#
#   ac_semi_anti_causal      : Y→X_z, Y→Z→X_y
#   ac_selection             : Y→X_z, selection on Z==Y
#   ac_confounding_proxy     : C→Y, C→Z→X_y, Y→X_z
#   ac_confounding_gamma     : same, gamma varies per env
#   ac_confounding_pc        : same, P(C=1) varies per env
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
    Generate n samples from the ANTI-CAUSAL semi-anti-causal model:

      Y*  ~ Ber(p_y)                  (label prior, p_y varies per env)
      Y   = Y* XOR Ber(label_flip)    (noisy label)
      X_z = w_true*(2Y*-1)*cs + ε    (ANTI-CAUSAL feature: Y → X_z)
      Z   = Y  XOR Ber(p_spur)        (spurious variable)
      X_y = u*Z + ε_y                 (spurious feature)

    The causal direction is reversed compared to the causal case:
    here Y *causes* X_z, not X_z causes Y.

    DAG : Y* → X_z    and    Y → Z → X_y
    """
    rng_global = _np_rng(42)
    rng        = _np_rng(seed)

    w_true = np.abs(rng_global.normal(0.0, 1.0, size=(dim_z,)))
    w_true = w_true / np.linalg.norm(w_true) * np.sqrt(dim_z)

    u = np.abs(rng_global.normal(0.0, 1.0, size=(dim_y,))).astype(np.float32)
    u = u / np.linalg.norm(u) * np.sqrt(dim_y)

    # 1) True label Y* (potentially unbalanced prior)
    Y_star = rng.binomial(1, p_y, size=(n, 1)).astype(np.float32)

    # 2) Noisy label (observed during training)
    Y = Y_star.copy()
    if label_flip > 0.0:
        mask = rng.uniform(0.0, 1.0, size=(n, 1)) < label_flip
        Y[mask] = 1.0 - Y[mask]

    # 3) Anti-causal: X_z is *generated from* Y_star
    shift = causal_strength * (2.0 * Y_star - 1.0) * w_true.reshape(1, -1)  # (n, dim_z)
    X_z   = shift + rng.normal(0.0, 1.0, size=(n, dim_z)).astype(np.float32)

    # 4) Spurious variable Z = Y XOR Ber(p_spur)
    Z = Y.copy()
    flips_z = rng.uniform(0.0, 1.0, size=(n, 1)) < p_spur
    Z[flips_z] = 1.0 - Z[flips_z]

    # 5) Spurious feature X_y
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

    p_y_train : prior P(Y*=1) per training environment.
        If None, uses 0.5 (balanced) for all envs.
        Example: [0.3, 0.7] → class 0 majority in env 0,
                               class 1 majority in env 1.
        Validation and test always use p_y=0.5.
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
    label_flip: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Anti-causal + confounding varying proxy.

    DAG :
            C   ~ Ber(p_c)
      Y   = sign(γ_scaled*(2C-1) + ε_Y)   (C → Y)
      X_z = w_true*(2Y-1)*cs + ε_Xz       (Y → X_z, anti-causal)
      Z   = C XOR Ber(a)                   (proxy spurieux, a varie)
      X_y = u*Z                             (spurious feature)
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

    Y_for_Xz = Y.copy()
    if label_flip > 0.0:
        mask = rng.uniform(0.0, 1.0, size=(n, 1)) < label_flip
        Y_for_Xz[mask] = 1.0 - Y_for_Xz[mask]

    shift = causal_strength * (2.0 * Y_for_Xz - 1.0) * w_true.reshape(1, -1)
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
    label_flip: float = 0.0,

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
            causal_strength=causal_strength, p_c=p_c_e, label_flip=label_flip,
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
    Anti-causal + selection bias (collider).

    DAG:
      Y*  ~ Ber(p_y)
      Y   = Y* XOR Ber(label_flip)
      X_z = w_true*(2Y*-1)*cs + ε    (Y → X_z, anti-causal)
      Z   ~ Ber(0.5)                  (independent spurious variable)
      X_y = u*Z                        (spurious feature)
      Selection: P(keep) = alpha if Z==Y, 1-alpha otherwise

    Environment variation: alpha (same interpretation as the causal case).
    p_y varies across envs to create class imbalance.
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

    p_y_train : prior P(Y*=1) per training environment.
        If None, uses 0.5 for all envs.
        Example: [0.3, 0.7] → reversed imbalance between the two envs.
        Validation and test always use p_y=0.5.
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


# Compatibility wrapper
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
    """Compatibility wrapper for semi anti-causal generation (also returns w_true and u)."""
    return _generate_semi_anti_causal(n, p_spur, seed, label_flip, dim_z, dim_y, causal_strength, x_shift)
