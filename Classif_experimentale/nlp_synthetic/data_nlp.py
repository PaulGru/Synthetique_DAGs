from __future__ import annotations

import sys
from pathlib import Path as _Path
# Ajoute la racine du projet + le dossier shared/ au chemin Python
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

# Générateurs d'environnements NLP (SMS Spam) pour expériences IRM
# Adaptation du DAG semi anti-causal : Text → Y → Z → X_y (spurious token embedding)

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import re
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from data_synth import Env  # Réutiliser la même structure Env


# =============================================================================
# Configuration des tokens trompeurs
# =============================================================================

def define_spurious_tokens() -> Dict[str, str]:
    """
    Définit les tokens textuels trompeurs pour SMS Spam.

    Choix : "sky" (ham) et "fire" (spam).
    - Tokens BERT uniques (un seul token WordPiece chacun).
    - Pas de connotation directe spam/ham : "sky" n'est pas associé aux
      conversations normales, "fire" n'apparaît pas dans les vocabulaires
      typiques de spam promotionnel (FREE, WIN, PRIZE...).
    - Injectés en préfixe : le texte original reste intact, BERT peut
      toujours encoder le contenu SMS normalement.
    """
    return {
        "spam_correlated": "fire",
        "ham_correlated":  "sky",
    }


# =============================================================================
# =============================================================================
# Chargement du dataset SMS Spam
# =============================================================================

def load_sms_spam_dataset(seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Charge le dataset SMS Spam depuis Hugging Face.
    
    Dataset: mshenoda/spam-messages (59k messages)
    
    Parameters
    ----------
    seed : int
        Graine pour le shuffle du dataset.
    
    Returns
    -------
    texts : List[str]
        Liste des SMS (textes bruts).
    labels : List[int]
        Labels binaires (0=ham, 1=spam).
    """
    from datasets import concatenate_datasets
    
    # Charger le nouveau dataset (59k messages)
    dataset = load_dataset("mshenoda/spam-messages")
    
    # Combiner les 3 splits (train + validation + test)
    all_data = concatenate_datasets([
        dataset['train'],
        dataset['validation'],
        dataset['test']
    ])
    
    # Shuffle avec seed fixe
    all_data = all_data.shuffle(seed=seed)
    
    # Extraire textes et convertir labels string → int
    texts = all_data['text']
    labels = [1 if label == 'spam' else 0 for label in all_data['label']]
    
    return texts, labels


# =============================================================================
# Injection de tokens trompeurs (semi anti-causal)
# =============================================================================

# Mots de haute fréquence, sémantiquement vides, présents dans les trois datasets.
# Le token spurieux REMPLACE chacune de leurs occurrences dans le texte.
# Résultat : le signal trompeur est distribué à travers tout le texte (plusieurs
# positions) plutôt que concentré en un seul préfixe, ce qui le rend beaucoup
# plus difficile à ignorer pour ERM.
NEUTRAL_WORDS: List[str] = [
    "the", "a", "an", "of", "in", "to", "is", "it",
    "and", "or", "at", "on", "for", "with", "as", "by",
]


def _prepend_token_to_neutral_words(text: str, neutral_words: List[str], token: str) -> Optional[str]:
    """
    Insère `token` devant chaque occurrence de mot neutre dans `text`.

    Ex : token="fire", text="I went to the store and it was great"
         → "I went to fire the store fire and fire it was great"

    Les mots neutres sont conservés (non remplacés). Le signal spurieux est
    ainsi distribué à travers tout le texte, ce qui le rend beaucoup plus
    difficile à ignorer pour ERM que le simple préfixe de phrase.
    Utilise des word boundaries (\\b) pour ne pas toucher aux sous-chaînes
    (ex. : "for" dans "format" n'est pas affecté).
    Retourne None si aucun mot neutre n'est trouvé (fallback préfixe).
    """
    result = text
    found_any = False
    for word in neutral_words:
        pattern = re.compile(r'\b(' + re.escape(word) + r')\b', re.IGNORECASE)
        new_result, count = pattern.subn(token + r' \1', result)
        if count > 0:
            found_any = True
            result = new_result
    return result if found_any else None


def inject_spurious_token(
    text: str,
    label: int,
    p_correct: float,
    spurious_tokens: Dict[str, str],
    rng: np.random.Generator,
    position: str = "prefix",
    neutral_words: Optional[List[str]] = NEUTRAL_WORDS,
) -> str:
    """
    Injecte un token textuel trompeur dans le texte avec corrélation contrôlée.

    Deux modes selon `neutral_words` :

    Mode 1 — préfixe devant chaque mot neutre (défaut, recommandé) :
      Insère le token spurieux devant chaque occurrence de mot fréquent
      sémantiquement vide ("the", "a", "of", ...) en conservant le mot.
      Ex : "fire the cat" (conserve "the"). Le signal est distribué à travers
      l'ensemble du texte → beaucoup plus difficile à ignorer pour ERM.
      Utilisé quand `neutral_words` est fourni ET que le texte contient des occurrences.

    Mode 2 — préfixe de phrase (fallback) :
      Quand aucun mot neutre n'est présent dans le texte (textes très courts),
      le token est ajouté en préfixe ou suffixe.

    Parameters
    ----------
    text : str            Texte SMS original.
    label : int           Label vrai (0=ham, 1=spam).
    p_correct : float     Probabilité d'associer le "bon" token au label.
    spurious_tokens : dict  {"spam_correlated": tok, "ham_correlated": tok}.
    rng : np.random.Generator
    position : str        "prefix" ou "suffix" (fallback uniquement).
    neutral_words : List[str] ou None
        Si None, mode préfixe de phrase seul.
    """
    spam_token = spurious_tokens["spam_correlated"]
    ham_token  = spurious_tokens["ham_correlated"]

    if rng.uniform() < p_correct:
        token = spam_token if label == 1 else ham_token
    else:
        token = ham_token  if label == 1 else spam_token

    if neutral_words is not None:
        prepended = _prepend_token_to_neutral_words(text, neutral_words, token)
        if prepended is not None:
            return prepended

    # Fallback préfixe/suffixe
    return f"{token} {text}" if position == "prefix" else f"{text} {token}"


# =============================================================================
# Tokenisation et extraction d'embeddings BERT
# =============================================================================

# =============================================================================
# Singleton BERT — chargé une seule fois par processus
# =============================================================================

_BERT_CACHE: Dict[str, Any] = {}   # clé : model_name → {"tokenizer": ..., "model": ...}


def _get_bert(model_name: str, device: str):
    """
    Retourne (tokenizer, model) en les chargeant une seule fois par model_name.
    Les appels suivants réutilisent l'instance en mémoire, évitant de recharger
    BERT depuis le disque (~500 Mo, ~2-5 s) à chaque build_envs_*.
    """
    if model_name not in _BERT_CACHE:
        print(f"  [BERT] Chargement de {model_name} (une seule fois)…")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        _BERT_CACHE[model_name] = {"tokenizer": tokenizer, "model": model}

    entry = _BERT_CACHE[model_name]
    # Déplacer sur le bon device si nécessaire
    current_device = next(entry["model"].parameters()).device
    if str(current_device) != str(device):
        entry["model"] = entry["model"].to(device)
    return entry["tokenizer"], entry["model"]


# =============================================================================
# Cache d'embeddings sur disque
# =============================================================================

def _embed_cache_path(texts: List[str], model_name: str,
                      max_length: int, pooling: str) -> str:
    """
    Calcule un chemin de cache unique basé sur un hash du contenu +
    des hyperparamètres d'embedding.  Stocké dans nlp_synthetic/.embed_cache/.
    """
    import hashlib, os
    digest = hashlib.md5(
        (repr(texts) + model_name + str(max_length) + pooling).encode()
    ).hexdigest()
    cache_dir = os.path.join(os.path.dirname(__file__), ".embed_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{digest}.npy")


def tokenize_and_embed_with_bert(
    texts: List[str],
    model_name: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    use_cache: bool = True,
    finetune_bert_layers: int = 0,
) -> np.ndarray:
    """
    Tokenise les textes et extrait les embeddings BERT.

    Optimisations par rapport à la version initiale :
    - **Singleton BERT** : le modèle est chargé une seule fois par processus
      (pas de rechargement à chaque build_envs_*).
    - **Cache disque** : les embeddings sont sauvegardés dans
      nlp_synthetic/.embed_cache/<hash>.npy.  Un second lancement avec les mêmes
      textes + hyperparamètres est quasi-instantané.
    - **Batch size adaptatif** : 64 sur GPU/MPS, 32 sur CPU.

    Parameters
    ----------
    texts : List[str]
    model_name : str
    max_length : int
    device : str
    pooling : str      "mean" | "cls" | "max"
    use_cache : bool   Mettre False pour forcer le re-calcul.
    """

    if finetune_bert_layers > 0:
        tokenizer, _ = _get_bert(model_name, "cpu")
        batch_size = 256
        all_stacked = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(
                batch_texts, padding="max_length", truncation=True,
                max_length=max_length, return_tensors="np"
            )
            stacked = np.stack([encoded["input_ids"], encoded["attention_mask"]], axis=-1)
            all_stacked.append(stacked)
        return np.concatenate(all_stacked, axis=0)

    import os

    # ── Cache disque ──────────────────────────────────────────────────────
    if use_cache:
        cache_path = _embed_cache_path(texts, model_name, max_length, pooling)
        if os.path.exists(cache_path):
            return np.load(cache_path)

    # ── Modèle (singleton) ────────────────────────────────────────────────
    tokenizer, model = _get_bert(model_name, device)
    model_device = next(model.parameters()).device

    if str(model_device) in ("cuda", "mps"):
        batch_size = 256 if max_length <= 128 else 128
    else:
        batch_size = 64

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            input_ids      = encoded["input_ids"].to(model_device)
            attention_mask = encoded["attention_mask"].to(model_device)

            outputs     = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # (B, seq_len, 768)

            if pooling == "mean":
                mask_exp = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_emb  = torch.sum(last_hidden * mask_exp, dim=1)
                sum_mask = torch.clamp(mask_exp.sum(dim=1), min=1e-9)
                batch_emb = sum_emb / sum_mask
            elif pooling == "cls":
                batch_emb = last_hidden[:, 0, :]
            elif pooling == "max":
                batch_emb = torch.max(last_hidden, dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")

            embeddings.append(batch_emb.cpu().numpy())

    result = np.concatenate(embeddings, axis=0).astype(np.float32)

    # ── Sauvegarde cache ─────────────────────────────────────────────────
    if use_cache:
        np.save(cache_path, result)
        print(f"  [cache] Embeddings sauvegardés → {os.path.basename(cache_path)}")

    return result


# =============================================================================
# Génération d'environnements NLP semi anti-causal
# =============================================================================

def make_env_nlp_semi_anti_causal(
    all_texts: List[str],
    all_labels: List[int],
    n: int,
    p_correct: float,
    seed: int,
    label_flip: float = 0.25,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Génère un environnement NLP avec corrélation emoji-label contrôlée.
    
    Parameters
    ----------
    all_texts : List[str]
        Pool de tous les SMS disponibles.
    all_labels : List[int]
        Labels correspondants.
    n : int
        Nombre d'exemples à échantillonner.
    p_correct : float
        Probabilité d'associer le "bon" emoji au label (0.0=inversé, 1.0=parfait).
    seed : int
        Graine aléatoire.
    bert_model : str
        Modèle BERT à utiliser.
    max_length : int
        Longueur max de séquence.
    device : str
        Device PyTorch.
    pooling : str
        Type de pooling pour les embeddings.
    
    Returns
    -------
    X : np.ndarray (n, 768)
        Embeddings BERT.
    Y : np.ndarray (n, 1)
        Labels.
    texts_original : List[str]
        Textes originaux (pour debug).
    texts_modified : List[str]
        Textes avec tokens trompeurs injectés.
    """
    rng = np.random.default_rng(seed)
    
    # 1) Échantillonner n exemples
    n_available = len(all_texts)
    if n > n_available:
        # Si on demande plus d'exemples que disponible, échantillonner avec remplacement
        indices = rng.choice(n_available, size=n, replace=True)
    else:
        indices = rng.choice(n_available, size=n, replace=False)
    
    sampled_texts = [all_texts[i] for i in indices]
    sampled_labels = np.array([all_labels[i] for i in indices])
    
    # 2) Appliquer le flip de label (AVANT l'injection d'emoji)
    #    Cela affaiblit le signal causal texte→label
    if label_flip > 0.0:
        flip_mask = rng.uniform(size=len(sampled_labels)) < label_flip
        sampled_labels_flipped = sampled_labels.copy()
        sampled_labels_flipped[flip_mask] = 1 - sampled_labels_flipped[flip_mask]
    else:
        sampled_labels_flipped = sampled_labels.copy()
    
    # 3) Injecter tokens trompeurs basés sur les labels BRUITÉS
    #    (La corrélation emoji-label sera donc avec le label bruité, pas le vrai)
    spurious_tokens = define_spurious_tokens()
    texts_modified = []
    
    for text, label in zip(sampled_texts, sampled_labels_flipped):
        modified_text = inject_spurious_token(
            text=text,
            label=int(label),
            p_correct=p_correct,
            spurious_tokens=spurious_tokens,
            rng=rng,
            position="prefix"  # Ajouter au début
        )
        texts_modified.append(modified_text)
    
    # 4) Tokeniser et extraire embeddings BERT
    X = tokenize_and_embed_with_bert(
        texts=texts_modified,
        model_name=bert_model,
        max_length=max_length,
        device=device,
        pooling=pooling, finetune_bert_layers=finetune_bert_layers
    )
    
    # 5) Préparer labels (on retourne les labels BRUITÉS)
    Y = sampled_labels_flipped.reshape(-1, 1).astype(np.float32)
    
    return X, Y, sampled_texts, texts_modified


# =============================================================================
# Construction de plusieurs environnements (train/val/test)
# =============================================================================

def _split_indices(n: int, val_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Retourne (train_idx, val_idx) comme dans data_synth.py"""

# =============================================================================
# Construction complète avec tout le dataset (80/10/10 split)
# =============================================================================

def build_envs_nlp_semi_anti_causal(
    n: int,  # IGNORÉ
    train_p_correct: List[float],
    test_p_correct: float,
    seed: int,
    val_frac: float = 0.2,  # IGNORÉ  
    n_test: Optional[int] = None,  # IGNORÉ
    label_flip: float = 0.25,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """Utilise TOUT le dataset avec split 80/10/10 et pas de chevauchement entre envs."""
    print("Chargement du dataset SMS Spam...")
    all_texts, all_labels = load_sms_spam_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset chargé : {n_total} SMS")
    
    # Split global
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test_split = int(n_total * 0.1)
    n_val_split = int(n_total * 0.1)
    
    test_indices = indices[:n_test_split]
    val_indices = indices[n_test_split:n_test_split + n_val_split]
    train_indices = indices[n_test_split + n_val_split:]
    
    print(f"\nSplit: Train {len(train_indices)} | Val {len(val_indices)} | Test {len(test_indices)}")
    
    n_envs = len(train_p_correct)
    samples_per_env = len(train_indices) // n_envs
    
    train_envs, val_envs = [], []
    spurious_tokens = define_spurious_tokens()
    
    # TRAIN ENVS
    for i, p_correct in enumerate(train_p_correct):
        print(f"\n=== Train Env {i} (p_correct={p_correct:.0%}) ===")
        start = i * samples_per_env
        end = (i + 1) * samples_per_env if i < n_envs - 1 else len(train_indices)
        env_idx = train_indices[start:end]
        
        texts = [all_texts[j] for j in env_idx]
        labels = np.array([all_labels[j] for j in env_idx])
        
        if label_flip > 0:
            mask = rng.uniform(size=len(labels)) < label_flip
            labels[mask] = 1 - labels[mask]
        
        texts_mod = [inject_spurious_token(t, int(l), p_correct, spurious_tokens, rng)
                     for t, l in zip(texts, labels)]
        X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        Y = labels.reshape(-1, 1).astype(np.float32)
        
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                             meta={"p_correct": p_correct, "label_flip": label_flip, "n_samples": len(X)}))
        
        # VAL ENV
        print(f"=== Val Env {i} ===")
        val_texts = [all_texts[j] for j in val_indices]
        val_labels = np.array([all_labels[j] for j in val_indices])
        # Même label_flip que le train pour que val ait la même distribution
        if label_flip > 0:
            val_rng = np.random.default_rng(seed + 5000 + i)
            mask_val = val_rng.uniform(size=len(val_labels)) < label_flip
            val_labels[mask_val] = 1 - val_labels[mask_val]
        val_texts_mod = [inject_spurious_token(t, int(l), p_correct, spurious_tokens,
                                              np.random.default_rng(seed+5000+i))
                        for t, l in zip(val_texts, val_labels)]
        X_val = tokenize_and_embed_with_bert(val_texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        Y_val = val_labels.reshape(-1, 1).astype(np.float32)
        
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(Y_val),
                           meta={"p_correct": p_correct, "n_samples": len(X_val)}))
    
    # TEST ENV
    print(f"\n=== Test OOD (p_correct={test_p_correct:.0%}) ===")
    test_texts = [all_texts[j] for j in test_indices]
    test_labels = np.array([all_labels[j] for j in test_indices])
    test_rng = np.random.default_rng(seed+777)
    test_texts_mod = [inject_spurious_token(t, int(l), test_p_correct, spurious_tokens, test_rng)
                     for t, l in zip(test_texts, test_labels)]
    X_test = tokenize_and_embed_with_bert(test_texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    Y_test = test_labels.reshape(-1, 1).astype(np.float32)
    
    test_env = Env(torch.from_numpy(X_test), torch.from_numpy(Y_test),
                  meta={"p_correct": test_p_correct, "n_samples": len(X_test)})
    
    print(f"\n✅ Done! Train: {sum(e.X.shape[0] for e in train_envs)} | Val: {val_envs[0].X.shape[0]} | Test: {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# Confounding helpers (shared by the 3 variants)
# =============================================================================
#
# DAG commun à tous les cas de confounding :
#
#   C   (confondeur latent binaire)
#   ├── C → Z → token injecté dans le texte   (chemin spurieux)
#   └── C → Y   (bruitage direct du label)
#   text → Y                                   (chemin causal invariant)
#
# Trois variants selon ce qui varie entre les environnements :
#   1. varying_proxy  : a_e = bruit sur le lien C → Z   (a_e varie)
#   2. varying_gamma  : gamma_e = force de C → Y         (gamma_e varie)
#   3. varying_pc     : p_e = prévalence de C             (p_e varie)
# =============================================================================

def _apply_conf_label_bias(
    labels: np.ndarray,
    C: np.ndarray,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Applique une influence directionnelle de C sur un label binaire.

    Quand le label et C diffèrent, le label est remplacé par C avec
    probabilité gamma. Ainsi :
      - C=1 augmente P(Y=1)
      - C=0 augmente P(Y=0)
    """
    out = labels.copy()
    if gamma > 0.0:
        mismatch = out != C
        align_mask = mismatch & (rng.uniform(size=len(out)) < gamma)
        out[align_mask] = C[align_mask]
    return out


def _conf_make_env(
    texts: List[str],
    labels: np.ndarray,     # labels ORIGINAUX (avant bruitage)
    C: np.ndarray,          # confondeur (n,) valeurs 0/1
    Z: np.ndarray,          # proxy spurieux (n,) valeurs 0/1
    gamma: float,           # force avec laquelle C attire Y vers sa propre valeur
    rng: np.random.Generator,
    bert_model: str,
    max_length: int,
    device: str,
    pooling: str,
    apply_gamma: bool = True,   # False pour val/test dans varying_gamma
    conf_tokens: Optional[Dict[str, str]] = None,  # Si None, utilise define_spurious_tokens()
    finetune_bert_layers: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit un environment confounding :
            - Pousse les labels vers C (si apply_gamma)
      - Injecte un token basé sur Z (pas sur Y) dans chaque texte
      - Encode avec BERT
    Retourne (X, Y).
    """
    spurious_tokens = conf_tokens if conf_tokens is not None else define_spurious_tokens()
    labels_obs = _apply_conf_label_bias(labels, C, gamma, rng) if apply_gamma else labels.copy()

    # Z=1 → token "spam_correlated" (fire), Z=0 → token "ham_correlated" (sky)
    # On passe z comme "label" avec p_correct=1.0 → token = class_tokens[z] toujours
    rng_inj = np.random.default_rng(int(rng.integers(0, 2**31)))
    texts_mod = [
        inject_spurious_token(text, int(z), 1.0, spurious_tokens, rng_inj)
        for text, z in zip(texts, Z)
    ]
    X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    Y = labels_obs.reshape(-1, 1).astype(np.float32)
    return X, Y


def _conf_global_split(seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Charge SMS Spam et retourne (all_texts, all_labels, indices_permuted, n_total)."""
    all_texts, all_labels = load_sms_spam_dataset(seed=seed)
    n_total = len(all_texts)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    return all_texts, np.array(all_labels), indices, n_total


# =============================================================================
# Confounding variant 1 — varying_proxy
# =============================================================================
# Le paramètre a_e (bruit dans Z = C XOR Ber(a_e)) varie entre les envs.
# gamma fixe contrôle à quel point C pousse Y vers sa propre valeur.
# a ≈ 0  → Z ≈ C  → token fortement corrélé avec Y (via C)
# a = 0.5 → Z aléatoire → pas de corrélation token–Y
# a ≈ 1  → Z ≈ NOT C → token anti-corrélé avec Y → ERM piégé en OOD
# C ~ Ber(p_c_flip) est un confondeur binaire latent, indépendant du texte.
# =============================================================================

def build_envs_nlp_conf_varying_proxy(
    a_train: List[float],
    a_test: float,
    seed: int,
    p_c_flip: float = 0.25,
    gamma: float = 0.5,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    SMS Spam — confounding avec variation du proxy Z = C XOR Ber(a_e).

    DAG :  C ~ Ber(p_c_flip) → Z(a_e) → token ; C → Y (Y est poussé vers C) ; text → Y
    Variation d'env : a_e (bruit sur C→Z).
    OOD : a_test ≈ 1 → token anti-corrélé avec Y.

    Parameters
    ----------
    a_train  : List[float]  Bruit proxy par env train (ex : [0.01, 0.11]).
    a_test   : float        Bruit proxy OOD (ex : 0.99).
    p_c_flip : float        P(C=1), prévalence du confondeur binaire (défaut 0.25).
    gamma    : float        Force fixe de C→Y. Si Y!=C, le label devient C avec
                            probabilité gamma.
    """
    print("Chargement du dataset SMS Spam (confounding – varying proxy)...")
    all_texts, all_labels, indices, n_total = _conf_global_split(seed)
    n_test  = int(n_total * 0.1)
    n_val   = int(n_total * 0.1)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"Dataset : {n_total} SMS | Split 80/10/10 : Train {len(train_idx)} | Val {len(n_val if False else val_idx)} | Test {len(test_idx)}")

    n_envs = len(a_train)
    spe = len(train_idx) // n_envs
    train_envs, val_envs = [], []
    rng_master = np.random.default_rng(seed + 1)

    for i, a_e in enumerate(a_train):
        print(f"\n=== Train Env {i} (a={a_e}) ===")
        env_idx = train_idx[i * spe:(i + 1) * spe if i < n_envs - 1 else len(train_idx)]
        texts  = [all_texts[j]  for j in env_idx]
        labels = all_labels[env_idx]

        rng_e = np.random.default_rng(seed + i * 7)
        C = rng_e.binomial(1, p_c_flip, size=len(labels))
        N = rng_e.binomial(1, a_e,  size=len(labels))
        Z = np.logical_xor(C, N).astype(int)

        X, Y = _conf_make_env(texts, labels, C, Z, gamma, rng_e, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                              meta={"kind": "nlp_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                                    "split": "train", "env_id": i, "n_samples": len(X)}))

        print(f"=== Val Env {i} (a={a_e}) ===")
        val_texts  = [all_texts[j]  for j in val_idx]
        val_labels = all_labels[val_idx]
        rng_v = np.random.default_rng(seed + 5000 + i)
        Cv = rng_v.binomial(1, p_c_flip, size=len(val_labels))
        Nv = rng_v.binomial(1, a_e,  size=len(val_labels))
        Zv = np.logical_xor(Cv, Nv).astype(int)
        X_val, Y_val = _conf_make_env(val_texts, val_labels, Cv, Zv, gamma, rng_v,
                                      bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(Y_val),
                            meta={"kind": "nlp_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                                  "split": "val", "env_id": i, "n_samples": len(X_val)}))

    print(f"\n=== Test OOD (a={a_test}) ===")
    test_texts  = [all_texts[j]  for j in test_idx]
    test_labels = all_labels[test_idx]
    rng_t = np.random.default_rng(seed + 777)
    Ct = rng_t.binomial(1, p_c_flip, size=len(test_labels))
    Nt = rng_t.binomial(1, a_test, size=len(test_labels))
    Zt = np.logical_xor(Ct, Nt).astype(int)
    X_test, Y_test = _conf_make_env(test_texts, test_labels, Ct, Zt, gamma, rng_t,
                                    bert_model, max_length, device, pooling,
                                    apply_gamma=False, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(torch.from_numpy(X_test), torch.from_numpy(Y_test),
                   meta={"kind": "nlp_conf_varying_proxy", "a": a_test, "p_c_flip": p_c_flip,
                         "split": "test_ood", "n_samples": len(X_test)})

    print(f"\n✅ Confounding varying proxy — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# Confounding variant 2 — varying_gamma
# =============================================================================
# gamma_e (force de C sur Y) varie entre les envs.
# gamma ≈ 1 → Y s'aligne souvent sur C → token très corrélé avec Y
# gamma = 0 → C n'affecte pas Y → token non corrélé avec Y → ERM piégé en OOD
# a = 0 (Z = C, proxy parfait)
# =============================================================================

def build_envs_nlp_conf_varying_gamma(
    gamma_train: List[float],
    gamma_test: float,
    seed: int,
    a: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    SMS Spam — confounding avec variation de l'influence C → Y (gamma_e).

    DAG :  C → Z=C → token ; C → Y (alignement vers C de force gamma_e) ; text → Y
    Variation d'env : gamma_e (force du confondeur sur Y).
    OOD : gamma_test = 0 → token sans corrélation avec Y → ERM piégé.

    Parameters
    ----------
    gamma_train : List[float]  Force de C→Y par env train (ex : [0.8, 0.5]).
    gamma_test  : float        Force en OOD (ex : 0.0).
    a           : float        Bruit proxy C→Z (0.0 = proxy parfait).
    """
    print("Chargement du dataset SMS Spam (confounding – varying gamma)...")
    all_texts, all_labels, indices, n_total = _conf_global_split(seed)
    n_test  = int(n_total * 0.1)
    n_val   = int(n_total * 0.1)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"Dataset : {n_total} SMS | Split 80/10/10 : Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    n_envs = len(gamma_train)
    spe = len(train_idx) // n_envs
    train_envs, val_envs = [], []

    for i, g in enumerate(gamma_train):
        print(f"\n=== Train Env {i} (gamma={g}) ===")
        env_idx = train_idx[i * spe:(i + 1) * spe if i < n_envs - 1 else len(train_idx)]
        texts  = [all_texts[j]  for j in env_idx]
        labels = all_labels[env_idx]

        rng_e = np.random.default_rng(seed + i * 7)
        C = rng_e.binomial(1, 0.5, size=len(labels))
        N = rng_e.binomial(1, a,   size=len(labels))
        Z = np.logical_xor(C, N).astype(int)

        X, Y = _conf_make_env(texts, labels, C, Z, g, rng_e, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                              meta={"kind": "nlp_conf_varying_gamma", "a": a, "gamma": g,
                                    "split": "train", "env_id": i, "n_samples": len(X)}))

        print(f"=== Val Env {i} (gamma={g}) ===")
        val_texts  = [all_texts[j]  for j in val_idx]
        val_labels = all_labels[val_idx]
        rng_v = np.random.default_rng(seed + 5000 + i)
        Cv = rng_v.binomial(1, 0.5, size=len(val_labels))
        Nv = rng_v.binomial(1, a,   size=len(val_labels))
        Zv = np.logical_xor(Cv, Nv).astype(int)
        # Val : pas de flip gamma (labels propres)
        X_val, Y_val = _conf_make_env(val_texts, val_labels, Cv, Zv, g, rng_v,
                                      bert_model, max_length, device, pooling, apply_gamma=False, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(Y_val),
                            meta={"kind": "nlp_conf_varying_gamma", "a": a, "gamma": g,
                                  "split": "val", "env_id": i, "n_samples": len(X_val)}))

    print(f"\n=== Test OOD (gamma={gamma_test}) ===")
    if gamma_test == 0.0:
        print("  gamma=0 → C n'affecte pas Y → token non corrélé avec Y → ERM piégé.")
    test_texts  = [all_texts[j]  for j in test_idx]
    test_labels = all_labels[test_idx]
    rng_t = np.random.default_rng(seed + 777)
    Ct = rng_t.binomial(1, 0.5, size=len(test_labels))
    Nt = rng_t.binomial(1, a,   size=len(test_labels))
    Zt = np.logical_xor(Ct, Nt).astype(int)
    X_test, Y_test = _conf_make_env(test_texts, test_labels, Ct, Zt, gamma_test, rng_t,
                                    bert_model, max_length, device, pooling, apply_gamma=False, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(torch.from_numpy(X_test), torch.from_numpy(Y_test),
                   meta={"kind": "nlp_conf_varying_gamma", "a": a, "gamma": gamma_test,
                         "split": "test_ood", "n_samples": len(X_test)})

    print(f"\n✅ Confounding varying gamma — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# Confounding variant 3 — varying_pc
# =============================================================================
# p_e = prévalence de C (C ~ Ber(p_e)) varie entre les envs.
# p_e élevé → C=1 plus fréquent → token plus souvent corrélé avec Y
# p_e faible → C rare → token peu corrélé avec Y en test OOD
# a et gamma sont fixés.
# =============================================================================

def build_envs_nlp_conf_varying_pc(
    pc_train: List[float],
    pc_test: float,
    seed: int,
    a: float = 0.0,
    gamma: float = 0.5,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    SMS Spam — confounding avec variation de la prévalence de C (p_e).

    DAG :  C~Ber(p_e) → Z=C⊕Ber(a) → token ; C → Y (alignement vers C de force gamma) ; text → Y
    Variation d'env : p_e (prévalence du confondeur).
    OOD : pc_test non vu en train → corrélation token–Y change de régime.

    Parameters
    ----------
    pc_train : List[float]  Prévalence de C par env train (ex : [0.8, 0.5]).
    pc_test  : float        Prévalence OOD (ex : 0.1).
    a        : float        Bruit proxy (0.0 = Z=C).
    gamma    : float        Force de C→Y (fixe).
    """
    print("Chargement du dataset SMS Spam (confounding – varying pc)...")
    all_texts, all_labels, indices, n_total = _conf_global_split(seed)
    n_test  = int(n_total * 0.1)
    n_val   = int(n_total * 0.1)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"Dataset : {n_total} SMS | Split 80/10/10 : Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    n_envs = len(pc_train)
    spe = len(train_idx) // n_envs
    train_envs, val_envs = [], []

    for i, p_c in enumerate(pc_train):
        print(f"\n=== Train Env {i} (p_c={p_c}) ===")
        env_idx = train_idx[i * spe:(i + 1) * spe if i < n_envs - 1 else len(train_idx)]
        texts  = [all_texts[j]  for j in env_idx]
        labels = all_labels[env_idx]

        rng_e = np.random.default_rng(seed + i * 7)
        C = rng_e.binomial(1, p_c, size=len(labels))
        N = rng_e.binomial(1, a,   size=len(labels))
        Z = np.logical_xor(C, N).astype(int)

        X, Y = _conf_make_env(texts, labels, C, Z, gamma, rng_e, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                              meta={"kind": "nlp_conf_varying_pc", "p_c": p_c, "a": a, "gamma": gamma,
                                    "split": "train", "env_id": i, "n_samples": len(X)}))

        print(f"=== Val Env {i} (p_c={p_c}) ===")
        val_texts  = [all_texts[j]  for j in val_idx]
        val_labels = all_labels[val_idx]
        rng_v = np.random.default_rng(seed + 5000 + i)
        Cv = rng_v.binomial(1, p_c, size=len(val_labels))
        Nv = rng_v.binomial(1, a,   size=len(val_labels))
        Zv = np.logical_xor(Cv, Nv).astype(int)
        X_val, Y_val = _conf_make_env(val_texts, val_labels, Cv, Zv, gamma, rng_v,
                                      bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(Y_val),
                            meta={"kind": "nlp_conf_varying_pc", "p_c": p_c, "a": a, "gamma": gamma,
                                  "split": "val", "env_id": i, "n_samples": len(X_val)}))

    print(f"\n=== Test OOD (p_c={pc_test}) ===")
    test_texts  = [all_texts[j]  for j in test_idx]
    test_labels = all_labels[test_idx]
    rng_t = np.random.default_rng(seed + 777)
    Ct = rng_t.binomial(1, pc_test, size=len(test_labels))
    Nt = rng_t.binomial(1, a,       size=len(test_labels))
    Zt = np.logical_xor(Ct, Nt).astype(int)
    X_test, Y_test = _conf_make_env(test_texts, test_labels, Ct, Zt, gamma, rng_t,
                                    bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(torch.from_numpy(X_test), torch.from_numpy(Y_test),
                   meta={"kind": "nlp_conf_varying_pc", "p_c": pc_test, "a": a, "gamma": gamma,
                         "split": "test_ood", "n_samples": len(X_test)})

    print(f"\n✅ Confounding varying pc — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# Size-based selection helpers
# =============================================================================


# =============================================================================
# Size-based selection helpers
# =============================================================================

def compute_size_thresholds(
    texts: List[str],
    labels: List[int],
    method: str = "quartile"
) -> Tuple[float, float]:
    """
    Calcule les seuils de taille pour la sélection basée sur la longueur.
    
    Parameters
    ----------
    texts : List[str]
        Liste des textes.
    labels : List[int]
        Labels (0=ham, 1=spam).
    method : str
        Méthode de calcul ("quartile", "median", ou "auto").
    
    Returns
    -------
    t1 : float
        Seuil pour HAM (HAM sélectionné si len < t1).
    t2 : float
        Seuil pour SPAM (SPAM sélectionné si len > t2).
    
    Notes
    -----
    - method="quartile": Q1(HAM) et Q3(SPAM) → forte séparation (25% vs 75%)
    - method="median": médianes des distributions (50% vs 50%)
    - method="auto": choisit des seuils pour ~40% d'exemples typiques
    """
    lengths = [len(text) for text in texts]
    spam_lengths = [lengths[i] for i in range(len(labels)) if labels[i] == 1]
    ham_lengths = [lengths[i] for i in range(len(labels)) if labels[i] == 0]
    
    if method == "quartile":
        # Q1 pour HAM (25% plus courts) et Q3 pour SPAM (75% plus longs)
        # → Signal spurious FORT : HAM très courts, SPAM très longs
        t1 = np.percentile(ham_lengths, 25)   # Q1
        t2 = np.percentile(spam_lengths, 75)   # Q3
    elif method == "median":
        t1 = np.median(ham_lengths)
        t2 = np.median(spam_lengths)
    elif method == "auto":
        # Choisir des seuils qui créent ~40% d'exemples typiques
        t1 = np.percentile(ham_lengths, 60)
        t2 = np.percentile(spam_lengths, 40)
    elif method == "soft":
        # "Soft" selection: HAM < 40th, SPAM > 60th
        # Less extreme than quartile (25/75), keeps more examples and reduces size gap
        t1 = np.percentile(ham_lengths, 35)
        t2 = np.percentile(spam_lengths, 65)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"  Seuils calculés ({method}): HAM < {t1:.0f} chars, SPAM > {t2:.0f} chars")
    return t1, t2


def is_typical_by_size(
    text: str,
    label: int,
    t1: float,
    t2: float
) -> bool:
    """
    Vérifie si un message est "typique" selon sa taille.
    
    Parameters
    ----------
    text : str
        Texte du message.
    label : int
        Label (0=ham, 1=spam).
    t1 : float
        Seuil pour HAM.
    t2 : float
        Seuil pour SPAM.
    
    Returns
    -------
    bool
        True si typique (HAM court ou SPAM long), False sinon.
    
    Examples
    --------
    >>> is_typical_by_size("Hi", 0, 333, 457)  # HAM court
    True
    >>> is_typical_by_size("Very long message...", 1, 333, 457)  # SPAM long
    True
    >>> is_typical_by_size("Very long message...", 0, 333, 457)  # HAM long (atypique)
    False
    """
    text_len = len(text)
    
    if label == 0:  # HAM
        return text_len < t1
    else:  # SPAM (1)
        return text_len > t2


# =============================================================================
# Construction d'environnements avec sélection basée sur la taille
# =============================================================================

def build_envs_nlp_size_selection(
    train_p_select: List[float],
    seed: int,
    threshold_method: str = "quartile",
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    Construit des environnements NLP avec sélection basée sur la TAILLE des messages.
    
    **DAG**: Y → Z (taille) → S (sélection)
    
    **Amélioration** : Signal spurious FORT via quartiles + OOD extrêmes opposés
    
    **Mécanisme**:
    1. Diviser le dataset en 2 moitiés (une par environnement)
    2. Calculer seuils t1 (HAM) et t2 (SPAM) sur chaque moitié
    3. Sélectionner avec p_select si "typique" (HAM court ou SPAM long)
    4. Rejeter les "atypiques" (HAM longs, SPAM courts) → test OOD
    
    Parameters
    ----------
    train_p_select : List[float]
        Probabilités de sélection par env (ex: [0.9, 0.8]).
    seed : int
        Graine aléatoire.
    threshold_method : str
        Méthode de calcul des seuils ("median" ou "auto").
    val_frac : float
        Fraction pour validation.
    label_flip : float
        Taux de bruit symetrique sur les labels train/val. Applique apres la
        selection pour affaiblir le signal causal sans changer le mecanisme de
        selection lui-meme.
    bert_model : str
        Modèle BERT.
    max_length : int
        Longueur max séquence.
    device : str
        Device PyTorch.
    pooling : str
        Type de pooling.
    
    Returns
    -------
    train_envs : List[Env]
        Environnements d'entraînement (messages typiques).
    val_envs : List[Env]
        Environnements de validation.
    test_env : Env
        Environnement de test OOD (messages atypiques).
    """
    print("Chargement du dataset SMS Spam (mshenoda/spam-messages)...")
    all_texts, all_labels = load_sms_spam_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset chargé : {n_total} SMS")
    
    rng = np.random.default_rng(seed)
    
    # 1) SPLIT EN 2 MOITIÉS (une par environnement)
    indices = rng.permutation(n_total)
    n_envs = len(train_p_select)
    samples_per_env = n_total // n_envs
    
    print(f"\n📊 Split en {n_envs} environnements:")
    
    # Pool for EXTREME OPPOSITE examples only (not all rejected)
    extreme_opposite_texts = []
    extreme_opposite_labels = []
    
    train_envs, val_envs = [], []
    spurious_tokens = define_spurious_tokens()
    
    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        
        # Partition spécifique à cet environnement
        env_start = i * samples_per_env
        env_end = (i + 1) * samples_per_env if i < n_envs - 1 else n_total
        env_indices = indices[env_start:env_end]
        
        env_texts = [all_texts[j] for j in env_indices]
        env_labels = [all_labels[j] for j in env_indices]
        
        print(f"  {len(env_texts)} SMS dans cette partition")
        
        # 2) CALCULER SEUILS pour cet environnement
        t1, t2 = compute_size_thresholds(env_texts, env_labels, threshold_method)
        
        # 3) SÉLECTIONNER selon la taille + collecter OPPOSÉS EXTRÊMES
        selected_texts = []
        selected_labels = []
        
        for text, label in zip(env_texts, env_labels):
            text_len = len(text)
            
            if is_typical_by_size(text, label, t1, t2):
                # Typique → sélectionner avec p_select
                if rng.uniform() < p_select:
                    selected_texts.append(text)
                    selected_labels.append(label)
            else:
                # Atypique → vérifier si EXTRÊME OPPOSÉ pour OOD
                # HAM long (> t2) OU SPAM court (< t1) = opposé extrême
                if label == 0 and text_len > t2:  # HAM très long (> seuil SPAM)
                    extreme_opposite_texts.append(text)
                    extreme_opposite_labels.append(label)
                elif label == 1 and text_len < t1:  # SPAM très court (< seuil HAM)
                    extreme_opposite_texts.append(text)
                    extreme_opposite_labels.append(label)
        
        print(f"  Sélectionné: {len(selected_texts)} SMS typiques")
        
        # 4) Split train/val
        n_selected = len(selected_texts)
        n_val = int(n_selected * val_frac)
        n_train = n_selected - n_val
        
        indices_shuffled = rng.permutation(n_selected)
        train_idx = indices_shuffled[:n_train]
        val_idx = indices_shuffled[n_train:]
        
        # Train env
        train_texts = [selected_texts[j] for j in train_idx]
        train_labels = np.array([selected_labels[j] for j in train_idx])
        if label_flip > 0.0:
            rng_train_flip = np.random.default_rng(seed + 7000 + i)
            flip_mask = rng_train_flip.uniform(size=len(train_labels)) < label_flip
            train_labels[flip_mask] = 1 - train_labels[flip_mask]
        
        X_train = tokenize_and_embed_with_bert(train_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        Y_train = train_labels.reshape(-1, 1).astype(np.float32)
        
        train_envs.append(Env(
            torch.from_numpy(X_train),
            torch.from_numpy(Y_train),
            meta={"p_select": p_select, "kind": "nlp_size_train", "env_id": i,
                  "t1": t1, "t2": t2, "label_flip": label_flip,
                  "n_samples": len(X_train)}
        ))
        
        # Val env
        val_texts = [selected_texts[j] for j in val_idx]
        val_labels = np.array([selected_labels[j] for j in val_idx])
        if label_flip > 0.0:
            rng_val_flip = np.random.default_rng(seed + 8000 + i)
            flip_mask = rng_val_flip.uniform(size=len(val_labels)) < label_flip
            val_labels[flip_mask] = 1 - val_labels[flip_mask]
        
        X_val = tokenize_and_embed_with_bert(val_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        Y_val = val_labels.reshape(-1, 1).astype(np.float32)
        
        val_envs.append(Env(
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val),
            meta={"p_select": p_select, "kind": "nlp_size_val", "env_id": i,
                  "label_flip": label_flip,
                  "n_samples": len(X_val)}
        ))
    
    # 5) TEST OOD (exemples EXTRÊMES OPPOSÉS uniquement)
    print(f"\n=== Test OOD (opposés extrêmes) ===")
    print(f"  {len(extreme_opposite_texts)} SMS extrêmes:")
    print(f"    - HAM très longs (> seuil SPAM)")
    print(f"    - SPAM très courts (< seuil HAM)")
    print(f"  → Signal spurious INVERSÉ au maximum !")
    
    extreme_labels = np.array(extreme_opposite_labels)
    
    X_test = tokenize_and_embed_with_bert(extreme_opposite_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    Y_test = extreme_labels.reshape(-1, 1).astype(np.float32)
    
    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(Y_test),
        meta={"kind": "nlp_size_test_ood", "n_samples": len(X_test),
              "description": "extreme_opposite_by_size"}
    )
    
    print(f"\n✅ Environnements créés avec sélection par taille (forte séparation) !")
    print(f"   - {len(train_envs)} envs d'entraînement ({sum(e.X.shape[0] for e in train_envs)} SMS typiques)")
    print(f"   - {len(val_envs)} envs de validation ({sum(e.X.shape[0] for e in val_envs)} SMS)")
    print(f"   - 1 env de test OOD ({test_env.X.shape[0]} SMS extrêmes opposés)")
    
    return train_envs, val_envs, test_env


# =============================================================================
# SST-2 — Dataset ANTI-CAUSAL (Y → X : sentiment → texte)
# =============================================================================
#
# SST-2 (Stanford Sentiment Treebank, Socher et al. 2013) est constitué de
# critiques de films tirées de Rotten Tomatoes.  C'est un dataset ANTI-CAUSAL :
# le SENTIMENT du critique (Y) *cause* ce qu'il écrit (X_z = le texte).
#
# Contrairement à AG News (topique → article) ou SMS Spam (contenu → label),
# dans SST-2 le mécanisme stable est P(X|Y) et non P(Y|X).
#
# Deux expériences (analogues à Rosenfeld et al. 2021 "An Empirical Study of
# Invariant Risk Minimization" et Ahmad & Bhatt "Environment Agnostic IRM
# for Classification of Sequential Datasets") :
#
#   1) sst2_semi_anti_causal : injection d'un token spurieux binaire
#      DAG : Y → X_z   ET   Y → Z = token(Y, p_correct) → X total
#      Variation env : force de corrélation Z–Y (p_correct)
#
#   2) sst2_selection : sélection par lexique de sentiment fort
#      DAG : Y → Z (présence de lexique) → S (sélection)
#      Typique = critique avec mots de sentiment évidents  (train)
#      Atypique = critique subtile, sans marqueur lexical (test OOD)
# =============================================================================

SST2_CLASS_NAMES: Dict[int, str] = {0: "negative", 1: "positive"}

# Tokens spurieux pour SST-2 — directions cardinales, sémantiquement neutres
# vis-à-vis du sentiment. "north"/"south" n'ont aucune association positive/négative
# dans les corpus de pré-entraînement BERT (Wikipedia/BooksCorpus).
SST2_TOKENS: Dict[int, str] = {
    0: "north",   # négatif
    1: "south",   # positif
}

# Mots de sentiment pour le biais de sélection SST-2.
# Listes larges de mots courants dans les critiques de films —
# les critiques CONTENANT ces mots sont les "cas typiques" (train/val),
# les autres constituent le pool de test OOD.
SST2_POSITIVE_WORDS: List[str] = [
    # Jugements généraux positifs
    "good", "great", "best", "love", "loved", "enjoy", "enjoyed",
    "like", "liked", "nice", "fine", "solid", "strong", "smart",
    # Qualité narrative / artistique
    "funny", "fun", "sweet", "clever", "witty", "sharp", "rich",
    "moving", "touching", "emotional", "heartfelt", "warm", "tender",
    "beautiful", "gorgeous", "stunning", "striking", "vivid",
    "compelling", "engaging", "absorbing", "gripping", "riveting",
    "entertaining", "enjoyable", "satisfying", "rewarding", "pleasing",
    # Intensifs
    "wonderful", "excellent", "brilliant", "outstanding", "perfect",
    "masterpiece", "superb", "terrific", "magnificent", "exceptional",
    "charming", "delightful", "remarkable", "unforgettable", "inspired",
    "stunning", "breathtaking", "glorious", "spectacular", "phenomenal",
    "extraordinary", "exquisite", "thrilling", "uplifting", "hilarious",
    "powerful", "refreshing", "impressive", "interesting", "intriguing",
    # Expressions
    "must see", "must-see", "highly recommended", "worth watching",
    "well done", "well-done", "well made", "well-made",
]
SST2_NEGATIVE_WORDS: List[str] = [
    # Jugements généraux négatifs
    "bad", "worst", "poor", "weak", "flat", "dull", "slow", "bland",
    "boring", "tired", "stale", "thin", "cheap", "hollow", "empty",
    "mess", "failure", "fail", "fails", "failed",
    # Qualité narrative / artistique
    "stupid", "silly", "lame", "weak", "clumsy", "lazy", "sloppy",
    "predictable", "formulaic", "clichéd", "cliched", "derivative",
    "contrived", "forced", "unconvincing", "uninteresting", "tedious",
    "forgettable", "pointless", "aimless", "incoherent", "confusing",
    "annoying", "irritating", "painful", "unwatchable", "unbearable",
    # Intensifs
    "terrible", "awful", "horrible", "dreadful", "pathetic",
    "disappointing", "atrocious", "abysmal", "ridiculous", "laughable",
    "pretentious", "insufferable", "excruciating", "soulless", "joyless",
    "vapid", "shallow", "lifeless", "numbing",
    # Expressions
    "complete waste", "waste of time", "don't bother", "avoid",
    "fell flat", "falls flat", "doesn't work", "doesn't work",
]


# =============================================================================
# Chargement de SST-2
# =============================================================================

def load_sst2_dataset(seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Charge SST-2 depuis Hugging Face (glue / sst2).

    On combine le split « train » (67 349 exemples) et « validation »
    (872 exemples labelisés).  Le split « test » officiel n'a pas de labels
    (GLUE benchmark) et est ignoré.

    Returns
    -------
    texts  : List[str]   – phrases de critiques de films
    labels : List[int]   – 0 = négatif, 1 = positif
    """
    from datasets import concatenate_datasets

    dataset = load_dataset("glue", "sst2")
    labeled = concatenate_datasets([dataset["train"], dataset["validation"]])
    labeled = labeled.shuffle(seed=seed)

    texts  = list(labeled["sentence"])
    labels = [int(l) for l in labeled["label"]]
    return texts, labels


def load_sst2_setfit_dataset(seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Charge SST-2 depuis Hugging Face (SetFit/sst2).

    Dataset SetFit pour SST-2, avec textes plus courts que GLUE SST-2.

    Returns
    -------
    texts  : List[str]   – phrases de critiques de films
    labels : List[int]   – 0 = négatif, 1 = positif
    """
    from datasets import concatenate_datasets

    dataset = load_dataset("SetFit/sst2")
    
    # SetFit/sst2 a généralement les splits 'train' et 'test'
    if 'validation' in dataset:
        labeled = concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
    else:
        labeled = concatenate_datasets([dataset["train"], dataset["test"]])
    
    labeled = labeled.shuffle(seed=seed)

    texts  = list(labeled["text"])
    labels = [int(l) for l in labeled["label"]]
    return texts, labels


# =============================================================================
# 1) SST-2 semi anti-causal — injection de token spurieux
# =============================================================================

def build_envs_sst2_semi_anti_causal(
    train_p_correct: List[float],
    test_p_correct: float,
    seed: int,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    SST-2 — expérience semi anti-causale par injection de token spurieux.

    Chaque exemple (texte, label y) est modifié :
      - avec proba p_correct  : injecter SST2_TOKENS[y]
      - avec proba 1-p_correct : injecter SST2_TOKENS[1-y]

    La corrélation Z–Y est donc exactement p_correct.
    En test OOD (p_correct=0.0), la corrélation est totalement inversée :
    un ERM qui utilise Z obtient 0% de bonnes prédictions sur cette dimension.

    Paramètres
    ----------
    train_p_correct : List[float]
        Corrélation Z–Y par env d'entraînement (ex : [0.9, 0.8]).
    test_p_correct : float
        Corrélation en test OOD (souvent 0.0).
    seed : int
        Graine aléatoire globale.
    label_flip : float
        Fraction de labels bruités en train (affaiblit le signal causal texte→label).
    bert_model / max_length / device / pooling : voir tokenize_and_embed_with_bert.

    Retourne
    --------
    train_envs, val_envs, test_env
    """
    print("Chargement du dataset SST-2...")
    all_texts, all_labels = load_sst2_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels)
    neg = int((all_labels_arr == 0).sum())
    pos = int((all_labels_arr == 1).sum())
    print(f"Dataset chargé : {n_total} exemples  (négatif={neg}, positif={pos})")

    # ── Split global 80/10/10 ──────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test_split = int(n_total * 0.10)
    n_val_split  = int(n_total * 0.10)
    test_indices  = indices[:n_test_split]
    val_indices   = indices[n_test_split:n_test_split + n_val_split]
    train_indices = indices[n_test_split + n_val_split:]

    print(f"Split : Train {len(train_indices)} | Val {len(val_indices)} | Test {len(test_indices)}")

    n_envs = len(train_p_correct)
    samples_per_env = len(train_indices) // n_envs

    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    # ── Envs train ─────────────────────────────────────────────────────────
    for i, p_correct in enumerate(train_p_correct):
        print(f"\n=== Train Env {i} (p_correct={p_correct:.0%}) ===")
        start   = i * samples_per_env
        end     = (i + 1) * samples_per_env if i < n_envs - 1 else len(train_indices)
        env_idx = train_indices[start:end]

        texts  = [all_texts[int(j)]  for j in env_idx]
        labels = np.array([all_labels[int(j)] for j in env_idx], dtype=np.int64)

        # Label flip binaire (bruit sur le signal causal)
        if label_flip > 0.0:
            rng_flip = np.random.default_rng(seed + i * 13 + 1)
            flip_mask = rng_flip.uniform(size=len(labels)) < label_flip
            labels[flip_mask] = 1 - labels[flip_mask]

        # Injection de token spurieux
        rng_inject = np.random.default_rng(seed + i * 17 + 3)
        texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, SST2_TOKENS, rng_inject)
            for t, l in zip(texts, labels)
        ]
        n_correct = sum(
            SST2_TOKENS[int(l)] in tm.lower().split()
            for tm, l in zip(texts_mod, labels)
        )
        print(f"  Token correct : {n_correct}/{len(labels)} ({n_correct/len(labels):.1%})")

        X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        Y = labels.reshape(-1, 1).astype(np.float32)
        train_envs.append(Env(
            torch.from_numpy(X), torch.from_numpy(Y),
            meta={
                "p_correct": p_correct, "label_flip": label_flip,
                "kind": "sst2_semi_anti_causal_train",
                "n_samples": len(X), "dataset": "sst2",
            },
        ))

        # ── Val env (même p_correct, même label_flip que le train) ──────────────────
        print(f"=== Val Env {i} (p_correct={p_correct:.0%}) ===")
        val_texts  = [all_texts[int(j)]  for j in val_indices]
        val_labels = np.array([all_labels[int(j)] for j in val_indices], dtype=np.int64)

        # Même label_flip que le train pour que val ait la même distribution
        if label_flip > 0.0:
            rng_val_flip = np.random.default_rng(seed + 5000 + i + 1)
            flip_mask_val = rng_val_flip.uniform(size=len(val_labels)) < label_flip
            val_labels[flip_mask_val] = 1 - val_labels[flip_mask_val]

        rng_val = np.random.default_rng(seed + 5000 + i)
        val_texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, SST2_TOKENS, rng_val)
            for t, l in zip(val_texts, val_labels)
        ]
        X_val = tokenize_and_embed_with_bert(val_texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(
            torch.from_numpy(X_val),
            torch.from_numpy(val_labels.reshape(-1, 1).astype(np.float32)),
            meta={
                "p_correct": p_correct, "kind": "sst2_semi_anti_causal_val",
                "n_samples": len(X_val), "dataset": "sst2",
            },
        ))

    # ── Test OOD ──────────────────────────────────────────────────────────
    print(f"\n=== Test OOD SST-2 (p_correct={test_p_correct:.0%}) ===")
    if test_p_correct == 0.0:
        print("  Token toujours erroné → ERM piégé, IRM attendu robuste.")

    test_texts  = [all_texts[int(j)]  for j in test_indices]
    test_labels = np.array([all_labels[int(j)] for j in test_indices], dtype=np.int64)

    rng_test = np.random.default_rng(seed + 777)
    test_texts_mod = [
        inject_spurious_token_multiclass(t, int(l), test_p_correct, SST2_TOKENS, rng_test)
        for t, l in zip(test_texts, test_labels)
    ]
    X_test = tokenize_and_embed_with_bert(test_texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)

    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(test_labels.reshape(-1, 1).astype(np.float32)),
        meta={
            "p_correct": test_p_correct,
            "kind": "sst2_semi_anti_causal_test_ood",
            "n_samples": len(X_test), "dataset": "sst2",
        },
    )

    print(f"\n✅ SST-2 Semi Anti-Causal — Done !")
    print(f"   - {n_envs} envs train  ({sum(e.X.shape[0] for e in train_envs)} exemples)")
    print(f"   - {n_envs} envs val    ({val_envs[0].X.shape[0]} exemples / env)")
    print(f"   - 1 env test OOD  ({test_env.X.shape[0]} exemples, p_correct={test_p_correct:.0%})")

    return train_envs, val_envs, test_env


# =============================================================================
# 2) SST-2 selection — biais de sélection par lexique de sentiment fort
# =============================================================================

def is_typical_sst2(
    text: str,
    label: int,
    positive_words: List[str] = SST2_POSITIVE_WORDS,
    negative_words: List[str] = SST2_NEGATIVE_WORDS,
) -> bool:
    """
    Retourne True si la critique contient des marqueurs lexicaux forts
    correspondant à son label.

    - Positif (1) avec ≥ 1 mot de ``SST2_POSITIVE_WORDS``
    - Négatif (0) avec ≥ 1 mot de ``SST2_NEGATIVE_WORDS``

    Les critiques sans marqueur lexical fort sont des « cas subtils »
    utilisés comme test OOD.
    """
    text_lower = text.lower()
    if label == 1:
        return any(w in text_lower for w in positive_words)
    else:
        return any(w in text_lower for w in negative_words)


def is_cross_label_sst2(
    text: str,
    label: int,
    positive_words: List[str] = SST2_POSITIVE_WORDS,
    negative_words: List[str] = SST2_NEGATIVE_WORDS,
) -> bool:
    """
    Retourne True si la critique contient des marqueurs lexicaux forts du
    label OPPOSÉ au sien.

    - Positif (1) contenant ≥ 1 mot négatif fort
      (ex: « terrible acting but overall wonderful », label=1)
    - Négatif (0) contenant ≥ 1 mot positif fort
      (ex: « wonderful premise, terrible execution », label=0)

    Ces exemples sont utilisés comme test OOD adversarial : les mots-clés
    pointent dans la mauvaise direction pour ERM.
    """
    text_lower = text.lower()
    if label == 1:  # critique positive contenant des mots négatifs
        return any(w in text_lower for w in negative_words)
    else:           # critique négative contenant des mots positifs
        return any(w in text_lower for w in positive_words)


def build_envs_sst2_selection(
    train_p_select: List[float],
    seed: int = 1,
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    ood_strategy: str = "cross_label",
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    SST-2 — expérience de sélection par lexique de sentiment fort.

    IMPORTANT : La distribution des labels P(Y) reste CONSTANTE entre tous les 
    environnements d'entraînement et le test. Ceci est garanti par stratification 
    au niveau du label lors de la sélection.

    DAG : Y → Z (présence d'un marqueur lexical) → S (sélection d'entraînement)

    Mécanisme :
      - Typique (Z=1) : critique contenant ≥ 1 mot fort (ex : "wonderful")
        → sélectionné avec proba p_select dans les envs de train.
      - Test OOD selon ood_strategy :
          "atypical"    : critiques sans aucun mot fort (signal Z absent)
          "cross_label" : critiques où les mots forts *contredisent* le label
                          → ERM prédit à l'inverse du vrai label (défaut).

    Paramètres
    ----------
    train_p_select : List[float]
        Proba de garder un exemple typique par env (ex : [0.99, 0.5]).
    label_flip : float
        Taux de bruit symetrique sur les labels train/val. Applique apres la
        selection pour diminuer le signal causal sans changer la selection.
    ood_strategy : str
        Stratégie OOD : "cross_label" (défaut) ou "atypical".
    seed, val_frac, bert_model, max_length, device, pooling : cf. hab.

    Retourne
    --------
    train_envs, val_envs, test_env
    """
    print("Chargement du dataset SST-2...")
    all_texts, all_labels = load_sst2_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels)
    print(f"Dataset chargé : {n_total} critiques  |  OOD strategy : {ood_strategy}")

    rng = np.random.default_rng(seed)

    # ── Phase 1 : Classifier tous les exemples typiques vs OOD
    print("\n=== Phase 1 : Classification typique vs OOD ===")
    typical_indices_pos: List[int] = []  # Exemples typiques positifs
    typical_indices_neg: List[int] = []  # Exemples typiques négatifs
    ood_pool_texts:  List[str] = []      # Pool OOD pour construction test set
    ood_pool_labels: List[int] = []

    for idx in range(n_total):
        text, label = all_texts[idx], all_labels[idx]
        if is_typical_sst2(text, label):
            if label == 1:
                typical_indices_pos.append(idx)
            else:
                typical_indices_neg.append(idx)
        else:
            # Collecte OOD selon stratégie
            if ood_strategy == 'cross_label':
                if is_cross_label_sst2(text, label):
                    ood_pool_texts.append(text)
                    ood_pool_labels.append(label)
            else:  # 'atypical'
                if not is_cross_label_sst2(text, label):
                    ood_pool_texts.append(text)
                    ood_pool_labels.append(label)

    n_typ_pos = len(typical_indices_pos)
    n_typ_neg = len(typical_indices_neg)
    n_typical_total = n_typ_pos + n_typ_neg
    p_pos_typical = n_typ_pos / n_typical_total if n_typical_total > 0 else 0.5

    print(f"  Typiques positifs:  {n_typ_pos}")
    print(f"  Typiques négatifs:  {n_typ_neg}")
    print(f"  P(Y=1 | typique) = {p_pos_typical:.3f}")
    print(f"  OOD pool size:      {len(ood_pool_texts)}")

    # ── Phase 2 : Construire les environnements d'entraînement
    print("\n=== Phase 2 : Construction des environnements d'entraînement ===")
    
    # Shuffle les deux pools de manière déterministe
    rng_shuffle_pos = np.random.default_rng(seed + 100)
    rng_shuffle_neg = np.random.default_rng(seed + 101)
    typical_indices_pos = list(rng_shuffle_pos.permutation(typical_indices_pos))
    typical_indices_neg = list(rng_shuffle_neg.permutation(typical_indices_neg))

    n_envs = len(train_p_select)
    
    # Distribuer les exemples typiques par env, en respectant p_select et la distribution
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")

        # Chaque env reçoit une portion séquentielle équitable, stratifiée par label
        start_idx = (i * n_typ_pos) // n_envs
        end_idx = ((i + 1) * n_typ_pos) // n_envs
        start_neg = (i * n_typ_neg) // n_envs
        end_neg = ((i + 1) * n_typ_neg) // n_envs

        env_indices_pos = typical_indices_pos[start_idx:end_idx]
        env_indices_neg = typical_indices_neg[start_neg:end_neg]

        # Sélectionner avec probabilité p_select, indépendamment par label
        rng_select_pos = np.random.default_rng(seed + 2000 + i)
        rng_select_neg = np.random.default_rng(seed + 2100 + i)

        selected_indices_pos = [
            idx for idx in env_indices_pos 
            if rng_select_pos.uniform() < p_select
        ]
        selected_indices_neg = [
            idx for idx in env_indices_neg 
            if rng_select_neg.uniform() < p_select
        ]

        # Combiner et récupérer textes/labels
        selected_all_indices = selected_indices_pos + selected_indices_neg
        selected_texts = [all_texts[idx] for idx in selected_all_indices]
        selected_labels = [all_labels[idx] for idx in selected_all_indices]

        n_sel_pos = len(selected_indices_pos)
        n_sel_neg = len(selected_indices_neg)
        n_sel = n_sel_pos + n_sel_neg
        p_pos_env = n_sel_pos / n_sel if n_sel > 0 else 0.5

        print(f"  Sélectionnés positifs : {n_sel_pos} / {len(env_indices_pos)}")
        print(f"  Sélectionnés négatifs : {n_sel_neg} / {len(env_indices_neg)}")
        print(f"  P(Y=1 | sélectionné) = {p_pos_env:.3f} (attendu ~{p_pos_typical:.3f})")

        # ── Split train/val stratifié par label ──────────────────────────────
        rng_split = np.random.default_rng(seed + 3000 + i)
        
        # Shuffle les labels
        perm_idx = rng_split.permutation(n_sel)
        selected_texts = [selected_texts[j] for j in perm_idx]
        selected_labels_arr = np.array([selected_labels[j] for j in perm_idx])

        n_val = int(n_sel * val_frac)
        tr_idx = list(range(n_val, n_sel))
        va_idx = list(range(n_val))

        tr_texts  = [selected_texts[j]  for j in tr_idx]
        tr_labels = selected_labels_arr[tr_idx].astype(np.int64)
        
        if label_flip > 0.0:
            rng_train_flip = np.random.default_rng(seed + 9000 + i)
            flip_mask = rng_train_flip.uniform(size=len(tr_labels)) < label_flip
            tr_labels[flip_mask] = 1 - tr_labels[flip_mask]

        X_tr = tokenize_and_embed_with_bert(tr_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        Y_tr = tr_labels.reshape(-1, 1).astype(np.float32)
        train_envs.append(Env(
            torch.from_numpy(X_tr), torch.from_numpy(Y_tr),
            meta={"p_select": p_select, "kind": "sst2_selection_train",
                  "env_id": i, "n_samples": len(X_tr), "dataset": "sst2",
                  "label_flip": label_flip},
        ))

        va_texts  = [selected_texts[j]  for j in va_idx]
        va_labels = selected_labels_arr[va_idx].astype(np.int64)
        
        if label_flip > 0.0:
            rng_val_flip = np.random.default_rng(seed + 10000 + i)
            flip_mask = rng_val_flip.uniform(size=len(va_labels)) < label_flip
            va_labels[flip_mask] = 1 - va_labels[flip_mask]

        X_va = tokenize_and_embed_with_bert(va_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        Y_va = va_labels.reshape(-1, 1).astype(np.float32)
        val_envs.append(Env(
            torch.from_numpy(X_va), torch.from_numpy(Y_va),
            meta={"p_select": p_select, "kind": "sst2_selection_val",
                  "env_id": i, "n_samples": len(X_va), "dataset": "sst2",
                  "label_flip": label_flip},
        ))

    # ── Phase 3 : Construire le test set OOD stratifié
    print(f"\n=== Phase 3 : Construction du test set OOD ===")
    if ood_strategy == 'cross_label':
        ood_desc = "cross_label : mots forts contredisent le label (adversarial ERM)"
    else:
        ood_desc = "atypical : aucun mot fort (signal spurieux absent)"
    print(f"  Stratégie: {ood_strategy}")
    print(f"  OOD exemples collectés : {len(ood_pool_texts)}")
    
    # Stratifier le test set aussi pour maintenir P(Y=1)
    ood_labels_arr = np.array(ood_pool_labels)
    ood_pos_mask = ood_labels_arr == 1
    ood_neg_mask = ood_labels_arr == 0
    ood_pos_indices = list(np.where(ood_pos_mask)[0])
    ood_neg_indices = list(np.where(ood_neg_mask)[0])
    
    # Shuffle
    rng_test_pos = np.random.default_rng(seed + 4000)
    rng_test_neg = np.random.default_rng(seed + 4001)
    ood_pos_indices = list(rng_test_pos.permutation(ood_pos_indices))
    ood_neg_indices = list(rng_test_neg.permutation(ood_neg_indices))
    
    # Maintenir la proportion typique dans le test set
    n_test_pos_target = max(1, int(len(ood_pool_texts) * p_pos_typical))
    n_test_neg_target = len(ood_pool_texts) - n_test_pos_target
    
    ood_test_indices = (
        ood_pos_indices[:n_test_pos_target] + 
        ood_neg_indices[:n_test_neg_target]
    )
    
    # Shuffle final du test set
    rng_test_final = np.random.default_rng(seed + 4002)
    ood_test_indices = list(rng_test_final.permutation(ood_test_indices))
    
    ood_test_texts = [ood_pool_texts[j] for j in ood_test_indices]
    ood_test_labels = np.array([ood_pool_labels[j] for j in ood_test_indices])
    
    n_test_pos_actual = int((ood_test_labels == 1).sum())
    n_test_neg_actual = int((ood_test_labels == 0).sum())
    p_pos_test = n_test_pos_actual / len(ood_test_labels) if len(ood_test_labels) > 0 else 0.5
    
    print(f"  Test positifs:       {n_test_pos_actual}")
    print(f"  Test négatifs:       {n_test_neg_actual}")
    print(f"  P(Y=1 | test) = {p_pos_test:.3f} (attendu ~{p_pos_typical:.3f})")

    X_test = tokenize_and_embed_with_bert(ood_test_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    Y_test = ood_test_labels.reshape(-1, 1).astype(np.float32)

    test_env = Env(
        torch.from_numpy(X_test), torch.from_numpy(Y_test),
        meta={"kind": "sst2_selection_test_ood", "ood_strategy": ood_strategy,
              "n_samples": len(X_test), "dataset": "sst2", "description": ood_desc,
              "p_y1_target": p_pos_typical},
    )

    print(f"\n✅ SST-2 Selection — Done !")
    print(f"   - {n_envs} envs train  ({sum(e.X.shape[0] for e in train_envs)} critiques typiques)")
    print(f"   - {n_envs} envs val    ({sum(e.X.shape[0] for e in val_envs)} critiques)")
    print(f"   - 1 env test OOD  ({test_env.X.shape[0]} critiques)")
    print(f"   - Distribution des labels maintenue : P(Y=1) ≈ {p_pos_typical:.3f}")

    return train_envs, val_envs, test_env


# =============================================================================
# 3) SST-2 size selection — biais de sélection par longueur du texte
# =============================================================================
# 
# Utilise le dataset GLUE SST-2 (original avec ~68k exemples).
# 
# DAG : Y → Z (longueur) → S (sélection)
#
# Mécanisme :
#   - Typique positif  : critique longue (> Q3 positives)
#   - Typique négatif  : critique courte (< Q1 négatives)
#   - OOD adversarial  : critique longue négative (signal inversion)
#                        critique courte positive (signal inversion)
# =============================================================================

def build_envs_sst2_size_selection(
    train_p_select: List[float],
    seed: int = 1,
    val_frac: float = 0.1,
    threshold_method: str = "quartile",
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    SST-2 — expérience de sélection par longueur du texte.

    IMPORTANT : La distribution des labels P(Y) reste CONSTANTE entre tous les 
    environnements d'entraînement et le test. Ceci est garanti par stratification 
    au niveau du label lors de la sélection.

    Utilise le dataset GLUE SST-2 (original avec ~68k exemples).

    DAG : Y → Z (longueur du texte) → S (sélection d'entraînement)

    Mécanisme :
      - Typique (Z cohérent) : critique positive ET longue OU critique négative ET courte
        → sélectionné avec proba p_select dans les envs de train.
      - Test OOD (size_opposite) : critique positive ET courte OU critique négative ET longue
        → exemples où la longueur contredit la note → adversarial pour ERM.

    Parameters
    ----------
    train_p_select : List[float]
        Proba de garder un exemple typique par env (ex : [0.9, 0.8]).
    seed : int
        Graine aléatoire globale.
    val_frac : float
        Fraction pour validation.
    threshold_method : str
        Méthode de calcul des seuils ("quartile", "median", "auto", "soft").
    label_flip : float
        Taux de bruit symétrique sur les labels train/val.
    bert_model / max_length / device / pooling : cf. hab.
    
    Returns
    -------
    train_envs, val_envs, test_env
    """
    print("Chargement du dataset SST-2 (GLUE)...")
    all_texts, all_labels = load_sst2_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels)
    print(f"Dataset chargé : {n_total} critiques  |  Seuil: {threshold_method}")

    rng = np.random.default_rng(seed)

    # ── Phase 1 : Classifier tous les exemples typiques vs OOD
    print("\n=== Phase 1 : Classification typique vs size_opposite ===")
    
    # Calculer seuils de taille GLOBAUX
    t1, t2 = compute_size_thresholds(all_texts, all_labels, threshold_method)
    
    typical_indices_pos: List[int] = []  # Typiques positifs (longs)
    typical_indices_neg: List[int] = []  # Typiques négatifs (courts)
    ood_pool_texts:  List[str] = []      # Pool OOD
    ood_pool_labels: List[int] = []

    for idx in range(n_total):
        text, label = all_texts[idx], all_labels[idx]
        text_len = len(text)
        
        if label == 1 and text_len > t2:
            # Typique positif : positif ET long
            typical_indices_pos.append(idx)
        elif label == 0 and text_len < t1:
            # Typique négatif : négatif ET court
            typical_indices_neg.append(idx)
        else:
            # Atypique → OOD
            if label == 1 and text_len < t1:
                # Positif court (très opposé)
                ood_pool_texts.append(text)
                ood_pool_labels.append(label)
            elif label == 0 and text_len > t2:
                # Négatif long (très opposé)
                ood_pool_texts.append(text)
                ood_pool_labels.append(label)

    n_typ_pos = len(typical_indices_pos)
    n_typ_neg = len(typical_indices_neg)
    n_typical_total = n_typ_pos + n_typ_neg
    p_pos_typical = n_typ_pos / n_typical_total if n_typical_total > 0 else 0.5

    print(f"  Typiques positifs (longs):  {n_typ_pos}")
    print(f"  Typiques négatifs (courts):  {n_typ_neg}")
    print(f"  P(Y=1 | typique) = {p_pos_typical:.3f}")
    print(f"  OOD pool size (size_opposite):      {len(ood_pool_texts)}")

    # ── Phase 2 : Construire les environnements d'entraînement
    print("\n=== Phase 2 : Construction des environnements d'entraînement ===")
    
    # Shuffle les deux pools de manière déterministe
    rng_shuffle_pos = np.random.default_rng(seed + 100)
    rng_shuffle_neg = np.random.default_rng(seed + 101)
    typical_indices_pos = list(rng_shuffle_pos.permutation(typical_indices_pos))
    typical_indices_neg = list(rng_shuffle_neg.permutation(typical_indices_neg))

    n_envs = len(train_p_select)
    
    # Distribuer les exemples typiques par env, en respectant p_select et la distribution
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")

        # Chaque env reçoit une portion séquentielle équitable, stratifiée par label
        start_idx = (i * n_typ_pos) // n_envs
        end_idx = ((i + 1) * n_typ_pos) // n_envs
        start_neg = (i * n_typ_neg) // n_envs
        end_neg = ((i + 1) * n_typ_neg) // n_envs

        env_indices_pos = typical_indices_pos[start_idx:end_idx]
        env_indices_neg = typical_indices_neg[start_neg:end_neg]

        # Sélectionner avec probabilité p_select, indépendamment par label
        rng_select_pos = np.random.default_rng(seed + 2000 + i)
        rng_select_neg = np.random.default_rng(seed + 2100 + i)

        selected_indices_pos = [
            idx for idx in env_indices_pos 
            if rng_select_pos.uniform() < p_select
        ]
        selected_indices_neg = [
            idx for idx in env_indices_neg 
            if rng_select_neg.uniform() < p_select
        ]

        # Combiner et récupérer textes/labels
        selected_all_indices = selected_indices_pos + selected_indices_neg
        selected_texts = [all_texts[idx] for idx in selected_all_indices]
        selected_labels = [all_labels[idx] for idx in selected_all_indices]

        n_sel_pos = len(selected_indices_pos)
        n_sel_neg = len(selected_indices_neg)
        n_sel = n_sel_pos + n_sel_neg
        p_pos_env = n_sel_pos / n_sel if n_sel > 0 else 0.5

        print(f"  Sélectionnés positifs (longs) : {n_sel_pos} / {len(env_indices_pos)}")
        print(f"  Sélectionnés négatifs (courts) : {n_sel_neg} / {len(env_indices_neg)}")
        print(f"  P(Y=1 | sélectionné) = {p_pos_env:.3f} (attendu ~{p_pos_typical:.3f})")

        # ── Split train/val stratifié par label ──────────────────────────────
        rng_split = np.random.default_rng(seed + 3000 + i)
        
        # Shuffle les labels
        perm_idx = rng_split.permutation(n_sel)
        selected_texts = [selected_texts[j] for j in perm_idx]
        selected_labels_arr = np.array([selected_labels[j] for j in perm_idx])

        n_val = int(n_sel * val_frac)
        tr_idx = list(range(n_val, n_sel))
        va_idx = list(range(n_val))

        tr_texts  = [selected_texts[j]  for j in tr_idx]
        tr_labels = selected_labels_arr[tr_idx].astype(np.int64)
        
        if label_flip > 0.0:
            rng_train_flip = np.random.default_rng(seed + 9000 + i)
            flip_mask = rng_train_flip.uniform(size=len(tr_labels)) < label_flip
            tr_labels[flip_mask] = 1 - tr_labels[flip_mask]

        X_tr = tokenize_and_embed_with_bert(tr_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        Y_tr = tr_labels.reshape(-1, 1).astype(np.float32)
        train_envs.append(Env(
            torch.from_numpy(X_tr), torch.from_numpy(Y_tr),
            meta={"p_select": p_select, "kind": "sst2_size_selection_train",
                  "env_id": i, "n_samples": len(X_tr), "dataset": "sst2",
                  "label_flip": label_flip, "threshold_method": threshold_method},
        ))

        va_texts  = [selected_texts[j]  for j in va_idx]
        va_labels = selected_labels_arr[va_idx].astype(np.int64)
        
        if label_flip > 0.0:
            rng_val_flip = np.random.default_rng(seed + 10000 + i)
            flip_mask = rng_val_flip.uniform(size=len(va_labels)) < label_flip
            va_labels[flip_mask] = 1 - va_labels[flip_mask]

        X_va = tokenize_and_embed_with_bert(va_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        Y_va = va_labels.reshape(-1, 1).astype(np.float32)
        val_envs.append(Env(
            torch.from_numpy(X_va), torch.from_numpy(Y_va),
            meta={"p_select": p_select, "kind": "sst2_size_selection_val",
                  "env_id": i, "n_samples": len(X_va), "dataset": "sst2",
                  "label_flip": label_flip, "threshold_method": threshold_method},
        ))

    # ── Phase 3 : Construire le test set OOD stratifié
    print(f"\n=== Phase 3 : Construction du test set OOD ===")
    print(f"  OOD exemples collectés : {len(ood_pool_texts)}")
    
    # Stratifier le test set aussi pour maintenir P(Y=1)
    ood_labels_arr = np.array(ood_pool_labels)
    ood_pos_mask = ood_labels_arr == 1
    ood_neg_mask = ood_labels_arr == 0
    ood_pos_indices = list(np.where(ood_pos_mask)[0])
    ood_neg_indices = list(np.where(ood_neg_mask)[0])
    
    # Shuffle
    rng_test_pos = np.random.default_rng(seed + 4000)
    rng_test_neg = np.random.default_rng(seed + 4001)
    ood_pos_indices = list(rng_test_pos.permutation(ood_pos_indices))
    ood_neg_indices = list(rng_test_neg.permutation(ood_neg_indices))
    
    # Maintenir la proportion typique dans le test set
    n_test_pos_target = max(1, int(len(ood_pool_texts) * p_pos_typical))
    n_test_neg_target = len(ood_pool_texts) - n_test_pos_target
    
    ood_test_indices = (
        ood_pos_indices[:n_test_pos_target] + 
        ood_neg_indices[:n_test_neg_target]
    )
    
    # Shuffle final du test set
    rng_test_final = np.random.default_rng(seed + 4002)
    ood_test_indices = list(rng_test_final.permutation(ood_test_indices))
    
    ood_test_texts = [ood_pool_texts[j] for j in ood_test_indices]
    ood_test_labels = np.array([ood_pool_labels[j] for j in ood_test_indices])
    
    n_test_pos_actual = int((ood_test_labels == 1).sum())
    n_test_neg_actual = int((ood_test_labels == 0).sum())
    p_pos_test = n_test_pos_actual / len(ood_test_labels) if len(ood_test_labels) > 0 else 0.5
    
    print(f"  Test positifs (courts):       {n_test_pos_actual}")
    print(f"  Test négatifs (longs):       {n_test_neg_actual}")
    print(f"  P(Y=1 | test) = {p_pos_test:.3f} (attendu ~{p_pos_typical:.3f})")

    X_test = tokenize_and_embed_with_bert(ood_test_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    Y_test = ood_test_labels.reshape(-1, 1).astype(np.float32)

    test_env = Env(
        torch.from_numpy(X_test), torch.from_numpy(Y_test),
        meta={"kind": "sst2_size_selection_test_ood", "threshold_method": threshold_method,
              "n_samples": len(X_test), "dataset": "sst2", "description": "size_opposite : longueur contredit label",
              "p_y1_target": p_pos_typical},
    )

    print(f"\n✅ SST-2 Size Selection — Done !")
    print(f"   - {n_envs} envs train  ({sum(e.X.shape[0] for e in train_envs)} critiques typiques)")
    print(f"   - {n_envs} envs val    ({sum(e.X.shape[0] for e in val_envs)} critiques)")
    print(f"   - 1 env test OOD  ({test_env.X.shape[0]} critiques)")
    print(f"   - Distribution des labels maintenue : P(Y=1) ≈ {p_pos_typical:.3f}")

    return train_envs, val_envs, test_env


# =============================================================================
# AG News — Sélection basée sur les sources journalistiques (multiclasse)
# =============================================================================
#
# DAG : Y → Z (source) → S (sélection)
#
# Classes AG News :  0=World, 1=Sports, 2=Business, 3=Sci/Tech
#
# Mapping source ↔ label (choisi selon la corrélation naturelle dans AG News) :
#   World   (0) ↔ "(AFP)"       AFP = agence internationale → couvre le monde
#   Business(2) ↔ "(Reuters)"   Reuters = agence finance/éco → couvre business
#   Sci/Tech(3) ↔ "(AP)"        AP = agence US généraliste → couvre tech/sci
#   Sports  (1) ↔  rien         pas de source spurieuse → toujours typique
#
# - Typique   : l'article contient la source assignée à son label (ou Sports).
#             → sélectionné en train avec probabilité p_select.
#             → Sports plafonné pour équilibrage inter-classes.
# - Atypique  : l'article n'a pas la source de son label
#             → pool de test OOD, plafonné à n_ood_per_class exemples/classe.
# =============================================================================

# Mapping label → source spurieuse (None = pas de source assignée)
AG_NEWS_LABEL_TO_SOURCE: Dict[int, Optional[str]] = {
    0: "AFP",       # World   — AFP couvre les événements mondiaux
    1: None,        # Sports  — toujours typique, aucune source spurieuse
    2: "Reuters",   # Business — Reuters est très présent en finance/éco
    3: "AP",        # Sci/Tech — AP couvre la tech et la science
}

# Noms lisibles des classes AG News (pour logs et diagnostics)
AG_NEWS_CLASS_NAMES: Dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}


# =============================================================================
# Chargement du dataset AG News (4 classes)
# =============================================================================

def load_ag_news_dataset(seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Charge le dataset AG News depuis Hugging Face (4 classes originales).

    Dataset : fancyzhx/ag_news — 127 600 train + 7 600 test.
    Classes : 0=World, 1=Sports, 2=Business, 3=Sci/Tech.
    """
    from datasets import concatenate_datasets

    dataset = load_dataset("fancyzhx/ag_news")
    all_data = concatenate_datasets([dataset["train"], dataset["test"]])
    all_data = all_data.shuffle(seed=seed)

    texts = list(all_data["text"])
    labels = list(all_data["label"])  # déjà 0-3

    return texts, labels


# =============================================================================
# Diagnostic cross-table source × classe
# =============================================================================

def diagnose_ag_news_sources(
    texts: List[str],
    labels: List[int],
    sources: Optional[List[str]] = None,
) -> None:
    """
    Affiche une cross-table (source × classe) sur un échantillon du dataset.
    Utile pour choisir le mapping optimal source ↔ label.
    """
    if sources is None:
        sources = ["AFP", "Reuters", "AP", "cnn", "Sports Network"]

    class_names = [AG_NEWS_CLASS_NAMES[c] for c in range(4)]

    # Comptage
    counts: Dict[str, Dict[int, int]] = {s: {c: 0 for c in range(4)} for s in sources}
    total_per_class: Dict[int, int] = {c: 0 for c in range(4)}

    for text, label in zip(texts, labels):
        total_per_class[label] += 1
        for src in sources:
            if re.search(r"\(" + re.escape(src) + r"\)", text, re.IGNORECASE):
                counts[src][label] += 1

    # Affichage
    col_w = 10
    header = f"{'Source':15s}" + "".join(f"{n:>{col_w}s}" for n in class_names)
    print("\n  Cross-table : articles contenant (Source) par classe")
    print("  " + "-" * len(header))
    print("  " + header)
    print("  " + "-" * len(header))
    for src in sources:
        row = f"  ({src}):".ljust(15)
        for c in range(4):
            n = counts[src][c]
            pct = 100 * n / max(total_per_class[c], 1)
            row += f"{n:>{col_w-4}d} ({pct:4.1f}%)"[: col_w]
        print(row)
    print("  " + "-" * len(header))
    totals_row = f"  {'Total':15s}" + "".join(
        f"{total_per_class[c]:>{col_w}d}" for c in range(4)
    )
    print(totals_row)
    print()


# =============================================================================
# Détection de typicité pour AG News (multiclasse)
# =============================================================================

def is_typical_ag_news(
    text: str,
    label: int,
    label_to_source: Optional[Dict[int, Optional[str]]] = None,
) -> bool:
    """
    Indique si un article est "typique" selon la corrélation source↔label.

    - Label avec source assignée (World/Business/Sci/Tech) :
      typique ssi le texte contient ``(Source assignée)``.
    - Sports (label sans source) :
      typique ssi le texte ne contient AUCUNE source assignée aux autres classes.
      → un article Sports avec (AFP), (Reuters) ou (AP) est atypique
        (signal spurieux contradictoire → pool OOD).
    """
    if label_to_source is None:
        label_to_source = AG_NEWS_LABEL_TO_SOURCE

    source = label_to_source.get(label)

    if source is None:
        # Sports : atypique si l'article contient une source d'une autre classe
        other_sources = [s for s in label_to_source.values() if s is not None]
        for s in other_sources:
            if re.search(r"\(" + re.escape(s) + r"\)", text, re.IGNORECASE):
                return False  # source spurieuse présente → OOD
        return True

    return bool(re.search(r"\(" + re.escape(source) + r"\)", text, re.IGNORECASE))


# =============================================================================
# Construction d'environnements AG News avec selection bias (multiclasse)
# =============================================================================

def build_envs_ag_news_source_selection(
    train_p_select: List[float],
    seed: int,
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    n_ood_per_class: int = 250,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    pooling: str = "mean",
    class_dist_train: Optional[List[List[float]]] = None,
    class_dist_test: Optional[List[float]] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    Construit des environnements AG News multiclasse avec selection bias.

    **DAG** : Y → Z (source) → S (sélection)

    **Mapping source ↔ label** (corrélation naturelle dans AG News) :
      - World   (0) ↔ "(AFP)"       AFP = agence internationale
      - Business(2) ↔ "(Reuters)"   Reuters = agence finance/éco
      - Sci/Tech(3) ↔ "(AP)"        AP = agence US
      - Sports  (1) ↔ rien          toujours typique

    **Équilibrage** :
      - Sports est plafonné dans chaque env à la somme des autres classes
        sélectionnées, pour éviter qu'il domine l'entraînement.
      - L'OOD est plafonné à `n_ood_per_class` exemples par classe.

    Parameters
    ----------
    train_p_select : List[float]
        Ex : [0.9, 0.7] → env 0 garde 90% des typiques, env 1 garde 70%.
    seed : int
        Graine aléatoire.
    val_frac : float
        Fraction de validation.
    label_flip : float
        Taux de bruit multiclasse sur les labels train/val. Applique apres la
        selection, de sorte que la structure de selection reste intacte.
    n_ood_per_class : int
        Nombre maximal d'exemples OOD par classe (défaut 200).
    bert_model, max_length, device, pooling : str / int
        Config BERT.

    Returns
    -------
    train_envs, val_envs, test_env
        Labels en **torch.long** (LongTensor, valeurs 0–3) pour CrossEntropyLoss.
    """
    print("Chargement du dataset AG News (4 classes)...")
    all_texts, all_labels = load_ag_news_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset chargé : {n_total} articles (0=World, 1=Sports, 2=Business, 3=Sci/Tech)")

    # ── Diagnostic cross-table ──
    diagnose_ag_news_sources(all_texts, all_labels)

    # ── Comptage typiques ──
    counts = {k: 0 for k in range(4)}
    n_typical_per_class = {k: 0 for k in range(4)}
    for text, label in zip(all_texts, all_labels):
        counts[label] += 1
        if is_typical_ag_news(text, label):
            n_typical_per_class[label] += 1

    print("  Répartition typiques / atypiques par classe (mapping actuel) :")
    for cls, name in AG_NEWS_CLASS_NAMES.items():
        src = AG_NEWS_LABEL_TO_SOURCE[cls] or "—"
        n_typ = n_typical_per_class[cls]
        n_all = counts[cls]
        print(f"    {name:10s} (→ {src:8s}) : {n_typ:6d}/{n_all:6d} typiques "
              f"({100*n_typ/max(n_all,1):.1f}%)")

    rng = np.random.default_rng(seed)

    # Pools OOD séparés par classe (pour le plafonnement)
    ood_by_class: Dict[int, List[str]] = {c: [] for c in range(4)}

    n_envs = len(train_p_select)
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    # ───────────────── SÉLECTION PAR ENVIRONNEMENT ─────────────────────────
    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")

        # Collecter les typiques par classe séparément
        sel_by_class: Dict[int, List[str]] = {c: [] for c in range(4)}

        for text, label in zip(all_texts, all_labels):
            if is_typical_ag_news(text, label):
                if rng.uniform() < p_select:
                    sel_by_class[label].append(text)
            else:
                u = rng.uniform()
                if u < (1.0 - p_select):
                    # Atypique inclus dans le train avec proba (1-p_select)
                    # → donne une corrélation spurieuse effective de p_select dans cet env
                    sel_by_class[label].append(text)
                elif i == 0:
                    # Atypiques de env 0 non sélectionnés → test OOD
                    ood_by_class[label].append(text)

        # ── Équilibrage : Sports capé au max des autres classes individuelles ──
        # max(World, Business, Sci/Tech) pour éviter que Sports domine
        n_per_nonsports = {c: len(sel_by_class[c]) for c in [0, 2, 3]}
        cap_sports = max(n_per_nonsports.values())
        if len(sel_by_class[1]) > cap_sports:
            sports_idx = rng.choice(len(sel_by_class[1]), size=cap_sports, replace=False)
            sel_by_class[1] = [sel_by_class[1][j] for j in sports_idx]

        sel_texts:  List[str] = []
        sel_labels: List[int] = []
        for c in range(4):
            sel_texts  += sel_by_class[c]
            sel_labels += [c] * len(sel_by_class[c])

        sel_arr = np.array(sel_labels)
        dist = {AG_NEWS_CLASS_NAMES[c]: int((sel_arr == c).sum()) for c in range(4)}
        print(f"  Sélectionné (Sports ≤ max autre classe) : {len(sel_texts)} → {dist}")

        if class_dist_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            sel_texts, sel_arr = _subsample_to_class_dist(sel_texts, sel_arr, class_dist_train[i], rng_sub)
            sel_labels = sel_arr.tolist()

        # ── Split train / val ──────────────────────────────────────────────
        n_sel = len(sel_texts)
        idx = rng.permutation(n_sel)
        n_val   = int(n_sel * val_frac)
        tr_idx  = idx[n_val:]
        val_idx = idx[:n_val]

        def _make_env(indices, kind, p_sel=p_select, env_i=i):
            texts_e  = [sel_texts[j]  for j in indices]
            labels_e = np.array([sel_labels[j] for j in indices], dtype=np.int64)
            if label_flip > 0.0:
                rng_flip = np.random.default_rng(
                    seed + 11000 + env_i * 17 + (0 if kind == "train" else 1)
                )
                flip_mask = rng_flip.uniform(size=len(labels_e)) < label_flip
                for k in np.where(flip_mask)[0]:
                    others = [c for c in range(4) if c != labels_e[k]]
                    labels_e[k] = int(rng_flip.choice(others))
            X = tokenize_and_embed_with_bert(texts_e, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
            return Env(
                torch.from_numpy(X),
                torch.from_numpy(labels_e),
                meta={
                    "p_select": p_sel,
                    "kind": f"ag_news_source_selection_{kind}",
                    "env_id": env_i,
                    "n_classes": 4,
                    "label_flip": label_flip,
                    "n_samples": len(X),
                },
            )

        train_envs.append(_make_env(tr_idx,  "train"))
        val_envs.append(  _make_env(val_idx, "val"))

    # ──────────────────────── TEST OOD ─────────────────────────────────────
    print(f"\n=== Test OOD (plafonné à {n_ood_per_class} exemples/classe) ===")

    ood_texts_final:  List[str] = []
    ood_labels_final: List[int] = []

    for cls in range(4):
        pool = ood_by_class[cls]
        if not pool:
            continue  # Sports n'a pas d'OOD
        n_take = min(len(pool), n_ood_per_class)
        chosen = rng.choice(len(pool), size=n_take, replace=False)
        for j in chosen:
            ood_texts_final.append(pool[j])
            ood_labels_final.append(cls)

    ood_arr  = np.array(ood_labels_final)
    ood_dist = {AG_NEWS_CLASS_NAMES[c]: int((ood_arr == c).sum()) for c in range(4)}
    print(f"  {len(ood_texts_final)} articles OOD → {ood_dist}")

    if class_dist_test is not None:
        rng_sub_t = np.random.default_rng(seed + 22000)
        ood_texts_final, ood_arr = _subsample_to_class_dist(ood_texts_final, ood_arr, class_dist_test, rng_sub_t)
        ood_labels_final = ood_arr.tolist()

    print(f"  Signal spurieux Z absent → ERM piégé, IRM attendu robuste.")

    X_ood = tokenize_and_embed_with_bert(ood_texts_final, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(
        torch.from_numpy(X_ood),
        torch.from_numpy(ood_arr.astype(np.int64)),
        meta={
            "kind": "ag_news_source_selection_ood",
            "n_classes": 4,
            "n_samples": len(X_ood),
            "n_ood_per_class": n_ood_per_class,
            "description": "articles_without_assigned_source",
        },
    )

    print(f"\n✅ AG News multiclasse — selection bias (source-based) !")
    print(f"   - {len(train_envs)} envs train  "
          f"({sum(e.X.shape[0] for e in train_envs)} articles, équilibrés)")
    print(f"   - {len(val_envs)} envs val    "
          f"({sum(e.X.shape[0] for e in val_envs)} articles)")
    print(f"   - 1 env test OOD ({test_env.X.shape[0]} articles, ≤{n_ood_per_class}/classe)")

    return train_envs, val_envs, test_env


# =============================================================================
# AG News — Sélection par densité de vocabulaire thématique
# =============================================================================
#
# Contrairement à la sélection par agence (1–2 tokens rares dilués par le
# mean pooling sur 256 tokens), les mots-clés thématiques ci-dessous
# apparaissent ~10–30 fois par article → signal fort dans l'embedding.
#
# DAG : Y → Z (score = nb d'occurrences de mots-clés de la classe) → S
#
# "Riche"  : score ≥ Q3  → embedding saturé de termes de la classe
# "Pauvre" : score ≤ Q1  → embedding thématiquement neutre
#
# Train env i : p_select × riches  +  (1 − p_select) × pauvres
# OOD         : uniquement des articles pauvres (jamais vus en train)
#               → raccourci "densité → classe" brisé pour ERM

AG_NEWS_KEYWORD_VOCAB: Dict[int, List[str]] = {
    0: [  # World
        "war", "troops", "military", "conflict", "government", "minister", "president",
        "parliament", "forces", "attack", "killed", "soldiers", "rebel",
        "ceasefire", "diplomatic", "sanctions", "regime", "invasion", "officials", "security",
    ],
    1: [  # Sports
        "goal", "scored", "season", "match", "tournament", "championship", "coach",
        "player", "team", "league", "victory", "defeat", "points",
        "stadium", "fans", "game", "race", "ball", "title", "club",
    ],
    2: [  # Business
        "profit", "revenue", "earnings", "shares", "stock", "market", "dividend",
        "quarterly", "merger", "acquisition", "investor", "fiscal", "billion",
        "trading", "bank", "fund", "inflation", "forecast", "deficit", "growth",
    ],
    3: [  # Sci/Tech
        "software", "launch", "satellite", "processor", "network", "device", "patent",
        "technology", "research", "scientists", "computer", "internet",
        "data", "server", "mobile", "chip", "digital", "innovation", "system", "version",
    ],
}


def _count_keywords(text: str, keywords: List[str]) -> int:
    """Count total keyword occurrences in text (case-insensitive)."""
    text_lower = text.lower()
    return sum(text_lower.count(kw) for kw in keywords)


def build_envs_ag_news_keyword_selection(
    train_p_select: List[float],
    seed: int,
    threshold_method: str = "quartile",
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    n_ood_per_class: int = 250,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    pooling: str = "mean",
    class_dist_train: Optional[List[List[float]]] = None,
    class_dist_test: Optional[List[float]] = None,
    finetune_bert_layers: int = 0,
) -> Tuple[List[Env], List[Env], Env]:
    """
    AG News — sélection par densité de vocabulaire thématique.

    DAG : Y → Z (densité vocabulaire) → S (sélection d'entraînement)

    Pour chaque article de classe Y, le score Z est le nombre d'occurrences
    des mots du vocabulaire AG_NEWS_KEYWORD_VOCAB[Y] dans le texte.
    Les articles "riches" (Z ≥ Q3) ont un embedding BERT fortement teinté
    par leur thématique ; les "pauvres" (Z ≤ Q1) sont plus neutres.

    Les articles pauvres réservés pour l'OOD ne sont JAMAIS inclus en train :
    pas de fuite de données entre pool train et pool OOD.

    Parameters
    ----------
    train_p_select : List[float]
        Ex : [0.9, 0.7] → env 0 : 90 % de riches, env 1 : 70 % de riches.
    threshold_method : str
        "quartile" (Q1/Q3) ou "median" (P33/P67).
    n_ood_per_class : int
        Nombre maximal d'articles OOD par classe.
    """
    print("Chargement du dataset AG News (keyword density selection)...")
    all_texts, all_labels = load_ag_news_dataset(seed=seed)
    print(f"Dataset chargé : {len(all_texts)} articles")

    # ── Scores de densité par article ────────────────────────────────────
    scores: List[int] = [
        _count_keywords(text, AG_NEWS_KEYWORD_VOCAB[label])
        for text, label in zip(all_texts, all_labels)
    ]

    # ── Seuils par classe ─────────────────────────────────────────────────
    if threshold_method == "quartile":
        lo_pct, hi_pct = 25, 75
    elif threshold_method == "median":
        lo_pct, hi_pct = 33, 67
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method!r}")

    rich_by_class: Dict[int, List[int]] = {c: [] for c in range(4)}
    poor_by_class: Dict[int, List[int]] = {c: [] for c in range(4)}

    for c in range(4):
        c_indices = [i for i, lbl in enumerate(all_labels) if lbl == c]
        c_scores  = [scores[i] for i in c_indices]
        lo_thr = float(np.percentile(c_scores, lo_pct))
        hi_thr = float(np.percentile(c_scores, hi_pct))
        for idx, sc in zip(c_indices, c_scores):
            if sc >= hi_thr:
                rich_by_class[c].append(idx)
            elif sc <= lo_thr:
                poor_by_class[c].append(idx)
        print(f"  Classe {AG_NEWS_CLASS_NAMES[c]:10s}: "
              f"rich(≥{hi_thr:.0f}) = {len(rich_by_class[c])}, "
              f"poor(≤{lo_thr:.0f}) = {len(poor_by_class[c])}")

    rng = np.random.default_rng(seed)

    # ── Séparation pool OOD / pool train pour les pauvres ────────────────
    # Les pauvres OOD ne seront JAMAIS sélectionnés en entraînement.
    n_envs = len(train_p_select)
    poor_train_by_class: Dict[int, List[int]] = {}
    poor_ood_by_class:   Dict[int, List[int]] = {}
    for c in range(4):
        poor = poor_by_class[c][:]
        perm = rng.permutation(len(poor)).tolist()
        poor_shuf = [poor[j] for j in perm]
        n_ood_reserve = min(len(poor_shuf) // 3, n_ood_per_class * 2)
        poor_ood_by_class[c]   = poor_shuf[:n_ood_reserve]
        poor_train_by_class[c] = poor_shuf[n_ood_reserve:]

    # ── Construction des environnements d'entraînement ───────────────────
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        rng_env = np.random.default_rng(seed + 5000 + i)

        sel_texts:  List[str] = []
        sel_labels: List[int] = []

        for c in range(4):
            # Tranche round-robin pour éviter le chevauchement entre envs
            rich_env = rich_by_class[c][i::n_envs]
            poor_env = poor_train_by_class[c][i::n_envs]

            n_rich_take = int(len(rich_env) * p_select)
            n_poor_take = int(len(poor_env) * (1.0 - p_select))

            if n_rich_take > 0:
                rich_pos = rng_env.choice(len(rich_env), size=n_rich_take, replace=False)
                for j in rich_pos:
                    sel_texts.append(all_texts[rich_env[j]])
                    sel_labels.append(c)

            if n_poor_take > 0:
                poor_pos = rng_env.choice(len(poor_env), size=n_poor_take, replace=False)
                for j in poor_pos:
                    sel_texts.append(all_texts[poor_env[j]])
                    sel_labels.append(c)

        sel_arr = np.array(sel_labels)
        dist = {AG_NEWS_CLASS_NAMES[c]: int((sel_arr == c).sum()) for c in range(4)}
        print(f"  Sélectionné : {len(sel_texts)} → {dist}")

        if class_dist_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            sel_texts, sel_arr = _subsample_to_class_dist(
                sel_texts, sel_arr, class_dist_train[i], rng_sub
            )
            sel_labels = sel_arr.tolist()

        n_sel = len(sel_texts)
        idx   = rng.permutation(n_sel)
        n_val = int(n_sel * val_frac)
        tr_idx  = idx[n_val:]
        val_idx = idx[:n_val]

        def _make_env(indices, kind, p_sel=p_select, env_i=i,
                      _sel_texts=sel_texts, _sel_labels=sel_labels):
            texts_e  = [_sel_texts[j]  for j in indices]
            labels_e = np.array([_sel_labels[j] for j in indices], dtype=np.int64)
            if label_flip > 0.0:
                rng_flip = np.random.default_rng(
                    seed + 11000 + env_i * 17 + (0 if kind == "train" else 1)
                )
                flip_mask = rng_flip.uniform(size=len(labels_e)) < label_flip
                for k in np.where(flip_mask)[0]:
                    others = [c for c in range(4) if c != labels_e[k]]
                    labels_e[k] = int(rng_flip.choice(others))
            X = tokenize_and_embed_with_bert(
                texts_e, bert_model, max_length, device, pooling,
                finetune_bert_layers=finetune_bert_layers,
            )
            return Env(
                torch.from_numpy(X),
                torch.from_numpy(labels_e),
                meta={
                    "p_select": p_sel,
                    "kind": f"ag_news_keyword_selection_{kind}",
                    "env_id": env_i,
                    "n_classes": 4,
                    "label_flip": label_flip,
                    "n_samples": len(X),
                },
            )

        train_envs.append(_make_env(tr_idx,  "train"))
        val_envs.append(  _make_env(val_idx, "val"))

    # ── TEST OOD : articles pauvres du pool réservé ───────────────────────
    print(f"\n=== Test OOD (pauvres en vocabulaire, ≤{n_ood_per_class}/classe) ===")

    ood_texts_final:  List[str] = []
    ood_labels_final: List[int] = []

    for c in range(4):
        pool   = poor_ood_by_class[c]
        n_take = min(len(pool), n_ood_per_class)
        chosen = rng.choice(len(pool), size=n_take, replace=False)
        for j in chosen:
            ood_texts_final.append(all_texts[pool[j]])
            ood_labels_final.append(c)

    ood_arr  = np.array(ood_labels_final)
    ood_dist = {AG_NEWS_CLASS_NAMES[c]: int((ood_arr == c).sum()) for c in range(4)}
    print(f"  {len(ood_texts_final)} articles OOD → {ood_dist}")
    print(f"  Signal spurieux Z absent → ERM piégé, IRM attendu robuste.")

    if class_dist_test is not None:
        rng_sub_t = np.random.default_rng(seed + 22000)
        ood_texts_final, ood_arr = _subsample_to_class_dist(
            ood_texts_final, ood_arr, class_dist_test, rng_sub_t
        )

    X_ood = tokenize_and_embed_with_bert(
        ood_texts_final, bert_model, max_length, device, pooling,
        finetune_bert_layers=finetune_bert_layers,
    )
    test_env = Env(
        torch.from_numpy(X_ood),
        torch.from_numpy(ood_arr.astype(np.int64)),
        meta={
            "kind": "ag_news_keyword_selection_ood",
            "n_classes": 4,
            "n_samples": len(X_ood),
            "description": "low_keyword_density_articles",
        },
    )

    print(f"\n✅ AG News — Keyword Density Selection !")
    print(f"   - {len(train_envs)} envs train  "
          f"({sum(e.X.shape[0] for e in train_envs)} articles)")
    print(f"   - {len(val_envs)} envs val    "
          f"({sum(e.X.shape[0] for e in val_envs)} articles)")
    print(f"   - 1 env test OOD ({test_env.X.shape[0]} articles, ≤{n_ood_per_class}/classe)")

    return train_envs, val_envs, test_env


# =============================================================================
# AG News — Expérience semi anti-causale (injection de tokens, 4 classes)
# =============================================================================
#
# DAG : Y (classe réelle) → Z (token injecté) → X = BERT(texte + Z)
#
# Mécanisme :
#   Pour chaque article on injecte UN token de classe en préfixe :
#     - avec proba p_correct   : le token du vrai label
#     - avec proba 1-p_correct : un token tiré uniformément parmi les 3 autres
#
#   En train  : p_correct ∈ {0.9, 0.8} → forte corrélation Z–Y
#   En test OOD : p_correct = 0.0 → token toujours erroné → ERM piégé
#
# Tokens BERT (chacun tokenisé comme UN SEUL token, rares dans les news) :
#   alpha=World(0)  beta=Sports(1)  gamma=Business(2)  delta=Sci/Tech(3)
# =============================================================================

# Mapping classe AG News → token spurieux
# Tokens spurieux pour AG News — couleurs primaires/secondaires.
# Un token par classe, aucune association avec les catégories de news :
# "red" n'est pas une catégorie journalistique, idem pour blue/green/yellow.
# Chaque couleur est un token BERT unique avec un embedding bien défini.
AG_NEWS_TOKENS: Dict[int, str] = {
    0: "red",      # World
    1: "blue",     # Sports
    2: "green",    # Business
    3: "yellow",   # Sci/Tech
}

# Mapping fixe "classe → classe erronnée" pour l'expérience semi anti-causale
# avec token erroné unique (non tiré uniformément parmi les 3 autres classes).
# Permutation cyclique : World→Sports→Business→Sci/Tech→World
AG_NEWS_WRONG_CLASS: Dict[int, int] = {
    0: 1,  # World     → token de Sports  (blue)
    1: 2,  # Sports    → token de Business (green)
    2: 3,  # Business  → token de Sci/Tech (yellow)
    3: 0,  # Sci/Tech  → token de World   (red)
}


def inject_spurious_token_multiclass(
    text: str,
    label: int,
    p_correct: float,
    class_tokens: Dict[int, str],
    rng: np.random.Generator,
    position: str = "prefix",
    neutral_words: Optional[List[str]] = NEUTRAL_WORDS,
    wrong_class_map: Optional[Dict[int, int]] = None,
) -> str:
    """
    Injecte un token spurieux pour une classification multiclasse (N classes).

    - Avec proba p_correct   : injecte ``class_tokens[label]``.
    - Avec proba 1-p_correct :
        - Si ``wrong_class_map`` est fourni : injecte ``class_tokens[wrong_class_map[label]]``
          (token erroné fixe, une seule classe possible par classe).
        - Sinon : injecte un token tiré uniformément parmi les (N-1) autres classes.

    Par défaut, insère le token devant chaque mot neutre du texte (conservé).
    Fallback préfixe de phrase si aucun mot neutre présent.
    Passer neutral_words=None pour forcer le mode préfixe seul.
    """
    if rng.uniform() < p_correct:
        token = class_tokens[label]
    else:
        if wrong_class_map is not None:
            token = class_tokens[wrong_class_map[label]]
        else:
            others = [k for k in class_tokens if k != label]
            token = class_tokens[int(rng.choice(others))]

    if neutral_words is not None:
        prepended = _prepend_token_to_neutral_words(text, neutral_words, token)
        if prepended is not None:
            return prepended

    return f"{token} {text}" if position == "prefix" else f"{text} {token}"


def build_envs_ag_news_semi_anti_causal(
    train_p_correct: List[float],
    test_p_correct: float,
    seed: int,
    label_flip: float = 0.25,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    pooling: str = "mean",
    class_dist_train: Optional[List[List[float]]] = None,
    class_dist_test: Optional[List[float]] = None,
    finetune_bert_layers: int = 0,
    wrong_class_map: Optional[Dict[int, int]] = None) -> Tuple[List[Env], List[Env], Env]:
    """
    AG News multiclasse — expérience semi anti-causale par injection de tokens.

    DAG : Y → Z (token) → X = BERT(texte + Z)

    Un token de classe est injecté dans chaque article avant l'encodage BERT.
    La corrélation entre le token Z et le label Y est contrôlée par p_correct.
    Un modèle ERM qui exploite Z sera mis en défaut sur le test OOD (token
    toujours erroné). IRM doit apprendre à ignorer Z.

    Split global : 80 % train | 10 % val | 10 % test
    Chaque valeur de train_p_correct génère un environnement d'entraînement
    distinct (les exemples train sont répartis équitablement entre les envs).

    Parameters
    ----------
    train_p_correct : List[float]
        Corrélation Z–Y par env train (ex : [0.9, 0.8]).
    test_p_correct : float
        Corrélation en test OOD (0.0 → token toujours erroné).
    seed : int
        Graine aléatoire globale.
    label_flip : float
        Fraction de labels bruités en train (affaiblit le signal causal).
        Pour chaque label bruité, une autre classe est tirée uniformément.
    bert_model : str
        Modèle BERT Hugging Face (défaut : distilbert-base-uncased).
    max_length : int
        Longueur max de séquence BERT (256 recommandé pour AG News).
    device : str
        Device PyTorch ("cpu", "cuda", "mps").
    pooling : str
        Stratégie de pooling BERT ("mean", "cls", "max").

    Returns
    -------
    train_envs, val_envs, test_env
    """
    print("Chargement du dataset AG News...")
    all_texts, all_labels = load_ag_news_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels)
    print(f"Dataset chargé : {n_total} articles (4 classes)")
    class_dist = {AG_NEWS_CLASS_NAMES[c]: int((all_labels_arr == c).sum()) for c in range(4)}
    print(f"  Distribution : {class_dist}")

    # ── Split global 80/10/10 ──────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test_split = int(n_total * 0.1)
    n_val_split  = int(n_total * 0.1)
    test_indices  = indices[:n_test_split]
    val_indices   = indices[n_test_split:n_test_split + n_val_split]
    train_indices = indices[n_test_split + n_val_split:]

    print(f"\nSplit : Train {len(train_indices)} | Val {len(val_indices)} | Test {len(test_indices)}")

    # ── Label flip GLOBAL (avant split en envs) ────────────────────────────
    # Appliqué une seule fois sur tout le train avec un seed unique, pour éviter
    # que les envs diffèrent par leur pattern de bruitage (source d'hétérogénéité
    # inter-envs exploitable par IRM même quand p_correct est constant).
    all_labels_train = np.array([all_labels[int(j)] for j in train_indices], dtype=np.int64)
    if label_flip > 0:
        rng_flip_global = np.random.default_rng(seed + 999)
        flip_mask_global = rng_flip_global.uniform(size=len(all_labels_train)) < label_flip
        for k in np.where(flip_mask_global)[0]:
            others = [c for c in range(4) if c != all_labels_train[k]]
            all_labels_train[k] = int(rng_flip_global.choice(others))
        n_flipped = int(flip_mask_global.sum())
        print(f"  Label flip global : {n_flipped}/{len(all_labels_train)} "
              f"({n_flipped/len(all_labels_train):.1%}) exemples bruités")

    n_envs = len(train_p_correct)
    samples_per_env = len(train_indices) // n_envs

    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    # ── Envs train + val ───────────────────────────────────────────────────
    for i, p_correct in enumerate(train_p_correct):
        print(f"\n=== Train Env {i} (p_correct={p_correct:.0%}) ===")
        start   = i * samples_per_env
        end     = (i + 1) * samples_per_env if i < n_envs - 1 else len(train_indices)
        env_idx = train_indices[start:end]

        texts  = [all_texts[int(j)]  for j in env_idx]
        # Récupérer les labels depuis le tableau global pré-flippé
        labels = all_labels_train[start:end].copy()

        if class_dist_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            texts, labels = _subsample_to_class_dist(texts, labels, class_dist_train[i], rng_sub)

        # (Label flip déjà appliqué globalement avant le split)

        # Injection de token spurieux
        rng_inject = np.random.default_rng(seed + i * 17 + 3)
        texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, AG_NEWS_TOKENS, rng_inject,
                                             wrong_class_map=wrong_class_map)
            for t, l in zip(texts, labels)
        ]
        n_correct = sum(
            AG_NEWS_TOKENS[int(l)] in tm.lower().split()
            for tm, l in zip(texts_mod, labels)
        )
        print(f"  Token correct : {n_correct}/{len(labels)} ({n_correct/len(labels):.1%})")

        X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(
            torch.from_numpy(X),
            torch.from_numpy(labels),
            meta={
                "p_correct": p_correct,
                "label_flip": label_flip,
                "n_classes": 4,
                "kind": "ag_news_semi_anti_causal_train",
                "n_samples": len(X),
            },
        ))

        # ── Val env (même p_correct, même label_flip que le train) ──────────────────
        print(f"=== Val Env {i} (p_correct={p_correct:.0%}) ===")
        val_texts  = [all_texts[int(j)]  for j in val_indices]
        val_labels = np.array([all_labels[int(j)] for j in val_indices], dtype=np.int64)

        # Label flip global pour val (même seed pour tous les envs)
        if label_flip > 0:
            rng_val_flip = np.random.default_rng(seed + 5999)
            flip_mask_val = rng_val_flip.uniform(size=len(val_labels)) < label_flip
            for k in np.where(flip_mask_val)[0]:
                others = [c for c in range(4) if c != val_labels[k]]
                val_labels[k] = int(rng_val_flip.choice(others))

        if class_dist_train is not None:
            rng_sub_v = np.random.default_rng(seed + 21000 + i)
            val_texts, val_labels = _subsample_to_class_dist(val_texts, val_labels, class_dist_train[i], rng_sub_v)

        # (Label flip val déjà appliqué ci-dessus, avant class_dist pour cohérence)

        # Même label_flip multiclasse que le train → SUPPRIMÉ (flip global ci-dessus)

        rng_val = np.random.default_rng(seed + 5000 + i)
        val_texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, AG_NEWS_TOKENS, rng_val,
                                             wrong_class_map=wrong_class_map)
            for t, l in zip(val_texts, val_labels)
        ]
        X_val = tokenize_and_embed_with_bert(val_texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(
            torch.from_numpy(X_val),
            torch.from_numpy(val_labels),
            meta={
                "p_correct": p_correct,
                "n_classes": 4,
                "kind": "ag_news_semi_anti_causal_val",
                "n_samples": len(X_val),
            },
        ))

    # ── Test OOD ──────────────────────────────────────────────────────────
    print(f"\n=== Test OOD (p_correct={test_p_correct:.0%}) ===")
    if test_p_correct == 0.0:
        print("  Token toujours erroné → ERM piégé, IRM attendu robuste.")

    test_texts  = [all_texts[int(j)]  for j in test_indices]
    test_labels = np.array([all_labels[int(j)] for j in test_indices], dtype=np.int64)

    if class_dist_test is not None:
        rng_sub_t = np.random.default_rng(seed + 22000)
        test_texts, test_labels = _subsample_to_class_dist(test_texts, test_labels, class_dist_test, rng_sub_t)

    rng_test = np.random.default_rng(seed + 777)
    test_texts_mod = [
        inject_spurious_token_multiclass(t, int(l), test_p_correct, AG_NEWS_TOKENS, rng_test,
                                         wrong_class_map=wrong_class_map)
        for t, l in zip(test_texts, test_labels)
    ]
    X_test = tokenize_and_embed_with_bert(test_texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)

    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(test_labels),
        meta={
            "p_correct": test_p_correct,
            "n_classes": 4,
            "kind": "ag_news_semi_anti_causal_test_ood",
            "n_samples": len(X_test),
        },
    )

    print(f"\n✅ AG News Semi Anti-Causal — Done !")
    print(f"   - {n_envs} envs train  ({sum(e.X.shape[0] for e in train_envs)} articles)")
    print(f"   - {n_envs} envs val    ({val_envs[0].X.shape[0]} articles/env)")
    print(f"   - 1 env test OOD  ({test_env.X.shape[0]} articles, p_correct={test_p_correct:.0%})")

    return train_envs, val_envs, test_env


# =============================================================================
# Confounding — helper SST-2 (tokens binaires)
# =============================================================================
# Tokens binaires dédiés au confounding SST-2 (réutilise SST2_TOKENS).
_SST2_CONF_TOKENS: Dict[str, str] = {
    "ham_correlated":  SST2_TOKENS[0],   # "north" (label 0 = négatif)
    "spam_correlated": SST2_TOKENS[1],   # "south" (label 1 = positif)
}


# =============================================================================
# AG News — Confounding variant 1 : varying proxy (multiclasse)
# =============================================================================
# DAG : C ∈ {0,1,2,3}\{Y}  avec prob p_c_flip  (confondeur multiclasse)
#       Y_obs = (Y + C) mod 4
#       Z = C  avec prob (1-a_e),  sinon autre shift aléatoire
#       token = SHIFT_TOKENS[Z]   (red/blue/green/yellow — 4 tokens distincts)
#
# Le token code un proxy bruité du confondeur latent C (shift de classe), et
# non pas directement le label observé Y_obs. C'est donc un vrai confounder.
# Évaluation OOD : labels vrais (Y, pas Y_obs) pour mesure propre.
# Variation d'env : a_e.
# =============================================================================

_AG_NEWS_CONF_SHIFT_TOKENS: Dict[int, str] = {
    0: "red",
    1: "blue",
    2: "green",
    3: "yellow",
}


def _sample_ag_news_direct_confounder(
    labels: np.ndarray,
    p_c_flip: float,
    a_e: float,
    rng: np.random.Generator,
    n_classes: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Nouveau mécanisme de confounding multiclasse — remplacement direct.

    C ∈ {0,...,K-1} est la classe CIBLE (pas un montant de shift).
    Pour une fraction p_c_flip des exemples, C est tiré uniformément.
    Pour le reste, C = Y* (confounder inactif : pas de biais possible).

    Z est un proxy bruité de C : P(Z=C) = 1 - a_e.

    Avantage vs shift cyclique : Z ≈ C prédit directement Y_obs
    (via _apply_conf_label_bias), créant un raccourci linéaire pour ERM.
    """
    # Défaut : C = Y* (confounder inactif)
    confounder = labels.copy()
    fire_mask = rng.uniform(size=len(labels)) < p_c_flip
    if fire_mask.any():
        confounder[fire_mask] = rng.integers(0, n_classes, size=int(fire_mask.sum()))

    proxy = confounder.copy()
    noise_mask = rng.uniform(size=len(labels)) < a_e
    for k in np.where(noise_mask)[0]:
        others = [c for c in range(n_classes) if c != proxy[k]]
        proxy[k] = int(rng.choice(others))

    return confounder, proxy

def _conf_ag_news_make_env(
    texts: List[str],
    labels: np.ndarray,        # int64 vrais labels (avant bruitage)
    a_e: float,                # bruit sur Z (0=clean, 1=entièrement bruité)
    rng: np.random.Generator,
    bert_model: str,
    max_length: int,
    device: str,
    pooling: str,
    p_c_flip: float,
    apply_label_flip: bool = True,  # False pour test OOD → retourne vrais labels
    gamma: float = 0.8,        # force d'alignement Y_obs→C (0=aucun, 1=total)
    n_classes: int = 4,
    finetune_bert_layers: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construit un environnement AG News confounding (multiclasse, remplacement direct).

    Mécanisme :
      1. Échantillonne C ∈ {0,...,K-1} (classe CIBLE, uniforme) pour fraction
         p_c_flip des exemples ; C = Y* pour le reste (inactif).
      2. Construit Z comme proxy bruité de C, avec P(Z=C)=1-a_e.
         → Z ≈ C ≈ Y_obs (raccourci linéaire direct pour ERM).
      3. Injecte le token correspondant à Z dans le texte.
      4. Si apply_label_flip=True : Y_obs aligné sur C via _apply_conf_label_bias.
         Si apply_label_flip=False (test OOD) : retourne Y* (labels propres).
    """
    C, Z = _sample_ag_news_direct_confounder(labels, p_c_flip, a_e, rng, n_classes=n_classes)
    if apply_label_flip:
        rng_g = np.random.default_rng(int(rng.integers(0, 2**31)))
        Y_obs = _apply_conf_label_bias(labels, C, gamma, rng_g)
    else:
        Y_obs = labels.copy()

    # ── Étape 3 : injection du token spurieux ──────────────────────────────
    rng_inj = np.random.default_rng(int(rng.integers(0, 2**31)))
    texts_mod = [
        inject_spurious_token_multiclass(text, int(z), 1.0, _AG_NEWS_CONF_SHIFT_TOKENS, rng_inj)
        for text, z in zip(texts, Z)
    ]

    X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    return X, Y_obs.astype(np.int64)


def build_envs_ag_news_conf_varying_proxy(
    a_train: List[float],
    a_test: float,
    seed: int,
    p_c_flip: float = 0.25,
    gamma: float = 1.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    pooling: str = "mean",
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    AG News — confounding multiclasse avec variation du proxy.

        DAG : C ∈ {0,...,K-1} uniforme pour fraction p_c_flip des exemples.
            C = Y* (inactif) pour les autres.
            Y_obs ← C avec prob gamma si Y* ≠ C  (remplacement direct, pas shift).
            Z = proxy bruité de C (P(Z=C)=1-a_e) → Z ≈ C ≈ Y_obs.
            token = CONF_TOKENS[Z]

        Différence vs ancien mécanisme (shift cyclique) :
        Z encode maintenant directement la classe cible Y_obs, pas un montant de
        décalage. ERM peut donc utiliser Z seul comme raccourci linéaire.
        À l'OOD test (a_test=1.0, labels propres), Z est anti-corrélé avec C
        → ERM qui s'appuie sur Z se plante ; IRM (texte seul) tient.

    Parameters
    ----------
    a_train  : List[float]  Bruit proxy par env train (ex : [0.01, 0.1]).
    a_test   : float        Bruit proxy OOD (ex : 0.99).
    p_c_flip : float        Probabilité de flip de label (défaut 0.25).
    """
    print("Chargement du dataset AG News (confounding – varying proxy)...")
    all_texts, all_labels = load_ag_news_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels, dtype=np.int64)

    rng_split = np.random.default_rng(seed)
    indices = rng_split.permutation(n_total)
    n_test = int(n_total * 0.1)
    n_val  = int(n_total * 0.1)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"Dataset : {n_total} articles | Split 80/10/10 : "
          f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    n_envs = len(a_train)
    spe = len(train_idx) // n_envs
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    val_texts  = [all_texts[int(j)] for j in val_idx]
    val_labels = all_labels_arr[val_idx]

    for i, a_e in enumerate(a_train):
        print(f"\n=== Train Env {i} (a={a_e}) ===")
        env_idx = train_idx[i * spe:(i + 1) * spe if i < n_envs - 1 else len(train_idx)]
        texts  = [all_texts[int(j)] for j in env_idx]
        labels = all_labels_arr[env_idx]

        rng_e = np.random.default_rng(seed + i * 7)
        X, Y = _conf_ag_news_make_env(
            texts, labels, a_e, rng_e,
            bert_model, max_length, device, pooling, p_c_flip,
            apply_label_flip=True, gamma=gamma,
            finetune_bert_layers=finetune_bert_layers,
        )
        train_envs.append(Env(
            torch.from_numpy(X), torch.from_numpy(Y),
            meta={"kind": "ag_news_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                  "split": "train", "env_id": i, "n_samples": len(X), "n_classes": 4},
        ))

        print(f"=== Val Env {i} (a={a_e}) ===")
        rng_v = np.random.default_rng(seed + 5000 + i)
        X_val, Y_val = _conf_ag_news_make_env(
            val_texts, val_labels, a_e, rng_v,
            bert_model, max_length, device, pooling, p_c_flip,
            apply_label_flip=True, gamma=gamma,
            finetune_bert_layers=finetune_bert_layers,
        )
        val_envs.append(Env(
            torch.from_numpy(X_val), torch.from_numpy(Y_val),
            meta={"kind": "ag_news_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                  "split": "val", "env_id": i, "n_samples": len(X_val), "n_classes": 4},
        ))

    print(f"\n=== Test OOD (a={a_test}) ===")
    test_texts  = [all_texts[int(j)] for j in test_idx]
    test_labels = all_labels_arr[test_idx]
    rng_t = np.random.default_rng(seed + 777)
    X_test, Y_test = _conf_ag_news_make_env(
        test_texts, test_labels, a_test, rng_t,
        bert_model, max_length, device, pooling, p_c_flip,
        apply_label_flip=False, gamma=gamma,  # évaluation sur vrais labels
        finetune_bert_layers=finetune_bert_layers,
    )
    test_env = Env(
        torch.from_numpy(X_test), torch.from_numpy(Y_test),
        meta={"kind": "ag_news_conf_varying_proxy", "a": a_test, "p_c_flip": p_c_flip,
              "split": "test_ood", "n_samples": len(X_test), "n_classes": 4},
    )

    print(f"\n✅ AG News Confounding varying proxy — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# SST-2 — Confounding variant 1 : varying proxy
# =============================================================================
# Même DAG que SMS Spam confounding mais sur SST-2 (binaire, anti-causal Y→X).
# DAG : C ~ Ber(p_c_flip) → Z(a_e) = C XOR Ber(a_e) → token (north/south)
#       C → Y (flip déterministe si C=1)
#       texte → Y
# =============================================================================

def build_envs_sst2_conf_varying_proxy(
    a_train: List[float],
    a_test: float,
    seed: int,
    p_c_flip: float = 0.25,
    gamma: float = 0.5,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    SST-2 — confounding avec variation du proxy Z = C XOR Ber(a_e).

    IMPORTANT : La distribution des labels P(Y) reste CONSTANTE entre tous les 
    environnements d'entraînement et le test. Ceci est garanti par stratification 
    au niveau du label lors du partitionnement.

    Même DAG que nlp_sms_spam_conf_varying_proxy mais sur SST-2 (anti-causal).
    Les tokens spurieux binaires sont "north" (label 0) et "south" (label 1),
    cohérents avec les tokens SAC de SST-2.

    Parameters
    ----------
    a_train  : List[float]  Bruit proxy par env train (ex : [0.01, 0.1]).
    a_test   : float        Bruit proxy OOD (ex : 0.99).
    p_c_flip : float        P(C=1), prévalence du confondeur binaire (défaut 0.25).
    gamma    : float        Force fixe de C→Y. Si Y!=C, le label devient C avec
                            probabilité gamma.
    """
    print("Chargement du dataset SST-2 (confounding – varying proxy)...")
    all_texts, all_labels = load_sst2_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels, dtype=np.int64)
    
    # ── Stratification par label ──────────────────────────────────────────
    label_pos_indices = np.where(all_labels_arr == 1)[0]
    label_neg_indices = np.where(all_labels_arr == 0)[0]
    n_pos = len(label_pos_indices)
    n_neg = len(label_neg_indices)

    rng_split = np.random.default_rng(seed)
    # Shuffle chaque groupe
    label_pos_indices = rng_split.permutation(label_pos_indices)
    label_neg_indices = rng_split.permutation(label_neg_indices)
    
    # Split 80/10/10 stratifié
    n_test_pos = int(n_pos * 0.1)
    n_val_pos  = int(n_pos * 0.1)
    n_test_neg = int(n_neg * 0.1)
    n_val_neg  = int(n_neg * 0.1)
    
    test_idx  = np.concatenate([label_pos_indices[:n_test_pos], 
                                label_neg_indices[:n_test_neg]])
    val_idx   = np.concatenate([label_pos_indices[n_test_pos:n_test_pos+n_val_pos],
                                label_neg_indices[n_test_neg:n_test_neg+n_val_neg]])
    train_idx = np.concatenate([label_pos_indices[n_test_pos+n_val_pos:],
                                label_neg_indices[n_test_neg+n_val_neg:]])
    
    print(f"Dataset : {n_total} reviews | Split 80/10/10 stratifié : "
          f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")
    print(f"  Labels : {n_pos} positifs, {n_neg} négatifs")

    n_envs = len(a_train)
    spe = len(train_idx) // n_envs
    
    # ── Stratifier aussi les environnements d'entraînement ──────────────────
    train_pos = [idx for idx in train_idx if all_labels_arr[idx] == 1]
    train_neg = [idx for idx in train_idx if all_labels_arr[idx] == 0]
    
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, a_e in enumerate(a_train):
        print(f"\n=== Train Env {i} (a={a_e}) ===")
        
        # Distribuer train positifs/négatifs équitablement
        start_pos = (i * len(train_pos)) // n_envs
        end_pos = ((i + 1) * len(train_pos)) // n_envs
        start_neg = (i * len(train_neg)) // n_envs
        end_neg = ((i + 1) * len(train_neg)) // n_envs
        
        env_idx = np.concatenate([train_pos[start_pos:end_pos],
                                  train_neg[start_neg:end_neg]])
        
        texts  = [all_texts[int(j)]  for j in env_idx]
        labels = all_labels_arr[env_idx]
        p_pos_env = float((labels == 1).sum()) / len(labels)

        rng_e = np.random.default_rng(seed + i * 7)
        C = rng_e.binomial(1, p_c_flip, size=len(labels))
        N = rng_e.binomial(1, a_e,  size=len(labels))
        Z = np.logical_xor(C, N).astype(int)

        X, Y = _conf_make_env(
            texts, labels.astype(np.float32), C, Z, gamma, rng_e,
            bert_model, max_length, device, pooling,
            apply_gamma=True, conf_tokens=_SST2_CONF_TOKENS,
            finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                              meta={"kind": "sst2_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                                    "split": "train", "env_id": i, "n_samples": len(X),
                                    "p_y1": p_pos_env}))
        print(f"  P(Y=1) = {p_pos_env:.3f}")

        print(f"=== Val Env {i} (a={a_e}) ===")
        val_texts  = [all_texts[int(j)]  for j in val_idx]
        val_labels = all_labels_arr[val_idx]
        p_pos_val = float((val_labels == 1).sum()) / len(val_labels)
        
        rng_v = np.random.default_rng(seed + 5000 + i)
        Cv = rng_v.binomial(1, p_c_flip, size=len(val_labels))
        Nv = rng_v.binomial(1, a_e,  size=len(val_labels))
        Zv = np.logical_xor(Cv, Nv).astype(int)
        X_val, Y_val = _conf_make_env(
            val_texts, val_labels.astype(np.float32), Cv, Zv, gamma, rng_v,
            bert_model, max_length, device, pooling,
            apply_gamma=True, conf_tokens=_SST2_CONF_TOKENS,
            finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(Y_val),
                            meta={"kind": "sst2_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                                  "split": "val", "env_id": i, "n_samples": len(X_val),
                                  "p_y1": p_pos_val}))
        print(f"  P(Y=1) = {p_pos_val:.3f}")

    print(f"\n=== Test OOD (a={a_test}) ===")
    test_texts  = [all_texts[int(j)]  for j in test_idx]
    test_labels = all_labels_arr[test_idx]
    p_pos_test = float((test_labels == 1).sum()) / len(test_labels)
    
    rng_t = np.random.default_rng(seed + 777)
    Ct = rng_t.binomial(1, p_c_flip, size=len(test_labels))
    Nt = rng_t.binomial(1, a_test, size=len(test_labels))
    Zt = np.logical_xor(Ct, Nt).astype(int)
    X_test, Y_test = _conf_make_env(
        test_texts, test_labels.astype(np.float32), Ct, Zt, gamma, rng_t,
        bert_model, max_length, device, pooling,
        apply_gamma=False, conf_tokens=_SST2_CONF_TOKENS,
        finetune_bert_layers=finetune_bert_layers)
    test_env = Env(torch.from_numpy(X_test), torch.from_numpy(Y_test),
                   meta={"kind": "sst2_conf_varying_proxy", "a": a_test, "p_c_flip": p_c_flip,
                         "split": "test_ood", "n_samples": len(X_test),
                         "p_y1": p_pos_test})
    print(f"  P(Y=1) = {p_pos_test:.3f}")

    print(f"\n✅ SST-2 Confounding varying proxy — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    print(f"   Distribution des labels maintenue entre tous les splits")
    return train_envs, val_envs, test_env


# =============================================================================
# IMDB — Dataset ANTI-CAUSAL (Y → X : sentiment → texte)
# =============================================================================
# stanfordnlp/imdb : 25 000 critiques de films en train, 25 000 en test.
# Labels : 0 = négatif, 1 = positif.
# Comme SST-2, c'est un dataset ANTI-CAUSAL : le sentiment du critique (Y)
# cause ce qu'il écrit (X). Mais les textes sont beaucoup plus longs (~230 mots
# en moyenne vs ~20 pour SST-2), ce qui donne plus de signal causal à BERT.
# On utilise train+test pour avoir ~50 000 exemples, puis on re-splitte 80/10/10.
# Tokens spurieux : "sky" (négatif) et "fire" (positif) — mêmes tokens que SMS
# Spam (sémantiquement neutres par rapport au sentiment), injectés via
# _prepend_token_to_neutral_words pour distribuer le signal dans le texte.
# =============================================================================

IMDB_CLASS_NAMES: Dict[int, str] = {0: "negative", 1: "positive"}

# Tokens spurieux pour IMDB — mêmes que SMS Spam, sémantiquement neutres
# pour le sentiment dans les corpus de pré-entraînement BERT.
IMDB_TOKENS: Dict[int, str] = {
    0: "sky",    # négatif
    1: "fire",   # positif
}

_IMDB_CONF_TOKENS: Dict[str, str] = {
    "ham_correlated":  "sky",    # label 0 = négatif
    "spam_correlated": "fire",   # label 1 = positif
}

# Mots de sentiment pour le biais de sélection IMDB.
# Les critiques CONTENANT ces mots sont les "cas typiques" (train/val),
# les autres constituent le pool de test OOD.
# Listes identiques à SST-2 car les deux datasets sont des critiques de films.
IMDB_POSITIVE_WORDS: List[str] = [
    "good", "great", "best", "love", "loved", "enjoy", "enjoyed",
    "like", "liked", "nice", "fine", "solid", "strong", "smart",
    "funny", "fun", "sweet", "clever", "witty", "sharp", "rich",
    "moving", "touching", "emotional", "heartfelt", "warm", "tender",
    "beautiful", "gorgeous", "stunning", "striking", "vivid",
    "compelling", "engaging", "absorbing", "gripping", "riveting",
    "entertaining", "enjoyable", "satisfying", "rewarding", "pleasing",
    "wonderful", "excellent", "brilliant", "outstanding", "perfect",
    "masterpiece", "superb", "terrific", "magnificent", "exceptional",
    "charming", "delightful", "remarkable", "unforgettable", "inspired",
    "breathtaking", "glorious", "spectacular", "phenomenal",
    "extraordinary", "exquisite", "thrilling", "uplifting", "hilarious",
    "powerful", "refreshing", "impressive", "interesting", "intriguing",
    "must see", "must-see", "highly recommended", "worth watching",
    "well done", "well-done", "well made", "well-made",
]
IMDB_NEGATIVE_WORDS: List[str] = [
    "bad", "worst", "poor", "weak", "flat", "dull", "slow", "bland",
    "boring", "tired", "stale", "thin", "cheap", "hollow", "empty",
    "mess", "failure", "fail", "fails", "failed",
    "stupid", "silly", "lame", "clumsy", "lazy", "sloppy",
    "predictable", "formulaic", "clichéd", "cliched", "derivative",
    "contrived", "forced", "unconvincing", "uninteresting", "tedious",
    "forgettable", "pointless", "aimless", "incoherent", "confusing",
    "annoying", "irritating", "painful", "unwatchable", "unbearable",
    "terrible", "awful", "horrible", "dreadful", "pathetic",
    "disappointing", "atrocious", "abysmal", "ridiculous", "laughable",
    "pretentious", "insufferable", "excruciating", "soulless", "joyless",
    "vapid", "shallow", "lifeless", "numbing",
    "complete waste", "waste of time", "don't bother", "avoid",
    "fell flat", "falls flat", "doesn't work",
]


# =============================================================================
# Helper multiclasse : sous-échantillonnage pour déséquilibre de classes (N classes)
# =============================================================================

def _subsample_to_class_dist(
    texts: List[str],
    labels: np.ndarray,
    target_dist: List[float],
    rng: np.random.Generator,
    n_classes: int = 4,
) -> Tuple[List[str], np.ndarray]:
    """
    Sous-échantillonne (texts, labels) pour que P(Y=c) ≈ target_dist[c].

    Garde tous les exemples de la classe au ratio le plus contraint et
    sous-échantillonne les autres proportionnellement.

    Parameters
    ----------
    target_dist : List[float]  Distribution cible, ex [0.1, 0.5, 0.2, 0.2].
                               Doit sommer à 1.0 (normalisé sinon).
    n_classes   : int          Nombre de classes.
    """
    target_dist = np.array(target_dist, dtype=np.float64)
    target_dist /= target_dist.sum()  # normaliser

    # Indices par classe
    class_idx = {c: np.where(labels == c)[0] for c in range(n_classes)}
    class_counts = {c: len(class_idx[c]) for c in range(n_classes)}

    # Trouver la taille totale max atteignable
    # Pour chaque classe c : n_total_max = class_counts[c] / target_dist[c]
    # La contrainte la plus serrée détermine n_total
    n_total = int(min(
        class_counts[c] / target_dist[c]
        for c in range(n_classes)
        if target_dist[c] > 0
    ))

    kept_indices = []
    for c in range(n_classes):
        n_keep = int(round(n_total * target_dist[c]))
        n_keep = min(n_keep, class_counts[c])
        if n_keep > 0:
            chosen = rng.choice(class_idx[c], size=n_keep, replace=False)
            kept_indices.append(chosen)

    kept = rng.permutation(np.concatenate(kept_indices))
    texts_out  = [texts[int(j)] for j in kept]
    labels_out = labels[kept]
    actual = {c: float((labels_out == c).mean()) for c in range(n_classes)}
    print(f"  Sous-échantillonnage multiclasse → {len(kept)} samples  "
          f"(effectif : {actual}, cible : {dict(enumerate(target_dist.tolist()))})")
    return texts_out, labels_out


# =============================================================================
# IMDB — helper : sous-échantillonnage pour déséquilibre de classes
# =============================================================================

def _subsample_to_ratio(
    texts: List[str],
    labels: np.ndarray,
    target_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[str], np.ndarray]:
    """
    Sous-échantillonne (texts, labels) pour que P(Y=1) ≈ target_ratio.

    Garde tous les exemples de la classe minoritaire cible et sous-échantillonne
    la classe majoritaire.  Le résultat est permuté aléatoirement.

    Parameters
    ----------
    target_ratio : float  Fraction souhaitée de positifs (ex : 0.2 ou 0.8).
    """
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if target_ratio >= 0.5:
        # Positifs majoritaires → garder tous les positifs, sous-échantillonner négatifs
        n_neg_keep = int(n_pos * (1.0 - target_ratio) / target_ratio)
        n_neg_keep = min(n_neg_keep, n_neg)
        neg_kept = rng.choice(neg_idx, size=n_neg_keep, replace=False)
        pos_kept = pos_idx
    else:
        # Négatifs majoritaires → garder tous les négatifs, sous-échantillonner positifs
        n_pos_keep = int(n_neg * target_ratio / (1.0 - target_ratio))
        n_pos_keep = min(n_pos_keep, n_pos)
        pos_kept = rng.choice(pos_idx, size=n_pos_keep, replace=False)
        neg_kept = neg_idx

    kept = rng.permutation(np.concatenate([pos_kept, neg_kept]))
    texts_out  = [texts[int(j)]  for j in kept]
    labels_out = labels[kept]
    actual_ratio = float(labels_out.mean())
    print(f"  Sous-échantillonnage → {len(kept)} reviews  "
          f"(positifs : {actual_ratio:.1%}, cible : {target_ratio:.0%})")
    return texts_out, labels_out


# =============================================================================
# IMDB — 1) Semi anti-causal : injection de token spurieux
# =============================================================================

def build_envs_imdb_semi_anti_causal(
    train_p_correct: List[float],
    test_p_correct: float,
    seed: int,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 512,
    device: str = "cpu",
    pooling: str = "mean",
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    IMDB — expérience semi anti-causale par injection de token spurieux.

    DAG : Y → X_z  ET  Y → Z = token(Y, p_correct) → X total
    Variation d'env : force de corrélation Z–Y (p_correct).

    Parameters
    ----------
    train_p_correct  : List[float]          Corrélation Z–Y par env train (ex : [0.9, 0.8]).
    test_p_correct   : float                Corrélation en test OOD (souvent 0.0).
    label_flip       : float                Fraction de labels bruités en train.
    max_length       : int                  Max tokens BERT (défaut 512 pour textes longs).
    class_ratio_train: Optional[List[float]] Fraction de positifs par env train
                                             (ex : [0.2, 0.8]).  None = pas de rééquilibrage.
    class_ratio_test : Optional[float]      Fraction de positifs au test (ex : 0.5). None = inchangé.
    """
    print("Chargement du dataset IMDB (semi anti-causal)...")
    all_texts, all_labels = load_imdb_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels, dtype=np.int64)
    print(f"Dataset : {n_total} reviews  "
          f"(négatif={int((all_labels_arr==0).sum())}, positif={int((all_labels_arr==1).sum())})")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test = int(n_total * 0.10)
    n_val  = int(n_total * 0.10)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"Split 80/10/10 : Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    n_envs = len(train_p_correct)
    spe = len(train_idx) // n_envs
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    val_texts  = [all_texts[int(j)] for j in val_idx]
    val_labels = all_labels_arr[val_idx].copy()

    for i, p_correct in enumerate(train_p_correct):
        print(f"\n=== Train Env {i} (p_correct={p_correct:.0%}) ===")
        env_idx = train_idx[i * spe:(i + 1) * spe if i < n_envs - 1 else len(train_idx)]
        texts  = [all_texts[int(j)] for j in env_idx]
        labels = all_labels_arr[env_idx].copy()

        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            texts, labels = _subsample_to_ratio(texts, labels, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub)

        if label_flip > 0.0:
            rng_flip = np.random.default_rng(seed + i * 13 + 1)
            flip_mask = rng_flip.uniform(size=len(labels)) < label_flip
            labels[flip_mask] = 1 - labels[flip_mask]

        rng_inj = np.random.default_rng(seed + i * 17 + 3)
        texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, IMDB_TOKENS, rng_inj)
            for t, l in zip(texts, labels)
        ]
        X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        Y = labels.reshape(-1, 1).astype(np.float32)
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                              meta={"p_correct": p_correct, "label_flip": label_flip,
                                    "kind": "imdb_semi_anti_causal_train",
                                    "env_id": i, "n_samples": len(X)}))

        print(f"=== Val Env {i} (p_correct={p_correct:.0%}) ===")
        val_texts_e  = list(val_texts)
        val_labels_e = val_labels.copy()
        if class_ratio_train is not None:
            rng_sub_v = np.random.default_rng(seed + 21000 + i)
            val_texts_e, val_labels_e = _subsample_to_ratio(
                val_texts_e, val_labels_e, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub_v)
        if label_flip > 0.0:
            rng_vf = np.random.default_rng(seed + 5000 + i + 1)
            fmv = rng_vf.uniform(size=len(val_labels_e)) < label_flip
            val_labels_e[fmv] = 1 - val_labels_e[fmv]
        rng_v = np.random.default_rng(seed + 5000 + i)
        val_texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, IMDB_TOKENS, rng_v)
            for t, l in zip(val_texts_e, val_labels_e)
        ]
        X_val = tokenize_and_embed_with_bert(val_texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(torch.from_numpy(X_val),
                            torch.from_numpy(val_labels_e.reshape(-1, 1).astype(np.float32)),
                            meta={"p_correct": p_correct, "kind": "imdb_semi_anti_causal_val",
                                  "env_id": i, "n_samples": len(X_val)}))

    print(f"\n=== Test OOD (p_correct={test_p_correct:.0%}) ===")
    test_texts  = [all_texts[int(j)] for j in test_idx]
    test_labels = all_labels_arr[test_idx].copy()
    if class_ratio_test is not None:
        rng_sub_t = np.random.default_rng(seed + 22000)
        test_texts, test_labels = _subsample_to_ratio(
            test_texts, test_labels, class_ratio_test, rng_sub_t)
    rng_t = np.random.default_rng(seed + 777)
    test_texts_mod = [
        inject_spurious_token_multiclass(t, int(l), test_p_correct, IMDB_TOKENS, rng_t)
        for t, l in zip(test_texts, test_labels)
    ]
    X_test = tokenize_and_embed_with_bert(test_texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(torch.from_numpy(X_test),
                   torch.from_numpy(test_labels.reshape(-1, 1).astype(np.float32)),
                   meta={"p_correct": test_p_correct, "kind": "imdb_semi_anti_causal_test_ood",
                         "n_samples": len(X_test)})

    print(f"\n✅ IMDB Semi Anti-Causal — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# IMDB — 2) Selection bias : sélection par lexique de sentiment fort
# =============================================================================

def _is_typical_imdb(text: str, label: int) -> bool:
    """True si le texte contient ≥ 1 mot de sentiment correspondant à son label."""
    text_lower = text.lower()
    if label == 1:
        return any(w in text_lower for w in IMDB_POSITIVE_WORDS)
    else:
        return any(w in text_lower for w in IMDB_NEGATIVE_WORDS)


def _is_cross_label_imdb(text: str, label: int) -> bool:
    """True si le texte contient des marqueurs de sentiment du label OPPOSÉ."""
    text_lower = text.lower()
    if label == 1:
        return any(w in text_lower for w in IMDB_NEGATIVE_WORDS)
    else:
        return any(w in text_lower for w in IMDB_POSITIVE_WORDS)


def build_envs_imdb_selection(
    train_p_select: List[float],
    seed: int = 1,
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 512,
    device: str = "cpu",
    pooling: str = "mean",
    ood_strategy: str = "cross_label",
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    IMDB — expérience de sélection par lexique de sentiment fort.

    DAG : Y → Z (présence de lexique fort) → S (sélection d'entraînement)

    Typique  = critique contenant ≥ 1 mot de sentiment fort (train/val).
    Test OOD selon ood_strategy :
      "cross_label" : mots forts contredisent le label (adversarial pour ERM).
      "atypical"    : aucun marqueur lexical fort (signal Z absent).

    Parameters
    ----------
    train_p_select   : List[float]           Proba de garder un exemple typique par env.
    ood_strategy     : str                   "cross_label" (défaut) ou "atypical".
    max_length       : int                   Max tokens BERT (défaut 512).
    class_ratio_train: Optional[List[float]] Fraction de positifs par env (ex : [0.2, 0.8]).
    class_ratio_test : Optional[float]       Fraction de positifs au test (ex : 0.5).
    """
    print("Chargement du dataset IMDB (selection)...")
    all_texts, all_labels = load_imdb_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset : {n_total} reviews  |  OOD strategy : {ood_strategy}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_envs = len(train_p_select)
    samples_per_env = n_total // n_envs

    ood_texts:  List[str] = []
    ood_labels: List[int] = []
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        env_start  = i * samples_per_env
        env_end    = (i + 1) * samples_per_env if i < n_envs - 1 else n_total
        env_indices = indices[env_start:env_end]

        env_texts  = [all_texts[int(j)]  for j in env_indices]
        env_labels = [all_labels[int(j)] for j in env_indices]

        selected_texts:  List[str] = []
        selected_labels: List[int] = []

        for text, label in zip(env_texts, env_labels):
            if _is_typical_imdb(text, label):
                if rng.uniform() < p_select:
                    selected_texts.append(text)
                    selected_labels.append(label)
            elif i == 0:
                if ood_strategy == 'cross_label':
                    if _is_cross_label_imdb(text, label):
                        ood_texts.append(text)
                        ood_labels.append(label)
                else:  # atypical
                    if not _is_cross_label_imdb(text, label):
                        ood_texts.append(text)
                        ood_labels.append(label)

        print(f"  Sélectionné : {len(selected_texts)} reviews typiques")

        sel_texts_arr  = selected_texts
        sel_labels_arr = np.array(selected_labels)
        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            sel_texts_arr, sel_labels_arr = _subsample_to_ratio(
                sel_texts_arr, sel_labels_arr, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub)

        n_sel = len(sel_texts_arr)
        n_val = int(n_sel * val_frac)
        idx_sh = rng.permutation(n_sel)
        tr_idx, va_idx = idx_sh[n_val:], idx_sh[:n_val]

        tr_texts  = [sel_texts_arr[j]  for j in tr_idx]
        tr_labels = sel_labels_arr[tr_idx].copy()
        if label_flip > 0.0:
            rng_tf = np.random.default_rng(seed + 9000 + i)
            tr_labels[rng_tf.uniform(size=len(tr_labels)) < label_flip] ^= 1
        X_tr = tokenize_and_embed_with_bert(tr_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(torch.from_numpy(X_tr),
                              torch.from_numpy(tr_labels.reshape(-1, 1).astype(np.float32)),
                              meta={"p_select": p_select, "kind": "imdb_selection_train",
                                    "env_id": i, "n_samples": len(X_tr), "label_flip": label_flip}))

        va_texts  = [sel_texts_arr[j]  for j in va_idx]
        va_labels = sel_labels_arr[va_idx].copy()
        if label_flip > 0.0:
            rng_vf = np.random.default_rng(seed + 10000 + i)
            va_labels[rng_vf.uniform(size=len(va_labels)) < label_flip] ^= 1
        X_va = tokenize_and_embed_with_bert(va_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(torch.from_numpy(X_va),
                            torch.from_numpy(va_labels.reshape(-1, 1).astype(np.float32)),
                            meta={"p_select": p_select, "kind": "imdb_selection_val",
                                  "env_id": i, "n_samples": len(X_va), "label_flip": label_flip}))

    print(f"\n=== Test OOD ({ood_strategy}) — {len(ood_texts)} reviews ===")
    ood_texts_final  = ood_texts
    ood_labels_arr   = np.array(ood_labels)
    if class_ratio_test is not None:
        rng_sub_t = np.random.default_rng(seed + 22000)
        ood_texts_final, ood_labels_arr = _subsample_to_ratio(
            ood_texts_final, ood_labels_arr, class_ratio_test, rng_sub_t)
    X_test = tokenize_and_embed_with_bert(ood_texts_final, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(torch.from_numpy(X_test),
                   torch.from_numpy(ood_labels_arr.reshape(-1, 1).astype(np.float32)),
                   meta={"kind": "imdb_selection_test_ood", "ood_strategy": ood_strategy,
                         "n_samples": len(X_test)})

    print(f"\n✅ IMDB Selection — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {sum(e.X.shape[0] for e in val_envs)} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# IMDB — 3) Selection by size
# =============================================================================
# DAG : Y → Z (longueur de la critique) → S (sélection d'entraînement)
#
# Corrélation observée dans IMDB : les critiques négatives tendant à être plus
# courtes (frustration, rejet rapide) et les positives plus longues (enthousiasme,
# détail).  Cette corrélation est SPURIEUSE : la longueur ne cause pas le
# sentiment, mais elle y est corrélée via Y → style d'écriture.
#
# Typique : négatif COURT (< Q1 des négatifs) ou positif LONG (> Q3 des positifs)
# OOD     : extrêmes opposés — négatif très long, positif très court
# =============================================================================

def build_envs_imdb_size_selection(
    train_p_select: List[float],
    seed: int,
    threshold_method: str = "quartile",
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 512,
    device: str = "cpu",
    pooling: str = "mean",
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    IMDB — sélection par longueur de critique (corrélation Y↔Z contrôlée).

    DAG : Y → Z (longueur) → S (sélection d'entraînement)

    **Architecture de corrélation contrôlée** :
    - 4 groupes : short_pos, short_neg, long_pos, long_neg
    - Env i avec p_select :
      - p_select % des long_pos + (1-p_select) % des long_neg
        → P(Y=1 | Z=long) = p_select
      - p_select % des short_neg + (1-p_select) % des short_pos
        → P(Y=1 | Z=short) = 1 - p_select
      - Distribution Y globale ≈ stable 50/50
    
    **Entraînement** : 
      → Env 0 avec p_select=0.9 : P(Y=1|Z=long)=90%, P(Y=1|Z=short)=10%
      → Env 1 avec p_select=0.8 : P(Y=1|Z=long)=80%, P(Y=1|Z=short)=20%
    
    **Test OOD** : p_select=0.0 = corrélation COMPLÈTEMENT INVERSÉE
      → P(Y=1|Z=long)=0%, P(Y=1|Z=short)=100%
      → Labels ~50/50, mais corrélation inversée
      → IRM doit identifier que la longueur est spurieuse et l'ignorer

    Parameters
    ----------
    train_p_select   : List[float]           Force de corrélation (0-1) pour chaque env.
    threshold_method : str                   "quartile" (défaut), "median", "soft".
    val_frac         : float                 Fraction validation.
    label_flip       : float                 Taux de bruit symétrique sur les labels.
    max_length       : int                   Max tokens BERT (défaut 512).
    class_ratio_train: Optional[List[float]] Fraction de positifs par env (peut ignorer pour stabilité).
    class_ratio_test : Optional[float]       Fraction de positifs au test (~0.5).
    """
    print("Chargement du dataset IMDB (sélection par taille, corrélation contrôlée)...")
    all_texts, all_labels = load_imdb_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset : {n_total} reviews")

    # Seuils calculés sur le dataset complet (plus stable)
    t1, t2 = compute_size_thresholds(all_texts, all_labels, threshold_method)
    print(f"Seuils globaux : t1={t1}, t2={t2} (méthode={threshold_method})")

    # Catégoriser en 4 groupes : (Z_size, Y_label)
    short_pos: List[str] = []  # Z=short, Y=1
    short_neg: List[str] = []  # Z=short, Y=0
    long_pos:  List[str] = []  # Z=long, Y=1
    long_neg:  List[str] = []  # Z=long, Y=0

    for text, label in zip(all_texts, all_labels):
        text_len = len(text)
        if text_len < t1:       # court
            if label == 1:
                short_pos.append(text)
            else:
                short_neg.append(text)
        elif text_len > t2:     # long
            if label == 1:
                long_pos.append(text)
            else:
                long_neg.append(text)
        # Ignorer les moyens (t1 ≤ len ≤ t2) pour avoir des signaux clairs

    print(f"4 groupes créés :")
    print(f"  short_pos (Z=short, Y=1) : {len(short_pos)}")
    print(f"  short_neg (Z=short, Y=0) : {len(short_neg)}")
    print(f"  long_pos  (Z=long,  Y=1) : {len(long_pos)}")
    print(f"  long_neg  (Z=long,  Y=0) : {len(long_neg)}")

    # Mélanger les 4 groupes pour équilibre dans les tranches
    rng_shuffle = np.random.default_rng(seed + 5000)
    short_pos = [short_pos[j] for j in rng_shuffle.permutation(len(short_pos))]
    short_neg = [short_neg[j] for j in rng_shuffle.permutation(len(short_neg))]
    long_pos  = [long_pos[j] for j in rng_shuffle.permutation(len(long_pos))]
    long_neg  = [long_neg[j] for j in rng_shuffle.permutation(len(long_neg))]

    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        rng_env = np.random.default_rng(seed + 5000 + i)
        rng_mix = np.random.default_rng(seed + 6100 + i)

        # Répartir les 4 groupes entre envs (tranches non-chevauchantes)
        n_envs = len(train_p_select)
        sp_per_env = len(short_pos) // n_envs
        sn_per_env = len(short_neg) // n_envs
        lp_per_env = len(long_pos) // n_envs
        ln_per_env = len(long_neg) // n_envs

        # Tranches pour cet env
        sp_start, sp_end = i * sp_per_env, (i+1)*sp_per_env if i < n_envs-1 else len(short_pos)
        sn_start, sn_end = i * sn_per_env, (i+1)*sn_per_env if i < n_envs-1 else len(short_neg)
        lp_start, lp_end = i * lp_per_env, (i+1)*lp_per_env if i < n_envs-1 else len(long_pos)
        ln_start, ln_end = i * ln_per_env, (i+1)*ln_per_env if i < n_envs-1 else len(long_neg)

        env_short_pos = short_pos[sp_start:sp_end]
        env_short_neg = short_neg[sn_start:sn_end]
        env_long_pos  = long_pos[lp_start:lp_end]
        env_long_neg  = long_neg[ln_start:ln_end]

        # Mélanger selon p_select :
        # - p_select % des long_pos et (1-p_select) % des long_neg
        #   → P(Y=1 | Z=long) = p_select
        # - p_select % des short_neg et (1-p_select) % des short_pos
        #   → P(Y=1 | Z=short) = 1 - p_select
        
        n_lp_keep = int(len(env_long_pos) * p_select)
        n_ln_keep = int(len(env_long_neg) * (1 - p_select))
        n_sp_keep = int(len(env_short_pos) * (1 - p_select))
        n_sn_keep = int(len(env_short_neg) * p_select)

        lp_idx = rng_mix.choice(len(env_long_pos),  size=n_lp_keep, replace=False)
        ln_idx = rng_mix.choice(len(env_long_neg),  size=n_ln_keep, replace=False)
        sp_idx = rng_mix.choice(len(env_short_pos), size=n_sp_keep, replace=False)
        sn_idx = rng_mix.choice(len(env_short_neg), size=n_sn_keep, replace=False)

        # Construire la sélection complète
        selected_texts = [env_long_pos[j]  for j in lp_idx] + \
                        [env_long_neg[j]  for j in ln_idx] + \
                        [env_short_pos[j] for j in sp_idx] + \
                        [env_short_neg[j] for j in sn_idx]
        
        selected_labels = [1] * n_lp_keep + [0] * n_ln_keep + \
                         [1] * n_sp_keep + [0] * n_sn_keep

        # Calcul de la vraie corrélation Y↔Z
        # P(Y=1 | Z=long) et P(Y=1 | Z=short)
        p_pos_given_long = (n_lp_keep) / (n_lp_keep + n_ln_keep) if (n_lp_keep + n_ln_keep) > 0 else 0.5
        p_pos_given_short = (n_sp_keep) / (n_sp_keep + n_sn_keep) if (n_sp_keep + n_sn_keep) > 0 else 0.5

        print(f"  Sélectionné : {len(selected_texts)} reviews")
        print(f"    long_pos: {n_lp_keep} | long_neg: {n_ln_keep}")
        print(f"    short_pos: {n_sp_keep} | short_neg: {n_sn_keep}")
        print(f"  P(Y=1|Z=long)={p_pos_given_long:.1%}  (cible: {p_select:.0%})")
        print(f"  P(Y=1|Z=short)={p_pos_given_short:.1%} (cible: {1-p_select:.0%})")
        print(f"  Distribution Y globale: {np.mean(selected_labels):.1%} positifs")

        sel_texts_sz = selected_texts
        sel_labels_sz = np.array(selected_labels)
        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            sel_texts_sz, sel_labels_sz = _subsample_to_ratio(
                sel_texts_sz, sel_labels_sz, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub)

        n_sel = len(sel_texts_sz)
        n_val = int(n_sel * val_frac)
        idx_sh = rng_env.permutation(n_sel)
        tr_idx, va_idx = idx_sh[n_val:], idx_sh[:n_val]

        tr_texts  = [sel_texts_sz[j]  for j in tr_idx]
        tr_labels = sel_labels_sz[tr_idx].copy()
        if label_flip > 0.0:
            rng_tf = np.random.default_rng(seed + 7000 + i)
            tr_labels[rng_tf.uniform(size=len(tr_labels)) < label_flip] ^= 1
        X_tr = tokenize_and_embed_with_bert(tr_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(torch.from_numpy(X_tr),
                              torch.from_numpy(tr_labels.reshape(-1, 1).astype(np.float32)),
                              meta={"p_select": p_select, "kind": "imdb_size_selection_train",
                                    "env_id": i, "t1": t1, "t2": t2,
                                    "label_flip": label_flip, "n_samples": len(X_tr)}))

        va_texts  = [sel_texts_sz[j]  for j in va_idx]
        va_labels = sel_labels_sz[va_idx].copy()
        if label_flip > 0.0:
            rng_vf = np.random.default_rng(seed + 8000 + i)
            va_labels[rng_vf.uniform(size=len(va_labels)) < label_flip] ^= 1
        X_va = tokenize_and_embed_with_bert(va_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(torch.from_numpy(X_va),
                            torch.from_numpy(va_labels.reshape(-1, 1).astype(np.float32)),
                            meta={"p_select": p_select, "kind": "imdb_size_selection_val",
                                  "env_id": i, "label_flip": label_flip, "n_samples": len(X_va)}))

    print(f"\n=== Test OOD (p_select=0.0 = corrélation INVERSÉE) ===")
    
    # OOD : 100% inversé
    # - 0% long_pos + 100% long_neg  → P(Y=1|Z=long) = 0%
    # - 100% short_pos + 0% short_neg → P(Y=1|Z=short) = 100%
    
    rng_ood = np.random.default_rng(seed + 25000)
    n_ood_long = min(len(long_neg), 2000)  # limiter la taille
    n_ood_short = min(len(short_pos), 2000)
    
    ood_long_idx = rng_ood.choice(len(long_neg), size=n_ood_long, replace=False)
    ood_short_idx = rng_ood.choice(len(short_pos), size=n_ood_short, replace=False)
    
    ood_texts_final = [long_neg[j] for j in ood_long_idx] + \
                      [short_pos[j] for j in ood_short_idx]
    ood_labels_arr = np.array([0] * n_ood_long + [1] * n_ood_short)
    
    # Shuffle
    perm = rng_ood.permutation(len(ood_texts_final))
    ood_texts_final = [ood_texts_final[j] for j in perm]
    ood_labels_arr = ood_labels_arr[perm]
    
    print(f"  Composition : {n_ood_long} long_neg + {n_ood_short} short_pos = {len(ood_texts_final)} total")
    print(f"  P(Y=1|Z=long)=0%  P(Y=1|Z=short)=100% (INVERSÉ)")
    print(f"  Distribution Y: {ood_labels_arr.mean():.1%} positifs")
    
    if class_ratio_test is not None and abs(class_ratio_test - 0.5) > 1e-6:
        rng_sub_t = np.random.default_rng(seed + 22000)
        ood_texts_final, ood_labels_arr = _subsample_to_ratio(
            ood_texts_final, ood_labels_arr, class_ratio_test, rng_sub_t)
    
    X_test = tokenize_and_embed_with_bert(ood_texts_final, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(torch.from_numpy(X_test),
                   torch.from_numpy(ood_labels_arr.reshape(-1, 1).astype(np.float32)),
                   meta={"kind": "imdb_size_selection_test_ood",
                         "n_samples": len(X_test), "description": "inverted_size_correlation"})

    print(f"\n✅ IMDB Size Selection — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {sum(e.X.shape[0] for e in val_envs)} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# IMDB — 4) Selection by HTML <br> tags
# =============================================================================
# DAG : Y → Z (présence de balises <br>) → S (sélection d'entraînement)
#
# Observation dans IMDB brut : certaines critiques contiennent des balises HTML
# <br /> (line breaks), d'autres non.  On utilise cette caractéristique de
# formatage comme corrélation SPURIEUSE avec le sentiment.
#
# Typique : négatif AVEC <br>, positif SANS <br>
# OOD     : relation inversée — négatif SANS <br>, positif AVEC <br>
# =============================================================================

def _has_br_tag(text: str) -> bool:
    """Vérifie si un texte contient au moins une balise <br> (ou <br />)."""
    return "<br" in text.lower()


def _count_br_tags(text: str) -> int:
    """Compte le nombre de balises <br> (ou <br />) dans le texte."""
    text_lower = text.lower()
    count = 0
    idx = 0
    while True:
        pos = text_lower.find("<br", idx)
        if pos == -1:
            break
        count += 1
        idx = pos + 1
    return count


def build_envs_imdb_br_selection(
    train_p_select: List[float],
    seed: int,
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 512,
    device: str = "cpu",
    pooling: str = "mean",
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0,
    max_length_chars: Optional[int] = None) -> Tuple[List[Env], List[Env], Env]:
    """
    IMDB — sélection par nombre de balises HTML <br> (corrélation Y↔Z contrôlée).

    DAG : Y → Z (nombre <br>) → S (sélection d'entraînement)

    **Critère de sélection renforcé** :
    - Groupe "many BR" : exemples avec **> 4 balises <br>**
    - Groupe "no BR"   : exemples avec **0 balise <br>**
    - Autres (1-3 BR)  : **IGNORÉS** (pour renforcer le signal)

    **Architecture de corrélation contrôlée** :
    - 4 groupes : br_many_pos, br_many_neg, no_br_pos, no_br_neg
    - Env i avec p_select :
      - p_select % des br_many_pos + (1-p_select) % des br_many_neg
        → P(Y=1 | Z=many_br) = p_select
      - p_select % des no_br_neg + (1-p_select) % des no_br_pos
        → P(Y=1 | Z=no_br) = 1 - p_select
      - Distribution Y globale ≈ stable 50/50
    
    **Entraînement** : 
      → Env 0 avec p_select=0.9 : P(Y=1|Z=many_br)=90%, P(Y=1|Z=no_br)=10%
      → Env 1 avec p_select=0.8 : P(Y=1|Z=many_br)=80%, P(Y=1|Z=no_br)=20%
    
    **Test OOD** : p_select=0.0 = corrélation COMPLÈTEMENT INVERSÉE
      → P(Y=1|Z=many_br)=0%, P(Y=1|Z=no_br)=100%
      → Labels ~50/50, mais corrélation BR inversée
      → IRM doit identifier que les BR-tags sont spurieux et les ignorer

    Parameters
    ----------
    train_p_select   : List[float]           Force de corrélation (0-1) pour chaque env.
    val_frac         : float                 Fraction validation.
    label_flip       : float                 Taux de bruit symétrique sur les labels.
    max_length       : int                   Max tokens BERT (défaut 512).
    class_ratio_train: Optional[List[float]] Fraction de positifs par env (peut ignorer).
    class_ratio_test : Optional[float]       Fraction de positifs au test (~0.5).
    max_length_chars : Optional[int]         Max longueur texte en caractères (None = pas de limite).
    """
    print("Chargement du dataset IMDB (sélection par balises <br>, corrélation contrôlée)...")
    all_texts, all_labels = load_imdb_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset : {n_total} reviews")

    # Filtrer par longueur en caractères si spécifié
    if max_length_chars is not None:
        filtered_texts, filtered_labels = [], []
        for text, label in zip(all_texts, all_labels):
            if len(text) <= max_length_chars:
                filtered_texts.append(text)
                filtered_labels.append(label)
        all_texts, all_labels = filtered_texts, filtered_labels
        n_kept = len(all_texts)
        print(f"Après filtre longueur <= {max_length_chars} chars : {n_kept} reviews ({100*n_kept/n_total:.1f}%)")

    # Catégoriser en 4 groupes : (Z_br_count, Y_label)
    # Z=many_br (>4) ou Z=no_br (0) ; les autres (1-3) sont ignorés
    br_many_pos:  List[str] = []  # Z=many_br (>4), Y=1
    br_many_neg:  List[str] = []  # Z=many_br (>4), Y=0
    no_br_pos:    List[str] = []  # Z=no_br (0),   Y=1
    no_br_neg:    List[str] = []  # Z=no_br (0),   Y=0
    n_ignored = 0

    for text, label in zip(all_texts, all_labels):
        br_count = _count_br_tags(text)
        
        if br_count > 4:  # Many BR tags
            if label == 1:
                br_many_pos.append(text)
            else:
                br_many_neg.append(text)
        elif br_count == 0:  # No BR tags
            if label == 1:
                no_br_pos.append(text)
            else:
                no_br_neg.append(text)
        else:  # 1-3 BR tags → ignoré
            n_ignored += 1

    print(f"4 groupes créés (critère: >4 BR vs 0 BR) :")
    print(f"  br_many_pos (Z=many_br (>4), Y=1) : {len(br_many_pos)}")
    print(f"  br_many_neg (Z=many_br (>4), Y=0) : {len(br_many_neg)}")
    print(f"  no_br_pos   (Z=no_br (0),   Y=1) : {len(no_br_pos)}")
    print(f"  no_br_neg   (Z=no_br (0),   Y=0) : {len(no_br_neg)}")
    print(f"  IGNORÉS (1-3 BR tags)             : {n_ignored}")

    # Mélanger les 4 groupes pour équilibre dans les tranches
    rng_shuffle = np.random.default_rng(seed + 5000)
    br_many_pos = [br_many_pos[j] for j in rng_shuffle.permutation(len(br_many_pos))]
    br_many_neg = [br_many_neg[j] for j in rng_shuffle.permutation(len(br_many_neg))]
    no_br_pos = [no_br_pos[j] for j in rng_shuffle.permutation(len(no_br_pos))]
    no_br_neg = [no_br_neg[j] for j in rng_shuffle.permutation(len(no_br_neg))]

    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        rng_env = np.random.default_rng(seed + 5000 + i)
        rng_mix = np.random.default_rng(seed + 6100 + i)

        # Répartir les 4 groupes entre envs (tranches non-chevauchantes)
        n_envs = len(train_p_select)
        bp_per_env = len(br_many_pos) // n_envs
        bn_per_env = len(br_many_neg) // n_envs
        nbp_per_env = len(no_br_pos) // n_envs
        nbn_per_env = len(no_br_neg) // n_envs

        # Tranches pour cet env
        bp_start, bp_end = i * bp_per_env, (i+1)*bp_per_env if i < n_envs-1 else len(br_many_pos)
        bn_start, bn_end = i * bn_per_env, (i+1)*bn_per_env if i < n_envs-1 else len(br_many_neg)
        nbp_start, nbp_end = i * nbp_per_env, (i+1)*nbp_per_env if i < n_envs-1 else len(no_br_pos)
        nbn_start, nbn_end = i * nbn_per_env, (i+1)*nbn_per_env if i < n_envs-1 else len(no_br_neg)

        env_br_many_pos = br_many_pos[bp_start:bp_end]
        env_br_many_neg = br_many_neg[bn_start:bn_end]
        env_no_br_pos = no_br_pos[nbp_start:nbp_end]
        env_no_br_neg = no_br_neg[nbn_start:nbn_end]

        # Mélanger selon p_select :
        # - p_select % des br_many_pos et (1-p_select) % des br_many_neg
        #   → P(Y=1 | Z=many_br) = p_select
        # - p_select % des no_br_neg et (1-p_select) % des no_br_pos
        #   → P(Y=1 | Z=no_br) = 1 - p_select
        
        n_bp_keep = int(len(env_br_many_pos) * p_select)
        n_bn_keep = int(len(env_br_many_neg) * (1 - p_select))
        n_nbp_keep = int(len(env_no_br_pos) * (1 - p_select))
        n_nbn_keep = int(len(env_no_br_neg) * p_select)

        bp_idx = rng_mix.choice(len(env_br_many_pos),  size=n_bp_keep, replace=False)
        bn_idx = rng_mix.choice(len(env_br_many_neg),  size=n_bn_keep, replace=False)
        nbp_idx = rng_mix.choice(len(env_no_br_pos), size=n_nbp_keep, replace=False)
        nbn_idx = rng_mix.choice(len(env_no_br_neg), size=n_nbn_keep, replace=False)

        # Construire la sélection complète
        selected_texts = [env_br_many_pos[j]  for j in bp_idx] + \
                        [env_br_many_neg[j]  for j in bn_idx] + \
                        [env_no_br_pos[j] for j in nbp_idx] + \
                        [env_no_br_neg[j] for j in nbn_idx]
        
        selected_labels = [1] * n_bp_keep + [0] * n_bn_keep + \
                         [1] * n_nbp_keep + [0] * n_nbn_keep

        # Calcul de la vraie corrélation Y↔Z
        # P(Y=1 | Z=many_br) et P(Y=1 | Z=no_br)
        p_pos_given_many_br = (n_bp_keep) / (n_bp_keep + n_bn_keep) if (n_bp_keep + n_bn_keep) > 0 else 0.5
        p_pos_given_no_br = (n_nbp_keep) / (n_nbp_keep + n_nbn_keep) if (n_nbp_keep + n_nbn_keep) > 0 else 0.5

        print(f"  Sélectionné : {len(selected_texts)} reviews")
        print(f"    br_many_pos: {n_bp_keep} | br_many_neg: {n_bn_keep}")
        print(f"    no_br_pos: {n_nbp_keep} | no_br_neg: {n_nbn_keep}")
        print(f"  P(Y=1|Z=many_br)={p_pos_given_many_br:.1%}  (cible: {p_select:.0%})")
        print(f"  P(Y=1|Z=no_br)={p_pos_given_no_br:.1%} (cible: {1-p_select:.0%})")
        print(f"  Distribution Y globale: {np.mean(selected_labels):.1%} positifs")

        sel_texts_br = selected_texts
        sel_labels_br = np.array(selected_labels)
        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            sel_texts_br, sel_labels_br = _subsample_to_ratio(
                sel_texts_br, sel_labels_br, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub)

        n_sel = len(sel_texts_br)
        n_val = int(n_sel * val_frac)
        idx_sh = rng_env.permutation(n_sel)
        tr_idx, va_idx = idx_sh[n_val:], idx_sh[:n_val]

        tr_texts  = [sel_texts_br[j]  for j in tr_idx]
        tr_labels = sel_labels_br[tr_idx].copy()
        if label_flip > 0.0:
            rng_tf = np.random.default_rng(seed + 7000 + i)
            tr_labels[rng_tf.uniform(size=len(tr_labels)) < label_flip] ^= 1
        X_tr = tokenize_and_embed_with_bert(tr_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(torch.from_numpy(X_tr),
                              torch.from_numpy(tr_labels.reshape(-1, 1).astype(np.float32)),
                              meta={"p_select": p_select, "kind": "imdb_br_selection_train",
                                    "env_id": i, "label_flip": label_flip,
                                    "n_samples": len(X_tr)}))

        va_texts  = [sel_texts_br[j]  for j in va_idx]
        va_labels = sel_labels_br[va_idx].copy()
        if label_flip > 0.0:
            rng_vf = np.random.default_rng(seed + 8000 + i)
            va_labels[rng_vf.uniform(size=len(va_labels)) < label_flip] ^= 1
        X_va = tokenize_and_embed_with_bert(va_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(torch.from_numpy(X_va),
                            torch.from_numpy(va_labels.reshape(-1, 1).astype(np.float32)),
                            meta={"p_select": p_select, "kind": "imdb_br_selection_val",
                                  "env_id": i, "label_flip": label_flip,
                                  "n_samples": len(X_va)}))

    print(f"\n=== Test OOD (p_select=0.0 = corrélation INVERSÉE) ===")
    
    # OOD : 100% inversé
    # - 0% br_many_pos + 100% br_many_neg      → P(Y=1|Z=many_br) = 0%
    # - 100% no_br_pos + 0% no_br_neg          → P(Y=1|Z=no_br) = 100%
    
    rng_ood = np.random.default_rng(seed + 25000)
    n_ood_br = min(len(br_many_neg), 2000)  # limiter la taille
    n_ood_no_br = min(len(no_br_pos), 2000)
    
    ood_br_idx = rng_ood.choice(len(br_many_neg), size=n_ood_br, replace=False)
    ood_no_br_idx = rng_ood.choice(len(no_br_pos), size=n_ood_no_br, replace=False)
    
    ood_texts_final = [br_many_neg[j] for j in ood_br_idx] + \
                      [no_br_pos[j] for j in ood_no_br_idx]
    ood_labels_arr = np.array([0] * n_ood_br + [1] * n_ood_no_br)
    
    # Shuffle
    perm = rng_ood.permutation(len(ood_texts_final))
    ood_texts_final = [ood_texts_final[j] for j in perm]
    ood_labels_arr = ood_labels_arr[perm]
    
    print(f"  Composition : {n_ood_br} br_many_neg (>4 BR) + {n_ood_no_br} no_br_pos (0 BR) = {len(ood_texts_final)} total")
    print(f"  P(Y=1|Z=many_br)=0%  P(Y=1|Z=no_br)=100% (INVERSÉ)")
    print(f"  Distribution Y: {ood_labels_arr.mean():.1%} positifs")
    
    if class_ratio_test is not None and abs(class_ratio_test - 0.5) > 1e-6:
        rng_sub_t = np.random.default_rng(seed + 22000)
        ood_texts_final, ood_labels_arr = _subsample_to_ratio(
            ood_texts_final, ood_labels_arr, class_ratio_test, rng_sub_t)
    
    X_test = tokenize_and_embed_with_bert(ood_texts_final, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(torch.from_numpy(X_test),
                   torch.from_numpy(ood_labels_arr.reshape(-1, 1).astype(np.float32)),
                   meta={"kind": "imdb_br_selection_test_ood",
                         "n_samples": len(X_test),
                         "description": "many_br_vs_no_br_inverted"})

    print(f"\n✅ IMDB BR Selection (>4 BR vs 0 BR) — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {sum(e.X.shape[0] for e in val_envs)} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# IMDB Genres — Semi anti-causal
# =============================================================================
# DAG : Y (genre) → Z (spurious token) → X = BERT(description + Z)
# Tokens binaires réels ("pine" / "ash") injectés devant chaque mot neutre.
# Probabilité que le token corresponde au vrai label = p_correct[env].
# Test OOD : p_correct=0 → token toujours incorrect
# =============================================================================

# Mots réels, tokens uniques dans le vocabulaire DistilBERT, neutres pour le genre
IMDB_GENRES_SAC_TOKENS: Dict[int, str] = {0: "pine", 1: "ash"}

def build_envs_imdb_genres_semi_anti_causal(
    train_p_correct: List[float],
    test_p_correct: float,
    seed: int,
    label_flip: float = 0.25,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    pooling: str = "mean",
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    IMDB Genres — Semi-anti-causal : injection de tokens spurieux.

    DAG : Y (genre) → Z (token spurieux) → X = BERT(description + Z)

    Un token "thriller_marker" ou "romance_marker" est injecté dans chaque
    description. La corrélation token↔genre est contrôlée par p_correct.
    - En train : P(token correct) = p_correct[env] par env
    - En test OOD : P(token correct) = test_p_correct (souvent 0 → toujours incorrect)

    IRM doit apprendre que le token est spurieux et que le modèle doit reposer
    sur des patterns textuels robustes, pas sur le token.

    Parameters
    ----------
    train_p_correct : List[float]
        P(token correct) par env train (ex: [0.9, 0.7]).
    test_p_correct : float
        P(token correct) en test OOD (0.0 → token toujours erroné).
    seed : int
        Graine aléatoire.
    label_flip : float
        Fraction de labels bruités (0.0 = pas de bruit).
    bert_model, max_length, device, pooling : config BERT standard
    """
    print("Chargement du dataset IMDB Genres (semi anti-causal avec tokens)...")
    all_texts, all_labels = load_imdb_genres_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels)
    n_romance = int((all_labels_arr == 1).sum())
    n_thriller = n_total - n_romance
    print(f"Dataset : {n_total} descriptions — thriller={n_thriller}, romance={n_romance}")

    # Split global : 80% train, 10% val, 10% test
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test_split = int(n_total * 0.1)
    n_val_split  = int(n_total * 0.1)
    test_indices  = indices[:n_test_split]
    val_indices   = indices[n_test_split:n_test_split + n_val_split]
    train_indices = indices[n_test_split + n_val_split:]

    print(f"Split : Train {len(train_indices)} | Val {len(val_indices)} | Test {len(test_indices)}")

    n_envs = len(train_p_correct)
    samples_per_env = len(train_indices) // n_envs

    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_correct in enumerate(train_p_correct):
        print(f"\n=== Train Env {i} (p_correct={p_correct:.0%}) ===")
        start   = i * samples_per_env
        end     = (i + 1) * samples_per_env if i < n_envs - 1 else len(train_indices)
        env_idx = train_indices[start:end]

        texts  = [all_texts[int(j)]  for j in env_idx]
        labels = np.array([all_labels[int(j)] for j in env_idx], dtype=np.int32)

        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            texts, labels = _subsample_to_ratio(texts, labels,
                                               class_ratio_train[min(i, len(class_ratio_train)-1)], rng_sub)

        # Label flip : inverser aléatoirement le genre
        if label_flip > 0:
            rng_flip = np.random.default_rng(seed + i * 13 + 1)
            flip_mask = rng_flip.uniform(size=len(labels)) < label_flip
            labels[flip_mask] = 1 - labels[flip_mask]

        # Injecter token spurieux avec inject_spurious_token_multiclass
        # (distribué devant les mots neutres, signal fort comme Amazon/AGNews)
        rng_inject = np.random.default_rng(seed + i * 17 + 3)
        texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, IMDB_GENRES_SAC_TOKENS, rng_inject)
            for t, l in zip(texts, labels)
        ]
        n_correct = sum(
            IMDB_GENRES_SAC_TOKENS[int(l)] in tm.lower().split()
            for tm, l in zip(texts_mod, labels)
        )
        print(f"  Token correct : {n_correct}/{len(labels)} ({n_correct/len(labels):.1%})")

        X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling,
                                        finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(
            torch.from_numpy(X),
            torch.from_numpy(labels.reshape(-1, 1).astype(np.float32)),
            meta={
                "p_correct": p_correct,
                "label_flip": label_flip,
                "kind": "imdb_genres_semi_anti_causal_train",
                "env_id": i,
                "n_samples": len(X),
            }))

        # Val env
        print(f"=== Val Env {i} ===")
        val_texts  = [all_texts[int(j)]  for j in val_indices]
        val_labels = np.array([all_labels[int(j)] for j in val_indices], dtype=np.int32)

        if class_ratio_train is not None:
            rng_sub_v = np.random.default_rng(seed + 21000 + i)
            val_texts, val_labels = _subsample_to_ratio(val_texts, val_labels,
                                                        class_ratio_train[min(i, len(class_ratio_train)-1)], rng_sub_v)

        if label_flip > 0:
            rng_val_flip = np.random.default_rng(seed + 5000 + i + 1)
            flip_mask_v = rng_val_flip.uniform(size=len(val_labels)) < label_flip
            val_labels[flip_mask_v] = 1 - val_labels[flip_mask_v]

        rng_val = np.random.default_rng(seed + i * 19 + 11)
        val_texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, IMDB_GENRES_SAC_TOKENS, rng_val)
            for t, l in zip(val_texts, val_labels)
        ]

        X_val = tokenize_and_embed_with_bert(val_texts_mod, bert_model, max_length, device, pooling,
                                            finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(
            torch.from_numpy(X_val),
            torch.from_numpy(val_labels.reshape(-1, 1).astype(np.float32)),
            meta={
                "p_correct": p_correct,
                "label_flip": label_flip,
                "kind": "imdb_genres_semi_anti_causal_val",
                "env_id": i,
                "n_samples": len(X_val),
            }))

    # Test OOD : p_correct = test_p_correct (souvent 0)
    print(f"\n=== Test OOD (p_correct={test_p_correct:.0%}) ===")
    test_texts  = [all_texts[int(j)]  for j in test_indices]
    test_labels = np.array([all_labels[int(j)] for j in test_indices], dtype=np.int32)

    if class_ratio_test is not None and abs(class_ratio_test - 0.5) > 1e-6:
        rng_sub_t = np.random.default_rng(seed + 22000)
        test_texts, test_labels = _subsample_to_ratio(test_texts, test_labels, class_ratio_test, rng_sub_t)

    rng_test = np.random.default_rng(seed + 99000)
    test_texts_mod = [
        inject_spurious_token_multiclass(t, int(l), test_p_correct, IMDB_GENRES_SAC_TOKENS, rng_test)
        for t, l in zip(test_texts, test_labels)
    ]

    X_test = tokenize_and_embed_with_bert(test_texts_mod, bert_model, max_length, device, pooling,
                                         finetune_bert_layers=finetune_bert_layers)
    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(test_labels.reshape(-1, 1).astype(np.float32)),
        meta={
            "p_correct": test_p_correct,
            "kind": "imdb_genres_semi_anti_causal_test_ood",
            "n_samples": len(X_test),
        })

    print(f"\n✅ IMDB Genres Semi Anti-Causal — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {sum(e.X.shape[0] for e in val_envs)} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# IMDB Genres — Confounding varying proxy
# =============================================================================
# DAG : C ~ Ber(p_c) → Y (flip si C=1) ; C → Z = C XOR Ber(a_e) → token
# Cause commune C crée une corrélation spurieuse Z~Y via C, sans chemin direct Y→Z.
#
# Mécanisme appris par ERM :
#   BERT voit (texte, token).  Quand Z=1 (C≈1), les labels sont flipés par rapport
#   au texte → BERT apprend Z comme "context switch" (inverteur de prediction).
#   C'est une interaction (texte × Z) non-linéaire capturée par l'attention BERT.
#
# En test OOD (a_test=1.0) : Z = NOT C, labels propres (pas de flip)
#   → ERM utilise Z comme inverteur mais les labels ne sont plus flipés → échec
#   → IRM (qui a ignoré Z) utilise le texte et réussit
#
# Note : P(Y_obs=1|Z=1) = 0.5 marginalement (indépendance marginale).
#   Ce n'est PAS une contradiction — la spuriosité est dans la relation jointe
#   (texte, Z) → Y_obs, apprise via l'attention BERT.

# Tokens réels pour le proxy confoundeur (distincts des tokens SAC)
IMDB_GENRES_CONF_TOKENS: Dict[int, str] = {0: "oak", 1: "elm"}

def build_envs_imdb_genres_conf_varying_proxy(
    a_train: List[float],
    a_test: float,
    seed: int,
    p_c_flip: float = 0.25,
    gamma: float = 0.5,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    pooling: str = "mean",
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    IMDB Genres — Confounding varying proxy : C → Y et C → Z.

    DAG : C ~ Ber(p_c) → Y (Y est poussé vers C) ; C → Z (token proxy de C)
          Z = C XOR Ber(a_e)  [le token a du bruit ]

    Le confounder C affecte à la fois le genre ET le token proxy.
    En entraînement, le token se corrèle avec le genre (par le chemin C→Y et C→Z).
    En test OOD, le bruitC→Z (via a_test proche de 1) dégrade cette corrélation.
    IRM doit ignorer Z et apprendre le pattern causal dans le texte.

    Parameters
    ----------
    a_train : List[float]
        Bruit C→Z par env train. a_e ≈ 0 → Z = C (fort confounder).
        a_e ≈ 1 → Z aléatoire (bruit complet).
    a_test : float
        Bruit C→Z en test OOD (typiquement ≈ 1.0).
    p_c_flip : float
        Probabilité que C = 1.
    gamma : float
        Force fixe de C→Y. Si Y!=C, le label devient C avec probabilité gamma.
    """
    print("Chargement du dataset IMDB Genres (confounding varying proxy)...")
    all_texts, all_labels = load_imdb_genres_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset : {n_total} descriptions")

    # Split global : 80/10/10
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test_split = int(n_total * 0.1)
    n_val_split  = int(n_total * 0.1)
    test_indices  = indices[:n_test_split]
    val_indices   = indices[n_test_split:n_test_split + n_val_split]
    train_indices = indices[n_test_split + n_val_split:]

    print(f"Split : Train {len(train_indices)} | Val {len(val_indices)} | Test {len(test_indices)}")

    n_envs = len(a_train)
    samples_per_env = len(train_indices) // n_envs

    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, a_e in enumerate(a_train):
        print(f"\n=== Train Env {i} (a_e={a_e:.2f}) ===")
        start   = i * samples_per_env
        end     = (i + 1) * samples_per_env if i < n_envs - 1 else len(train_indices)
        env_idx = train_indices[start:end]

        texts  = [all_texts[int(j)]  for j in env_idx]
        labels = np.array([all_labels[int(j)] for j in env_idx], dtype=np.int32)

        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            texts, labels = _subsample_to_ratio(
                texts, labels,
                class_ratio_train[min(i, len(class_ratio_train)-1)], rng_sub)

        # Générer C APRÈS sous-échantillonnage (évite les décalages d'indices)
        rng_c_e = np.random.default_rng(seed + i * 7 + 100)
        C_env = rng_c_e.binomial(1, p_c_flip, size=len(labels))

        # C → Y : pousse le label vers C quand il lui est opposé
        rng_y_e = np.random.default_rng(seed + i * 17 + 3)
        labels_confounded = _apply_conf_label_bias(labels, C_env, gamma, rng_y_e)

        if label_flip > 0:
            rng_flip = np.random.default_rng(seed + i * 13 + 1)
            flip_mask = rng_flip.uniform(size=len(labels_confounded)) < label_flip
            labels_confounded[flip_mask] = 1 - labels_confounded[flip_mask]

        # C → Z : proxy bruité de C, varie par env
        # a_e petit → Z ≈ C (fort) ; a_e grand → Z bruité (faible)
        rng_z = np.random.default_rng(seed + i * 23 + 5)
        noise = (rng_z.uniform(size=len(C_env)) < a_e).astype(int)
        Z_env = C_env ^ noise

        # Injecter le token proxy (distribué sur mots neutres, comme Amazon)
        rng_inj = np.random.default_rng(seed + i * 41 + 13)
        texts_mod = [
            inject_spurious_token_multiclass(text, int(z), 1.0, IMDB_GENRES_CONF_TOKENS, rng_inj)
            for text, z in zip(texts, Z_env)
        ]

        X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling,
                                        finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(
            torch.from_numpy(X),
            torch.from_numpy(labels_confounded.reshape(-1, 1).astype(np.float32)),
            meta={
                "a_e": a_e,
                "p_c": p_c_flip,
                "label_flip": label_flip,
                "kind": "imdb_genres_conf_varying_proxy_train",
                "env_id": i,
                "n_samples": len(X),
            }))

        # Val env
        print(f"=== Val Env {i} ===")
        val_texts  = [all_texts[int(j)]  for j in val_indices]
        val_labels = np.array([all_labels[int(j)] for j in val_indices], dtype=np.int32)

        if class_ratio_train is not None:
            rng_sub_v = np.random.default_rng(seed + 21000 + i)
            val_texts, val_labels = _subsample_to_ratio(
                val_texts, val_labels,
                class_ratio_train[min(i, len(class_ratio_train)-1)], rng_sub_v)

        # C val généré après sous-échantillonnage, indépendant du C train
        rng_c_v = np.random.default_rng(seed + i * 31 + 200)
        C_val = rng_c_v.binomial(1, p_c_flip, size=len(val_labels))

        rng_y_v = np.random.default_rng(seed + i * 37 + 11)
        val_labels_conf = _apply_conf_label_bias(val_labels, C_val, gamma, rng_y_v)

        if label_flip > 0:
            rng_val_flip = np.random.default_rng(seed + 5000 + i + 1)
            flip_mask_v = rng_val_flip.uniform(size=len(val_labels_conf)) < label_flip
            val_labels_conf[flip_mask_v] = 1 - val_labels_conf[flip_mask_v]

        rng_z_v = np.random.default_rng(seed + i * 31 + 7)
        noise_v = (rng_z_v.uniform(size=len(C_val)) < a_e).astype(int)
        Z_val = C_val ^ noise_v

        rng_inj_v = np.random.default_rng(seed + i * 43 + 17)
        val_texts_mod = [
            inject_spurious_token_multiclass(text, int(z), 1.0, IMDB_GENRES_CONF_TOKENS, rng_inj_v)
            for text, z in zip(val_texts, Z_val)
        ]

        X_val = tokenize_and_embed_with_bert(val_texts_mod, bert_model, max_length, device, pooling,
                                            finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(
            torch.from_numpy(X_val),
            torch.from_numpy(val_labels_conf.reshape(-1, 1).astype(np.float32)),
            meta={
                "a_e": a_e,
                "p_c": p_c_flip,
                "kind": "imdb_genres_conf_varying_proxy_val",
                "env_id": i,
                "n_samples": len(X_val),
            }))

    # Test OOD — labels CLEAN (pas de flip C), Z = C_test XOR Ber(a_test)
    # Avec a_test=1.0 : Z_test = NOT C_test → le token inverse le signal habituel
    # ERM (qui a appris le context switch via Z) est trompé ; IRM (texte seul) tient
    print(f"\n=== Test OOD (a_test={a_test:.2f}) ===")
    test_texts  = [all_texts[int(j)]  for j in test_indices]
    test_labels_clean = np.array([all_labels[int(j)] for j in test_indices], dtype=np.int32)

    if class_ratio_test is not None and abs(class_ratio_test - 0.5) > 1e-6:
        rng_sub_t = np.random.default_rng(seed + 22000)
        test_texts, test_labels_clean = _subsample_to_ratio(test_texts, test_labels_clean, class_ratio_test, rng_sub_t)

    rng_c_t = np.random.default_rng(seed + 777)
    C_test = rng_c_t.binomial(1, p_c_flip, size=len(test_labels_clean))
    rng_z_t = np.random.default_rng(seed + 999)
    noise_t = (rng_z_t.uniform(size=len(C_test)) < a_test).astype(int)
    Z_test = C_test ^ noise_t

    rng_inj_t = np.random.default_rng(seed + 888)
    test_texts_mod = [
        inject_spurious_token_multiclass(text, int(z), 1.0, IMDB_GENRES_CONF_TOKENS, rng_inj_t)
        for text, z in zip(test_texts, Z_test)
    ]

    X_test = tokenize_and_embed_with_bert(test_texts_mod, bert_model, max_length, device, pooling,
                                         finetune_bert_layers=finetune_bert_layers)
    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(test_labels_clean.reshape(-1, 1).astype(np.float32)),
        meta={
            "a_e": a_test,
            "p_c": p_c_flip,
            "kind": "imdb_genres_conf_varying_proxy_test_ood",
            "n_samples": len(X_test),
        })

    print(f"\n✅ IMDB Genres Confounding Varying Proxy — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {sum(e.X.shape[0] for e in val_envs)} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


def load_imdb_genres_dataset(seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Charge jquigl/imdb-genres depuis Hugging Face.

    Fusionne tous les splits (train, validation, test), filtre uniquement
    les genres "thriller" et "romance", et ne conserve que les colonnes
    "description" et "genre".

    Returns
    -------
    texts  : List[str]  – descriptions de films
    labels : List[int]  – 0 = thriller, 1 = romance
    """
    from datasets import concatenate_datasets

    dataset = load_dataset("jquigl/imdb-genres")

    # Fusionner tous les splits disponibles
    splits = [dataset[s] for s in dataset.keys()]
    all_data = concatenate_datasets(splits)

    # Filtrer thriller et romance uniquement
    all_data = all_data.filter(
        lambda ex: ex["genre"] in ("Thriller", "Romance")
    )

    # Shuffle reproductible
    all_data = all_data.shuffle(seed=seed)

    texts  = [str(ex["description"]) for ex in all_data]
    labels = [1 if ex["genre"] == "Romance" else 0 for ex in all_data]

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"IMDB Genres (thriller/romance) : {len(texts)} exemples — "
          f"romance={n_pos}, thriller={n_neg}")
    return texts, labels


# =============================================================================
# IMDB Genres — Size selection
# =============================================================================
# Dataset : jquigl/imdb-genres (descriptions de films + genres)
# Tâche   : prédire le genre (thriller=0, romance=1)
# Signal spurieux Z : longueur de la description
#
# Architecture 4-pool identique à IMDB size selection :
#   short_pos (Z=court, Y=romance), short_neg (Z=court, Y=thriller),
#   long_pos  (Z=long,  Y=romance), long_neg  (Z=long,  Y=thriller)
# Corrélation contrôlée : P(Y=romance | Z=long) = p_select
# =============================================================================

def build_envs_imdb_genres_size_selection(
    train_p_select: List[float],
    seed: int,
    threshold_method: str = "quartile",
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 256,
    device: str = "cpu",
    pooling: str = "mean",
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    IMDB Genres (thriller/romance) — sélection par longueur de description.

    DAG : Y (genre) → Z (longueur description) → S (sélection d'entraînement)

    **Architecture 4-pool** :
    - short_pos : Z=court, Y=romance (1)
    - short_neg : Z=court, Y=thriller (0)
    - long_pos  : Z=long,  Y=romance (1)
    - long_neg  : Z=long,  Y=thriller (0)

    **Entraînement (Env i avec p_select[i])** :
      - p_select % des long_pos + (1-p_select) % des long_neg
        → P(Y=romance | Z=long) = p_select
      - p_select % des short_neg + (1-p_select) % des short_pos
        → P(Y=romance | Z=short) = 1 - p_select
      - P(Y=romance) global ≈ 50%

    **Test OOD** : corrélation taille↔genre complètement inversée
      → P(Y=romance | Z=long) = 0%
      → P(Y=romance | Z=short) = 100%

    Parameters
    ----------
    train_p_select   : List[float]           Force de corrélation (0-1) par env.
    threshold_method : str                   "quartile" (défaut), "median".
    val_frac         : float                 Fraction validation.
    label_flip       : float                 Taux de bruit symétrique sur les labels.
    max_length       : int                   Max tokens BERT (défaut 256).
    class_ratio_train: Optional[List[float]] Fraction de positifs par env.
    class_ratio_test : Optional[float]       Fraction de positifs au test.
    """
    print("Chargement du dataset IMDB Genres (sélection par taille)...")
    all_texts, all_labels = load_imdb_genres_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset : {n_total} descriptions (thriller/romance)")

    # Calcul des seuils : percentiles sur l'ensemble du corpus
    all_lengths = [len(t) for t in all_texts]
    if threshold_method == "quartile":
        t1 = float(np.percentile(all_lengths, 25))
        t2 = float(np.percentile(all_lengths, 75))
    elif threshold_method == "median":
        t1 = float(np.percentile(all_lengths, 33))
        t2 = float(np.percentile(all_lengths, 67))
    elif threshold_method == "soft":
        t1 = float(np.percentile(all_lengths, 35))
        t2 = float(np.percentile(all_lengths, 65))
    else:
        raise ValueError(f"Unknown threshold_method: {threshold_method}")
    print(f"Seuils ({threshold_method}): court < {t1:.0f} chars, long > {t2:.0f} chars")

    # Catégoriser en 4 groupes : (Z_size, Y_genre)
    short_pos: List[str] = []  # Z=court, Y=romance (1)
    short_neg: List[str] = []  # Z=court, Y=thriller (0)
    long_pos:  List[str] = []  # Z=long,  Y=romance (1)
    long_neg:  List[str] = []  # Z=long,  Y=thriller (0)

    for text, label in zip(all_texts, all_labels):
        text_len = len(text)
        if text_len < t1:       # court
            if label == 1:
                short_pos.append(text)
            else:
                short_neg.append(text)
        elif text_len > t2:     # long
            if label == 1:
                long_pos.append(text)
            else:
                long_neg.append(text)
        # textes de taille intermédiaire ignorés

    print(f"4 groupes créés :")
    print(f"  short_pos (Z=court, Y=romance)  : {len(short_pos)}")
    print(f"  short_neg (Z=court, Y=thriller) : {len(short_neg)}")
    print(f"  long_pos  (Z=long,  Y=romance)  : {len(long_pos)}")
    print(f"  long_neg  (Z=long,  Y=thriller) : {len(long_neg)}")

    # Mélanger les 4 groupes
    rng_shuffle = np.random.default_rng(seed + 5000)
    short_pos = [short_pos[j] for j in rng_shuffle.permutation(len(short_pos))]
    short_neg = [short_neg[j] for j in rng_shuffle.permutation(len(short_neg))]
    long_pos  = [long_pos[j]  for j in rng_shuffle.permutation(len(long_pos))]
    long_neg  = [long_neg[j]  for j in rng_shuffle.permutation(len(long_neg))]

    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    n_envs = len(train_p_select)
    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        rng_env = np.random.default_rng(seed + 5000 + i)
        rng_mix = np.random.default_rng(seed + 6100 + i)

        # Tranches non-chevauchantes par env
        sp_per_env = len(short_pos) // n_envs
        sn_per_env = len(short_neg) // n_envs
        lp_per_env = len(long_pos)  // n_envs
        ln_per_env = len(long_neg)  // n_envs

        sp_start, sp_end = i * sp_per_env, (i+1)*sp_per_env if i < n_envs-1 else len(short_pos)
        sn_start, sn_end = i * sn_per_env, (i+1)*sn_per_env if i < n_envs-1 else len(short_neg)
        lp_start, lp_end = i * lp_per_env, (i+1)*lp_per_env if i < n_envs-1 else len(long_pos)
        ln_start, ln_end = i * ln_per_env, (i+1)*ln_per_env if i < n_envs-1 else len(long_neg)

        env_short_pos = short_pos[sp_start:sp_end]
        env_short_neg = short_neg[sn_start:sn_end]
        env_long_pos  = long_pos[lp_start:lp_end]
        env_long_neg  = long_neg[ln_start:ln_end]

        n_lp_keep = int(len(env_long_pos)  * p_select)
        n_ln_keep = int(len(env_long_neg)  * (1 - p_select))
        n_sp_keep = int(len(env_short_pos) * (1 - p_select))
        n_sn_keep = int(len(env_short_neg) * p_select)

        lp_idx = rng_mix.choice(len(env_long_pos),  size=n_lp_keep, replace=False)
        ln_idx = rng_mix.choice(len(env_long_neg),  size=n_ln_keep, replace=False)
        sp_idx = rng_mix.choice(len(env_short_pos), size=n_sp_keep, replace=False)
        sn_idx = rng_mix.choice(len(env_short_neg), size=n_sn_keep, replace=False)

        selected_texts = ([env_long_pos[j]  for j in lp_idx] +
                         [env_long_neg[j]  for j in ln_idx] +
                         [env_short_pos[j] for j in sp_idx] +
                         [env_short_neg[j] for j in sn_idx])
        selected_labels = [1]*n_lp_keep + [0]*n_ln_keep + [1]*n_sp_keep + [0]*n_sn_keep

        p_pos_given_long  = n_lp_keep / (n_lp_keep + n_ln_keep) if (n_lp_keep + n_ln_keep) > 0 else 0.5
        p_pos_given_short = n_sp_keep / (n_sp_keep + n_sn_keep) if (n_sp_keep + n_sn_keep) > 0 else 0.5

        print(f"  long_pos: {n_lp_keep} | long_neg: {n_ln_keep} | "
              f"short_pos: {n_sp_keep} | short_neg: {n_sn_keep}")
        print(f"  P(romance|long)={p_pos_given_long:.1%}  (cible: {p_select:.0%})")
        print(f"  P(romance|short)={p_pos_given_short:.1%} (cible: {1-p_select:.0%})")
        print(f"  P(romance) global: {np.mean(selected_labels):.1%}")

        sel_texts  = selected_texts
        sel_labels = np.array(selected_labels)
        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            sel_texts, sel_labels = _subsample_to_ratio(
                sel_texts, sel_labels,
                class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub)

        n_sel = len(sel_texts)
        n_val = int(n_sel * val_frac)
        idx_sh = rng_env.permutation(n_sel)
        tr_idx, va_idx = idx_sh[n_val:], idx_sh[:n_val]

        tr_texts  = [sel_texts[j] for j in tr_idx]
        tr_labels = sel_labels[tr_idx].copy()
        if label_flip > 0.0:
            rng_tf = np.random.default_rng(seed + 7000 + i)
            tr_labels[rng_tf.uniform(size=len(tr_labels)) < label_flip] ^= 1
        X_tr = tokenize_and_embed_with_bert(
            tr_texts, bert_model, max_length, device, pooling,
            finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(
            torch.from_numpy(X_tr),
            torch.from_numpy(tr_labels.reshape(-1, 1).astype(np.float32)),
            meta={"p_select": p_select, "kind": "imdb_genres_size_selection_train",
                  "env_id": i, "t1": t1, "t2": t2,
                  "label_flip": label_flip, "n_samples": len(X_tr)}))

        va_texts  = [sel_texts[j] for j in va_idx]
        va_labels = sel_labels[va_idx].copy()
        if label_flip > 0.0:
            rng_vf = np.random.default_rng(seed + 8000 + i)
            va_labels[rng_vf.uniform(size=len(va_labels)) < label_flip] ^= 1
        X_va = tokenize_and_embed_with_bert(
            va_texts, bert_model, max_length, device, pooling,
            finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(
            torch.from_numpy(X_va),
            torch.from_numpy(va_labels.reshape(-1, 1).astype(np.float32)),
            meta={"p_select": p_select, "kind": "imdb_genres_size_selection_val",
                  "env_id": i, "label_flip": label_flip, "n_samples": len(X_va)}))

    print(f"\n=== Test OOD (corrélation taille↔genre INVERSÉE) ===")

    rng_ood = np.random.default_rng(seed + 25000)
    n_ood_long  = min(len(long_neg),  2000)
    n_ood_short = min(len(short_pos), 2000)

    ood_long_idx  = rng_ood.choice(len(long_neg),  size=n_ood_long,  replace=False)
    ood_short_idx = rng_ood.choice(len(short_pos), size=n_ood_short, replace=False)

    ood_texts_final = ([long_neg[j]  for j in ood_long_idx] +
                       [short_pos[j] for j in ood_short_idx])
    ood_labels_arr  = np.array([0]*n_ood_long + [1]*n_ood_short)

    perm = rng_ood.permutation(len(ood_texts_final))
    ood_texts_final = [ood_texts_final[j] for j in perm]
    ood_labels_arr  = ood_labels_arr[perm]

    print(f"  {n_ood_long} long_neg (thriller) + {n_ood_short} short_pos (romance)")
    print(f"  P(romance|long)=0%  P(romance|short)=100% (INVERSÉ)")
    print(f"  P(romance) global: {ood_labels_arr.mean():.1%}")

    if class_ratio_test is not None and abs(class_ratio_test - 0.5) > 1e-6:
        rng_sub_t = np.random.default_rng(seed + 22000)
        ood_texts_final, ood_labels_arr = _subsample_to_ratio(
            ood_texts_final, ood_labels_arr, class_ratio_test, rng_sub_t)

    X_test = tokenize_and_embed_with_bert(
        ood_texts_final, bert_model, max_length, device, pooling,
        finetune_bert_layers=finetune_bert_layers)
    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(ood_labels_arr.reshape(-1, 1).astype(np.float32)),
        meta={"kind": "imdb_genres_size_selection_test_ood",
              "n_samples": len(X_test), "description": "inverted_size_correlation"})

    print(f"\n✅ IMDB Genres Size Selection — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {sum(e.X.shape[0] for e in val_envs)} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


def load_imdb_dataset(seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Charge IMDB depuis Hugging Face (stanfordnlp/imdb).

    On combine train (25 000) et test (25 000) et on ignore le split
    « unsupervised » (pas de labels).  Résultat shufflé avec seed fixe.

    Returns
    -------
    texts  : List[str]   – critiques de films (textes longs)
    labels : List[int]   – 0 = négatif, 1 = positif
    """
    from datasets import concatenate_datasets

    dataset = load_dataset("stanfordnlp/imdb")
    labeled = concatenate_datasets([dataset["train"], dataset["test"]])
    labeled = labeled.shuffle(seed=seed)

    texts  = list(labeled["text"])
    labels = [int(l) for l in labeled["label"]]
    return texts, labels


# =============================================================================
# IMDB — Confounding variant 1 : varying proxy
# =============================================================================
# DAG identique à SMS Spam conf_varying_proxy :
#   C ~ Ber(p_c_flip) → Z(a_e) = C XOR Ber(a_e) → token ; C → Y (flip si C=1) ; texte → Y
# Variation d'env : a_e (bruit sur C→Z).
# OOD : a_test ≈ 1 → token anti-corrélé avec Y_obs → Z ⊥ Y sur vrais labels.
# =============================================================================

def build_envs_imdb_conf_varying_proxy(
    a_train: List[float],
    a_test: float,
    seed: int,
    p_c_flip: float = 0.25,
    gamma: float = 0.5,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 512,
    device: str = "cpu",
    pooling: str = "mean",
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    IMDB — confounding avec variation du proxy Z = C XOR Ber(a_e).

    DAG : C ~ Ber(p_c_flip) → Z(a_e) → token ; C → Y (Y est poussé vers C) ; texte → Y
    Variation d'env : a_e (bruit sur C→Z).
    OOD : a_test ≈ 1 → token anti-corrélé avec Y_obs.

    Parameters
    ----------
    a_train          : List[float]           Bruit proxy par env train (ex : [0.01, 0.1]).
    a_test           : float                 Bruit proxy OOD (ex : 1.0).
    p_c_flip         : float                 P(C=1), prévalence du confondeur binaire.
    gamma            : float                 Force fixe de C→Y. Si Y!=C, le label devient C avec
                                             probabilité gamma.
    max_length       : int                   Max tokens BERT (défaut 512 pour textes longs IMDB).
    class_ratio_train: Optional[List[float]] Fraction de positifs par env (ex : [0.2, 0.8]).
    class_ratio_test : Optional[float]       Fraction de positifs au test (ex : 0.5).
    """
    print("Chargement du dataset IMDB (confounding – varying proxy)...")
    all_texts, all_labels = load_imdb_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels, dtype=np.int64)

    rng_split = np.random.default_rng(seed)
    indices = rng_split.permutation(n_total)
    n_test = int(n_total * 0.1)
    n_val  = int(n_total * 0.1)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"Dataset : {n_total} reviews | Split 80/10/10 : "
          f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    n_envs = len(a_train)
    spe = len(train_idx) // n_envs
    train_envs: List[Env] = []
    val_envs:   List[Env] = []
    rng_master = np.random.default_rng(seed + 1)  # noqa: F841

    val_texts  = [all_texts[int(j)] for j in val_idx]
    val_labels = all_labels_arr[val_idx]

    for i, a_e in enumerate(a_train):
        print(f"\n=== Train Env {i} (a={a_e}) ===")
        env_idx = train_idx[i * spe:(i + 1) * spe if i < n_envs - 1 else len(train_idx)]
        texts  = [all_texts[int(j)] for j in env_idx]
        labels = all_labels_arr[env_idx].copy()

        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            texts, labels = _subsample_to_ratio(texts, labels, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub)

        rng_e = np.random.default_rng(seed + i * 7)
        C = rng_e.binomial(1, p_c_flip, size=len(labels))
        N = rng_e.binomial(1, a_e, size=len(labels))
        Z = np.logical_xor(C, N).astype(int)

        X, Y = _conf_make_env(
            texts, labels.astype(np.float32), C, Z, gamma, rng_e,
            bert_model, max_length, device, pooling,
            apply_gamma=True, conf_tokens=_IMDB_CONF_TOKENS,
         finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                              meta={"kind": "imdb_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                                    "split": "train", "env_id": i, "n_samples": len(X)}))

        print(f"=== Val Env {i} (a={a_e}) ===")
        val_texts_e  = list(val_texts)
        val_labels_e = all_labels_arr[val_idx].copy()
        if class_ratio_train is not None:
            rng_sub_v = np.random.default_rng(seed + 21000 + i)
            val_texts_e, val_labels_e = _subsample_to_ratio(
                val_texts_e, val_labels_e, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub_v)
        rng_v = np.random.default_rng(seed + 5000 + i)
        Cv = rng_v.binomial(1, p_c_flip, size=len(val_labels_e))
        Nv = rng_v.binomial(1, a_e, size=len(val_labels_e))
        Zv = np.logical_xor(Cv, Nv).astype(int)
        X_val, Y_val = _conf_make_env(
            val_texts_e, val_labels_e.astype(np.float32), Cv, Zv, gamma, rng_v,
            bert_model, max_length, device, pooling,
            apply_gamma=True, conf_tokens=_IMDB_CONF_TOKENS,
         finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(Y_val),
                            meta={"kind": "imdb_conf_varying_proxy", "a": a_e, "p_c_flip": p_c_flip,
                                  "split": "val", "env_id": i, "n_samples": len(X_val)}))

    print(f"\n=== Test OOD (a={a_test}) ===")
    test_texts  = [all_texts[int(j)] for j in test_idx]
    test_labels = all_labels_arr[test_idx].copy()
    if class_ratio_test is not None:
        rng_sub_t = np.random.default_rng(seed + 22000)
        test_texts, test_labels = _subsample_to_ratio(
            test_texts, test_labels, class_ratio_test, rng_sub_t)
    rng_t = np.random.default_rng(seed + 777)
    Ct = rng_t.binomial(1, p_c_flip, size=len(test_labels))
    Nt = rng_t.binomial(1, a_test, size=len(test_labels))
    Zt = np.logical_xor(Ct, Nt).astype(int)
    X_test, Y_test = _conf_make_env(
        test_texts, test_labels.astype(np.float32), Ct, Zt, gamma, rng_t,
        bert_model, max_length, device, pooling,
        apply_gamma=False, conf_tokens=_IMDB_CONF_TOKENS,
     finetune_bert_layers=finetune_bert_layers)
    test_env = Env(torch.from_numpy(X_test), torch.from_numpy(Y_test),
                   meta={"kind": "imdb_conf_varying_proxy", "a": a_test, "p_c_flip": p_c_flip,
                         "split": "test_ood", "n_samples": len(X_test)})

    print(f"\n✅ IMDB Confounding varying proxy — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# ██████  Amazon Reviews – Books  ██████
# =============================================================================
# Dataset CAUSAL (X → Y) : le contenu du texte détermine si la review est utile.
# Source : McAuley-Lab/Amazon-Reviews-2023, catégorie Books (jsonl streaming).
# Binarisation : helpful_vote = 0 → inutile (0), helpful_vote ≥ 5 → utile (1).
# Zone 1-4 écartée (ambiguë).
# =============================================================================

AMAZON_CLASS_NAMES: Dict[int, str] = {0: "not_helpful", 1: "helpful"}
AMAZON_TOKENS: Dict[int, str] = {0: "moon", 1: "sun"}
_AMAZON_CONF_TOKENS: Dict[str, str] = {
    "ham_correlated":  "moon",   # corrélé au label 0 (inutile)
    "spam_correlated": "sun",    # corrélé au label 1 (utile)
}


def load_amazon_books(
    seed: int = 42,
    n_target: int = 100_000,
    helpful_threshold: int = 5,
) -> Tuple[List[str], List[int]]:
    """
    Charge Amazon Reviews Books depuis HuggingFace (streaming jsonl).

    Binarise sur helpful_vote :
      - helpful_vote = 0           → inutile (0)
      - helpful_vote ≥ threshold   → utile   (1)
      - entre 1 et threshold-1     → écarté (ambigu)

    Collecte en streaming puis sous-échantillonne à *n_target* (équilibré 50/50).

    Returns
    -------
    texts  : List[str]
    labels : List[int]   – 0 = inutile, 1 = utile
    """
    print(f"Chargement Amazon Books (streaming, cible {n_target} reviews, "
          f"seuil helpful_vote ≥ {helpful_threshold})...")
    ds = load_dataset(
        "json",
        data_files="hf://datasets/McAuley-Lab/Amazon-Reviews-2023/"
                   "raw/review_categories/Books.jsonl",
        split="train",
        streaming=True,
    )

    target_per_class = n_target // 2
    helpful_texts:     List[str] = []
    not_helpful_texts: List[str] = []

    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        vote = int(row.get("helpful_vote", 0) or 0)
        if vote == 0 and len(not_helpful_texts) < target_per_class:
            not_helpful_texts.append(text)
        elif vote >= helpful_threshold and len(helpful_texts) < target_per_class:
            helpful_texts.append(text)
        if (len(not_helpful_texts) >= target_per_class
                and len(helpful_texts) >= target_per_class):
            break

    rng = np.random.default_rng(seed)
    n_per = min(target_per_class, len(helpful_texts), len(not_helpful_texts))

    texts  = not_helpful_texts[:n_per] + helpful_texts[:n_per]
    labels = [0] * n_per + [1] * n_per

    perm = rng.permutation(len(texts))
    texts  = [texts[int(i)]  for i in perm]
    labels = [labels[int(i)] for i in perm]

    print(f"  Collecté {n_per} inutiles + {n_per} utiles = {len(texts)} total")
    return texts, labels


# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Semi Anti-Causal
# ─────────────────────────────────────────────────────────────────────────────

def build_envs_amazon_semi_anti_causal(
    train_p_correct: List[float],
    test_p_correct: float,
    seed: int,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 512,
    device: str = "cpu",
    pooling: str = "mean",
    n_target: int = 100_000,
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    Semi anti-causal sur Amazon Books (utilité des reviews).

    DAG : Text → Y ; Y → Z (token spurieux injecté avec proba p_correct) ; Text ⊕ Z → X.
    OOD : p_correct ≈ 0 (tokens inversés).
    """
    all_texts, all_labels = load_amazon_books(seed=seed, n_target=n_target)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels, dtype=np.int64)
    print(f"Dataset : {n_total} reviews  "
          f"(négatif={int((all_labels_arr==0).sum())}, positif={int((all_labels_arr==1).sum())})")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test = int(n_total * 0.10)
    n_val  = int(n_total * 0.10)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"Split 80/10/10 : Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    n_envs = len(train_p_correct)
    spe = len(train_idx) // n_envs
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    val_texts  = [all_texts[int(j)] for j in val_idx]
    val_labels = all_labels_arr[val_idx].copy()

    for i, p_correct in enumerate(train_p_correct):
        print(f"\n=== Train Env {i} (p_correct={p_correct:.0%}) ===")
        env_idx = train_idx[i * spe:(i + 1) * spe if i < n_envs - 1 else len(train_idx)]
        texts  = [all_texts[int(j)] for j in env_idx]
        labels = all_labels_arr[env_idx].copy()

        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            texts, labels = _subsample_to_ratio(texts, labels, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub)

        if label_flip > 0.0:
            rng_flip = np.random.default_rng(seed + i * 13 + 1)
            flip_mask = rng_flip.uniform(size=len(labels)) < label_flip
            labels[flip_mask] = 1 - labels[flip_mask]

        rng_inj = np.random.default_rng(seed + i * 17 + 3)
        texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, AMAZON_TOKENS, rng_inj)
            for t, l in zip(texts, labels)
        ]
        X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        Y = labels.reshape(-1, 1).astype(np.float32)
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                              meta={"p_correct": p_correct, "label_flip": label_flip,
                                    "kind": "amazon_semi_anti_causal_train",
                                    "env_id": i, "n_samples": len(X)}))

        print(f"=== Val Env {i} (p_correct={p_correct:.0%}) ===")
        val_texts_e  = list(val_texts)
        val_labels_e = val_labels.copy()
        if class_ratio_train is not None:
            rng_sub_v = np.random.default_rng(seed + 21000 + i)
            val_texts_e, val_labels_e = _subsample_to_ratio(
                val_texts_e, val_labels_e, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub_v)
        if label_flip > 0.0:
            rng_vf = np.random.default_rng(seed + 5000 + i + 1)
            fmv = rng_vf.uniform(size=len(val_labels_e)) < label_flip
            val_labels_e[fmv] = 1 - val_labels_e[fmv]
        rng_v = np.random.default_rng(seed + 5000 + i)
        val_texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, AMAZON_TOKENS, rng_v)
            for t, l in zip(val_texts_e, val_labels_e)
        ]
        X_val = tokenize_and_embed_with_bert(val_texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(torch.from_numpy(X_val),
                            torch.from_numpy(val_labels_e.reshape(-1, 1).astype(np.float32)),
                            meta={"p_correct": p_correct, "kind": "amazon_semi_anti_causal_val",
                                  "env_id": i, "n_samples": len(X_val)}))

    print(f"\n=== Test OOD (p_correct={test_p_correct:.0%}) ===")
    test_texts  = [all_texts[int(j)] for j in test_idx]
    test_labels = all_labels_arr[test_idx].copy()
    if class_ratio_test is not None:
        rng_sub_t = np.random.default_rng(seed + 22000)
        test_texts, test_labels = _subsample_to_ratio(
            test_texts, test_labels, class_ratio_test, rng_sub_t)
    rng_t = np.random.default_rng(seed + 777)
    test_texts_mod = [
        inject_spurious_token_multiclass(t, int(l), test_p_correct, AMAZON_TOKENS, rng_t)
        for t, l in zip(test_texts, test_labels)
    ]
    X_test = tokenize_and_embed_with_bert(test_texts_mod, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(torch.from_numpy(X_test),
                   torch.from_numpy(test_labels.reshape(-1, 1).astype(np.float32)),
                   meta={"p_correct": test_p_correct, "kind": "amazon_semi_anti_causal_test_ood",
                         "n_samples": len(X_test)})

    print(f"\n✅ Amazon Books Semi Anti-Causal — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Selection bias par taille
# ─────────────────────────────────────────────────────────────────────────────
# DAG : Y → Z (longueur de la review) → S (sélection d'entraînement)
#
# Corrélation naturelle dans Amazon Books : les reviews utiles tendent à être
# plus longues (plus détaillées et structurées) que les reviews inutiles.
# Cette corrélation est SPURIEUSE pour la tâche : la longueur ne cause pas
# l'utilité, mais elle y est corrélée via Y → style d'écriture.
#
# Typique : inutile (0) COURT  (< Q1 des inutiles)
#           utile   (1) LONG   (> Q3 des utiles)
# OOD     : inutile (0) très LONG, utile (1) très COURT → signal spurieux inversé
# ─────────────────────────────────────────────────────────────────────────────

def build_envs_amazon_size_selection(
    train_p_select: List[float],
    seed: int = 1,
    threshold_method: str = "quartile",
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 512,
    device: str = "cpu",
    pooling: str = "mean",
    n_target: int = 100_000,
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    Biais de sélection par taille sur Amazon Books (utilité des reviews).

    Typique  : inutile (0) COURT (< Q1) ou utile (1) LONG (> Q3).
    OOD      : extrêmes opposés — inutile très long, utile très court.

    Parameters
    ----------
    threshold_method : "quartile" (défaut), "median", "soft".
    """
    print("Chargement Amazon Books (sélection par taille)...")
    all_texts, all_labels = load_amazon_books(seed=seed, n_target=n_target)
    n_total = len(all_texts)
    print(f"Dataset : {n_total} reviews")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_envs = len(train_p_select)
    spe = n_total // n_envs

    extreme_texts:  List[str] = []
    extreme_labels: List[int] = []
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        env_start  = i * spe
        env_end    = (i + 1) * spe if i < n_envs - 1 else n_total
        env_indices = indices[env_start:env_end]

        env_texts  = [all_texts[int(j)] for j in env_indices]
        env_labels = [all_labels[int(j)] for j in env_indices]

        t1, t2 = compute_size_thresholds(env_texts, env_labels, threshold_method)

        selected_texts:  List[str] = []
        selected_labels: List[int] = []

        for text, label in zip(env_texts, env_labels):
            text_len = len(text)
            if is_typical_by_size(text, label, t1, t2):
                if rng.uniform() < p_select:
                    selected_texts.append(text)
                    selected_labels.append(label)
            elif i == 0:
                # Extrêmes opposés : inutile très long OU utile très court
                if label == 0 and text_len > t2:
                    extreme_texts.append(text)
                    extreme_labels.append(label)
                elif label == 1 and text_len < t1:
                    extreme_texts.append(text)
                    extreme_labels.append(label)

        print(f"  Sélectionné : {len(selected_texts)} reviews typiques")

        sel_texts_sz  = selected_texts
        sel_labels_sz = np.array(selected_labels)
        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            sel_texts_sz, sel_labels_sz = _subsample_to_ratio(
                sel_texts_sz, sel_labels_sz, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub)

        n_sel = len(sel_texts_sz)
        n_val = int(n_sel * val_frac)
        idx_sh = rng.permutation(n_sel)
        tr_idx, va_idx = idx_sh[n_val:], idx_sh[:n_val]

        tr_texts  = [sel_texts_sz[j]  for j in tr_idx]
        tr_labels = sel_labels_sz[tr_idx].copy()
        if label_flip > 0.0:
            rng_tf = np.random.default_rng(seed + 7000 + i)
            tr_labels[rng_tf.uniform(size=len(tr_labels)) < label_flip] ^= 1
        X_tr = tokenize_and_embed_with_bert(tr_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(torch.from_numpy(X_tr),
                              torch.from_numpy(tr_labels.reshape(-1, 1).astype(np.float32)),
                              meta={"p_select": p_select, "kind": "amazon_size_selection_train",
                                    "env_id": i, "t1": t1, "t2": t2,
                                    "label_flip": label_flip, "n_samples": len(X_tr)}))

        va_texts  = [sel_texts_sz[j]  for j in va_idx]
        va_labels = sel_labels_sz[va_idx].copy()
        if label_flip > 0.0:
            rng_vf = np.random.default_rng(seed + 8000 + i)
            va_labels[rng_vf.uniform(size=len(va_labels)) < label_flip] ^= 1
        X_va = tokenize_and_embed_with_bert(va_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(torch.from_numpy(X_va),
                            torch.from_numpy(va_labels.reshape(-1, 1).astype(np.float32)),
                            meta={"p_select": p_select, "kind": "amazon_size_selection_val",
                                  "env_id": i, "label_flip": label_flip, "n_samples": len(X_va)}))

    print(f"\n=== Test OOD (extrêmes opposés) — {len(extreme_texts)} reviews ===")
    extreme_texts_final  = extreme_texts
    extreme_labels_arr   = np.array(extreme_labels)
    if class_ratio_test is not None:
        rng_sub_t = np.random.default_rng(seed + 22000)
        extreme_texts_final, extreme_labels_arr = _subsample_to_ratio(
            extreme_texts_final, extreme_labels_arr, class_ratio_test, rng_sub_t)
    X_test = tokenize_and_embed_with_bert(
        extreme_texts_final, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(torch.from_numpy(X_test),
                   torch.from_numpy(extreme_labels_arr.reshape(-1, 1).astype(np.float32)),
                   meta={"kind": "amazon_size_selection_test_ood",
                         "n_samples": len(X_test), "description": "extreme_opposite_by_size"})

    print(f"\n✅ Amazon Books Size Selection — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {sum(e.X.shape[0] for e in val_envs)} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Confounding variant 1 : varying proxy
# ─────────────────────────────────────────────────────────────────────────────

def build_envs_amazon_conf_varying_proxy(
    a_train: List[float],
    a_test: float,
    seed: int,
    p_c_flip: float = 0.25,
    gamma: float = 0.5,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 512,
    device: str = "cpu",
    pooling: str = "mean",
    n_target: int = 100_000,
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    Confounding avec proxy variable sur Amazon Books (utilité des reviews).

    DAG : C ~ Ber(p_c_flip) → Z = C ⊕ Ber(a_e) → token injecté ;
            C → Y (Y est poussé vers C) ; Text → Y.
    OOD : a_test = 1.0  ⇒  Z ⊥ C  ⇒  token non informatif.
    """
    all_texts, all_labels = load_amazon_books(seed=seed, n_target=n_target)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels, dtype=np.int64)

    rng_split = np.random.default_rng(seed)
    indices = rng_split.permutation(n_total)
    n_test = int(n_total * 0.1)
    n_val  = int(n_total * 0.1)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"Dataset : {n_total} reviews | Split 80/10/10 : "
          f"Train {len(train_idx)} | Val {len(val_idx)} | Test {len(test_idx)}")

    n_envs = len(a_train)
    spe = len(train_idx) // n_envs
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    val_texts  = [all_texts[int(j)] for j in val_idx]

    for i, a_e in enumerate(a_train):
        print(f"\n=== Train Env {i} (a={a_e}) ===")
        env_idx = train_idx[i * spe:(i + 1) * spe if i < n_envs - 1 else len(train_idx)]
        texts  = [all_texts[int(j)] for j in env_idx]
        labels = all_labels_arr[env_idx].copy()

        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            texts, labels = _subsample_to_ratio(texts, labels, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub)

        rng_e = np.random.default_rng(seed + i * 7)
        C = rng_e.binomial(1, p_c_flip, size=len(labels))
        N = rng_e.binomial(1, a_e, size=len(labels))
        Z = np.logical_xor(C, N).astype(int)

        X, Y = _conf_make_env(
            texts, labels.astype(np.float32), C, Z, gamma, rng_e,
            bert_model, max_length, device, pooling,
            apply_gamma=True, conf_tokens=_AMAZON_CONF_TOKENS,
         finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                              meta={"kind": "amazon_conf_varying_proxy", "a": a_e,
                                    "p_c_flip": p_c_flip,
                                    "split": "train", "env_id": i, "n_samples": len(X)}))

        print(f"=== Val Env {i} (a={a_e}) ===")
        val_texts_e  = list(val_texts)
        val_labels_e = all_labels_arr[val_idx].copy()
        if class_ratio_train is not None:
            rng_sub_v = np.random.default_rng(seed + 21000 + i)
            val_texts_e, val_labels_e = _subsample_to_ratio(
                val_texts_e, val_labels_e, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub_v)
        rng_v = np.random.default_rng(seed + 5000 + i)
        Cv = rng_v.binomial(1, p_c_flip, size=len(val_labels_e))
        Nv = rng_v.binomial(1, a_e, size=len(val_labels_e))
        Zv = np.logical_xor(Cv, Nv).astype(int)
        X_val, Y_val = _conf_make_env(
            val_texts_e, val_labels_e.astype(np.float32), Cv, Zv, gamma, rng_v,
            bert_model, max_length, device, pooling,
            apply_gamma=True, conf_tokens=_AMAZON_CONF_TOKENS,
         finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(torch.from_numpy(X_val), torch.from_numpy(Y_val),
                            meta={"kind": "amazon_conf_varying_proxy", "a": a_e,
                                  "p_c_flip": p_c_flip,
                                  "split": "val", "env_id": i, "n_samples": len(X_val)}))

    print(f"\n=== Test OOD (a={a_test}) ===")
    test_texts  = [all_texts[int(j)] for j in test_idx]
    test_labels = all_labels_arr[test_idx].copy()
    if class_ratio_test is not None:
        rng_sub_t = np.random.default_rng(seed + 22000)
        test_texts, test_labels = _subsample_to_ratio(
            test_texts, test_labels, class_ratio_test, rng_sub_t)
    rng_t = np.random.default_rng(seed + 777)
    Ct = rng_t.binomial(1, p_c_flip, size=len(test_labels))
    Nt = rng_t.binomial(1, a_test, size=len(test_labels))
    Zt = np.logical_xor(Ct, Nt).astype(int)
    X_test, Y_test = _conf_make_env(
        test_texts, test_labels.astype(np.float32), Ct, Zt, gamma, rng_t,
        bert_model, max_length, device, pooling,
        apply_gamma=False, conf_tokens=_AMAZON_CONF_TOKENS,
     finetune_bert_layers=finetune_bert_layers)
    test_env = Env(torch.from_numpy(X_test), torch.from_numpy(Y_test),
                   meta={"kind": "amazon_conf_varying_proxy", "a": a_test,
                         "p_c_flip": p_c_flip,
                         "split": "test_ood", "n_samples": len(X_test)})

    print(f"\n✅ Amazon Books Confounding varying proxy — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {val_envs[0].X.shape[0]} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Rating Natural (biais longueur naturel, envs = notes)
# ─────────────────────────────────────────────────────────────────────────────
# Env 0 : reviews 5★   → "long = utile" tient fortement
# Env 1 : reviews 3-4★ → corrélation longueur-utilité plus faible
# Test  : reviews 1★   → un avis bref négatif peut être très utile
#         → le raccourci longueur casse
# ─────────────────────────────────────────────────────────────────────────────

def _load_amazon_books_by_rating(
    seed: int = 42,
    n_per_group: int = 20_000,
    helpful_threshold: int = 5,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Charge Amazon Books en streaming et regroupe par note.

    Groupes : "5" (5★), "3-4" (3★ ou 4★), "1" (1★).
    Pour chaque groupe, collecte n_per_group//2 utiles + n_per_group//2 inutiles.

    Returns
    -------
    dict  : {"5": (texts, labels), "3-4": ..., "1": ...}
    """
    print(f"Chargement Amazon Books par note (streaming, "
          f"cible {n_per_group} par groupe, seuil helpful_vote ≥ {helpful_threshold})...")

    ds = load_dataset(
        "json",
        data_files="hf://datasets/McAuley-Lab/Amazon-Reviews-2023/"
                   "raw/review_categories/Books.jsonl",
        split="train",
        streaming=True,
    )

    target_half = n_per_group // 2
    # {group: {0: [...], 1: [...]}}
    buckets: Dict[str, Dict[int, List[str]]] = {
        "5":   {0: [], 1: []},
        "3-4": {0: [], 1: []},
        "1":   {0: [], 1: []},
    }

    def _group_full(g: str) -> bool:
        return (len(buckets[g][0]) >= target_half
                and len(buckets[g][1]) >= target_half)

    for row in ds:
        if all(_group_full(g) for g in buckets):
            break

        text = (row.get("text") or "").strip()
        if not text:
            continue

        rating = float(row.get("rating", 0) or 0)
        vote   = int(row.get("helpful_vote", 0) or 0)

        # Déterminer le groupe
        if rating == 5.0:
            grp = "5"
        elif rating in (3.0, 4.0):
            grp = "3-4"
        elif rating == 1.0:
            grp = "1"
        else:
            continue  # 2★ écarté

        if _group_full(grp):
            continue

        # Binariser l'utilité
        if vote == 0 and len(buckets[grp][0]) < target_half:
            buckets[grp][0].append(text)
        elif vote >= helpful_threshold and len(buckets[grp][1]) < target_half:
            buckets[grp][1].append(text)

    rng = np.random.default_rng(seed)
    result: Dict[str, Tuple[List[str], List[int]]] = {}
    for grp in ("5", "3-4", "1"):
        n0 = len(buckets[grp][0])
        n1 = len(buckets[grp][1])
        n  = min(n0, n1, target_half)
        texts  = buckets[grp][0][:n] + buckets[grp][1][:n]
        labels = [0] * n + [1] * n
        perm = rng.permutation(len(texts))
        texts  = [texts[int(i)]  for i in perm]
        labels = [labels[int(i)] for i in perm]
        result[grp] = (texts, labels)
        print(f"  Groupe {grp}★ : {n} inutiles + {n} utiles = {2*n}")

    return result


def build_envs_amazon_rating_natural(
    seed: int = 1,
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 512,
    device: str = "cpu",
    pooling: str = "mean",
    n_per_group: int = 20_000,
    helpful_threshold: int = 5,
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    Expérience naturelle : envs définis par la note (rating).

    Env 0 : reviews 5★   — corrélation longueur–utilité forte
    Env 1 : reviews 3-4★ — corrélation longueur–utilité modérée
    Test  : reviews 1★   — corrélation longueur–utilité faible / inversée
            (un avis négatif bref peut être très utile)

    Le label est toujours helpful_vote=0 → 0, helpful_vote≥5 → 1.
    Aucune manipulation synthétique : le biais est 100% naturel.
    """
    groups = _load_amazon_books_by_rating(
        seed=seed, n_per_group=n_per_group,
        helpful_threshold=helpful_threshold,
    )

    train_envs: List[Env] = []
    val_envs:   List[Env] = []
    rng = np.random.default_rng(seed)

    # ── Train / Val : 5★ (env 0) et 3-4★ (env 1) ──
    for i, grp in enumerate(("5", "3-4")):
        texts, labels_list = groups[grp]
        labels = np.array(labels_list)

        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            texts, labels = _subsample_to_ratio(texts, labels, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub)

        n = len(texts)
        n_val = int(n * val_frac)
        perm = rng.permutation(n)
        tr_idx, va_idx = perm[n_val:], perm[:n_val]

        # ─── Train ───
        tr_texts  = [texts[int(j)] for j in tr_idx]
        tr_labels = labels[tr_idx].copy()
        if label_flip > 0.0:
            rng_f = np.random.default_rng(seed + 7000 + i)
            tr_labels[rng_f.uniform(size=len(tr_labels)) < label_flip] ^= 1

        # Stats longueur
        lens_0 = [len(tr_texts[k]) for k in range(len(tr_texts)) if tr_labels[k] == 0]
        lens_1 = [len(tr_texts[k]) for k in range(len(tr_texts)) if tr_labels[k] == 1]
        print(f"\n=== Env {i} ({grp}★) — {len(tr_texts)} train ===")
        print(f"  Longueur inutiles : médiane={np.median(lens_0):.0f}, "
              f"moyenne={np.mean(lens_0):.0f}")
        print(f"  Longueur utiles   : médiane={np.median(lens_1):.0f}, "
              f"moyenne={np.mean(lens_1):.0f}")

        X_tr = tokenize_and_embed_with_bert(
            tr_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(
            torch.from_numpy(X_tr),
            torch.from_numpy(tr_labels.reshape(-1, 1).astype(np.float32)),
            meta={"kind": "amazon_rating_natural_train", "rating_group": grp,
                  "env_id": i, "label_flip": label_flip,
                  "n_samples": len(X_tr)}))

        # ─── Val ───
        va_texts  = [texts[int(j)] for j in va_idx]
        va_labels = labels[va_idx].copy()
        if label_flip > 0.0:
            rng_vf = np.random.default_rng(seed + 8000 + i)
            va_labels[rng_vf.uniform(size=len(va_labels)) < label_flip] ^= 1

        X_va = tokenize_and_embed_with_bert(
            va_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(
            torch.from_numpy(X_va),
            torch.from_numpy(va_labels.reshape(-1, 1).astype(np.float32)),
            meta={"kind": "amazon_rating_natural_val", "rating_group": grp,
                  "env_id": i, "n_samples": len(X_va)}))

    # ── Test OOD : 1★ ──
    test_texts, test_labels_list = groups["1"]
    test_labels = np.array(test_labels_list)
    if class_ratio_test is not None:
        rng_sub_t = np.random.default_rng(seed + 22000)
        test_texts, test_labels = _subsample_to_ratio(
            test_texts, test_labels, class_ratio_test, rng_sub_t)
    lens_0t = [len(test_texts[k]) for k in range(len(test_texts)) if test_labels[k] == 0]
    lens_1t = [len(test_texts[k]) for k in range(len(test_texts)) if test_labels[k] == 1]
    print(f"\n=== Test OOD (1★) — {len(test_texts)} reviews ===")
    print(f"  Longueur inutiles : médiane={np.median(lens_0t):.0f}, "
          f"moyenne={np.mean(lens_0t):.0f}")
    print(f"  Longueur utiles   : médiane={np.median(lens_1t):.0f}, "
          f"moyenne={np.mean(lens_1t):.0f}")

    X_test = tokenize_and_embed_with_bert(
        test_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(test_labels.reshape(-1, 1).astype(np.float32)),
        meta={"kind": "amazon_rating_natural_test_ood", "rating_group": "1",
              "n_samples": len(X_test)})

    print(f"\n✅ Amazon Books Rating Natural — Done!")
    print(f"   Env 0 (5★): {train_envs[0].X.shape[0]} train, {val_envs[0].X.shape[0]} val")
    print(f"   Env 1 (3-4★): {train_envs[1].X.shape[0]} train, {val_envs[1].X.shape[0]} val")
    print(f"   Test OOD (1★): {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Keyword Selection (lexique de recommandation)
# ─────────────────────────────────────────────────────────────────────────────
# DAG : Y → Z (présence de mots de recommandation) → S (sélection)
#
# Corrélation observée : les reviews utiles contiennent souvent un vocabulaire
# de recommandation explicite ("recommend", "must read", "page-turner"…).
# Cette corrélation est SPURIEUSE : ce n'est pas la recommandation qui cause
# l'utilité, c'est le contenu détaillé. Un "Must read! Amazing!" sans détail
# n'est pas utile ; une analyse factuelle sans mots de reco peut l'être.
#
# Typique : utile (1) AVEC mots de recommandation
#           OU inutile (0) SANS mots de recommandation
# OOD cross_label : inutile AVEC mots de recommandation (enthousiaste mais vide)
#                   + utile SANS mots de recommandation (factuel mais utile)
# ─────────────────────────────────────────────────────────────────────────────

AMAZON_RECO_WORDS: List[str] = [
    # Recommandation explicite
    "recommend", "highly recommend", "definitely recommend",
    "must read", "must-read",
    "page-turner", "page turner",
    "couldn't put it down", "could not put it down",
    # Qualité d'écriture
    "well written", "well-written",
    "beautifully written", "wonderfully written",
    # Jugement superlatif
    "one of the best", "best book", "best books",
    "five stars", "5 stars", "5 star",
    "loved this book", "loved this series", "love this book", "love this author",
    "thoroughly enjoyed", "truly enjoyed",
    # Qualificatifs d'immersion
    "captivating", "engrossing", "riveting", "gripping",
    "masterpiece",
]


def _is_typical_amazon(text: str, label: int) -> bool:
    """Typique = utile + mots de recommandation, OU inutile + sans mots de reco."""
    has_reco = any(w in text.lower() for w in AMAZON_RECO_WORDS)
    if label == 1:
        return has_reco       # utile + recommandation
    else:
        return not has_reco   # inutile + pas de recommandation


def _is_cross_label_amazon(text: str, label: int) -> bool:
    """Cross = les mots de recommandation contredisent le label."""
    has_reco = any(w in text.lower() for w in AMAZON_RECO_WORDS)
    if label == 0:
        return has_reco       # inutile AVEC mots de recommandation
    else:
        return not has_reco   # utile SANS mots de recommandation


def build_envs_amazon_keyword_selection(
    train_p_select: List[float],
    seed: int = 1,
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 512,
    device: str = "cpu",
    pooling: str = "mean",
    n_target: int = 100_000,
    ood_strategy: str = "cross_label",
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    Amazon Books — sélection par lexique de recommandation.

    DAG : Y → Z (présence de mots de recommandation) → S (sélection)

    Typique  : utile (1) + mots de reco, OU inutile (0) + pas de mots de reco.
    OOD cross_label : inutile + mots de reco (enthousiaste mais vide)
                      + utile + pas de mots de reco (factuel mais utile).
    OOD atypical    : reviews sans pattern clair (ni typique ni cross).

    Parameters
    ----------
    train_p_select : List[float]
        Proba de garder un exemple typique par env.
    ood_strategy : str
        "cross_label" (défaut) ou "atypical".
    n_target : int
        Nombre total de reviews à charger (équilibré 50/50).
    """
    print("Chargement Amazon Books (keyword selection)...")
    all_texts, all_labels = load_amazon_books(seed=seed, n_target=n_target)
    n_total = len(all_texts)
    print(f"Dataset : {n_total} reviews  |  OOD strategy : {ood_strategy}")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_envs = len(train_p_select)
    spe = n_total // n_envs

    ood_texts:  List[str] = []
    ood_labels: List[int] = []
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        env_start  = i * spe
        env_end    = (i + 1) * spe if i < n_envs - 1 else n_total
        env_indices = indices[env_start:env_end]

        env_texts  = [all_texts[int(j)]  for j in env_indices]
        env_labels = [all_labels[int(j)] for j in env_indices]

        selected_texts:  List[str] = []
        selected_labels: List[int] = []

        for text, label in zip(env_texts, env_labels):
            if _is_typical_amazon(text, label):
                if rng.uniform() < p_select:
                    selected_texts.append(text)
                    selected_labels.append(label)
            elif i == 0:
                if ood_strategy == 'cross_label':
                    if _is_cross_label_amazon(text, label):
                        ood_texts.append(text)
                        ood_labels.append(label)
                else:  # atypical
                    if not _is_cross_label_amazon(text, label):
                        ood_texts.append(text)
                        ood_labels.append(label)

        print(f"  Sélectionné : {len(selected_texts)} reviews typiques")

        sel_texts_arr  = selected_texts
        sel_labels_arr = np.array(selected_labels)
        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            sel_texts_arr, sel_labels_arr = _subsample_to_ratio(
                sel_texts_arr, sel_labels_arr, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub)

        n_sel = len(sel_texts_arr)
        n_val = int(n_sel * val_frac)
        idx_sh = rng.permutation(n_sel)
        tr_idx, va_idx = idx_sh[n_val:], idx_sh[:n_val]

        # ─── Train ───
        tr_texts  = [sel_texts_arr[j] for j in tr_idx]
        tr_labels = sel_labels_arr[tr_idx].copy()
        if label_flip > 0.0:
            rng_tf = np.random.default_rng(seed + 9000 + i)
            tr_labels[rng_tf.uniform(size=len(tr_labels)) < label_flip] ^= 1
        X_tr = tokenize_and_embed_with_bert(
            tr_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(
            torch.from_numpy(X_tr),
            torch.from_numpy(tr_labels.reshape(-1, 1).astype(np.float32)),
            meta={"p_select": p_select, "kind": "amazon_keyword_selection_train",
                  "env_id": i, "label_flip": label_flip, "n_samples": len(X_tr)}))

        # ─── Val ───
        va_texts  = [sel_texts_arr[j] for j in va_idx]
        va_labels = sel_labels_arr[va_idx].copy()
        if label_flip > 0.0:
            rng_vf = np.random.default_rng(seed + 10000 + i)
            va_labels[rng_vf.uniform(size=len(va_labels)) < label_flip] ^= 1
        X_va = tokenize_and_embed_with_bert(
            va_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(
            torch.from_numpy(X_va),
            torch.from_numpy(va_labels.reshape(-1, 1).astype(np.float32)),
            meta={"p_select": p_select, "kind": "amazon_keyword_selection_val",
                  "env_id": i, "n_samples": len(X_va)}))

    # ─── Test OOD ───
    print(f"\n=== Test OOD ({ood_strategy}) — {len(ood_texts)} reviews ===")
    ood_texts_final  = ood_texts
    ood_labels_arr   = np.array(ood_labels)
    if class_ratio_test is not None:
        rng_sub_t = np.random.default_rng(seed + 22000)
        ood_texts_final, ood_labels_arr = _subsample_to_ratio(
            ood_texts_final, ood_labels_arr, class_ratio_test, rng_sub_t)
    X_test = tokenize_and_embed_with_bert(
        ood_texts_final, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(ood_labels_arr.reshape(-1, 1).astype(np.float32)),
        meta={"kind": "amazon_keyword_selection_test_ood",
              "ood_strategy": ood_strategy, "n_samples": len(X_test)})

    print(f"\n✅ Amazon Books Keyword Selection — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {sum(e.X.shape[0] for e in val_envs)} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Sentiment Selection (note comme proxy trompeur d'utilité)
# ─────────────────────────────────────────────────────────────────────────────
# DAG : Y → Z (note ≥ 4 ou ≤ 2) → S (sélection)
#
# Corrélation spurieuse entre note et utilité :
#   - Typique : utile (1) + positif (4-5★)  OU  inutile (0) + négatif (1-2★)
#   - OOD     : utile (1) + négatif (1-2★)  OU  inutile (0) + positif (4-5★)
#
# Intuition : un modèle ERM apprend "sentiment positif → utile" comme raccourci.
# Ce raccourci casse au test OOD (reviews utiles mais négatives, reviews inutiles
# mais positives). IRM devrait ignorer le sentiment car la corrélation
# varie entre envs.
# ─────────────────────────────────────────────────────────────────────────────

def _load_amazon_books_with_rating(
    seed: int = 42,
    n_target: int = 100_000,
    helpful_threshold: int = 5,
) -> Tuple[List[str], List[int], List[float]]:
    """
    Charge Amazon Books avec rating, équilibré label × rating.

    Collecte dans 4 buckets (label × rating) puis équilibre pour avoir
    autant de 1★ que de 5★ et autant d'utiles que d'inutiles.

    Returns
    -------
    texts   : List[str]
    labels  : List[int]    – 0=inutile, 1=utile
    ratings : List[float]  – 1.0 ou 5.0
    """
    print(f"Chargement Amazon Books avec rating (streaming, cible {n_target}, "
          f"seuil helpful_vote ≥ {helpful_threshold})...")
    ds = load_dataset(
        "json",
        data_files="hf://datasets/McAuley-Lab/Amazon-Reviews-2023/"
                   "raw/review_categories/Books.jsonl",
        split="train",
        streaming=True,
    )

    # 4 buckets : (label, rating) = (0,1★), (0,5★), (1,1★), (1,5★)
    target_per_bucket = n_target // 4
    buckets: Dict[Tuple[int, float], List[str]] = {
        (0, 1.0): [], (0, 5.0): [],
        (1, 1.0): [], (1, 5.0): [],
    }

    for row in ds:
        if all(len(b) >= target_per_bucket for b in buckets.values()):
            break

        text = (row.get("text") or "").strip()
        if not text:
            continue
        vote = int(row.get("helpful_vote", 0) or 0)
        rating = float(row.get("rating", 0) or 0)

        if rating not in (1.0, 5.0):
            continue
        if vote >= 1 and vote < helpful_threshold:
            continue

        label = 1 if vote >= helpful_threshold else 0
        key = (label, rating)
        if len(buckets[key]) < target_per_bucket:
            buckets[key].append(text)

    # Équilibrer : min des 4 buckets
    n_per = min(len(b) for b in buckets.values())
    print(f"  Buckets : " + ", ".join(
        f"({'utile' if l else 'inutile'},{r:.0f}★)={len(buckets[(l,r)])}"
        for l in (0, 1) for r in (1.0, 5.0)))
    print(f"  Équilibré à {n_per} par bucket → {4 * n_per} total")

    rng = np.random.default_rng(seed)
    texts:   List[str]   = []
    labels:  List[int]   = []
    ratings: List[float] = []
    for key, bucket in buckets.items():
        rng.shuffle(bucket)
        for t in bucket[:n_per]:
            texts.append(t)
            labels.append(key[0])
            ratings.append(key[1])

    perm = rng.permutation(len(texts))
    texts   = [texts[int(i)]   for i in perm]
    labels  = [labels[int(i)]  for i in perm]
    ratings = [ratings[int(i)] for i in perm]

    n_pos = sum(1 for r in ratings if r == 5.0)
    n_neg = sum(1 for r in ratings if r == 1.0)
    print(f"  Notes : {n_pos} positives (5★), {n_neg} négatives (1★)")
    return texts, labels, ratings


def _is_typical_sentiment(label: int, rating: float) -> bool:
    """Typique = utile + 5★ OU inutile + 1★."""
    if label == 1:
        return rating == 5.0
    else:
        return rating == 1.0


def _is_cross_sentiment(label: int, rating: float) -> bool:
    """Cross = utile + 1★ OU inutile + 5★."""
    if label == 1:
        return rating == 1.0
    else:
        return rating == 5.0


def build_envs_amazon_sentiment_selection(
    train_p_select: List[float],
    seed: int = 1,
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 512,
    device: str = "cpu",
    pooling: str = "mean",
    n_target: int = 100_000,
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    Amazon Books — sélection par sentiment (note) comme proxy d'utilité.

    DAG : Y → Z (sentiment positif/négatif) → S (sélection)

    Typique  : utile (1) + positif (4-5★), OU inutile (0) + négatif (1-2★).
    OOD      : utile (1) + négatif (1-2★), OU inutile (0) + positif (4-5★).

    Le raccourci « sentiment positif ↔ utile » est très fort dans les envs
    de train mais casse au test OOD.

    Parameters
    ----------
    train_p_select : List[float]
        Proba de garder un exemple typique par env.
    n_target : int
        Nombre total de reviews à charger (équilibré 50/50 en label).
    """
    print("Chargement Amazon Books (sentiment selection)...")
    all_texts, all_labels, all_ratings = _load_amazon_books_with_rating(
        seed=seed, n_target=n_target,
    )
    n_total = len(all_texts)
    print(f"Dataset : {n_total} reviews")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_envs = len(train_p_select)
    spe = n_total // n_envs

    ood_texts:  List[str] = []
    ood_labels: List[int] = []
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        env_start = i * spe
        env_end   = (i + 1) * spe if i < n_envs - 1 else n_total
        env_indices = indices[env_start:env_end]

        env_texts   = [all_texts[int(j)]   for j in env_indices]
        env_labels  = [all_labels[int(j)]  for j in env_indices]
        env_ratings = [all_ratings[int(j)] for j in env_indices]

        selected_texts:  List[str] = []
        selected_labels: List[int] = []
        n_typical = 0
        n_cross   = 0

        for text, label, rating in zip(env_texts, env_labels, env_ratings):
            if _is_typical_sentiment(label, rating):
                n_typical += 1
                if rng.uniform() < p_select:
                    selected_texts.append(text)
                    selected_labels.append(label)
            elif _is_cross_sentiment(label, rating):
                n_cross += 1
                u = rng.uniform()
                if u < (1.0 - p_select):
                    # Cross inclus dans le train avec proba (1-p_select)
                    # → donne une corrélation spurieuse effective de p_select dans cet env
                    selected_texts.append(text)
                    selected_labels.append(label)
                elif i == 0:
                    # Cross de env 0 non sélectionnés pour le train → test OOD
                    ood_texts.append(text)
                    ood_labels.append(label)

        print(f"  Typiques dans partition : {n_typical}, Cross : {n_cross}")
        corr_effective = p_select  # P(typique | sélectionné) ≈ p_select (pool équilibré)
        print(f"  Corrélation spurieuse effective : {corr_effective:.0%}")
        print(f"  Sélectionné : {len(selected_texts)} reviews")

        sel_texts_arr  = selected_texts
        sel_labels_arr = np.array(selected_labels)
        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            sel_texts_arr, sel_labels_arr = _subsample_to_ratio(
                sel_texts_arr, sel_labels_arr, class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub)

        n_sel = len(sel_texts_arr)
        n_val = int(n_sel * val_frac)
        idx_sh = rng.permutation(n_sel)
        tr_idx, va_idx = idx_sh[n_val:], idx_sh[:n_val]

        # ─── Train ───
        tr_texts  = [sel_texts_arr[j] for j in tr_idx]
        tr_labels = sel_labels_arr[tr_idx].copy()
        if label_flip > 0.0:
            rng_tf = np.random.default_rng(seed + 9000 + i)
            tr_labels[rng_tf.uniform(size=len(tr_labels)) < label_flip] ^= 1
        X_tr = tokenize_and_embed_with_bert(
            tr_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(
            torch.from_numpy(X_tr),
            torch.from_numpy(tr_labels.reshape(-1, 1).astype(np.float32)),
            meta={"p_select": p_select, "kind": "amazon_sentiment_selection_train",
                  "env_id": i, "label_flip": label_flip, "n_samples": len(X_tr)}))

        # ─── Val ───
        va_texts  = [sel_texts_arr[j] for j in va_idx]
        va_labels = sel_labels_arr[va_idx].copy()
        if label_flip > 0.0:
            rng_vf = np.random.default_rng(seed + 10000 + i)
            va_labels[rng_vf.uniform(size=len(va_labels)) < label_flip] ^= 1
        X_va = tokenize_and_embed_with_bert(
            va_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(
            torch.from_numpy(X_va),
            torch.from_numpy(va_labels.reshape(-1, 1).astype(np.float32)),
            meta={"p_select": p_select, "kind": "amazon_sentiment_selection_val",
                  "env_id": i, "n_samples": len(X_va)}))

    # ─── Test OOD ───
    print(f"\n=== Test OOD (cross sentiment) — {len(ood_texts)} reviews ===")
    ood_texts_final  = ood_texts
    ood_labels_arr   = np.array(ood_labels)
    if class_ratio_test is not None:
        rng_sub_t = np.random.default_rng(seed + 22000)
        ood_texts_final, ood_labels_arr = _subsample_to_ratio(
            ood_texts_final, ood_labels_arr, class_ratio_test, rng_sub_t)
    n0_ood = int((ood_labels_arr == 0).sum())
    n1_ood = int((ood_labels_arr == 1).sum())
    print(f"  Inutiles+positif : {n0_ood}, Utiles+négatif : {n1_ood}")
    X_test = tokenize_and_embed_with_bert(
        ood_texts_final, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(ood_labels_arr.reshape(-1, 1).astype(np.float32)),
        meta={"kind": "amazon_sentiment_selection_test_ood",
              "n_samples": len(X_test)})

    print(f"\n✅ Amazon Books Sentiment Selection — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {sum(e.X.shape[0] for e in val_envs)} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# McAuley-Lab Amazon Reviews 2023 — Category Selection
# =============================================================================
# Dataset : McAuley-Lab/Amazon-Reviews-2023 (textes courts, catégories disponibles)
#
# DAG : Y → Z (catégorie) → S (sélection d'entraînement)
#
# Corrélation SPURIEUSE : une catégorie (ex : Electronics) est sur-représentée
# chez les positifs en entraînement, l'autre (ex : Movies & TV) chez les négatifs.
# Au test OOD la relation est INVERSÉE.
#
# Avantage : reviews plus courtes que IMDB (50-150 tokens) → le signal trompeur
# (catégorie) ne se noie pas. Catégories metadata pures (pas d'influence
# sémantique directe sur le sentiment).
#
# Typique (pool) : positif + cat_pos  OU  négatif + cat_neg
# OOD            : positif + cat_neg  OU  négatif + cat_pos
# =============================================================================

def load_amazon_reviews_by_category(
    seed: int = 42,
    cat_typical_pos: str = "Electronics",
    cat_typical_neg: str = "Movies_and_TV",
    n_target: int = 60_000,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Charge McAuley-Lab/Amazon-Reviews-2023 depuis deux catégories séparées.
    Binarise le label sur rating (1-2★ = négatif, 4-5★ = positif).

    Parameters
    ----------
    cat_typical_pos : str   Catégorie corrélée aux positifs (ex: "Electronics")
    cat_typical_neg : str   Catégorie corrélée aux négatifs (ex: "Movies_and_TV")
    n_target        : int   Nombre total cible de reviews (≤ disponible)

    Returns
    -------
    texts      : List[str]
    labels     : List[int]   – 0 = négatif (1-2★), 1 = positif (4-5★)
    categories : List[str]   – catégorie de chaque review
    """
    categories = [cat_typical_pos, cat_typical_neg]
    print(f"Chargement Amazon Reviews 2023 (catégories: {categories}, cible {n_target})...")

    target_per_cat = n_target // (len(categories) * 2)
    buckets: Dict[str, Dict[int, List[str]]] = {
        cat: {0: [], 1: []} for cat in categories
    }

    for cat in categories:
        # Charger le fichier .jsonl de cette catégorie
        try:
            ds = load_dataset(
                "json",
                data_files=f"hf://datasets/McAuley-Lab/Amazon-Reviews-2023/"
                           f"raw/review_categories/{cat}.jsonl",
                split="train",
                streaming=True,
            )
        except Exception as e:
            print(f"  ⚠ Catégorie '{cat}' non trouvée : {type(e).__name__}")
            continue

        for row in ds:
            if len(buckets[cat][0]) >= target_per_cat and len(buckets[cat][1]) >= target_per_cat:
                break

            text = (row.get("text") or "").strip()
            if not text or len(text) < 10:
                continue

            rating = float(row.get("rating", 0) or 0)
            if rating in (4.0, 5.0):
                label = 1  # positif
            elif rating in (1.0, 2.0):
                label = 0  # négatif
            else:
                continue  # 3★ = ambigu, écarté

            if len(buckets[cat][label]) < target_per_cat:
                buckets[cat][label].append(text)

    # Assembler le résultat
    texts_out:  List[str] = []
    labels_out: List[int] = []
    cats_out:   List[str] = []

    for cat in categories:
        for label in (0, 1):
            n = min(len(buckets[cat][label]), target_per_cat)
            texts_out.extend(buckets[cat][label][:n])
            labels_out.extend([label] * n)
            cats_out.extend([cat] * n)
            label_str = "positif (4-5★)" if label == 1 else "négatif (1-2★)"
            print(f"  {cat} | {label_str} : {n} reviews")

    # Shuffle
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(texts_out))
    texts_out  = [texts_out[int(i)]  for i in perm]
    labels_out = [labels_out[int(i)] for i in perm]
    cats_out   = [cats_out[int(i)]   for i in perm]

    print(f"  Total chargé : {len(texts_out)} reviews")
    return texts_out, labels_out, cats_out


def build_envs_amazon_category_selection(
    train_p_select: List[float],
    seed: int = 1,
    val_frac: float = 0.1,
    label_flip: float = 0.0,
    bert_model: str = "distilbert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    n_target: int = 60_000,
    cat_typical_pos: str = "Electronics",
    cat_typical_neg: str = "Movies_and_TV",
    class_ratio_train: Optional[List[float]] = None,
    class_ratio_test: Optional[float] = None,
    finetune_bert_layers: int = 0) -> Tuple[List[Env], List[Env], Env]:
    """
    Amazon Reviews Polarity 2013 — sélection par catégorie de produit (4-pool architecture).

    DAG : Y → Z (catégorie) → S (sélection d'entraînement)

    **Architecture 4-pool** :
    - Groupe 1 : positif + cat_typical_pos  → Z_typical, Y=1 (corrélation positive)
    - Groupe 2 : négatif + cat_typical_neg  → Z_typical, Y=0 (corrélation positive)
    - Groupe 3 : positif + cat_typical_neg  → Z_opposé, Y=1  (corrélation négative)
    - Groupe 4 : négatif + cat_typical_pos  → Z_opposé, Y=0  (corrélation négative)
    
    **Entraînement (Env i avec p_select[i])** :
      Mélange selon formula :
      - n_typ_pos_keep = int(len(typ_pos) * p_select)
      - n_typ_neg_keep = int(len(typ_neg) * (1 - p_select))
      - n_opp_pos_keep = int(len(opp_pos) * (1 - p_select))
      - n_opp_neg_keep = int(len(opp_neg) * p_select)
      
      Résultat : P(Y=1|Z=typique) = p_select exactement
                 P(Y=1|Z=opposé)  = 1 - p_select exactement
                 P(Y=1) global = ~50%
      
      Exemple : Env 0 avec p_select=0.9 → P(Y=1|Z_typical)=90%, P(Y=1|Z_opposé)=10%, P(Y)=50%
    
    **Test OOD** : 100% opposé = corrélation INVERSÉE COMPLÈTE
      → P(Y=1|Z_typique) = 0%
      → P(Y=1|Z_opposé) = 100%
      → P(Y=1) = 50%
      → IRM doit identifier que Y↔Z est spurieux (varie entre train et test) et l'ignorer

    Parameters
    ----------
    train_p_select   : List[float]           Fraction p_select pour chaque env train (contrôle corrélation)
    n_target         : int                   Nombre total de reviews à charger.
    cat_typical_pos  : str                   Catégorie corrélée aux positifs (train).
    cat_typical_neg  : str                   Catégorie corrélée aux négatifs (train).
    val_frac         : float                 Fraction validation.
    label_flip       : float                 Taux de bruit symétrique sur les labels.
    max_length       : int                   Max tokens BERT (défaut 128, adapté aux reviews courtes).
    class_ratio_train: Optional[List[float]] Fraction de positifs par env.
    class_ratio_test : Optional[float]       Fraction de positifs au test (devrait être ~0.5).
    """
    all_texts, all_labels, all_cats = load_amazon_reviews_by_category(
        seed=seed,
        cat_typical_pos=cat_typical_pos,
        cat_typical_neg=cat_typical_neg,
        n_target=n_target,
    )
    n_total = len(all_texts)
    print(f"Dataset : {n_total} reviews (2 catégories)")

    # **Architecture 4-pool** : Partitionner en (Y_label, Z_signal)
    # typ_pos   : Y=1, Z=typique (corrélation positive)
    # typ_neg   : Y=0, Z=typique (corrélation positive)
    # opp_pos   : Y=1, Z=opposé  (corrélation négative)
    # opp_neg   : Y=0, Z=opposé  (corrélation négative)
    
    typ_pos_texts:  List[str] = []
    typ_neg_texts:  List[str] = []
    opp_pos_texts:  List[str] = []
    opp_neg_texts:  List[str] = []

    for text, label, cat in zip(all_texts, all_labels, all_cats):
        if label == 1 and cat == cat_typical_pos:
            typ_pos_texts.append(text)
        elif label == 0 and cat == cat_typical_neg:
            typ_neg_texts.append(text)
        elif label == 1 and cat == cat_typical_neg:
            opp_pos_texts.append(text)
        elif label == 0 and cat == cat_typical_pos:
            opp_neg_texts.append(text)

    print(f"  Pool typ_pos  : {len(typ_pos_texts)} ({cat_typical_pos}, Y=1)")
    print(f"  Pool typ_neg  : {len(typ_neg_texts)} ({cat_typical_neg}, Y=0)")
    print(f"  Pool opp_pos  : {len(opp_pos_texts)} ({cat_typical_neg}, Y=1)")
    print(f"  Pool opp_neg  : {len(opp_neg_texts)} ({cat_typical_pos}, Y=0)")

    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        rng_env = np.random.default_rng(seed + 6000 + i)
        rng_mix = np.random.default_rng(seed + 6100 + i)

        # **Mélange 4-pool** : Sélectionner de chaque groupe selon p_select
        # Formule : P(Y=1|Z=typique) = p_select exactement
        #          P(Y=1|Z=opposé)  = 1 - p_select exactement
        #          P(Y=1) global = ~50%
        
        n_tp_keep = int(len(typ_pos_texts) * p_select)
        n_tn_keep = int(len(typ_neg_texts) * (1 - p_select))
        n_op_keep = int(len(opp_pos_texts) * (1 - p_select))
        n_on_keep = int(len(opp_neg_texts) * p_select)
        
        # Sélectionner aléatoirement de chaque groupe
        tp_idx = rng_mix.choice(len(typ_pos_texts), size=n_tp_keep, replace=False)
        tn_idx = rng_mix.choice(len(typ_neg_texts), size=n_tn_keep, replace=False)
        op_idx = rng_mix.choice(len(opp_pos_texts), size=n_op_keep, replace=False)
        on_idx = rng_mix.choice(len(opp_neg_texts), size=n_on_keep, replace=False)
        
        # Construire l'env
        selected_texts = ([typ_pos_texts[j] for j in tp_idx] +
                         [typ_neg_texts[j] for j in tn_idx] +
                         [opp_pos_texts[j] for j in op_idx] +
                         [opp_neg_texts[j] for j in on_idx])
        selected_labels = np.array([1]*n_tp_keep + [0]*n_tn_keep + 
                                  [1]*n_op_keep + [0]*n_on_keep, dtype=np.int32)
        
        # Calcul des corrélations exactes
        p_pos_given_typ = n_tp_keep / (n_tp_keep + n_tn_keep) if (n_tp_keep + n_tn_keep) > 0 else 0.5
        p_pos_given_opp = n_op_keep / (n_op_keep + n_on_keep) if (n_op_keep + n_on_keep) > 0 else 0.5
        p_global = (n_tp_keep + n_op_keep) / len(selected_labels) if len(selected_labels) > 0 else 0.5
        
        print(f"  Mélange 4-pool : {n_tp_keep} typ_pos + {n_tn_keep} typ_neg + {n_op_keep} opp_pos + {n_on_keep} opp_neg")
        print(f"  P(Y=1|Z=typique) = {p_pos_given_typ:.1%} (cible: {p_select:.0%})")
        print(f"  P(Y=1|Z=opposé)  = {p_pos_given_opp:.1%} (cible: {1-p_select:.0%})")
        print(f"  P(Y=1) global    = {p_global:.1%}")

        sel_texts  = selected_texts
        sel_labels = selected_labels
        if class_ratio_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            sel_texts, sel_labels = _subsample_to_ratio(
                sel_texts, sel_labels,
                class_ratio_train[min(i, len(class_ratio_train) - 1)], rng_sub)

        n_sel = len(sel_texts)
        n_val = int(n_sel * val_frac)
        idx_sh = rng_env.permutation(n_sel)
        tr_idx, va_idx = idx_sh[n_val:], idx_sh[:n_val]

        tr_texts  = [sel_texts[j]  for j in tr_idx]
        tr_labels = sel_labels[tr_idx].copy()
        if label_flip > 0.0:
            rng_tf = np.random.default_rng(seed + 7000 + i)
            tr_labels[rng_tf.uniform(size=len(tr_labels)) < label_flip] ^= 1
        X_tr = tokenize_and_embed_with_bert(
            tr_texts, bert_model, max_length, device, pooling,
            finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(
            torch.from_numpy(X_tr),
            torch.from_numpy(tr_labels.reshape(-1, 1).astype(np.float32)),
            meta={"p_select": p_select, "kind": "amazon_category_selection_train",
                  "env_id": i, "label_flip": label_flip, "n_samples": len(X_tr)}))

        va_texts  = [sel_texts[j]  for j in va_idx]
        va_labels = sel_labels[va_idx].copy()
        if label_flip > 0.0:
            rng_vf = np.random.default_rng(seed + 8000 + i)
            va_labels[rng_vf.uniform(size=len(va_labels)) < label_flip] ^= 1
        X_va = tokenize_and_embed_with_bert(
            va_texts, bert_model, max_length, device, pooling,
            finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(
            torch.from_numpy(X_va),
            torch.from_numpy(va_labels.reshape(-1, 1).astype(np.float32)),
            meta={"p_select": p_select, "kind": "amazon_category_selection_val",
                  "env_id": i, "label_flip": label_flip, "n_samples": len(X_va)}))

    print(f"\n=== Test OOD (100% inversé) ===")
    
    # Test OOD : 100% opposé = corrélation INVERSÉE COMPLÈTE
    # P(Y=1|Z=typique) = 0%  P(Y=1|Z=opposé) = 100%
    rng_test = np.random.default_rng(seed + 25000)
    
    # Sélectionner tous les opposés
    all_opp_pos_idx = rng_test.permutation(len(opp_pos_texts))
    all_opp_neg_idx = rng_test.permutation(len(opp_neg_texts))
    
    ood_texts_final = ([opp_pos_texts[j] for j in all_opp_pos_idx] +
                       [opp_neg_texts[j] for j in all_opp_neg_idx])
    ood_labels_arr  = np.array([1]*len(all_opp_pos_idx) + [0]*len(all_opp_neg_idx), dtype=np.int32)
    
    # Shuffle
    perm = rng_test.permutation(len(ood_texts_final))
    ood_texts_final = [ood_texts_final[int(j)] for j in perm]
    ood_labels_arr = ood_labels_arr[perm]
    
    p_pos_opp = np.mean(ood_labels_arr)
    print(f"  Composition : 100% opposé ({len(ood_texts_final)} reviews)")
    print(f"  P(Y=1|Z=opposé) = {p_pos_opp:.1%} (should be ~50% after balance)")
    
    # Rééquilibrer stratifié si ratio classe fourni
    if class_ratio_test is not None and abs(class_ratio_test - 0.5) > 1e-6:
        rng_sub_t = np.random.default_rng(seed + 22000)
        ood_texts_final, ood_labels_arr = _subsample_to_ratio(
            ood_texts_final, ood_labels_arr, class_ratio_test, rng_sub_t)
    else:
        # Si ratio_test non fourni, équilibrer à 50/50 (corrélation inversée complète)
        pos_idx_ood = np.where(ood_labels_arr == 1)[0]
        neg_idx_ood = np.where(ood_labels_arr == 0)[0]
        n_pos_ood = len(pos_idx_ood)
        n_neg_ood = len(neg_idx_ood)
        
        n_keep = min(n_pos_ood, n_neg_ood)
        rng_bal = np.random.default_rng(seed + 25100)
        pos_kept = rng_bal.choice(pos_idx_ood, size=n_keep, replace=False)
        neg_kept = rng_bal.choice(neg_idx_ood, size=n_keep, replace=False)
        
        balanced_idx = rng_bal.permutation(np.concatenate([pos_kept, neg_kept]))
        ood_texts_final = [ood_texts_final[int(j)] for j in balanced_idx]
        ood_labels_arr = ood_labels_arr[balanced_idx]

    print(f"  Après équilibrage : {len(ood_texts_final)} reviews, "
          f"P(Y=1) = {float(ood_labels_arr.mean()):.1%}")

    X_test = tokenize_and_embed_with_bert(
        ood_texts_final, bert_model, max_length, device, pooling,
        finetune_bert_layers=finetune_bert_layers)
    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(ood_labels_arr.reshape(-1, 1).astype(np.float32)),
        meta={"kind": "amazon_category_selection_test_ood",
              "n_samples": len(X_test),
              "description": "100_percent_inverted_correlation"})

    print(f"\n✅ Amazon Category Selection — Done!")
    print(f"   Train : {sum(e.X.shape[0] for e in train_envs)} | "
          f"Val : {sum(e.X.shape[0] for e in val_envs)} | Test : {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env
