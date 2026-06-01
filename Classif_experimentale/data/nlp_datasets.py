from __future__ import annotations

import sys
from pathlib import Path as _Path
# Add project root and shared/ to Python path
_ROOT = _Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "irm") not in sys.path:
    sys.path.insert(0, str(_ROOT / "irm"))

# NLP environment builders for IRM experiments (AG News, IMDB Genres, Amazon Books).

from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import re
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from env import Env

# =============================================================================
# Spurious token configuration
# =============================================================================

def define_spurious_tokens() -> Dict[str, str]:
    """
    Return the spurious token dictionary for binary datasets.

    Tokens: "sky" (ham/negative) and "fire" (spam/positive).
    Chosen to be semantically neutral and tokenised as single WordPiece tokens.
    """
    return {
        "spam_correlated": "fire",
        "ham_correlated":  "sky",
    }

# =============================================================================
# SMS Spam dataset loader
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

    # Load dataset (~59k messages)
    dataset = load_dataset("mshenoda/spam-messages")

    # Merge all splits
    all_data = concatenate_datasets([
        dataset['train'],
        dataset['validation'],
        dataset['test']
    ])

    # Shuffle with fixed seed
    all_data = all_data.shuffle(seed=seed)

    # Extract texts and convert string labels to int
    texts = all_data['text']
    labels = [1 if label == 'spam' else 0 for label in all_data['label']]

    return texts, labels

# =============================================================================
# Spurious token injection (SAC mechanism)
# =============================================================================

# High-frequency, semantically empty words present across all three datasets.
# The spurious token is inserted before each occurrence, distributing the signal
# throughout the text (harder for ERM to ignore than a single prefix).
NEUTRAL_WORDS: List[str] = [
    "the", "a", "an", "of", "in", "to", "is", "it",
    "and", "or", "at", "on", "for", "with", "as", "by",
]

def _prepend_token_to_neutral_words(text: str, neutral_words: List[str], token: str) -> Optional[str]:
    """
    Insert `token` before each occurrence of a neutral word in `text` (word boundaries).

    Ex: token="fire", text="I went to the store and it was great"
        -> "I went to fire the store fire and fire it was great"

    Returns None if no neutral word is found (caller falls back to sentence prefix).
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
    Inject a spurious text token into `text` with controlled label correlation.

    Mode 1 (default): insert token before each neutral word occurrence.
      The signal is distributed across the text, harder for ERM to ignore.
      Used when `neutral_words` is provided and the text contains matches.

    Mode 2 (fallback): sentence prefix/suffix when no neutral word is found.

    Parameters
    ----------
    text : str
    label : int               True label (0=ham, 1=spam).
    p_correct : float         P(token matches true label).
    spurious_tokens : dict    {"spam_correlated": tok, "ham_correlated": tok}.
    rng : np.random.Generator
    position : str            "prefix" or "suffix" (fallback only).
    neutral_words : List[str] or None
        If None, force sentence-prefix mode.
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

    # Fallback: sentence prefix/suffix
    return f"{token} {text}" if position == "prefix" else f"{text} {token}"

# =============================================================================
# BERT embedding extraction
# =============================================================================

_BERT_CACHE: Dict[str, Any] = {}

def _get_bert(model_name: str, device: str):
    """Return (tokenizer, model), loading from disk only once per model_name."""
    if model_name not in _BERT_CACHE:
        print(f"  [BERT] Loading {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        _BERT_CACHE[model_name] = {"tokenizer": tokenizer, "model": model}

    entry = _BERT_CACHE[model_name]
    # Move to target device if needed
    current_device = next(entry["model"].parameters()).device
    if str(current_device) != str(device):
        entry["model"] = entry["model"].to(device)
    return entry["tokenizer"], entry["model"]

# =============================================================================
# On-disk embedding cache
# =============================================================================

def _embed_cache_path(texts: List[str], model_name: str,
                      max_length: int, pooling: str) -> str:
    """Compute a unique cache path based on a hash of texts + embedding hyperparameters."""
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
    Tokenize texts and extract BERT embeddings.

    Optimisations:
    - Singleton BERT: model loaded once per process.
    - Disk cache: embeddings saved to nlp_synthetic/.embed_cache/<hash>.npy.
    - Adaptive batch size: 256 on GPU/MPS, 64 on CPU.

    Parameters
    ----------
    texts : List[str]
    model_name : str
    max_length : int
    device : str
    pooling : str      "mean" | "cls" | "max"
    use_cache : bool   Set False to force recomputation.
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

    # Disk cache
    if use_cache:
        cache_path = _embed_cache_path(texts, model_name, max_length, pooling)
        if os.path.exists(cache_path):
            return np.load(cache_path)

    # BERT model (singleton)
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

    # Save to cache
    if use_cache:
        np.save(cache_path, result)
        print(f"  [cache] Saved embeddings -> {os.path.basename(cache_path)}")

    return result

# Noms lisibles des classes AG News (pour logs et diagnostics)
AG_NEWS_CLASS_NAMES: Dict[int, str] = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech",
}

# =============================================================================
# AG News dataset loader (4 classes)
# =============================================================================

def load_ag_news_dataset(seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Load AG News from Hugging Face (fancyzhx/ag_news, 4 classes).

    Merges train and test splits (135 200 articles total).
    Labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech.
    """
    from datasets import concatenate_datasets

    dataset = load_dataset("fancyzhx/ag_news")
    all_data = concatenate_datasets([dataset["train"], dataset["test"]])
    all_data = all_data.shuffle(seed=seed)

    texts = list(all_data["text"])
    labels = list(all_data["label"])  # already 0-3

    return texts, labels

# =============================================================================
# AG News — Size-based selection (4 global length bins)
# =============================================================================
#
# DAG: Y (class) -> Z (length bin) -> S (train selection)
#
# 4 global length bins (corpus-level quartiles), one canonical bin per class:
#   Class 0 World    -> bin 0: length <= P25   (very short)
#   Class 1 Sports   -> bin 1: P25 < length <= P50
#   Class 2 Business -> bin 2: P50 < length <= P75
#   Class 3 Sci/Tech -> bin 3: length > P75   (very long)
#
# Typical   for class c: article in bin c -> over-sampled in train with prob p_select
# Atypical  for class c: article outside  -> held out as OOD pool
#
# Creates a distinct spurious Z-Y correlation per class (analogous to the
# 4 tokens in the SAC experiment).
# =============================================================================

# AG News class -> canonical length bin (global quartile)
AG_NEWS_CLASS_TO_LENGTH_BIN: Dict[int, int] = {
    0: 0,   # World    -> very short  (<= P25)
    1: 1,   # Sports   -> short-mid   (P25-P50)
    2: 2,   # Business -> mid-long    (P50-P75)
    3: 3,   # Sci/Tech -> very long   (> P75)
}

# OOD bin per class: the most distant bin
#   bins 0,1 (short) -> OOD in bin 3 (> P75, very long)
#   bins 2,3 (long)  -> OOD in bin 0 (<= P25, very short)
AG_NEWS_CLASS_TO_OOD_BIN: Dict[int, int] = {
    0: 3,   # World    (bin 0) -> OOD bin 3
    1: 3,   # Sports   (bin 1) -> OOD bin 3
    2: 0,   # Business (bin 2) -> OOD bin 0
    3: 0,   # Sci/Tech (bin 3) -> OOD bin 0
}

def build_envs_ag_news_size_selection(
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
    AG News (4-class) — size-based selection environments.

    DAG: Y (class) -> Z (global length bin) -> S (train selection)

    Each class maps to a distinct global length bin (corpus-level quartiles).
    Train: P(article in canonical bin | label=c) = p_select -> Z-Y correlation.
    OOD  : articles in the opposite bin -> ERM biased, IRM expected to generalise.

    Parameters
    ----------
    train_p_select : List[float]
        e.g. [0.9, 0.7] -> env 0: 90% typical, env 1: 70% typical.
    threshold_method : str
        "quartile" (P25/P50/P75, default), "tertile", "quintile".
    n_ood_per_class : int
        Max OOD articles per class.
    """
    print("Loading AG News dataset (size-based selection, 4 global bins)...")
    all_texts, all_labels = load_ag_news_dataset(seed=seed)
    all_lengths = [len(t) for t in all_texts]
    print(f"Loaded {len(all_texts)} articles")

    # ── Seuils globaux de longueur (3 seuils → 4 bins) ───────────────────
    if threshold_method == "quartile":
        pcts = [25, 50, 75]
    elif threshold_method == "tertile":
        pcts = [25, 50, 75]   # default: same as quartile (tertile would be 33/67, 3 bins)
    elif threshold_method == "quintile":
        pcts = [20, 40, 60, 80]
        raise ValueError("quintile produces 5 bins but we only have 4 classes.")
    else:
        pcts = [25, 50, 75]

    thresholds = [float(np.percentile(all_lengths, p)) for p in pcts]
    print(f"  Global thresholds: P{pcts[0]}={thresholds[0]:.0f}c, "
          f"P{pcts[1]}={thresholds[1]:.0f}c, P{pcts[2]}={thresholds[2]:.0f}c")

    def _length_bin(length: int) -> int:
        if length <= thresholds[0]:
            return 0
        elif length <= thresholds[1]:
            return 1
        elif length <= thresholds[2]:
            return 2
        else:
            return 3

    # Partition articles into: typical (canonical bin), atypical-train, and OOD pool
    # typical_by_class[c]     : bin == canonical bin -> train
    # atyp_train_by_class[c]  : other bins (not OOD) -> train minority
    # ood_by_class[c]         : opposite bin         -> OOD test pool
    # Pre-flip labels globally so that selection is based on noisy Y.
    # Z must align with ~Y so the spurious signal stays strong as noise grows.
    all_noisy_labels = list(all_labels)
    if label_flip > 0.0:
        rng_global_flip = np.random.default_rng(seed + 4999)
        gfm = rng_global_flip.uniform(size=len(all_noisy_labels)) < label_flip
        for k in np.where(gfm)[0]:
            others = [c for c in range(4) if c != all_noisy_labels[k]]
            all_noisy_labels[k] = int(rng_global_flip.choice(others))
        print(f"  Label flip (global, before selection): "
              f"{int(gfm.sum())}/{len(all_noisy_labels)} ({gfm.mean():.1%})")

    # Training partitions: NOISY labels (Z aligned to ~Y)
    # OOD partition      : TRUE  labels (evaluation on Y)
    typique_by_class:    Dict[int, List[int]] = {c: [] for c in range(4)}
    atyp_train_by_class: Dict[int, List[int]] = {c: [] for c in range(4)}
    ood_by_class:        Dict[int, List[int]] = {c: [] for c in range(4)}

    for idx, (true_label, noisy_label, length) in enumerate(
            zip(all_labels, all_noisy_labels, all_lengths)):
        b = _length_bin(length)
        # Training: use noisy label to decide typical/atypical
        bin_c_noisy = AG_NEWS_CLASS_TO_LENGTH_BIN[noisy_label]
        ood_b_noisy = AG_NEWS_CLASS_TO_OOD_BIN[noisy_label]
        if b == bin_c_noisy:
            typique_by_class[noisy_label].append(idx)
        elif b != ood_b_noisy:
            atyp_train_by_class[noisy_label].append(idx)
        # OOD: use true label
        ood_b_true = AG_NEWS_CLASS_TO_OOD_BIN[true_label]
        if b == ood_b_true:
            ood_by_class[true_label].append(idx)

    for c in range(4):
        bin_c = AG_NEWS_CLASS_TO_LENGTH_BIN[c]
        ood_b = AG_NEWS_CLASS_TO_OOD_BIN[c]
        print(f"  Classe {AG_NEWS_CLASS_NAMES[c]:10s} (bin {bin_c}): "
              f"typiques={len(typique_by_class[c])}, "
              f"atyp_train={len(atyp_train_by_class[c])}, "
              f"ood(bin {ood_b})={len(ood_by_class[c])}")

    rng = np.random.default_rng(seed)
    n_envs = len(train_p_select)

    # Build train/val environments
    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        rng_env = np.random.default_rng(seed + 5000 + i)

        sel_texts:  List[str] = []
        sel_labels: List[int] = []

        for c in range(4):
            # Round-robin slices to avoid overlap between environments
            typ_env  = typique_by_class[c][i::n_envs]
            atyp_env = atyp_train_by_class[c][i::n_envs]

            n_typ_take  = int(len(typ_env)  * p_select)
            n_atyp_take = int(len(atyp_env) * (1.0 - p_select))

            if n_typ_take > 0:
                typ_pos = rng_env.choice(len(typ_env), size=n_typ_take, replace=False)
                for j in typ_pos:
                    sel_texts.append(all_texts[typ_env[j]])
                    sel_labels.append(c)

            if n_atyp_take > 0:
                atyp_pos = rng_env.choice(len(atyp_env), size=n_atyp_take, replace=False)
                for j in atyp_pos:
                    sel_texts.append(all_texts[atyp_env[j]])
                    sel_labels.append(c)

        sel_arr = np.array(sel_labels)
        dist = {AG_NEWS_CLASS_NAMES[c]: int((sel_arr == c).sum()) for c in range(4)}
        print(f"  Selected {len(sel_texts)} -> {dist}")

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
            # labels_e are already noisy (flip applied globally before selection)
            X = tokenize_and_embed_with_bert(
                texts_e, bert_model, max_length, device, pooling,
                finetune_bert_layers=finetune_bert_layers,
            )
            return Env(
                torch.from_numpy(X),
                torch.from_numpy(labels_e),
                meta={
                    "p_select": p_sel,
                    "kind": f"ag_news_size_selection_{kind}",
                    "env_id": env_i,
                    "n_classes": 4,
                    "label_flip": label_flip,
                    "n_samples": len(X),
                },
            )

        train_envs.append(_make_env(tr_idx,  "train"))
        val_envs.append(  _make_env(val_idx, "val"))

    # OOD test: articles in the opposite bin (never seen during training)
    print(f"\n=== OOD test (opposite bin, max {n_ood_per_class}/class) ===")
    for c in range(4):
        print(f"  Classe {AG_NEWS_CLASS_NAMES[c]:10s}: "
              f"bin train={AG_NEWS_CLASS_TO_LENGTH_BIN[c]} → "
              f"bin OOD={AG_NEWS_CLASS_TO_OOD_BIN[c]} "
              f"({len(ood_by_class[c])} candidats)")

    ood_texts_final:  List[str] = []
    ood_labels_final: List[int] = []

    rng_ood = np.random.default_rng(seed + 25000)  # dedicated rng, independent of train envs
    for c in range(4):
        pool   = ood_by_class[c]
        n_take = min(len(pool), n_ood_per_class)
        chosen = rng_ood.choice(len(pool), size=n_take, replace=False)
        for j in chosen:
            ood_texts_final.append(all_texts[pool[j]])
            ood_labels_final.append(c)

    ood_arr  = np.array(ood_labels_final)
    ood_dist = {AG_NEWS_CLASS_NAMES[c]: int((ood_arr == c).sum()) for c in range(4)}
    print(f"  {len(ood_texts_final)} OOD articles -> {ood_dist}")
    print(f"  Z (length bin) points to wrong class -> ERM maximally biased.")

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
            "kind": "ag_news_size_selection_ood",
            "n_classes": 4,
            "n_samples": len(X_ood),
            "description": "atypical_length_bin_articles",
        },
    )

    print(f"\nAG News Size Selection (4 global bins) done.")
    print(f"   train: {len(train_envs)} envs, {sum(e.X.shape[0] for e in train_envs)} samples")
    print(f"   val  : {len(val_envs)} envs, {sum(e.X.shape[0] for e in val_envs)} samples")
    print(f"   test : {test_env.X.shape[0]} OOD samples (max {n_ood_per_class}/class)")

    return train_envs, val_envs, test_env

# =============================================================================
# AG News — Semi anti-causal (token injection, 4 classes)
# =============================================================================
#
# DAG: Y (true class) -> Z (injected token) -> X = BERT(text + Z)
#
# One class token is injected per article:
#   - with prob p_correct  : the token for the true label
#   - with prob 1-p_correct: a token sampled uniformly from the other 3 classes
#
# Train : p_correct in {0.9, 0.8} -> strong Z-Y correlation
# OOD   : p_correct = 0.0 -> token always wrong -> ERM fails
# =============================================================================

# AG News spurious tokens — primary colours, no association with news topics.
# Each colour is a single DistilBERT token with a well-defined embedding.
AG_NEWS_TOKENS: Dict[int, str] = {
    0: "red",    # World
    1: "blue",   # Sports
    2: "green",  # Business
    3: "yellow", # Sci/Tech
}

# Fixed wrong-class map for SAC with a single wrong token per class.
# Cyclic permutation: World->Sports->Business->Sci/Tech->World
AG_NEWS_WRONG_CLASS: Dict[int, int] = {
    0: 1,  # World    -> Sports  (blue)
    1: 2,  # Sports   -> Business (green)
    2: 3,  # Business -> Sci/Tech (yellow)
    3: 0,  # Sci/Tech -> World   (red)
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
    Inject a spurious token for multi-class classification (N classes).

    - With prob p_correct  : inject class_tokens[label].
    - With prob 1-p_correct:
        - If wrong_class_map is provided: use class_tokens[wrong_class_map[label]].
        - Otherwise: sample uniformly from the other (N-1) class tokens.

    By default, inserts the token before each neutral word (distributed signal).
    Falls back to sentence prefix if no neutral word is found.
    Pass neutral_words=None to force sentence-prefix mode.
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
    AG News (4-class) — semi anti-causal environments (token injection).

    DAG: Y -> Z (token) -> X = BERT(text + Z)

    One class token is injected per article before BERT encoding.
    The Z-Y correlation is controlled by p_correct per environment.
    ERM that exploits Z fails on OOD (token always wrong).
    IRM must learn to ignore Z.

    Global split: 80% train | 10% val | 10% test.
    Each value in train_p_correct generates one training environment.

    Parameters
    ----------
    train_p_correct : List[float]  e.g. [0.9, 0.8]
    test_p_correct  : float        typically 0.0 (token always wrong)
    seed            : int
    label_flip      : float        fraction of noisy labels in train
    bert_model      : str
    max_length      : int
    device          : str
    pooling         : str
    """
    print("Loading AG News dataset...")
    all_texts, all_labels = load_ag_news_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels)
    print(f"Loaded {n_total} articles (4 classes)")
    class_dist = {AG_NEWS_CLASS_NAMES[c]: int((all_labels_arr == c).sum()) for c in range(4)}
    print(f"  Class distribution: {class_dist}")

    # Global 80/10/10 split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test_split = int(n_total * 0.1)
    n_val_split  = int(n_total * 0.1)
    test_indices  = indices[:n_test_split]
    val_indices   = indices[n_test_split:n_test_split + n_val_split]
    train_indices = indices[n_test_split + n_val_split:]

    print(f"  Split: train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}")

    # Global label flip applied once before env split, so envs share the same
    # noise pattern (prevents IRM from exploiting env-specific noise).
    all_labels_train = np.array([all_labels[int(j)] for j in train_indices], dtype=np.int64)
    if label_flip > 0:
        rng_flip_global = np.random.default_rng(seed + 999)
        flip_mask_global = rng_flip_global.uniform(size=len(all_labels_train)) < label_flip
        for k in np.where(flip_mask_global)[0]:
            others = [c for c in range(4) if c != all_labels_train[k]]
            all_labels_train[k] = int(rng_flip_global.choice(others))
        n_flipped = int(flip_mask_global.sum())
        print(f"  Label flip: {n_flipped}/{len(all_labels_train)} "
              f"({n_flipped/len(all_labels_train):.1%}) noisy labels")

    n_envs = len(train_p_correct)
    samples_per_env = len(train_indices) // n_envs

    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    # Train and val environments
    for i, p_correct in enumerate(train_p_correct):
        print(f"\n=== Train Env {i} (p_correct={p_correct:.0%}) ===")
        start   = i * samples_per_env
        end     = (i + 1) * samples_per_env if i < n_envs - 1 else len(train_indices)
        env_idx = train_indices[start:end]

        texts  = [all_texts[int(j)]  for j in env_idx]
        # Labels from the globally pre-flipped array
        labels = all_labels_train[start:end].copy()

        if class_dist_train is not None:
            rng_sub = np.random.default_rng(seed + 20000 + i)
            texts, labels = _subsample_to_class_dist(texts, labels, class_dist_train[i], rng_sub)

        # Inject spurious token
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
        print(f"  Correct token: {n_correct}/{len(labels)} ({n_correct/len(labels):.1%})")

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

        # Val environment (same p_correct as train)
        print(f"=== Val Env {i} (p_correct={p_correct:.0%}) ===")
        val_texts  = [all_texts[int(j)]  for j in val_indices]
        val_labels = np.array([all_labels[int(j)] for j in val_indices], dtype=np.int64)

        # Global label flip for val (same seed for all envs)
        if label_flip > 0:
            rng_val_flip = np.random.default_rng(seed + 5999)
            flip_mask_val = rng_val_flip.uniform(size=len(val_labels)) < label_flip
            for k in np.where(flip_mask_val)[0]:
                others = [c for c in range(4) if c != val_labels[k]]
                val_labels[k] = int(rng_val_flip.choice(others))

        if class_dist_train is not None:
            rng_sub_v = np.random.default_rng(seed + 21000 + i)
            val_texts, val_labels = _subsample_to_class_dist(val_texts, val_labels, class_dist_train[i], rng_sub_v)

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

    # OOD test environment
    print(f"\n=== OOD test (p_correct={test_p_correct:.0%}) ===")
    if test_p_correct == 0.0:
        print("  Token always wrong -> ERM fails, IRM expected to be robust.")

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

    print(f"\nAG News Semi Anti-Causal done.")
    print(f"   train: {n_envs} envs, {sum(e.X.shape[0] for e in train_envs)} samples")
    print(f"   val  : {n_envs} envs, {val_envs[0].X.shape[0]} samples/env")
    print(f"   test : {test_env.X.shape[0]} OOD samples, p_correct={test_p_correct:.0%}")

    return train_envs, val_envs, test_env

# =============================================================================
# SST-2 confounding tokens (binary)
# =============================================================================
SST2_TOKENS: Dict[int, str] = {
    0: "north",  # negative
    1: "south",  # positive
}

_SST2_CONF_TOKENS: Dict[str, str] = {
    "ham_correlated":  SST2_TOKENS[0],  # "north" (label 0 = negative)
    "spam_correlated": SST2_TOKENS[1],  # "south" (label 1 = positive)
}

# =============================================================================
# Confounding helpers (shared between datasets)
# =============================================================================
#
# Shared DAG for all confounding experiments:
#   C  (binary latent confounder)
#   |-- C -> Z -> token injected into text   (spurious path)
#   `-- C -> Y  (direct label bias)
#   text -> Y                                 (invariant causal path)
# =============================================================================

def _apply_conf_label_bias(
    labels: np.ndarray,
    C: np.ndarray,
    gamma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Apply directional influence of C on a binary label array.

    When label != C, replace label with C with probability gamma:
      - C=1 increases P(Y=1)
      - C=0 increases P(Y=0)
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
    apply_gamma: bool = True,   # False for val/test in varying_gamma experiments
    conf_tokens: Optional[Dict[str, str]] = None,  # defaults to define_spurious_tokens()
    finetune_bert_layers: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a confounding environment:
      - Bias labels towards C (if apply_gamma)
      - Inject a token based on Z (not Y) into each text
      - Encode with BERT
    Returns (X, Y).
    """
    spurious_tokens = conf_tokens if conf_tokens is not None else define_spurious_tokens()
    labels_obs = _apply_conf_label_bias(labels, C, gamma, rng) if apply_gamma else labels.copy()

    # Z=1 -> token "spam_correlated", Z=0 -> token "ham_correlated"
    rng_inj = np.random.default_rng(int(rng.integers(0, 2**31)))
    texts_mod = [
        inject_spurious_token(text, int(z), 1.0, spurious_tokens, rng_inj)
        for text, z in zip(texts, Z)
    ]
    X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling,
                                     finetune_bert_layers=finetune_bert_layers)
    Y = labels_obs.reshape(-1, 1).astype(np.float32)
    return X, Y

# =============================================================================
# AG News — Confounding (varying proxy, multi-class)
# =============================================================================
# C in {0,...,K-1}, sampled uniformly for fraction p_c_flip of examples.
# Z = noisy proxy of C: P(Z=C) = 1 - a_e.
# token = _AG_NEWS_CONF_SHIFT_TOKENS[Z]  (red/blue/green/yellow)
# OOD eval: true labels (not Y_obs) -> clean measurement.
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
    Sample multi-class direct confounder (target-class replacement).

    C in {0,...,K-1} is the TARGET class (not a cyclic shift amount).
    For fraction p_c_flip of examples, C is drawn uniformly; otherwise C = Y* (inactive).

    Z is a noisy proxy of C: P(Z=C) = 1 - a_e.
    Since Z ~ C ~ Y_obs, ERM can use Z as a linear shortcut.
    """
    # Default: C = Y* (inactive confounder)
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
    a_e: float,                # Z noise (0=clean, 1=fully noisy)
    rng: np.random.Generator,
    bert_model: str,
    max_length: int,
    device: str,
    pooling: str,
    p_c_flip: float,
    apply_label_flip: bool = True,  # False for OOD test -> returns true labels
    gamma: float = 0.8,        # force d'alignement Y_obs→C (0=aucun, 1=total)
    n_classes: int = 4,
    finetune_bert_layers: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build an AG News confounding environment (multi-class, direct replacement).

    Steps:
      1. Sample C in {0,...,K-1} (target class, uniform) for fraction p_c_flip;
         C = Y* for the rest (inactive confounder).
      2. Build Z as noisy proxy of C: P(Z=C) = 1 - a_e.
         Z ~ C ~ Y_obs -> linear shortcut for ERM.
      3. Inject token for Z into the text.
      4. If apply_label_flip=True: align Y_obs towards C via _apply_conf_label_bias.
         If apply_label_flip=False (OOD test): return true labels Y*.
    """
    C, Z = _sample_ag_news_direct_confounder(labels, p_c_flip, a_e, rng, n_classes=n_classes)
    if apply_label_flip:
        rng_g = np.random.default_rng(int(rng.integers(0, 2**31)))
        Y_obs = _apply_conf_label_bias(labels, C, gamma, rng_g)
    else:
        Y_obs = labels.copy()

    # Step 3: inject spurious token
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
    AG News (4-class) — confounding with varying proxy.

    DAG: C in {0,...,K-1} (uniform) for fraction p_c_flip of examples;
         C = Y* (inactive) otherwise.
         Y_obs <- C with prob gamma when Y* != C (direct replacement).
         Z = noisy proxy of C (P(Z=C) = 1 - a_e) -> Z ~ C ~ Y_obs.
         token = CONF_TOKENS[Z]

    OOD test (a_test=1.0, clean labels): Z is anti-correlated with C
    -> ERM relying on Z fails; IRM (text only) generalises.

    Parameters
    ----------
    a_train  : List[float]  Proxy noise per train env (e.g. [0.01, 0.1]).
    a_test   : float        Proxy noise at OOD test (e.g. 0.99).
    p_c_flip : float        Fraction of examples with active confounder.
    """
    print("Loading AG News dataset (confounding – varying proxy)...")
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
    print(f"Loaded {n_total} articles | split: "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

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

    print(f"\n=== OOD test (a={a_test}) ===")
    test_texts  = [all_texts[int(j)] for j in test_idx]
    test_labels = all_labels_arr[test_idx]
    rng_t = np.random.default_rng(seed + 777)
    X_test, Y_test = _conf_ag_news_make_env(
        test_texts, test_labels, a_test, rng_t,
        bert_model, max_length, device, pooling, p_c_flip,
        apply_label_flip=False, gamma=gamma,  # evaluate on true labels
        finetune_bert_layers=finetune_bert_layers,
    )
    test_env = Env(
        torch.from_numpy(X_test), torch.from_numpy(Y_test),
        meta={"kind": "ag_news_conf_varying_proxy", "a": a_test, "p_c_flip": p_c_flip,
              "split": "test_ood", "n_samples": len(X_test), "n_classes": 4},
    )

    print(f"\nAG News Confounding (varying proxy) done.")
    print(f"   train={sum(e.X.shape[0] for e in train_envs)} "
          f"val={val_envs[0].X.shape[0]} test={test_env.X.shape[0]}")
    return train_envs, val_envs, test_env

# =============================================================================
# Multi-class subsampling helper
# =============================================================================

def _subsample_to_class_dist(
    texts: List[str],
    labels: np.ndarray,
    target_dist: List[float],
    rng: np.random.Generator,
    n_classes: int = 4,
) -> Tuple[List[str], np.ndarray]:
    """
    Subsample (texts, labels) so that P(Y=c) ~ target_dist[c].

    Keeps all examples of the most constrained class and subsamples the rest.

    Parameters
    ----------
    target_dist : List[float]  Target distribution, e.g. [0.1, 0.5, 0.2, 0.2] (auto-normalised).
    n_classes   : int
    """
    target_dist = np.array(target_dist, dtype=np.float64)
    target_dist /= target_dist.sum()  # normaliser

    # Indices par classe
    class_idx = {c: np.where(labels == c)[0] for c in range(n_classes)}
    class_counts = {c: len(class_idx[c]) for c in range(n_classes)}

    # Trouver la taille totale max atteignable
    # Pour chaque classe c : n_total_max = class_counts[c] / target_dist[c]
    # The tightest constraint determines n_total
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
    print(f"  Multi-class subsample: {len(kept)} samples "
          f"(actual={actual}, target={dict(enumerate(target_dist.tolist()))})")
    return texts_out, labels_out

# =============================================================================
# Binary subsampling helper (IMDB / Amazon)
# =============================================================================

def _subsample_to_ratio(
    texts: List[str],
    labels: np.ndarray,
    target_ratio: float,
    rng: np.random.Generator,
) -> Tuple[List[str], np.ndarray]:
    """
    Subsample (texts, labels) so that fraction p_pos of examples are positive.

    Keeps all minority-class examples and subsamples the majority class.
    """
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if target_ratio >= 0.5:
        # Positive majority -> keep all positives, subsample negatives
        n_neg_keep = int(n_pos * (1.0 - target_ratio) / target_ratio)
        n_neg_keep = min(n_neg_keep, n_neg)
        neg_kept = rng.choice(neg_idx, size=n_neg_keep, replace=False)
        pos_kept = pos_idx
    else:
        # Negative majority -> keep all negatives, subsample positives
        n_pos_keep = int(n_neg * target_ratio / (1.0 - target_ratio))
        n_pos_keep = min(n_pos_keep, n_pos)
        pos_kept = rng.choice(pos_idx, size=n_pos_keep, replace=False)
        neg_kept = neg_idx

    kept = rng.permutation(np.concatenate([pos_kept, neg_kept]))
    texts_out  = [texts[int(j)]  for j in kept]
    labels_out = labels[kept]
    actual_ratio = float(labels_out.mean())
    print(f"  Subsampled: {len(kept)} samples "
          f"(actual positive rate: {actual_ratio:.1%}, target: {target_ratio:.0%})")
    return texts_out, labels_out

# =============================================================================
# IMDB Genres — Semi anti-causal
# =============================================================================
# DAG: Y (genre) -> Z (spurious token) -> X = BERT(description + Z)
# Binary tokens ("pine" / "ash") inserted before each neutral word.
# P(token matches true label) = p_correct[env].
# OOD test: p_correct=0 -> token always wrong.
# =============================================================================

# Real words, single DistilBERT tokens, genre-neutral
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
    IMDB Genres (Thriller/Romance) — semi anti-causal environments (token injection).

    DAG: Y (genre) -> Z (spurious token) -> X = BERT(description + Z)

    One token is injected per description; P(token correct) = p_correct[env].
    OOD test: p_correct=0 -> token always wrong.
    IRM must learn to ignore Z and rely on text features.

    Parameters
    ----------
    train_p_correct : List[float]   e.g. [0.9, 0.7]
    test_p_correct  : float         typically 0.0
    seed            : int
    label_flip      : float         fraction of noisy labels
    bert_model, max_length, device, pooling : standard BERT config
    """
    print("Loading IMDB Genres dataset (semi anti-causal token injection)...")
    all_texts, all_labels = load_imdb_genres_dataset(seed=seed)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels)
    n_romance = int((all_labels_arr == 1).sum())
    n_thriller = n_total - n_romance
    print(f"Loaded {n_total} descriptions: thriller={n_thriller}, romance={n_romance}")

    # Global 80/10/10 split
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test_split = int(n_total * 0.1)
    n_val_split  = int(n_total * 0.1)
    test_indices  = indices[:n_test_split]
    val_indices   = indices[n_test_split:n_test_split + n_val_split]
    train_indices = indices[n_test_split + n_val_split:]

    print(f"  Split: train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}")

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

        # Label flip (random genre swap)
        if label_flip > 0:
            rng_flip = np.random.default_rng(seed + i * 13 + 1)
            flip_mask = rng_flip.uniform(size=len(labels)) < label_flip
            labels[flip_mask] = 1 - labels[flip_mask]

        # Inject spurious token (distributed before neutral words)
        rng_inject = np.random.default_rng(seed + i * 17 + 3)
        texts_mod = [
            inject_spurious_token_multiclass(t, int(l), p_correct, IMDB_GENRES_SAC_TOKENS, rng_inject)
            for t, l in zip(texts, labels)
        ]
        n_correct = sum(
            IMDB_GENRES_SAC_TOKENS[int(l)] in tm.lower().split()
            for tm, l in zip(texts_mod, labels)
        )
        print(f"  Correct token: {n_correct}/{len(labels)} ({n_correct/len(labels):.1%})")

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

    print(f"\nIMDB Genres Semi Anti-Causal done.")
    print(f"   train={sum(e.X.shape[0] for e in train_envs)} "
          f"val={sum(e.X.shape[0] for e in val_envs)} test={test_env.X.shape[0]}")
    return train_envs, val_envs, test_env

# =============================================================================
# IMDB Genres — Confounding (varying proxy)
# =============================================================================
# DAG: C ~ Ber(p_c) -> Y (flipped when C=1); C -> Z = C XOR Ber(a_e) -> token
# Common cause C creates spurious Z~Y correlation without a direct Y->Z path.
#
# ERM mechanism:
#   BERT sees (text, token). When Z=1 (C~1), labels are flipped relative to
#   text -> BERT learns Z as a context switch (prediction inverter).
#   This is a nonlinear (text x Z) interaction captured by attention.
#
# OOD test (a_test=1.0): Z = NOT C, clean labels (no flip)
#   -> ERM uses Z as inverter but labels are no longer flipped -> fails
#   -> IRM (ignoring Z) uses text and succeeds
#
# Note: P(Y_obs=1|Z=1) = 0.5 marginally (marginal independence).
#   Not a contradiction: spuriousness is in the joint (text, Z) -> Y_obs
#   relationship learned by attention.

# Proxy confounding tokens (distinct from SAC tokens)
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
    IMDB Genres — confounding with varying proxy (C -> Y and C -> Z).

    DAG: C ~ Ber(p_c) -> Y (label biased towards C);
         C -> Z = C XOR Ber(a_e) (noisy proxy token)

    C affects both label and token. IRM must ignore Z and learn from text.

    Parameters
    ----------
    a_train  : List[float]  C->Z noise per env. a_e ~ 0: strong; a_e ~ 1: noisy.
    a_test   : float        C->Z noise at OOD (typically ~1.0).
    p_c_flip : float        P(C=1).
    gamma    : float        C->Y strength (if Y!=C, flip with prob gamma).
    """
    print("Loading IMDB Genres dataset (confounding varying proxy)...")
    all_texts, all_labels = load_imdb_genres_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Loaded {n_total} descriptions")

    # Split global : 80/10/10
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test_split = int(n_total * 0.1)
    n_val_split  = int(n_total * 0.1)
    test_indices  = indices[:n_test_split]
    val_indices   = indices[n_test_split:n_test_split + n_val_split]
    train_indices = indices[n_test_split + n_val_split:]

    print(f"  Split: train={len(train_indices)} val={len(val_indices)} test={len(test_indices)}")

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

        # Sample C after subsampling (avoids index misalignment)
        rng_c_e = np.random.default_rng(seed + i * 7 + 100)
        C_env = rng_c_e.binomial(1, p_c_flip, size=len(labels))

        # C -> Y: bias label towards C
        rng_y_e = np.random.default_rng(seed + i * 17 + 3)
        labels_confounded = _apply_conf_label_bias(labels, C_env, gamma, rng_y_e)

        if label_flip > 0:
            rng_flip = np.random.default_rng(seed + i * 13 + 1)
            flip_mask = rng_flip.uniform(size=len(labels_confounded)) < label_flip
            labels_confounded[flip_mask] = 1 - labels_confounded[flip_mask]

        # C -> Z: noisy proxy of C, a_e small -> Z~C (strong); a_e large -> noisy
        rng_z = np.random.default_rng(seed + i * 23 + 5)
        noise = (rng_z.uniform(size=len(C_env)) < a_e).astype(int)
        Z_env = C_env ^ noise

        # Inject proxy token (distributed before neutral words)
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

        # C for val generated independently after subsampling
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

    # OOD test: clean labels (no flip), Z = C XOR Ber(a_test)
    # With a_test=1.0: Z = NOT C -> token inverts the habitual signal
    # ERM (context-switch via Z) is tricked; IRM (text only) holds
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

    print(f"\nIMDB Genres Confounding (varying proxy) done.")
    print(f"   train={sum(e.X.shape[0] for e in train_envs)} "
          f"val={sum(e.X.shape[0] for e in val_envs)} test={test_env.X.shape[0]}")
    return train_envs, val_envs, test_env

def load_imdb_genres_dataset(seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Load jquigl/imdb-genres from Hugging Face.

    Merges all splits, keeps only Thriller and Romance genres.

    Returns
    -------
    texts  : List[str]  -- movie plot descriptions
    labels : List[int]  -- 0 = Thriller, 1 = Romance
    """
    from datasets import concatenate_datasets

    dataset = load_dataset("jquigl/imdb-genres")

    # Merge all available splits
    splits = [dataset[s] for s in dataset.keys()]
    all_data = concatenate_datasets(splits)

    # Keep only Thriller and Romance
    all_data = all_data.filter(
        lambda ex: ex["genre"] in ("Thriller", "Romance")
    )

    # Reproducible shuffle
    all_data = all_data.shuffle(seed=seed)

    texts  = [str(ex["description"]) for ex in all_data]
    labels = [1 if ex["genre"] == "Romance" else 0 for ex in all_data]

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    print(f"IMDB Genres (Thriller/Romance): {len(texts)} examples "
          f"romance={n_pos} thriller={n_neg}")
    return texts, labels

# =============================================================================
# IMDB Genres — Size-based selection
# =============================================================================
# Dataset: jquigl/imdb-genres (movie plot descriptions + genres)
# Task: predict genre (thriller=0, romance=1)
# Spurious signal Z: description length
#
# 4-pool design: short_pos, short_neg, long_pos, long_neg
# Controlled correlation: P(Y=romance | Z=long) = p_select
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
    IMDB Genres (Thriller/Romance) — size-based selection environments.

    DAG: Y (genre) -> Z (description length) -> S (train selection)

    4-pool design:
      short_pos: Z=short, Y=Romance (1)
      short_neg: Z=short, Y=Thriller (0)
      long_pos : Z=long,  Y=Romance (1)
      long_neg : Z=long,  Y=Thriller (0)

    Train env i (p_select):
      P(Y=Romance | Z=long)  = p_select
      P(Y=Romance | Z=short) = 1 - p_select
      P(Y=Romance) overall  ~ 50%

    OOD test: fully inverted size-genre correlation.

    Parameters
    ----------
    train_p_select   : List[float]
    threshold_method : str           "quartile" (default) or "median".
    val_frac         : float
    label_flip       : float
    max_length       : int
    class_ratio_train: Optional[List[float]]
    class_ratio_test : Optional[float]
    """
    print("Loading IMDB Genres dataset (size-based selection)...")
    all_texts, all_labels = load_imdb_genres_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Loaded {n_total} descriptions (Thriller/Romance)")

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
    print(f"Thresholds ({threshold_method}): short < {t1:.0f} chars, long > {t2:.0f} chars")

    # Pre-flip labels globally so that selection is based on noisy Y.
    # Z must align with ~Y so the spurious signal stays strong as noise grows.
    all_noisy_labels = list(all_labels)
    if label_flip > 0.0:
        rng_global_flip = np.random.default_rng(seed + 6999)
        gfm = rng_global_flip.uniform(size=len(all_noisy_labels)) < label_flip
        for k in np.where(gfm)[0]:
            all_noisy_labels[k] ^= 1
        print(f"  Label flip (global, before selection): "
              f"{int(gfm.sum())}/{len(all_noisy_labels)} ({gfm.mean():.1%})")

    # Training pools: partitioned with NOISY labels (Z aligned to ~Y)
    # OOD pools    : partitioned with TRUE  labels (evaluation on Y)
    short_pos: List[str] = []  # Z=short, Y_noisy=romance (1)  [training]
    short_neg: List[str] = []  # Z=short, Y_noisy=thriller (0) [training]
    long_pos:  List[str] = []  # Z=long,  Y_noisy=romance (1)  [training]
    long_neg:  List[str] = []  # Z=long,  Y_noisy=thriller (0) [training]

    short_pos_ood: List[str] = []  # Z=short, Y_true=romance   [OOD]
    long_neg_ood:  List[str] = []  # Z=long,  Y_true=thriller  [OOD]

    for text, true_label, noisy_label in zip(all_texts, all_labels, all_noisy_labels):
        text_len = len(text)
        if text_len < t1:
            if noisy_label == 1: short_pos.append(text)
            else:                short_neg.append(text)
            if true_label  == 1: short_pos_ood.append(text)
        elif text_len > t2:
            if noisy_label == 1: long_pos.append(text)
            else:                long_neg.append(text)
            if true_label  == 0: long_neg_ood.append(text)
        # mid-range texts are discarded

    print(f"Training pools (noisy labels):")
    print(f"  short_pos (Z=short, Y_noisy=Romance) : {len(short_pos)}")
    print(f"  short_neg (Z=short, Y_noisy=Thriller): {len(short_neg)}")
    print(f"  long_pos  (Z=long,  Y_noisy=Romance) : {len(long_pos)}")
    print(f"  long_neg  (Z=long,  Y_noisy=Thriller): {len(long_neg)}")
    print(f"OOD pools (true labels): short_pos_ood={len(short_pos_ood)}  long_neg_ood={len(long_neg_ood)}")

    # Shuffle training pools
    rng_shuffle = np.random.default_rng(seed + 5000)
    short_pos = [short_pos[j] for j in rng_shuffle.permutation(len(short_pos))]
    short_neg = [short_neg[j] for j in rng_shuffle.permutation(len(short_neg))]
    long_pos  = [long_pos[j]  for j in rng_shuffle.permutation(len(long_pos))]
    long_neg  = [long_neg[j]  for j in rng_shuffle.permutation(len(long_neg))]
    # Shuffle OOD pools
    rng_shuffle_ood = np.random.default_rng(seed + 5050)
    short_pos_ood = [short_pos_ood[j] for j in rng_shuffle_ood.permutation(len(short_pos_ood))]
    long_neg_ood  = [long_neg_ood[j]  for j in rng_shuffle_ood.permutation(len(long_neg_ood))]

    train_envs: List[Env] = []
    val_envs:   List[Env] = []

    n_envs = len(train_p_select)
    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        rng_env = np.random.default_rng(seed + 5000 + i)
        rng_mix = np.random.default_rng(seed + 6100 + i)

        # Non-overlapping slices per env
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
        print(f"  P(Romance|long)={p_pos_given_long:.1%}  (target: {p_select:.0%})")
        print(f"  P(Romance|short)={p_pos_given_short:.1%} (target: {1-p_select:.0%})")
        print(f"  P(Romance) overall: {np.mean(selected_labels):.1%}")

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
        tr_labels = sel_labels[tr_idx].copy()  # already noisy (flip applied before selection)
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
        va_labels = sel_labels[va_idx].copy()  # already noisy (flip applied before selection)
        X_va = tokenize_and_embed_with_bert(
            va_texts, bert_model, max_length, device, pooling,
            finetune_bert_layers=finetune_bert_layers)
        val_envs.append(Env(
            torch.from_numpy(X_va),
            torch.from_numpy(va_labels.reshape(-1, 1).astype(np.float32)),
            meta={"p_select": p_select, "kind": "imdb_genres_size_selection_val",
                  "env_id": i, "label_flip": label_flip, "n_samples": len(X_va)}))

    print(f"\n=== OOD test (size-genre correlation INVERTED) ===")

    rng_ood = np.random.default_rng(seed + 25000)
    n_ood_long  = min(len(long_neg_ood),  2000)
    n_ood_short = min(len(short_pos_ood), 2000)

    ood_long_idx  = rng_ood.choice(len(long_neg_ood),  size=n_ood_long,  replace=False)
    ood_short_idx = rng_ood.choice(len(short_pos_ood), size=n_ood_short, replace=False)

    ood_texts_final = ([long_neg_ood[j]  for j in ood_long_idx] +
                       [short_pos_ood[j] for j in ood_short_idx])
    ood_labels_arr  = np.array([0]*n_ood_long + [1]*n_ood_short)

    perm = rng_ood.permutation(len(ood_texts_final))
    ood_texts_final = [ood_texts_final[j] for j in perm]
    ood_labels_arr  = ood_labels_arr[perm]

    print(f"  {n_ood_long} long_neg_true (Thriller) + {n_ood_short} short_pos_true (Romance)")
    print(f"  P(Romance|long)=0%  P(Romance|short)=100% (INVERTED)")
    print(f"  P(Romance) overall: {ood_labels_arr.mean():.1%}")

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

    print(f"\nIMDB Genres Size Selection done.")
    print(f"   train={sum(e.X.shape[0] for e in train_envs)} "
          f"val={sum(e.X.shape[0] for e in val_envs)} test={test_env.X.shape[0]}")
    return train_envs, val_envs, test_env

# =============================================================================
# Amazon Reviews – Books
# =============================================================================
# Causal dataset (X -> Y): text content determines review helpfulness.
# Source: McAuley-Lab/Amazon-Reviews-2023, Books category.
# Binarisation: helpful_vote=0 -> not helpful (0), helpful_vote>=5 -> helpful (1).
# =============================================================================

AMAZON_CLASS_NAMES: Dict[int, str] = {0: "not_helpful", 1: "helpful"}
AMAZON_TOKENS: Dict[int, str] = {0: "moon", 1: "sun"}
_AMAZON_CONF_TOKENS: Dict[str, str] = {
    "ham_correlated":  "moon",   # label 0 (not helpful)
    "spam_correlated": "sun",    # label 1 (helpful)
}

def load_amazon_books(
    seed: int = 42,
    n_target: int = 100_000,
    helpful_threshold: int = 5,
) -> Tuple[List[str], List[int]]:
    """
    Load Amazon Reviews Books from HuggingFace (streaming jsonl).

    Binarises on helpful_vote:
      - helpful_vote = 0          -> not helpful (0)
      - helpful_vote >= threshold -> helpful (1)
      - 1 .. threshold-1          -> discarded (ambiguous)

    Streams until n_target samples (50/50 balanced by label) are collected.

    Returns
    -------
    texts  : List[str]
    labels : List[int]  -- 0 = not helpful, 1 = helpful
    """
    print(f"Loading Amazon Books (streaming, target {n_target} reviews, "
          f"helpful_vote >= {helpful_threshold})...")
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

    print(f"  Collected {n_per} not-helpful + {n_per} helpful = {len(texts)} total")
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
    Amazon Books — semi anti-causal environments (token injection).

    DAG: Text -> Y; Y -> Z (spurious token with prob p_correct); Text + Z -> X.
    OOD: p_correct ~ 0 (tokens always wrong).
    """
    all_texts, all_labels = load_amazon_books(seed=seed, n_target=n_target)
    n_total = len(all_texts)
    all_labels_arr = np.array(all_labels, dtype=np.int64)
    print(f"Loaded {n_total} reviews "
          f"(not-helpful={int((all_labels_arr==0).sum())}, helpful={int((all_labels_arr==1).sum())})")

    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test = int(n_total * 0.10)
    n_val  = int(n_total * 0.10)
    test_idx  = indices[:n_test]
    val_idx   = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]
    print(f"  Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

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

    print(f"\nAmazon Books Semi Anti-Causal done.")
    print(f"   train={sum(e.X.shape[0] for e in train_envs)} "
          f"val={val_envs[0].X.shape[0]} test={test_env.X.shape[0]}")
    return train_envs, val_envs, test_env

# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Size-based selection bias
# ─────────────────────────────────────────────────────────────────────────────
# DAG: Y -> Z (review length) -> S (train selection)
#
# Natural correlation: helpful reviews tend to be longer.
# Spurious shortcut: length correlates with helpfulness but does not cause it.
#
# Typical : not-helpful (0) SHORT (< Q1), helpful (1) LONG (> Q3)
# OOD     : not-helpful (0) very LONG,    helpful (1) very SHORT
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
    Amazon Books — confounding with varying proxy.

    DAG: C ~ Ber(p_c_flip) -> Z = C XOR Ber(a_e) -> token;
         C -> Y (label biased towards C); Text -> Y.
    OOD: a_test = 1.0 => Z independent of C => token uninformative.
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
    print(f"Loaded {n_total} reviews | split: "
          f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

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

    print(f"\nAmazon Books Confounding (varying proxy) done.")
    print(f"   train={sum(e.X.shape[0] for e in train_envs)} "
          f"val={val_envs[0].X.shape[0]} test={test_env.X.shape[0]}")
    return train_envs, val_envs, test_env

# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Rating-stratified environments
# ─────────────────────────────────────────────────────────────────────────────
# Env 0: 5-star reviews  -> length-helpfulness correlation is strong
# Env 1: 3-4-star reviews -> weaker correlation
# Test : 1-star reviews  -> a short negative review can be very helpful
#        -> length shortcut breaks
# ─────────────────────────────────────────────────────────────────────────────

def _load_amazon_books_with_rating(
    seed: int = 42,
    n_target: int = 100_000,
    helpful_threshold: int = 5,
) -> Tuple[List[str], List[int], List[float]]:
    """
    Load Amazon Books with star rating, balanced label x rating.

    Collects into 4 buckets (label x rating) and balances to equal counts.

    Returns
    -------
    texts   : List[str]
    labels  : List[int]    -- 0=not helpful, 1=helpful
    ratings : List[float]  -- 1.0 or 5.0
    """
    print(f"Loading Amazon Books with rating (streaming, target {n_target}, "
          f"helpful_vote >= {helpful_threshold})...")
    ds = load_dataset(
        "json",
        data_files="hf://datasets/McAuley-Lab/Amazon-Reviews-2023/"
                   "raw/review_categories/Books.jsonl",
        split="train",
        streaming=True,
    )

    # 4 buckets : (label, rating) = (0,1*), (0,5*), (1,1*), (1,5*)
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

    # Balance to smallest bucket
    n_per = min(len(b) for b in buckets.values())
    print(f"  Buckets: " + ", ".join(
        f"({'helpful' if l else 'not-helpful'},{r:.0f}*)={len(buckets[(l,r)])}"
        for l in (0, 1) for r in (1.0, 5.0)))
    print(f"  Balanced to {n_per} per bucket -> {4 * n_per} total")

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
    print(f"  Ratings: {n_pos} positive (5*), {n_neg} negative (1*)")
    return texts, labels, ratings

def _is_typical_sentiment(label: int, rating: float) -> bool:
    """Return True if (label, rating) is a typical pair: helpful+5* or not-helpful+1*."""
    if label == 1:
        return rating == 5.0
    else:
        return rating == 1.0

def _is_cross_sentiment(label: int, rating: float) -> bool:
    """Return True if (label, rating) is a cross pair: helpful+1* or not-helpful+5*."""
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
    Amazon Books — sentiment-based selection environments.

    DAG: Y -> Z (positive/negative rating) -> S (train selection)

    Typical : helpful (1) + positive (4-5 stars), OR not-helpful (0) + negative (1-2 stars).
    OOD     : helpful (1) + negative (1-2 stars), OR not-helpful (0) + positive (4-5 stars).

    The "positive rating <-> helpful" shortcut is strong in train but breaks at OOD.

    Parameters
    ----------
    train_p_select : List[float]  P(keep typical example) per env.
    n_target       : int          Total reviews to load (50/50 balanced by label).
    """
    print("Loading Amazon Books (sentiment selection)...")
    all_texts, all_labels, all_ratings = _load_amazon_books_with_rating(
        seed=seed, n_target=n_target,
    )
    n_total = len(all_texts)
    print(f"Loaded {n_total} reviews")

    # Pre-flip labels globally so that selection is based on noisy Y.
    # Z (rating) must align with ~Y so the spurious signal stays strong as noise grows.
    all_noisy_labels = list(all_labels)
    if label_flip > 0.0:
        rng_global_flip = np.random.default_rng(seed + 7999)
        gfm = rng_global_flip.uniform(size=len(all_noisy_labels)) < label_flip
        for k in np.where(gfm)[0]:
            all_noisy_labels[k] ^= 1
        print(f"  Label flip (global, before selection): "
              f"{int(gfm.sum())}/{n_total} ({gfm.mean():.1%})")

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

        env_texts        = [all_texts[int(j)]        for j in env_indices]
        env_labels       = [all_labels[int(j)]       for j in env_indices]
        env_noisy_labels = [all_noisy_labels[int(j)] for j in env_indices]
        env_ratings      = [all_ratings[int(j)]      for j in env_indices]

        selected_texts:  List[str] = []
        selected_labels: List[int] = []
        n_typical = 0
        n_cross   = 0

        for text, true_label, noisy_label, rating in zip(
                env_texts, env_labels, env_noisy_labels, env_ratings):
            if _is_typical_sentiment(noisy_label, rating):
                n_typical += 1
                if rng.uniform() < p_select:
                    selected_texts.append(text)
                    selected_labels.append(noisy_label)
            elif _is_cross_sentiment(noisy_label, rating):
                n_cross += 1
                u = rng.uniform()
                if u < (1.0 - p_select):
                    # Cross included in train with prob (1-p_select)
                    # -> effective spurious correlation = p_select in this env
                    selected_texts.append(text)
                    selected_labels.append(noisy_label)
                elif i == 0 and _is_cross_sentiment(true_label, rating):
                    # OOD: cross in the TRUE-label sense, not selected for training
                    ood_texts.append(text)
                    ood_labels.append(true_label)

        print(f"  Typical: {n_typical}, Cross: {n_cross}")
        print(f"  Effective spurious correlation: {p_select:.0%}")
        print(f"  Selected {len(selected_texts)} reviews")

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
        tr_labels = sel_labels_arr[tr_idx].copy()  # already noisy (flip applied before selection)
        X_tr = tokenize_and_embed_with_bert(
            tr_texts, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
        train_envs.append(Env(
            torch.from_numpy(X_tr),
            torch.from_numpy(tr_labels.reshape(-1, 1).astype(np.float32)),
            meta={"p_select": p_select, "kind": "amazon_sentiment_selection_train",
                  "env_id": i, "label_flip": label_flip, "n_samples": len(X_tr)}))

        # ─── Val ───
        va_texts  = [sel_texts_arr[j] for j in va_idx]
        va_labels = sel_labels_arr[va_idx].copy()  # already noisy (flip applied before selection)
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
    print(f"  not-helpful+positive: {n0_ood}, helpful+negative: {n1_ood}")
    X_test = tokenize_and_embed_with_bert(
        ood_texts_final, bert_model, max_length, device, pooling, finetune_bert_layers=finetune_bert_layers)
    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(ood_labels_arr.reshape(-1, 1).astype(np.float32)),
        meta={"kind": "amazon_sentiment_selection_test_ood",
              "n_samples": len(X_test)})

    print(f"\nAmazon Books Sentiment Selection done.")
    print(f"   train={sum(e.X.shape[0] for e in train_envs)} "
          f"val={sum(e.X.shape[0] for e in val_envs)} test={test_env.X.shape[0]}")
    return train_envs, val_envs, test_env

