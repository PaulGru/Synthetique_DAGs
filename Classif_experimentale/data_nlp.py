# Générateurs d'environnements NLP (SMS Spam) pour expériences IRM
# Adaptation du DAG semi anti-causal : Text → Y → Z → X_y (spurious token embedding)

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertModel
from data_synth import Env  # Réutiliser la même structure Env


# =============================================================================
# Configuration des tokens trompeurs
# =============================================================================

def define_spurious_tokens() -> Dict[str, str]:
    """
    Définit les tokens textuels trompeurs.
    
    Returns
    -------
    dict : {"spam_correlated": token, "ham_correlated": token}
    
    Notes
    -----
    red : normalement corrélé avec spam
    green : normalement corrélé avec ham
    (La corrélation peut être inversée selon l'environnement)
    
    Ces tokens sont tokenisés par BERT comme UN SEUL TOKEN chacun:
    - red → ['red'] (1 token, ID: 2417)
    - green → ['green'] (1 token, ID: 2665)
    
    Avantage: Signal spurious maximum car 1 seul token au lieu de 3 ([RED] → ['[', 'red', ']'])
    """
    return {
        "spam_correlated": "red",      # Rouge/spam
        "ham_correlated": "green",     # Vert/ham
    }


# =============================================================================
# Characteristic words for selection bias
# =============================================================================

# Mots fortement associés au spam (promotions, gains, urgence)
SPAM_WORDS = [
    "winner", "win", "congratulations", "free", "premium",
    "private", "claim", "offer", "call", "award", "prize", "urgent"
]

# Mots fortement associés aux conversations normales (ham)
# Analysés depuis le dataset: mots >50x plus fréquents dans ham vs spam
HAM_WORDS = [
    "home", "later", "come", "happy", "way", "ask", "said",
    "doing", "really", "yeah", "but", "she", "much", "already", "too"
]


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

def inject_spurious_token(
    text: str,
    label: int,
    p_correct: float,
    spurious_tokens: Dict[str, str],
    rng: np.random.Generator,
    position: str = "prefix"
) -> str:
    """
    Injecte un token textuel trompeur dans le texte avec corrélation contrôlée.
    
    Nouveau mécanisme (direct) :
    - p_correct = probabilité d'avoir le "bon" token selon le label
    - Si label=1 (spam) : p_correct% → red, (1-p_correct)% → green
    - Si label=0 (ham) : p_correct% → green, (1-p_correct)% → red
    
    Exemples :
    - Env 1 : p_correct=0.9 → 90% corrélation
    - Env 2 : p_correct=0.8 → 80% corrélation
    - Test OOD : p_correct=0.0 → 100% INVERSÉ
    
    Parameters
    ----------
    text : str
        Texte SMS original.
    label : int
        Label vrai (0=ham, 1=spam).
    p_correct : float
        Probabilité d'associer le "bon" token au label.
        - 1.0 = corrélation maximale
        - 0.5 = aléatoire
        - 0.0 = inversion totale
    spurious_tokens : dict
        Dict with {"spam_correlated": "red", "ham_correlated": "green"}.
    rng : np.random.Generator
        Générateur aléatoire.
    position : str
        Position d'insertion ("prefix" ou "suffix").
    
    Returns
    -------
    text_with_token : str
        Texte avec token textuel injecté (avec espaces pour tokenisation indépendante).
    """
    # Extraire les tokens
    red_token = spurious_tokens["spam_correlated"]  # red
    green_token = spurious_tokens["ham_correlated"]  # green
    
    # Déterminer quel token ajouter selon le label et p_correct
    if rng.uniform() < p_correct:
        # Corrélation "normale"
        token = red_token if label == 1 else green_token
    else:
        # Corrélation inversée
        token = green_token if label == 1 else red_token
    
    # Injecter le token avec espaces (pour tokenisation indépendante)
    if position == "prefix":
        return f"{token} {text}"
    else:  # suffix
        return f"{text} {token}"


# =============================================================================
# Tokenisation et extraction d'embeddings BERT
# =============================================================================

def tokenize_and_embed_with_bert(
    texts: List[str],
    model_name: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean"
) -> np.ndarray:
    """
    Tokenise les textes et extrait les embeddings BERT.
    
    Parameters
    ----------
    texts : List[str]
        Liste des textes à tokeniser.
    model_name : str
        Nom du modèle BERT à utiliser.
    max_length : int
        Longueur maximale de séquence.
    device : str
        Device PyTorch ("cpu" ou "cuda").
    pooling : str
        Type de pooling ("mean", "cls", ou "max").
    
    Returns
    -------
    embeddings : np.ndarray (N, 768)
        Embeddings BERT pour chaque texte.
    """
    # Charger tokenizer et modèle
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    # Freeze le modèle (pas de fine-tuning)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    model = model.to(device)
    
    embeddings = []
    batch_size = 32
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokeniser
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            
            # Déplacer sur device
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Extraire embeddings
            # outputs.last_hidden_state: (batch, seq_len, 768)
            last_hidden = outputs.last_hidden_state
            
            if pooling == "mean":
                # Mean pooling (en tenant compte du masque d'attention)
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                batch_embeddings = sum_embeddings / sum_mask
            elif pooling == "cls":
                # CLS token (premier token)
                batch_embeddings = last_hidden[:, 0, :]
            elif pooling == "max":
                # Max pooling
                batch_embeddings = torch.max(last_hidden, dim=1)[0]
            else:
                raise ValueError(f"Unknown pooling: {pooling}")
            
            embeddings.append(batch_embeddings.cpu().numpy())
    
    return np.concatenate(embeddings, axis=0).astype(np.float32)


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
    bert_model: str = "bert-base-uncased",
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
        pooling=pooling
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
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
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
        
        texts_mod = [inject_spurious_token(t, int(l), p_correct, spurious_tokens, rng) for t, l in zip(texts, labels)]
        X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling)
        Y = labels.reshape(-1, 1).astype(np.float32)
        
        train_envs.append(Env(torch.from_numpy(X), torch.from_numpy(Y),
                             meta={"p_correct": p_correct, "label_flip": label_flip, "n_samples": len(X)}))
        
        # VAL ENV
        print(f"=== Val Env {i} ===")
        val_texts = [all_texts[j] for j in val_indices]
        val_labels = np.array([all_labels[j] for j in val_indices])
        val_texts_mod = [inject_spurious_token(t, int(l), p_correct, spurious_tokens, np.random.default_rng(seed+5000+i))
                        for t, l in zip(val_texts, val_labels)]
        X_val = tokenize_and_embed_with_bert(val_texts_mod, bert_model, max_length, device, pooling)
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
    X_test = tokenize_and_embed_with_bert(test_texts_mod, bert_model, max_length, device, pooling)
    Y_test = test_labels.reshape(-1, 1).astype(np.float32)
    
    test_env = Env(torch.from_numpy(X_test), torch.from_numpy(Y_test),
                  meta={"p_correct": test_p_correct, "n_samples": len(X_test)})
    
    print(f"\n✅ Done! Train: {sum(e.X.shape[0] for e in train_envs)} | Val: {val_envs[0].X.shape[0]} | Test: {test_env.X.shape[0]}")
    return train_envs, val_envs, test_env


# =============================================================================
# Selection bias helper
# =============================================================================

def contains_characteristic_word(
    text: str,
    label: int,
    spam_words: List[str] = SPAM_WORDS,
    ham_words: List[str] = HAM_WORDS
) -> bool:
    """
    Vérifie si le texte contient des mots caractéristiques de son label.
    
    Parameters
    ----------
    text : str
        Texte SMS à vérifier.
    label : int
        Label (0=ham, 1=spam).
    spam_words : List[str]
        Liste de mots typiques du spam.
    ham_words : List[str]
        Liste de mots typiques du ham.
    
    Returns
    -------
    bool
        True si le texte contient au moins un mot caractéristique de son label.
    
    Examples
    --------
    >>> contains_characteristic_word("Win FREE prize now!", 1, SPAM_WORDS, HAM_WORDS)
    True  # contient "win" et "free"
    >>> contains_characteristic_word("See you later at home", 0, SPAM_WORDS, HAM_WORDS)
    True  # contient "later" et "home"
    >>> contains_characteristic_word("OK thanks", 0, SPAM_WORDS, HAM_WORDS)
    False  # pas de mots caractéristiques (atypique)
    """
    text_lower = text.lower()
    
    if label == 1:  # spam
        return any(word in text_lower for word in spam_words)
    else:  # ham (0)
        return any(word in text_lower for word in ham_words)


# =============================================================================
# Construction d'environnements avec selection bias
# =============================================================================

def build_envs_nlp_selection_bias(
    train_p_select: List[float],
    seed: int,
    val_frac: float = 0.1,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
    """
    Construit des environnements NLP avec biais de sélection.
    
    **DAG**: Y → Z → S (sélection dépend de Y ET Z)
    - Y = label (spam/ham)
    - Z = présence de mots caractéristiques
    - S = sélection (probabilité dépend de Y AND Z)
    
    **Mécanisme de sélection**:
    - Si le texte contient des mots caractéristiques de son label:
      → sélectionner avec probabilité `p_select`
    - Sinon (texte "atypique"):
      → rejeter (ajouter au pool OOD)
    
    **Résultat**:
    - Train envs: exemples "typiques" (spam avec mots spam, ham avec mots ham)
    - Test OOD: exemples "atypiques" (rejetés, sans mots caractéristiques)
    
    Parameters
    ----------
    train_p_select : List[float]
        Probabilités de sélection pour chaque env d'entraînement.
        Ex: [0.9, 0.8] → env1 garde 90% des typiques, env2 garde 80%.
    seed : int
        Graine aléatoire.
    val_frac : float
        Fraction des exemples sélectionnés à utiliser pour validation.
    bert_model : str
        Modèle BERT.
    max_length : int
        Longueur max de séquence.
    device : str
        Device PyTorch.
    pooling : str
        Type de pooling.
    
    Returns
    -------
    train_envs : List[Env]
        Environnements d'entraînement (exemples typiques).
    val_envs : List[Env]
        Environnements de validation.
    test_env : Env
        Environnement de test OOD (exemples atypiques rejetés).
    
    Examples
    --------
    >>> trains, vals, test = build_envs_nlp_selection_bias([0.9, 0.8], seed=42)
    >>> # trains[0]: 90% des spam/ham typiques
    >>> # trains[1]: 80% des spam/ham typiques
    >>> # test: tous les spam/ham atypiques (sans mots caractéristiques)
    """
    print("Chargement du dataset SMS Spam...")
    all_texts, all_labels = load_sms_spam_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset chargé : {n_total} SMS")
    
    rng = np.random.default_rng(seed)
    
    # Pool pour les exemples rejetés (atypiques) → test OOD
    rejected_pool_texts = []
    rejected_pool_labels = []
    
    n_envs = len(train_p_select)
    train_envs, val_envs = [], []
    
    # =================== SÉLECTION PAR ENVIRONNEMENT ===================
    for i, p_select in enumerate(train_p_select):
        print(f"\n=== Env {i} (p_select={p_select:.0%}) ===")
        
        selected_texts = []
        selected_labels = []
        
        # Parcourir tous les SMS
        for text, label in zip(all_texts, all_labels):
            # Vérifier si le texte contient des mots caractéristiques
            has_char_word = contains_characteristic_word(text, label)
            
            if has_char_word:
                # Texte "typique" → sélectionner avec p_select
                if rng.uniform() < p_select:
                    selected_texts.append(text)
                    selected_labels.append(label)
                else:
                    # Pas sélectionné this time (peut être sélectionné dans autre env)
                    pass
            else:
                # Texte "atypique" → rejeter systématiquement (pour test OOD)
                # On l'ajoute au pool uniquement la première fois (env 0)
                if i == 0:
                    rejected_pool_texts.append(text)
                    rejected_pool_labels.append(label)
        
        print(f"  Sélectionné: {len(selected_texts)} SMS")
        
        # Split train/val
        n_selected = len(selected_texts)
        n_val = int(n_selected * val_frac)
        n_train = n_selected - n_val
        
        indices_shuffled = rng.permutation(n_selected)
        train_idx = indices_shuffled[:n_train]
        val_idx = indices_shuffled[n_train:]
        
        # Train env
        train_texts = [selected_texts[j] for j in train_idx]
        train_labels = np.array([selected_labels[j] for j in train_idx])
        
        X_train = tokenize_and_embed_with_bert(train_texts, bert_model, max_length, device, pooling)
        Y_train = train_labels.reshape(-1, 1).astype(np.float32)
        
        train_envs.append(Env(
            torch.from_numpy(X_train),
            torch.from_numpy(Y_train),
            meta={"p_select": p_select, "kind": "nlp_selection_train", "env_id": i,
                  "n_samples": len(X_train)}
        ))
        
        # Val env
        val_texts = [selected_texts[j] for j in val_idx]
        val_labels = np.array([selected_labels[j] for j in val_idx])
        
        X_val = tokenize_and_embed_with_bert(val_texts, bert_model, max_length, device, pooling)
        Y_val = val_labels.reshape(-1, 1).astype(np.float32)
        
        val_envs.append(Env(
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val),
            meta={"p_select": p_select, "kind": "nlp_selection_val", "env_id": i,
                  "n_samples": len(X_val)}
        ))
    
    # =================== TEST OOD (exemples atypiques) ===================
    print(f"\n=== Test OOD (exemples atypiques rejetés) ===")
    print(f"  {len(rejected_pool_texts)} SMS atypiques (sans mots caractéristiques)")
    
    rejected_labels = np.array(rejected_pool_labels)
    
    X_test = tokenize_and_embed_with_bert(rejected_pool_texts, bert_model, max_length, device, pooling)
    Y_test = rejected_labels.reshape(-1, 1).astype(np.float32)
    
    test_env = Env(
        torch.from_numpy(X_test),
        torch.from_numpy(Y_test),
        meta={"kind": "nlp_selection_test_ood", "n_samples": len(X_test),
              "description": "atypical_examples"}
    )
    
    print(f"\n✅ Environnements créés avec selection bias !")
    print(f"   - {len(train_envs)} envs d'entraînement (exemples typiques)")
    print(f"   - {len(val_envs)} envs de validation")
    print(f"   - 1 env de test OOD ({test_env.X.shape[0]} exemples atypiques)")
    
    return train_envs, val_envs, test_env


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
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
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
        
        X_train = tokenize_and_embed_with_bert(train_texts, bert_model, max_length, device, pooling)
        Y_train = train_labels.reshape(-1, 1).astype(np.float32)
        
        train_envs.append(Env(
            torch.from_numpy(X_train),
            torch.from_numpy(Y_train),
            meta={"p_select": p_select, "kind": "nlp_size_train", "env_id": i,
                  "t1": t1, "t2": t2, "n_samples": len(X_train)}
        ))
        
        # Val env
        val_texts = [selected_texts[j] for j in val_idx]
        val_labels = np.array([selected_labels[j] for j in val_idx])
        
        X_val = tokenize_and_embed_with_bert(val_texts, bert_model, max_length, device, pooling)
        Y_val = val_labels.reshape(-1, 1).astype(np.float32)
        
        val_envs.append(Env(
            torch.from_numpy(X_val),
            torch.from_numpy(Y_val),
            meta={"p_select": p_select, "kind": "nlp_size_val", "env_id": i, 
                  "n_samples": len(X_val)}
        ))
    
    # 5) TEST OOD (exemples EXTRÊMES OPPOSÉS uniquement)
    print(f"\n=== Test OOD (opposés extrêmes) ===")
    print(f"  {len(extreme_opposite_texts)} SMS extrêmes:")
    print(f"    - HAM très longs (> seuil SPAM)")
    print(f"    - SPAM très courts (< seuil HAM)")
    print(f"  → Signal spurious INVERSÉ au maximum !")
    
    extreme_labels = np.array(extreme_opposite_labels)
    
    X_test = tokenize_and_embed_with_bert(extreme_opposite_texts, bert_model, max_length, device, pooling)
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
# Varying Confounder Helper (Text + Token -> Y)
# =============================================================================

def get_base_logit(text, model):
    """
    Returns the logit of the base classifier for a given text.
    """
    # Proba class 1
    prob = model.predict_proba([text])[0][1]
    # Logit = log(p / (1-p))
    epsilon = 1e-6
    prob = np.clip(prob, epsilon, 1 - epsilon)
    return np.log(prob / (1 - prob))

# =============================================================================
# Construction d'environnements avec Varying Confounder (NLP)
# =============================================================================

def build_envs_nlp_varying_confounder(
    n: int,
    train_gammas: List[float],
    test_gamma: float,
    seed: int,
    val_frac: float = 0.2,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
) -> Tuple[List[Env], List[Env], Env]:
    """
    Construit des environnements NLP avec un confondeur variable.
    
    Y dépend de X_base (contenu) ET de C (confondeur injecté).
    
    Logique:
    1. On prend un SMS.
    2. On calcule son "logit naturel" via un classifieur pré-entraîné (proxy pour W*X_z).
    3. On standardise ces logits (mean=0, std=1) pour que gamma ait un sens.
    4. On tire un confondeur C (+1 ou -1) aléatoirement.
    5. On injecte le token correspondant à C dans le texte.
    6. On calcule le nouveau Label Y:
       Logit_Final = Logit_Naturel_Std + gamma * C
       Y = 1 si Logit_Final > 0 sinon 0.
       
    Tokens:
    C = +1  -> "winner"
    C = -1  -> "news"
    
    Parameters
    ----------
    n : int
        Nombre d'exemples par environnement.
    train_gammas : List[float]
        Force du confondeur pour chaque environnement d'entraînement.
    test_gamma : float
        Force du confondeur pour le test (généralement 0 ou inversé).
    """
    import joblib
    
    print("Chargement du dataset SMS Spam...")
    all_texts, all_labels = load_sms_spam_dataset(seed=seed)
    
    print("Chargement du modèle de base (Oracle)...")
    try:
        base_model = joblib.load('base_spam_model.pkl')
    except:
        print("Erreur: base_spam_model.pkl introuvable. Veuillez lancer train_base_model.py d'abord.")
        return [], [], None

    # --- Pre-calcul et Standardisation des Logits ---
    print("Pré-calcul des logits de base pour tout le dataset...")
    all_probs = base_model.predict_proba(all_texts)[:, 1]
    epsilon = 1e-6
    all_probs = np.clip(all_probs, epsilon, 1 - epsilon)
    all_raw_logits = np.log(all_probs / (1 - all_probs))
    
    logit_mean = np.mean(all_raw_logits)
    logit_std = np.std(all_raw_logits)
    print(f"Stats Logits Brut: Mean={logit_mean:.2f}, Std={logit_std:.2f}")
    
    all_base_logits = (all_raw_logits - logit_mean) / logit_std
    print(f"Logits Standardisés: Mean={np.mean(all_base_logits):.2f}, Std={np.std(all_base_logits):.2f}")

    # --- Calcule des Directions pour le Tracking des Poids ---
    print("Calcul des directions sémantique et confounder...")
    # 1. Direction Sémantique (Mean Spam - Mean Ham sur textes ORIGINAUX)
    # On prend un subset pour aller vite
    sub_indices = np.random.choice(len(all_texts), min(2000, len(all_texts)), replace=False)
    sub_texts = [all_texts[i] for i in sub_indices]
    sub_X = tokenize_and_embed_with_bert(sub_texts, bert_model, max_length, device, pooling)
    
    sub_Y = np.array([all_labels[i] for i in sub_indices])
    
    mean_spam = np.mean(sub_X[sub_Y == 1], axis=0)
    mean_ham = np.mean(sub_X[sub_Y == 0], axis=0)
    dir_sem = mean_spam - mean_ham
    dir_sem = dir_sem / np.linalg.norm(dir_sem)
    
    # 2. Direction Confounder (Token "winner" - Token "news")
    # On embedde juste ces deux mots
    X_tokens = tokenize_and_embed_with_bert(["winner", "news"], bert_model, max_length, device, pooling)
    dir_conf = X_tokens[0] - X_tokens[1]
    dir_conf = dir_conf / np.linalg.norm(dir_conf)
    
    # Orthogonalité ?
    print(f"Angle entre Sémantique et Confounder: {np.degrees(np.arccos(np.clip(np.dot(dir_sem, dir_conf), -1, 1))):.2f}°")
    
    meta_dirs = {"dir_sem": dir_sem, "dir_conf": dir_conf}

    rng = np.random.default_rng(seed)
    
    # Tokens confondeurs
    token_pos = "winner" # C=+1 (pousse vers Spam)
    token_neg = "news"   # C=-1 (pousse vers Ham)
    
    envs = []
    
    # Fonction interne pour créer un env
    def create_env(n_samples, gamma, seed_env, kind="train"):
        rng_env = np.random.default_rng(seed_env)
        
        # 1. Echantillonner n indices au hasard
        indices = rng_env.choice(len(all_texts), n_samples, replace=True)
        batch_texts = [all_texts[i] for i in indices]
        base_logits = all_base_logits[indices] # Utiliser les versions standardisées
        
        # 2. Tirer C ~ Rademacher (+1 ou -1)
        C = rng_env.choice([-1, 1], size=n_samples)
        
        new_texts = []
        new_labels = []
        
        # 3. Injecter token et calculer Y
        final_logits = base_logits + gamma * C
        new_labels = (final_logits > 0).astype(np.float32)
        
        for t, c in zip(batch_texts, C):
            token = token_pos if c == 1 else token_neg
            new_texts.append(f"{token} {t}")
            
        # 4. Embeddings BERT
        X = tokenize_and_embed_with_bert(new_texts, bert_model, max_length, device, pooling)
        Y = new_labels.reshape(-1, 1)
        
        # Fusionner meta et directions
        meta = {"gamma": gamma, "kind": kind}
        meta.update(meta_dirs)
        
        return Env(torch.from_numpy(X), torch.from_numpy(Y), meta=meta)

    train_envs = []
    val_envs = []
    
    # --- TRAIN ---
    for i, gamma in enumerate(train_gammas):
        print(f"Génération Train Env {i} (gamma={gamma})...")
        # Train
        env_train = create_env(n, gamma, seed + i, kind="train")
        train_envs.append(env_train)
        
        # Val (10% de n)
        env_val = create_env(int(n * val_frac), gamma, seed + 1000 + i, kind="val")
        val_envs.append(env_val)
        
    # --- TEST ---
    print(f"Génération Test Env (gamma={test_gamma})...")
    test_env = create_env(n, test_gamma, seed + 999, kind="test")
    
    return train_envs, val_envs, test_env


# =============================================================================
# NLP Custom Confounding (analogue de custom_confounding synthétique)
# =============================================================================
#
# DAG :
#   Text  ──────────────────────────────────────────▶  Y (label)
#   U ~ Bern(0.5)  ──── α ──────────────────────────▶  Y (confondeur latent)
#   U  ──── Z = U ⊕ Bern(a_e)  ──── token injecté  ──▶  X_spurious
#
# Ce qui varie entre les envs : a_e (bruit sur le lien U → Z)
#   a_e faible → token très corrélé à U (et donc à Y)   [train]
#   a_e fort   → token presque indépendant de U           [test OOD]
#
# Paramètres clés :
#   train_a  : liste des a_e pour les envs de train
#   test_a   : a_e pour le test OOD
#   alpha    : force de la perturbation du label par U (0=U sans effet, 1=U domine)
# =============================================================================

def _inject_token_from_z(
    text: str,
    z: int,
    spurious_tokens: Dict[str, str],
    position: str = "prefix",
) -> str:
    """Injecte le token correspondant à Z∈{0,1} dans le texte.

    Z=1 → token "spam_correlated" (rouge / winner / ...)
    Z=0 → token "ham_correlated"  (vert  / news   / ...)
    """
    token = spurious_tokens["spam_correlated"] if z == 1 else spurious_tokens["ham_correlated"]
    if position == "prefix":
        return f"{token} {text}"
    return f"{text} {token}"


def build_envs_nlp_custom_confounding(
    train_a: List[float],
    test_a: float,
    seed: int = 1,
    alpha: float = 0.5,
    val_frac: float = 0.1,
    bert_model: str = "bert-base-uncased",
    max_length: int = 128,
    device: str = "cpu",
    pooling: str = "mean",
    # paramètres ignorés (pour compatibilité avec la signature NLP générique)
    n: int = 0,
    n_test: Optional[int] = None,
) -> Tuple[List[Env], List[Env], Env]:
    """Construit des environnements NLP avec le mécanisme custom_confounding.

    DAG :
        Text ──→ Y_base  (signal causal texte → label)
        U ~Bernoulli(0.5) ──→ Y_final  (confondeur latent perturbe le label via alpha)
        Z = U ⊕ Bernoulli(a_e) ──→ token injecté  (proxy bruité de U)

    Ce qui varie entre les envs : a_e (fiabilité du token comme proxy de U).
    En train : a_e faible → token très corrélé à U (et donc à Y).
    En test OOD : a_e fort → corrélation inversée / bruit maximal.

    Parameters
    ----------
    train_a : List[float]
        Liste des taux de flip a_e pour les environs de train.
        a_e=0.05 → token = U avec 95% de fiabilité.
        a_e=0.15 → token = U avec 85% de fiabilité.
    test_a : float
        Taux de flip pour le test OOD.
        a_e=0.90 → token est INVERSÉ par rapport à U (corrélation spurieuse inversée).
    alpha : float
        Force de la perturbation du label par U.
        0.0 = U n'afecte pas Y (pas de confounding).
        0.5 = U et texte ont une influence comparable.
        1.0 = U domine totalement Y (signal text ignoré).
    seed : int
        Graine aléatoire.
    val_frac : float
        Fraction de validation (prise dans split train global).
    bert_model, max_length, device, pooling : str / int
        Config BERT (identique aux autres fonctions NLP).

    Returns
    -------
    train_envs, val_envs, test_env
    """
    print("Chargement du dataset SMS Spam...")
    all_texts, all_labels = load_sms_spam_dataset(seed=seed)
    n_total = len(all_texts)
    print(f"Dataset chargé : {n_total} SMS")

    # ── Split global 80/10/10 (même logique que build_envs_nlp_semi_anti_causal) ──
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_total)
    n_test_split = int(n_total * 0.1)
    n_val_split  = int(n_total * 0.1)

    test_indices  = indices[:n_test_split]
    val_indices   = indices[n_test_split:n_test_split + n_val_split]
    train_indices = indices[n_test_split + n_val_split:]

    print(f"\nSplit: Train {len(train_indices)} | Val {len(val_indices)} | Test {len(test_indices)}")

    n_envs = len(train_a)
    samples_per_env = len(train_indices) // n_envs

    spurious_tokens = define_spurious_tokens()
    train_envs, val_envs = [], []

    # ────────────────────────── FONCTION INTERNE ──────────────────────────────
    def _make_env(text_indices: np.ndarray, a_e: float, rng_e: np.random.Generator,
                  kind: str, env_seed: int) -> Env:
        """Génère un Env NLP custom-confounding pour un jeu d'indices donné."""
        texts  = [all_texts[j] for j in text_indices]
        y_base = np.array([all_labels[j] for j in text_indices], dtype=np.float32)
        n_e    = len(texts)

        # 1. Confondeur latent U ~ Bernoulli(0.5)
        U = rng_e.integers(0, 2, size=n_e).astype(np.float32)  # {0, 1}

        # 2. Perturber le label via U : override (pas flip)
        #    Avec probabilité alpha, Y = U (U détermine directement le label)
        #    Sinon, Y = Y_base (signal causal textuel)
        #
        #    Résultat :
        #      P(Y=1 | U=1) = (1-alpha) × P(Y_base=1) + alpha × 1  = haute
        #      P(Y=1 | U=0) = (1-alpha) × P(Y_base=1) + alpha × 0  = basse
        #    → corrélation forte entre U (et donc Z) et Y_final
        override_mask = rng_e.uniform(size=n_e) < alpha
        y_final = y_base.copy()
        y_final[override_mask] = U[override_mask]  # Y = U pour ces exemples

        # 3. Z = U ⊕ Bernoulli(a_e)  (proxy bruité)
        noise = rng_e.uniform(size=n_e) < a_e
        Z = U.copy()
        Z[noise] = 1.0 - Z[noise]
        Z = Z.astype(int)

        # 4. Injecter le token basé sur Z
        texts_mod = [_inject_token_from_z(t, int(z), spurious_tokens) for t, z in zip(texts, Z)]

        # 5. Embeddings BERT
        X = tokenize_and_embed_with_bert(texts_mod, bert_model, max_length, device, pooling)
        Y = y_final.reshape(-1, 1)

        return Env(
            torch.from_numpy(X), torch.from_numpy(Y),
            meta={
                "kind": f"nlp_custom_confounding_{kind}",
                "a": a_e,
                "alpha": alpha,
                "n_samples": n_e,
            }
        )

    # ───────────────────────── TRAIN + VAL ENVS ─────────────────────────────
    for i, a_e in enumerate(train_a):
        print(f"\n=== Train Env {i} (a={a_e}, alpha={alpha}) ===")
        start = i * samples_per_env
        end   = (i + 1) * samples_per_env if i < n_envs - 1 else len(train_indices)
        env_idx = train_indices[start:end]

        rng_tr = np.random.default_rng(seed + i)
        train_envs.append(_make_env(env_idx, a_e, rng_tr, kind="train", env_seed=seed + i))

        # Validation : même pool de validation commun, réutilisé avec le même a_e
        print(f"=== Val Env {i} ===")
        rng_val = np.random.default_rng(seed + 5000 + i)
        val_envs.append(_make_env(val_indices, a_e, rng_val, kind="val", env_seed=seed + 5000 + i))

    # ───────────────────────────── TEST OOD ─────────────────────────────────
    print(f"\n=== Test OOD (a={test_a}, alpha={alpha}) ===")
    rng_test = np.random.default_rng(seed + 777)
    test_env = _make_env(test_indices, test_a, rng_test, kind="test_ood", env_seed=seed + 777)

    print(
        f"\n✅ Done! Train: {sum(e.X.shape[0] for e in train_envs)} | "
        f"Val: {val_envs[0].X.shape[0]} | Test: {test_env.X.shape[0]}"
    )
    return train_envs, val_envs, test_env
