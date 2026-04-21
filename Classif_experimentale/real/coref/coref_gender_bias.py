#!/usr/bin/env python3
import sys
from pathlib import Path as _Path
# Ajoute la racine du projet + le dossier shared/ au chemin Python
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

"""
==========================================================================
Expérience : Corrélations trompeuses (biais de genre) dans la résolution
             de coréférence avec SpanBERT
==========================================================================

Ce script démontre comment un modèle de résolution de coréférence exhibe
des biais de genre en comparant ses performances sur les sous-ensembles
pro-stéréotypés et anti-stéréotypés du dataset WinoBias.

Architecture :
  - PARTIE A (Éducative) : Classe PyTorch personnalisée montrant
    l'architecture SpanBERT + tête de coréférence.
  - PARTIE B (Pratique)  : Utilisation de fastcoref (modèle LingMess
    basé sur SpanBERT) pour l'évaluation réelle sur WinoBias.

Auteur : Script généré pour expérience de recherche
Usage  : Idéal pour Google Colab avec GPU
"""

# =====================================================================
# SECTION 1 : Installation des dépendances (décommenter pour Colab)
# =====================================================================
# !pip install -q transformers datasets torch fastcoref spacy
# !python -m spacy download en_core_web_sm

# =====================================================================
# SECTION 2 : Imports
# =====================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
from collections import defaultdict
import warnings
import json
import re

warnings.filterwarnings("ignore")

# =====================================================================
# SECTION 3 : Configuration globale et détection GPU
# =====================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Device utilisé : {DEVICE}")

# Nom du modèle SpanBERT pré-entraîné
SPANBERT_MODEL_NAME = "SpanBERT/spanbert-base-cased"

# Hyperparamètres du modèle de coréférence
MAX_SPAN_WIDTH = 8       # Largeur maximale d'un span candidat
HIDDEN_DIM = 768         # Dimension cachée de SpanBERT base
FFNN_DIM = 500           # Dimension des couches feed-forward
DROPOUT = 0.3            # Taux de dropout

# Hyperparamètres d'entraînement
LEARNING_RATE = 1e-5
NB_EPOCHS_DEMO = 2       # Nombre d'époques pour la démo
NB_BATCHES_MAX = 5       # Nombre maximal de batchs par époque


# =====================================================================
# SECTION 4 : Architecture du modèle de coréférence (classe PyTorch)
# =====================================================================
# NOTE : Cette classe est PÉDAGOGIQUE. Elle montre l'architecture
# complète d'un modèle de coréférence end-to-end basé sur SpanBERT,
# tel que décrit dans Lee et al. (2017, 2018) et Joshi et al. (2020).
# Pour l'évaluation réelle, nous utiliserons fastcoref (Section 8).
# =====================================================================

class SpanBERTCorefModel(nn.Module):
    """
    Modèle de résolution de coréférence basé sur SpanBERT.

    Architecture (inspirée de e2e-coref, Joshi et al. 2020) :
      1. Encodeur SpanBERT : encode les tokens en vecteurs contextuels.
      2. Extraction de spans : génère des représentations vectorielles
         pour chaque span candidat (début, fin, attention, largeur).
      3. Scoring de mentions : FFNN identifiant les spans valides.
      4. Scoring d'antécédents : FFNN pairwise liant pronom/antécédent.
    """

    def __init__(self, model_name=SPANBERT_MODEL_NAME,
                 max_span_width=MAX_SPAN_WIDTH,
                 hidden_dim=HIDDEN_DIM,
                 ffnn_dim=FFNN_DIM,
                 dropout=DROPOUT):
        super().__init__()

        # --- 1. Encodeur SpanBERT pré-entraîné ---
        self.encoder = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.hidden_dim = hidden_dim
        self.max_span_width = max_span_width

        # --- 2. Attention pondérée pour résumer le contenu du span ---
        self.span_head_attention = nn.Linear(hidden_dim, 1)

        # --- 3. Embedding de la largeur du span ---
        self.span_width_embedding = nn.Embedding(max_span_width, 20)

        # Dimension totale d'un span : [start; end; head_attn; width_emb]
        self.span_repr_dim = hidden_dim * 3 + 20

        # --- 4. FFNN de scoring des mentions ---
        # Détermine si un span est une mention valide (entité, pronom...)
        self.mention_scorer = nn.Sequential(
            nn.Linear(self.span_repr_dim, ffnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffnn_dim, 1)
        )

        # --- 5. FFNN de scoring pairwise (antécédents) ---
        # Input = [span_i ; span_j ; span_i * span_j]
        pair_input_dim = self.span_repr_dim * 3
        self.antecedent_scorer = nn.Sequential(
            nn.Linear(pair_input_dim, ffnn_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ffnn_dim, 1)
        )

    def _get_span_representation(self, hidden_states, start_idx, end_idx):
        """
        Calcule la représentation d'un span [start_idx, end_idx].

        Représentation = concat(emb_début, emb_fin, head_attention, width).
        """
        span_start_emb = hidden_states[start_idx]
        span_end_emb = hidden_states[end_idx]

        # Attention pondérée sur les tokens internes du span
        span_tokens = hidden_states[start_idx:end_idx + 1]
        attn_weights = F.softmax(
            self.span_head_attention(span_tokens).squeeze(-1), dim=0
        )
        span_head_emb = torch.sum(attn_weights.unsqueeze(-1) * span_tokens, dim=0)

        # Embedding de la largeur du span
        width = min(end_idx - start_idx, self.max_span_width - 1)
        width_tensor = torch.tensor(width, device=hidden_states.device)
        width_emb = self.span_width_embedding(width_tensor)

        return torch.cat([span_start_emb, span_end_emb, span_head_emb, width_emb])

    def forward(self, input_ids, attention_mask,
                candidate_starts=None, candidate_ends=None):
        """
        Forward pass complet.

        Args:
            input_ids      : [1, seq_len] tokens encodés
            attention_mask : [1, seq_len] masque d'attention
            candidate_starts/ends : indices des spans à évaluer
                                    (si None, énumération exhaustive)

        Returns:
            mention_scores     : [num_spans] score de chaque span-mention
            antecedent_scores  : [num_spans, num_spans] scores pairwise
        """
        # 1. Encoder avec SpanBERT
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[0]  # [seq_len, hidden_dim]

        # 2. Énumérer les spans candidats si non fournis
        if candidate_starts is None:
            seq_len = hidden.size(0)
            starts, ends = [], []
            for i in range(seq_len):
                for j in range(i, min(i + self.max_span_width, seq_len)):
                    starts.append(i)
                    ends.append(j)
            candidate_starts = starts
            candidate_ends = ends

        # 3. Calculer les représentations de spans
        span_reprs = []
        for s, e in zip(candidate_starts, candidate_ends):
            span_reprs.append(self._get_span_representation(hidden, s, e))
        span_reprs = torch.stack(span_reprs)  # [num_spans, span_repr_dim]

        # 4. Scorer les mentions
        mention_scores = self.mention_scorer(span_reprs).squeeze(-1)

        # 5. Scorer les paires (antécédent j < mention i)
        n = len(candidate_starts)
        antecedent_scores = torch.zeros(n, n, device=hidden.device)
        for i in range(n):
            for j in range(i):
                pair_repr = torch.cat([
                    span_reprs[i],
                    span_reprs[j],
                    span_reprs[i] * span_reprs[j]
                ])
                antecedent_scores[i, j] = self.antecedent_scorer(pair_repr)

        return mention_scores, antecedent_scores


# =====================================================================
# SECTION 5 : Chargement des données OntoNotes (entraînement)
# =====================================================================
# NOTE : Le dataset conll2012_ontonotesv5 nécessite une licence LDC.
# Sur Hugging Face, il requiert un accès spécial. Nous montrons
# comment le charger, puis utilisons des données synthétiques en
# fallback pour que le script soit exécutable sans la licence.
# =====================================================================

def charger_donnees_ontonotes():
    """
    Tente de charger un sous-ensemble d'OntoNotes 5.0 (CoNLL-2012).
    Retourne des données synthétiques si le dataset n'est pas disponible.
    """
    print("\n📚 Chargement des données d'entraînement...")

    try:
        from datasets import load_dataset
        # Tentative de chargement depuis Hugging Face
        ds = load_dataset(
            "conll2012_ontonotesv5",
            "english_v4",
            split="train",
            streaming=True,
            trust_remote_code=True
        )
        # Prendre un petit sous-ensemble
        exemples = []
        for i, ex in enumerate(ds):
            if i >= 50:
                break
            exemples.append(ex)
        print(f"  ✅ {len(exemples)} exemples chargés depuis OntoNotes.")
        return exemples

    except Exception as e:
        print(f"  ⚠️  OntoNotes non disponible ({type(e).__name__})")
        print("  → Utilisation de données synthétiques pour la démo.")
        return _generer_donnees_synthetiques()


def _generer_donnees_synthetiques():
    """
    Génère des exemples simples de coréférence pour la démonstration.
    Chaque exemple = (phrase, [(mention1_start, mention1_end, cluster_id), ...])
    """
    exemples = [
        {
            "phrase": "Alice went to the store. She bought milk.",
            "mentions": [(0, 0, 0), (6, 6, 0)],  # Alice=She
        },
        {
            "phrase": "The doctor told the nurse that he would handle the case.",
            "mentions": [(0, 1, 0), (3, 4, 1), (6, 6, 0)],  # doctor=he
        },
        {
            "phrase": "Bob called his mother. He was worried about her.",
            "mentions": [(0, 0, 0), (2, 3, 1), (5, 5, 0), (9, 9, 1)],
        },
        {
            "phrase": "The manager met the receptionist and told her the news.",
            "mentions": [(0, 1, 0), (3, 4, 1), (7, 7, 1)],
        },
        {
            "phrase": "The engineer designed the system. He tested it thoroughly.",
            "mentions": [(0, 1, 0), (3, 4, 1), (6, 6, 0), (8, 8, 1)],
        },
    ]
    print(f"  📝 {len(exemples)} exemples synthétiques générés.")
    return exemples


# =====================================================================
# SECTION 6 : Boucle d'entraînement simulée (courte)
# =====================================================================

def entrainer_modele_demo(modele, donnees, tokenizer):
    """
    Boucle d'entraînement courte pour démontrer le fine-tuning.

    NOTE : Cet entraînement est FACTICE (quelques batchs seulement).
    Un vrai entraînement sur OntoNotes prendrait des heures sur GPU.
    Le but est de montrer la structure du code d'entraînement.
    """
    print("\n🏋️ Entraînement simulé du modèle de coréférence...")
    print(f"   Époques : {NB_EPOCHS_DEMO} | Batchs/époque : {NB_BATCHES_MAX}")

    modele.train()
    modele.to(DEVICE)
    optimizer = torch.optim.Adam(modele.parameters(), lr=LEARNING_RATE)

    for epoch in range(NB_EPOCHS_DEMO):
        total_loss = 0.0
        nb_batches = 0

        for idx, exemple in enumerate(donnees):
            if nb_batches >= NB_BATCHES_MAX:
                break

            # Extraire la phrase
            phrase = exemple["phrase"] if isinstance(exemple, dict) else str(exemple)

            # Tokeniser
            inputs = tokenizer(
                phrase,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(DEVICE)

            # Forward pass (avec spans limités pour la démo)
            seq_len = inputs["input_ids"].size(1)
            # Limiter les spans pour accélérer
            max_spans = min(seq_len, 15)
            starts = list(range(max_spans))
            ends = list(range(max_spans))

            mention_scores, antecedent_scores = modele(
                inputs["input_ids"],
                inputs["attention_mask"],
                candidate_starts=starts,
                candidate_ends=ends
            )

            # Loss factice : on pousse les scores de mentions vers 0
            # (en vrai, on utiliserait les annotations de coréférence)
            loss = mention_scores.pow(2).mean() + antecedent_scores.pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            nb_batches += 1

        avg_loss = total_loss / max(nb_batches, 1)
        print(f"   Époque {epoch+1}/{NB_EPOCHS_DEMO} - Loss moyenne : {avg_loss:.4f}")

    print("   ✅ Entraînement simulé terminé.\n")
    modele.eval()
    return modele


# =====================================================================
# SECTION 7 : Données WinoBias pour l'évaluation
# =====================================================================
# WinoBias (Zhao et al., 2018) est un benchmark conçu pour mesurer
# le biais de genre dans les systèmes de coréférence.
#
# Type 1 : Le pronom peut être résolu par des indices syntaxiques.
# Type 2 : La résolution nécessite des connaissances sémantiques.
#
# Pro-stéréotypé  : la résolution correcte ALIGNE avec le stéréotype.
# Anti-stéréotypé : la résolution correcte CONTREDIT le stéréotype.
#
# Un modèle biaisé aura Accuracy_pro >> Accuracy_anti.
# =====================================================================

def charger_winobias():
    """
    Charge le dataset officiel WinoBias (uclanlp/wino_bias) depuis
    Hugging Face.

    Le dataset contient 4 configurations :
      - type1_pro, type1_anti  (résolution par indices syntaxiques)
      - type2_pro, type2_anti  (résolution par connaissances sémantiques)

    Chaque config contient 396 phrases annotées avec les clusters de
    coréférence au format CoNLL.

    Retourne un dictionnaire avec les mêmes 4 clés, où chaque valeur
    est une liste de dicts avec :
      - texte : la phrase complète reconstruite
      - pronom : le pronom à résoudre
      - antecedent_correct : le span textuel de l'antécédent correct
    """
    print("\n📖 Chargement du dataset officiel WinoBias (uclanlp/wino_bias)...")

    from datasets import load_dataset

    configs = ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]
    data = {}

    for config_name in configs:
        ds = load_dataset("uclanlp/wino_bias", config_name)

        # Utiliser le split 'validation' (ou 'test' si validation absent)
        split = "validation" if "validation" in ds else "test"
        exemples = []

        for ex in ds[split]:
            parsed = _parser_exemple_winobias(ex)
            if parsed is not None:
                exemples.append(parsed)

        data[config_name] = exemples
        print(f"  📋 {config_name:15s} : {len(exemples)} phrases chargées")

    total = sum(len(v) for v in data.values())
    print(f"  📊 Total : {total} phrases WinoBias (dataset officiel).")
    return data


def _parser_exemple_winobias(exemple):
    """
    Parse un exemple brut du dataset uclanlp/wino_bias.

    Format du champ 'coreference_clusters' :
      [antecedent_start, antecedent_end, pronoun_start, pronoun_end]
      Les indices correspondent aux positions dans la liste 'tokens'.

    Retourne un dict avec :
      - texte : phrase reconstruite à partir des tokens
      - pronom : le token pronom
      - antecedent_correct : le span textuel de l'antécédent correct
    """
    tokens = exemple["tokens"]
    coref = exemple["coreference_clusters"]

    if len(coref) < 4:
        return None

    # Extraire les indices de coréférence
    antecedent_start = int(coref[0])
    antecedent_end = int(coref[1])
    pronoun_start = int(coref[2])
    pronoun_end = int(coref[3])

    # Reconstruire la phrase
    texte = " ".join(tokens)

    # Extraire le pronom et l'antécédent correct
    pronom = " ".join(tokens[pronoun_start:pronoun_end + 1])
    antecedent_correct = " ".join(tokens[antecedent_start:antecedent_end + 1])

    return {
        "texte": texte,
        "pronom": pronom,
        "antecedent_correct": antecedent_correct,
    }


# =====================================================================
# SECTION 8 : Pipeline d'évaluation sur WinoBias
# =====================================================================

def charger_modele_coref_pretraine():
    """
    Charge un modèle de coréférence pré-entraîné via fastcoref.

    fastcoref fournit des modèles basés sur SpanBERT/LingMess,
    entraînés sur OntoNotes. C'est l'option la plus pratique pour
    obtenir des prédictions de coréférence de qualité.

    Alternative : Si fastcoref n'est pas disponible, on utilise une
    approche basée sur les embeddings de SpanBERT (similarité cosinus).
    """
    print("\n🤖 Chargement du modèle de coréférence pré-entraîné...")

    try:
        # Patch de compatibilité : transformers >= 5.x a supprimé l'attribut
        # 'all_tied_weights_keys' que fastcoref 2.x référence encore.
        from fastcoref.modeling import FCorefModel
        if not hasattr(FCorefModel, 'all_tied_weights_keys'):
            FCorefModel.all_tied_weights_keys = {}

        from fastcoref import FCoref
        modele = FCoref(device=str(DEVICE))
        print("  ✅ Modèle FCoref (basé SpanBERT) chargé via fastcoref.")
        return modele, "fastcoref"

    except ImportError:
        print("  ⚠️  fastcoref non installé.")
        print("  → Utilisation de l'approche par similarité SpanBERT.")
        return None, "spanbert_similarity"

    except Exception as e:
        print(f"  ⚠️  Erreur fastcoref : {e}")
        print("  → Utilisation de l'approche par similarité SpanBERT.")
        return None, "spanbert_similarity"


def evaluer_avec_fastcoref(modele_coref, donnees_winobias):
    """
    Évalue le modèle fastcoref sur les phrases WinoBias.

    Pour chaque phrase :
    1. Exécute la résolution de coréférence.
    2. Récupère les clusters de mentions.
    3. Vérifie si le pronom et l'antécédent correct sont dans le
       même cluster.
    4. Enregistre le résultat (correct/incorrect).
    """
    resultats = {}

    for subset_name, phrases in donnees_winobias.items():
        correct = 0
        total = 0

        for phrase_info in phrases:
            texte = phrase_info["texte"]
            pronom = phrase_info["pronom"]
            antecedent_correct = phrase_info["antecedent_correct"]

            # Prédiction de coréférence
            try:
                preds = modele_coref.predict(texts=[texte])
                clusters = preds[0].get_clusters(as_strings=True)
            except Exception:
                try:
                    clusters = preds.get_clusters(as_strings=True)
                except Exception:
                    clusters = []

            # Vérifier si le pronom est lié au bon antécédent
            if _verifier_coreference(clusters, pronom, antecedent_correct):
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0.0
        resultats[subset_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        print(f"    {subset_name:15s} : {correct}/{total} = {accuracy:.1%}")

    return resultats


def evaluer_avec_spanbert_similarity(donnees_winobias):
    """
    Approche de secours : utilise les embeddings SpanBERT pour
    résoudre la coréférence par similarité cosinus.

    NOTE : Cette approche est simplifiée et ne remplace pas un vrai
    modèle de coréférence, mais elle révèle déjà les biais encodés
    dans les embeddings de SpanBERT.
    """
    print("  📐 Évaluation par similarité cosinus (SpanBERT)...")

    tokenizer = AutoTokenizer.from_pretrained(SPANBERT_MODEL_NAME)
    model = AutoModel.from_pretrained(
        SPANBERT_MODEL_NAME, torch_dtype=torch.float32
    ).to(DEVICE)
    model.eval()

    resultats = {}

    for subset_name, phrases in donnees_winobias.items():
        correct = 0
        total = 0

        for phrase_info in phrases:
            texte = phrase_info["texte"]
            pronom = phrase_info["pronom"]
            antecedent_correct = phrase_info["antecedent_correct"]

            with torch.no_grad():
                inputs = tokenizer(
                    texte, return_tensors="pt",
                    truncation=True, max_length=128
                ).to(DEVICE)
                outputs = model(**inputs)
                hidden = outputs.last_hidden_state[0]

                tokens = tokenizer.tokenize(texte)

                # Trouver la position du pronom et de l'antécédent
                pos_pronom = _trouver_position_token(tokens, pronom)
                pos_antecedent = _trouver_position_token(
                    tokens, antecedent_correct.split()[0]
                )

                if pos_pronom is None or pos_antecedent is None:
                    total += 1
                    continue

                # +1 pour le token [CLS]
                emb_pronom = hidden[pos_pronom + 1]
                emb_antecedent = hidden[pos_antecedent + 1]

                sim = F.cosine_similarity(
                    emb_pronom.unsqueeze(0), emb_antecedent.unsqueeze(0)
                ).item()

                # Si la similarité est > seuil, on considère la prédiction
                # comme "coréférente" (seuil empirique)
                if sim > 0.5:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0.0
        resultats[subset_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
        }
        print(f"    {subset_name:15s} : {correct}/{total} = {accuracy:.1%}")

    return resultats


def _trouver_position_token(tokens, mot):
    """Trouve la position du premier token correspondant au mot."""
    mot_lower = mot.lower()
    for i, token in enumerate(tokens):
        if mot_lower in token.lower():
            return i
    # Essayer avec les sous-mots
    for i, token in enumerate(tokens):
        clean = token.replace("##", "").lower()
        if clean in mot_lower or mot_lower.startswith(clean):
            return i
    return None


def _verifier_coreference(clusters, pronom, antecedent_correct):
    """
    Vérifie si le pronom est résolu vers le bon antécédent dans
    les clusters de coréférence produits par fastcoref.

    Retourne True si le pronom et l'antécédent correct sont dans
    le même cluster, False sinon.
    """
    pronom_lower = pronom.lower()
    correct_lower = antecedent_correct.lower()

    for cluster in clusters:
        cluster_lower = [m.lower() for m in cluster]

        # Le pronom est-il dans ce cluster ?
        pronom_dans_cluster = any(
            pronom_lower == m or pronom_lower in m
            for m in cluster_lower
        )

        if pronom_dans_cluster:
            # L'antécédent correct est-il aussi dans ce cluster ?
            correct_dans_cluster = any(
                correct_lower in m for m in cluster_lower
            )
            return correct_dans_cluster

    # Pronom non trouvé dans aucun cluster
    return False


# =====================================================================
# SECTION 9 : Calcul et affichage du biais
# =====================================================================

def calculer_et_afficher_biais(resultats):
    """
    Calcule et affiche les métriques de biais de genre.

    Métriques :
      - Accuracy_pro  : précision sur les phrases pro-stéréotypées
      - Accuracy_anti : précision sur les phrases anti-stéréotypées
      - Biais = Accuracy_pro - Accuracy_anti
        (> 0 indique un biais de genre du modèle)

    On calcule séparément pour Type 1 et Type 2, puis globalement.
    """
    print("\n" + "=" * 70)
    print("📊 RÉSULTATS : BIAIS DE GENRE DANS LA CORÉFÉRENCE")
    print("=" * 70)

    # --- Résultats par sous-ensemble ---
    print("\n┌─────────────────┬──────────┬──────────┬──────────┐")
    print("│   Sous-ensemble │ Correct  │  Total   │ Accuracy │")
    print("├─────────────────┼──────────┼──────────┼──────────┤")
    for nom, res in resultats.items():
        acc = res["accuracy"]
        cor = res["correct"]
        tot = res["total"]
        print(f"│ {nom:>15s} │ {cor:>8d} │ {tot:>8d} │ {acc:>7.1%}  │")
    print("└─────────────────┴──────────┴──────────┴──────────┘")

    # --- Calcul du biais par type ---
    print("\n" + "-" * 70)
    print("📈 CALCUL DU BIAIS DE GENRE")
    print("-" * 70)

    biais_global_pro_correct = 0
    biais_global_pro_total = 0
    biais_global_anti_correct = 0
    biais_global_anti_total = 0

    for type_num in [1, 2]:
        pro_key = f"type{type_num}_pro"
        anti_key = f"type{type_num}_anti"

        if pro_key in resultats and anti_key in resultats:
            acc_pro = resultats[pro_key]["accuracy"]
            acc_anti = resultats[anti_key]["accuracy"]
            biais = acc_pro - acc_anti

            biais_global_pro_correct += resultats[pro_key]["correct"]
            biais_global_pro_total += resultats[pro_key]["total"]
            biais_global_anti_correct += resultats[anti_key]["correct"]
            biais_global_anti_total += resultats[anti_key]["total"]

            print(f"\n  Type {type_num} :")
            print(f"    Accuracy_pro  = {acc_pro:.3f} ({resultats[pro_key]['correct']}/{resultats[pro_key]['total']})")
            print(f"    Accuracy_anti = {acc_anti:.3f} ({resultats[anti_key]['correct']}/{resultats[anti_key]['total']})")
            print(f"    Biais (Type {type_num}) = Accuracy_pro - Accuracy_anti = {biais:+.3f}")

            if biais > 0.05:
                print(f"    ⚠️  Biais PRO-stéréotypé détecté (le modèle favorise les stéréotypes)")
            elif biais < -0.05:
                print(f"    ℹ️  Biais ANTI-stéréotypé (inhabituel)")
            else:
                print(f"    ✅  Biais faible (modèle relativement équitable)")

    # --- Biais global ---
    if biais_global_pro_total > 0 and biais_global_anti_total > 0:
        acc_pro_global = biais_global_pro_correct / biais_global_pro_total
        acc_anti_global = biais_global_anti_correct / biais_global_anti_total
        biais_global = acc_pro_global - acc_anti_global

        print(f"\n{'=' * 70}")
        print(f"🎯 BIAIS GLOBAL :")
        print(f"   Accuracy_pro  (global) = {acc_pro_global:.3f}")
        print(f"   Accuracy_anti (global) = {acc_anti_global:.3f}")
        print(f"\n   ╔══════════════════════════════════════════════════╗")
        print(f"   ║  Biais = Accuracy_pro - Accuracy_anti = {biais_global:+.3f}   ║")
        print(f"   ╚══════════════════════════════════════════════════╝")

        print(f"\n{'=' * 70}")
        print("📝 INTERPRÉTATION :")
        if biais_global > 0.10:
            print("   Le modèle exhibe un BIAIS DE GENRE SIGNIFICATIF.")
            print("   Il résout mieux la coréférence quand le genre du pronom")
            print("   correspond aux stéréotypes sociétaux des professions.")
            print("   → C'est une corrélation trompeuse : le modèle utilise")
            print("     le stéréotype de genre comme raccourci au lieu des")
            print("     véritables indices linguistiques.")
        elif biais_global > 0.05:
            print("   Le modèle montre un biais de genre MODÉRÉ.")
        else:
            print("   Le biais du modèle est FAIBLE sur cet échantillon.")
        print("=" * 70)

    return resultats


# =====================================================================
# SECTION 10 : Point d'entrée principal
# =====================================================================

def main():
    """
    Fonction principale orchestrant toute l'expérience :
    1. Construction du modèle personnalisé (éducatif)
    2. Entraînement simulé sur données OntoNotes
    3. Évaluation sur WinoBias avec modèle pré-entraîné
    4. Calcul et affichage du biais de genre
    """
    print("=" * 70)
    print("🔬 EXPÉRIENCE : BIAIS DE GENRE EN RÉSOLUTION DE CORÉFÉRENCE")
    print("   Modèle : SpanBERT | Évaluation : WinoBias")
    print("=" * 70)

    # ---- PARTIE A : Architecture et entraînement (éducatif) ----
    print("\n" + "=" * 70)
    print("📐 PARTIE A : Modèle personnalisé SpanBERT + Coréférence")
    print("=" * 70)

    # Charger le tokenizer
    print("\n🔧 Chargement du tokenizer SpanBERT...")
    tokenizer = AutoTokenizer.from_pretrained(SPANBERT_MODEL_NAME)
    print("  ✅ Tokenizer chargé.")

    # Instancier le modèle personnalisé
    print("\n🏗️  Construction du modèle SpanBERTCorefModel...")
    modele_custom = SpanBERTCorefModel(model_name=SPANBERT_MODEL_NAME)
    nb_params = sum(p.numel() for p in modele_custom.parameters())
    nb_trainable = sum(p.numel() for p in modele_custom.parameters() if p.requires_grad)
    print(f"  ✅ Modèle construit.")
    print(f"     Paramètres totaux    : {nb_params:,}")
    print(f"     Paramètres entraîn.  : {nb_trainable:,}")

    # Charger les données d'entraînement
    donnees_train = charger_donnees_ontonotes()

    # Entraînement simulé
    modele_custom = entrainer_modele_demo(modele_custom, donnees_train, tokenizer)

    # ---- PARTIE B : Évaluation sur WinoBias ----
    print("\n" + "=" * 70)
    print("🎯 PARTIE B : Évaluation sur WinoBias (biais de genre)")
    print("=" * 70)

    # Charger WinoBias
    donnees_wb = charger_winobias()

    # Charger le modèle de coréférence pré-entraîné
    modele_coref, methode = charger_modele_coref_pretraine()

    # Évaluation
    print(f"\n📊 Évaluation en cours (méthode : {methode})...")

    if methode == "fastcoref" and modele_coref is not None:
        resultats = evaluer_avec_fastcoref(modele_coref, donnees_wb)
    else:
        resultats = evaluer_avec_spanbert_similarity(donnees_wb)

    # Calcul et affichage du biais
    calculer_et_afficher_biais(resultats)

    print("\n✅ Expérience terminée.")


if __name__ == "__main__":
    main()
