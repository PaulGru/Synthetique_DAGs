#!/usr/bin/env python3
"""
coref_irm_debiasing.py — Pipeline IRM pour le débiaisage de la résolution
                         de coréférence genrée.

Protocole :
  1. Charger le dataset GAP (Google Ambiguous Pronouns) — ~4 400 exemples
  2. Construire 2 environnements : pronoms masculins vs féminins
  3. Entraîner un classifieur pairwise (SpanBERT gelé + tête MLP) :
       • ERM (baseline biaisée)
       • IRM (débiaisé)
  4. Évaluer les deux modèles sur WinoBias (100 % non vu)
  5. Comparer les biais : Acc_pro − Acc_anti

Auteur : Script généré pour les expériences de corrélations trompeuses.
"""

import os
import re
import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel

# =====================================================================
# Configuration
# =====================================================================
# Modèle pré-entraîné
# On utilise l'encodeur de base de fastcoref (fine-tuné sur OntoNotes)
# afin d'obtenir des embeddings déjà adaptés à la coréférence.
SPANBERT_MODEL = "biu-nlp/f-coref"

# Hyperparamètres d'entraînement
LR_ENCODER = 1e-5   # Inutilisé car on gèle l'encodeur
LR_HEAD = 1e-3      # Taux d'apprentissage plus élevé pour l'MLP seul
EPOCHS_ERM = 10     # Baseline ERM classique
EPOCHS_IRM = 30     # Plus de temps pour IRM (apprentissage + débiaisage)
BATCH_SIZE = 64     # Plus grand batch size car l'encodeur est gelé
IRM_LAMBDA = 5000.0  # Pénalité IRM cible
WARMUP_FRAC = 0.20  # 10% des steps en warmup strict. Assez pour décoller de 0 sans sur-apprendre.

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pronoms pour la classification des environnements
MASCULINE_PRONOUNS = {"he", "him", "his", "himself"}
FEMININE_PRONOUNS = {"she", "her", "hers", "herself"}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# SECTION 1 : Modèle — CorefPairwiseClassifier
# =====================================================================

class CorefPairwiseClassifier(nn.Module):
    """
    Classifieur pairwise pour la résolution de coréférence.

    Architecture :
    - Encodeur SpanBERT (partiellement dégelé)
    - Tête MLP : [emb_pronom; emb_candidat; emb_pronom ⊙ emb_candidat]
                  → score scalaire

    Pour une phrase donnée avec un pronom et deux candidats A/B,
    on calcule score_A et score_B, puis softmax → probabilité.
    """

    def __init__(self, encoder_name=SPANBERT_MODEL, num_unfrozen_layers=0):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(
            encoder_name, torch_dtype=torch.float32
        )
        hidden_size = self.encoder.config.hidden_size  # 768 pour SpanBERT

        # Geler toutes les couches
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        # Dégeler les "num_unfrozen_layers" dernières couches
        if num_unfrozen_layers > 0:
            for i in range(12 - num_unfrozen_layers, 12):
                for param in self.encoder.encoder.layer[i].parameters():
                    param.requires_grad = True
            # Dégeler aussi le pooler
            if hasattr(self.encoder, "pooler"):
                for param in self.encoder.pooler.parameters():
                    param.requires_grad = True

        # Tête pairwise : [pronom; candidat; pronom⊙candidat] → score
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def _get_span_embedding(self, hidden_states, start, end):
        """Moyenne des embeddings sur un span [start, end] (inclus)."""
        if start > end or start < 0 or end >= hidden_states.size(0):
            return hidden_states.mean(dim=0)
        return hidden_states[start : end + 1].mean(dim=0)

    def forward_single(self, hidden_states, pronoun_span, candidate_span):
        """
        Score un seul candidat par rapport au pronom.

        Args:
            hidden_states: [seq_len, hidden_size]
            pronoun_span: (start, end) indices de tokens
            candidate_span: (start, end) indices de tokens

        Returns:
            score: scalaire
        """
        emb_p = self._get_span_embedding(
            hidden_states, pronoun_span[0], pronoun_span[1]
        )
        emb_c = self._get_span_embedding(
            hidden_states, candidate_span[0], candidate_span[1]
        )

        features = torch.cat([emb_p, emb_c, emb_p * emb_c])
        return self.scorer(features).squeeze(-1)

    def forward(self, input_ids, attention_mask,
                pronoun_spans, candidate_a_spans, candidate_b_spans):
        """
        Calcule les logits [score_A, score_B] pour un batch.

        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            pronoun_spans: list of (start, end) — batch taille
            candidate_a_spans: list of (start, end)
            candidate_b_spans: list of (start, end)

        Returns:
            logits: [batch, 2] — scores pour A et B
        """
        with torch.no_grad():
            outputs = self.encoder(input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [batch, seq_len, hidden]

        batch_logits = []
        for i in range(hidden.size(0)):
            h = hidden[i]  # [seq_len, hidden]
            score_a = self.forward_single(
                h, pronoun_spans[i], candidate_a_spans[i]
            )
            score_b = self.forward_single(
                h, pronoun_spans[i], candidate_b_spans[i]
            )
            batch_logits.append(torch.stack([score_a, score_b]))

        return torch.stack(batch_logits)  # [batch, 2]


# =====================================================================
# SECTION 2 : Dataset GAP
# =====================================================================

class GAPDataset(Dataset):
    """
    Dataset PyTorch pour GAP (Gendered Ambiguous Pronouns).

    Chaque exemple contient :
    - input_ids, attention_mask (tokenisation SpanBERT)
    - pronoun_span : (start_token, end_token)
    - candidate_a_span : (start_token, end_token)
    - candidate_b_span : (start_token, end_token)
    - label : 0 si A est correct, 1 si B est correct
    """

    def __init__(self, exemples, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for ex in exemples:
            processed = self._process_example(ex)
            if processed is not None:
                self.data.append(processed)

    def _char_to_token_span(self, encoding, char_start, char_end):
        """Convertit un span caractère en span token."""
        token_start = encoding.char_to_token(char_start)
        # char_end est le dernier caractère (inclus)
        token_end = encoding.char_to_token(max(0, char_end - 1))

        if token_start is None or token_end is None:
            return None
        return (token_start, token_end)

    def _process_example(self, ex):
        """Convertit un exemple GAP brut en tenseurs."""
        text = ex["Text"]
        pronoun = ex["Pronoun"]
        pronoun_offset = ex["Pronoun-offset"]

        a_name = ex["A"]
        a_offset = ex["A-offset"]
        a_coref = ex["A-coref"]

        b_name = ex["B"]
        b_offset = ex["B-offset"]
        b_coref = ex["B-coref"]

        # Filtrer les cas ambigus (ni A ni B, ou les deux)
        if a_coref == b_coref:
            return None

        label = 0 if a_coref else 1  # 0=A correct, 1=B correct

        # Tokenisation
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )

        # Convertir offsets caractère → offsets token
        pronoun_span = self._char_to_token_span(
            encoding, pronoun_offset, pronoun_offset + len(pronoun)
        )
        a_span = self._char_to_token_span(
            encoding, a_offset, a_offset + len(a_name)
        )
        b_span = self._char_to_token_span(
            encoding, b_offset, b_offset + len(b_name)
        )

        if pronoun_span is None or a_span is None or b_span is None:
            return None

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pronoun_span": pronoun_span,
            "candidate_a_span": a_span,
            "candidate_b_span": b_span,
            "label": label,
            "gender": pronoun.lower(),  # pour le split en environnements
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def gap_collate_fn(batch):
    """Collate function avec padding pour le DataLoader."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    attention_masks = []
    pronoun_spans = []
    candidate_a_spans = []
    candidate_b_spans = []
    labels = []

    for item in batch:
        seq_len = item["input_ids"].size(0)
        pad_len = max_len - seq_len

        input_ids.append(
            F.pad(item["input_ids"], (0, pad_len), value=0)
        )
        attention_masks.append(
            F.pad(item["attention_mask"], (0, pad_len), value=0)
        )
        pronoun_spans.append(item["pronoun_span"])
        candidate_a_spans.append(item["candidate_a_span"])
        candidate_b_spans.append(item["candidate_b_span"])
        labels.append(item["label"])

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_masks),
        "pronoun_spans": pronoun_spans,
        "candidate_a_spans": candidate_a_spans,
        "candidate_b_spans": candidate_b_spans,
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# =====================================================================
# SECTION 3 : Chargement et préparation des données
# =====================================================================

def charger_gap():
    """
    Charge le dataset GAP depuis les fichiers TSV locaux et le sépare
    en deux environnements : masculin et féminin.

    Les fichiers TSV proviennent du repo officiel :
    https://github.com/google-research-datasets/gap-coreference

    Returns:
        env_masc: list d'exemples GAP avec pronoms masculins
        env_fem: list d'exemples GAP avec pronoms féminins
    """
    print("\n📖 Chargement du dataset GAP...")
    import csv

    gap_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "data_gap"
    )
    tsv_files = [
        os.path.join(gap_dir, "gap-development.tsv"),
        os.path.join(gap_dir, "gap-validation.tsv"),
        os.path.join(gap_dir, "gap-test.tsv"),
    ]

    all_examples = []
    for fpath in tsv_files:
        if not os.path.exists(fpath):
            print(f"  ⚠️  Fichier manquant : {fpath}")
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                all_examples.append({
                    "Text": row["Text"],
                    "Pronoun": row["Pronoun"],
                    "Pronoun-offset": int(row["Pronoun-offset"]),
                    "A": row["A"],
                    "A-offset": int(row["A-offset"]),
                    "A-coref": row["A-coref"].strip().upper() == "TRUE",
                    "B": row["B"],
                    "B-offset": int(row["B-offset"]),
                    "B-coref": row["B-coref"].strip().upper() == "TRUE",
                })

    print(f"  📊 Total brut : {len(all_examples)} exemples GAP")

    # Séparer par genre du pronom
    env_masc = []
    env_fem = []
    skipped = 0

    for ex in all_examples:
        pronoun = ex["Pronoun"].lower().strip()

        # Filtrer les cas ambigus
        if ex["A-coref"] == ex["B-coref"]:
            skipped += 1
            continue

        if pronoun in MASCULINE_PRONOUNS:
            env_masc.append(ex)
        elif pronoun in FEMININE_PRONOUNS:
            env_fem.append(ex)
        else:
            skipped += 1

    print(f"  🔵 Environnement masculin : {len(env_masc)} exemples")
    print(f"  🔴 Environnement féminin  : {len(env_fem)} exemples")
    print(f"  ⏭️  Exemples filtrés       : {skipped}")

    return env_masc, env_fem


def charger_winobias_pour_eval():
    """
    Charge le dataset WinoBias officiel (uclanlp/wino_bias) et
    le prépare pour l'évaluation pairwise.

    Pour chaque phrase WinoBias, on identifie :
    - Le pronom et sa position
    - L'antécédent correct (depuis l'annotation)
    - Le deuxième candidat (l'autre mention "The [profession]")
    """
    print("\n📖 Chargement de WinoBias pour évaluation...")
    from datasets import load_dataset

    configs = ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]
    data = {}

    for config_name in configs:
        ds = load_dataset("uclanlp/wino_bias", config_name)
        split = "validation" if "validation" in ds else "test"
        exemples = []

        for ex in ds[split]:
            parsed = _parser_winobias_pairwise(ex)
            if parsed is not None:
                exemples.append(parsed)

        data[config_name] = exemples
        print(f"  📋 {config_name:15s} : {len(exemples)} phrases")

    total = sum(len(v) for v in data.values())
    print(f"  📊 Total : {total} phrases WinoBias prêtes.")
    return data


def _parser_winobias_pairwise(exemple):
    """
    Parse un exemple WinoBias pour l'évaluation pairwise.

    Identifie les deux mentions "The [profession]" dans la phrase
    et détermine laquelle est l'antécédent correct.
    """
    tokens = exemple["tokens"]
    coref = exemple["coreference_clusters"]

    if len(coref) < 4:
        return None

    antecedent_start = int(coref[0])
    antecedent_end = int(coref[1])
    pronoun_start = int(coref[2])
    pronoun_end = int(coref[3])

    texte = " ".join(tokens)

    # Trouver les deux mentions "The [profession]" dans la phrase
    # En WinoBias, les mentions sont toujours "The [profession]"
    mentions = []
    for i, tok in enumerate(tokens):
        if tok.lower() == "the" and i + 1 < len(tokens):
            # Vérifier si c'est une des deux mentions de profession
            # (pas un article avant un nom commun dans la subordonnée)
            if i < pronoun_start:  # Les professions sont avant le pronom
                mentions.append((i, i + 1))

    if len(mentions) < 2:
        # Fallback : utiliser l'antécédent correct et chercher l'autre
        mention_correct = (antecedent_start, antecedent_end)
        # L'autre mention est celle que l'on n'a pas identifiée
        for i, tok in enumerate(tokens):
            if tok.lower() == "the" and i + 1 < len(tokens) and i != antecedent_start:
                if i < pronoun_start:
                    mention_autre = (i, i + 1)
                    mentions = [mention_correct, mention_autre]
                    break

    if len(mentions) < 2:
        return None

    # Déterminer quelle mention est le candidat correct
    # L'annotation dit que (antecedent_start, antecedent_end) est correct
    correct_mention = (antecedent_start, antecedent_end)

    if mentions[0] == correct_mention:
        label = 0  # Candidat A est correct
        mention_a = mentions[0]
        mention_b = mentions[1]
    elif mentions[1] == correct_mention:
        label = 1  # Candidat B est correct
        mention_a = mentions[0]
        mention_b = mentions[1]
    else:
        # L'antécédent ne correspond à aucune mention trouvée
        # Fallback : assigner directement
        mention_a = correct_mention
        # Chercher l'autre mention
        for m in mentions:
            if m != correct_mention:
                mention_b = m
                break
        else:
            mention_b = mentions[1] if mentions[0] == correct_mention else mentions[0]
        label = 0

    pronom = " ".join(tokens[pronoun_start : pronoun_end + 1])
    antecedent_correct = " ".join(tokens[antecedent_start : antecedent_end + 1])

    return {
        "texte": texte,
        "tokens": tokens,
        "pronom": pronom,
        "pronoun_span_tokens": (pronoun_start, pronoun_end),
        "candidate_a_span_tokens": mention_a,
        "candidate_b_span_tokens": mention_b,
        "label": label,  # 0 si A est correct, 1 si B
        "antecedent_correct": antecedent_correct,
    }


# =====================================================================
# SECTION 4 : Entraînement ERM
# =====================================================================

def train_erm(model, env_masc, env_fem, tokenizer, epochs=EPOCHS_ERM):
    """
    Entraîne le modèle avec ERM (Empirical Risk Minimization).

    Combine tous les environnements et minimise la loss globale.
    C'est la baseline biaisée.
    """
    print("\n" + "=" * 60)
    print("🏋️ ENTRAÎNEMENT ERM (baseline)")
    print("=" * 60)

    # Combiner les deux environnements
    all_data = env_masc + env_fem
    random.shuffle(all_data)

    dataset = GAPDataset(all_data, tokenizer)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=gap_collate_fn
    )

    print(f"  📊 Exemples valides : {len(dataset)}")
    print(f"  📊 Batchs/époque   : {len(loader)}")

    criterion = nn.CrossEntropyLoss()
    
    # Differential learning rates
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for p in model.scorer.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam([
        {'params': encoder_params, 'lr': LR_ENCODER},
        {'params': head_params, 'lr': LR_HEAD}
    ], weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(
                input_ids, attention_mask,
                batch["pronoun_spans"],
                batch["candidate_a_spans"],
                batch["candidate_b_spans"],
            )

            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total
        print(
            f"  Époque {epoch + 1:2d}/{epochs} — "
            f"Loss: {avg_loss:.4f} — Acc: {accuracy:.1%}"
        )

    print("  ✅ Entraînement ERM terminé.")
    return model


# =====================================================================
# SECTION 5 : Entraînement IRM
# =====================================================================

def train_irm(model, env_masc, env_fem, tokenizer,
              epochs=EPOCHS_IRM, irm_lambda=IRM_LAMBDA):
    """
    Entraîne le modèle avec IRM (Invariant Risk Minimization).

    Deux environnements :
      - env_masc : phrases avec pronoms masculins
      - env_fem  : phrases avec pronoms féminins

    La pénalité IRM force le modèle à apprendre des features
    invariantes entre les deux environnements, supprimant ainsi
    la corrélation spurieuse genre → profession.

    Implémentation IRMv1 avec le trick du scale parameter,
    identique à models_training.py.
    """
    print("\n" + "=" * 60)
    print("🧪 ENTRAÎNEMENT IRM (débiaisage)")
    print("=" * 60)

    dataset_masc = GAPDataset(env_masc, tokenizer)
    dataset_fem = GAPDataset(env_fem, tokenizer)

    loader_masc = DataLoader(
        dataset_masc, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=gap_collate_fn, drop_last=True
    )
    loader_fem = DataLoader(
        dataset_fem, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=gap_collate_fn, drop_last=True
    )

    print(f"  🔵 Env masculin : {len(dataset_masc)} exemples")
    print(f"  🔴 Env féminin  : {len(dataset_fem)} exemples")

    criterion = nn.CrossEntropyLoss()
    
    # Differential learning rates
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for p in model.scorer.parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam([
        {'params': encoder_params, 'lr': LR_ENCODER},
        {'params': head_params, 'lr': LR_HEAD}
    ], weight_decay=1e-4)

    total_steps = epochs * min(len(loader_masc), len(loader_fem))
    # Le warmup doit être suffisamment long pour que le modèle ERM ait le temps
    # de s'adapter doucement à la pénalité sans s'effondrer (collapse).
    warmup_steps = max(10, int(total_steps * WARMUP_FRAC))

    print(f"  ⚙️  Steps totaux  : {total_steps}")
    print(f"  ⚙️  Warmup steps  : {warmup_steps} ({(WARMUP_FRAC*100):.0f}%)")
    print(f"  ⚙️  IRM λ         : {irm_lambda}")

    model.train()
    step = 0

    for epoch in range(epochs):
        total_loss = 0.0
        total_penalty = 0.0
        correct = 0
        total = 0

        for batch_masc, batch_fem in zip(loader_masc, loader_fem):
            step += 1

            # --- Risque empirique + pénalité IRM par environnement ---
            emp_risk = 0.0
            penalties = []

            for batch_e in [batch_masc, batch_fem]:
                input_ids = batch_e["input_ids"].to(DEVICE)
                attention_mask = batch_e["attention_mask"].to(DEVICE)
                labels = batch_e["labels"].to(DEVICE)

                logits = model(
                    input_ids, attention_mask,
                    batch_e["pronoun_spans"],
                    batch_e["candidate_a_spans"],
                    batch_e["candidate_b_spans"],
                )

                # Risque empirique pour cet env
                loss_e = criterion(logits, labels)
                emp_risk = emp_risk + loss_e

                # Pénalité IRM (IRMv1 : gradient du scale parameter)
                scale = torch.tensor(1.0, device=DEVICE, requires_grad=True)
                loss_scaled = criterion(logits * scale, labels)
                grad_scale = grad(
                    loss_scaled, [scale], create_graph=True
                )[0]
                penalty_e = grad_scale ** 2
                penalties.append(penalty_e)

                # Stats
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            emp_risk = emp_risk / 2.0
            penalty = torch.stack(penalties).mean()

            # On utilise un "step" net plutôt qu'un warmup linéaire pour ne pas 
            # dégrader les performances précocement. 
            # Pendant le warmup : penalty weight = 1.0 (très faible).
            # Après le warmup : penalty weight = irm_lambda.
            if step < warmup_steps:
                lambda_t = 1.0
            else:
                lambda_t = irm_lambda

            objective = emp_risk + lambda_t * penalty
            if lambda_t > 1.0:
                objective = objective / lambda_t

            optimizer.zero_grad()
            objective.backward()
            optimizer.step()

            total_loss += emp_risk.item()
            total_penalty += penalty.item()

        n_batches = min(len(loader_masc), len(loader_fem))
        avg_loss = total_loss / n_batches if n_batches > 0 else 0
        avg_pen = total_penalty / n_batches if n_batches > 0 else 0
        accuracy = correct / total if total > 0 else 0

        print(
            f"  Époque {epoch + 1:2d}/{epochs} — "
            f"Loss: {avg_loss:.4f} — "
            f"Pénalité: {avg_pen:.6f} — "
            f"Acc: {accuracy:.1%}"
        )

    print("  ✅ Entraînement IRM terminé.")
    return model


# =====================================================================
# SECTION 6 : Évaluation sur WinoBias
# =====================================================================

def evaluer_sur_winobias(model, tokenizer, donnees_winobias):
    """
    Évalue le modèle pairwise sur WinoBias.

    Pour chaque phrase WinoBias :
    1. Tokenise la phrase avec SpanBERT
    2. Convertit les spans token en positions dans la tokenisation
    3. Prédit le candidat (A ou B) avec le score le plus élevé
    4. Compare avec le label correct
    """
    model.eval()
    resultats = {}

    with torch.no_grad():
        for subset_name, phrases in donnees_winobias.items():
            correct = 0
            total = 0

            for phrase_info in phrases:
                texte = phrase_info["texte"]
                tokens_wb = phrase_info["tokens"]
                label = phrase_info["label"]

                # Tokeniser avec SpanBERT
                encoding = tokenizer(
                    texte,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256,
                    return_offsets_mapping=True,
                ).to(DEVICE)

                # Convertir les spans WinoBias (indices de mots) en spans
                # de tokens SpanBERT (indices de sous-mots)
                pronoun_span = _word_to_token_span(
                    tokens_wb, phrase_info["pronoun_span_tokens"],
                    encoding, texte
                )
                cand_a_span = _word_to_token_span(
                    tokens_wb, phrase_info["candidate_a_span_tokens"],
                    encoding, texte
                )
                cand_b_span = _word_to_token_span(
                    tokens_wb, phrase_info["candidate_b_span_tokens"],
                    encoding, texte
                )

                if pronoun_span is None or cand_a_span is None or cand_b_span is None:
                    total += 1
                    continue

                logits = model(
                    encoding["input_ids"],
                    encoding["attention_mask"],
                    [pronoun_span],
                    [cand_a_span],
                    [cand_b_span],
                )

                pred = logits.argmax(dim=1).item()
                if pred == label:
                    correct += 1
                total += 1

            accuracy = correct / total if total > 0 else 0.0
            resultats[subset_name] = {
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            }

    return resultats


def _word_to_token_span(words, word_span, encoding, full_text):
    """
    Convertit un span d'indices de mots en span d'indices de sous-mots
    (tokens SpanBERT).

    Args:
        words: list de mots originaux
        word_span: (start_word, end_word) — indices dans `words`
        encoding: sortie du tokenizer avec offset_mapping
        full_text: texte complet (pour calculer les offsets char)
    """
    start_word, end_word = word_span

    # Calculer l'offset caractère du début du premier mot
    char_offset = 0
    for i in range(start_word):
        char_offset += len(words[i])
        if i < len(words) - 1:
            char_offset += 1  # espace

    char_start = char_offset
    span_text = " ".join(words[start_word : end_word + 1])
    char_end = char_start + len(span_text)

    # Convertir en indices de tokens
    token_start = encoding.char_to_token(0, char_start)
    token_end = encoding.char_to_token(0, max(0, char_end - 1))

    if token_start is None or token_end is None:
        return None

    return (token_start, token_end)


# =====================================================================
# SECTION 7 : Affichage des résultats
# =====================================================================

def afficher_resultats(resultats_erm, resultats_irm):
    """Affiche le tableau comparatif ERM vs IRM."""

    print("\n" + "=" * 70)
    print("📊 RÉSULTATS : COMPARAISON ERM vs IRM SUR WINOBIAS")
    print("=" * 70)

    # Tableau détaillé
    print(f"\n{'':4s}{'Sous-ensemble':17s} │ {'ERM':>10s} │ {'IRM':>10s} │ {'Δ':>8s}")
    print("─" * 55)

    for subset in ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]:
        acc_erm = resultats_erm[subset]["accuracy"]
        acc_irm = resultats_irm[subset]["accuracy"]
        delta = acc_irm - acc_erm
        sign = "+" if delta >= 0 else ""

        print(
            f"    {subset:17s} │ {acc_erm:9.1%}  │ {acc_irm:9.1%}  │ {sign}{delta:7.1%}"
        )

    # Biais par type
    print("\n" + "─" * 55)

    for type_name in ["type1", "type2"]:
        pro_key = f"{type_name}_pro"
        anti_key = f"{type_name}_anti"

        bias_erm = (
            resultats_erm[pro_key]["accuracy"]
            - resultats_erm[anti_key]["accuracy"]
        )
        bias_irm = (
            resultats_irm[pro_key]["accuracy"]
            - resultats_irm[anti_key]["accuracy"]
        )

        print(f"\n  {type_name.upper()} :")
        print(f"    Biais ERM = Acc_pro − Acc_anti = {bias_erm:+.3f}")
        print(f"    Biais IRM = Acc_pro − Acc_anti = {bias_irm:+.3f}")
        reduction = abs(bias_erm) - abs(bias_irm)
        if reduction > 0:
            print(f"    ✅ Réduction du biais : {reduction:+.3f}")
        else:
            print(f"    ⚠️  Biais non réduit : {reduction:+.3f}")

    # Biais global
    acc_pro_erm = (
        resultats_erm["type1_pro"]["correct"]
        + resultats_erm["type2_pro"]["correct"]
    )
    acc_anti_erm = (
        resultats_erm["type1_anti"]["correct"]
        + resultats_erm["type2_anti"]["correct"]
    )
    total_pro = (
        resultats_erm["type1_pro"]["total"]
        + resultats_erm["type2_pro"]["total"]
    )
    total_anti = (
        resultats_erm["type1_anti"]["total"]
        + resultats_erm["type2_anti"]["total"]
    )

    bias_global_erm = acc_pro_erm / total_pro - acc_anti_erm / total_anti

    acc_pro_irm = (
        resultats_irm["type1_pro"]["correct"]
        + resultats_irm["type2_pro"]["correct"]
    )
    acc_anti_irm = (
        resultats_irm["type1_anti"]["correct"]
        + resultats_irm["type2_anti"]["correct"]
    )
    bias_global_irm = acc_pro_irm / total_pro - acc_anti_irm / total_anti

    print("\n" + "=" * 70)
    print("🎯 BIAIS GLOBAL :")
    print(f"   ╔══════════════════════════════════════════════════╗")
    print(f"   ║  ERM : Biais = {bias_global_erm:+.3f}                          ║")
    print(f"   ║  IRM : Biais = {bias_global_irm:+.3f}                          ║")
    reduction = abs(bias_global_erm) - abs(bias_global_irm)
    if reduction > 0:
        print(f"   ║  ✅ Réduction globale : {reduction:+.3f}                  ║")
    else:
        print(f"   ║  ⚠️  Pas de réduction : {reduction:+.3f}                  ║")
    print(f"   ╚══════════════════════════════════════════════════╝")
    print("=" * 70)

    # Interprétation
    print("\n📝 INTERPRÉTATION :")
    if reduction > 0.02:
        print(
            "   IRM réduit significativement le biais de genre.\n"
            "   Le modèle apprend à résoudre la coréférence par la\n"
            "   syntaxe et la sémantique, plutôt que par le stéréotype\n"
            "   de genre — l'invariance entre environnements supprime\n"
            "   la corrélation trompeuse."
        )
    elif reduction > 0:
        print(
            "   IRM montre une légère réduction du biais.\n"
            "   Un ajustement des hyperparamètres (λ, lr, epochs)\n"
            "   pourrait améliorer les résultats."
        )
    else:
        print(
            "   IRM n'a pas réussi à réduire le biais avec ces\n"
            "   hyperparamètres. Essayer :\n"
            "   - Augmenter irm_lambda\n"
            "   - Plus d'époques de warmup\n"
            "   - Dégeler les dernières couches de SpanBERT"
        )


# =====================================================================
# SECTION 8 : Main
# =====================================================================

def main():
    print("=" * 70)
    print("🔬 EXPÉRIENCE : DÉBIAISAGE IRM DE LA CORÉFÉRENCE GENRÉE")
    print("   Entraînement : GAP | Évaluation : WinoBias")
    print("=" * 70)
    print(f"🖥️  Device : {DEVICE}")
    print(f"🔧 Encodeur : {SPANBERT_MODEL} (fine-tuné partiellement)")

    set_seed(SEED)

    # --- 1. Charger les données ---
    env_masc, env_fem = charger_gap()
    donnees_winobias = charger_winobias_pour_eval()

    # --- 2. Initialisation du tokenizer ---
    print("\n🔧 Chargement du tokenizer SpanBERT...")
    tokenizer = AutoTokenizer.from_pretrained(SPANBERT_MODEL)

    # --- 3. Entraîner ERM (baseline) ---
    print("\n" + "━" * 70)
    print("  PHASE 1 : Entraînement ERM (baseline biaisée)")
    print("━" * 70)
    model_erm = CorefPairwiseClassifier(
        SPANBERT_MODEL, num_unfrozen_layers=0
    ).to(DEVICE)
    model_erm = train_erm(model_erm, env_masc, env_fem, tokenizer, epochs=EPOCHS_ERM)

    print("\n  📊 Évaluation ERM sur WinoBias...")
    resultats_erm = evaluer_sur_winobias(model_erm, tokenizer, donnees_winobias)
    for subset, res in resultats_erm.items():
        print(f"    {subset:17s} : {res['correct']}/{res['total']} = {res['accuracy']:.1%}")

    # --- 4. Entraîner IRM (débiaisé) ---
    print("\n" + "━" * 70)
    print("  PHASE 2 : Entraînement IRM (débiaisage)")
    print("━" * 70)
    # ⚠️ CRUCIAL : On DOIT réinitialiser un nouveau modèle à zéro.
    # Ne PAS copier model_erm ! Si le modèle est déjà parfait, les logits
    # sont très élevés, et la dérivée de la pénalité vaut mathématiquement 0.0.
    # IRM ne fonctionnerait alors pas.
    model_irm = CorefPairwiseClassifier(
        SPANBERT_MODEL, num_unfrozen_layers=0
    ).to(DEVICE)
    
    model_irm = train_irm(model_irm, env_masc, env_fem, tokenizer, epochs=EPOCHS_IRM, irm_lambda=IRM_LAMBDA)

    print("\n  📊 Évaluation IRM sur WinoBias...")
    resultats_irm = evaluer_sur_winobias(model_irm, tokenizer, donnees_winobias)
    for k, v in resultats_irm.items():
        print(f"    {k:17s} : {v['correct']}/{v['total']} = {v['accuracy']:.1%}")

    # --- 5. Comparaison ---
    afficher_resultats(resultats_erm, resultats_irm)

    print("\n✅ Expérience terminée.")


if __name__ == "__main__":
    main()
