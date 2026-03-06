"""
find_similar_professions.py — Analyse de similarité sémantique entre les professions.

Objectif : Trouver deux professions qui sont :
 1. Très proches sémantiquement (vocabulaire textuel similaire).
 2. Avec des distributions de genre opposées (ex: l'un majoritairement masculin, l'autre féminin).

Méthode :
 - Vecteurs TF-IDF sur le split train du dataset.
 - Calcul du centroïde moyen (profil moyen) pour chaque profession.
 - Calcul de la similarité cosinus entre toutes les paires.
 - Tri et affichage des paires les plus proches.
"""

from __future__ import annotations

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# On reprend les 22 professions déséquilibrées définies dans data_bios.py
_ALL_PROFESSIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker", "interior_designer",
    "journalist", "model", "nurse", "painter", "paralegal",
    "pastor", "personal_trainer", "photographer", "physician", "poet",
    "professor", "psychologist", "rapper", "software_engineer", "surgeon",
    "teacher", "yoga_teacher"
]
_BALANCED = {"journalist", "personal_trainer", "photographer", "model", "paralegal", "physician"}
PROFESSIONS = [p for p in _ALL_PROFESSIONS if p not in _BALANCED]
PROF2IDX = {p: i for i, p in enumerate(PROFESSIONS)}


def main():
    print("Chargement du dataset Bias in Bios...")
    ds = load_dataset("LabHC/bias_in_bios", split="train")

    texts = []
    labels = []
    genders = []

    print("Filtrage des 22 professions déséquilibrées...")
    for example in ds:
        bio = example.get("hard_text", "").strip()
        prof_ds_idx = example.get("profession")
        gender = example.get("gender")

        if not bio or prof_ds_idx is None or gender is None:
            continue

        prof_name = _ALL_PROFESSIONS[int(prof_ds_idx)]
        if prof_name in PROF2IDX:
            texts.append(bio)
            labels.append(PROF2IDX[prof_name])
            genders.append(int(gender))

    print(f"Total exemples conservés : {len(texts)}")

    # Calcul de la probabilité d'être une femme P(F) par profession
    print("Calcul des statistiques de genre...")
    labels_np = np.array(labels)
    genders_np = np.array(genders)
    
    p_female = {}
    for k, prof in enumerate(PROFESSIONS):
        mask = (labels_np == k)
        if mask.sum() > 0:
            p_female[prof] = float(genders_np[mask].mean())
        else:
            p_female[prof] = 0.5

    # Vectorisation TF-IDF
    print("Vectorisation TF-IDF (max 10000 features)...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english")
    X_tfidf = vectorizer.fit_transform(texts)

    # Calcul des centroïdes
    print("Calcul des centroïdes sémantiques...")
    centroids = np.zeros((len(PROFESSIONS), X_tfidf.shape[1]))
    for k in range(len(PROFESSIONS)):
        mask = (labels_np == k)
        if mask.any():
            # Moyenne des vecteurs TF-IDF pour cette profession
            centroids[k] = np.asarray(X_tfidf[mask].mean(axis=0)).flatten()

    # Similarité cosinus
    print("Calcul de la similarité cosinus pairwise...")
    sim_matrix = cosine_similarity(centroids)

    # Recherche des meilleures paires
    results = []
    for i in range(len(PROFESSIONS)):
        for j in range(i + 1, len(PROFESSIONS)):
            prof1 = PROFESSIONS[i]
            prof2 = PROFESSIONS[j]
            sim = sim_matrix[i, j]
            
            p_f1 = p_female[prof1]
            p_f2 = p_female[prof2]
            
            # Écart de genre absolu (plus il est grand, plus les genres sont opposés)
            gender_diff = abs(p_f1 - p_f2)
            
            results.append({
                "prof1": prof1,
                "prof2": prof2,
                "sim": sim,
                "p_f1": p_f1,
                "p_f2": p_f2,
                "gender_diff": gender_diff,
            })

    # Tri par similarité décroissante
    results.sort(key=lambda x: x["sim"], reverse=True)

    print("\n" + "="*80)
    print(" TOP 15 DES PAIRES LES PLUS SIMILAIRES SÉMANTIQUEMENT")
    print("="*80)
    fmt = "{:<20} & {:<20} | Sim: {:.3f} | Genre 1: {:>4.0f}% F | Genre 2: {:>4.0f}% F | Δ Genre: {:.0f}%"
    
    for r in results[:15]:
        print(fmt.format(
            r["prof1"], r["prof2"], r["sim"], 
            r["p_f1"]*100, r["p_f2"]*100, r["gender_diff"]*100
        ))

    print("\n" + "="*80)
    print(" TOP 10 PAIRES (SIMILARITÉ > 0.5) AVEC LE PLUS FORT ÉCART DE GENRE")
    print("="*80)
    # On filtre les paires qui sont un minimum similaires (>0.5)
    candidates_diff = [r for r in results if r["sim"] > 0.5]
    candidates_diff.sort(key=lambda x: x["gender_diff"], reverse=True)
    
    for r in candidates_diff[:10]:
         print(fmt.format(
            r["prof1"], r["prof2"], r["sim"], 
            r["p_f1"]*100, r["p_f2"]*100, r["gender_diff"]*100
        ))

if __name__ == "__main__":
    main()
