#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate spurious-environment splits for the Bias in Bios dataset
avec injection de préfixes sur tokens fréquents et équilibrage par genre.

- Charge les splits train/test de Bias in Bios depuis Hugging Face.
- Équilibre les splits train et test par genre pour neutraliser le biais naturel
  (même nombre d'hommes et de femmes par profession).
- Sélectionne un ensemble restreint de tokens fréquents pour injection de préfixes.
- Crée deux catégories fictives (0/1) pour le choix des préfixes (<AAA>/<BBB>).
- Génère 3 environnements d'entraînement avec p={0.9,0.7,0.5}.
- Génère un environnement de validation/test avec p=0.1 (inversion de corrélation) sur le test équilibré.
- Sauvegarde les fichiers sous donnees/envs/, donnees/val_test/ et donnees/train_erm.txt.
"""
import os
import argparse
import random
from datasets import load_dataset, concatenate_datasets

# Reproductibilité
random.seed(0)

# 3. Fonction d'équilibrage par genre pour un split donné
def balance_by_gender(ds):
    sample = ds[0]['gender']
    if isinstance(sample, int):
        male_val, female_val = 0, 1
    else:
        male_val, female_val = 'male', 'female'
    professions = sorted(set(ds['profession']))
    indices = []
    for prof in professions:
        idxs = [i for i, ex in enumerate(ds) if ex['profession'] == prof]
        male_idxs   = [i for i in idxs if ds[i]['gender'] == male_val]
        female_idxs = [i for i in idxs if ds[i]['gender'] == female_val]
        k = min(len(male_idxs), len(female_idxs))
        if k > 0:
            # Échantillonnage équilibré par genre
            indices.extend(random.sample(male_idxs, k))
            indices.extend(random.sample(female_idxs, k))
    if not indices:
        raise ValueError("Aucun exemple équilibré trouvé : vérifiez les labels de genre.")
    return ds.select(indices)



def main():
    p = argparse.ArgumentParser()
    # Mode historique (tu peux toujours donner p1/p2 directement en 'proba d'injection alignée')
    p.add_argument("--p1", type=float, default=None, help="(legacy) Proba d'injection alignée pour l'environnement 1")
    p.add_argument("--p2", type=float, default=None, help="(legacy) Proba d'injection alignée pour l'environnement 2")
    p.add_argument("--val_p", type=float, default=None, help="(legacy) Proba d'injection alignée pour la validation/test (OoD)")

    # Mode papier (Empirical Study of IRM) : on définit les 'coloring probabilities'
    p.add_argument("--gap", type=float, default=None,
                   help="gap = |p1 - p2| défini sur les 'coloring probabilities' du papier (active le mode papier si non nul)")
    p.add_argument("--color_p_mean", type=float, default=0.2,
                   help="moyenne des 'coloring probabilities' (papier) -> (p1+p2)/2 (par défaut 0.2)")
    p.add_argument("--val_color_p", type=float, default=0.9,
                   help="coloring probability pour la validation/test OoD (papier). La proba d'injection alignée utilisée sera 1 - val_color_p.")
    p.add_argument("--m", type=int, default=2,
               help="Nombre d'environnements d'entraînement en mode 'papier'.")
               
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--balance_test", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    env_dir = os.path.join(args.out_dir, "envs")
    val_dir = os.path.join(args.out_dir, "val_test")
    os.makedirs(env_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    random.seed(args.seed)

    train_ds = load_dataset("LabHC/bias_in_bios", split="train")
    validation_ds = load_dataset("LabHC/bias_in_bios", split="dev")
    train_ds = concatenate_datasets([train_ds, validation_ds])
    test_ds  = load_dataset("LabHC/bias_in_bios", split="test")

    # 2. Détection automatique de la colonne texte
    text_col = next(c for c in train_ds.column_names if c not in ['profession', 'gender'])

    # 4. Équilibrage des splits
    train_ds = balance_by_gender(train_ds)
    test_ds = balance_by_gender(test_ds)

    # 5. Mapping professions → IDs entiers
    professions = sorted(set(train_ds['profession']))
    label2id    = {prof: idx for idx, prof in enumerate(professions)}

    # 6. Catégories fictives pour préfixes (0/1)
    group0 = set(range(len(professions) // 2))
    def get_fictive_label(prof):
        return 0 if label2id[prof] in group0 else 1

    # 7. Tokens cibles pour injection
    target_tokens = {"the", "a", "and", "to", "of", "in", "for", "is", "with"}

    # 9. Fonction d'injection de préfixe
    def inject_prefix(text, prof, p):
        y = get_fictive_label(prof)
        out = []
        for t in text.split():
            if t.lower() in target_tokens:
                prefix = ('<AAA>' if y == 0 else '<BBB>') if random.random() < p else ('<BBB>' if y == 0 else '<AAA>')
                out.append(prefix + ' ' + t)
            else:
                out.append(t)
        return ' '.join(out)
    
    # 10. Répartition aléatoire du train équilibré en environnements
    train_shuf = train_ds.shuffle(seed=args.seed)
    
    # --- Choix des probabilités selon le mode ---
    if args.gap is not None:
        m = max(2, args.m)
        half = args.gap / 2.0
        # offsets également espacés de -gap/2 à +gap/2 (m points)
        offsets = [(-half + j * (args.gap / (m - 1))) if m > 1 else 0.0 for j in range(m)]
        p_colors = [max(0.0, min(1.0, args.color_p_mean + off)) for off in offsets]
        env_probs = {i + 1: 1.0 - p_colors[i] for i in range(m)}  # proba d'injection alignée
        val_p = 1.0 - args.val_color_p
        print(f"[paper-mode] gap={args.gap:.2f} | p_color={p_colors} → p_inj={[round(env_probs[i],2) for i in env_probs]} | val_inj={val_p:.2f}")

    else:
        # ---- Mode legacy (p1/p2 fournis directement en proba d'injection alignée) ----
        if args.p1 is None or args.p2 is None:
            raise ValueError("En mode legacy, spécifie --p1 et --p2 (probas d'injection alignée). "
                             "Sinon, utilise --gap pour activer le mode 'papier'.")
        env_probs = {1: args.p1, 2: args.p2}
        val_p = args.val_p if args.val_p is not None else 0.10
        print(f"[legacy-mode] p_inj=(e1:{args.p1:.2f}, e2:{args.p2:.2f}) | val_inj={val_p:.2f}")


    env_ids = sorted(env_probs.keys())
    n_env      = len(env_probs)
    size       = len(train_shuf) // n_env
    env_slices = {
        env_id: (range(i*size, (i+1)*size) if i < n_env-1 else range(i*size, len(train_shuf)))
        for i, env_id in enumerate(env_ids)
    }

    # 11. Préparation des dossiers de sortie
    os.makedirs(args.out_dir, exist_ok=True)
    train_erm = []

    # 12. Génération des environnements d'entraînement
    for env_id, sl in env_slices.items():
        p = env_probs[env_id]
        with open(os.path.join(env_dir, f"env_{env_id}.txt"), "w", encoding="utf-8") as fout:
            for i in sl:
                ex = train_shuf[i]
                txt = inject_prefix(ex[text_col], ex["profession"], env_probs[env_id])
                lab = label2id[ex['profession']]
                fout.write(f"{txt}\t{lab}\n")
                train_erm.append(f"{txt}\t{lab}\n")

    # 13. Génération du set validation/test équilibré par genre
    with open(os.path.join(val_dir, "val.txt"), "w", encoding="utf-8") as fout:
        for ex in test_ds:
            txt = inject_prefix(ex[text_col], ex["profession"], val_p)
            lab = label2id[ex['profession']]
            fout.write(f"{txt}\t{lab}\n")

    # 14. Agrégation ERM et shuffle
    with open(os.path.join(args.out_dir, "train_erm.txt"), 'w', encoding='utf-8') as f:
        random.shuffle(train_erm)
        f.writelines(train_erm)

    print(f"OK → {args.out_dir}/envs/env_1.txt, envs/env_2.txt, val_test/val.txt, train_erm.txt")


if __name__ == "__main__":
    main()



"""
Expe_tentative_3envs :
env_probs = {1: 0.9, 2: 0.8, 3: 0.7}
val_p     = 0.05
"""