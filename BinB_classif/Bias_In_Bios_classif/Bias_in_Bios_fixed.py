#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bias_in_Bios_fixed_testbalanced.py

Pipeline demandé :
- pool = train+dev+test (HF)
- équilibrage 50/50 par genre à l'intérieur de CHAQUE profession (on downsample le genre majoritaire)
- TEST = K exemples par classe (K/2 M + K/2 F) tirés au hasard depuis le pool équilibré
- TRAIN = le reste (on ne rééquilibre pas davantage)
- Environnements d'entraînement (m, gap, color_p_mean) + val OoD (val_color_p)
- Injection <AAA>/<BBB> sur une petite liste de mots neutres (comme ton script)

Sorties :
  out_dir/
    envs/env_1.txt ... env_m.txt
    val_test/val.txt
    train_erm.txt
"""
import os, argparse, random
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets

def set_seed(s): random.seed(s)

# 1) même détection de colonne texte que ton script
def detect_text_col(ds):
    return next(c for c in ds.column_names if c not in ["profession","gender"])  # idem ton code :contentReference[oaicite:2]{index=2}

# 2) même équilibrage 50/50 par genre dans chaque profession (on downsample le majoritaire)
def balance_by_gender(ds):
    sample = ds[0]['gender']
    male_val, female_val = (0,1) if isinstance(sample, int) else ('male','female')
    professions = sorted(set(ds['profession']))
    keep = []
    for p in professions:
        idxs = [i for i, ex in enumerate(ds) if ex['profession']==p]
        m = [i for i in idxs if ds[i]['gender']==male_val]
        f = [i for i in idxs if ds[i]['gender']==female_val]
        k = min(len(m), len(f))
        if k>0:
            keep += random.sample(m, k) + random.sample(f, k)
    if not keep:
        raise ValueError("Aucun exemple équilibré trouvé (vérifie les labels de genre).")
    return ds.select(keep)  # cf. ton implémentation :contentReference[oaicite:3]{index=3}

def main():
    ap = argparse.ArgumentParser()
    # Spurious (mode papier)
    ap.add_argument("--gap", type=float, default=0.40, help="|p_color(env_i) - p_color(env_j)| ; m>=2")
    ap.add_argument("--m",   type=int,   default=2,    help="nb d'environnements (>=2)")
    ap.add_argument("--color_p_mean", type=float, default=0.2, help="mean des coloring probs")
    ap.add_argument("--val_color_p",  type=float, default=0.9, help="coloring prob pour val (=> p_align = 1 - val_color_p)")
    # Test équilibré
    ap.add_argument("--k_test_per_class", type=int, default=250, help="nombre total par classe en test (=> 50/50 M/F)")
    # I/O
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.join(args.out_dir, "envs"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "val_test"), exist_ok=True)

    # 0) Charger pool global
    tr = load_dataset("LabHC/bias_in_bios", split="train")
    dv = load_dataset("LabHC/bias_in_bios", split="dev")
    te = load_dataset("LabHC/bias_in_bios", split="test")
    pool = concatenate_datasets([tr, dv, te])

    text_col = detect_text_col(pool)  # même logique que ton script :contentReference[oaicite:4]{index=4}

    # 1) Équilibrage par genre par profession (sur le POOL entier)
    pool_bal = balance_by_gender(pool)  # même fonction que ton script :contentReference[oaicite:5]{index=5}

    # 2) TEST
    by_prof = defaultdict(list)
    for i, p in enumerate(pool_bal["profession"]):
        by_prof[p].append(i)

    K_target = args.k_test_per_class
    min_count = min(len(idxs) for idxs in by_prof.values())
    K_eff = min(K_target, min_count)
    if K_eff <= 0:
        raise ValueError("Impossible de construire un test 50/50. Baisse --k_test_per_class.")

    test_idx = []
    for p, idxs in by_prof.items():
        test_idx += random.sample(idxs, K_eff)

    test_idx = sorted(test_idx)
    test_ds  = pool_bal.select(test_idx)
    test_set = set(test_idx)
    # 3) TRAIN = reste du pool équilibré (pas d'autre équilibrage)
    train_ds = pool_bal.filter(lambda ex, idx: idx not in test_set, with_indices=True)

    # 4) Mapping label↔id (sur train)
    professions = sorted(set(train_ds["profession"]))
    label2id = {p:i for i,p in enumerate(professions)}  # même ordre que ton script :contentReference[oaicite:6]{index=6}

    # 5) Préfixes spurious : même logique <AAA>/<BBB> + liste de mots neutres
    group0 = set(range(len(professions)//2))
    def fictive_label(prof):  # 0 ou 1
        return 0 if label2id[prof] in group0 else 1

    target_tokens = {"the","a","and","to","of","in","for","is","with"}  # même liste que ton script :contentReference[oaicite:7]{index=7}
    def inject_prefix(text, prof, p_align):
        y = fictive_label(prof)
        out = []
        for t in text.split():
            if t.lower() in target_tokens:
                pref = ('<AAA>' if y==0 else '<BBB>') if random.random()<p_align else ('<BBB>' if y==0 else '<AAA>')
                out.append(pref + " " + t)
            else:
                out.append(t)
        return " ".join(out)  # identique au style de ton script :contentReference[oaicite:8]{index=8}

    # 6) Probas d’alignement (mode papier)
    m = max(2, int(args.m))
    half = args.gap/2.0
    offsets   = [(-half + j*(args.gap/(m-1))) if m>1 else 0.0 for j in range(m)]
    p_colors  = [max(0.0, min(1.0, args.color_p_mean + off)) for off in offsets]
    p_aligns  = [1.0 - pc for pc in p_colors]   # proba d'injection alignée par env :contentReference[oaicite:9]{index=9}
    p_align_val = 1.0 - args.val_color_p        # val inversée si val_color_p>0.5 :contentReference[oaicite:10]{index=10}
    print(f"[paper-mode] p_color={p_colors} → p_align_train={p_aligns} | p_align_val={p_align_val:.2f}")

    # 7) Répartition TRAIN en m environnements (round-robin par classe pour stabilité)
    by_prof_train = defaultdict(list)
    for i, p in enumerate(train_ds["profession"]):
        by_prof_train[p].append(i)
    env_indices = {e: [] for e in range(1, m+1)}
    for p, idxs in by_prof_train.items():
        idxs = idxs[:]
        random.shuffle(idxs)
        for r, idx in enumerate(idxs):
            env_id = (r % m) + 1
            env_indices[env_id].append(idx)

    # 8) Écriture : envs/* + train_erm + val
    train_erm_lines = []
    for e in range(1, m+1):
        with open(os.path.join(args.out_dir, "envs", f"env_{e}.txt"), "w", encoding="utf-8") as f:
            p_align = p_aligns[e-1]
            for idx in env_indices[e]:
                ex = train_ds[idx]
                txt = inject_prefix(ex[text_col], ex["profession"], p_align)
                lab = label2id[ex["profession"]]
                line = f"{txt}\t{lab}\n"
                f.write(line)
                train_erm_lines.append(line)

    with open(os.path.join(args.out_dir, "train_erm.txt"), "w", encoding="utf-8") as f:
        random.shuffle(train_erm_lines)
        f.writelines(train_erm_lines)

    with open(os.path.join(args.out_dir, "val_test", "val.txt"), "w", encoding="utf-8") as f:
        for ex in test_ds:
            txt = inject_prefix(ex[text_col], ex["profession"], p_align_val)
            lab = label2id[ex["profession"]]
            f.write(f"{txt}\t{lab}\n")

    print(f"OK → {args.out_dir}/envs/env_1.txt..env_{m}.txt ; val_test/val.txt ; train_erm.txt")
    print(f"[Stats] classes={len(professions)} | test K_eff={K_eff}/classe | train={len(train_ds)} ex")

if __name__ == "__main__":
    main()
