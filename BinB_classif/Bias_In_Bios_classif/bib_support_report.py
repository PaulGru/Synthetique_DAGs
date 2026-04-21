#!/usr/bin/env python3
import argparse, numpy as np
from collections import Counter
from datasets import load_dataset, concatenate_datasets

def balance_by_gender(ds):
    sample = ds[0]['gender']
    male_val, female_val = (0,1) if isinstance(sample,int) else ('male','female')
    professions = sorted(set(ds['profession']))
    idxs_keep = []
    import random
    for prof in professions:
        idxs = [i for i, ex in enumerate(ds) if ex['profession']==prof]
        m = [i for i in idxs if ds[i]['gender']==male_val]
        f = [i for i in idxs if ds[i]['gender']==female_val]
        k = min(len(m), len(f))
        if k>0:
            idxs_keep += random.sample(m, k) + random.sample(f, k)
    if not idxs_keep:
        raise ValueError("Aucun exemple équilibré (genre) trouvé.")
    return ds.select(idxs_keep)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min_train_per_class", type=int, default=0)
    ap.add_argument("--min_val_per_class", type=int, default=0)
    args = ap.parse_args()

    tr = load_dataset("LabHC/bias_in_bios", split="train")
    dev = load_dataset("LabHC/bias_in_bios", split="dev")
    te  = load_dataset("LabHC/bias_in_bios", split="test")

    train = concatenate_datasets([tr, dev])
    train = balance_by_gender(train)   # idem ton script
    val   = balance_by_gender(te)

    profs = sorted(set(train['profession']))
    def counts(ds):
        c = Counter(ds['profession'])
        return {p: c.get(p,0) for p in profs}

    train_counts = counts(train)
    val_counts   = counts(val)

    # résumé
    n_total = len(profs)
    n_low_train = sum(1 for p in profs if train_counts[p] < args.min_train_per_class)
    n_low_val   = sum(1 for p in profs if val_counts[p]   < args.min_val_per_class)

    print(f"[Résumé] {n_total} professions après équilibrage (train+dev).")
    if args.min_train_per_class>0:
        print(f" - {n_low_train} classes < {args.min_train_per_class} en TRAIN")
    if args.min_val_per_class>0:
        print(f" - {n_low_val} classes < {args.min_val_per_class} en VAL")

    # top/bottom pour inspection rapide
    top_val = sorted(val_counts.items(), key=lambda kv: kv[1], reverse=True)[:10]
    low_val = sorted(val_counts.items(), key=lambda kv: kv[1])[:10]
    print("\n[VAL] Top 10 supports:", top_val)
    print("[VAL] Bottom 10 supports:", low_val)

    # CSV minimal
    import csv
    with open("bib_support_report.csv","w",newline="",encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["profession","train_count","val_count"])
        for p in profs:
            w.writerow([p, train_counts[p], val_counts[p]])
    print("→ bib_support_report.csv écrit.")

if __name__ == "__main__":
    main()
