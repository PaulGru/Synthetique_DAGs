# pip install wandb pandas
import re
import pandas as pd
import wandb

ENTITY  = "paul-grunenwaldplecy-institut-polytechnique-de-paris"
PROJECT = "Comparaison" # InvariantLM
METRIC  = "evaluation/ood_perplexity"

# "last" = dernière valeur loggée | "min" = meilleure (min) sur tout l'history
AGG_STRATEGY_PER_RUN = "last"   # mets "min" si tu veux la meilleure perplexité

"""
def parse_from_name(name: str):
    # Extrait algo, lr, seed, model depuis un nom de run style:
    # eval-erm-lr_5e-5-seed_3-model-25000 (variantes acceptées).
    s = (name or "").strip().lower()
    # tolérant aux séparateurs -, _, espace
    algo  = re.search(r"eval[\-_ ](?P<algo>.+?)[\-_ ]lr", s)
    lr    = re.search(r"lr[\-_ ](?P<lr>.+?)[\-_ ]seed", s)
    seed  = re.search(r"seed[\-_ ](?P<seed>\d+)", s)
    model = re.search(r"model[\-_ ](?P<model>[\d _]+)", s)

    algo  = algo.group("algo") if algo else None
    lr    = lr.group("lr") if lr else None
    seed  = int(seed.group("seed")) if seed else None
    if model:
        # enlève espaces/underscores pour 25 000 -> 25000
        m = re.sub(r"[^\d]", "", model.group("model"))
        model = int(m) if m else None
    else:
        model = None

    return algo, lr, seed, model
"""

def parse_from_name(name: str):
    import re
    s = (name or "").strip().lower()

    # algo = juste après "eval-"
    algo_m = re.search(r"^eval[\-_ ](?P<algo>[a-z0-9]+)", s)
    algo = algo_m.group("algo") if algo_m else None

    # lr: "lr_*" OU juste après l'algo ("eval-irm_<lr>"), mais on s'arrête avant 'seed'
    lr_m = re.search(r"lr[\-_ ](?P<lr>[0-9eE.\+\-]+)(?=[\-_ ]seed)", s)
    if lr_m:
        lr = lr_m.group("lr")
    else:
        lr_after = re.search(r"^eval[\-_ ][a-z0-9]+[\-_ ](?P<lr>[0-9eE.\+\-]+)(?=[\-_ ]seed)", s)
        lr = lr_after.group("lr") if lr_after else None

    # seed: accepte "seed4" ou "seed_4"
    seed_m = re.search(r"seed_?(?P<seed>\d+)", s)
    seed = int(seed_m.group("seed")) if seed_m else None

    # model: "model-5000", "model 25 000", etc.
    model_m = re.search(r"model[\-_ ](?P<model>[\d _]+)", s)
    if model_m:
        model = int(re.sub(r"[^\d]", "", model_m.group("model")))
    else:
        model = None

    return algo, lr, seed, model


def get_metric_value(run, metric, strategy="last"):
    hist = run.history(keys=["_step", metric])
    if not hist.empty and metric in hist:
        df = hist.dropna(subset=[metric])
        if df.empty:
            val = run.summary.get(metric); step = run.summary.get("_step")
        else:
            row = df.loc[df[metric].idxmin()] if strategy == "min" else df.iloc[-1]
            val = float(row[metric]); step = int(row["_step"]) if "_step" in row else None
    else:
        val = run.summary.get(metric); step = run.summary.get("_step")
    return val, step

def main():
    api = wandb.Api()
    runs = api.runs(f"{ENTITY}/{PROJECT}")

    rows, unmatched = [], []
    for r in runs:
        algo, lr, seed, model = parse_from_name(r.name)
        if None in (algo, lr, seed, model):
            unmatched.append(r.name)
            continue

        val, step = get_metric_value(r, METRIC, strategy=AGG_STRATEGY_PER_RUN)
        if val is None:
            continue

        rows.append({
            "run": r.name, "algo": algo, "lr": lr, "seed": seed, "model": model,
            "_step": step, METRIC: float(val),
        })

    print(f"Runs totaux: {len(list(runs))}, parsés: {len(rows)}, non parsés: {len(unmatched)}")
    if unmatched:
        print("\nNoms non reconnus (exemples):")
        for nm in unmatched[:15]:
            print(" -", nm)

    if not rows:
        print("\nAucun run parsé → vérifie que l’URL projet est correcte et les noms contiennent bien lr_/seed_/model-.")
        return

    df = pd.DataFrame(rows)

    # Agrégation sur les seeds
    group_keys = ["algo", "lr", "model"]
    agg = (df.groupby(group_keys)[METRIC]
             .agg(["count","mean","std"])
             .rename(columns={"count":"n_seeds",
                              "mean":f"mean_{METRIC}",
                              "std":f"std_{METRIC}"})
             .reset_index())

    # Tri: lr converti en float pour garder l’ordre 1e-5 < 5e-5
    def sort_key(col):
        if col.name == "lr":
            return col.map(lambda x: float(str(x).replace(" ", "")))
        return col
    agg = agg.sort_values(by=["algo", "lr", "model"], key=sort_key)

    print("\nAgrégations par (algo, lr, model):")
    print(agg.to_string(index=False))

    agg.to_csv("wandb_seed_aggregations_ilm.csv", index=False)
    print("\nCSV écrit : wandb_seed_aggregations_ilm.csv")

    # Matrice optionnelle (model x lr)
    pivot = agg.pivot(index="model", columns="lr", values=f"mean_{METRIC}")
    print("\nMatrice (mean par model x lr):")
    print(pivot)

if __name__ == "__main__":
    main()
