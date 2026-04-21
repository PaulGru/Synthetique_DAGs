import warnings
from datasets import load_dataset
from transformers import DistilBertTokenizerFast
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

# Optionnel : Supprimer les warnings relatifs à la longueur de séquence
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum")

# Charger le dataset (petite version de ThePile)
dataset = load_dataset("ola13/small-the_pile", split="train")

# Afficher un exemple pour vérifier la structure du dataset
print("Exemple de données :", dataset[0])

# Initialiser le tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Regrouper les exemples par environnement en cherchant la clé "pile_set_name"
env_examples = defaultdict(list)
for example in tqdm(dataset, desc="Regroupement par environnement"):
    # Tenter de récupérer l'environnement directement ou via le champ "meta"
    if "pile_set_name" in example:
        env = example["pile_set_name"]
    elif "meta" in example and isinstance(example["meta"], dict) and "pile_set_name" in example["meta"]:
        env = example["meta"]["pile_set_name"]
    else:
        env = "unknown"
    env_examples[env].append(example)

# Calculer les statistiques par environnement
stats = {}
for env, examples in env_examples.items():
    num_examples = len(examples)
    total_tokens = 0
    total_bytes = 0
    for ex in tqdm(examples, desc=f"Traitement de {env}", leave=False):
        text = ex["text"]
        # Utiliser tokenize() pour éviter le warning lié à la longueur des séquences
        tokens = tokenizer.tokenize(text)
        total_tokens += len(tokens)
        total_bytes += len(text.encode("utf-8"))
    stats[env] = {
        "num_examples": num_examples,
        "token_count": total_tokens,
        "byte_count": total_bytes,
    }

# Calcul des totaux sur tous les environnements
global_tokens = sum(s["token_count"] for s in stats.values())
global_bytes  = sum(s["byte_count"]  for s in stats.values())

# Calculer les pourcentages
for env, s in stats.items():
    s["token_percentage"] = s["token_count"] / global_tokens * 100
    s["byte_percentage"]  = s["byte_count"] / global_bytes * 100

# Affichage des résultats dans un DataFrame pour une lecture claire
df = pd.DataFrame.from_dict(stats, orient="index")
print(df)
