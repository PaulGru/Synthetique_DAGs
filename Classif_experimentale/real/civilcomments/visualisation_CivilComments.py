import sys
from pathlib import Path as _Path
# Ajoute la racine du projet + le dossier shared/ au chemin Python
_ROOT = _Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "shared") not in sys.path:
    sys.path.insert(0, str(_ROOT / "shared"))

from wilds import get_dataset

# 1. Téléchargement et initialisation du dataset
# (Le téléchargement peut prendre un peu de temps la première fois)
print("Chargement du dataset CivilComments...")
dataset = get_dataset(dataset="civilcomments", download=True)

# 2. On récupère le sous-ensemble d'entraînement
train_data = dataset.get_subset('train')

print(f"Nombre total d'exemples d'entraînement : {len(train_data)}\n")

# 3. Les champs de métadonnées : c'est ici que se trouve notre "Prairie/Désert"
# Cela liste toutes les variables d'environnement annotées pour chaque texte
print("Variables d'environnement (Identités démographiques) :")
print(dataset.metadata_fields)
print("-" * 50)

# 4. Regardons comment un exemple est structuré
# Chaque élément du dataset renvoie un tuple : (texte, label, métadonnées)
index_exemple = 150 # Prenons un exemple au hasard dans le dataset

texte, label, metadata = train_data[index_exemple]

print("\n### EXEMPLE DÉCORTIQUÉ ###")
print(f"TEXTE (L'équivalent de l'animal) :\n> \"{texte}\"\n")

# Le label cible (ce qu'on veut prédire)
est_toxique = "Toxique" if label.item() == 1 else "Non toxique"
print(f"LABEL :\n> {est_toxique} (Valeur brute: {label.item()})\n")

# 5. Extraction de l'environnement (la corrélation trompeuse)
print("ENVIRONNEMENT (L'équivalent du décor) :")
identites_mentionnees = []

# Les métadonnées sont un tenseur contenant des 0 ou des 1 pour chaque groupe
for nom_du_groupe, presence in zip(dataset.metadata_fields, metadata):
    # On ignore la colonne 'y' qui est juste une répétition du label
    if nom_du_groupe != 'y' and presence.item() == 1: 
        identites_mentionnees.append(nom_du_groupe)

if identites_mentionnees:
    print(f"> Identités mentionnées dans ce texte : {', '.join(identites_mentionnees)}")
else:
    print("> Aucune identité démographique spécifique mentionnée.")