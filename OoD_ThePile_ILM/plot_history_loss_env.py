import pandas as pd
import matplotlib.pyplot as plt

# Chemin vers le fichier CSV
log_file_path = "output/thePile_ensLM_InD/training_loss_history.csv"

# Chargement des données
df = pd.read_csv(log_file_path)

# Vérification de la présence de la colonne 'epoch'
if 'Epoch' not in df.columns:
    raise ValueError("La colonne 'epoch' est introuvable dans le fichier CSV.")

# Récupérer les epochs pour l'axe des abscisses
epochs = df['Epoch']

# Sélectionner toutes les colonnes à tracer (excepté 'epoch')
columns_to_plot = [col for col in df.columns if col != 'Epoch']

# Création de la figure
plt.figure(figsize=(10, 6))
for col in columns_to_plot:
    plt.plot(epochs, df[col], marker='o', label=col)

plt.xlabel("Epochs")
plt.ylabel("Valeur")
plt.title("Evolution des métriques au cours des epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Enregistrer le plot dans un fichier
output_plot_path = "training_plot.png"
plt.savefig(output_plot_path)
plt.close()

print(f"Le plot a été enregistré sous le nom: {output_plot_path}")