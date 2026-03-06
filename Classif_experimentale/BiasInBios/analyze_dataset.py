from datasets import load_dataset
import pandas as pd

def analyze_bias_in_bios_per_split():
    # 1. Chargement du dataset
    print("Chargement du dataset 'LabHC/bias_in_bios' depuis le Hub Hugging Face...\n")
    dataset = load_dataset("LabHC/bias_in_bios")
    
    # Identification du nom exact du set de validation/dev
    dev_split = 'dev' if 'dev' in dataset else 'validation'
    
    # Création d'un dictionnaire contenant les DataFrames pour chaque split
    splits = {
        'Train': dataset['train'].to_pandas(),
        'Test': dataset['test'].to_pandas(),
        'Dev': dataset[dev_split].to_pandas()
    }
    
    # Récupération du mapping des professions (de l'ID vers le nom de la profession en texte)
    profession_feature = dataset['train'].features.get('profession')
    
    # 2. Analyse pour chaque split
    for split_name, df in splits.items():
        print(f"==================================================")
        print(f" ANALYSE DU SET : {split_name.upper()} ({len(df)} exemples)")
        print(f"==================================================\n")
        
        # Application du mapping des professions
        if hasattr(profession_feature, 'int2str'):
            df['profession_name'] = df['profession'].apply(profession_feature.int2str)
        else:
            df['profession_name'] = df['profession']
            
        # A. Nombre et proportion des professions dans ce set
        prof_counts = df['profession_name'].value_counts()
        prof_props = df['profession_name'].value_counts(normalize=True) * 100
        
        df_stats = pd.DataFrame({
            'Nombre d\'exemples': prof_counts,
            'Proportion du set (%)': prof_props
        }).round(2)
        
        print(f"--- Répartition des professions ({split_name}) ---")
        print(df_stats)
        print("\n")
        
        # B. Propension Masculin / Féminin par profession dans ce set
        if 'gender' in df.columns:
            # pd.crosstab avec normalize='index' calcule le pourcentage sur la ligne (intra-profession)
            gender_dist = pd.crosstab(df['profession_name'], df['gender'], normalize='index') * 100
            gender_dist = gender_dist.round(2)
            
            # Renommage des colonnes (0 = Male, 1 = Female dans ce dataset)
            if 0 in gender_dist.columns and 1 in gender_dist.columns:
                gender_dist = gender_dist.rename(columns={0: 'Male (%)', 1: 'Female (%)'})
                
            print(f"--- Propension Homme / Femme par profession ({split_name}) ---")
            print(gender_dist)
        else:
            print("Erreur : La colonne 'gender' n'a pas été trouvée dans le dataset.")
            
        print("\n\n")

if __name__ == "__main__":
    analyze_bias_in_bios_per_split()