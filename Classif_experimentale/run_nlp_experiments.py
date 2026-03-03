#!/usr/bin/env python3
"""
Script pour lancer les expériences NLP (SMS Spam) :
1. Semi Anti-Causal (Tokens injectés 'red'/'green')
2. Biais de sélection basé sur la longueur (Size-based)

Sauvegarde les résultats dans des dossiers plot_nlp_*.
"""

import subprocess
import os
import shutil
from pathlib import Path

# Configuration des expériences
"""
"nlp_semi_anti_causal": {
        "dataset": "nlp_sms_spam",
        "nlp_n_samples": 3000,
        "nlp_p_correct_train": "0.99 0.9",
        "nlp_p_correct_test": 0.0,
        "nlp_label_flip": 0.25,
        "nlp_bert_model": "bert-base-uncased",
        
        # Hyperparamètres d'entraînement
        "erm_steps": 25000,
        "erm_lr": 5e-4,
        "irm_steps": 25000,
        "irm_lr": 5e-4,
        "irm_lambda": 100.0,
        
        "model_kind": "logreg",
        "eval_every": 20,
        "output_dir": "plot_nlp_semi_anti_causal"
    },
    
"""

experiments = {
    "nlp_size_selection": {
        "dataset": "nlp_sms_spam_size_selection",
        "nlp_n_samples": 3000,
        "nlp_selection_p_train": "0.9 0.8",
        "nlp_size_threshold_method": "soft",
        "val_frac": 0.1,
        
        # Hyperparamètres d'entraînement
        "erm_steps": 25000,
        "erm_lr": 5e-4,
        "irm_steps": 25000,
        "irm_lr": 5e-4,
        "irm_lambda": 1000.0,
        
        "model_kind": "logreg",
        "eval_every": 20,
        "output_dir": "plot_nlp_size_selection"
    }
}

def build_command(exp_name, config):
    """Construit la ligne de commande pour une expérience."""
    cmd = ["uv", "run", "main.py"]
    
    # Arguments communs
    cmd.extend(["--dataset", config["dataset"]])
    
    # Entraînement
    cmd.extend(["--erm_steps", str(config["erm_steps"])])
    cmd.extend(["--erm_lr", str(config["erm_lr"])])
    cmd.extend(["--irm_steps", str(config["irm_steps"])])
    cmd.extend(["--irm_lr", str(config["irm_lr"])])
    cmd.extend(["--irm_lambda", str(config["irm_lambda"])])
    
    cmd.extend(["--model_kind", config["model_kind"]])
    cmd.extend(["--eval_every", str(config["eval_every"])])
    cmd.extend(["--device", "auto"])
    
    # Arguments spécifiques NLP
    if "nlp_n_samples" in config:
        cmd.extend(["--nlp_n_samples", str(config["nlp_n_samples"])])
    
    if exp_name == "nlp_semi_anti_causal":
        cmd.extend(["--nlp_p_correct_train"] + config["nlp_p_correct_train"].split())
        cmd.extend(["--nlp_p_correct_test", str(config["nlp_p_correct_test"])])
        cmd.extend(["--nlp_label_flip", str(config["nlp_label_flip"])])
        cmd.extend(["--nlp_bert_model", config.get("nlp_bert_model", "bert-base-uncased")])
        
    elif exp_name == "nlp_size_selection":
        cmd.extend(["--nlp_selection_p_train"] + config["nlp_selection_p_train"].split())
        cmd.extend(["--nlp_size_threshold_method", config["nlp_size_threshold_method"]])
        cmd.extend(["--val_frac", str(config.get("val_frac", 0.1))])
    
    return cmd

def run_experiment(exp_name, config):
    """Lance une expérience et déplace les plots dans le dossier de sortie."""
    print(f"\n{'='*80}")
    print(f"🚀 Lancement de l'expérience NLP: {exp_name}")
    print(f"{'='*80}\n")
    
    # Construire la commande
    cmd = build_command(exp_name, config)
    
    # Afficher la commande
    print(f"Commande: {' '.join(cmd)}\n")
    
    # Lancer l'expérience
    try:
        # Check if uv is installed, otherwise try python directly if in venv
        if shutil.which("uv") is None:
             print("⚠️ 'uv' introuvable, essai avec 'python'...")
             cmd[0:2] = ["python"]
        
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        
        # Créer le dossier de sortie
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(exist_ok=True)
        
        # Déplacer les plots
        plot_dir = Path("plot")
        copied_count = 0
        if plot_dir.exists():
            for plot_file in ["comparison_accuracy.png", "comparison_weights.png", "comparison_loss.png", "comparison_alignment.png", "comparison_distance.png"]:
                src = plot_dir / plot_file
                if src.exists():
                    dst = output_dir / plot_file
                    shutil.copy2(src, dst)
                    print(f"✅ Plot copié: {dst}")
                    copied_count += 1
        
        if copied_count == 0:
            print("⚠️ Aucun plot n'a été trouvé à copier. Vérifiez si l'entraînement a produit des graphiques.")

        print(f"\n✅ Expérience '{exp_name}' terminée avec succès!")
        print(f"📊 Plots sauvegardés dans: {output_dir}/")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Erreur lors de l'exécution de l'expérience '{exp_name}'")
        print(f"Code de sortie: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        return False

def main():
    """Lance toutes les expériences NLP."""
    print("\n" + "="*80)
    print("🔬 LANCEMENT DES EXPÉRIENCES NLP (SMS SPAM)")
    print("="*80)
    
    results = {}
    
    # Lancer chaque expérience
    for exp_name, config in experiments.items():
        success = run_experiment(exp_name, config)
        results[exp_name] = success
    
    # Résumé final
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES EXPÉRIENCES NLP")
    print("="*80)
    
    for exp_name, success in results.items():
        status = "✅ Succès" if success else "❌ Échec"
        output_dir = experiments[exp_name]["output_dir"]
        print(f"{status} - {exp_name:25s} → {output_dir}/")
    
    # Vérifier si toutes les expériences ont réussi
    if all(results.values()):
        print("\n🎉 Toutes les expériences NLP sont terminées avec succès!")
    else:
        print("\n⚠️  Certaines expériences ont échoué. Vérifiez les logs ci-dessus.")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
