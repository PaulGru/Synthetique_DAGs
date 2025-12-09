#!/usr/bin/env python3
"""
Script pour lancer les 3 expériences (semi-anti-causal, confounding, selection)
et sauvegarder les résultats dans des dossiers séparés.
"""

import subprocess
import os
import shutil
from pathlib import Path

# Configuration des expériences
experiments = {
    "anti_causal": {
        "dataset": "synthetic_semi_anti_causal",
        "n": 200000,
        "n_test": 10000,
        "ps_train": "0.1 0.2",
        "p_test_ood": 1.0,
        "label_flip": 0.25,
        "erm_steps": 45000,
        "irm_steps": 45000,
        "irm_lambda": 220.0,
        "model_kind": "logreg",
        "eval_every": 10,
        "output_dir": "plot_anti_causal"
    },
    "confounding": {
        "dataset": "synthetic_confounding",
        "n": 200000,
        "n_test": 10000,
        "conf_a_train": "0.01 0.11",
        "conf_a_test": 0.99,
        "label_flip": 0.0,
        "erm_steps": 45000,
        "irm_steps": 45000,
        "irm_lambda": 250.0,
        "model_kind": "logreg",
        "eval_every": 10,
        "output_dir": "plot_confounding"
    },
    "selection": {
        "dataset": "synthetic_selection",
        "n": 200000,
        "n_test": 10000,
        "sel_alpha_train": "0.9 0.8",
        "sel_alpha_test": 0.0,
        "label_flip": 0.25,
        "erm_steps": 45000,
        "irm_steps": 45000,
        "irm_lambda": 225.0,
        "model_kind": "logreg",
        "eval_every": 10,
        "output_dir": "plot_selection"
    }
}

def build_command(exp_name, config):
    """Construit la ligne de commande pour une expérience."""
    cmd = ["uv", "run", "main.py"]
    
    # Arguments communs
    cmd.extend(["--dataset", config["dataset"]])
    cmd.extend(["--n", str(config["n"])])
    cmd.extend(["--n_test", str(config["n_test"])])
    cmd.extend(["--label_flip", str(config["label_flip"])])
    cmd.extend(["--erm_steps", str(config["erm_steps"])])
    cmd.extend(["--irm_steps", str(config["irm_steps"])])
    cmd.extend(["--irm_lambda", str(config["irm_lambda"])])
    cmd.extend(["--model_kind", config["model_kind"]])
    cmd.extend(["--eval_every", str(config["eval_every"])])
    cmd.extend(["--device", "auto"])
    
    # Arguments spécifiques au dataset
    if exp_name == "anti_causal":
        cmd.extend(["--ps_train"] + config["ps_train"].split())
        cmd.extend(["--p_test_ood", str(config["p_test_ood"])])
    elif exp_name == "confounding":
        cmd.extend(["--conf_a_train"] + config["conf_a_train"].split())
        cmd.extend(["--conf_a_test", str(config["conf_a_test"])])
    elif exp_name == "selection":
        cmd.extend(["--sel_alpha_train"] + config["sel_alpha_train"].split())
        cmd.extend(["--sel_alpha_test", str(config["sel_alpha_test"])])
    
    return cmd

def run_experiment(exp_name, config):
    """Lance une expérience et déplace les plots dans le dossier de sortie."""
    print(f"\n{'='*80}")
    print(f"🚀 Lancement de l'expérience: {exp_name}")
    print(f"{'='*80}\n")
    
    # Construire la commande
    cmd = build_command(exp_name, config)
    
    # Afficher la commande
    print(f"Commande: {' '.join(cmd)}\n")
    
    # Lancer l'expérience
    try:
        result = subprocess.run(cmd, check=True, cwd=os.getcwd())
        
        # Créer le dossier de sortie
        output_dir = Path(config["output_dir"])
        output_dir.mkdir(exist_ok=True)
        
        # Déplacer les plots
        plot_dir = Path("plot")
        if plot_dir.exists():
            for plot_file in ["comparison_accuracy.png", "comparison_weights.png", "comparison_loss.png"]:
                src = plot_dir / plot_file
                if src.exists():
                    dst = output_dir / plot_file
                    shutil.copy2(src, dst)
                    print(f"✅ Plot copié: {dst}")
        
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
    """Lance toutes les expériences."""
    print("\n" + "="*80)
    print("🔬 LANCEMENT DES 3 EXPÉRIENCES IRM")
    print("="*80)
    
    results = {}
    
    # Lancer chaque expérience
    for exp_name, config in experiments.items():
        success = run_experiment(exp_name, config)
        results[exp_name] = success
    
    # Résumé final
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES EXPÉRIENCES")
    print("="*80)
    
    for exp_name, success in results.items():
        status = "✅ Succès" if success else "❌ Échec"
        output_dir = experiments[exp_name]["output_dir"]
        print(f"{status} - {exp_name:20s} → {output_dir}/")
    
    # Vérifier si toutes les expériences ont réussi
    all_success = all(results.values())
    
    if all_success:
        print("\n🎉 Toutes les expériences sont terminées avec succès!")
    else:
        print("\n⚠️  Certaines expériences ont échoué. Vérifiez les logs ci-dessus.")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
