import os
import json
import math
import subprocess
import shutil
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

base_dir = "runs_ilmg" # ilm
# 1) On déclare nos deux jeux de test
test_sets = {
    "in_dist":  "data/val_test/val_ind.txt",
    "out_dist": "data/val_test/val_ood.txt"
}

# 2) On prépare un dictionnaire pour collecter les résultats
all_records = { mode: [] for mode in test_sets }

# 3) On boucle d'abord sur chaque mode (in/out)
for mode, test_file in test_sets.items():
    print(f"\n Lancement des évaluations [{mode}] avec `{test_file}`\n")
    for seed_dir in sorted(os.listdir(base_dir)):
        seed_path = os.path.join(base_dir, seed_dir)
        if not os.path.isdir(seed_path):
            continue

        for model_folder in sorted(os.listdir(seed_path)):
            if not model_folder.startswith("model-"):
                continue

            model_path = os.path.join(seed_path, model_folder)
            print(f"[{mode}] Seed `{seed_dir}` - Modèle `{model_folder}`")

            eval_out = os.path.join(model_path, f"eval_{mode}")
            os.makedirs(eval_out, exist_ok=True)

            cmd = [
                "python3", "-m", "torch.distributed.run",
                "--nproc_per_node=1",
                "--master_port", "29500",
                "run_invariant_mlm.py",
                "--model_name_or_path", model_path,
                "--do_eval",
                "--output_dir", eval_out,
                "--validation_file", test_file
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Échec pour {model_folder} : {e}")
                continue
