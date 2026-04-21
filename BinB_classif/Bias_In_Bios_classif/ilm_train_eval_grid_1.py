import subprocess
import sys
import time
import pandas as pd
import os
from pathlib import Path

# ------------------ CONFIG ------------------
GAPS = [0.10, 0.20, 0.30, 0.40] # 0.00, 
SEED = 0 # seed de génération des datasets
LEARNING_RATES = [5e-3, 5e-5]
NB_STEPS = 10000
N_ENVS = 2

BASE_DIRS = {
    "elm": "runs_elm",
    "ilm": "runs_ilm",
    "ilmg": "runs_ilmg" 
}

GPU_ID = 0

# def make_dataset(gap: float):
#     """
#     Génère un dossier donnees_gap_XXX en MODE PAPIER :
#     p_color1 = 0.2 + gap/2, p_color2 = 0.2 - gap/2
#     proba d'injection alignée = 1 - p_color (gérée dans Bias_in_Bios.py)
#     """
#     root = f"donnees_gap_{int(gap*100):03d}"
#     Path(root).mkdir(parents=True, exist_ok=True)
#     cmd = [
#         sys.executable, "Bias_in_Bios.py",
#         "--gap", f"{gap:.2f}",
#         "--m", str(N_ENVS),
#         "--out_dir", root,
#         "--seed", str(SEED),
#     ]
#     print("[DATA]", " ".join(cmd))
#     subprocess.run(cmd, check=True)
#     # on s'assure que les chemins attendus existent
#     for i in range(1, N_ENVS + 1):
#         assert Path(root, "envs", f"env_{i}.txt").exists(), f"Missing file {root}/envs/env_{i}.txt"
#     assert Path(root, "val_test", "val.txt").exists()
#     assert Path(root, "train_erm.txt").exists()
#     return root


def make_dataset(gap: float):
    """
    Génère un dossier donnees_gap_XXX en MODE PAPIER pour le jouet:
    p_align_e = 1 - p_color_e avec mean=0.2 et gap variable.
    """
    root = f"donnees_gap_{int(gap*100):03d}"
    Path(root).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "toy_letters.py",
        "--gap", f"{gap:.2f}",
        "--m", str(N_ENVS),
        "--out_dir", root,
        "--seed", str(SEED),
        "--n_train_per_env", "20000",   # ajuste si tu veux
        "--n_val", "5000",
        "--val_color_p", "1.0",         # ⇒ p_align_val = 0.0 (OOD "parfait")
    ]
    print("[DATA]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    # garde les asserts:
    for i in range(1, N_ENVS + 1):
        assert Path(root, "envs", f"env_{i}.txt").exists()
    assert Path(root, "val_test", "val.txt").exists()
    assert Path(root, "train_erm.txt").exists()
    return root


# ------------------ TRAINING ------------------
def launch_training(model_key: str, dataset_root: str, lr: float, seed: int,
            nb_steps: int, head_updates: int | None = None, freeze_phi: bool | None = None,
            gpu_id: int = GPU_ID):
    """
    Lance un entraînement pour un dataset_root donné :
      - ERM :           train_erm.txt, mode=ilm (mono-env)
      - iLM :           envs/,        mode=ilm
      - IRM-Games :     envs/,        mode=game + --head_updates_per_encoder_update
    """

    base_dir = BASE_DIRS[model_key]
    Path(base_dir).mkdir(exist_ok=True)
    gap_suffix = Path(dataset_root).name.split("_")[-1]

    if model_key == "elm":
        train_file = os.path.join(dataset_root, "train_erm.txt")
        mode = "ilm"

    elif model_key == "ilm":
        train_file = os.path.join(dataset_root, "envs")
        mode = "ilm"
    
    elif model_key == "ilmg":
        train_file = os.path.join(dataset_root, "envs")
        mode = "game"

    run_name = f"{model_key}_gap{gap_suffix}_lr{lr}_seed{seed}_steps{nb_steps}" + \
               ("" if head_updates is None else f"_K{head_updates}") + \
               ("" if freeze_phi is None else f"_freeze{int(bool(freeze_phi))}")
    
    out_dir = os.path.join(base_dir, run_name)
    Path(out_dir).mkdir(exist_ok=True)

    cmd = [
        sys.executable, "run_invariant_mlm_1.py",
        "--model_name_or_path", "distilbert-base-uncased",
        "--mode", mode,
        "--train_file", train_file,
        "--validation_file", os.path.join(dataset_root, "val_test", "val.txt"),
        "--do_train", "--do_eval",
        "--output_dir", out_dir, "--overwrite_output_dir",
        "--max_seq_length", "128",
        "--per_device_train_batch_size", "64",
        "--gradient_accumulation_steps", "4",
        "--preprocessing_num_workers", "16",
        "--per_device_eval_batch_size", "64",
        "--learning_rate", str(lr),
        "--nb_steps", str(nb_steps),
        "--seed", str(seed),
        "--run_name", run_name,
        "--fp16", "--half_precision_backend", "cuda_amp",
        "--evaluation_strategy", "steps", "--eval_steps", "250",
        "--local_rank", "-1",
    ]

    if mode == "game":
        if head_updates is not None:
            cmd += ["--head_updates_per_encoder_update", str(head_updates)]
        if freeze_phi:
            cmd += ["--freeze_phi"]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print("[TRAIN]", " ".join(cmd), f"(GPU {gpu_id})")
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    t0 = time.time()
    for gap in GAPS:
        root = make_dataset(gap)
        for lr in LEARNING_RATES:
            # ERM (elm) et iLM (ilm)
            launch_training("elm",  root, lr, SEED, NB_STEPS, gpu_id=GPU_ID)
            # launch_training("ilm",  root, lr, SEED, NB_STEPS, gpu_id=GPU_ID)
            # IRM-Games : freeze φ (F-IRM)
            launch_training("ilmg", root, lr, SEED, NB_STEPS, head_updates=1, freeze_phi=True, gpu_id=GPU_ID)
            # IRM-Games : V-IRM avec K = 15, 30
            for K in (15, 30):
                launch_training("ilmg", root, lr, SEED, NB_STEPS, head_updates=K, freeze_phi=False, gpu_id=GPU_ID)
            
    print(f"[DONE] Temps total : {round(time.time() - t0, 2)}s")
