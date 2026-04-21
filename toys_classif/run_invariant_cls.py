#!/usr/bin/env python
# coding=utf-8
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict

from datasets import load_dataset

from invariant_trainer import InvariantTrainer
from invariant_roberta_cls import InvariantRobertaForSequenceClassification, InvariantRobertaConfig
from invariant_distilbert_cls import InvariantDistilBertForSequenceClassification, InvariantDistilBertConfig

import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process, EvalPrediction

# --- type resolution guard for some HF versions
try:
    from transformers.training_args import ParallelismConfig  # noqa: F401
except Exception:
    class ParallelismConfig:  # type: ignore
        pass

logger = logging.getLogger(__name__)
RESULTS_ROOT = os.path.join(os.getcwd(), "eval_results_cls")
os.makedirs(RESULTS_ROOT, exist_ok=True)


# ---- metrics (accuracy) ----
def preprocess_logits_for_metrics(logits, labels):
    import torch
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits.argmax(dim=-1).detach().to("cpu", dtype=torch.int32)

def build_compute_metrics():
    def compute(eval_pred: "EvalPrediction"):
        preds = eval_pred.predictions      # argmax déjà fait (ints)
        labels = eval_pred.label_ids
        mask = (labels != -100) if (labels.ndim == 2 or labels.ndim == 1) else labels >= 0
        # Cas standard: labels shape [B], mask inutile
        if mask.shape != labels.shape:
            mask = labels >= 0
        total = mask.sum()
        if total == 0:
            return {"accuracy": 0.0}
        correct = (preds == labels) & mask
        acc = correct.sum() / total
        try:
            acc = float(acc)
        except Exception:
            acc = float(acc.item())
        return {"accuracy": acc}
    return compute


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None, metadata={"help": "ex: roberta-base, distilbert-base-uncased"})
    config_name: Optional[str] = field(default=None)
    tokenizer_name: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    use_fast_tokenizer: bool = field(default=True)
    model_revision: str = field(default="main")
    use_auth_token: bool = field(default=False)
    mode: Optional[str] = field(default="ilm", metadata={"help": "ensemble | ilm | game"})
    nb_steps_heads_saving: int = field(default=0)
    nb_steps_model_saving: int = field(default=0)
    init_base: bool = field(default=False, metadata={"help": "Réinit encoder + têtes"})


@dataclass
class DataTrainingArguments:
    train_file: Optional[str] = field(default=None, metadata={"help": "Dossier envs/ (plusieurs *.txt TSV) ou fichier train_erm.txt"})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Fichier TSV val.txt"})
    overwrite_cache: bool = field(default=False)
    max_seq_length: Optional[int] = field(default=128)
    preprocessing_num_workers: Optional[int] = field(default=None)
    pad_to_max_length: bool = field(default=False)
    nb_steps: int = field(default=0, metadata={"help": "nb de steps d'entraînement (sinon déduit par epoch)"})
    num_labels: int = field(default=2)


@dataclass
class CustomTrainingArguments(TrainingArguments):
    # Par défaut, logs toutes les 100 steps si l'utilisateur ne précise pas
    logging_strategy: str = field(default="steps")
    logging_steps: int = field(default=100)
    head_updates_per_encoder_update: int = field(default=1, metadata={"help": "IRM-Games: #updates têtes / update encodeur"})
    freeze_phi: bool = field(default=False, metadata={"help": "IRM-Games: geler l'encodeur"})


def _load_tsv_dataset(path: str, split_name: str = "train"):
    return load_dataset(
        "csv",
        data_files={split_name: path},
        delimiter="\t",
        column_names=["text", "labels"],
        quoting=3,  # QUOTE_NONE
    )


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Defaults intelligents si l'utilisateur n'a rien fixé
    cli = " ".join(sys.argv)
    if "--logging_steps" not in cli:
        training_args.logging_steps = 100
    if "--logging_strategy" not in cli:
        training_args.logging_strategy = "steps"

    if not model_args.model_name_or_path and not model_args.config_name:
        raise ValueError("Fournir --model_name_or_path (ex: 'distilbert-base-uncased' ou 'roberta-base').")

    # Logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    set_seed(training_args.seed)

    # --- Charger datasets (TSV text\tlabel)
    irm_datasets: Dict[str, "datasets.DatasetDict"] = {}
    train_input = data_args.train_file
    if train_input is None:
        raise ValueError("--train_file requis (fichier unique ERM ou dossier envs/).")

    if os.path.isdir(train_input):
        for file in os.listdir(train_input):
            if file.endswith(".txt"):
                env_name = os.path.splitext(file)[0]
                irm_datasets[env_name] = _load_tsv_dataset(os.path.join(train_input, file), split_name="train")
    elif os.path.isfile(train_input):
        env_name = os.path.splitext(os.path.basename(train_input))[0]
        irm_datasets[env_name] = _load_tsv_dataset(train_input, split_name="train")
    else:
        raise ValueError(f"train_file introuvable: {train_input}")

    if data_args.validation_file is not None:
        irm_datasets["validation-file"] = _load_tsv_dataset(data_args.validation_file, split_name="validation")

    # --- Config & tokenizer
    config_kwargs = {"cache_dir": model_args.cache_dir, "revision": model_args.model_revision,
                     "use_auth_token": True if model_args.use_auth_token else None}
    if model_args.config_name:
        base_config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    else:
        base_config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    base_config.num_labels = int(data_args.num_labels)

    tok_kwargs = {"cache_dir": model_args.cache_dir, "use_fast": model_args.use_fast_tokenizer,
                  "revision": model_args.model_revision, "use_auth_token": True if model_args.use_auth_token else None}
    tokenizer = (AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tok_kwargs)
                 if model_args.tokenizer_name else
                 AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tok_kwargs))

    # base encoder pour copier les poids
    base_encoder = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        config=base_config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Env names (hors validation)
    envs = [k for k in irm_datasets.keys() if "validation" not in k]

    # Construire modèle invariant (selon nom du modèle)
    if "distil" in (model_args.model_name_or_path or "").lower():
        inv_config = InvariantDistilBertConfig(envs=envs, num_labels=base_config.num_labels, **base_config.to_dict())
        irm_model = InvariantDistilBertForSequenceClassification(inv_config, base_encoder)
    else:
        inv_config = InvariantRobertaConfig(envs=envs, num_labels=base_config.num_labels, **base_config.to_dict())
        irm_model = InvariantRobertaForSequenceClassification(inv_config, base_encoder)

    if model_args.init_base:
        # réinit complète si demandé
        irm_model.encoder.init_weights()
        for k in irm_model.lm_heads:
            irm_model.lm_heads[k].apply(type(irm_model.lm_heads[k]).__init__)

    # --- Tokenisation
    def tok_map_fn(examples):
        # labels string->int si besoin
        lbls = examples["labels"]
        if isinstance(lbls, list):
            lbls = [int(x) for x in lbls]
        else:
            lbls = int(lbls)
        out = tokenizer(
            examples["text"],
            truncation=True,
            max_length=data_args.max_seq_length or tokenizer.model_max_length,
        )
        out["labels"] = lbls
        return out

    irm_tokenized = {}
    remove_cols = ["text", "labels"]
    for env_name, ds in irm_datasets.items():
        if "train" in ds:
            irm_tokenized[env_name] = ds.map(
                tok_map_fn,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=remove_cols,
                load_from_cache_file=not data_args.overwrite_cache,
            )
        elif "validation" in ds:
            irm_tokenized[env_name] = ds.map(
                tok_map_fn,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=remove_cols,
                load_from_cache_file=not data_args.overwrite_cache,
            )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=None)

    train_tokenized = {k: v for k, v in irm_tokenized.items() if "validation-file" not in k}
    eval_tokenized = None
    if "validation-file" in irm_tokenized and "validation" in irm_tokenized["validation-file"]:
        eval_tokenized = irm_tokenized["validation-file"]["validation"]

    # --- Trainer
    trainer = InvariantTrainer(
        model=irm_model,
        args=training_args,
        eval_dataset=eval_tokenized if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics() if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    # --- Training
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    nb_steps = int(data_args.nb_steps) if data_args.nb_steps else None

    if training_args.do_train:
        ckpt = last_checkpoint
        if model_args.mode == "ensemble":
            train_result = trainer.ensemble_train(
                training_set=train_tokenized,
                nb_steps=nb_steps,
                nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                nb_steps_model_saving=model_args.nb_steps_model_saving,
                resume_from_checkpoint=ckpt,
            )
        elif model_args.mode == "ilm":
            train_result = trainer.invariant_train(
                training_set=train_tokenized,
                nb_steps=nb_steps,
                nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                nb_steps_model_saving=model_args.nb_steps_model_saving,
                resume_from_checkpoint=ckpt,
            )
        elif model_args.mode == "game":
            train_result = trainer.invariant_train_games(
                training_set=train_tokenized,
                nb_steps=nb_steps,
                nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                nb_steps_model_saving=model_args.nb_steps_model_saving,
                resume_from_checkpoint=ckpt,
            )
        else:
            raise ValueError(f"Unknown --mode {model_args.mode}")

        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

        if trainer.is_world_process_zero():
            with open(os.path.join(training_args.output_dir, "train_results.txt"), "w") as w:
                for k, v in sorted(train_result["metrics"].items()):
                    w.write(f"{k} = {v}\n")

    # --- Evaluation
    results = {}
    if training_args.do_eval and eval_tokenized is not None:
        logger.info("*** Evaluate ***")
        eval_output = trainer.evaluate()
        results.update(eval_output)

        if trainer.is_world_process_zero():
            model_name = os.path.basename(os.path.normpath(training_args.output_dir))
            with open(os.path.join(RESULTS_ROOT, f"{model_name}_eval_results.txt"), "w") as w:
                for k, v in sorted(results.items()):
                    w.write(f"{k} = {v}\n")

    return results


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
