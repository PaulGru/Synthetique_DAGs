#!/usr/bin/env python
# coding=utf-8
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import sys
from transformers.trainer_utils import IntervalStrategy

from datasets import load_dataset

from invariant_trainer import InvariantTrainer
from invariant_roberta import InvariantRobertaForMaskedLM, InvariantRobertaConfig
from invariant_distilbert import InvariantDistilBertForMaskedLM, InvariantDistilBertConfig

import transformers
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

# --- Guard import so HfArgumentParser can resolve forward-ref types on some versions
try:
    # Ensure TrainingArguments' forward-ref type hints like `ParallelismConfig` resolve
    from transformers.training_args import ParallelismConfig  # noqa: F401
except Exception:
    class ParallelismConfig:  # type: ignore
        pass

RESULTS_ROOT = os.path.join(os.getcwd(), "eval_results_perplexity")
os.makedirs(RESULTS_ROOT, exist_ok=True)
logger = logging.getLogger(__name__)

# Réduit drastiquement la mémoire: on ne garde que l'argmax des logits (indices vocab)
def preprocess_logits_for_metrics(logits, labels):
    import torch
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    # on renvoie directement des int CPU (bien plus compacts, et hors GPU)
    return logits.argmax(dim=-1).detach().to("cpu", dtype=torch.int32)

# ---- compute_metrics: MLM accuracy on masked positions ----
def build_compute_metrics():
    import numpy as np
    from transformers.trainer_utils import EvalPrediction
    def compute(eval_pred: "EvalPrediction"):
        preds = eval_pred.predictions      # déjà argmax (int CPU)
        labels = eval_pred.label_ids
        mask = labels != -100
        total = mask.sum()
        if total == 0:
            return {"mlm_accuracy": 0.0}
        correct = (preds == labels) & mask
        acc = correct.sum() / total
        try:
            acc = float(acc)
        except Exception:
            acc = float(acc.item())
        return {"mlm_accuracy": acc}
    return compute



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization (e.g., roberta-base, distilbert-base-uncased)."
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use the fast tokenizer (backed by tokenizers) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `huggingface-cli login` (necessary for private models)."
        },
    )
    init_head: Optional[bool] = field(
        default=False,
        metadata={"help": "Re-initialize the language modeling heads to random weights before training"},
    )
    init_base: Optional[bool] = field(
        default=False,
        metadata={"help": "Re-initialize the base language model (and thus the language modeling heads) before training"},
    )
    mode: Optional[str] = field(
        default="ilm",
        metadata={
            "help": "Training mode: 'ensemble', 'ilm' (per-env sequential), or 'game' (IRM Games dynamics)."
        },
    )
    nb_steps_heads_saving: Optional[int] = field(
        default=0,
        metadata={
            "help": "Number of training steps between saving the head weights (0 disables periodic head saving)."
        },
    )
    nb_steps_model_saving: Optional[int] = field(
        default=0,
        metadata={
            "help": "Number of training steps between saving the full model (0 disables periodic model saving)."
        },
    )
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "Lower-case during tokenization."},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "Path to a folder with *.txt envs or a single .txt (ERM)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "Optional validation text file to evaluate the perplexity on."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    nb_steps: Optional[int] = field(
        default=0,
        metadata={"help": "Number of training steps. If 0, we'll compute from epochs + dataset size."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")


@dataclass
class CustomTrainingArguments(TrainingArguments):
    head_updates_per_encoder_update: Optional[int] = field(
        default=1, metadata={"help": "Number of head updates per encoder update (IRM Games)"}
    )
    freeze_phi: bool = field(default=False, metadata={"help": "Freeze encoder φ (no updates at all)"})


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, CustomTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not model_args.model_name_or_path and not model_args.config_name:
        raise ValueError(
            "Please provide --model_name_or_path (recommended, e.g. 'roberta-base' or 'distilbert-base-uncased'). "
            "Initializing entirely from scratch is not supported in this updated script."
        )
    
    cli = " ".join(sys.argv)
    if "--logging_steps" not in cli:
        training_args.logging_steps = 100  # <-- ton défaut
    if "--logging_strategy" not in cli:
        training_args.logging_strategy = IntervalStrategy.STEPS

    nb_steps = data_args.nb_steps

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Prepare IRM datasets: accept a folder with *.txt files (multiple envs) OR a single .txt (ERM)
    irm_input = data_args.train_file
    irm_datasets = {}
    if irm_input is not None:
        if os.path.isdir(irm_input):
            for file in os.listdir(irm_input):
                if file.endswith(".txt"):
                    env_name = os.path.splitext(file)[0]
                    data_files = {"train": os.path.join(irm_input, file)}
                    datasets = load_dataset("text", data_files=data_files)
                    irm_datasets[env_name] = datasets
        elif os.path.isfile(irm_input):
            env_name = os.path.splitext(os.path.basename(irm_input))[0]
            datasets = load_dataset("text", data_files={"train": irm_input})
            irm_datasets[env_name] = datasets
        else:
            raise ValueError(f"--train_file path not found: {irm_input}")

    if data_args.validation_file is not None:
        data_files = {"validation": data_args.validation_file}
        eval_datasets = load_dataset("text", data_files=data_files)
        irm_datasets["validation-file"] = eval_datasets

    # Load config & tokenizer
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        raise ValueError("A config or a model_name_or_path must be provided.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError("You must provide a tokenizer via --tokenizer_name or --model_name_or_path.")

    if model_args.model_name_or_path:
        base_model = AutoModelForMaskedLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in str(model_args.model_name_or_path)),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        base_model = AutoModelForMaskedLM.from_config(config)

    envs = [k for k in irm_datasets.keys() if "validation" not in k]

    # Wrap with invariant heads
    if not hasattr(config, "envs"):
        if "distil" in (model_args.model_name_or_path or "").lower():
            inv_config = InvariantDistilBertConfig(envs=envs, **config.to_dict())
            irm_model = InvariantDistilBertForMaskedLM(inv_config, base_model)
        else:
            inv_config = InvariantRobertaConfig(envs=envs, **config.to_dict())
            irm_model = InvariantRobertaForMaskedLM(inv_config, base_model)
    else:
        irm_model = base_model

    irm_model.resize_token_embeddings(len(tokenizer))

    if model_args.init_head:
        irm_model.init_head()
    if model_args.init_base:
        irm_model.init_base()

    # Tokenization & chunking
    irm_tokenized_datasets = {}
    for env_name, datasets in irm_datasets.items():
        if training_args.do_train and "validation" not in env_name:
            column_names = datasets["train"].column_names
        elif training_args.do_eval and "validation" in env_name:
            column_names = datasets["validation"].column_names
        else:
            continue

        text_column_name = "text" if "text" in column_names else column_names[0]

        if data_args.max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
            if max_seq_length > 1024:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that by passing --max_seq_length xxx."
                )
                max_seq_length = 1024
        else:
            if data_args.max_seq_length > tokenizer.model_max_length:
                logger.warning(
                    f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        if data_args.line_by_line:
            padding = "max_length" if data_args.pad_to_max_length else False

            def tokenize_function(examples):
                examples["text"] = [line for line in examples["text"] if len(line) > 0 and not line.isspace()]
                return tokenizer(
                    examples["text"],
                    padding=padding,
                    truncation=True,
                    max_length=max_seq_length,
                    return_special_tokens_mask=True,
                )

            tokenized = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
            )
            irm_tokenized_datasets[env_name] = tokenized
        else:
            def tokenize_function(examples):
                return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

            tokenized = datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )

            def group_texts(examples):
                concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                total_length = (total_length // max_seq_length) * max_seq_length
                result = {
                    k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                    for k, t in concatenated_examples.items()
                }
                return result

            tokenized = tokenized.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
            )
            irm_tokenized_datasets[env_name] = tokenized

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=data_args.mlm_probability)

    train_tokenized = {k: v for k, v in irm_tokenized_datasets.items() if "validation-file" not in k}
    eval_tokenized = None
    if "validation-file" in irm_tokenized_datasets and "validation" in irm_tokenized_datasets["validation-file"]:
        eval_tokenized = irm_tokenized_datasets["validation-file"]["validation"]

    trainer = InvariantTrainer(
        model=irm_model,
        args=training_args,
        eval_dataset=eval_tokenized if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics() if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None

        if model_args.mode == "ensemble":
            logger.info("TRAINING WITH ENSEMBLE -- NOT FOLLOWING IRM-GAMES DYNAMIC")
            train_result = trainer.ensemble_train(
                training_set=train_tokenized,
                nb_steps=nb_steps,
                nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                nb_steps_model_saving=model_args.nb_steps_model_saving,
                resume_from_checkpoint=checkpoint,
            )

        elif model_args.mode == "ilm":
            train_result = trainer.invariant_train(
                training_set=train_tokenized,
                nb_steps=nb_steps,
                nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                nb_steps_model_saving=model_args.nb_steps_model_saving,
                resume_from_checkpoint=checkpoint,
            )

        elif model_args.mode == "game":
            train_result = trainer.invariant_train_games(
                training_set=train_tokenized,
                nb_steps=nb_steps,
                nb_steps_heads_saving=model_args.nb_steps_heads_saving,
                nb_steps_model_saving=model_args.nb_steps_model_saving,
                resume_from_checkpoint=checkpoint,
            )
        else:
            raise ValueError(f"Unknown --mode {model_args.mode} (expected 'ensemble' | 'ilm' | 'game').")

        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result["metrics"].items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # Evaluation
    results = {}
    if training_args.do_eval and eval_tokenized is not None:
        logger.info("*** Evaluate ***")
        eval_output = trainer.evaluate()
        eval_loss = eval_output["eval_loss"]
        results["eval_loss"] = eval_loss
        results["perplexity"] = math.exp(eval_loss)

        if trainer.is_world_process_zero():
            model_name = os.path.basename(os.path.normpath(training_args.output_dir))
            output_txt = os.path.join(RESULTS_ROOT, f"{model_name}_eval_results.txt")
            with open(output_txt, "w") as writer:
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

    return results


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
