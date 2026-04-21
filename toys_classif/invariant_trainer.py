#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import math
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import transformers
from transformers.optimization import Adafactor, get_scheduler
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import IntervalStrategy
from transformers.utils import logging
from torch.optim import AdamW

from tqdm import tqdm
import numpy as np
import pandas as pd

logger = logging.get_logger(__name__)


def _move_to_device(obj, device):
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(v, device) for v in obj)
    if torch.is_tensor(obj):
        return obj.to(device)
    return obj


class InvariantTrainer(transformers.Trainer):
    def create_optimizer_and_scheduler(self, model, num_training_steps: int):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False, "lr": self.args.learning_rate}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
                "lr": self.args.learning_rate,
            }
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )
        return optimizer, lr_scheduler

    def remove_dataparallel_wrapper(self):
        if hasattr(self.model, "module"):
            self.model = self.model.module

    # ------------------------------ Training Loops ------------------------------
    def invariant_train(
        self,
        training_set: Dict[str, Dict[str, torch.utils.data.Dataset]],
        nb_steps: Optional[int] = None,
        nb_steps_heads_saving: Optional[int] = 0,
        resume_from_checkpoint: Optional[str] = None,
        num_train_epochs: Optional[int] = 1,
        nb_steps_model_saving: Optional[int] = 0,
        **kwargs,
    ):
        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")

        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None at the same time")
        if len(kwargs) > 0:
            raise TypeError(f"train() received unexpected kwargs: {', '.join(list(kwargs.keys()))}.")

        min_train_set_size = min(len(data["train"]) for _, data in training_set.items())
        per_device_bs = self.args.per_device_train_batch_size
        grad_accum = max(1, self.args.gradient_accumulation_steps)

        if nb_steps is not None and nb_steps > 0:
            max_steps = nb_steps
            num_update_steps_per_epoch = max(1, math.floor(min_train_set_size / (grad_accum * per_device_bs)))
            num_train_epochs_calc = max(1, math.floor(max_steps / max(1, num_update_steps_per_epoch)))
        else:
            num_update_steps_per_epoch = max(1, math.floor(min_train_set_size / (grad_accum * per_device_bs)))
            num_train_epochs_calc = num_train_epochs or 1
            max_steps = num_update_steps_per_epoch * num_train_epochs_calc

        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(env_name, data_features["train"])
            optimizer, lr_scheduler = self.create_optimizer_and_scheduler(self.model.lm_heads[env_name], num_training_steps=max_steps)
            optimizers[env_name] = optimizer
            lr_schedulers[env_name] = lr_scheduler

        enc_optim, enc_sched = self.create_optimizer_and_scheduler(self.model.encoder, num_training_steps=max_steps)

        self.state = TrainerState()

        device = self.args.device
        self.model.to(device)
        self.model.train()

        total_train_batch_size = per_device_bs * grad_accum
        num_examples = total_train_batch_size * max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  num_update_steps_per_epoch = {num_update_steps_per_epoch}")
        logger.info(f"  Total optimization steps = {max_steps}")

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        total_trained_steps = 0

        log_every = max(1, self.args.logging_steps)
        os.makedirs(self.args.output_dir, exist_ok=True)
        csv_total_loss_path = os.path.join(self.args.output_dir, "train_total_loss.csv")
        csv_heads_loss_path = os.path.join(self.args.output_dir, "train_env_losses.csv")
        train_loss_log, heads_loss_log = [], []

        use_fp16 = bool(self.args.fp16) and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

        last_total_loss_step = 0.0

        for epoch in range(num_train_epochs_calc):
            iter_loaders = {env: iter(loader) for env, loader in dataloaders.items()}

            for _ in tqdm(range(num_update_steps_per_epoch)):
                if total_trained_steps >= max_steps:
                    break

                env_step_losses = {"step": total_trained_steps}
                total_loss_step = 0.0

                for env_name in training_set.keys():
                    enc_optim.zero_grad(set_to_none=True)
                    optimizers[env_name].zero_grad(set_to_none=True)

                    try:
                        inputs = next(iter_loaders[env_name])
                    except StopIteration:
                        iter_loaders[env_name] = iter(dataloaders[env_name])
                        inputs = next(iter_loaders[env_name])

                    inputs = _move_to_device(inputs, device)

                    for e_n in self.model.lm_heads.keys():
                        self.model.lm_heads[e_n].requires_grad_(False)
                    self.model.lm_heads[env_name].requires_grad_(True)
                    self.model.encoder.requires_grad_(True)

                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        outputs = self.model(**inputs)
                        loss = outputs.loss

                    total_loss_step += float(loss.detach().item())
                    env_step_losses[env_name] = float(loss.detach().item())

                    scaler.scale(loss).backward()

                    # Clip grads
                    scaler.unscale_(enc_optim)
                    scaler.unscale_(optimizers[env_name])
                    torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(self.model.lm_heads[env_name].parameters(), self.args.max_grad_norm)

                    scaler.step(enc_optim)
                    scaler.step(optimizers[env_name])
                    scaler.update()

                    enc_sched.step()
                    lr_schedulers[env_name].step()

                    total_trained_steps += 1

                    if saving_heads and total_trained_steps % nb_steps_heads_saving == 0:
                        self.save_heads(total_trained_steps)
                    if saving_intermediary_models and total_trained_steps % nb_steps_model_saving == 0:
                        self.save_intermediary_model(total_trained_steps)

                avg_loss_step = total_loss_step / max(1, len(training_set))
                last_total_loss_step = avg_loss_step
                train_loss_log.append({"step": total_trained_steps, "loss": avg_loss_step})
                heads_loss_log.append(env_step_losses)

                if total_trained_steps % log_every == 0:
                    pd.DataFrame(train_loss_log).to_csv(csv_total_loss_path, index=False)
                    pd.DataFrame(heads_loss_log).to_csv(csv_heads_loss_path, index=False)

                # ---- Logs & Eval (W&B) ----
                self.state.global_step = int(total_trained_steps)
                if (
                    self.args.logging_strategy == IntervalStrategy.STEPS
                    and (total_trained_steps % max(1, self.args.logging_steps)) == 0
                ):
                    self.log({"train/loss": avg_loss_step})

                if (
                    self.eval_dataset is not None
                    and self.args.evaluation_strategy == IntervalStrategy.STEPS
                    and (total_trained_steps % max(1, self.args.eval_steps)) == 0
                ):
                    metrics = self.evaluate()
                    self.log(metrics)

        if train_loss_log:
            pd.DataFrame(train_loss_log).to_csv(csv_total_loss_path, index=False)
        if heads_loss_log:
            pd.DataFrame(heads_loss_log).to_csv(csv_heads_loss_path, index=False)

        return {
            "metrics": {
                "final_loss": float(last_total_loss_step),
                "nb_steps": int(max_steps),
                "global_step": int(total_trained_steps),
            }
        }

    def ensemble_train(
        self,
        training_set,
        nb_steps: Optional[int] = None,
        nb_steps_heads_saving: Optional[int] = 0,
        resume_from_checkpoint: Optional[str] = None,
        num_train_epochs: Optional[int] = 1,
        nb_steps_model_saving: Optional[int] = 0,
        **kwargs,
    ):
        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")

        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None at the same time")
        if len(kwargs) > 0:
            raise TypeError(f"train() received unexpected kwargs: {', '.join(list(kwargs.keys()))}.")

        min_train_set_size = min(len(data["train"]) for _, data in training_set.items())
        per_device_bs = self.args.per_device_train_batch_size
        grad_accum = max(1, self.args.gradient_accumulation_steps)

        if nb_steps is not None and nb_steps > 0:
            max_steps = nb_steps
            num_update_steps_per_epoch = max(1, math.floor(min_train_set_size / (grad_accum * per_device_bs)))
            num_train_epochs_calc = max(1, math.floor(max_steps / max(1, num_update_steps_per_epoch)))
        else:
            num_update_steps_per_epoch = max(1, math.floor(min_train_set_size / (grad_accum * per_device_bs)))
            num_train_epochs_calc = num_train_epochs or 1
            max_steps = num_update_steps_per_epoch * num_train_epochs_calc

        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(env_name, data_features["train"])
            optimizer, lr_scheduler = self.create_optimizer_and_scheduler(self.model.lm_heads[env_name], num_training_steps=max_steps)
            optimizers[env_name] = optimizer
            lr_schedulers[env_name] = lr_scheduler

        enc_optim, enc_sched = self.create_optimizer_and_scheduler(self.model.encoder, num_training_steps=max_steps)

        device = self.args.device
        self.model.to(device)
        self.model.train()

        total_trained_steps = 0
        use_fp16 = bool(self.args.fp16) and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

        for epoch in range(max(1, math.ceil(max_steps / max(1, min_train_set_size // max(1, (per_device_bs * grad_accum)))))):
            iter_loaders = {env: iter(loader) for env, loader in dataloaders.items()}
            for _ in tqdm(range(max(1, min_train_set_size // max(1, (per_device_bs * grad_accum))))):
                if total_trained_steps >= max_steps:
                    break

                for env_name in training_set.keys():
                    enc_optim.zero_grad(set_to_none=True)
                    for e_n in training_set.keys():
                        optimizers[e_n].zero_grad(set_to_none=True)

                    try:
                        batch = next(iter_loaders[env_name])
                    except StopIteration:
                        iter_loaders[env_name] = iter(dataloaders[env_name])
                        batch = next(iter_loaders[env_name])

                    batch = _move_to_device(batch, device)

                    with torch.cuda.amp.autocast(enabled=use_fp16):
                        outputs = self.model(**batch)
                        loss = outputs.loss

                    scaler.scale(loss).backward()
                    scaler.unscale_(enc_optim)
                    for e_n in training_set.keys():
                        scaler.unscale_(optimizers[e_n])

                    torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.args.max_grad_norm)
                    for e_n in training_set.keys():
                        torch.nn.utils.clip_grad_norm_(self.model.lm_heads[e_n].parameters(), self.args.max_grad_norm)

                    scaler.step(enc_optim)
                    for e_n in training_set.keys():
                        scaler.step(optimizers[e_n])
                    scaler.update()

                    enc_sched.step()
                    for e_n in training_set.keys():
                        lr_schedulers[e_n].step()

                    total_trained_steps += 1

                    # Logs/Eval périodiques
                    self.state.global_step = int(total_trained_steps)
                    if (
                        self.args.logging_strategy == IntervalStrategy.STEPS
                        and (total_trained_steps % max(1, self.args.logging_steps)) == 0
                    ):
                        self.log({"train/loss": float(loss.detach().item())})

                    if (
                        self.eval_dataset is not None
                        and self.args.evaluation_strategy == IntervalStrategy.STEPS
                        and (total_trained_steps % max(1, self.args.eval_steps)) == 0
                    ):
                        metrics = self.evaluate()
                        self.log(metrics)

        return {
            "metrics": {
                "final_loss": float(loss.detach().item()),
                "nb_steps": int(max_steps),
                "global_step": int(total_trained_steps),
            }
        }

    def invariant_train_games(
        self,
        training_set,
        nb_steps: Optional[int] = None,
        nb_steps_heads_saving: Optional[int] = 0,
        resume_from_checkpoint: Optional[str] = None,
        num_train_epochs: Optional[int] = 1,
        nb_steps_model_saving: Optional[int] = 0,
        **kwargs,
    ):
        K = getattr(self.args, "head_updates_per_encoder_update", 1)
        freeze_phi = getattr(self.args, "freeze_phi", False)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")  # noqa: F841

        if nb_steps is None and num_train_epochs is None:
            raise ValueError("Both nb_steps and num_train_epochs can't be None at the same time")
        if len(kwargs) > 0:
            raise TypeError(f"train() received unexpected kwargs: {', '.join(list(kwargs.keys()))}.")

        min_train_set_size = min(len(data["train"]) for _, data in training_set.items())
        per_device_bs = self.args.per_device_train_batch_size
        grad_accum = max(1, self.args.gradient_accumulation_steps)

        if nb_steps is not None and nb_steps > 0:
            max_steps = nb_steps
            num_update_steps_per_epoch = max(1, math.floor(min_train_set_size / (grad_accum * per_device_bs)))
            num_train_epochs_calc = max(1, math.floor(max_steps / max(1, num_update_steps_per_epoch)))
        else:
            num_update_steps_per_epoch = max(1, math.floor(min_train_set_size / (grad_accum * per_device_bs)))
            num_train_epochs_calc = num_train_epochs or 1
            max_steps = num_update_steps_per_epoch * num_train_epochs_calc

        E = len(training_set)
        head_total_steps = math.ceil(max_steps / E)
        enc_total_steps = math.ceil(max_steps / (K * E))

        dataloaders, optimizers, lr_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(env_name, data_features["train"])
            optimizer, lr_scheduler = self.create_optimizer_and_scheduler(self.model.lm_heads[env_name], num_training_steps=head_total_steps)
            optimizers[env_name] = optimizer
            lr_schedulers[env_name] = lr_scheduler

        if not freeze_phi:
            enc_optim, enc_sched = self.create_optimizer_and_scheduler(self.model.encoder, num_training_steps=enc_total_steps)
        else:
            enc_optim = enc_sched = None
            self.model.encoder.requires_grad_(False)

        device = self.args.device
        self.model.to(device)
        self.model.train()

        logger.info("***** Running training (IRM Games) *****")
        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        total_trained_steps = 0

        log_every = max(1, self.args.logging_steps)
        os.makedirs(self.args.output_dir, exist_ok=True)
        csv_total_loss_path = os.path.join(self.args.output_dir, "train_total_loss.csv")
        csv_heads_loss_path = os.path.join(self.args.output_dir, "train_env_losses.csv")
        last_heads_mean_loss = None
        total_loss_val = None

        train_loss_log, heads_loss_log = [], []

        use_fp16 = bool(self.args.fp16) and torch.cuda.is_available()
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

        env_step_losses = {}

        for epoch in range(num_train_epochs_calc):
            iter_loaders = {env: iter(loader) for env, loader in dataloaders.items()}

            for _ in tqdm(range(num_update_steps_per_epoch)):
                if total_trained_steps >= max_steps:
                    break

                # === Phase 1: update des têtes (K tours) ===
                env_step_losses = {"step": total_trained_steps}

                for _k in range(K):
                    for env_name in training_set.keys():
                        # On n'autorise le gradient que pour la tête courante
                        for e_n in training_set.keys():
                            self.model.lm_heads[e_n].requires_grad_(False)
                        self.model.lm_heads[env_name].requires_grad_(True)

                        try:
                            inputs = next(iter_loaders[env_name])
                        except StopIteration:
                            iter_loaders[env_name] = iter(dataloaders[env_name])
                            inputs = next(iter_loaders[env_name])

                        inputs = _move_to_device(inputs, device)

                        optimizers[env_name].zero_grad(set_to_none=True)

                        self.model.encoder.requires_grad_(False)
                        for e_n in self.model.lm_heads.keys():
                            self.model.lm_heads[e_n].requires_grad_(False)
                        self.model.lm_heads[env_name].requires_grad_(True)

                        with torch.cuda.amp.autocast(enabled=use_fp16):
                            outputs = self.model(**inputs)
                            loss = outputs.loss

                        env_step_losses[env_name] = float(loss.detach().item())

                        scaler.scale(loss).backward()

                        scaler.unscale_(optimizers[env_name])
                        torch.nn.utils.clip_grad_norm_(self.model.lm_heads[env_name].parameters(), self.args.max_grad_norm)

                        scaler.step(optimizers[env_name])
                        scaler.update()

                        lr_schedulers[env_name].step()

                        total_trained_steps += 1

                        if saving_heads and total_trained_steps % nb_steps_heads_saving == 0:
                            self.save_heads(total_trained_steps)
                        if saving_intermediary_models and total_trained_steps % nb_steps_model_saving == 0:
                            self.save_intermediary_model(total_trained_steps)

                    heads_loss_log.append(env_step_losses)
                    vals = [v for k, v in env_step_losses.items() if k != "step"]
                    if vals:
                        last_heads_mean_loss = float(np.mean(vals))

                # ---- Logs/Eval en fin de Phase 1 si encodeur figé ----
                if freeze_phi:
                    self.state.global_step = int(total_trained_steps)
                    if (
                        self.args.logging_strategy == IntervalStrategy.STEPS
                        and (total_trained_steps % max(1, self.args.logging_steps)) == 0
                    ):
                        val_to_log = last_heads_mean_loss if last_heads_mean_loss is not None else 0.0
                        self.log({"train/loss": float(val_to_log)})

                    if (
                        self.eval_dataset is not None
                        and self.args.evaluation_strategy == IntervalStrategy.STEPS
                        and (total_trained_steps % max(1, self.args.eval_steps)) == 0
                    ):
                        metrics = self.evaluate()
                        self.log(metrics)

                # === Phase 2: update encodeur partagé ===
                if not freeze_phi:
                    for env_name in training_set.keys():
                        self.model.lm_heads[env_name].requires_grad_(False)
                    self.model.encoder.requires_grad_(True)
                    for e_n in self.model.lm_heads.keys():
                        self.model.lm_heads[e_n].requires_grad_(False)

                    enc_optim.zero_grad(set_to_none=True)
                    total_loss = 0.0
                    for env_name in training_set.keys():
                        try:
                            inputs = next(iter_loaders[env_name])
                        except StopIteration:
                            iter_loaders[env_name] = iter(dataloaders[env_name])
                            inputs = next(iter_loaders[env_name])

                        inputs = _move_to_device(inputs, device)

                        with torch.cuda.amp.autocast(enabled=use_fp16):
                            outputs = self.model(**inputs)
                            loss = outputs.loss
                        total_loss = total_loss + loss

                    scaler.scale(total_loss).backward()
                    scaler.unscale_(enc_optim)
                    torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.args.max_grad_norm)

                    scaler.step(enc_optim)
                    scaler.update()
                    enc_sched.step()

                    total_loss_val = float(total_loss.detach().item())
                    train_loss_log.append({"step": total_trained_steps, "loss": total_loss_val / max(1, len(training_set))})

                    if total_trained_steps % log_every == 0:
                        pd.DataFrame(train_loss_log).to_csv(csv_total_loss_path, index=False)
                        pd.DataFrame(heads_loss_log).to_csv(csv_heads_loss_path, index=False)

                    self.state.global_step = int(total_trained_steps)
                    if (
                        self.args.logging_strategy == IntervalStrategy.STEPS
                        and (total_trained_steps % max(1, self.args.logging_steps)) == 0
                    ):
                        val_to_log = (total_loss_val / max(1, len(training_set))) if total_loss_val is not None else (
                            float(np.mean([v for k, v in env_step_losses.items() if k != "step"])) if env_step_losses else 0.0
                        )
                        self.log({"train/loss": float(val_to_log)})

                    if (
                        self.eval_dataset is not None
                        and self.args.evaluation_strategy == IntervalStrategy.STEPS
                        and (total_trained_steps % max(1, self.args.eval_steps)) == 0
                    ):
                        metrics = self.evaluate()
                        self.log(metrics)

        if train_loss_log:
            pd.DataFrame(train_loss_log).to_csv(csv_total_loss_path, index=False)
        if heads_loss_log:
            pd.DataFrame(heads_loss_log).to_csv(csv_heads_loss_path, index=False)

        metrics = {"nb_steps": int(max_steps), "global_step": int(total_trained_steps)}
        if total_loss_val is not None:
            metrics["eval_loss"] = float(total_loss_val / max(1, len(training_set)))
        elif last_heads_mean_loss is not None:
            metrics["eval_loss"] = float(last_heads_mean_loss)
        return {"metrics": metrics}

    # ------------------------------ Utilities ------------------------------
    def save_intermediary_model(self, n_steps):
        fname = os.path.join(self.args.output_dir, f"model-{n_steps}")
        self.save_model(output_dir=fname)

    def save_heads(self, step_count):
        logger.info("saving-heads")
        out_dir = os.path.join(self.args.output_dir, "heads")
        os.makedirs(out_dir, exist_ok=True)
        for env, head in self.model.lm_heads.items():
            # compat: MLM et CLF
            if hasattr(head, "vocab_projector"):              # DistilBERT MLM
                weight = head.vocab_projector.weight
            elif hasattr(head, "decoder") and hasattr(head.decoder, "weight"):  # Roberta MLM
                weight = head.decoder.weight
            elif hasattr(head, "classifier"):                 # DistilBERT CLF
                weight = head.classifier.weight
            elif hasattr(head, "out_proj"):                   # Roberta CLF
                weight = head.out_proj.weight
            else:
                continue
            np.save(os.path.join(out_dir, f"{env}-{step_count}.npy"), weight.data.cpu().numpy())

    def get_single_train_dataloader(self, env_name, train_dataset):
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        use_distributed = False
        try:
            import torch.distributed as dist
            use_distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        except Exception:
            use_distributed = False

        train_sampler = DistributedSampler(train_dataset) if use_distributed else RandomSampler(train_dataset)
        return DataLoader(
            train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
        )
