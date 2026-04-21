import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import transformers
from transformers.optimization import Adafactor, get_scheduler
from torch.optim import AdamW
from transformers.trainer_callback import TrainerState
from transformers.utils import logging

from tqdm import tqdm

import wandb
import math
import os
import numpy as np
from itertools import cycle
import random
from torch.amp import autocast, GradScaler
from typing import Optional

logger = logging.get_logger(__name__)

def compute_moving_average(values, window_size=10):
        if len(values) < window_size:
            return sum(values) / max(len(values), 1)
        return sum(values[-window_size:]) / window_size

class InvariantTrainer(transformers.Trainer):

    def create_optimizer_and_scheduler(self, model, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        optimizer, lr_scheduler = None, None
        # if self.optimizer is None:
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer_cls = Adafactor if self.args.adafactor else AdamW
        if self.args.adafactor:
            optimizer_cls = Adafactor
            optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
        else:
            optimizer_cls = AdamW
            optimizer_kwargs = {
                "betas": (self.args.adam_beta1, self.args.adam_beta2),
                "eps": self.args.adam_epsilon,
            }
        optimizer_kwargs["lr"] = self.args.learning_rate
        
        optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        lr_scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps,
        )

        return optimizer, lr_scheduler
    

    def invariant_train(
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
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )

        min_train_set_size = min([len(data["train"]) for _, data in training_set.items()])
        
        if nb_steps is not None:
            max_steps = nb_steps
            steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
            )
            num_train_epochs = max(1, math.floor(max_steps / steps_per_epoch))
        else:
            steps_per_epoch = math.floor(
                min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
            )
            max_steps = steps_per_epoch * num_train_epochs

        dataloaders, head_optimizers, head_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(data_features["train"])
            optimizer, head_scheduler = self.create_optimizer_and_scheduler(
                self.model.lm_heads[env_name],
                num_training_steps=max_steps
            )
            head_optimizers[env_name] = optimizer
            head_schedulers[env_name] = head_scheduler

        phi_optimizer, phi_scheduler = self.create_optimizer_and_scheduler(
            self.model.encoder,
            num_training_steps=max_steps
        )

        self.state = TrainerState()

        if self.args.n_gpu > 0:
            self.model.to(self.args.device)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        num_examples = total_train_batch_size * max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  steps_per_epoch = {steps_per_epoch}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        self.state.global_step = 0
        
        recent_losses = []

        self.scaler = GradScaler()
        
        self.use_amp = bool(self.args.fp16 and torch.cuda.is_available())
        print(f"use_amp = {self.use_amp}")

        iter_loaders = {env_name: cycle(dataloaders[env_name]) for env_name in training_set.keys()}
        for epoch in range(int(num_train_epochs)):
            logger.info(f" Epoch: {epoch}")

            # iter_loaders = {}
            # for env_name in training_set.keys():
            #     iter_loaders[env_name] = iter(dataloaders[env_name])

            for round_idx in tqdm(range(steps_per_epoch)):
                if self.state.global_step >= max_steps :
                    break

                for env_name in training_set.keys():
                    logger.info(f" Update on environement {env_name}")

                    phi_optimizer.zero_grad()
                    head_optimizers[env_name].zero_grad()

                    batch = next(iter_loaders[env_name])     
                    
                    self.model.train()
                    batch = self._prepare_inputs(batch)

                    if self.use_amp:
                        with autocast("cuda", enabled=self.use_amp):
                            loss = self.compute_loss(self.model, batch)
                    else:
                        loss = self.compute_loss(self.model, batch)

                    if self.args.n_gpu > 1:
                        loss = loss.mean()
                    
                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                    
                    if self.use_amp:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    loss = loss.detach()
                    
                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        if self.use_amp:
                            self.scaler.unscale_(phi_optimizer)
                            self.scaler.unscale_(head_optimizers[env_name])

                        if hasattr(phi_optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            phi_optimizer.clip_grad_norm(self.args.max_grad_norm)
                            head_optimizers[env_name].clip_grad_norm(self.args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.args.max_grad_norm,
                            )

                    if self.use_amp:
                        self.scaler.step(phi_optimizer)
                        self.scaler.step(head_optimizers[env_name])
                        self.scaler.update()
                    else:
                        phi_optimizer.step()
                        head_optimizers[env_name].step()

                    # Mise à jour des schedulers
                    phi_scheduler.step()
                    head_schedulers[env_name].step()

                    self.state.global_step += 1
                    
                    recent_losses.append(loss.item())
                    moving_avg_loss = compute_moving_average(recent_losses, window_size=20)
                    if self.is_world_process_zero() and (nb_steps_model_saving > 0) and self.state.global_step % nb_steps_model_saving == 0:
                        wandb.log({
                            "training/train_loss": loss.item(),
                            "training/train_loss_moving_avg": moving_avg_loss
                        }, step=self.state.global_step
                        )

                    if saving_heads and self.state.global_step % nb_steps_heads_saving == 0:
                        self.save_heads(self.state.global_step)
                    if saving_intermediary_models and self.state.global_step % nb_steps_model_saving == 0:
                        self.save_intermediary_model(self.state.global_step)
                    
                
        print("=== Entraînement du modèle terminé. Nombre total de rounds:", self.state.global_step/len(training_set.keys()))
 

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

        head_updates_per_encoder_update = getattr(self.args, "head_updates_per_encoder_update", 5)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )

        min_train_size = min(len(data["train"]) for _, data in training_set.items())
        
        steps_per_epoch = math.floor(
            min_train_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
        )
        if nb_steps is not None:
            num_train_epochs = max(1, math.floor(nb_steps / steps_per_epoch))
            max_steps = nb_steps
        else:
            max_steps = steps_per_epoch * num_train_epochs

        dataloaders, head_optimizers, head_schedulers = {}, {}, {}
        for env_name, data_features in training_set.items():
            dataloaders[env_name] = self.get_single_train_dataloader(data_features["train"])
            head_optimizers[env_name], head_schedulers[env_name] = self.create_optimizer_and_scheduler(
                self.model.lm_heads[env_name],
                num_training_steps=max_steps
            )
        
        phi_optimizer, phi_scheduler = self.create_optimizer_and_scheduler(
            self.model.encoder,
            num_training_steps=max_steps
        )

        self.state = TrainerState()

        # Move model to device
        if self.args.n_gpu > 0:
            self.model.to(self.args.device)
        if self.args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model)
        
        total_train_batch_size = self.args.train_batch_size * self.args.gradient_accumulation_steps
        num_examples = total_train_batch_size * max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  steps_per_epoch = {steps_per_epoch}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        saving_heads = bool(nb_steps_heads_saving > 0)
        saving_intermediary_models = bool(nb_steps_model_saving > 0)
        self.state.global_step = 0

        recent_head_losses = []
        recent_phi_losses = []

        self.scaler = GradScaler()
        
        iter_loaders = {env_name: cycle(dataloaders[env_name]) for env_name in training_set.keys()}
        for epoch in range(int(num_train_epochs)):
            logger.info(f" Epoch: {epoch}")

            for round_idx in tqdm(range(steps_per_epoch)):
                if self.state.global_step >= max_steps:
                    break
                
                for _ in range(head_updates_per_encoder_update):
                    self.model.encoder.requires_grad_(False)
                    for env_name in training_set.keys():
                        logger.info(f" Update on environement {env_name}")
                        self.model.lm_heads[env_name].requires_grad_(True)
                        head_optimizers[env_name].zero_grad()

                        batch = next(iter_loaders[env_name])
                        
                        self.model.train()
                        batch = self._prepare_inputs(batch)

                        if self.use_apex:
                            with autocast():
                                loss_head = self.compute_loss(self.model, batch)
                        else:
                            loss_head = self.compute_loss(self.model, batch)

                        if self.args.n_gpu > 1:
                            loss_head = loss_head.mean()
                    
                        if self.args.gradient_accumulation_steps > 1:
                            loss_head = loss_head / self.args.gradient_accumulation_steps
                        
                        if self.use_apex:
                            self.scaler.scale(loss_head).backward()
                        else:
                            loss_head.backward()

                        loss_head = loss_head.detach()
                        recent_head_losses.append(loss_head.item())

                        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                            if self.use_apex:
                                self.scaler.unscale_(head_optimizers[env_name])
                            
                            if hasattr(head_optimizers[env_name], "clip_grad_norm"):
                                head_optimizers[env_name].clip_grad_norm(self.args.max_grad_norm)
                            else:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.lm_heads[env_name].parameters(),
                                    self.args.max_grad_norm,
                                )
                        
                        if self.use_apex:
                            self.scaler.step(head_optimizers[env_name])
                        else:
                            head_optimizers[env_name].step()

                        head_schedulers[env_name].step()

                        self.state.global_step += 1

                        moving_avg_head = compute_moving_average(recent_head_losses, window_size=20)
                        if self.is_world_process_zero() and nb_steps_model_saving > 0 and self.state.global_step % nb_steps_model_saving == 0:
                            wandb.log({
                                "training/head_loss": loss_head.item(),
                                "training/head_loss_moving_avg": moving_avg_head,
                            }, step=self.state.global_step)

                        if nb_steps_heads_saving and self.state.global_step % nb_steps_heads_saving == 0:
                            self.save_heads(self.state.global_step)
                        if nb_steps_model_saving and self.state.global_step % nb_steps_model_saving == 0:
                            self.save_intermediary_model(self.state.global_step)


                # === Phase 2: update shared encoder ===
                self.model.encoder.requires_grad_(True)
                for env_name in training_set.keys():
                    self.model.lm_heads[env_name].requires_grad_(False)

                phi_optimizer.zero_grad()
                total_phi_loss = 0.0

                for env_name in training_set.keys():
                    batch = next(iter_loaders[env_name])
                    
                    self.model.train()
                    batch = self._prepare_inputs(batch)

                    if self.use_apex:
                        with autocast():
                            phi_loss = self.compute_loss(self.model, batch)
                    else:
                        phi_loss = self.compute_loss(self.model, batch)

                    if self.args.n_gpu > 1:
                        phi_loss = phi_loss.mean()
                
                    if self.args.gradient_accumulation_steps > 1:
                        phi_loss = phi_loss / self.args.gradient_accumulation_steps
                    
                    if self.use_apex:
                        self.scaler.scale(phi_loss).backward()
                    else:
                        phi_loss.backward()

                    phi_loss = phi_loss.detach()
                    total_phi_loss += phi_loss.item()

                    if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                        if self.use_apex:
                            self.scaler.unscale_(phi_optimizer)
                        
                        if hasattr(phi_optimizer, "clip_grad_norm"):
                            phi_optimizer.clip_grad_norm(self.args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.encoder.parameters(),
                                self.args.max_grad_norm,
                            )
                    
                    if self.use_apex:
                        self.scaler.step(phi_optimizer)
                        self.scaler.update()
                    else:
                        phi_optimizer.step()

                    phi_scheduler.step()

                    recent_phi_losses.append(total_phi_loss)
                    moving_avg_phi = compute_moving_average(recent_phi_losses, window_size=20)
                    if self.is_world_process_zero() and nb_steps_model_saving > 0 and self.state.global_step % nb_steps_model_saving == 0:
                        wandb.log({
                            "training/phi_loss": total_phi_loss,
                            "training/phi_loss_moving_avg": moving_avg_phi,
                        }, step=self.state.global_step)

                        
        if self.is_world_process_zero():
            print("=== Training complete. Total rounds:", self.state.global_step / len(training_set))


    def save_intermediary_model(self, n_steps):
        fname = os.path.join(self.args.output_dir, f"model-{n_steps}")
        self.save_model(output_dir=fname)

    def save_heads(self, step_count):
        # Ne sauvegarder que si ce processus est le principal
        if not self.is_world_process_zero():
            return
        
        print("saving-heads")
        if not os.path.exists("lm_heads"):
            os.makedirs("lm_heads")

        for env, lm_head in self.model.lm_heads.items():
            filepath = os.path.join("lm_heads", "{}-{}".format(env, step_count))
            
            if hasattr(lm_head, "dense"):
                np.save(filepath, lm_head.dense.weight.data.cpu().numpy())
            elif hasattr(lm_head, "decoder"):
                np.save(filepath, lm_head.decoder.weight.data.cpu().numpy())
            elif hasattr(lm_head, "vocab_projector"):
                np.save(filepath, lm_head.vocab_projector.weight.data.cpu().numpy())
            else:
                print(f"La tête pour l'environnement {env} ne possède pas d'attribut de sauvegarde connu.")


    def get_single_train_dataloader(self, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        train_sampler = (
            RandomSampler(train_dataset)
            if self.args.local_rank == -1
            else DistributedSampler(train_dataset)
        )

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator
        )