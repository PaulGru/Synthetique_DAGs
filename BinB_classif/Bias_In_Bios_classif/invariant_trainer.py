import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
import torch.distributed as dist

import transformers
from transformers.optimization import Adafactor, get_scheduler
from torch.optim import AdamW
from transformers.trainer_callback import TrainerState
from transformers.utils import logging

from tqdm import tqdm
import warnings
import wandb
import math
import os
import csv
import numpy as np
from itertools import cycle
import random
from torch.cuda.amp import autocast, GradScaler
from typing import Optional
from collections import defaultdict

logger = logging.get_logger(__name__)

def compute_moving_average(values, window_size=10):
        if len(values) < window_size:
            return sum(values) / max(len(values), 1)
        return sum(values[-window_size:]) / window_size


class InvariantTrainer(transformers.Trainer):

    # --- CSV LOGGING: small helpers ---
    def _csv_root(self):
        return os.path.join(self.args.output_dir, "csv_logs")

    def _append_row(self, filename, columns, row):
        """
        Append a row to OUTPUT_DIR/csv_logs/{filename}.
        Creates folder & header if needed. Only runs on world process zero.
        """
        if not self.is_world_process_zero():
            return
        os.makedirs(self._csv_root(), exist_ok=True)
        fp = os.path.join(self._csv_root(), filename)
        file_exists = os.path.exists(fp) and os.path.getsize(fp) > 0
        with open(fp, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            if not file_exists:
                writer.writeheader()
            # ensure every column is present
            safe_row = {c: row.get(c, "") for c in columns}
            writer.writerow(safe_row)

    def _log_training_to_csv(self, step, loss, moving_avg=None, round_idx=None):
        cols = ["step", "train_loss", "train_loss_moving_avg", "round"]
        row = {
            "step": int(step),
            "train_loss": float(loss),
        }
        if moving_avg is not None:
            row["train_loss_moving_avg"] = float(moving_avg)
        if round_idx is not None:
            row["round"] = int(round_idx)
        self._append_row("training.csv", cols, row)

    def _log_eval_to_csv(self, cleaned_metrics, step):
        # cleaned_metrics are the eval_* metrics with "eval_" stripped (see _log_eval_to_wandb)
        cols = [
            "step", "loss", "accuracy", "f1_macro",
            "corr_support_f1_pearson_r", "corr_support_f1_spearman_rho",
            "f1_head", "f1_tail", "gap_head_tail",
        ]
        row = {"step": int(step)}
        row.update({k: cleaned_metrics.get(k, "") for k in cols if k != "step"})
        self._append_row("eval.csv", cols, row)

    def _log_heads_to_csv(self, stats, step):
        # stats keys look like "heads/l2_mean", "heads/js_mean", ...
        cols = ["step", "l2_mean", "cosine_mean", "js_mean", "disagree_mean"]
        row = {"step": int(step)}
        for k in ["l2_mean", "cosine_mean", "js_mean", "disagree_mean"]:
            v = stats.get(f"heads/{k}")
            if v is not None:
                row[k] = float(v)
        self._append_row("heads.csv", cols, row)



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
    
    def _log_eval_to_wandb(self, metrics, step):
        if not (self.is_world_process_zero() and wandb.run):
            # même si WandB est OFF, on écrit le CSV si possible
            keep = {
                "eval_loss", "eval_accuracy", "eval_f1_macro",
                "eval_corr_support_f1_pearson_r", "eval_corr_support_f1_spearman_rho",
                "eval_f1_head", "eval_f1_tail", "eval_gap_head_tail",
            }
            cleaned = {k.replace("eval_", ""): v for k, v in metrics.items() if k in keep}
            if cleaned:
                self._log_eval_to_csv(cleaned, step)
            return
        
        keep = {
            "eval_loss", "eval_accuracy", "eval_f1_macro",
            "eval_corr_support_f1_pearson_r", "eval_corr_support_f1_spearman_rho",
            "eval_f1_head", "eval_f1_tail", "eval_gap_head_tail",
        }
        cleaned = {k.replace("eval_", ""): v for k, v in metrics.items() if k in keep}
        if cleaned:
            wandb.log({f"eval/{k}": v for k, v in cleaned.items()}, step=step)
            self._log_eval_to_csv(cleaned, step)
    

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
        
        steps_per_epoch = math.floor(
            min_train_set_size / (self.args.gradient_accumulation_steps * self.args.train_batch_size)
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
                self.model.classifier_heads[env_name],
                num_training_steps=max_steps
            )

        phi_optimizer, phi_scheduler = self.create_optimizer_and_scheduler(
            self.model.encoder,
            num_training_steps=max_steps
        )

        self.state = TrainerState()

        if self.args.n_gpu > 0:
            self.model.to(self.args.device)
        
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

        self.scaler = GradScaler(enabled=self.args.fp16)

        iter_loaders = {env_name: cycle(dataloaders[env_name]) for env_name in training_set.keys()}

        pbar = tqdm(total=max_steps, desc="Training progress")
        while self.state.global_step < max_steps:
            for env_name in training_set.keys():

                logger.info(f" Update on environement {env_name}")
                phi_optimizer.zero_grad()
                head_optimizers[env_name].zero_grad()

                batch = next(iter_loaders[env_name])     
                
                self.model.train()
                # gèle toutes les têtes sauf celle de l'env courant
                for name, head in self.model.classifier_heads.items():
                    head.requires_grad_(name == env_name)
                batch = self._prepare_inputs(batch)

                if self.args.fp16:
                    with autocast():
                        loss = self.compute_loss(self.model, batch)
                else:
                    loss = self.compute_loss(self.model, batch)
                
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps
                
                if self.args.fp16:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                loss = loss.detach()
                
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    if self.args.fp16:
                        self.scaler.unscale_(phi_optimizer)
                        self.scaler.unscale_(head_optimizers[env_name])

                    if hasattr(phi_optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        phi_optimizer.clip_grad_norm(self.args.max_grad_norm)
                        head_optimizers[env_name].clip_grad_norm(self.args.max_grad_norm)
                    
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.args.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(self.model.classifier_heads[env_name].parameters(), self.args.max_grad_norm)

                if self.args.fp16:
                    self.scaler.step(phi_optimizer)
                    self.scaler.step(head_optimizers[env_name])
                    self.scaler.update()
                else:
                    phi_optimizer.step()
                    head_optimizers[env_name].step()

                # Mise à jour des schedulers
                phi_scheduler.step()
                head_schedulers[env_name].step()

                recent_losses.append(loss.item())
                moving_avg_loss = compute_moving_average(recent_losses, window_size=20)

                if self.state.global_step % 100 == 0:
                    self._log_training_to_csv(
                        step=self.state.global_step,
                        loss=loss.item(),
                        moving_avg=moving_avg_loss
                    )

                if self.is_world_process_zero() and self.state.global_step % 100 == 0 and wandb.run:
                    wandb.log({
                        "training/train_loss": loss.item(),
                        "training/train_loss_moving_avg": moving_avg_loss
                    }, step=self.state.global_step)

                if saving_heads and self.state.global_step % nb_steps_heads_saving == 0:
                    self.save_heads(self.state.global_step)
                if saving_intermediary_models and self.state.global_step % nb_steps_model_saving == 0:
                    self.save_intermediary_model(self.state.global_step)

                if (
                    self.args.do_eval
                    and getattr(self.args, "evaluation_strategy", None) == "steps"
                    and getattr(self.args, "eval_steps", 0) > 0
                    and self.state.global_step % self.args.eval_steps == 0
                ):
                    # metrics = self.evaluate()  # évalue sur self.eval_dataset
                    # logger.info(f"*** Evaluation at step {self.state.global_step}: {metrics}")
                    # # log vers WandB si processus principal
                    # if self.is_world_process_zero() and wandb.run:
                    #     wandb.log(
                    #         {f"eval/{k}": v for k, v in metrics.items()},
                    #         step=self.state.global_step,
                    #     )
                    
                    metrics = self.evaluate()
                    logger.info(f"*** Evaluation at step {self.state.global_step}: {metrics}")
                    self._log_eval_to_wandb(metrics, step=self.state.global_step)
                    self._log_heads_weight_divergence(step=self.state.global_step)
                    self._log_heads_prediction_divergence(step=self.state.global_step, eval_dataset=self.eval_dataset, max_batches=50)

                self.state.global_step += 1
                pbar.update(1)
        
        pbar.close()              
        print("=== Entraînement du modèle terminé. Nombre total de steps :", self.state.global_step)
 

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

        # Flag F-IRM : si défini, on ne met jamais à jour phi
        freeze_phi = getattr(self, 'freeze_phi', False) #True
        K = getattr(self.args, "head_updates_per_encoder_update")
        print(f"[IRM-GAMES] freeze_phi={freeze_phi} | K={K}")

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
                self.model.classifier_heads[env_name],
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

        self.scaler = GradScaler(enabled=self.args.fp16)
        
        iter_loaders = {env_name: cycle(dataloaders[env_name]) for env_name in training_set.keys()}
        grad_accum = max(1, self.args.gradient_accumulation_steps)
        accum_heads = defaultdict(int)
        pbar = tqdm(total=max_steps, desc="Training progress")

        round_idx = 0
        recent_round_losses = []
        while self.state.global_step < max_steps:

            # === Phase 1: update heads ===  
            for _ in range(K):
                self.model.encoder.requires_grad_(False)
                for env_name in training_set.keys():
                    for name, head in self.model.classifier_heads.items():
                        head.requires_grad_(name == env_name)

                    if accum_heads[env_name] == 0:
                        head_optimizers[env_name].zero_grad(set_to_none=True)

                    batch = next(iter_loaders[env_name])
                    
                    self.model.train()
                    batch = self._prepare_inputs(batch)

                    use_amp = bool(self.args.fp16 and self.scaler is not None and self.scaler.is_enabled())
                    if use_amp:
                        with autocast():
                            loss_head = self.compute_loss(self.model, batch)
                    else:
                        loss_head = self.compute_loss(self.model, batch)

                    loss_head = loss_head / grad_accum
                    
                    if use_amp:
                        self.scaler.scale(loss_head).backward()
                    else:
                        loss_head.backward()

                    recent_head_losses.append(loss_head.detach().item())
                    accum_heads[env_name] += 1
                    do_step = (accum_heads[env_name] % grad_accum == 0)

                    if do_step:

                        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                            if use_amp:
                                self.scaler.unscale_(head_optimizers[env_name])
                            
                            torch.nn.utils.clip_grad_norm_(
                                self.model.classifier_heads[env_name].parameters(),
                                self.args.max_grad_norm,
                            )
                        
                        if use_amp:
                            self.scaler.step(head_optimizers[env_name])
                            self.scaler.update()
                        else:
                            head_optimizers[env_name].step()

                        head_schedulers[env_name].step()

                        accum_heads[env_name] = 0

                        moving_avg_head = compute_moving_average(recent_head_losses, window_size=20)


                        if nb_steps_heads_saving and self.state.global_step % nb_steps_heads_saving == 0:
                            self.save_heads(self.state.global_step)
                        if saving_intermediary_models and self.state.global_step % nb_steps_model_saving == 0:
                            self.save_intermediary_model(self.state.global_step)

                        # === Évaluation périodique tous les eval_steps ===
                        if (
                            self.args.do_eval
                            and getattr(self.args, "evaluation_strategy", None) == "steps"
                            and getattr(self.args, "eval_steps", 0) > 0
                            and self.state.global_step % self.args.eval_steps == 0
                        ):
                            
                            
                            metrics = self.evaluate()
                            logger.info(f"*** Evaluation at step {self.state.global_step}: {metrics}")
                            self._log_eval_to_wandb(metrics, step=self.state.global_step)
                            self._log_heads_weight_divergence(step=self.state.global_step)
                            self._log_heads_prediction_divergence(step=self.state.global_step, eval_dataset=self.eval_dataset, max_batches=50)

                        self.state.global_step += 1
                        pbar.update(1)

            # === Phase 2: update shared encoder ===
            if not getattr(self, "freeze_phi", False):
                self.model.encoder.requires_grad_(True)
                phi_optimizer.zero_grad(set_to_none=True)
                losses = []

                for env_name in training_set.keys():
                    self.model.classifier_heads[env_name].requires_grad_(False)

                use_amp = bool(self.args.fp16 and self.scaler is not None and self.scaler.is_enabled())

                for env_name in training_set.keys():
                    batch = next(iter_loaders[env_name])
                    
                    self.model.train()
                    batch = self._prepare_inputs(batch)

                    if use_amp:
                        with autocast():
                            loss_e = self.compute_loss(self.model, batch)
                    else:
                        loss_e = self.compute_loss(self.model, batch)
                    losses.append(loss_e)

                phi_loss = torch.stack(losses).mean()

                if self.args.gradient_accumulation_steps > 1:
                    phi_loss = phi_loss / self.args.gradient_accumulation_steps

                if use_amp:
                    self.scaler.scale(phi_loss).backward()
                    self.scaler.unscale_(phi_optimizer)
                else:
                    phi_loss.backward()
                    
                if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.encoder.parameters(), self.args.max_grad_norm)
                        
                if use_amp:
                    self.scaler.step(phi_optimizer)
                    self.scaler.update()
                else:
                    phi_optimizer.step()

                phi_scheduler.step()

                round_loss = phi_loss.detach().item()         # perte harmonisée du round
                recent_round_losses.append(round_loss)
                moving_avg = compute_moving_average(recent_round_losses, window_size=20)

                if self.is_world_process_zero() and wandb.run:
                    wandb.log({
                        "training/train_loss": round_loss,
                        "training/train_loss_moving_avg": moving_avg,
                        "training/round": round_idx,   # pour pouvoir mettre l’axe X = training/round dans W&B
                    }, step=self.state.global_step)

                round_idx += 1

                self._log_training_to_csv(
                    step=self.state.global_step,
                    loss=round_loss,
                    moving_avg=moving_avg,
                    round_idx=round_idx - 1,
                )

        pbar.close()                  
        print("=== Training complete. Total rounds:", self.state.global_step / len(training_set))


    def save_intermediary_model(self, n_steps):
        fname = os.path.join(self.args.output_dir, f"model-{n_steps}")
        self.save_model(output_dir=fname)

    def save_heads(self, step_count):
        # Ne sauvegarder que si ce processus est le principal
        if not self.is_world_process_zero():
            return
        
        print("saving-heads")
        if not os.path.exists("classifier_heads"):
            os.makedirs("classifier_heads")

        for env, classifier_heads in self.model.classifier_heads.items():
            filepath = os.path.join("classifier_heads", "{}-{}".format(env, step_count))
            
            if hasattr(classifier_heads, "dense"):
                np.save(filepath, classifier_heads.dense.weight.data.cpu().numpy())
            elif hasattr(classifier_heads, "decoder"):
                np.save(filepath, classifier_heads.decoder.weight.data.cpu().numpy())
            elif hasattr(classifier_heads, "vocab_projector"):
                np.save(filepath, classifier_heads.vocab_projector.weight.data.cpu().numpy())
            else:
                print(f"La tête pour l'environnement {env} ne possède pas d'attribut de sauvegarde connu.")
            torch.save(classifier_heads.state_dict(), filepath + ".pt")
            

    def get_single_train_dataloader(self, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        is_ddp = dist.is_available() and dist.is_initialized()
        train_sampler = DistributedSampler(train_dataset) if is_ddp else RandomSampler(train_dataset)

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    
    # --- Divergence des poids des têtes ---
    def _compute_heads_weight_divergence(self):
        import torch, numpy as np
        heads = getattr(self.model, "classifier_heads", None)
        if heads is None or len(heads) < 2:
            return None

        names = list(heads.keys())
        # Empile les poids aplatis de out_proj (DistilBERT)
        Ws = []
        for n in names:
            # DistilBERT: tête = DistilBertClassificationHead, poids => .out_proj.weight
            w = heads[n].out_proj.weight.detach().float().view(-1)   # [num_labels*hidden]
            Ws.append(w)
        W = torch.stack(Ws)  # [E, D]

        # L2 pairwise
        diffs = W[:, None, :] - W[None, :, :]
        l2 = torch.norm(diffs, dim=-1)  # [E, E]

        # Cosine pairwise (1 - cos sim)
        Wn = torch.nn.functional.normalize(W, p=2, dim=-1)
        cos = 1.0 - (Wn @ Wn.T)  # [E, E]

        # moyenne sur la partie supérieure (i<j)
        tri = torch.triu(torch.ones_like(l2), diagonal=1).bool()
        l2_mean  = l2[tri].mean().item()
        cos_mean = cos[tri].mean().item()
        return {"heads/l2_mean": l2_mean, "heads/cosine_mean": cos_mean}

    def _log_heads_weight_divergence(self, step):
        import wandb
        if not (self.is_world_process_zero() and wandb.run):
            stats = self._compute_heads_weight_divergence()
            if stats:
                # --- CSV LOGGING even if WandB off ---
                self._log_heads_to_csv(stats, step)
            return
        stats = self._compute_heads_weight_divergence()
        if stats:
            wandb.log(stats, step=step)
            # --- CSV LOGGING mirror ---
            self._log_heads_to_csv(stats, step)


    # --- Divergence prédictive des têtes (JS / désaccord) ---
    def _compute_heads_prediction_divergence(self, eval_dataset=None, max_batches=50, eps=1e-12):
        import torch, numpy as np, itertools
        heads = getattr(self.model, "classifier_heads", None)
        if heads is None or len(heads) < 2:
            return None

        envs = list(heads.keys())
        # dataloader d'éval
        dl = self.get_eval_dataloader(eval_dataset) if eval_dataset is not None else self.get_eval_dataloader()

        js_sum, dis_sum, n_batches = 0.0, 0.0, 0
        self.model.eval()
        with torch.no_grad():
            for bi, batch in enumerate(dl):
                if bi >= max_batches: break
                batch = self._prepare_inputs(batch)

                # 1) forward encodeur une seule fois
                enc = self.model.encoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask", None),
                )
                hidden = enc.last_hidden_state  # [B, L, H]

                # 2) logits/probas de chaque tête
                probs = {}
                preds = {}
                for e in envs:
                    logits_e = self.model.classifier_heads[e](hidden)  # [B, C]
                    p = torch.softmax(logits_e, dim=-1).clamp_min(eps)
                    probs[e] = p
                    preds[e] = p.argmax(dim=-1)

                # 3) agrégation pairwise (moyenne par batch)
                pair_js, pair_dis, pairs = 0.0, 0.0, 0
                for i, a in enumerate(envs):
                    for b in envs[i+1:]:
                        Pa, Pb = probs[a], probs[b]
                        M = 0.5 * (Pa + Pb)
                        js = 0.5 * ((Pa * (Pa.log() - M.log())).sum(-1) + (Pb * (Pb.log() - M.log())).sum(-1))
                        dis = (preds[a] != preds[b]).float()
                        pair_js  += js.mean().item()
                        pair_dis += dis.mean().item()
                        pairs += 1
                if pairs > 0:
                    js_sum  += pair_js  / pairs
                    dis_sum += pair_dis / pairs
                    n_batches += 1

        if n_batches == 0:
            return None
        return {
            "heads/js_mean":       float(js_sum  / n_batches),
            "heads/disagree_mean": float(dis_sum / n_batches),
        }

    def _log_heads_prediction_divergence(self, step, eval_dataset=None, max_batches=50):
        import wandb
        if not (self.is_world_process_zero() and wandb.run):
            stats = self._compute_heads_prediction_divergence(eval_dataset=eval_dataset, max_batches=max_batches)
            if stats:
                self._log_heads_to_csv(stats, step)
            return
        stats = self._compute_heads_prediction_divergence(eval_dataset=eval_dataset, max_batches=max_batches)
        if stats:
            wandb.log(stats, step=step)
            self._log_heads_to_csv(stats, step)
