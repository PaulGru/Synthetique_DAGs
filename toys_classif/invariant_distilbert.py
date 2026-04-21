import copy
import warnings
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertPreTrainedModel,
    DistilBertModel,
)
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class DistilBertLMHead(nn.Module):
    """DistilBert Head for masked language modeling."""

    def __init__(self, config: DistilBertConfig):
        super().__init__()
        self.vocab_transform = nn.Linear(config.dim, config.dim)
        self.vocab_layer_norm = nn.LayerNorm(config.dim, eps=1e-12)
        self.vocab_projector = nn.Linear(config.dim, config.vocab_size)

    def forward(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.vocab_transform(features)        # (bs, seq_length, dim)
        x = F.gelu(x)                             # (bs, seq_length, dim)
        x = self.vocab_layer_norm(x)              # (bs, seq_length, dim)
        x = self.vocab_projector(x)               # (bs, seq_length, vocab_size)
        return x


class InvariantDistilBertConfig(DistilBertConfig):
    # Keep base type name so tokenizers/configs work out of the box
    model_type = "distilbert"

    def __init__(self, envs: Optional[List[str]] = None, **kwargs):
        """Constructs InvariantDistilBertConfig.
        Args:
            envs: list of environment names. If None/empty, defaults to ["erm"].
        """
        super().__init__(**kwargs)
        self.envs = list(envs) if envs else ["erm"]


class InvariantDistilBertForMaskedLM(DistilBertPreTrainedModel):
    """DistilBERT with one LM head per environment."""

    def __init__(self, config: InvariantDistilBertConfig, model: Optional[nn.Module] = None):
        super().__init__(config)
        self.config = config

        if getattr(config, "is_decoder", False):
            logger.warning(
                "If you want to use `InvariantDistilBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.encoder = DistilBertModel(config)

        self.envs: List[str] = list(config.envs) if getattr(config, "envs", None) else ["erm"]
        self.lm_heads: nn.ModuleDict = nn.ModuleDict({env: DistilBertLMHead(config) for env in self.envs})

        if model is not None:
            # Initialize from a pretrained DistilBertForMaskedLM-like module
            if hasattr(model, "distilbert"):
                self.encoder = copy.deepcopy(model.distilbert)
            elif hasattr(model, "encoder"):
                self.encoder = copy.deepcopy(model.encoder)

            # Copy the single LM head weights to all env heads if present
            src_vocab_transform = getattr(model, "vocab_transform", None)
            src_vocab_layer_norm = getattr(model, "vocab_layer_norm", None)
            src_vocab_projector = getattr(model, "vocab_projector", None)

            for env in self.envs:
                if src_vocab_transform is not None:
                    self.lm_heads[env].vocab_transform = copy.deepcopy(src_vocab_transform)
                if src_vocab_layer_norm is not None:
                    self.lm_heads[env].vocab_layer_norm = copy.deepcopy(src_vocab_layer_norm)
                if src_vocab_projector is not None:
                    self.lm_heads[env].vocab_projector = copy.deepcopy(src_vocab_projector)

        self.n_environments: int = len(self.lm_heads)

        # Initialize weights and apply final processing
        self.post_init()

    def tie_weights(self):
        # Disable tying to avoid conflicts at save time with multiple heads
        return

    # Utilities -------------------------------------------------
    def print_lm_w(self):
        for env, lm_h in self.lm_heads.items():
            print(f"[{env}] vocab_transform.weight shape: {tuple(lm_h.vocab_transform.weight.shape)}")

    def init_head(self):
        for env_name in self.envs:
            self.lm_heads[env_name] = DistilBertLMHead(self.config)

    def init_base(self):
        self.encoder.init_weights()
        self.init_head()

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    def get_output_embeddings(self):
        # Return first head's projector for tie-weights API compatibility
        first_env = next(iter(self.lm_heads))
        return self.lm_heads[first_env].vocab_projector

    def set_output_embeddings(self, new_embeddings):
        for _, lm_head in self.lm_heads.items():
            lm_head.vocab_projector = new_embeddings

    # Forward ---------------------------------------------------
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> MaskedLMOutput:

        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        # --- Ensure inputs are on the same device as the model (defensive) ---
        # Use embedding weight device (robust even under DataParallel)
        emb_device = self.get_input_embeddings().weight.device

        def _to_dev(x):
            return x.to(emb_device) if (x is not None and getattr(x, "device", emb_device) != emb_device) else x

        input_ids = _to_dev(input_ids)
        attention_mask = _to_dev(attention_mask)
        head_mask = _to_dev(head_mask)
        inputs_embeds = _to_dev(inputs_embeds)
        labels = _to_dev(labels)

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        if self.n_environments == 1:
            env_name = next(iter(self.lm_heads))
            prediction_scores = self.lm_heads[env_name](sequence_output)
        else:
            # Average logits across env heads
            logits = None
            for _, lm_head in self.lm_heads.items():
                head_logits = lm_head(sequence_output)
                logits = head_logits if logits is None else (logits + head_logits)
            prediction_scores = logits / float(self.n_environments)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
