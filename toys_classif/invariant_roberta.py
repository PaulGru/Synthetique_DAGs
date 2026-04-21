import copy
import warnings
from typing import List, Optional

import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
    RobertaLMHead,
)
from transformers.modeling_outputs import MaskedLMOutput
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class InvariantRobertaConfig(RobertaConfig):
    # Keep base type so tokenizers/configs work
    model_type = "roberta"

    def __init__(self, envs: Optional[List[str]] = None, **kwargs):
        super().__init__(**kwargs)
        self.envs = list(envs) if envs else ["erm"]


class InvariantRobertaForMaskedLM(RobertaPreTrainedModel):
    def __init__(self, config: InvariantRobertaConfig, model: Optional[nn.Module] = None):
        super().__init__(config)

        self.config = config
        if getattr(config, "is_decoder", False):
            logger.warning(
                "If you want to use `InvariantRobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.encoder = RobertaModel(config, add_pooling_layer=False)

        self.envs = list(config.envs) if getattr(config, "envs", None) else ["erm"]
        self.lm_heads: nn.ModuleDict = nn.ModuleDict({env: RobertaLMHead(config) for env in self.envs})

        if model is not None:
            if hasattr(model, "roberta"):
                self.encoder = copy.deepcopy(model.roberta)
            elif hasattr(model, "encoder"):
                self.encoder = copy.deepcopy(model.encoder)

            for env in self.envs:
                if hasattr(model, "lm_head"):
                    self.lm_heads[env] = copy.deepcopy(model.lm_head)

        self.n_environments = len(self.lm_heads)

        self.post_init()

    def tie_weights(self):
        # Disable tying to avoid conflicts at save time with multiple heads
        return

    def print_lm_w(self):
        for env, lm_h in self.lm_heads.items():
            print(f"[{env}] dense.weight shape: {tuple(lm_h.dense.weight.shape)}")

    def init_head(self):
        for env_name in self.envs:
            self.lm_heads[env_name] = RobertaLMHead(self.config)

    def init_base(self):
        self.encoder.init_weights()
        self.init_head()

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)

    def get_output_embeddings(self):
        first_env = next(iter(self.lm_heads))
        return self.lm_heads[first_env].decoder

    def set_output_embeddings(self, new_embeddings):
        for _, lm_head in self.lm_heads.items():
            lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # --- Ensure inputs are on the same device as the model (defensive) ---
        emb_device = self.get_input_embeddings().weight.device

        def _to(x):
            return x.to(emb_device) if (x is not None and getattr(x, "device", emb_device) != emb_device) else x

        input_ids = _to(input_ids)
        attention_mask = _to(attention_mask)
        token_type_ids = _to(token_type_ids)
        position_ids = _to(position_ids)
        head_mask = _to(head_mask)
        inputs_embeds = _to(inputs_embeds)
        encoder_hidden_states = _to(encoder_hidden_states)
        encoder_attention_mask = _to(encoder_attention_mask)
        labels = _to(labels)

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        if self.n_environments == 1:
            env = next(iter(self.lm_heads))
            prediction_scores = self.lm_heads[env](sequence_output)
        else:
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
