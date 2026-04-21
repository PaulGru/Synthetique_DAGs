#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel,
    RobertaModel,
)
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class InvariantRobertaConfig(RobertaConfig):
    model_type = "roberta"
    def __init__(self, envs: Optional[List[str]] = None, num_labels: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.envs = list(envs) if envs else ["erm"]
        self.num_labels = int(num_labels)


class RobertaCLSHead(nn.Module):
    def __init__(self, config: InvariantRobertaConfig):
        super().__init__()
        hidden = config.hidden_size
        self.dropout = nn.Dropout(getattr(config, "hidden_dropout_prob", 0.1))
        self.dense = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features[:, 0, :]          # token <s> (équiv. CLS)
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class InvariantRobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config: InvariantRobertaConfig, model: Optional[nn.Module] = None):
        super().__init__(config)
        self.config = config

        self.encoder = RobertaModel(config, add_pooling_layer=False)
        if model is not None:
            if hasattr(model, "roberta"):
                self.encoder = copy.deepcopy(model.roberta)
            elif isinstance(model, RobertaModel):
                self.encoder = copy.deepcopy(model)
            elif hasattr(model, "encoder") and isinstance(model.encoder, RobertaModel):
                self.encoder = copy.deepcopy(model.encoder)

        self.envs: List[str] = list(getattr(config, "envs", ["erm"]))
        self.lm_heads = nn.ModuleDict({env: RobertaCLSHead(config) for env in self.envs})
        self.n_environments = len(self.lm_heads)

        self.post_init()

    def tie_weights(self):
        return

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels: Optional[torch.Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        emb_device = self.get_input_embeddings().weight.device
        def _to(x):
            return x.to(emb_device) if (x is not None and getattr(x, "device", emb_device) != emb_device) else x

        input_ids = _to(input_ids)
        attention_mask = _to(attention_mask)
        token_type_ids = _to(token_type_ids)
        position_ids = _to(position_ids)
        head_mask = _to(head_mask)
        inputs_embeds = _to(inputs_embeds)
        labels = _to(labels)

        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state

        if self.n_environments == 1:
            env = next(iter(self.lm_heads))
            logits = self.lm_heads[env](hidden)
        else:
            agg = None
            for _, head in self.lm_heads.items():
                lg = head(hidden)
                agg = lg if agg is None else (agg + lg)
            logits = agg / float(self.n_environments)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, outputs.hidden_states, outputs.attentions)
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
