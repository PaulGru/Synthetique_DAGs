#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.distilbert.modeling_distilbert import (
    DistilBertPreTrainedModel,
    DistilBertModel,
)
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class InvariantDistilBertConfig(DistilBertConfig):
    model_type = "distilbert"
    def __init__(self, envs: Optional[List[str]] = None, num_labels: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.envs = list(envs) if envs else ["erm"]
        self.num_labels = int(num_labels)


class DistilBertCLSHead(nn.Module):
    def __init__(self, config: InvariantDistilBertConfig):
        super().__init__()
        # mimique légère de la tête HF (dropout + FC)
        dropout_p = getattr(config, "seq_classif_dropout", None)
        if dropout_p is None:
            dropout_p = getattr(config, "dropout", 0.1)
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(config.dim, config.num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # token [CLS] = position 0 (DistilBERT)
        x = hidden_states[:, 0, :]
        x = self.dropout(x)
        logits = self.classifier(x)
        return logits


class InvariantDistilBertForSequenceClassification(DistilBertPreTrainedModel):
    """DistilBERT avec une tête de classification par environnement (ModuleDict)."""
    def __init__(self, config: InvariantDistilBertConfig, model: Optional[nn.Module] = None):
        super().__init__(config)
        self.config = config

        # Encodeur
        self.encoder = DistilBertModel(config)
        if model is not None:
            # Copier les poids d'un DistilBertModel/for* déjà chargé
            if hasattr(model, "distilbert"):
                self.encoder = copy.deepcopy(model.distilbert)
            elif isinstance(model, DistilBertModel):
                self.encoder = copy.deepcopy(model)
            elif hasattr(model, "encoder") and isinstance(model.encoder, DistilBertModel):
                self.encoder = copy.deepcopy(model.encoder)

        # Têtes par environnement
        self.envs: List[str] = list(getattr(config, "envs", ["erm"]))
        self.lm_heads = nn.ModuleDict({env: DistilBertCLSHead(config) for env in self.envs})
        self.n_environments = len(self.lm_heads)

        self.post_init()

    def tie_weights(self):
        # neutralise tout tying résiduel (par sécurité)
        return

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels: Optional[torch.Tensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # s'assurer que tout est sur le device de l'embed (robuste DP)
        emb_device = self.get_input_embeddings().weight.device
        def _to(x):
            return x.to(emb_device) if (x is not None and getattr(x, "device", emb_device) != emb_device) else x

        input_ids = _to(input_ids)
        attention_mask = _to(attention_mask)
        head_mask = _to(head_mask)
        inputs_embeds = _to(inputs_embeds)
        labels = _to(labels)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state

        logits_list = []
        for _, head in self.lm_heads.items():
            logits_list.append(head(hidden))        # [B, C]
        logits = torch.stack(logits_list, dim=0).mean(dim=0)   # [B, C]

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
