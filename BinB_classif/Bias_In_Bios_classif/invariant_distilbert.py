import copy
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.distilbert.modeling_distilbert import DistilBertPreTrainedModel, DistilBertModel
from transformers.activations import gelu
from transformers.models.distilbert.configuration_distilbert import DistilBertConfig
from transformers.utils import logging
logger = logging.get_logger(__name__)

class DistilBertClassificationHead(nn.Module):
    """DistilBert Head for masked language modeling."""

    # tête de prédiction, MLP classique
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)  
        self.out_proj = nn.Linear(config.dim, config.num_labels)

    def forward(self, features, **kwargs):
        # features: [batch_size, seq_len, hidden_size]
        # on prend le token [CLS] en position 0
        cls_token = features[:, 0, :]            # [batch_size, hidden_size]
        x = self.dropout(cls_token)
        logits = self.out_proj(x)                # [batch_size, num_labels]
        
        return logits


# On prend la config déjà définie pour DistilBert, mais on ajoute un attribut "envs" pour gérer les environnements.
class InvariantDistilBertConfig(DistilBertConfig):
    """
    Modèle DistilBert adapté à la classification de texte
    avec invariance par environnement (IRM-style).
    On possède une tête de classification par environnement, 
    puis on moyenne les logits pour la perte et l'inférence.
    """
    model_type = "invariant-distilbert"

    def __init__(self, envs=1, num_labels=28, **kwargs):
        """Constructs InvariantDistilBertConfig."""
        super().__init__(**kwargs)
        self.envs = envs
        self.num_labels = num_labels


class InvariantDistilBertForSequenceClassification(DistilBertPreTrainedModel):
    authorized_missing_keys = [r"position_ids", r"predictions.decoder.bias"]
    authorized_unexpected_keys = [r"pooler"]

    def __init__(self, config, model=None):  # , model, envs):

        super().__init__(config)

        self.config = config
        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.encoder = DistilBertModel(config)
        
        if isinstance(config.envs, int):
            self.envs = [f"env_{i+1}" for i in range(config.envs)]
        elif isinstance(config.envs, (list, tuple)):
            self.envs = list(config.envs)
        else:
            self.envs = ['erm']

        self.classifier_heads = nn.ModuleDict()
        for env_name in self.envs:
            self.classifier_heads[env_name] = DistilBertClassificationHead(config)

        if model is not None:
            self.encoder = copy.deepcopy(model.distilbert)
        
        for env_name, head in self.classifier_heads.items():
            self.__setattr__(env_name + '_head', self.classifier_heads[env_name])

        # self.encoder.to('cuda')
        # for _, classifier_head in self.classifier_heads.items():
        #     classifier_head.to('cuda')
        
        self.n_environments = len(self.classifier_heads)

    def print_lm_w(self):
        for env, head in self.classifier_heads.items():
            print(head.out_proj.weight)

    def init_base(self):
        self.encoder.init_weights()
        self.init_head()

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.encoder.set_input_embeddings(value)
        # self.embeddings.word_embeddings = value

    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings):
        return None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated … use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")

        # Ignorer special_tokens_mask injecté par DataCollatorWithPadding
        if "special_tokens_mask" in kwargs:
            kwargs.pop("special_tokens_mask")

        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

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
            # Cas simple : une seule tête
            single_head = list(self.classifier_heads.values())[0]
            logits = single_head(sequence_output)               # (batch_size, num_labels)
        else:
            # Moyenne des logits de toutes les têtes
            logits = 0.0
            for head in self.classifier_heads.values():
                logits = logits + head(sequence_output) / self.n_environments

        # 3) Calcul de la loss si on a des labels
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # logits: (batch_size, num_labels), labels: (batch_size,)
            loss = loss_fct(logits, labels)

        # 4) Retour pour return_dict=False
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 5) Packaging dans SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )