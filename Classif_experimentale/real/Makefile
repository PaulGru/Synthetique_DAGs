.PHONY: help moji civilcomments civilcomments_bias civilcomments_identity \
        amazon coref multinli multinli_ft multinli_irm_ft multinli_hans \
        multinli_snli_injection all

THIS_DIR = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT     = $(abspath $(THIS_DIR)/..)

# Variables
PYTHON = uv run
DEVICE = auto
SEED   = 42

# Chemins des scripts
SCRIPT_MOJI        = real/moji/run_moji_erm_vs_irm.py
SCRIPT_CC_ERM_IRM  = real/civilcomments/run_civilcomments_erm_vs_irm.py
SCRIPT_CC_BIAS     = real/civilcomments/run_civilcomments_bias_envs.py
SCRIPT_CC_IDENTITY = real/civilcomments/run_civilcomments_identity_envs.py
SCRIPT_AMAZON      = real/amazon/custom_split_train.py
SCRIPT_COREF       = real/coref/coref_irm_debiasing.py
SCRIPT_MULTINLI    = real/multinli/run_multinli_erm_vs_irm.py
SCRIPT_MULTINLI_FT_IRM = real/multinli/run_multinli_finetune_irm.py
SCRIPT_MULTINLI_HANS   = real/multinli/run_multinli_hans.py
SCRIPT_MULTINLI_SNLI_INJ = real/multinli/run_nli_snli_injection.py

help:
	@echo "Expériences NLP réelles disponibles :"
	@echo "  make moji                - ERM vs IRM sur TweetEval-Emoji (Moji)"
	@echo "  make civilcomments       - ERM vs IRM sur CivilComments"
	@echo "  make civilcomments_bias  - CivilComments avec environnements de biais"
	@echo "  make civilcomments_identity - CivilComments avec environnements d'identité"
	@echo "  make amazon              - Amazon Reviews (split custom)"
	@echo "  make coref               - Débiaisage gender sur coref (IRM)"
	@echo "  make multinli            - ERM vs IRM sur MultiNLI → ANLI OOD"
	@echo "  make multinli_ft         - idem avec DistilBERT fine-tuné"
	@echo "  make multinli_snli_injection - ERM vs IRM MNLI+SNLI injecté → HANS/ANLI/SNLI-Hard"
	@echo "  make all                 - Lance moji + civilcomments + amazon + multinli"
	@echo ""
	@echo "Variables overridables :"
	@echo "  DEVICE=$(DEVICE)  SEED=$(SEED)"
	@echo "  MOJI_ERM_STEPS=$(MOJI_ERM_STEPS)  MOJI_IRM_STEPS=$(MOJI_IRM_STEPS)"
	@echo "  CC_ERM_STEPS=$(CC_ERM_STEPS)  CC_IRM_STEPS=$(CC_IRM_STEPS)"

# ---- Moji ----
MOJI_ERM_STEPS ?= 25000
MOJI_IRM_STEPS ?= 25000
MOJI_ERM_LR    ?= 1e-5
MOJI_IRM_LR    ?= 1e-5
MOJI_IRM_LAMBDA ?= 1000.0

moji:
	cd $(ROOT) && $(PYTHON) $(SCRIPT_MOJI) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--erm_steps $(MOJI_ERM_STEPS) \
		--erm_lr $(MOJI_ERM_LR) \
		--irm_steps $(MOJI_IRM_STEPS) \
		--irm_lr $(MOJI_IRM_LR) \
		--irm_lambda $(MOJI_IRM_LAMBDA)

# ---- CivilComments ----
CC_ERM_STEPS  ?= 20000
CC_IRM_STEPS  ?= 20000
CC_ERM_LR     ?= 1e-3
CC_IRM_LR     ?= 1e-3
CC_IRM_LAMBDA ?= 1200.0

civilcomments:
	cd $(ROOT) && $(PYTHON) $(SCRIPT_CC_ERM_IRM) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--erm_steps $(CC_ERM_STEPS) \
		--erm_lr $(CC_ERM_LR) \
		--irm_steps $(CC_IRM_STEPS) \
		--irm_lr $(CC_IRM_LR) \
		--irm_lambda $(CC_IRM_LAMBDA)

civilcomments_bias:
	cd $(ROOT) && $(PYTHON) $(SCRIPT_CC_BIAS) \
		--seed $(SEED) \
		--device $(DEVICE)

civilcomments_identity:
	cd $(ROOT) && $(PYTHON) $(SCRIPT_CC_IDENTITY) \
		--seed $(SEED) \
		--device $(DEVICE)

# ---- Amazon ----
amazon:
	cd $(ROOT) && $(PYTHON) $(SCRIPT_AMAZON)

# ---- Coref ----
coref:
	cd $(ROOT) && $(PYTHON) $(SCRIPT_COREF)

# ---- MultiNLI ----
MNLI_ERM_STEPS  ?= 35000
MNLI_IRM_STEPS  ?= 35000
MNLI_ERM_LR     ?= 1e-3
MNLI_IRM_LR     ?= 1e-3
MNLI_IRM_LAMBDA ?= 50.0
MNLI_BATCH      ?= 256
MNLI_EVAL_EVERY ?= 100
MNLI_MAX_LENGTH ?= 256
MNLI_EMBED_BATCH ?= 256
MNLI_CACHE_DIR  ?= real/multinli/.embed_cache
MNLI_BERT_MODEL ?= bert-base-uncased

multinli:
	cd $(ROOT) && $(PYTHON) $(SCRIPT_MULTINLI) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--erm_steps $(MNLI_ERM_STEPS) \
		--erm_lr $(MNLI_ERM_LR) \
		--irm_steps $(MNLI_IRM_STEPS) \
		--irm_lr $(MNLI_IRM_LR) \
		--irm_lambda $(MNLI_IRM_LAMBDA) \
		--cache_dir $(MNLI_CACHE_DIR) \
		--batch $(MNLI_BATCH) \
		--eval_every $(MNLI_EVAL_EVERY) \
		--max_length $(MNLI_MAX_LENGTH) \
		--embed_batch $(MNLI_EMBED_BATCH)

# ---- MultiNLI + fine-tuning DistilBERT ----
# Pipeline 2 étapes :
#   1. Fine-tune DistilBERT (cross-entropie standard sur les 5 genres)
#   2. Geler le backbone → extraire embeddings CLS enrichis
#   3. LogisticReg ERM puis IRM sur ces embeddings (--no_mlp)
MNLI_FT_EPOCHS  ?= 2
MNLI_FT_LR      ?= 2e-5
MNLI_FT_BATCH   ?= 32

multinli_ft:
	cd $(ROOT) && $(PYTHON) $(SCRIPT_MULTINLI) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--bert_model distilbert-base-uncased \
		--finetune_epochs $(MNLI_FT_EPOCHS) \
		--finetune_lr $(MNLI_FT_LR) \
		--finetune_batch $(MNLI_FT_BATCH) \
		--no_mlp \
		--erm_steps $(MNLI_ERM_STEPS) \
		--erm_lr $(MNLI_ERM_LR) \
		--irm_steps $(MNLI_IRM_STEPS) \
		--irm_lr $(MNLI_IRM_LR) \
		--irm_lambda $(MNLI_IRM_LAMBDA) \
		--batch $(MNLI_BATCH) \
		--eval_every $(MNLI_EVAL_EVERY) \
		--max_length $(MNLI_MAX_LENGTH) \
		--embed_batch $(MNLI_EMBED_BATCH) \
		--out_dir real/multinli/plots_ft

all: moji civilcomments amazon multinli

# ---- MultiNLI + fine-tuning IRM end-to-end ----
# Fine-tune les 2 dernières couches BERT avec IRMv1 (genre = env)
# Compare avec un fine-tuning ERM classique
MNLI_IRM_FT_EPOCHS    ?= 1
MNLI_IRM_FT_LR_BERT   ?= 2e-5
MNLI_IRM_FT_LR_HEAD   ?= 1e-3
MNLI_IRM_FT_LAMBDA    ?= 100.0
MNLI_IRM_FT_BATCH     ?= 16
MNLI_IRM_FT_UNFROZEN  ?= 2
MNLI_IRM_FT_EVAL      ?= 500
MNLI_IRM_FT_SNLI_MAX  ?= 90000

multinli_irm_ft:
	cd $(ROOT) && $(PYTHON) $(SCRIPT_MULTINLI_FT_IRM) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--epochs $(MNLI_IRM_FT_EPOCHS) \
		--batch_size $(MNLI_IRM_FT_BATCH) \
		--lr_bert $(MNLI_IRM_FT_LR_BERT) \
		--lr_head $(MNLI_IRM_FT_LR_HEAD) \
		--irm_lambda $(MNLI_IRM_FT_LAMBDA) \
		--n_unfrozen_layers $(MNLI_IRM_FT_UNFROZEN) \
		--eval_every $(MNLI_IRM_FT_EVAL) \
		--max_length $(MNLI_MAX_LENGTH) \
		--snli_max_samples $(MNLI_IRM_FT_SNLI_MAX)

# ---- MultiNLI → HANS (exp. i) ----
# Fine-tune ERM vs IRM sur MNLI uniquement (5 genres),
# évaluation OOD sur val_mismatched + HANS (heuristiques NLI)
MNLI_HANS_EPOCHS   ?= 8
MNLI_HANS_LR_BERT  ?= 1e-5
MNLI_HANS_LR_HEAD  ?= 5e-4
MNLI_HANS_LAMBDA   ?= 25.0
MNLI_HANS_BATCH    ?= 64
MNLI_HANS_UNFROZEN ?= 5
MNLI_HANS_EVAL     ?= 200
MNLI_HANS_MODEL    ?= distilbert-base-uncased

multinli_hans:
	cd $(ROOT) && $(PYTHON) $(SCRIPT_MULTINLI_HANS) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--epochs $(MNLI_HANS_EPOCHS) \
		--batch_size $(MNLI_HANS_BATCH) \
		--lr_bert $(MNLI_HANS_LR_BERT) \
		--lr_head $(MNLI_HANS_LR_HEAD) \
		--irm_lambda $(MNLI_HANS_LAMBDA) \
		--n_unfrozen_layers $(MNLI_HANS_UNFROZEN) \
		--eval_every $(MNLI_HANS_EVAL) \
		--max_length $(MNLI_MAX_LENGTH) \
		--bert_model $(MNLI_HANS_MODEL)

# ---- MultiNLI + injection SNLI (exp. j) ----
# ERM vs IRM sur 5 envs = MNLI genres + fraction croissante de SNLI.
# Évaluation : Val Matched/Mismatched · ANLI R1-R3 · HANS · SNLI-Hard.
# Les embeddings MNLI/SNLI sont réutilisés depuis le cache de multinli.
MNLI_INJ_ERM_STEPS  ?= 25000
MNLI_INJ_IRM_STEPS  ?= 25000
MNLI_INJ_ERM_LR     ?= 1e-3
MNLI_INJ_IRM_LR     ?= 1e-3
MNLI_INJ_IRM_LAMBDA ?= 500.0
MNLI_INJ_BATCH      ?= 512
MNLI_INJ_EVAL_EVERY ?= 100
MNLI_INJ_SNLI_RATIOS ?= 0.05 0.10 0.15 0.20 0.25

multinli_snli_injection:
	cd $(ROOT) && $(PYTHON) $(SCRIPT_MULTINLI_SNLI_INJ) \
		--seed $(SEED) \
		--device $(DEVICE) \
		--bert_model $(MNLI_BERT_MODEL) \
		--max_length $(MNLI_MAX_LENGTH) \
		--embed_batch $(MNLI_EMBED_BATCH) \
		--snli_ratios $(MNLI_INJ_SNLI_RATIOS) \
		--erm_steps $(MNLI_INJ_ERM_STEPS) \
		--erm_lr $(MNLI_INJ_ERM_LR) \
		--irm_steps $(MNLI_INJ_IRM_STEPS) \
		--irm_lr $(MNLI_INJ_IRM_LR) \
		--irm_lambda $(MNLI_INJ_IRM_LAMBDA) \
		--batch $(MNLI_INJ_BATCH) \
		--eval_every $(MNLI_INJ_EVAL_EVERY) \
		--cache_dir $(MNLI_CACHE_DIR)
