.PHONY: help \
        nlp_sms_spam nlp_sms_spam_size_selection \
        nlp_sms_spam_conf_varying_proxy nlp_sms_spam_conf_varying_gamma nlp_sms_spam_conf_varying_pc \
        nlp_agnews_semi_anti_causal nlp_agnews_source_selection nlp_agnews_conf_varying_proxy \
        nlp_sst2_semi_anti_causal nlp_sst2_selection nlp_sst2_genre_selection nlp_sst2_conf_varying_proxy \
        nlp_imdb_conf_varying_proxy nlp_imdb_semi_anti_causal nlp_imdb_selection nlp_imdb_size_selection nlp_imdb_br_selection \
        nlp_amazon_semi_anti_causal nlp_amazon_size_selection nlp_amazon_conf_varying_proxy nlp_amazon_rating_natural nlp_amazon_keyword_selection nlp_amazon_sentiment_selection \
        all all_sst2 all_conf

THIS_DIR  = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT      = $(abspath $(THIS_DIR)/..)

PYTHON    = uv run
DEVICE    = auto
EVAL_EVERY ?= 20
FINETUNE_BERT ?=
FINETUNE_BERT_FLAG = $(if $(FINETUNE_BERT),--finetune_bert,)

# ─────────────────────────────────────────────────────────────────────────────
# AG News — Semi Anti-Causal
# ─────────────────────────────────────────────────────────────────────────────
# Déséquilibre de classes pour AG News (4 classes : World, Sports, Business, Sci/Tech)
# Env 0 : World=10%, Sports=50%, Business=20%, Sci/Tech=20%
# Env 1 : équilibré (25% chacune)
# Test  : équilibré (25% chacune)
AGNEWS_CLASS_DIST_TRAIN ?= 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25
AGNEWS_CLASS_DIST_TEST  ?= 0.25 0.25 0.25 0.25

AGNN_P_CORRECT_TRAIN ?= 0.9 0.8
AGNN_P_CORRECT_TEST  ?= 0.0
AGNN_LABEL_FLIP      ?= 0.2
AGNN_ERM_STEPS       ?= 25_000
AGNN_ERM_LR          ?= 1e-4
AGNN_IRM_STEPS       ?= 25_000
AGNN_IRM_LR          ?= 1e-4
AGNN_IRM_LAMBDA      ?= 100.0

nlp_agnews_semi_anti_causal:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_agnews_semi_anti_causal \
		--nlp_p_correct_train $(AGNN_P_CORRECT_TRAIN) \
		--nlp_p_correct_test $(AGNN_P_CORRECT_TEST) \
		--nlp_label_flip $(AGNN_LABEL_FLIP) \
		--nlp_max_length 256 \
		--nlp_agnews_class_dist_train $(AGNEWS_CLASS_DIST_TRAIN) \
		--nlp_agnews_class_dist_test $(AGNEWS_CLASS_DIST_TEST) \
		--erm_steps $(AGNN_ERM_STEPS) \
		--erm_lr $(AGNN_ERM_LR) \
		--irm_steps $(AGNN_IRM_STEPS) \
		--irm_lr $(AGNN_IRM_LR) \
		--irm_lambda $(AGNN_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# AG News — Source Selection Bias
# ─────────────────────────────────────────────────────────────────────────────
AGNS_P_TRAIN     ?= 0.9 0.8
AGNS_LABEL_FLIP  ?= 0.2
AGNS_N_OOD       ?= 250
AGNS_ERM_STEPS   ?= 25000
AGNS_ERM_LR      ?= 5e-4
AGNS_IRM_STEPS   ?= 25000
AGNS_IRM_LR      ?= 5e-4
AGNS_IRM_LAMBDA  ?= 100.0

nlp_agnews_source_selection:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_agnews_source_selection \
		--nlp_selection_p_train $(AGNS_P_TRAIN) \
		--nlp_label_flip $(AGNS_LABEL_FLIP) \
		--nlp_n_ood_per_class $(AGNS_N_OOD) \
		--nlp_max_length 256 \
		--nlp_agnews_class_dist_train $(AGNEWS_CLASS_DIST_TRAIN) \
		--nlp_agnews_class_dist_test $(AGNEWS_CLASS_DIST_TEST) \
		--erm_steps $(AGNS_ERM_STEPS) \
		--erm_lr $(AGNS_ERM_LR) \
		--irm_steps $(AGNS_IRM_STEPS) \
		--irm_lr $(AGNS_IRM_LR) \
		--irm_lambda $(AGNS_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# AG News — Confounding variant 1 : varying proxy
# ─────────────────────────────────────────────────────────────────────────────
AGNN_CONF_PROXY_A_TRAIN  ?= 0.01 0.1
AGNN_CONF_PROXY_A_TEST   ?= 1.0
AGNN_CONF_PROXY_P_C_FLIP ?= 0.25
AGNN_CONF_ERM_STEPS      ?= 25000
AGNN_CONF_ERM_LR         ?= 1e-4
AGNN_CONF_IRM_STEPS      ?= 25000
AGNN_CONF_IRM_LR         ?= 1e-4
AGNN_CONF_IRM_LAMBDA     ?= 150.0

nlp_agnews_conf_varying_proxy:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_agnews_conf_varying_proxy \
		--nlp_conf_a_train $(AGNN_CONF_PROXY_A_TRAIN) \
		--nlp_conf_a_test $(AGNN_CONF_PROXY_A_TEST) \
		--nlp_conf_p_c_flip $(AGNN_CONF_PROXY_P_C_FLIP) \
		--nlp_max_length 256 \
		--erm_steps $(AGNN_CONF_ERM_STEPS) \
		--erm_lr $(AGNN_CONF_ERM_LR) \
		--irm_steps $(AGNN_CONF_IRM_STEPS) \
		--irm_lr $(AGNN_CONF_IRM_LR) \
		--irm_lambda $(AGNN_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
all_nlp_agnews: nlp_agnews_semi_anti_causal nlp_agnews_source_selection nlp_agnews_conf_varying_proxy


# Déséquilibre de classes pour les expériences IMDB
# Env 0 : 20% positifs / 80% négatifs
# Env 1 : 80% positifs / 20% négatifs
# Test  : équilibré (50/50)
IMDB_CLASS_RATIO_TRAIN ?= 0.5 0.5
IMDB_CLASS_RATIO_TEST  ?= 0.5

# ─────────────────────────────────────────────────────────────────────────────
# IMDB — Semi Anti-Causal
# ─────────────────────────────────────────────────────────────────────────────
IMDB_SAC_P_CORRECT_TRAIN ?= 0.9 0.7
IMDB_SAC_P_CORRECT_TEST  ?= 0.0
IMDB_SAC_LABEL_FLIP      ?= 0.25
IMDB_SAC_ERM_STEPS       ?= 25000
IMDB_SAC_ERM_LR          ?= 5e-4
IMDB_SAC_IRM_STEPS       ?= 25000
IMDB_SAC_IRM_LR          ?= 5e-4
IMDB_SAC_IRM_LAMBDA      ?= 100.0

nlp_imdb_semi_anti_causal:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_imdb_semi_anti_causal \
		--nlp_p_correct_train $(IMDB_SAC_P_CORRECT_TRAIN) \
		--nlp_p_correct_test $(IMDB_SAC_P_CORRECT_TEST) \
		--nlp_label_flip $(IMDB_SAC_LABEL_FLIP) \
		--nlp_max_length 512 \
		--nlp_imdb_class_ratio_train $(IMDB_CLASS_RATIO_TRAIN) \
		--nlp_imdb_class_ratio_test $(IMDB_CLASS_RATIO_TEST) \
		--erm_steps $(IMDB_SAC_ERM_STEPS) \
		--erm_lr $(IMDB_SAC_ERM_LR) \
		--irm_steps $(IMDB_SAC_IRM_STEPS) \
		--irm_lr $(IMDB_SAC_IRM_LR) \
		--irm_lambda $(IMDB_SAC_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# IMDB — Size Selection bias
# ─────────────────────────────────────────────────────────────────────────────
# Typique : négatif COURT (< Q1 négatifs) ou positif LONG (> Q3 positifs)
# OOD     : négatif très long + positif très court (signal spurieux inversé)
IMDB_SIZE_P_TRAIN    ?= 0.9 0.8
IMDB_SIZE_THRESHOLD  ?= quartile
IMDB_SIZE_LABEL_FLIP ?= 0.25
IMDB_SIZE_ERM_STEPS  ?= 25000
IMDB_SIZE_ERM_LR     ?= 5e-4
IMDB_SIZE_IRM_STEPS  ?= 25000
IMDB_SIZE_IRM_LR     ?= 5e-4
IMDB_SIZE_IRM_LAMBDA ?= 150.0

nlp_imdb_size_selection:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_imdb_size_selection \
		--nlp_selection_p_train $(IMDB_SIZE_P_TRAIN) \
		--nlp_size_threshold_method $(IMDB_SIZE_THRESHOLD) \
		--nlp_label_flip $(IMDB_SIZE_LABEL_FLIP) \
		--nlp_max_length 512 \
		--nlp_imdb_class_ratio_train $(IMDB_CLASS_RATIO_TRAIN) \
		--nlp_imdb_class_ratio_test $(IMDB_CLASS_RATIO_TEST) \
		--erm_steps $(IMDB_SIZE_ERM_STEPS) \
		--erm_lr $(IMDB_SIZE_ERM_LR) \
		--irm_steps $(IMDB_SIZE_IRM_STEPS) \
		--irm_lr $(IMDB_SIZE_IRM_LR) \
		--irm_lambda $(IMDB_SIZE_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# IMDB — BR tag Selection bias
# ─────────────────────────────────────────────────────────────────────────────
# Typique : négatif AVEC <br>, positif SANS <br>
# OOD     : négatif sans <br> + positif avec <br> (signal spurieux inversé)
IMDB_BR_P_TRAIN    ?= 0.9 0.8
IMDB_BR_LABEL_FLIP ?= 0.1
IMDB_BR_ERM_STEPS  ?= 25000
IMDB_BR_ERM_LR     ?= 1e-4
IMDB_BR_IRM_STEPS  ?= 25000
IMDB_BR_IRM_LR     ?= 1e-4
IMDB_BR_IRM_LAMBDA ?= 50.0

nlp_imdb_br_selection:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_imdb_br_selection \
		--nlp_selection_p_train $(IMDB_BR_P_TRAIN) \
		--nlp_label_flip $(IMDB_BR_LABEL_FLIP) \
		--nlp_max_length 512 \
		--nlp_imdb_class_ratio_train $(IMDB_CLASS_RATIO_TRAIN) \
		--nlp_imdb_class_ratio_test $(IMDB_CLASS_RATIO_TEST) \
		--erm_steps $(IMDB_BR_ERM_STEPS) \
		--erm_lr $(IMDB_BR_ERM_LR) \
		--irm_steps $(IMDB_BR_IRM_STEPS) \
		--irm_lr $(IMDB_BR_IRM_LR) \
		--irm_lambda $(IMDB_BR_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# IMDB — Confounding varying proxy
# ─────────────────────────────────────────────────────────────────────────────
IMDB_CONF_PROXY_A_TRAIN  ?= 0.01 0.1
IMDB_CONF_PROXY_A_TEST   ?= 1.0
IMDB_CONF_PROXY_P_C_FLIP ?= 0.25
IMDB_CONF_ERM_STEPS      ?= 25000
IMDB_CONF_ERM_LR         ?= 5e-4
IMDB_CONF_IRM_STEPS      ?= 25000
IMDB_CONF_IRM_LR         ?= 5e-4
IMDB_CONF_IRM_LAMBDA     ?= 100.0

nlp_imdb_conf_varying_proxy:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_imdb_conf_varying_proxy \
		--nlp_conf_a_train $(IMDB_CONF_PROXY_A_TRAIN) \
		--nlp_conf_a_test $(IMDB_CONF_PROXY_A_TEST) \
		--nlp_conf_p_c_flip $(IMDB_CONF_PROXY_P_C_FLIP) \
		--nlp_max_length 512 \
		--nlp_imdb_class_ratio_train $(IMDB_CLASS_RATIO_TRAIN) \
		--nlp_imdb_class_ratio_test $(IMDB_CLASS_RATIO_TEST) \
		--erm_steps $(IMDB_CONF_ERM_STEPS) \
		--erm_lr $(IMDB_CONF_ERM_LR) \
		--irm_steps $(IMDB_CONF_IRM_STEPS) \
		--irm_lr $(IMDB_CONF_IRM_LR) \
		--irm_lambda $(IMDB_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)


all_nlp_imdb: nlp_imdb_semi_anti_causal nlp_imdb_size_selection nlp_imdb_br_selection nlp_imdb_conf_varying_proxy 


# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Semi Anti-Causal
# ─────────────────────────────────────────────────────────────────────────────
AMAZON_N_TARGET        ?= 100000

# Déséquilibre de classes pour les expériences Amazon
# Env 0 : 20% positifs / 80% négatifs
# Env 1 : 30% positifs / 70% négatifs
# Test  : 90% positifs / 10% négatifs
AMAZON_CLASS_RATIO_TRAIN ?= 0.5 0.5
AMAZON_CLASS_RATIO_TEST  ?= 0.5

AMAZON_SAC_P_CORRECT_TRAIN ?= 0.9 0.8
AMAZON_SAC_P_CORRECT_TEST  ?= 0.0
AMAZON_SAC_LABEL_FLIP      ?= 0.25
AMAZON_SAC_ERM_STEPS       ?= 25000
AMAZON_SAC_ERM_LR          ?= 1e-4
AMAZON_SAC_IRM_STEPS       ?= 25000
AMAZON_SAC_IRM_LR          ?= 1e-4
AMAZON_SAC_IRM_LAMBDA      ?= 100

nlp_amazon_semi_anti_causal:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_amazon_semi_anti_causal \
		--nlp_p_correct_train $(AMAZON_SAC_P_CORRECT_TRAIN) \
		--nlp_p_correct_test $(AMAZON_SAC_P_CORRECT_TEST) \
		--nlp_label_flip $(AMAZON_SAC_LABEL_FLIP) \
		--nlp_max_length 512 \
		--nlp_amazon_n_target $(AMAZON_N_TARGET) \
		--nlp_amazon_class_ratio_train $(AMAZON_CLASS_RATIO_TRAIN) \
		--nlp_amazon_class_ratio_test $(AMAZON_CLASS_RATIO_TEST) \
		--erm_steps $(AMAZON_SAC_ERM_STEPS) \
		--erm_lr $(AMAZON_SAC_ERM_LR) \
		--irm_steps $(AMAZON_SAC_IRM_STEPS) \
		--irm_lr $(AMAZON_SAC_IRM_LR) \
		--irm_lambda $(AMAZON_SAC_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Sentiment Selection (note comme proxy trompeur d'utilité)
# ─────────────────────────────────────────────────────────────────────────────
# Typique : utile + positif (5★), OU inutile + négatif (1★)
# OOD     : utile + négatif (1★), OU inutile + positif (5★)
# Le modèle apprend "sentiment → utilité", raccourci qui casse au test OOD
AMAZON_SENT_P_TRAIN    ?= 0.9 0.7
AMAZON_SENT_LABEL_FLIP ?= 0.25
AMAZON_SENT_ERM_STEPS  ?= 25000
AMAZON_SENT_ERM_LR     ?= 1e-4
AMAZON_SENT_IRM_STEPS  ?= 25000
AMAZON_SENT_IRM_LR     ?= 1e-4
AMAZON_SENT_IRM_LAMBDA ?= 200.0

nlp_amazon_sentiment_selection:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_amazon_sentiment_selection \
		--nlp_selection_p_train $(AMAZON_SENT_P_TRAIN) \
		--nlp_label_flip $(AMAZON_SENT_LABEL_FLIP) \
		--nlp_max_length 512 \
		--nlp_amazon_n_target $(AMAZON_N_TARGET) \
		--nlp_amazon_class_ratio_train $(AMAZON_CLASS_RATIO_TRAIN) \
		--nlp_amazon_class_ratio_test $(AMAZON_CLASS_RATIO_TEST) \
		--erm_steps $(AMAZON_SENT_ERM_STEPS) \
		--erm_lr $(AMAZON_SENT_ERM_LR) \
		--irm_steps $(AMAZON_SENT_IRM_STEPS) \
		--irm_lr $(AMAZON_SENT_IRM_LR) \
		--irm_lambda $(AMAZON_SENT_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)
		
# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Confounding varying proxy
# ─────────────────────────────────────────────────────────────────────────────
AMAZON_CONF_PROXY_A_TRAIN  ?= 0.01 0.1
AMAZON_CONF_PROXY_A_TEST   ?= 1.0
AMAZON_CONF_PROXY_P_C_FLIP ?= 0.25
AMAZON_CONF_ERM_STEPS      ?= 25000
AMAZON_CONF_ERM_LR         ?= 1e-4
AMAZON_CONF_IRM_STEPS      ?= 25000
AMAZON_CONF_IRM_LR         ?= 1e-4
AMAZON_CONF_IRM_LAMBDA     ?= 100.0

nlp_amazon_conf_varying_proxy:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_amazon_conf_varying_proxy \
		--nlp_conf_a_train $(AMAZON_CONF_PROXY_A_TRAIN) \
		--nlp_conf_a_test $(AMAZON_CONF_PROXY_A_TEST) \
		--nlp_conf_p_c_flip $(AMAZON_CONF_PROXY_P_C_FLIP) \
		--nlp_max_length 512 \
		--nlp_amazon_n_target $(AMAZON_N_TARGET) \
		--nlp_amazon_class_ratio_train $(AMAZON_CLASS_RATIO_TRAIN) \
		--nlp_amazon_class_ratio_test $(AMAZON_CLASS_RATIO_TEST) \
		--erm_steps $(AMAZON_CONF_ERM_STEPS) \
		--erm_lr $(AMAZON_CONF_ERM_LR) \
		--irm_steps $(AMAZON_CONF_IRM_STEPS) \
		--irm_lr $(AMAZON_CONF_IRM_LR) \
		--irm_lambda $(AMAZON_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

all_nlp_amazon: nlp_amazon_semi_anti_causal nlp_amazon_conf_varying_proxy nlp_amazon_sentiment_selection