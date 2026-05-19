.PHONY: help \
		nlp_agnews_semi_anti_causal nlp_agnews_semi_anti_causal_fixed_wrong nlp_agnews_source_selection nlp_agnews_keyword_selection nlp_agnews_conf_varying_proxy \
        nlp_imdb_genres_semi_anti_causal nlp_imdb_genres_size_selection nlp_imdb_genres_conf_varying_proxy \
        nlp_amazon_semi_anti_causal nlp_amazon_size_selection nlp_amazon_conf_varying_proxy nlp_amazon_rating_natural nlp_amazon_keyword_selection nlp_amazon_sentiment_selection \
        gap_sweep_agnews_sac gap_sweep_agnews_source gap_sweep_agnews_keyword gap_sweep_agnews_conf \
        gap_sweep_imdb_genres_sac gap_sweep_imdb_genres_size_selection gap_sweep_imdb_genres_conf \
        gap_sweep_amazon_sac gap_sweep_amazon_sentiment_selection gap_sweep_amazon_conf \
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
AGNEWS_CLASS_DIST_TRAIN ?= 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25
AGNEWS_CLASS_DIST_TEST  ?= 0.25 0.25 0.25 0.25

AGNN_P_CORRECT_TRAIN ?= 0.9 0.8
AGNN_P_CORRECT_TEST  ?= 0.0
AGNN_LABEL_FLIP      ?= 0.25
AGNN_ERM_STEPS       ?= 25_000
AGNN_ERM_LR          ?= 1e-4
AGNN_IRM_STEPS       ?= 25_000
AGNN_IRM_LR          ?= 1e-4
AGNN_IRM_LAMBDA      ?= 35.0

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
# AG News — Semi Anti-Causal avec token erroné fixe (permutation cyclique)
# ─────────────────────────────────────────────────────────────────────────────
# Comme nlp_agnews_semi_anti_causal mais le token erroné est fixe par classe :
#   World(0)→Sports(blue)  Sports(1)→Business(green)
#   Business(2)→Sci/Tech(yellow)  Sci/Tech(3)→World(red)
nlp_agnews_semi_anti_causal_fixed_wrong:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_agnews_semi_anti_causal_fixed_wrong \
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
		--irm_lambda 10 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# AG News — Source Selection Bias
# ─────────────────────────────────────────────────────────────────────────────
AGNS_P_TRAIN     ?= 1. 1.
AGNS_LABEL_FLIP  ?= 0.35 #0.2
AGNS_N_OOD       ?= 250
AGNS_ERM_STEPS   ?= 25_000
AGNS_ERM_LR      ?= 1e-3
AGNS_IRM_STEPS   ?= 25_000
AGNS_IRM_LR      ?= 1e-3
AGNS_IRM_LAMBDA  ?= 50.0

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
# AG News — Keyword Density Selection Bias
# ─────────────────────────────────────────────────────────────────────────────
# Signal spurieux Z = nombre d'occurrences du vocabulaire thématique de la classe
# "Riche" (score ≥ Q3) : embedding BERT fortement teinté de termes de la classe
# "Pauvre" (score ≤ Q1) : embedding plus neutre → OOD exclusivement pauvres
AGNS_KW_P_TRAIN    ?= 0.9 0.7
AGNS_KW_LABEL_FLIP ?= 0.25
AGNS_KW_N_OOD      ?= 250
AGNS_KW_THRESHOLD  ?= quartile
AGNS_KW_ERM_STEPS  ?= 25_000
AGNS_KW_ERM_LR     ?= 1e-3
AGNS_KW_IRM_STEPS  ?= 25_000
AGNS_KW_IRM_LR     ?= 1e-3
AGNS_KW_IRM_LAMBDA ?= 50.0

nlp_agnews_keyword_selection:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_agnews_keyword_selection \
		--nlp_selection_p_train $(AGNS_KW_P_TRAIN) \
		--nlp_label_flip $(AGNS_KW_LABEL_FLIP) \
		--nlp_n_ood_per_class $(AGNS_KW_N_OOD) \
		--nlp_size_threshold_method $(AGNS_KW_THRESHOLD) \
		--nlp_max_length 256 \
		--nlp_agnews_class_dist_train $(AGNEWS_CLASS_DIST_TRAIN) \
		--nlp_agnews_class_dist_test $(AGNEWS_CLASS_DIST_TEST) \
		--erm_steps $(AGNS_KW_ERM_STEPS) \
		--erm_lr $(AGNS_KW_ERM_LR) \
		--irm_steps $(AGNS_KW_IRM_STEPS) \
		--irm_lr $(AGNS_KW_IRM_LR) \
		--irm_lambda $(AGNS_KW_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# AG News — Confounding varying proxy
# ─────────────────────────────────────────────────────────────────────────────
AGNN_CONF_PROXY_A_TRAIN  ?= 0.05 0.1 #0.01 0.1
AGNN_CONF_PROXY_A_TEST   ?= 1.0
AGNN_CONF_PROXY_P_C_FLIP ?= 0.5
AGNN_CONF_ERM_STEPS      ?= 25_000
AGNN_CONF_ERM_LR         ?= 1e-4
AGNN_CONF_IRM_STEPS      ?= 25_000
AGNN_CONF_IRM_LR         ?= 1e-4
AGNN_CONF_IRM_LAMBDA     ?= 50.0
AGNN_CONF_GAMMA          ?= 0.8

nlp_agnews_conf_varying_proxy:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_agnews_conf_varying_proxy \
		--nlp_conf_a_train $(AGNN_CONF_PROXY_A_TRAIN) \
		--nlp_conf_a_test $(AGNN_CONF_PROXY_A_TEST) \
		--nlp_conf_p_c_flip $(AGNN_CONF_PROXY_P_C_FLIP) \
		--nlp_conf_gamma $(AGNN_CONF_GAMMA) \
		--nlp_max_length 256 \
		--erm_steps $(AGNN_CONF_ERM_STEPS) \
		--erm_lr $(AGNN_CONF_ERM_LR) \
		--irm_steps $(AGNN_CONF_IRM_STEPS) \
		--irm_lr $(AGNN_CONF_IRM_LR) \
		--irm_lambda $(AGNN_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
all_nlp_agnews: nlp_agnews_semi_anti_causal nlp_agnews_semi_anti_causal_fixed_wrong nlp_agnews_source_selection nlp_agnews_keyword_selection nlp_agnews_conf_varying_proxy


# ─────────────────────────────────────────────────────────────────────────────
# IMDB Genres — Semi Anti-Causal
# ─────────────────────────────────────────────────────────────────────────────
IMDB_GENRES_CLASS_RATIO_TRAIN ?= 0.5 0.5
IMDB_GENRES_CLASS_RATIO_TEST  ?= 0.5

IMDB_GENRES_SAC_P_CORRECT_TRAIN ?= 0.9 0.7
IMDB_GENRES_SAC_P_CORRECT_TEST  ?= 0.0
IMDB_GENRES_SAC_LABEL_FLIP      ?= 0.25
IMDB_GENRES_SAC_ERM_STEPS       ?= 25_000
IMDB_GENRES_SAC_ERM_LR          ?= 5e-4
IMDB_GENRES_SAC_IRM_STEPS       ?= 25_000
IMDB_GENRES_SAC_IRM_LR          ?= 5e-4
IMDB_GENRES_SAC_IRM_LAMBDA      ?= 100.0

nlp_imdb_genres_semi_anti_causal:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_imdb_genres_semi_anti_causal \
		--nlp_p_correct_train $(IMDB_GENRES_SAC_P_CORRECT_TRAIN) \
		--nlp_p_correct_test $(IMDB_GENRES_SAC_P_CORRECT_TEST) \
		--nlp_label_flip $(IMDB_GENRES_SAC_LABEL_FLIP) \
		--nlp_max_length 256 \
		--nlp_imdb_class_ratio_train $(IMDB_GENRES_CLASS_RATIO_TRAIN) \
		--nlp_imdb_class_ratio_test $(IMDB_GENRES_CLASS_RATIO_TEST) \
		--erm_steps $(IMDB_GENRES_SAC_ERM_STEPS) \
		--erm_lr $(IMDB_GENRES_SAC_ERM_LR) \
		--irm_steps $(IMDB_GENRES_SAC_IRM_STEPS) \
		--irm_lr $(IMDB_GENRES_SAC_IRM_LR) \
		--irm_lambda $(IMDB_GENRES_SAC_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# IMDB Genres — Size Selection bias
# ─────────────────────────────────────────────────────────────────────────────
IMDB_GENRES_SIZE_P_TRAIN    ?= 0.9 0.7
IMDB_GENRES_SIZE_THRESHOLD  ?= quartile
IMDB_GENRES_SIZE_LABEL_FLIP ?= 0.0
IMDB_GENRES_SIZE_ERM_STEPS  ?= 25_000
IMDB_GENRES_SIZE_ERM_LR     ?= 1e-3
IMDB_GENRES_SIZE_IRM_STEPS  ?= 25_000
IMDB_GENRES_SIZE_IRM_LR     ?= 1e-3
IMDB_GENRES_SIZE_IRM_LAMBDA ?= 50.0

nlp_imdb_genres_size_selection:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_imdb_genres_size_selection \
		--nlp_selection_p_train $(IMDB_GENRES_SIZE_P_TRAIN) \
		--nlp_size_threshold_method $(IMDB_GENRES_SIZE_THRESHOLD) \
		--nlp_label_flip $(IMDB_GENRES_SIZE_LABEL_FLIP) \
		--nlp_max_length 256 \
		--nlp_imdb_class_ratio_train $(IMDB_GENRES_CLASS_RATIO_TRAIN) \
		--nlp_imdb_class_ratio_test $(IMDB_GENRES_CLASS_RATIO_TEST) \
		--erm_steps $(IMDB_GENRES_SIZE_ERM_STEPS) \
		--erm_lr $(IMDB_GENRES_SIZE_ERM_LR) \
		--irm_steps $(IMDB_GENRES_SIZE_IRM_STEPS) \
		--irm_lr $(IMDB_GENRES_SIZE_IRM_LR) \
		--irm_lambda $(IMDB_GENRES_SIZE_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# IMDB Genres — Confounding varying proxy
# ─────────────────────────────────────────────────────────────────────────────
IMDB_GENRES_CONF_PROXY_A_TRAIN  ?= 0.01 0.1
IMDB_GENRES_CONF_PROXY_A_TEST   ?= 1.0
IMDB_GENRES_CONF_PROXY_P_C_FLIP ?= 0.5
IMDB_GENRES_CONF_ERM_STEPS      ?= 25_000
IMDB_GENRES_CONF_ERM_LR         ?= 1e-4
IMDB_GENRES_CONF_IRM_STEPS      ?= 25_000
IMDB_GENRES_CONF_IRM_LR         ?= 1e-4
IMDB_GENRES_CONF_IRM_LAMBDA     ?= 100.0
IMDB_GENRES_CONF_GAMMA          ?= 0.5

nlp_imdb_genres_conf_varying_proxy:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_imdb_genres_conf_varying_proxy \
		--nlp_conf_a_train $(IMDB_GENRES_CONF_PROXY_A_TRAIN) \
		--nlp_conf_a_test $(IMDB_GENRES_CONF_PROXY_A_TEST) \
		--nlp_conf_p_c_flip $(IMDB_GENRES_CONF_PROXY_P_C_FLIP) \
		--nlp_max_length 256 \
		--nlp_imdb_class_ratio_train $(IMDB_GENRES_CLASS_RATIO_TRAIN) \
		--nlp_imdb_class_ratio_test $(IMDB_GENRES_CLASS_RATIO_TEST) \
		--erm_steps $(IMDB_GENRES_CONF_ERM_STEPS) \
		--erm_lr $(IMDB_GENRES_CONF_ERM_LR) \
		--irm_steps $(IMDB_GENRES_CONF_IRM_STEPS) \
		--irm_lr $(IMDB_GENRES_CONF_IRM_LR) \
		--irm_lambda $(IMDB_GENRES_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
all_nlp_imdb_genres: nlp_imdb_genres_semi_anti_causal nlp_imdb_genres_size_selection nlp_imdb_genres_conf_varying_proxy


# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Semi Anti-Causal
# ─────────────────────────────────────────────────────────────────────────────
AMAZON_N_TARGET        ?= 100000

AMAZON_CLASS_RATIO_TRAIN ?= 0.5 0.5
AMAZON_CLASS_RATIO_TEST  ?= 0.5

AMAZON_SAC_P_CORRECT_TRAIN ?= 0.9 0.7
AMAZON_SAC_P_CORRECT_TEST  ?= 0.0
AMAZON_SAC_LABEL_FLIP      ?= 0.25 #0.0
AMAZON_SAC_ERM_STEPS       ?= 25_000
AMAZON_SAC_ERM_LR          ?= 1e-4
AMAZON_SAC_IRM_STEPS       ?= 25_000
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
AMAZON_SENT_P_TRAIN    ?= 0.9 0.7 #0.9 0.7 
AMAZON_SENT_LABEL_FLIP ?= 0.0
AMAZON_SENT_ERM_STEPS  ?= 25_000
AMAZON_SENT_ERM_LR     ?= 1e-3
AMAZON_SENT_IRM_STEPS  ?= 25_000
AMAZON_SENT_IRM_LR     ?= 1e-3
AMAZON_SENT_IRM_LAMBDA ?= 100.0 # 200 balanced

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
AMAZON_CONF_PROXY_A_TRAIN  ?= 0.1 0.1
AMAZON_CONF_PROXY_A_TEST   ?= 1.0
AMAZON_CONF_PROXY_P_C_FLIP ?= 0.5
AMAZON_CONF_ERM_STEPS      ?= 25_000
AMAZON_CONF_ERM_LR         ?= 5e-4
AMAZON_CONF_IRM_STEPS      ?= 25_000
AMAZON_CONF_IRM_LR         ?= 5e-4
AMAZON_CONF_IRM_LAMBDA     ?= 50.0
AMAZON_CONF_GAMMA          ?= 0.8

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
		--nlp_conf_gamma $(AMAZON_CONF_GAMMA) \
		--erm_steps $(AMAZON_CONF_ERM_STEPS) \
		--erm_lr $(AMAZON_CONF_ERM_LR) \
		--irm_steps $(AMAZON_CONF_IRM_STEPS) \
		--irm_lr $(AMAZON_CONF_IRM_LR) \
		--irm_lambda $(AMAZON_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

all_nlp_amazon: nlp_amazon_semi_anti_causal nlp_amazon_conf_varying_proxy nlp_amazon_sentiment_selection

# =============================================================================
# Gap sweep NLP — balayage de la force de corrélation trompeuse
# =============================================================================
# Chaque target passe EXACTEMENT les mêmes hyperparamètres que l'expérience
# simple correspondante (ERM lr, IRM lr, IRM lambda, label_flip, steps, …).

# ── AG News ──────────────────────────────────────────────────────────────────
gap_sweep_agnews_sac:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/run_gap_sweep_nlp.py \
		--dataset nlp_agnews_semi_anti_causal \
		--nlp_max_length 256 \
		--nlp_label_flip $(AGNN_LABEL_FLIP) \
		--nlp_agnews_class_dist_train $(AGNEWS_CLASS_DIST_TRAIN) \
		--nlp_agnews_class_dist_test $(AGNEWS_CLASS_DIST_TEST) \
		--erm_steps $(AGNN_ERM_STEPS) --erm_lr $(AGNN_ERM_LR) \
		--irm_steps $(AGNN_IRM_STEPS) --irm_lr $(AGNN_IRM_LR) --irm_lambda $(AGNN_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

gap_sweep_agnews_source:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/run_gap_sweep_nlp.py \
		--dataset nlp_agnews_source_selection \
		--nlp_max_length 256 \
		--nlp_label_flip $(AGNS_LABEL_FLIP) \
		--nlp_agnews_class_dist_train $(AGNEWS_CLASS_DIST_TRAIN) \
		--nlp_agnews_class_dist_test $(AGNEWS_CLASS_DIST_TEST) \
		--erm_steps $(AGNS_ERM_STEPS) --erm_lr $(AGNS_ERM_LR) \
		--irm_steps $(AGNS_IRM_STEPS) --irm_lr $(AGNS_IRM_LR) --irm_lambda $(AGNS_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

gap_sweep_agnews_keyword:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/run_gap_sweep_nlp.py \
		--dataset nlp_agnews_keyword_selection \
		--nlp_max_length 256 \
		--nlp_label_flip $(AGNS_KW_LABEL_FLIP) \
		--nlp_size_threshold_method $(AGNS_KW_THRESHOLD) \
		--nlp_agnews_class_dist_train $(AGNEWS_CLASS_DIST_TRAIN) \
		--nlp_agnews_class_dist_test $(AGNEWS_CLASS_DIST_TEST) \
		--erm_steps $(AGNS_KW_ERM_STEPS) --erm_lr $(AGNS_KW_ERM_LR) \
		--irm_steps $(AGNS_KW_IRM_STEPS) --irm_lr $(AGNS_KW_IRM_LR) --irm_lambda $(AGNS_KW_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

gap_sweep_agnews_conf:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/run_gap_sweep_nlp.py \
		--dataset nlp_agnews_conf_varying_proxy \
		--nlp_max_length 256 \
		--nlp_conf_p_c_flip $(AGNN_CONF_PROXY_P_C_FLIP) \
		--nlp_conf_gamma $(AGNN_CONF_GAMMA) \
		--erm_steps $(AGNN_CONF_ERM_STEPS) --erm_lr $(AGNN_CONF_ERM_LR) \
		--irm_steps $(AGNN_CONF_IRM_STEPS) --irm_lr $(AGNN_CONF_IRM_LR) --irm_lambda $(AGNN_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 --device $(DEVICE)


# ── IMDB Genres ────────────────────────────────────────────────────────
gap_sweep_imdb_genres_sac:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/run_gap_sweep_nlp.py \
		--dataset nlp_imdb_genres_semi_anti_causal \
		--nlp_max_length 256 \
		--nlp_label_flip 0.25 \
		--erm_steps 25_000 --erm_lr 5e-4 \
		--irm_steps 25_000 --irm_lr 5e-4 --irm_lambda 100.0 \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

gap_sweep_imdb_genres_size_selection:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/run_gap_sweep_nlp.py \
		--dataset nlp_imdb_genres_size_selection \
		--nlp_max_length 256 \
		--nlp_size_threshold_method quartile \
		--nlp_label_flip 0.25 \
		--erm_steps 25_000 --erm_lr 5e-4 \
		--irm_steps 25_000 --irm_lr 5e-4 --irm_lambda 100.0 \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

gap_sweep_imdb_genres_conf:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/run_gap_sweep_nlp.py \
		--dataset nlp_imdb_genres_conf_varying_proxy \
		--nlp_conf_p_c_flip $(IMDB_GENRES_CONF_PROXY_P_C_FLIP) \
		--nlp_max_length 256 \
		--nlp_conf_gamma $(IMDB_GENRES_CONF_GAMMA) \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 --irm_lambda 100.0 \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

# ── Amazon ────────────────────────────────────────────────────────────────────
gap_sweep_amazon_sac:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/run_gap_sweep_nlp.py \
		--dataset nlp_amazon_semi_anti_causal \
		--nlp_max_length 512 \
		--nlp_label_flip $(AMAZON_SAC_LABEL_FLIP) \
		--nlp_amazon_n_target $(AMAZON_N_TARGET) \
		--erm_steps $(AMAZON_SAC_ERM_STEPS) --erm_lr $(AMAZON_SAC_ERM_LR) \
		--irm_steps $(AMAZON_SAC_IRM_STEPS) --irm_lr $(AMAZON_SAC_IRM_LR) --irm_lambda $(AMAZON_SAC_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

gap_sweep_amazon_sentiment_selection:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/run_gap_sweep_nlp.py \
		--dataset nlp_amazon_sentiment_selection \
		--nlp_max_length 512 \
		--nlp_label_flip $(AMAZON_SENT_LABEL_FLIP) \
		--nlp_amazon_n_target $(AMAZON_N_TARGET) \
		--erm_steps $(AMAZON_SENT_ERM_STEPS) --erm_lr $(AMAZON_SENT_ERM_LR) \
		--irm_steps $(AMAZON_SENT_IRM_STEPS) --irm_lr $(AMAZON_SENT_IRM_LR) --irm_lambda $(AMAZON_SENT_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

gap_sweep_amazon_conf:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/run_gap_sweep_nlp.py \
		--dataset nlp_amazon_conf_varying_proxy \
		--nlp_max_length 512 \
		--nlp_conf_p_c_flip $(AMAZON_CONF_PROXY_P_C_FLIP) \
		--nlp_amazon_n_target $(AMAZON_N_TARGET) \
		--nlp_conf_gamma $(AMAZON_CONF_GAMMA) \
		--erm_steps $(AMAZON_CONF_ERM_STEPS) --erm_lr $(AMAZON_CONF_ERM_LR) \
		--irm_steps $(AMAZON_CONF_IRM_STEPS) --irm_lr $(AMAZON_CONF_IRM_LR) --irm_lambda $(AMAZON_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)
