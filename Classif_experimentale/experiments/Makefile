.PHONY: help \
		nlp_agnews_semi_anti_causal nlp_agnews_size_selection nlp_agnews_conf_varying_proxy \
        nlp_imdb_genres_semi_anti_causal nlp_imdb_genres_size_selection nlp_imdb_genres_conf_varying_proxy \
        nlp_amazon_semi_anti_causal nlp_amazon_sentiment_selection nlp_amazon_conf_varying_proxy \
        gap_sweep_agnews_sac gap_sweep_agnews_size gap_sweep_agnews_conf \
        gap_sweep_imdb_genres_sac gap_sweep_imdb_genres_size_selection gap_sweep_imdb_genres_conf \
        gap_sweep_amazon_sac gap_sweep_amazon_sentiment_selection gap_sweep_amazon_conf \
        noise_sweep_agnews_sac noise_sweep_agnews_size noise_sweep_agnews_conf \
        noise_sweep_imdb_genres_sac noise_sweep_imdb_genres_size_selection noise_sweep_imdb_genres_conf \
        noise_sweep_amazon_sac noise_sweep_amazon_sentiment_selection noise_sweep_amazon_conf \
        grand_test grand_test_resume \
        grand_test_agnews grand_test_imdb grand_test_amazon \
        grand_test_resume_agnews grand_test_resume_imdb grand_test_resume_amazon \
        grand_test_part0 grand_test_part1 grand_test_part2 grand_test_part3 \
        grand_test_resume_part0 grand_test_resume_part1 grand_test_resume_part2 grand_test_resume_part3 \


THIS_DIR  = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT      = $(abspath $(THIS_DIR)/..)

# ─────────────────────────────────────────────────────────────────────────────
# help
# ─────────────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Usage: make <target> [DEVICE=cuda:0] [NLP_SEED=0] [SEEDS='0 1 2']"
	@echo ""
	@echo "Single-run targets (1 dataset × 1 mechanism, seed = NLP_SEED)"
	@echo "  nlp_agnews_semi_anti_causal          AG News  – SAC (token injection)"
	@echo "  nlp_agnews_size_selection             AG News  – Size selection bias"
	@echo "  nlp_agnews_conf_varying_proxy         AG News  – Confounding proxy"
	@echo "  nlp_imdb_genres_semi_anti_causal      IMDB     – SAC"
	@echo "  nlp_imdb_genres_size_selection        IMDB     – Size selection"
	@echo "  nlp_imdb_genres_conf_varying_proxy    IMDB     – Confounding"
	@echo "  nlp_amazon_semi_anti_causal           Amazon   – SAC"
	@echo "  nlp_amazon_sentiment_selection        Amazon   – Sentiment selection"
	@echo "  nlp_amazon_conf_varying_proxy         Amazon   – Confounding"
	@echo ""
	@echo "Multi-seed targets (loop over SEEDS, then aggregate)"
	@echo "  all_nlp_agnews                        All 3 mechanisms for AG News"
	@echo "  all_nlp_imdb_genres                   All 3 mechanisms for IMDB"
	@echo "  all_nlp_amazon                        All 3 mechanisms for Amazon"
	@echo ""
	@echo "Gap sweep targets (sweep train/test correlation gap)"
	@echo "  gap_sweep_agnews_sac/size/conf"
	@echo "  gap_sweep_imdb_genres_sac/size_selection/conf"
	@echo "  gap_sweep_amazon_sac/sentiment_selection/conf"
	@echo ""
	@echo "Noise sweep targets (sweep noise level at fixed gap)"
	@echo "  noise_sweep_agnews_sac/size/conf"
	@echo "  noise_sweep_imdb_genres_sac/size_selection/conf"
	@echo "  noise_sweep_amazon_sac/sentiment_selection/conf"
	@echo "  all_noise_sweep_agnews / all_noise_sweep_imdb_genres / all_noise_sweep_amazon"
	@echo ""
	@echo "Grand test targets (full grid)"
	@echo "  grand_test                  Full grid (sequential)"
	@echo "  grand_test_resume           Resume interrupted run"
	@echo "  grand_test_agnews           Only AG News cells"
	@echo "  grand_test_imdb             Only IMDB cells"
	@echo "  grand_test_amazon           Only Amazon cells"
	@echo "  grand_test_part0/1/2/3      Quarter of the grid (for 4-GPU parallelism)"
	@echo ""
	@echo "Key variables:"
	@echo "  DEVICE=auto|cpu|cuda|cuda:N   Target device           (default: auto)"
	@echo "  NLP_SEED=N                    Seed for single-run targets (default: 0)"
	@echo "  SEEDS='0 1 2'                 Seeds for all_nlp_* targets (default: 0 1 2)"
	@echo "  GRAND_TEST_RUN_NAME=name      Output folder name for grand test"
	@echo "  GRAND_TEST_SEEDS='0 1 2'      Seeds for grand test grid"
	@echo ""

PYTHON    = uv run
DEVICE    = auto
EVAL_EVERY ?= 20
FINETUNE_BERT ?=
FINETUNE_BERT_FLAG = $(if $(FINETUNE_BERT),--finetune_bert,)

# ─── Multi-seed support ───────────────────────────────────────────────────────
# NLP_SEED  : seed used by individual targets (default 0)
# SEEDS     : list used by all_nlp_* targets to loop and aggregate
NLP_SEED  ?= 0
SEEDS     ?= 0 1 2

# ─────────────────────────────────────────────────────────────────────────────
# AG News — Semi Anti-Causal
# ─────────────────────────────────────────────────────────────────────────────
AGNEWS_CLASS_DIST_TRAIN ?= 0.25 0.25 0.25 0.25 0.25 0.25 0.25 0.25
AGNEWS_CLASS_DIST_TEST  ?= 0.25 0.25 0.25 0.25

AGNN_P_CORRECT_TRAIN ?= 0.9 0.7
AGNN_P_CORRECT_TEST  ?= 0.0
AGNN_LABEL_FLIP      ?= 0.2
AGNN_ERM_STEPS       ?= 25_000
AGNN_ERM_LR          ?= 1e-4
AGNN_IRM_STEPS       ?= 25_000
AGNN_IRM_LR          ?= 1e-4
AGNN_IRM_LAMBDA      ?= 100.0

nlp_agnews_semi_anti_causal:
	cd $(ROOT) && $(PYTHON) experiments/main.py \
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
		--seed $(NLP_SEED) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# AG News — Size Selection Bias
# ─────────────────────────────────────────────────────────────────────────────
AGNS_SIZE_P_TRAIN     ?= 0.9 0.7
AGNS_SIZE_LABEL_FLIP  ?= 0.0
AGNS_SIZE_N_OOD       ?= 250
AGNS_SIZE_THRESHOLD   ?= quartile
AGNS_SIZE_ERM_STEPS   ?= 25_000
AGNS_SIZE_ERM_LR      ?= 1e-4
AGNS_SIZE_IRM_STEPS   ?= 25_000
AGNS_SIZE_IRM_LR      ?= 1e-4
AGNS_SIZE_IRM_LAMBDA  ?= 100.0

nlp_agnews_size_selection:
	cd $(ROOT) && $(PYTHON) experiments/main.py \
		--dataset nlp_agnews_size_selection \
		--nlp_selection_p_train $(AGNS_SIZE_P_TRAIN) \
		--nlp_label_flip $(AGNS_SIZE_LABEL_FLIP) \
		--nlp_n_ood_per_class $(AGNS_SIZE_N_OOD) \
		--nlp_size_threshold_method $(AGNS_SIZE_THRESHOLD) \
		--nlp_max_length 256 \
		--nlp_agnews_class_dist_train $(AGNEWS_CLASS_DIST_TRAIN) \
		--nlp_agnews_class_dist_test $(AGNEWS_CLASS_DIST_TEST) \
		--erm_steps $(AGNS_SIZE_ERM_STEPS) \
		--erm_lr $(AGNS_SIZE_ERM_LR) \
		--irm_steps $(AGNS_SIZE_IRM_STEPS) \
		--irm_lr $(AGNS_SIZE_IRM_LR) \
		--irm_lambda $(AGNS_SIZE_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--seed $(NLP_SEED) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# AG News — Confounding varying proxy
# ─────────────────────────────────────────────────────────────────────────────
AGNN_CONF_PROXY_A_TRAIN  ?= 0.01 0.1
AGNN_CONF_PROXY_A_TEST   ?= 1.0
AGNN_CONF_PROXY_P_C_FLIP ?= 0.5
AGNN_CONF_ERM_STEPS      ?= 25_000
AGNN_CONF_ERM_LR         ?= 1e-4
AGNN_CONF_IRM_STEPS      ?= 25_000
AGNN_CONF_IRM_LR         ?= 1e-4
AGNN_CONF_IRM_LAMBDA     ?= 100.0 #50
AGNN_CONF_GAMMA          ?= 0.8

nlp_agnews_conf_varying_proxy:
	cd $(ROOT) && $(PYTHON) experiments/main.py \
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
		--seed $(NLP_SEED) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
all_nlp_agnews:
	@for seed in $(SEEDS); do \
		$(MAKE) -f $(THIS_DIR)Makefile nlp_agnews_semi_anti_causal   NLP_SEED=$$seed DEVICE=$(DEVICE) || exit 1; \
		$(MAKE) -f $(THIS_DIR)Makefile nlp_agnews_size_selection     NLP_SEED=$$seed DEVICE=$(DEVICE) || exit 1; \
		$(MAKE) -f $(THIS_DIR)Makefile nlp_agnews_conf_varying_proxy NLP_SEED=$$seed DEVICE=$(DEVICE) || exit 1; \
	done
	cd $(ROOT) && $(PYTHON) experiments/aggregate_seeds.py \
		--datasets nlp_agnews_semi_anti_causal nlp_agnews_size_selection nlp_agnews_conf_varying_proxy


# ─────────────────────────────────────────────────────────────────────────────
# IMDB Genres — Semi Anti-Causal
# ─────────────────────────────────────────────────────────────────────────────
IMDB_GENRES_CLASS_RATIO_TRAIN ?= 0.5 0.5
IMDB_GENRES_CLASS_RATIO_TEST  ?= 0.5

IMDB_GENRES_SAC_P_CORRECT_TRAIN ?= 0.9 0.7
IMDB_GENRES_SAC_P_CORRECT_TEST  ?= 0.0
IMDB_GENRES_SAC_LABEL_FLIP      ?= 0.2
IMDB_GENRES_SAC_ERM_STEPS       ?= 25_000
IMDB_GENRES_SAC_ERM_LR          ?= 1e-4
IMDB_GENRES_SAC_IRM_STEPS       ?= 25_000
IMDB_GENRES_SAC_IRM_LR          ?= 1e-4
IMDB_GENRES_SAC_IRM_LAMBDA      ?= 100.0

nlp_imdb_genres_semi_anti_causal:
	cd $(ROOT) && $(PYTHON) experiments/main.py \
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
		--seed $(NLP_SEED) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# IMDB Genres — Size Selection bias
# ─────────────────────────────────────────────────────────────────────────────
IMDB_GENRES_SIZE_P_TRAIN    ?= 0.9 0.7
IMDB_GENRES_SIZE_THRESHOLD  ?= quartile
IMDB_GENRES_SIZE_LABEL_FLIP ?= 0.2
IMDB_GENRES_SIZE_ERM_STEPS  ?= 25_000
IMDB_GENRES_SIZE_ERM_LR     ?= 1e-4
IMDB_GENRES_SIZE_IRM_STEPS  ?= 25_000
IMDB_GENRES_SIZE_IRM_LR     ?= 1e-4
IMDB_GENRES_SIZE_IRM_LAMBDA ?= 100.0

nlp_imdb_genres_size_selection:
	cd $(ROOT) && $(PYTHON) experiments/main.py \
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
		--seed $(NLP_SEED) \
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
IMDB_GENRES_CONF_IRM_LAMBDA     ?= 100.0 #50
IMDB_GENRES_CONF_GAMMA          ?= 0.5

nlp_imdb_genres_conf_varying_proxy:
	cd $(ROOT) && $(PYTHON) experiments/main.py \
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
		--seed $(NLP_SEED) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
all_nlp_imdb_genres:
	@for seed in $(SEEDS); do \
		$(MAKE) -f $(THIS_DIR)Makefile nlp_imdb_genres_semi_anti_causal   NLP_SEED=$$seed DEVICE=$(DEVICE) || exit 1; \
		$(MAKE) -f $(THIS_DIR)Makefile nlp_imdb_genres_size_selection     NLP_SEED=$$seed DEVICE=$(DEVICE) || exit 1; \
		$(MAKE) -f $(THIS_DIR)Makefile nlp_imdb_genres_conf_varying_proxy NLP_SEED=$$seed DEVICE=$(DEVICE) || exit 1; \
	done
	cd $(ROOT) && $(PYTHON) experiments/aggregate_seeds.py \
		--datasets nlp_imdb_genres_semi_anti_causal nlp_imdb_genres_size_selection nlp_imdb_genres_conf_varying_proxy


# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Semi Anti-Causal
# ─────────────────────────────────────────────────────────────────────────────
AMAZON_N_TARGET        ?= 100000

AMAZON_CLASS_RATIO_TRAIN ?= 0.5 0.5
AMAZON_CLASS_RATIO_TEST  ?= 0.5

AMAZON_SAC_P_CORRECT_TRAIN ?= 0.9 0.7
AMAZON_SAC_P_CORRECT_TEST  ?= 0.0
AMAZON_SAC_LABEL_FLIP      ?= 0.2
AMAZON_SAC_ERM_STEPS       ?= 25_000
AMAZON_SAC_ERM_LR          ?= 1e-4
AMAZON_SAC_IRM_STEPS       ?= 25_000
AMAZON_SAC_IRM_LR          ?= 1e-4
AMAZON_SAC_IRM_LAMBDA      ?= 100

nlp_amazon_semi_anti_causal:
	cd $(ROOT) && $(PYTHON) experiments/main.py \
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
		--seed $(NLP_SEED) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Sentiment Selection (usefulness as a misleading proxy)
# ─────────────────────────────────────────────────────────────────────────────
# Typical : useful + positive (5★), OR useless + negative (1★)
# OOD     : useful + negative (1★), OR useless + positive (5★)
# The model learns "sentiment → usefulness", a shortcut that breaks at OOD test
AMAZON_SENT_P_TRAIN    ?= 0.9 0.7 
AMAZON_SENT_LABEL_FLIP ?= 0.0
AMAZON_SENT_ERM_STEPS  ?= 25_000
AMAZON_SENT_ERM_LR     ?= 1e-4
AMAZON_SENT_IRM_STEPS  ?= 25_000
AMAZON_SENT_IRM_LR     ?= 1e-4
AMAZON_SENT_IRM_LAMBDA ?= 100.0

nlp_amazon_sentiment_selection:
	cd $(ROOT) && $(PYTHON) experiments/main.py \
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
		--seed $(NLP_SEED) \
		--device $(DEVICE)
		
# ─────────────────────────────────────────────────────────────────────────────
# Amazon Books — Confounding varying proxy
# ─────────────────────────────────────────────────────────────────────────────
AMAZON_CONF_PROXY_A_TRAIN  ?= 0.01 0.1
AMAZON_CONF_PROXY_A_TEST   ?= 1.0
AMAZON_CONF_PROXY_P_C_FLIP ?= 0.5
AMAZON_CONF_ERM_STEPS      ?= 25_000
AMAZON_CONF_ERM_LR         ?= 1e-4
AMAZON_CONF_IRM_STEPS      ?= 25_000
AMAZON_CONF_IRM_LR         ?= 1e-4
AMAZON_CONF_IRM_LAMBDA     ?= 100.0 #50
AMAZON_CONF_GAMMA          ?= 0.8

nlp_amazon_conf_varying_proxy:
	cd $(ROOT) && $(PYTHON) experiments/main.py \
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
		--seed $(NLP_SEED) \
		--device $(DEVICE)

all_nlp_amazon:
	@for seed in $(SEEDS); do \
		$(MAKE) -f $(THIS_DIR)Makefile nlp_amazon_semi_anti_causal      NLP_SEED=$$seed DEVICE=$(DEVICE) || exit 1; \
		$(MAKE) -f $(THIS_DIR)Makefile nlp_amazon_conf_varying_proxy    NLP_SEED=$$seed DEVICE=$(DEVICE) || exit 1; \
		$(MAKE) -f $(THIS_DIR)Makefile nlp_amazon_sentiment_selection   NLP_SEED=$$seed DEVICE=$(DEVICE) || exit 1; \
	done
	cd $(ROOT) && $(PYTHON) experiments/aggregate_seeds.py \
		--datasets nlp_amazon_semi_anti_causal nlp_amazon_conf_varying_proxy nlp_amazon_sentiment_selection


# =============================================================================
# Gap sweep NLP — sweep of the spurious correlation strength
# =============================================================================

# ── AG News ──────────────────────────────────────────────────────────────────
gap_sweep_agnews_sac:
	cd $(ROOT) && $(PYTHON) experiments/run_gap_sweep.py \
		--dataset nlp_agnews_semi_anti_causal \
		--nlp_max_length 256 \
		--nlp_label_flip $(AGNN_LABEL_FLIP) \
		--nlp_agnews_class_dist_train $(AGNEWS_CLASS_DIST_TRAIN) \
		--nlp_agnews_class_dist_test $(AGNEWS_CLASS_DIST_TEST) \
		--erm_steps $(AGNN_ERM_STEPS) --erm_lr $(AGNN_ERM_LR) \
		--irm_steps $(AGNN_IRM_STEPS) --irm_lr $(AGNN_IRM_LR) --irm_lambda $(AGNN_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

gap_sweep_agnews_size:
	cd $(ROOT) && $(PYTHON) experiments/run_gap_sweep.py \
		--dataset nlp_agnews_size_selection \
		--nlp_max_length 256 \
		--nlp_label_flip $(AGNS_SIZE_LABEL_FLIP) \
		--nlp_size_threshold_method $(AGNS_SIZE_THRESHOLD) \
		--nlp_agnews_class_dist_train $(AGNEWS_CLASS_DIST_TRAIN) \
		--nlp_agnews_class_dist_test $(AGNEWS_CLASS_DIST_TEST) \
		--erm_steps $(AGNS_SIZE_ERM_STEPS) --erm_lr $(AGNS_SIZE_ERM_LR) \
		--irm_steps $(AGNS_SIZE_IRM_STEPS) --irm_lr $(AGNS_SIZE_IRM_LR) --irm_lambda $(AGNS_SIZE_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

gap_sweep_agnews_conf:
	cd $(ROOT) && $(PYTHON) experiments/run_gap_sweep.py \
		--dataset nlp_agnews_conf_varying_proxy \
		--nlp_max_length 256 \
		--nlp_conf_p_c_flip $(AGNN_CONF_PROXY_P_C_FLIP) \
		--nlp_conf_gamma $(AGNN_CONF_GAMMA) \
		--erm_steps $(AGNN_CONF_ERM_STEPS) --erm_lr $(AGNN_CONF_ERM_LR) \
		--irm_steps $(AGNN_CONF_IRM_STEPS) --irm_lr $(AGNN_CONF_IRM_LR) --irm_lambda $(AGNN_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)


# ── IMDB Genres ────────────────────────────────────────────────────────
gap_sweep_imdb_genres_sac:
	cd $(ROOT) && $(PYTHON) experiments/run_gap_sweep.py \
		--dataset nlp_imdb_genres_semi_anti_causal \
		--nlp_max_length 256 \
		--nlp_label_flip 0.25 \
		--erm_steps 25_000 --erm_lr 5e-4 \
		--irm_steps 25_000 --irm_lr 5e-4 --irm_lambda 100.0 \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

gap_sweep_imdb_genres_size_selection:
	cd $(ROOT) && $(PYTHON) experiments/run_gap_sweep.py \
		--dataset nlp_imdb_genres_size_selection \
		--nlp_max_length 256 \
		--nlp_size_threshold_method quartile \
		--nlp_label_flip 0.0 \
		--erm_steps 25_000 --erm_lr 5e-4 \
		--irm_steps 25_000 --irm_lr 5e-4 --irm_lambda 100.0 \
		--eval_every $(EVAL_EVERY) --seeds 0 --device $(DEVICE)

gap_sweep_imdb_genres_conf:
	cd $(ROOT) && $(PYTHON) experiments/run_gap_sweep.py \
		--dataset nlp_imdb_genres_conf_varying_proxy \
		--nlp_conf_p_c_flip $(IMDB_GENRES_CONF_PROXY_P_C_FLIP) \
		--nlp_max_length 256 \
		--nlp_conf_gamma $(IMDB_GENRES_CONF_GAMMA) \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 --irm_lambda 100.0 \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

# ── Amazon ────────────────────────────────────────────────────────────────────
gap_sweep_amazon_sac:
	cd $(ROOT) && $(PYTHON) experiments/run_gap_sweep.py \
		--dataset nlp_amazon_semi_anti_causal \
		--nlp_max_length 512 \
		--nlp_label_flip $(AMAZON_SAC_LABEL_FLIP) \
		--nlp_amazon_n_target $(AMAZON_N_TARGET) \
		--erm_steps $(AMAZON_SAC_ERM_STEPS) --erm_lr $(AMAZON_SAC_ERM_LR) \
		--irm_steps $(AMAZON_SAC_IRM_STEPS) --irm_lr $(AMAZON_SAC_IRM_LR) --irm_lambda $(AMAZON_SAC_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

gap_sweep_amazon_sentiment_selection:
	cd $(ROOT) && $(PYTHON) experiments/run_gap_sweep.py \
		--dataset nlp_amazon_sentiment_selection \
		--nlp_max_length 512 \
		--nlp_label_flip $(AMAZON_SENT_LABEL_FLIP) \
		--nlp_amazon_n_target $(AMAZON_N_TARGET) \
		--erm_steps $(AMAZON_SENT_ERM_STEPS) --erm_lr $(AMAZON_SENT_ERM_LR) \
		--irm_steps $(AMAZON_SENT_IRM_STEPS) --irm_lr $(AMAZON_SENT_IRM_LR) --irm_lambda $(AMAZON_SENT_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

gap_sweep_amazon_conf:
	cd $(ROOT) && $(PYTHON) experiments/run_gap_sweep.py \
		--dataset nlp_amazon_conf_varying_proxy \
		--nlp_max_length 512 \
		--nlp_conf_p_c_flip $(AMAZON_CONF_PROXY_P_C_FLIP) \
		--nlp_amazon_n_target $(AMAZON_N_TARGET) \
		--nlp_conf_gamma $(AMAZON_CONF_GAMMA) \
		--erm_steps $(AMAZON_CONF_ERM_STEPS) --erm_lr $(AMAZON_CONF_ERM_LR) \
		--irm_steps $(AMAZON_CONF_IRM_STEPS) --irm_lr $(AMAZON_CONF_IRM_LR) --irm_lambda $(AMAZON_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)


# =============================================================================
# Noise sweep NLP — sweep of noise (label_flip or gamma) at fixed gap
# =============================================================================
# For semi_anti_causal / selection : noise = label_flip (0 → 0.5)
# For conf_varying_proxy           : noise = gamma      (0 → 1.0)

# ── AG News ──────────────────────────────────────────────────────────────────
noise_sweep_agnews_sac:
	cd $(ROOT) && $(PYTHON) experiments/run_noise_sweep.py \
		--dataset nlp_agnews_semi_anti_causal \
		--nlp_max_length 256 \
		--nlp_agnews_class_dist_train $(AGNEWS_CLASS_DIST_TRAIN) \
		--nlp_agnews_class_dist_test $(AGNEWS_CLASS_DIST_TEST) \
		--erm_steps $(AGNN_ERM_STEPS) --erm_lr $(AGNN_ERM_LR) \
		--irm_steps $(AGNN_IRM_STEPS) --irm_lr $(AGNN_IRM_LR) --irm_lambda $(AGNN_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

noise_sweep_agnews_size:
	cd $(ROOT) && $(PYTHON) experiments/run_noise_sweep.py \
		--dataset nlp_agnews_size_selection \
		--nlp_max_length 256 \
		--nlp_size_threshold_method $(AGNS_SIZE_THRESHOLD) \
		--nlp_agnews_class_dist_train $(AGNEWS_CLASS_DIST_TRAIN) \
		--nlp_agnews_class_dist_test $(AGNEWS_CLASS_DIST_TEST) \
		--p1 $(word 1,$(AGNS_SIZE_P_TRAIN)) \
		--p2 $(word 2,$(AGNS_SIZE_P_TRAIN)) \
		--erm_steps $(AGNS_SIZE_ERM_STEPS) --erm_lr $(AGNS_SIZE_ERM_LR) \
		--irm_steps $(AGNS_SIZE_IRM_STEPS) --irm_lr $(AGNS_SIZE_IRM_LR) --irm_lambda $(AGNS_SIZE_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 --device $(DEVICE)

noise_sweep_agnews_conf:
	cd $(ROOT) && $(PYTHON) experiments/run_noise_sweep.py \
		--dataset nlp_agnews_conf_varying_proxy \
		--nlp_max_length 256 \
		--nlp_conf_p_c_flip $(AGNN_CONF_PROXY_P_C_FLIP) \
		--erm_steps $(AGNN_CONF_ERM_STEPS) --erm_lr $(AGNN_CONF_ERM_LR) \
		--irm_steps $(AGNN_CONF_IRM_STEPS) --irm_lr $(AGNN_CONF_IRM_LR) --irm_lambda $(AGNN_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)


# ── IMDB Genres ──────────────────────────────────────────────────────────────
noise_sweep_imdb_genres_sac:
	cd $(ROOT) && $(PYTHON) experiments/run_noise_sweep.py \
		--dataset nlp_imdb_genres_semi_anti_causal \
		--nlp_max_length 256 \
		--erm_steps 25_000 --erm_lr 5e-4 \
		--irm_steps 25_000 --irm_lr 5e-4 --irm_lambda 100.0 \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

noise_sweep_imdb_genres_size_selection:
	cd $(ROOT) && $(PYTHON) experiments/run_noise_sweep.py \
		--dataset nlp_imdb_genres_size_selection \
		--nlp_max_length 256 \
		--nlp_size_threshold_method quartile \
		--p1 $(word 1,$(IMDB_GENRES_SIZE_P_TRAIN)) \
		--p2 $(word 2,$(IMDB_GENRES_SIZE_P_TRAIN)) \
		--erm_steps 25_000 --erm_lr $(IMDB_GENRES_SIZE_ERM_LR) \
		--irm_steps 25_000 --irm_lr $(IMDB_GENRES_SIZE_IRM_LR) --irm_lambda 100 \
		--eval_every $(EVAL_EVERY) --seeds 0 --device $(DEVICE)

noise_sweep_imdb_genres_conf:
	cd $(ROOT) && $(PYTHON) experiments/run_noise_sweep.py \
		--dataset nlp_imdb_genres_conf_varying_proxy \
		--nlp_conf_p_c_flip $(IMDB_GENRES_CONF_PROXY_P_C_FLIP) \
		--nlp_max_length 256 \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 --irm_lambda 50.0 \
		--eval_every $(EVAL_EVERY) --seeds 0 --device $(DEVICE)


# ── Amazon ────────────────────────────────────────────────────────────────────
noise_sweep_amazon_sac:
	cd $(ROOT) && $(PYTHON) experiments/run_noise_sweep.py \
		--dataset nlp_amazon_semi_anti_causal \
		--nlp_max_length 512 \
		--nlp_amazon_n_target $(AMAZON_N_TARGET) \
		--erm_steps $(AMAZON_SAC_ERM_STEPS) --erm_lr $(AMAZON_SAC_ERM_LR) \
		--irm_steps $(AMAZON_SAC_IRM_STEPS) --irm_lr $(AMAZON_SAC_IRM_LR) --irm_lambda $(AMAZON_SAC_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

noise_sweep_amazon_sentiment_selection:
	cd $(ROOT) && $(PYTHON) experiments/run_noise_sweep.py \
		--dataset nlp_amazon_sentiment_selection \
		--nlp_max_length 512 \
		--nlp_amazon_n_target $(AMAZON_N_TARGET) \
		--erm_steps $(AMAZON_SENT_ERM_STEPS) --erm_lr $(AMAZON_SENT_ERM_LR) \
		--irm_steps $(AMAZON_SENT_IRM_STEPS) --irm_lr $(AMAZON_SENT_IRM_LR) --irm_lambda 100 \
		--eval_every $(EVAL_EVERY) --seeds 0 --device $(DEVICE)

noise_sweep_amazon_conf:
	cd $(ROOT) && $(PYTHON) experiments/run_noise_sweep.py \
		--dataset nlp_amazon_conf_varying_proxy \
		--nlp_max_length 512 \
		--nlp_conf_p_c_flip $(AMAZON_CONF_PROXY_P_C_FLIP) \
		--nlp_amazon_n_target $(AMAZON_N_TARGET) \
		--erm_steps $(AMAZON_CONF_ERM_STEPS) --erm_lr $(AMAZON_CONF_ERM_LR) \
		--irm_steps $(AMAZON_CONF_IRM_STEPS) --irm_lr $(AMAZON_CONF_IRM_LR) --irm_lambda $(AMAZON_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) --seeds 0 1 2 --device $(DEVICE)

all_noise_sweep_amazon: noise_sweep_amazon_sac noise_sweep_amazon_sentiment_selection noise_sweep_amazon_conf
all_noise_sweep_imdb_genres: noise_sweep_imdb_genres_sac noise_sweep_imdb_genres_size_selection noise_sweep_imdb_genres_conf
all_noise_sweep_agnews: noise_sweep_agnews_sac noise_sweep_agnews_size noise_sweep_agnews_conf


# =============================================================================
# Grand test — 3 datasets × 3 mechanisms × 4 noise levels × 2 correlations (72 runs)
# =============================================================================
# Grid:
#   Noise (label_flip / gamma)       : 0.0  0.1  0.2  0.25
#   Spurious correlation SAC/selection : p=[1.0,0.8]  p=[0.9,0.7]
#   Spurious correlation confounder    : a=[0.01,0.1] a=[0.05,0.15]
#
# Useful options:
#   make grand_test DEVICE=cuda:1
#   make grand_test GRAND_TEST_RUN_NAME=my_run
#   make grand_test GRAND_TEST_SEEDS="0 1 2"
#   make grand_test GRAND_TEST_DATASETS="nlp_agnews_semi_anti_causal nlp_imdb_genres_conf_varying_proxy"
#   make grand_test_resume    # resume an interrupted run (same run_name)

GRAND_TEST_RUN_NAME ?=
GRAND_TEST_SEEDS    ?= 0
GRAND_TEST_DATASETS ?=

# Fallback timestamp if GRAND_TEST_RUN_NAME is empty (even if explicitly passed empty)
_GT_RUN_NAME = $(or $(GRAND_TEST_RUN_NAME),$(shell date +%Y%m%d_%H%M%S))

_GRAND_TEST_DATASETS_FLAG = $(if $(GRAND_TEST_DATASETS),--datasets $(GRAND_TEST_DATASETS),)

grand_test:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--seeds $(GRAND_TEST_SEEDS) \
		$(_GRAND_TEST_DATASETS_FLAG) \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

grand_test_resume:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--seeds $(GRAND_TEST_SEEDS) \
		$(_GRAND_TEST_DATASETS_FLAG) \
		--resume \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ── Split into 3 for parallelisation (same run_name → same output folder) ───────
# Launch on 3 different tmux terminals with the same GRAND_TEST_RUN_NAME:
#   make grand_test_agnews  GRAND_TEST_RUN_NAME=myrun DEVICE=cuda:0 &
#   make grand_test_imdb    GRAND_TEST_RUN_NAME=myrun DEVICE=cuda:0 &
#   make grand_test_amazon  GRAND_TEST_RUN_NAME=myrun DEVICE=cuda:0 &

_AGNEWS_DATASETS  = nlp_agnews_semi_anti_causal nlp_agnews_size_selection nlp_agnews_conf_varying_proxy
_IMDB_DATASETS    = nlp_imdb_genres_semi_anti_causal nlp_imdb_genres_size_selection nlp_imdb_genres_conf_varying_proxy
_AMAZON_DATASETS  = nlp_amazon_semi_anti_causal nlp_amazon_sentiment_selection nlp_amazon_conf_varying_proxy

grand_test_agnews:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_AGNEWS_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

grand_test_imdb:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_IMDB_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

grand_test_amazon:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_AMAZON_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

grand_test_resume_agnews:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_AGNEWS_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) --resume \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

grand_test_resume_imdb:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_IMDB_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) --resume \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

grand_test_resume_amazon:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_AMAZON_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) --resume \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ── Split into 4 for parallelisation of SAC + Selection (no confounder) ───────
# 48 cells / 4 = 12 per part (round-robin: each part contains a balanced mix
# of the 3 datasets and 2 mechanisms).
# Launch on 4 tmux terminals with the same GRAND_TEST_RUN_NAME and different GPUs:
#   make grand_test_part0 GRAND_TEST_RUN_NAME=myrun DEVICE=cuda:0
#   make grand_test_part1 GRAND_TEST_RUN_NAME=myrun DEVICE=cuda:1
#   make grand_test_part2 GRAND_TEST_RUN_NAME=myrun DEVICE=cuda:2
#   make grand_test_part3 GRAND_TEST_RUN_NAME=myrun DEVICE=cuda:3

_SAC_SEL_DATASETS = nlp_agnews_semi_anti_causal nlp_agnews_size_selection \
                    nlp_imdb_genres_semi_anti_causal nlp_imdb_genres_size_selection \
                    nlp_amazon_semi_anti_causal nlp_amazon_sentiment_selection

grand_test_part0:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_SAC_SEL_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) \
		--n_parts 4 --part 0 \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

grand_test_part1:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_SAC_SEL_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) \
		--n_parts 4 --part 1 \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

grand_test_part2:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_SAC_SEL_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) \
		--n_parts 4 --part 2 \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

grand_test_part3:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_SAC_SEL_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) \
		--n_parts 4 --part 3 \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

grand_test_resume_part0:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_SAC_SEL_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) --resume \
		--n_parts 4 --part 0 \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

grand_test_resume_part1:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_SAC_SEL_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) --resume \
		--n_parts 4 --part 1 \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

grand_test_resume_part2:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_SAC_SEL_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) --resume \
		--n_parts 4 --part 2 \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

grand_test_resume_part3:
	cd $(ROOT) && $(PYTHON) experiments/run_grand_test.py \
		--run_name $(_GT_RUN_NAME) \
		--datasets $(_SAC_SEL_DATASETS) \
		--seeds $(GRAND_TEST_SEEDS) --resume \
		--n_parts 4 --part 3 \
		--erm_steps 25_000 --erm_lr 1e-4 \
		--irm_steps 25_000 --irm_lr 1e-4 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)
