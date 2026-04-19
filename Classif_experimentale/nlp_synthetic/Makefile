.PHONY: help \
        nlp_sms_spam nlp_sms_spam_size_selection \
        nlp_sms_spam_conf_varying_proxy nlp_sms_spam_conf_varying_gamma nlp_sms_spam_conf_varying_pc \
        nlp_agnews_semi_anti_causal nlp_agnews_source_selection nlp_agnews_conf_varying_proxy \
        nlp_sst2_semi_anti_causal nlp_sst2_selection nlp_sst2_genre_selection nlp_sst2_conf_varying_proxy \
        all all_sst2 all_conf

THIS_DIR  = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT      = $(abspath $(THIS_DIR)/..)

PYTHON    = uv run
DEVICE    = auto
EVAL_EVERY ?= 20

# ─────────────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "=== NLP Experiments ==="
	@echo ""
	@echo "  --- CAS CAUSAL (X → Y) ---"
	@echo "  SMS Spam:"
	@echo "    make nlp_sms_spam                      → Semi-anti-causal (token injection)"
	@echo "    make nlp_sms_spam_size_selection       → Selection bias by message length"
	@echo "    make nlp_sms_spam_conf_varying_proxy   → Confounding (proxy noise varies)"
	@echo "    make nlp_sms_spam_conf_varying_gamma   → Confounding (C→Y strength varies)"
	@echo "    make nlp_sms_spam_conf_varying_pc      → Confounding (prevalence of C varies)"
	@echo ""
	@echo "  AG News:"
	@echo "    make nlp_agnews_semi_anti_causal → Semi-anti-causal (token injection)"
	@echo "    make nlp_agnews_source_selection → Selection bias by news source"
	@echo ""
	@echo "  --- CAS ANTI-CAUSAL (Y → X) ---"
	@echo "  SST-2 (sentiment de film, Rotten Tomatoes) :"
	@echo "    make nlp_sst2_semi_anti_causal   → Semi-anti-causal (token injection)"
	@echo "    make nlp_sst2_selection          → Selection bias by sentiment lexicon"
	@echo ""
	@echo "  make all                           → all 4 causal NLP experiments"
	@echo "  make all_conf                      → all 3 confounding SMS Spam experiments"
	@echo "  make all_sst2                      → both anti-causal SST-2 experiments"
	@echo ""
	@echo "  Plots saved to nlp_synthetic/plots/<dataset>/"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# SMS Spam — Semi Anti-Causal
# ─────────────────────────────────────────────────────────────────────────────
SAC_P_CORRECT_TRAIN ?= 0.9 0.8
SAC_P_CORRECT_TEST  ?= 0.0
SAC_LABEL_FLIP      ?= 0.2
SAC_ERM_STEPS       ?= 25_000
SAC_ERM_LR          ?= 5e-4
SAC_IRM_STEPS       ?= 25_000
SAC_IRM_LR          ?= 5e-4
SAC_IRM_LAMBDA      ?= 100.0

nlp_sms_spam_semi_anti_causal:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_sms_spam \
		--nlp_p_correct_train $(SAC_P_CORRECT_TRAIN) \
		--nlp_p_correct_test $(SAC_P_CORRECT_TEST) \
		--nlp_label_flip $(SAC_LABEL_FLIP) \
		--erm_steps $(SAC_ERM_STEPS) \
		--erm_lr $(SAC_ERM_LR) \
		--irm_steps $(SAC_IRM_STEPS) \
		--irm_lr $(SAC_IRM_LR) \
		--irm_lambda $(SAC_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# SMS Spam — Size Selection Bias
# ─────────────────────────────────────────────────────────────────────────────
SEL_P_TRAIN    ?= 0.9 0.8
SEL_THRESHOLD  ?= quartile
SEL_LABEL_FLIP ?= 0.0
SEL_ERM_STEPS  ?= 25_000
SEL_ERM_LR     ?= 5e-4
SEL_IRM_STEPS  ?= 25_000
SEL_IRM_LR     ?= 5e-4
SEL_IRM_LAMBDA ?= 400.0

nlp_sms_spam_size_selection:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_sms_spam_size_selection \
		--nlp_selection_p_train $(SEL_P_TRAIN) \
		--nlp_size_threshold_method $(SEL_THRESHOLD) \
		--nlp_label_flip $(SEL_LABEL_FLIP) \
		--erm_steps $(SEL_ERM_STEPS) \
		--erm_lr $(SEL_ERM_LR) \
		--irm_steps $(SEL_IRM_STEPS) \
		--irm_lr $(SEL_IRM_LR) \
		--irm_lambda $(SEL_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# SMS Spam — Confounding (3 variants)
# ─────────────────────────────────────────────────────────────────────────────
# DAG commun : C → Z → token ; C → Y (flip gamma) ; text → Y
# Variant 1 : bruit proxy a_e varie (a ≈ 0 en train → a ≈ 1 en test OOD)
# Variant 2 : force C→Y gamma_e varie (gamma > 0 en train → gamma=0 en OOD)
# Variant 3 : prévalence p_e varie (forte en train → faible en OOD)

CONF_ERM_STEPS   ?= 25000
CONF_ERM_LR      ?= 1e-4
CONF_IRM_STEPS   ?= 25000
CONF_IRM_LR      ?= 1e-4
CONF_IRM_LAMBDA  ?= 150.0

# ── Variant 1 : varying proxy ──
CONF_PROXY_A_TRAIN  ?= 0.01 0.1
CONF_PROXY_A_TEST   ?= 1.0
CONF_PROXY_GAMMA    ?= 0.5

nlp_sms_spam_conf_varying_proxy:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_sms_spam_conf_varying_proxy \
		--nlp_conf_a_train $(CONF_PROXY_A_TRAIN) \
		--nlp_conf_a_test $(CONF_PROXY_A_TEST) \
		--nlp_conf_gamma $(CONF_PROXY_GAMMA) \
		--erm_steps $(CONF_ERM_STEPS) \
		--erm_lr $(CONF_ERM_LR) \
		--irm_steps $(CONF_IRM_STEPS) \
		--irm_lr $(CONF_IRM_LR) \
		--irm_lambda $(CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# AG News — Semi Anti-Causal
# ─────────────────────────────────────────────────────────────────────────────
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
AGNS_LABEL_FLIP  ?= 0.0
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
AGNN_CONF_PROXY_GAMMA    ?= 0.5
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
		--nlp_conf_gamma $(AGNN_CONF_PROXY_GAMMA) \
		--nlp_max_length 256 \
		--erm_steps $(AGNN_CONF_ERM_STEPS) \
		--erm_lr $(AGNN_CONF_ERM_LR) \
		--irm_steps $(AGNN_CONF_IRM_STEPS) \
		--irm_lr $(AGNN_CONF_IRM_LR) \
		--irm_lambda $(AGNN_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
# SST-2 — Semi Anti-Causal (token injection, anti-causal Y → X)
# ─────────────────────────────────────────────────────────────────────────────
# Correspondance avec la littérature IRM :
#   Rosenfeld et al. (2021) "An Empirical Study of IRM"
#   Ahmad & Bhatt "Environment Agnostic IRM for Sequential Datasets"
# Envs train : p_correct=0.9 et 0.8 (forte corrélation Z–Y)
# Test OOD   : p_correct=0.0 (corrélation totalement inversée)

SST2_SAC_P_CORRECT_TRAIN ?= 0.9 0.8
SST2_SAC_P_CORRECT_TEST  ?= 0.0
SST2_SAC_LABEL_FLIP      ?= 0.2
SST2_SAC_ERM_STEPS       ?= 25000
SST2_SAC_ERM_LR          ?= 5e-4
SST2_SAC_IRM_STEPS       ?= 25000
SST2_SAC_IRM_LR          ?= 5e-4
SST2_SAC_IRM_LAMBDA      ?= 150.0

nlp_sst2_semi_anti_causal:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_sst2_semi_anti_causal \
		--nlp_p_correct_train $(SST2_SAC_P_CORRECT_TRAIN) \
		--nlp_p_correct_test $(SST2_SAC_P_CORRECT_TEST) \
		--nlp_label_flip $(SST2_SAC_LABEL_FLIP) \
		--erm_steps $(SST2_SAC_ERM_STEPS) \
		--erm_lr $(SST2_SAC_ERM_LR) \
		--irm_steps $(SST2_SAC_IRM_STEPS) \
		--irm_lr $(SST2_SAC_IRM_LR) \
		--irm_lambda $(SST2_SAC_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE) \

# ─────────────────────────────────────────────────────────────────────────────
# SST-2 — Selection bias (lexique de sentiment fort vs. critiques subtiles)
# ─────────────────────────────────────────────────────────────────────────────
# Typique  = review avec marqueur lexical fort (ex : "wonderful", "terrible")
# OOD cross_label : mots forts contredisent le label (adversarial pour ERM)
# OOD atypical   : review sans aucun marqueur lexical

SST2_SEL_P_TRAIN    ?= 0.9 0.8
SST2_SEL_OOD_STRAT  ?= cross_label
SST2_SEL_LABEL_FLIP ?= 0.0
SST2_SEL_ERM_STEPS  ?= 25000 
SST2_SEL_ERM_LR     ?= 5e-4
SST2_SEL_IRM_STEPS  ?= 25000
SST2_SEL_IRM_LR     ?= 5e-4
SST2_SEL_IRM_LAMBDA ?= 75.0

nlp_sst2_selection:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_sst2_selection \
		--nlp_selection_p_train $(SST2_SEL_P_TRAIN) \
		--nlp_sst2_ood_strategy $(SST2_SEL_OOD_STRAT) \
		--nlp_label_flip $(SST2_SEL_LABEL_FLIP) \
		--erm_steps $(SST2_SEL_ERM_STEPS) \
		--erm_lr $(SST2_SEL_ERM_LR) \
		--irm_steps $(SST2_SEL_IRM_STEPS) \
		--irm_lr $(SST2_SEL_IRM_LR) \
		--irm_lambda $(SST2_SEL_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# SST-2 — Selection bias par genre de film (documentaire/indie vs suite/remake)
# ─────────────────────────────────────────────────────────────────────────────
# Typique positif  : critique positive + vocabulaire doc/indie/arthouse
# Typique négatif  : critique négative + vocabulaire suite/remake/reboot
# OOD cross_genre  : critique positive d’une suite (bonne suite)
#                    critique négative d’un doc  (mauvais documentaire)

SST2_GENRE_P_TRAIN    ?= 0.9 0.7
SST2_GENRE_LABEL_FLIP ?= 0.0
SST2_GENRE_ERM_STEPS  ?= 25000
SST2_GENRE_ERM_LR     ?= 5e-4
SST2_GENRE_IRM_STEPS  ?= 25000
SST2_GENRE_IRM_LR     ?= 5e-4
SST2_GENRE_IRM_LAMBDA ?= 350.0

nlp_sst2_genre_selection:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_sst2_genre_selection \
		--nlp_selection_p_train $(SST2_GENRE_P_TRAIN) \
		--nlp_label_flip $(SST2_GENRE_LABEL_FLIP) \
		--erm_steps $(SST2_GENRE_ERM_STEPS) \
		--erm_lr $(SST2_GENRE_ERM_LR) \
		--irm_steps $(SST2_GENRE_IRM_STEPS) \
		--irm_lr $(SST2_GENRE_IRM_LR) \
		--irm_lambda $(SST2_GENRE_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ─────────────────────────────────────────────────────────────────────────────
# SST-2 — Confounding variant 1 : varying proxy
# ─────────────────────────────────────────────────────────────────────────────
SST2_CONF_PROXY_A_TRAIN  ?= 0.01 0.1
SST2_CONF_PROXY_A_TEST   ?= 1.0
SST2_CONF_PROXY_GAMMA    ?= 0.5
SST2_CONF_ERM_STEPS      ?= 25000
SST2_CONF_ERM_LR         ?= 5e-4
SST2_CONF_IRM_STEPS      ?= 25000
SST2_CONF_IRM_LR         ?= 5e-4
SST2_CONF_IRM_LAMBDA     ?= 150.0

nlp_sst2_conf_varying_proxy:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_sst2_conf_varying_proxy \
		--nlp_conf_a_train $(SST2_CONF_PROXY_A_TRAIN) \
		--nlp_conf_a_test $(SST2_CONF_PROXY_A_TEST) \
		--nlp_conf_gamma $(SST2_CONF_PROXY_GAMMA) \
		--erm_steps $(SST2_CONF_ERM_STEPS) \
		--erm_lr $(SST2_CONF_ERM_LR) \
		--irm_steps $(SST2_CONF_IRM_STEPS) \
		--irm_lr $(SST2_CONF_IRM_LR) \
		--irm_lambda $(SST2_CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)


# ─────────────────────────────────────────────────────────────────────────────
#all: nlp_sms_spam_conf_varying_proxy nlp_agnews_conf_varying_proxy nlp_sst2_conf_varying_proxy
all: nlp_sms_spam_size_selection nlp_sst2_selection
# nlp_sms_spam_semi_anti_causal nlp_agnews_semi_anti_causal nlp_agnews_source_selection nlp_sst2_semi_anti_causal

# ── Variant 2 : varying gamma ──
CONF_GAMMA_TRAIN    ?= 0.8 0.5
CONF_GAMMA_TEST     ?= 0.0
CONF_GAMMA_A        ?= 0.0

nlp_sms_spam_conf_varying_gamma:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_sms_spam_conf_varying_gamma \
		--nlp_conf_gamma_train $(CONF_GAMMA_TRAIN) \
		--nlp_conf_gamma_test $(CONF_GAMMA_TEST) \
		--nlp_conf_a $(CONF_GAMMA_A) \
		--erm_steps $(CONF_ERM_STEPS) \
		--erm_lr $(CONF_ERM_LR) \
		--irm_steps $(CONF_IRM_STEPS) \
		--irm_lr $(CONF_IRM_LR) \
		--irm_lambda $(CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

# ── Variant 3 : varying pc ──
CONF_PC_TRAIN       ?= 0.8 0.5
CONF_PC_TEST        ?= 0.1
CONF_PC_A           ?= 0.0
CONF_PC_GAMMA       ?= 0.5

nlp_sms_spam_conf_varying_pc:
	cd $(ROOT) && $(PYTHON) nlp_synthetic/main_nlp.py \
		--dataset nlp_sms_spam_conf_varying_pc \
		--nlp_conf_pc_train $(CONF_PC_TRAIN) \
		--nlp_conf_pc_test $(CONF_PC_TEST) \
		--nlp_conf_a $(CONF_PC_A) \
		--nlp_conf_gamma $(CONF_PC_GAMMA) \
		--erm_steps $(CONF_ERM_STEPS) \
		--erm_lr $(CONF_ERM_LR) \
		--irm_steps $(CONF_IRM_STEPS) \
		--irm_lr $(CONF_IRM_LR) \
		--irm_lambda $(CONF_IRM_LAMBDA) \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)