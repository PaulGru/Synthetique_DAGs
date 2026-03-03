.PHONY: help semi-anti-causal confounding selection confounding_yz selection_yz gap_sweep_sac nlp nlp_custom_confounding all

# Variables
PYTHON = uv run
N = 200000
N_TEST = 10000
VAL_FRAC = 0.1
ERM_STEPS = 25000
ERM_LR = 5e-3
IRM_STEPS = 25000
IRM_LR = 5e-3
MODEL_KIND = logreg
DEVICE = auto

help:
	@echo "Expériences disponibles:"
	@echo "  make semi-anti-causal       - Lance l'expérience semi-anti-causal"
	@echo "  make confounding            - Lance l'expérience confounding (simple)"
	@echo "  make confounding_yz         - Lance confounding + X^perp_{Y,Z} (variable mixte)"
	@echo "  make selection              - Lance l'expérience selection (simple)"
	@echo "  make selection_yz           - Lance selection + X^perp_{Y,Z} (variable mixte)"
	@echo "  make gap_sweep_sac          - Sweep du gap entre envs (semi-anti-causal)"
	@echo "  make nlp                    - Lance l'expérience NLP (semi-anti-causal)"
	@echo "  make nlp_custom_confounding - Lance NLP avec le mécanisme custom confounding"
	@echo "  make all                    - Lance les 3 expériences de base"

semi-anti-causal:
	$(PYTHON) main.py \
		--dataset synthetic_semi_anti_causal \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--ps_train 0.1 0.2 \
		--p_test_ood 1.0 \
		--label_flip 0.25 \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 750.0 \
		--model_kind $(MODEL_KIND) \
		--eval_every 10 \
		--device $(DEVICE)

confounding:
	$(PYTHON) main.py \
		--dataset synthetic_confounding \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--conf_a_train 0.01 0.11 \
		--conf_a_test 0.99 \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 250.0 \
		--model_kind $(MODEL_KIND) \
		--eval_every 10 \
		--device $(DEVICE)

custom_confounding:
	$(PYTHON) main.py \
		--dataset synthetic_custom_confounding \
		--n $(N) \
		--val_frac $(VAL_FRAC) \
		--custom_conf_a_train 0.1 0.2 \
		--custom_conf_a_test 1.0 \
		--custom_conf_alpha 1.0 \
		--erm_steps 15000 \
		--erm_lr $(ERM_LR) \
		--irm_steps 15000 \
		--irm_lr $(IRM_LR) \
		--irm_lambda 250.0 \
		--model_kind $(MODEL_KIND) \
		--eval_every 10 \
		--device $(DEVICE)

selection:
	$(PYTHON) main.py \
		--dataset synthetic_selection \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--sel_alpha_train 0.9 0.8 \
		--sel_alpha_test 0.0 \
		--sel_label_flip 0.25 \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 225.0 \
		--model_kind $(MODEL_KIND) \
		--eval_every 10 \
		--device $(DEVICE)

all: semi-anti-causal confounding selection

# ---- Variantes avec X^perp_{Y,Z} (variable mixte causal+spurieux) ----
# CONF_GAMMA_YZ : force de l'effet causal de X^perp_{Y,Z} sur Y (defaut: 0.5)
# Exemple : make confounding_yz CONF_GAMMA_YZ=1.0
CONF_GAMMA_YZ ?= 0.5
SEL_GAMMA_YZ  ?= 0.5

confounding_yz:
	$(PYTHON) main.py \
		--dataset synthetic_confounding \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--conf_a_train 0.01 0.11 \
		--conf_a_test 0.99 \
		--conf_include_yz \
		--conf_gamma_yz $(CONF_GAMMA_YZ) \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 5000.0 \
		--model_kind $(MODEL_KIND) \
		--eval_every 10 \
		--device $(DEVICE)

selection_yz:
	$(PYTHON) main.py \
		--dataset synthetic_selection \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--sel_alpha_train 0.9 0.8 \
		--sel_alpha_test 0.0 \
		--sel_label_flip 0.25 \
		--sel_include_yz \
		--sel_gamma_yz $(SEL_GAMMA_YZ) \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 225.0 \
		--model_kind $(MODEL_KIND) \
		--eval_every 10 \
		--device $(DEVICE)

# ---- Gap sweep – semi-anti-causal ----
# Exemple : make gap_sweep_sac IRM_LAMBDA_SAC=1000 GAP_STEP=0.025
IRM_LAMBDA_SAC ?= 750.0
GAP_STEP       ?= 0.05

gap_sweep_sac:
	$(PYTHON) run_gap_sweep_sac.py \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--p_center 0.2 \
		--p_test_ood 1.0 \
		--label_flip 0.25 \
		--gap_step $(GAP_STEP) \
		--gap_max 0.40 \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda $(IRM_LAMBDA_SAC) \
		--device $(DEVICE)

nlp:
	$(PYTHON) run_nlp_experiments.py


nlp_confounding:
	$(PYTHON) main.py \
		--dataset nlp_varying_confounder \
		--nlp_n_samples 3000 \
		--val_frac $(VAL_FRAC) \
		--conf_gamma_train 1.2 1.5 \
		--conf_gamma_test 0.0 \
		--erm_steps 25000 \
		--erm_lr 1e-3 \
		--irm_steps 25000 \
		--irm_lr 1e-3 \
		--irm_lambda 250.0 \
		--model_kind logreg \
		--eval_every 10 \
		--device $(DEVICE)

# ---- NLP Custom Confounding (U ~ Bernoulli, token basé sur Z=U⊕flip) ----
# NLP_CONF_ALPHA : force du confondeur U sur Y (defaut: 0.5)
# NLP_CONF_A_TRAIN : taux de flip U->Z pour les envs train (ex: "0.05 0.15")
# NLP_CONF_A_TEST  : taux de flip pour le test OOD (ex: 0.90)
NLP_CONF_ALPHA   ?= 0.7
NLP_CONF_A_TRAIN ?= 0.05 0.15
NLP_CONF_A_TEST  ?= 0.90

nlp_custom_confounding:
	$(PYTHON) main.py \
		--dataset nlp_custom_confounding \
		--nlp_conf_a_train $(NLP_CONF_A_TRAIN) \
		--nlp_conf_a_test $(NLP_CONF_A_TEST) \
		--nlp_conf_alpha $(NLP_CONF_ALPHA) \
		--val_frac $(VAL_FRAC) \
		--erm_steps $(ERM_STEPS) \
		--erm_lr 1e-3 \
		--irm_steps $(IRM_STEPS) \
		--irm_lr 1e-3 \
		--irm_lambda 750.0 \
		--model_kind logreg \
		--eval_every 10 \
		--device $(DEVICE)
