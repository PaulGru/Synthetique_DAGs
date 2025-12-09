.PHONY: help semi-anti-causal confounding selection all

# Variables
PYTHON = uv run
N = 200000
N_TEST = 10000
VAL_FRAC = 0.1
ERM_STEPS = 45
ERM_LR = 5e-3
IRM_STEPS = 45000
IRM_LR = 5e-3
MODEL_KIND = logreg
DEVICE = auto

help:
	@echo "Expériences disponibles:"
	@echo "  make semi-anti-causal  - Lance l'expérience semi-anti-causal"
	@echo "  make confounding       - Lance l'expérience confounding"
	@echo "  make selection         - Lance l'expérience selection"
	@echo "  make all               - Lance toutes les expériences"

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
		--irm_lambda 220.0 \
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
		--irm_lambda 720.0 \
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
