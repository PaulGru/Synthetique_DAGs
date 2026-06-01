.PHONY: help semi_anti_causal confounding_varying_proxy selection \
        gap_sweep_semi_anti_causal gap_sweep_selection gap_sweep_confounding_varying_proxy \
        all all_gap \
        ac_semi_anti_causal ac_selection ac_confounding_varying_proxy \
		ac_imb_semi_anti_causal ac_imb_selection ac_imb_confounding_varying_proxy \
        gap_sweep_ac_semi_anti_causal gap_sweep_ac_selection gap_sweep_ac_confounding_varying_proxy \ \
		all_ac all_ac_imb all_gap_ac \
		noise_sweep_semi_anti_causal noise_sweep_selection noise_sweep_ac_semi_anti_causal noise_sweep_ac_selection \
		noise_sweep_confounding_varying_proxy noise_sweep_ac_confounding_varying_proxy all_noise

THIS_DIR = $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT     = $(THIS_DIR)

# Variables
PYTHON = uv run
N = 200000
N_TEST = 10000
VAL_FRAC = 0.1
EVAL_EVERY = 10
ERM_STEPS = 25000
ERM_LR = 1e-3
IRM_STEPS = 25000
IRM_LR = 1e-3
DEVICE = auto

help:
	@echo "Synthetic experiments  (plots saved to plots/<dataset>/)"
	@echo ""
	@echo "  --- CAS CAUSAL (X → Y) ---"
	@echo "  make semi-anti-causal              → plots/semi_anti_causal/"
	@echo "  make selection                     → plots/selection/"
	@echo "  make confounding_varying_proxy     → plots/confounding_varying_proxy/"
	@echo "  make all                           → all 5 causal experiments in sequence"
	@echo ""
	@echo "  --- CAS ANTI-CAUSAL (Y → X) ---"
	@echo "  make ac_semi_anti_causal           → plots/ac_semi_anti_causal/"
	@echo "  make ac_selection                  → plots/ac_selection/"
	@echo "  make ac_confounding_varying_proxy  → plots/ac_confounding_varying_proxy/"
	@echo "  make all_ac                        → all 5 anti-causal experiments in sequence"
	@echo ""
	@echo "  --- CAS ANTI-CAUSAL + class imbalance ---"
	@echo "  make ac_imb_semi_anti_causal       → plots/ac_imb_semi_anti_causal/"
	@echo "  make ac_imb_selection              → plots/ac_imb_selection/"
	@echo "  make ac_imb_confounding_varying_proxy"
	@echo "  make"
	@echo "  make"
	@echo "  make all_ac_imb                    → all 5 anti-causal imbalanced experiments"
	@echo ""
	@echo "  --- Gap sweeps (causal) ---"
	@echo "  make gap_sweep_semi_anti_causal    → gap sweep (semi-anti-causal)"
	@echo "  make gap_sweep_selection           → gap sweep (selection)"
	@echo "  make gap_sweep_confounding_varying_proxy"
	@echo "  make"
	@echo "  make all_gap                        → all 4 causal gap sweeps in sequence"
	@echo ""
	@echo "  --- Gap sweeps (anti-causal) ---"
	@echo "  make gap_sweep_ac_semi_anti_causal"
	@echo "  make gap_sweep_ac_selection"
	@echo "  make gap_sweep_ac_confounding_varying_proxy"
	@echo "  make"
	@echo "  make"
	@echo "  make all_gap_ac                     → all 5 anti-causal gap sweeps in sequence"
	@echo ""
	@echo "  --- Noise sweeps (label_flip 0 → 0.25, gap fixé) ---"
	@echo "  make noise_sweep_semi_anti_causal   → noise sweep (semi-anti-causal)"
	@echo "  make noise_sweep_selection          → noise sweep (selection)"
	@echo "  make noise_sweep_ac_semi_anti_causal"
	@echo "  make noise_sweep_ac_selection"
	@echo "  make noise_sweep_confounding_varying_proxy"
	@echo "  make noise_sweep_ac_confounding_varying_proxy"
	@echo "  make all_noise                      → causal semi-anti-causal + selection"
	@echo "  Overrideable: NOISE_STEP=0.05"

semi_anti_causal:
	cd $(ROOT) && uv run main.py \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--ps_train 0.1 0.2 \
		--p_test_ood 1.0 \
		--label_flip 0.0 \
		--erm_steps 25_000 \
		--erm_lr $(ERM_LR) \
		--irm_steps 25_000 \
		--irm_lr $(IRM_LR) \
		--irm_lambda 75.0 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

selection:
	cd $(ROOT) && uv run main.py \
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
		--irm_lambda 100.0 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

confounding_varying_proxy:
	cd $(ROOT) && uv run main.py \
		--dataset synthetic_confounding_varying_proxy \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--conf_a_train 0.01 0.11 \
		--conf_a_test 0.99 \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 200.0 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)


all_causal: semi_anti_causal selection confounding_varying_proxy
# 

# ---- Anti-causal experiments ----
ac_semi_anti_causal:
	cd $(ROOT) && uv run main.py \
		--dataset synthetic_ac_semi_anti_causal \
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
		--irm_lambda 100.0 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

ac_selection:
	cd $(ROOT) && uv run main.py \
		--dataset synthetic_ac_selection \
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
		--irm_lambda 100.0 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

ac_confounding_varying_proxy:
	cd $(ROOT) && uv run main.py \
		--dataset synthetic_ac_confounding_varying_proxy \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--conf_a_train 0.01 0.11 \
		--conf_a_test 0.99 \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 300.0 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)


all_ac: ac_semi_anti_causal ac_selection ac_confounding_varying_proxy
#

# ---- Anti-causal imbalanced experiments ----
ac_imb_semi_anti_causal:
	cd $(ROOT) && uv run main.py \
		--dataset synthetic_ac_semi_anti_causal \
		--plot_dir plots/ac_imb_semi_anti_causal \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--ps_train 0.1 0.2 \
		--p_test_ood 1.0 \
		--label_flip 0.25 \
		--ac_py_train 0.1 0.9 \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 100.0 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

ac_imb_selection:
	cd $(ROOT) && uv run main.py \
		--dataset synthetic_ac_selection \
		--plot_dir plots/ac_imb_selection \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--sel_alpha_train 0.9 0.8 \
		--sel_alpha_test 0.0 \
		--sel_label_flip 0.25 \
		--ac_py_train 0.1 0.9 \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 100.0 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)

ac_imb_confounding_varying_proxy:
	cd $(ROOT) && uv run main.py \
		--dataset synthetic_ac_confounding_varying_proxy \
		--plot_dir plots/ac_imb_confounding_varying_proxy \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--conf_a_train 0.01 0.11 \
		--conf_a_test 0.99 \
		--ac_proxy_pc_train 0.1 0.9 \
		--ac_proxy_pc_test 0.5 \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 200.0 \
		--eval_every $(EVAL_EVERY) \
		--device $(DEVICE)



all_ac_imb: ac_imb_semi_anti_causal ac_imb_selection ac_imb_confounding_varying_proxy
#



# ---- Gap sweeps – script générique run_gap_sweep.py ----
# Exemple : make gap_sweep_semi_anti_causal GAP_STEP=0.025
GAP_STEP       ?= 0.05

gap_sweep_semi_anti_causal:
	cd $(ROOT) && uv run run_gap_sweep.py \
		--dataset synthetic_semi_anti_causal \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--label_flip 0.25 \
		--gap_step $(GAP_STEP) \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 100 \
		--device $(DEVICE)

gap_sweep_selection:
	cd $(ROOT) && uv run run_gap_sweep.py \
		--dataset synthetic_selection \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--label_flip 0.25 \
		--gap_step $(GAP_STEP) \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 100 \
		--device $(DEVICE)

gap_sweep_confounding_varying_proxy:
	cd $(ROOT) && uv run run_gap_sweep.py \
		--dataset synthetic_confounding_varying_proxy \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--gap_step 0.02 \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 200 \
		--device $(DEVICE)


all_gap: gap_sweep_semi_anti_causal gap_sweep_selection gap_sweep_confounding_varying_proxy

# ---- Anti-causal gap sweeps ----
gap_sweep_ac_semi_anti_causal:
	cd $(ROOT) && uv run run_gap_sweep.py \
		--dataset synthetic_ac_semi_anti_causal \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--label_flip 0.25 \
		--gap_step $(GAP_STEP) \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 100 \
		--device $(DEVICE)

gap_sweep_ac_selection:
	cd $(ROOT) && uv run run_gap_sweep.py \
		--dataset synthetic_ac_selection \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--label_flip 0.25 \
		--gap_step $(GAP_STEP) \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 100 \
		--device $(DEVICE)

gap_sweep_ac_confounding_varying_proxy:
	cd $(ROOT) && uv run run_gap_sweep.py \
		--dataset synthetic_ac_confounding_varying_proxy \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--gap_step 0.02 \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 200 \
		--device $(DEVICE)

all_gap_ac: gap_sweep_ac_semi_anti_causal gap_sweep_ac_selection gap_sweep_ac_confounding_varying_proxy


# ---- Noise sweeps – script run_noise_sweep.py ----
# Fait varier le taux de bruit (label_flip) de 0 à 0.25 (gap fixé).
# Exemple : make noise_sweep_semi_anti_causal NOISE_STEP=0.05
NOISE_STEP     ?= 0.025

noise_sweep_semi_anti_causal:
	cd $(ROOT) && uv run run_noise_sweep.py \
		--dataset synthetic_semi_anti_causal \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--noise_step $(NOISE_STEP) \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 100 \
		--device $(DEVICE)

noise_sweep_selection:
	cd $(ROOT) && uv run run_noise_sweep.py \
		--dataset synthetic_selection \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--noise_step $(NOISE_STEP) \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 100 \
		--device $(DEVICE)

noise_sweep_ac_semi_anti_causal:
	cd $(ROOT) && uv run run_noise_sweep.py \
		--dataset synthetic_ac_semi_anti_causal \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--noise_step $(NOISE_STEP) \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 100 \
		--device $(DEVICE)

noise_sweep_ac_selection:
	cd $(ROOT) && uv run run_noise_sweep.py \
		--dataset synthetic_ac_selection \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--noise_step $(NOISE_STEP) \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 100 \
		--device $(DEVICE)

noise_sweep_confounding_varying_proxy:
	cd $(ROOT) && uv run run_noise_sweep.py \
		--dataset synthetic_confounding_varying_proxy \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--noise_step 0.5 \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 200 \
		--device $(DEVICE)

noise_sweep_ac_confounding_varying_proxy:
	cd $(ROOT) && uv run run_noise_sweep.py \
		--dataset synthetic_ac_confounding_varying_proxy \
		--n $(N) \
		--n_test $(N_TEST) \
		--val_frac $(VAL_FRAC) \
		--noise_step $(NOISE_STEP) \
		--erm_steps $(ERM_STEPS) \
		--erm_lr $(ERM_LR) \
		--irm_steps $(IRM_STEPS) \
		--irm_lr $(IRM_LR) \
		--irm_lambda 300 \
		--device $(DEVICE)

all_noise: noise_sweep_semi_anti_causal noise_sweep_selection noise_sweep_confounding_varying_proxy
