.PHONY: help bios bios_quick bios_mlp

# Variables
PYTHON = uv run
DEVICE = auto
SEED   = 42

help:
	@echo "Expériences Bias in Bios :"
	@echo "  make bios        - Run complet ERM+IRM (logreg, 10k steps)"
	@echo "  make bios_quick  - Run rapide pour tester le pipeline (200 steps)"
	@echo "  make bios_mlp    - Run avec MLP au lieu de logreg"

# Run principal — logreg, paramètres par défaut
bios:
	$(PYTHON) main_bios.py \
		--train_rho 0.8 0.6 \
		--test_rho 0.2 \
		--erm_steps 10000 --erm_lr 1e-3 \
		--irm_steps 10000 --irm_lr 1e-3 --irm_lambda 500.0 \
		--model_kind logreg \
		--eval_every 500 \
		--device $(DEVICE) \
		--seed $(SEED) \
		--outdir plots/

# Run rapide : peu de samples, peu de steps (pour valider le pipeline)
bios_quick:
	$(PYTHON) main_bios.py \
		--train_rho 0.8 0.6 \
		--test_rho 0.2 \
		--max_samples 5000 \
		--erm_steps 200 --erm_lr 1e-3 \
		--irm_steps 200 --irm_lr 1e-3 --irm_lambda 500.0 \
		--model_kind logreg \
		--eval_every 50 \
		--device $(DEVICE) \
		--seed $(SEED) \
		--outdir plots/

# Run avec MLP
bios_mlp:
	$(PYTHON) main_bios.py \
		--train_rho 0.8 0.6 \
		--test_rho 0.2 \
		--erm_steps 10000 --erm_lr 1e-3 \
		--irm_steps 10000 --irm_lr 1e-3 --irm_lambda 500.0 \
		--model_kind mlp --mlp_hidden 512 --mlp_layers 2 \
		--eval_every 500 \
		--device $(DEVICE) \
		--seed $(SEED) \
		--outdir plots/
