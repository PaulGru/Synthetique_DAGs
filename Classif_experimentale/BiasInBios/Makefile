.PHONY: help bios bios_quick bios_mlp

PYTHON = uv run
DEVICE = auto
SEED   = 0

help:
	@echo "Bias in Bios — ERM vs IRM (split simple)"
	@echo "  make bios        - Run complet (logreg, 10k steps)"
	@echo "  make bios_quick  - Run rapide avec max 5000 exemples"
	@echo "  make bios_mlp    - Run avec MLP"
	@echo ""
	@echo "  Le split 'train' est divisé en 2 environnements d'entraînement."
	@echo "  Le split 'dev' est utilisé pour la validation."
	@echo "  Le split 'test' est utilisé pour l'évaluation."

bios:
	$(PYTHON) main_bios.py \
		--n_train_envs 2 \
		--erm_steps 25000 --erm_lr 1e-3 \
		--irm_steps 25000 --irm_lr 1e-3 --irm_lambda 1000.0 \
		--model_kind logreg \
		--eval_every 500 \
		--device $(DEVICE) \
		--seed $(SEED) \
		--outdir plots/

bios_quick:
	$(PYTHON) main_bios.py \
		--n_train_envs 2 \
		--max_samples 5000 \
		--erm_steps 200 --erm_lr 1e-3 \
		--irm_steps 200 --irm_lr 1e-3 --irm_lambda 500.0 \
		--model_kind logreg \
		--eval_every 50 \
		--device $(DEVICE) \
		--seed $(SEED) \
		--outdir plots/

bios_mlp:
	$(PYTHON) main_bios.py \
		--n_train_envs 2 \
		--erm_steps 10000 --erm_lr 1e-3 \
		--irm_steps 10000 --irm_lr 1e-3 --irm_lambda 500.0 \
		--model_kind mlp --mlp_hidden 512 --mlp_layers 2 \
		--eval_every 500 \
		--device $(DEVICE) \
		--seed $(SEED) \
		--outdir plots/
