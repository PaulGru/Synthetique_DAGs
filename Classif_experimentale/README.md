# IRM vs ERM on NLP Data with Spurious Correlations

Comparison of **ERM** (Empirical Risk Minimization) and **IRM** (Invariant Risk
Minimization) on text classification tasks where spurious correlations are
deliberately injected at training time and broken at test time.

The key question: does IRM correctly identify causal features and generalise
out-of-distribution, while ERM exploits the spurious signal and fails?

---

## Project Structure

```
Classif_experimentale/
│
├── pyproject.toml           # Dependencies + build system (single source of truth)
├── uv.lock                  # Pinned exact versions for reproducibility
├── .python-version          # Python 3.13
│
├── core/                    # ── Models, training loops, evaluation
│   ├── env.py              #    Env dataclass: container for (X, y, y_true, meta)
│   ├── evaluation.py       #    resolve_device(), evaluate_binary/multiclass/env()
│   └── training.py         #    LogisticReg, BertClassifier, train_erm(), train_irm()
│
├── data/                    # ── Data loading, embedding, bias injection
│   ├── nlp_datasets.py     #    Dataset loaders, BERT embedding extraction (cached),
│   │                       #    spurious token injection, build_envs_*() factories
│   └── .embed_cache/       #    On-disk BERT embedding cache (git-ignored)
│
├── experiments/             # ── Entry points and launchers
│   ├── args.py             #    Argument parsers: base_parser() + make_nlp_parser()
│   ├── main.py             #    Single run: 1 dataset × 1 mechanism × 1 seed
│   ├── run_gap_sweep.py    #    Sweep train/test correlation gap
│   ├── run_noise_sweep.py  #    Sweep noise level at fixed gap
│   ├── run_grand_test.py   #    Full grid: 3 datasets × 3 mechanisms × noise × corr
│   ├── aggregate_seeds.py  #    Merge per-seed JSONs → mean ± std
│   └── Makefile            #    All ready-to-run targets (run `make help`)
│
├── plotting/                # ── All visualization scripts
│   ├── training_curves.py  #    Training curves (loss, accuracy, feature weights)
│   ├── gap_sweep.py        #    Combined gap-sweep figures (IMDB + Amazon)
│   ├── grand_test.py       #    Grand test: OOD acc vs noise, faceted
│   └── hparam_search.py    #    IRM λ sensitivity, heatmaps, LaTeX table
│
└── results/                 # ── Generated outputs (git-ignored)
    ├── <dataset_slug>/      #    Single-run outputs (main.py)
    ├── gap_sweep/           #    Gap sweep outputs
    ├── noise_sweep/         #    Noise sweep outputs
    └── grand_test/          #    Grand test outputs
```

---

## Installation

**Prerequisites:** Python ≥ 3.13, GPU recommended (CUDA) for BERT embedding
extraction.

```bash
# 1. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone <repo-url>
cd Classif_experimentale

# 3. Create the virtual environment and install all dependencies
uv sync
```

`uv sync` installs the project as an editable package, so all internal imports
(`from core.training import ...`, `from data.nlp_datasets import ...`) work
without any `sys.path` manipulation.

---

## Datasets

| Dataset | Task | Classes | Approx. size | HuggingFace ID |
|---------|------|---------|-------------|----------------|
| AG News | Topic classification | 4 | 135k articles | `fancyzhx/ag_news` |
| IMDB Genres | Action vs. Romance | 2 | ~50k reviews | `imdb` |
| Amazon Books | Sentiment | 2 | up to 200k reviews | `amazon_polarity` |

BERT embeddings are computed once and cached on disk under `data/.embed_cache/`.
Subsequent runs skip the embedding step entirely.

---

## Spurious Correlation Mechanisms

Three bias injection strategies, each corresponding to a different causal DAG:

| ID | Name | Description | Spurious signal |
|----|------|-------------|-----------------|
| `semi_anti_causal` | SAC | Token correlated with label injected into text | Token present at train, absent/flipped at test |
| `size_selection` | Selection bias | Selection probability depends on text length | Short texts over-represented in one env, OOD breaks this |
| `conf_varying_proxy` | Confounding | Confounder C observed through noisy proxy Z | High proxy alignment at train, weak/reversed at test |

The key hyperparameter: the **gap** — the difference in spurious correlation
strength between the two training environments. A larger gap gives IRM a stronger
invariance signal.

---

## Execution Workflow

All commands are run from the project root. The Makefile is in `experiments/`.

### Level 1 — Single run

Run ERM and IRM on one (dataset, mechanism, seed) combination.

```bash
# Using Makefile (recommended)
cd experiments/
make nlp_agnews_semi_anti_causal
make nlp_imdb_genres_size_selection DEVICE=cuda:0
make nlp_amazon_conf_varying_proxy  NLP_SEED=1

# Or directly from root
uv run experiments/main.py \
    --dataset nlp_agnews_semi_anti_causal \
    --nlp_p_correct_train 0.9 0.7 \
    --nlp_p_correct_test 0.0 \
    --erm_steps 25000 --erm_lr 1e-4 \
    --irm_steps 25000 --irm_lr 1e-4 --irm_lambda 100 \
    --seed 0 --device auto
```

Results → `results/<dataset_slug>/`

To run all mechanisms for a dataset and aggregate over seeds:

```bash
cd experiments/
make all_nlp_agnews    SEEDS="0 1 2"
make all_nlp_imdb_genres
make all_nlp_amazon
```

To aggregate manually:

```bash
uv run experiments/aggregate_seeds.py --datasets nlp_agnews_semi_anti_causal nlp_amazon_conf_varying_proxy
```

---

### Level 2 — Sweep experiments

#### Gap sweep — how does performance change as the train/test gap grows?

```bash
cd experiments/
make gap_sweep_imdb_genres_sac      DEVICE=cuda:0
make gap_sweep_amazon_conf          DEVICE=cuda:0

# Combined figures (IMDB + Amazon on same axes)
uv run plotting/gap_sweep.py
```

Results → `results/gap_sweep/<dataset_slug>/<timestamp>/`

#### Noise sweep — how does performance change as the causal signal degrades?

```bash
cd experiments/
make noise_sweep_imdb_genres_sac    DEVICE=cuda:0
make all_noise_sweep_amazon         DEVICE=cuda:0

# Re-plot without rerunning training
uv run experiments/run_noise_sweep.py \
    --dataset nlp_amazon_semi_anti_causal \
    --plot_only \
    --out_dir results/noise_sweep/causal_amazon_sac/<timestamp>
```

Results → `results/noise_sweep/<dataset_slug>/<timestamp>/`

---

### Level 3 — Grand test (full grid)

Full grid: **3 datasets × 3 mechanisms × 4 noise × 2 correlation × N seeds**.

```bash
cd experiments/

# Sequential
make grand_test DEVICE=cuda:0

# Resume interrupted run
make grand_test_resume DEVICE=cuda:0 GRAND_TEST_RUN_NAME=my_run

# Parallel by dataset (recommended with multiple GPUs)
make grand_test_agnews  GRAND_TEST_RUN_NAME=run1 DEVICE=cuda:0   # Terminal 1
make grand_test_imdb    GRAND_TEST_RUN_NAME=run1 DEVICE=cuda:1   # Terminal 2
make grand_test_amazon  GRAND_TEST_RUN_NAME=run1 DEVICE=cuda:2   # Terminal 3

# Fine-grained 4-way parallelism
make grand_test_part0 GRAND_TEST_RUN_NAME=run1 DEVICE=cuda:0
make grand_test_part1 GRAND_TEST_RUN_NAME=run1 DEVICE=cuda:1
make grand_test_part2 GRAND_TEST_RUN_NAME=run1 DEVICE=cuda:2
make grand_test_part3 GRAND_TEST_RUN_NAME=run1 DEVICE=cuda:3
```

**Generate figures from results:**

```bash
# OOD accuracy vs noise
uv run plotting/grand_test.py --run_name run1

# IRM lambda sensitivity
uv run plotting/hparam_search.py --run_name run1
```

Results → `results/grand_test/<run_name>/`

---

## Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset × mechanism key (see below) | required |
| `--erm_steps` | ERM gradient steps | 25 000 |
| `--erm_lr` | ERM learning rate | 5e-3 |
| `--irm_steps` | IRM gradient steps | 25 000 |
| `--irm_lr` | IRM learning rate | 5e-3 |
| `--irm_lambda` | IRM penalty weight | 200.0 |
| `--seed` | Random seed | 1 |
| `--device` | `cuda` / `cuda:N` / `cpu` / `auto` | auto |
| `--nlp_bert_model` | HuggingFace model | `distilbert-base-uncased` |
| `--nlp_max_length` | Max token length | 128 |
| `--nlp_pooling` | Pooling: `mean` or `cls` | mean |

Full list: `uv run experiments/main.py --help`

### Dataset keys

```
nlp_agnews_semi_anti_causal          nlp_agnews_size_selection          nlp_agnews_conf_varying_proxy
nlp_imdb_genres_semi_anti_causal     nlp_imdb_genres_size_selection     nlp_imdb_genres_conf_varying_proxy
nlp_amazon_semi_anti_causal          nlp_amazon_sentiment_selection     nlp_amazon_conf_varying_proxy
```

---

## Code Architecture

```
experiments/Makefile
  └─ experiments/main.py / run_gap_sweep.py / run_noise_sweep.py / run_grand_test.py
        │
        ├── experiments/args.py         argument parsing
        │
        ├── data/nlp_datasets.py        load dataset → inject bias → BERT embed → List[Env]
        │     └── core/env.py           Env dataclass
        │
        └── core/training.py            train_erm() / train_irm() → (model, history)
              ├── core/evaluation.py    evaluate_env() → accuracy
              └── plotting/training_curves.py  → PNG figures

  Post-processing (reads JSON, no training):
        experiments/aggregate_seeds.py   merge per-seed JSONs → mean ± std
        plotting/gap_sweep.py            combined gap-sweep figures
        plotting/grand_test.py           OOD acc vs noise figures
        plotting/hparam_search.py        lambda sensitivity + heatmaps + LaTeX
```

---

## Output Structure

```
results/
├── causal_agnews_sac/           ← single-run outputs
│   ├── results_seed0.json
│   ├── results_aggregated.json  ← from aggregate_seeds.py
│   └── *.png
│
├── gap_sweep/
│   └── <slug>/<timestamp>/
│       ├── gap_sweep_results.json
│       └── gap_sweep.png
│
├── noise_sweep/
│   └── <slug>/<timestamp>/
│       ├── noise_sweep_results.json
│       └── noise_sweep.png
│
└── grand_test/
    └── <run_name>/
        └── <slug>/noise<N>__<corr_tag>/
            ├── config.json
            ├── results.json
            └── training_curves.png
```
