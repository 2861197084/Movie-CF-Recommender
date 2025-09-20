# Repository Guidelines

## Project Structure & Module Organization
The pipeline centers on `main.py`, orchestrating data loading, collaborative filtering runs, evaluation, and visualization. Configuration defaults live in `config.py`. Data access helpers, logging, and download logic are in `utils/`, while model implementations sit in `models/`. Evaluation metrics reside in `evaluation/metrics.py`; analytics and reporting tools (visualizations, hyperparameter search, reports) are in `analysis/`. Raw datasets populate `data/`, run artifacts and plots land in `results/`, `logs/`, and `plots/`. Keep large artifacts out of version control.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install Python dependencies (Python ≥3.8).
- `python main.py`: execute the full experiment pipeline on the configured dataset.
- `python main.py --quick-test`: reduced sample run that auto-downloads `ml-latest-small` for smoke testing.
- `python main.py --hyperparameter-search --search-method random_search --n-trials 50`: launch optimizer defined in `analysis/hyperparameter_optimizer.py`; adjust trials and folds per experiment needs.

## Coding Style & Naming Conventions
Use 4-space indentation, descriptive snake_case for modules, functions, and variables, and PascalCase for class names (see `models/user_based_cf.py`). Type hints and informative docstrings are expected for new public APIs. Reuse the structured logging provided by `utils/logger.py`. Place configuration constants in `config.py` rather than hard-coding values.

## Testing Guidelines
No dedicated unit-test suite exists yet; rely on quick experiments to validate changes. For fast regression checks run `python main.py --quick-test` before proposing changes, and inspect generated metrics in `results/` plus plots in `plots/`. When modifying evaluators or models, capture before/after RMSE or precision values in your PR description.

## Commit & Pull Request Guidelines
Follow Conventional Commit prefixes observed in history (e.g., `feat:`, `feat(component):`). Keep messages concise in English or Chinese, describing the behavioural change. Each PR should summarize intent, list datasets or flags used, attach key metrics or screenshots of plots when relevant, and reference related issues or experiments. Ensure large data files remain untracked and clean up temporary artifacts before review.
