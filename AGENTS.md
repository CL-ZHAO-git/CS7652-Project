# Repository Guidelines

## Project Structure & Module Organization

This CS7652 research project investigates single-agent versus multi-agent AutoML for time-series forecasting under matched computational budgets, including forecast reliability and abstention.

The current workspace contains `CS7652 Project Proposal.docx`; no source, tests, datasets, or build configuration are present. Treat the proposal as the research scope reference. As implementation begins, organize reusable code under `src/`, tests under `tests/`, experiment settings under `configs/`, and reproducible evaluation scripts under `eval/`. These directories are proposed conventions, not existing components.

## Build, Test, and Development Commands

No build, installation, training, or test commands are configured yet. Do not assume a package manager or installed forecasting framework. Useful inspection commands from the repository root include:

- `ls -la`: inspect workspace contents, including hidden files.
- `rg --files`: list visible project files.

When introducing executable code, add a README with exact environment setup, dependency installation, experiment, and test commands. Include a small reproducible run before documenting expensive evaluations.

## Coding Style & Naming Conventions

No formatter, linter, or language configuration is established. If adding Python, use four-space indentation, `snake_case` functions and modules, and `PascalCase` classes. Keep data preparation, forecasting tools, agent orchestration, and evaluation logic separate. Give experiment configurations descriptive names and record random seeds explicitly.

## Testing & Experimental Validation

No test framework or coverage threshold is configured. For Python code, adopt `pytest` with `tests/test_*.py` files and document its installation. Test chronological splits, leakage prevention, corruption generation, budget accounting, and abstention behavior. Compare agent systems using identical data splits and limits on LLM calls, tokens, model fits, tool calls, and runtime. Record forecasting accuracy, selective-prediction metrics, and computational cost.

## Commit & Pull Request Guidelines

Git history is unavailable in this workspace, so no existing commit convention can be verified. Use short, imperative subjects such as `Add rolling validation baseline`. Keep changes focused. Pull requests should explain the research motivation, affected configurations, validation commands and results, and any changes to computational budgets or evaluation assumptions. Link relevant issues when available.

## Data & Configuration

Keep credentials and restricted CRSP/WRDS data out of version control. Document access requirements and use small synthetic fixtures for tests. Record dataset versions and dependency versions so experiments can be reproduced.
