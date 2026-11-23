# Forecasting Agents

Research infrastructure for comparing one coding agent with four specialized
agents under common resource constraints. The forecasting task pools multiple
stocks and predicts each stock's next **1, 3, and 5 trading-session cumulative
returns**. See [PROJECT_DESIGN.md](PROJECT_DESIGN.md) for the research decisions.
For the current team handoff, verified progress, and prioritized next steps, see
[DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md).

## Implemented Scope

- Shared agent loop, separate role histories, artifact handoffs, and bounded
  revisions across diagnosis, feature engineering, modeling, and auditing.
- Provider-independent messages, tool calls, usage records, and stop reasons.
  Adapters: OpenAI-compatible Chat Completions and Anthropic Messages.
- File tools, bounded synchronous Python execution, historical validation, and
  explicit stage completion. No implicit model/provider fallback or API retries.
- Shared call/tool/token ledger, wall-time checks, and JSONL event records.
- Frozen candidate snapshots and independent fixed-fit, causal-prefix evaluation.
- Per-horizon MAE/RMSE, zero-return comparison, abstention coverage, zero-return
  fallback, and post-hoc risk-coverage curves with equal-risk observations grouped.
- A deterministic, offline MockBackend that writes and executes a pooled ridge
  regression pipeline using only Python's standard library.

**MockBackend is a scripted integration fixture.** It does not reason, search
models, or establish a single-agent/multi-agent performance advantage. Its risk
score is an uncalibrated heuristic; its audit script only checks syntax. Real
agents receive role instructions and tools and must create their own code.

## Quick Start

Run from the repository root on macOS, Linux, or Windows with Python 3.11 or newer.
The offline demo requires no third-party packages, API credentials, Docker, or
downloaded data. On macOS or Linux:

```bash
PYTHONPATH=src python -m forecast_agents run --config configs/demo.toml --output runs/demo-multi
PYTHONPATH=src python -m forecast_agents run --config configs/demo.toml --mode single --output runs/demo-single
```

In Windows PowerShell, set `PYTHONPATH` once per session (the editable installation
below needs no `PYTHONPATH`):

```powershell
$env:PYTHONPATH = "src"
python -m forecast_agents run --config configs/demo.toml --output runs/demo-multi
python -m forecast_agents run --config configs/demo.toml --mode single --output runs/demo-single
```

Every output directory must be new. Existing runs are never overwritten. To run
both conditions and save an aggregate smoke-test summary:

```bash
python eval/run_smoke.py --output-root runs/smoke-comparison
```

The mock conditions intentionally produce the same forecasting sources. Their
different role and call traces test orchestration, not scientific hypotheses.

For an editable installation and pytest on macOS or Linux:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/forecast-agents run --config configs/demo.toml --output runs/installed-demo
```

In Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\forecast-agents run --config configs/demo.toml --output runs/installed-demo
```

Package installation needs access to the configured Python package index. The
same tests also have an offline standard-library entry point:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v              # macOS/Linux
$env:PYTHONPATH = "src"; python -m unittest discover -s tests -v    # PowerShell
```

Tests cover label maturity, session-based horizons, causal observation prefixes,
fresh evaluation workspaces, prediction completeness, abstention/fallback metrics,
snapshot integrity, budget exhaustion, provider continuation state, end-to-end
single/multi runs, child-process cleanup, platform-independent paths and hashes,
and the CRSP fetch/build steps against a fake WRDS connection. On Windows without
Developer Mode, the two symlink tests are skipped. No numerical coverage threshold
is prescribed.

## Real LLM Backends

Real LLM-generated code requires the Docker execution mode; the configuration
loader rejects real backends paired with local execution. Docker must be installed
and its daemon accessible; on macOS and Windows, Docker Desktop provides it. Windows
hosts have no POSIX user ID to map, so containers there run as UID 1000. Pull or
build the configured image before running:

```bash
docker pull python:3.13-slim
```

For an OpenAI-compatible endpoint, set `LLM_MODEL`, `LLM_BASE_URL` (the API prefix,
usually ending in `/v1`), and `LLM_API_KEY` through your environment, then run:

```bash
PYTHONPATH=src python -m forecast_agents run --config configs/openai_compatible.toml --output runs/api-multi
```

`token_parameter` explicitly selects `max_completion_tokens` or `max_tokens` to
match the endpoint. Compatibility includes tool calls and usage fields, not just
text completion. Unsupported responses or unknown usage stop the run. Provider
requirements beyond the implemented contract need an adapter extension.

For Anthropic, set `LLM_MODEL` and `ANTHROPIC_API_KEY` and use
`--config configs/anthropic.toml`. Native assistant content, including continuation
blocks, is retained. API credentials are read by the controller and are not passed
to generated-code environments. Request headers are not logged.

The first image contains only Python's standard library. To use external ML/DL
packages, build a versioned image containing the selected dependencies and set
`execution.image` to that image. The resolved local image ID is recorded, and
single-agent and multi-agent comparisons must use the same image and resources.
GPU execution is not implemented yet.

The adapters have fixture-based protocol tests. Live paid API calls and Docker
execution have not been verified in the initial development environment.

## Data and Prediction Protocol

Synthetic data use seeded, correlated stock returns and weekday sessions. They
are neither real market data nor an exchange holiday calendar.

For preprocessed data, configure the source in a copy of the TOML configuration:

```toml
[data]
source = "json"
path = "data/panel.json"
train_sessions = 70
validation_sessions = 25
horizons = [1, 3, 5]
```

Instead of session counts, `train_end` and `validation_end` (YYYY-MM-DD) set
calendar boundaries; each resolves to the last panel session on or before that
date. Use one form or the other, as in [crsp_dev_mock.toml](configs/crsp_dev_mock.toml).

The JSON file is an array of `{ "symbol": "A", "date": "2020-01-02", "close": 100.0 }`
records. Use ISO dates, unique stock-date pairs, and positive finite prices with a
consistent adjustment/return convention. This initial loader requires a balanced
panel with the same observed sessions for every stock. It rejects missing rows
rather than silently changing horizon lengths. [CRSP/WRDS Data](#crspwrds-data)
builds this format from CRSP. Keep restricted data and credentials out of version
control.

Training uses the first `train_sessions` dates. The following
`validation_sessions` dates form the development validation period; subsequent
dates form the final test period. Validation/test origins are scored only if all
requested horizons mature within that period. The convention is prediction after
observing the origin's close. No annualization or excess-return adjustment occurs.

`refit` selects the evaluation protocol:

- `"none"` (default): one fit at the end of training serves validation and the test.
- `"year"`, `"quarter"`, or `"month"`: walk-forward. Validation splits into folds at
  each period end, and each fold's model is fitted on all history before it. During
  the test, the frozen pipeline is refitted at each period end on all history to
  date. Reports give metrics per fold or test block as well as overall.

`validation_stride` and `test_stride` score every n-th eligible origin (default 1).
A stride of 5 makes 5-day returns non-overlapping and cuts run time.

Agents see development history only. `validate_candidate` returns historical
validation feedback. Final scoring begins after the agent workflow ends, requires
the submitted sources to match the last validated snapshot, and never returns
test feedback to a live agent.

The pipeline interface is:

```python
def fit(history, horizons):
    return json_serializable_state

def predict(history, state, horizons):
    return prediction_rows
```

Rows contain `symbol`, `horizon`, `prediction`, `risk`, `abstain`, and `fallback`.
The runtime writes a detailed contract into each workspace. Fit receives only
history through its cutoff, so any constructed forward labels must mature by that
cutoff; under walk-forward it is called once per fold and per test block. Every
predict call gets a fresh workspace and only history through the current origin;
future prices and final labels are not mounted. Persist model state in returned
JSON, not mutable files or Python globals.

Abstention excludes a row from served-error metrics. Fallback replaces the served
prediction with zero. Raw candidate errors and zero-baseline errors are reported
separately. The zero-return baseline (`baseline_report.json`) is fixed code, so it
is scored in-process on the same origins and labels. Risk-coverage curves use raw forecasts and are post-hoc descriptive
results, not threshold calibration. No confidence intervals or significance claims
are produced by the smoke evaluator.

## CRSP/WRDS Data

`scripts/crsp_data.py` turns WRDS access into the panel JSON above in two steps.
Each experiment window has a dataset spec in `configs/datasets/` and a run config:

| Window | Dataset spec | Run config | Use |
|---|---|---|---|
| 2005–2014 | `crsp_lc30_2005_2014.toml` | `crsp_dev_mock.toml` | Development: build and tune agents and prompts |
| 2015–2024 | `crsp_lc30_2015_2024.toml` | `crsp_final_mock.toml` | Final experiments, only after prompts are frozen |

Both windows use a 70/30 walk-forward protocol. Agents see the first seven years and
validate on four yearly folds, each trained on all earlier years. The last three
years are the hidden test, with the pipeline refitted each January. Validation folds
are 2008–2011 and 2018–2021; test periods are 2012–2014 and 2022–2024.

The commands are identical on macOS, Linux, and Windows:

```bash
python -m pip install -e ".[wrds]"   # once; wrds currently needs Python 3.11-3.13
python scripts/crsp_data.py fetch configs/datasets/crsp_lc30_2005_2014.toml
python scripts/crsp_data.py build configs/datasets/crsp_lc30_2005_2014.toml
python -m forecast_agents run --config configs/crsp_dev_mock.toml --output runs/crsp-dev-mock
```

The first `fetch` asks for your WRDS password (and possibly a Duo approval); the
`wrds` package can save it to your own pgpass file, outside the repository. Set
`WRDS_USERNAME` or pass `--wrds-username`. The last command needs `PYTHONPATH=src`
or the editable installation.

- **fetch** (needs WRDS) ranks companies by market cap on the formation date (the
  last CRSP session before `start`) among ordinary common shares (SHRCD 10/11) on
  NYSE, NYSE American, and NASDAQ (EXCHCD 1/2/3), keeping one share class per
  company. For the top `candidates`, it downloads daily `crsp.dsf` rows (RET, RETX,
  PRC, VOL, SHROUT, adjustment factors, open/high/low, bid/ask) with their
  `crsp.dsenames` and `crsp.dsedelist` records, plus the `crsp.dsi` market indices
  (VWRETD, EWRETD, SPRTRN), into `data/raw/<name>/` with a manifest.
- **build** (offline) keeps the `stocks` highest-ranked candidates that have a CRSP
  row and a valid RET on every session, and writes `data/panels/<name>.json`.
  `close` is a total-return index compounded forward from 1.0, so
  `close[t+h]/close[t] - 1` is the cumulative return including dividends. Symbols
  are anonymized (`S000`, ...); the PERMNO/ticker map (`<name>.symbols.csv`) and a
  manifest listing dropped candidates stay beside the panel.
- **Same data for everyone:** a spec's `expected_sha256` records the fingerprint of
  the team's reference build. `build` refuses to write a panel with a different
  fingerprint, so a teammate who fetches with their own WRDS account gets either the
  identical panel or an error explaining that their extract differs.

Known limitations:

- Requiring every session conditions on survival through the window end. This
  survivorship bias is acceptable for development runs only; `PROJECT_DESIGN.md`
  excludes it for final experiments. The build manifest lists each company dropped
  for this reason. Removing the bias requires a loader that allows entries and exits.
- The loader accepts only `symbol`, `date`, and `close`, so agents do not yet see
  volume, market cap, or market returns, although the raw extract contains them.
- The fetch reads CRSP's legacy SIZ tables, which CRSP has replaced with the CIZ
  format; on WRDS they end on 2024-12-31. The fetch rejects windows beyond the
  available coverage; later windows need a CIZ variant of the queries.
- Raw extracts, panels, and run directories contain CRSP data, and real agents send
  tool output to the configured LLM provider. Check your WRDS/CRSP license terms
  before sharing files or running real agents on CRSP data.

## Execution and Budget Semantics

`local` execution is exclusively for the trusted mock fixtures. Process deadlines,
memory/file limits, environment filtering, and path checks do **not** make local
Python a security sandbox; it can access host files. Do not use it for untrusted
code or formal leakage-isolation claims. Limits use each platform's mechanism:
rlimits and a process group on Linux and macOS (macOS does not enforce the memory
rlimit), and a job object on Windows, which caps memory and kills child processes
when a script ends.

Docker execution disables networking, drops capabilities, uses a non-root UID,
limits CPU/memory/processes, and mounts development data read-only. The evaluator
keeps the full panel outside mounted workspaces. The controller performs API
calls; generated code has no API credentials or Docker socket.

All roles share call and tool limits. LLM usage comes from provider reports; mock
usage is explicitly synthetic. Cached input and reasoning details are retained.
Unknown usage stops the run instead of being charged as zero. Token limits are
**observed-use stops**: the last response can exceed the remaining budget because
provider-independent exact input pre-counting is not implemented. No tool from
that over-budget response is executed. Model fits inside arbitrary code are not
claimed to be exactly counted or capped.

Agent development has one wall-time budget, which includes every
`validate_candidate` call. Final candidate execution has a separately configured
evaluation wall-time limit; the zero baseline is scored in-process. The run records the development ledger separately from final evaluation durations.
An exhausted budget, rejected candidate, invalid submission, or execution failure
is a failed run, not an automatic successful fallback. Runs are not resumable yet.

## Repository and Run Layout

```text
src/forecast_agents/
  agents/          common agent loop and role prompts
  llm/             contracts, real adapters, scripted mock
  tasks/           panel loader, labels, chronological task definitions
  tools/           shared file/execution/validation tools
  orchestration/   single-agent and multi-agent policies
  runtime/         execution, budget ledger, event log
  evaluation/      frozen-pipeline worker and independent scoring
  datasets/        CRSP/WRDS fetch and panel construction, outside agent runs
configs/           reproducible TOML settings
configs/datasets/  dataset specs for scripts/crsp_data.py
scripts/           data preparation entry points
tests/             pytest-compatible unittest tests
eval/              offline smoke-comparison entry point
runs/<run>/
  run.json         configuration, status, image ID, budgets, metrics
  events.jsonl     ordered model/tool/stage events
  handoffs.json    completed role handoffs
  workspace/      development data, generated code, historical validation
  submission/     frozen candidate sources
  test_report.json
  baseline_report.json
  private/        full panel, never mounted into execution containers
```

Full development traces and datasets can be sensitive; review run artifacts before
sharing them. The repository ignores run output, `.env`, and `data/` by default.

## Working Across macOS and Windows

`.gitattributes` stores text files with LF line endings and checks them out with LF
on every OS, regardless of each person's `core.autocrlf` setting. Run artifacts are
UTF-8 with LF, agent tools use forward-slash paths and the same case-insensitive
input protection everywhere, and candidate hashes match across platforms. Run both
conditions of a paired comparison on the same machine: Docker image IDs differ
between Apple Silicon and x86, and wall-time budgets depend on hardware.

## Next Research Milestones

Select a live model and develop real agents on the 2005–2014 CRSP window. Extend
the loader with covariates and with stock entries and exits so final windows avoid
survivorship bias, and settle the model dependencies. Then add
controlled corruptions, calibrated reliability decisions, matched-budget repeated
experiments, role ablations, and uncertainty estimates that respect temporal and
cross-stock dependence. Broad ML/DL search and autonomous audit quality require
real-model experiments and are not demonstrated by the deterministic fixture.

Adapter references:
[Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create),
[Anthropic tool calls](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls).
Execution references:
[Docker resource limits](https://docs.docker.com/engine/containers/resource_constraints/),
[Docker network isolation](https://docs.docker.com/engine/network/drivers/none).
