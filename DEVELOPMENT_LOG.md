# Development Progress and Next Steps

Updated November 22, 2026.

The initial experimental framework is implemented and runs end to end with
synthetic data and a deterministic MockBackend. Both single-agent and four-role
workflows complete forecasting and independent scoring. The runtime now runs on
macOS, Linux, and Windows, and a CRSP/WRDS data layer builds the panel from CRSP.
The next milestones are a live WRDS fetch for the agreed windows and real LLM
execution in Docker. There are no empirical results about the benefits of
multi-agent specialization yet.

## Agreed Research Design

- Predict each stock's cumulative return over the next **1, 3, and 5 trading
  sessions** using models trained on a shared stock panel.
- Compare one agent performing the full workflow with four specialized agents:
  **data diagnosis, feature engineering, modeling, and auditing**.
- Allow every role to write and execute code as well as use shared tools.
- Keep data, underlying LLM, execution environment, and applicable resource limits
  consistent across comparison conditions.
- Evaluate forecasting accuracy and reliability separately. Distinguish withholding
  a forecast from replacing it with a predefined fallback.
- Develop a pipeline once per experimental window, then execute its predictions
  through an independent evaluator. Do not launch a new team per stock per day.

The full planning record is in [PROJECT_DESIGN.md](PROJECT_DESIGN.md). Installation,
execution, data contracts, and implementation limitations are in
[README.md](README.md).

## Completed Implementation

| Component | Current behavior | Main location |
| --- | --- | --- |
| Experiment entry point | Loads configuration, prepares development data, runs agents, freezes a submission, and saves final and baseline reports. | [cli.py](src/forecast_agents/cli.py) |
| Agent loop | Calls the model, executes requested tools, returns results, and requires explicit stage completion. Each role retains its own conversation history. | [runner.py](src/forecast_agents/agents/runner.py) |
| Coordination | Supports single-agent execution and sequential four-role execution, structured handoffs, revision requests, and a revision limit. | [policies.py](src/forecast_agents/orchestration/policies.py) |
| LLM abstraction | Defines common messages, tool calls, usage, and responses. Implements OpenAI-compatible Chat Completions, Anthropic Messages, and a scripted mock. | [llm/](src/forecast_agents/llm) |
| Shared tools | Lists, reads, and writes files; executes Python; validates candidates on historical data; finishes stages. | [workspace.py](src/forecast_agents/tools/workspace.py) |
| Execution and records | Provides local fixture execution and a Docker execution path, process deadlines, a shared budget ledger, and ordered JSONL events. | [runtime/](src/forecast_agents/runtime) |
| Forecasting task | Generates seeded synthetic panels, loads preprocessed JSON panels, defines chronological windows, and checks label maturity. | [panel.py](src/forecast_agents/tasks/panel.py) |
| Independent evaluation | Fits at the training cutoff and predicts from causal history prefixes in fresh workspaces. Checks output completeness and computes per-horizon metrics. | [evaluation/](src/forecast_agents/evaluation) |
| Reproducible settings | Provides mock, OpenAI-compatible, and Anthropic configurations, plus calendar-date split boundaries. | [configs/](configs) |
| CRSP data preparation | Fetches a point-in-time large-cap universe and daily CRSP data from WRDS; builds an anonymized total-return panel with a private symbol map and manifest. | [crsp.py](src/forecast_agents/datasets/crsp.py), [crsp_data.py](scripts/crsp_data.py) |
| Cross-platform execution | Local execution uses rlimits and process groups on POSIX and a job object on Windows; artifacts, tool paths, and candidate hashes are identical across OSes. | [executor.py](src/forecast_agents/runtime/executor.py) |

The mock workflow writes executable feature and pooled ridge-regression code,
runs diagnostics, requests historical validation, and performs syntax smoke
checks. Its tool interactions exercise the same framework used by real agents.
Its behavior is predefined; it does not autonomously choose models or establish
audit quality.

## Verification Snapshot

The latest local verification passed **27 tests** through the standard-library
test runner. The suite is also compatible with pytest. Tests cover session-based
horizons, label maturity, causal observation prefixes, fresh evaluation workspaces,
prediction validation, snapshot integrity, abstention and fallback metrics,
budget exhaustion, timeouts, local imports, and provider tool-call continuation.

Both conditions completed the default offline smoke comparison:

| Condition | Model calls | Tool calls | Outcome |
| --- | --- | --- | --- |
| Single agent mock | 9 | 9 | Completed |
| Four-role mock | 12 | 12 | Completed |

Both deliberately produced the same candidate sources. The call counts describe
scripted orchestration only and do not demonstrate comparative efficiency or
forecasting skill. Mock token usage is synthetic.

Local comparison artifacts are under `runs/smoke-comparison-final/`, including
`comparison.json` and each condition's detailed reports. `runs/` is ignored by
version control; teammates should reproduce these artifacts using the commands
below. No live API requests or Docker execution have been verified.

### November 22, 2026: Cross-Platform Runtime, CRSP Data, and Walk-Forward Evaluation

- **Windows 11, Python 3.13.** 48 tests: 46 pass and 2 symlink tests are skipped
  (no Developer Mode), with both `python -m unittest discover -s tests` and
  `pytest -q`. The new child-process and memory-limit tests fail when the Windows
  job object is disabled, so they test it.
- **Full-size walk-forward dry run.** A fake 10-year extract shaped like the final
  window (100 candidates, 2015-2024) went through `scripts/crsp_data.py build`
  (30 stocks x 2,608 weekday sessions) and a single-mode mock run with the
  `configs/crsp_final_mock.toml` protocol: four validation folds of 52-53 origins
  and three yearly test blocks of 257-260 origins. Local execution on Windows took
  6.3 minutes in total: 56 seconds per validation call and about 4.3 minutes for
  the final test. Docker adds container start-up to every origin, so real-agent
  configs on these windows need larger time budgets than the synthetic examples.
- **Live CRSP data.** Both windows were fetched from WRDS (legacy SIZ tables, which
  end on 2024-12-31) and built: 30 stocks x 2,517 sessions for 2005-2014 and 30 x
  2,516 for 2015-2024. Every panel daily return matches CRSP RET to within 4e-16.
  The 2005-2014 build skipped Wachovia (acquired 2008) and Dell (taken private 2013)
  for incomplete history; the 2015-2024 build skipped none. Each dataset spec records
  its panel's SHA-256, and rebuilding reproduces it. Mock runs completed on both
  windows (multi-agent on 2005-2014, single-agent on 2015-2024), each in about 6.5
  minutes while running in parallel.
- **Not verified.** The suite has not been run on macOS since these changes; POSIX
  code paths changed only to apply rlimits as best effort, to resolve the Docker
  binary path, and to add `-X utf8`. Docker execution remains unverified on every OS.

## Reproduce the Current Progress

Run from the repository root on macOS, Linux, or Windows with Python 3.11 or newer
(see the README for PowerShell syntax). These commands need no API credentials or
third-party packages:

```bash
# Run both mock conditions and save reports. Choose a new output directory.
python eval/run_smoke.py --output-root runs/team-smoke-01

# Run the regression suite without installing development dependencies.
PYTHONPATH=src python -m unittest discover -s tests -v
```

For a normal development environment with package-index access:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m pytest -q
```

Development-package installation was unavailable in the initial restricted-network
environment, so verification used the offline test entry point.

## Current Limits That Affect Experiments

- **Real execution:** the API adapters have protocol tests using fixtures. Docker
  and live-model integration still need verification. Local execution is for
  trusted mock scripts and is not a security sandbox.
- **Data:** the loader requires a balanced panel, so CRSP panels keep only stocks
  with every session, which biases them toward survivors; final windows need entries
  and exits. Agents see only a total-return index, not the volume, market-cap, or
  market-index fields already in the raw extract. The legacy SIZ tables end on
  2024-12-31.
- **Models:** the mock demonstrates pooled ridge regression. Broad ML/DL search,
  prebuilt scientific-computing images, and GPU support are not implemented.
- **Evaluation:** walk-forward refits with per-fold and per-year metrics are
  implemented. Controlled corruption scenarios, confidence intervals,
  or the complete ablation matrix.
- **Reliability:** output fields and descriptive metrics exist. The mock risk
  score is an uncalibrated heuristic, and its audit checks syntax only. Useful
  failure detection and threshold calibration remain research work.
- **Budgets:** call/tool limits and time checks exist. Token limits stop work after
  reported usage reaches the limit; the last response can exceed it. Arbitrary
  code can fit multiple models, so exact fit-count matching is not enforced.
- **Operations:** Python jobs are synchronous and bounded. Resuming interrupted
  runs and asynchronous job management are not implemented. Failed runs remain
  failures rather than being converted into successful fallback results.

## Next Steps by Priority

### Priority 0 Verify the Real Execution Path

Select one accessible model and a Docker-capable machine. Start with synthetic
data, the existing adapter, and the same model for all four roles. Check tool-call
compatibility, code execution, historical validation, stage handoffs, and submission.

**Completion criteria:** a real-model run completes without scripted mock actions;
generated code executes in Docker; image identity and resource usage are recorded;
final evaluation finishes without exposing test feedback to agents. A timeout or
invalid submission must produce an explicit failed-run record.

### Priority 0 Finalize and Integrate the Data Contract

Confirm CRSP/WRDS access or another approved source, then define the stock universe,
date range, available fields, adjustment convention, and forecast issuance time.
Define how missing observations, entries, exits, and delistings are handled before
relaxing the balanced-panel restriction. Fix chronological development and final
evaluation windows and the rules for label availability.

**Completion criteria:** a small approved panel runs through the baseline and
evaluator; manually checked 1/3/5-session labels match the source convention;
features and labels respect their information cutoffs; restricted data remain
outside version control.

**Status (November 22):** both windows (development 2005–2014, final 2015–2024)
are fetched from WRDS and built, and each dataset spec records its panel's
fingerprint. Next: check a few stocks' returns by hand against an independent
source, and extend the loader for entries, exits, and covariates.

### Priority 1 Establish the First Real Comparison

Build a versioned environment for the selected ML/DL dependencies. Strengthen role
instructions and evidence-based handoffs using development tasks. Implement the
chosen fixed/non-agent baselines and run single-agent and four-role conditions
on paired tasks with the same model, image, data, and resource settings.

**Completion criteria:** configurations and artifacts reproduce the comparison;
valid and failed runs are both counted; per-horizon prediction errors, baseline
results, actual resource consumption, and run-to-run variation are reported.
Freeze prompts and settings before evaluating reserved final tasks.

### Priority 1 Implement Reliability Experiments

Define risk-score direction, candidate reliability signals, operational abstention
and fallback rules, and validation-only calibration. Add controlled input
corruptions with clean evaluation targets and separately specified shift scenarios.

**Completion criteria:** reliability decisions are submitted before final scoring;
thresholds use development data; clean/stressed forecasting and risk-coverage
results are reported separately. Agent judgment disagreement and predictive-model
disagreement are measured as distinct signals.

### Priority 2 Explain the Results with Ablations

Compare feature engineering inside modeling against a separate feature role, and
compare systems with and without auditing. Finalize repetitions and statistical
comparisons that account for overlapping return horizons and cross-stock
dependence. Add more model families or additional providers when
the experiment requires them.

**Completion criteria:** the results identify which role changes affect accuracy,
reliability, and cost, with uncertainty reported under a documented procedure.

## Suggested Team Workstreams

Owners have not been assigned. The following division supports independent work
once the data and pipeline interfaces are agreed:

| Workstream | Focus | Integration handoff |
| --- | --- | --- |
| Data and task protocol | Access, preprocessing, universe rules, targets, and chronological splits. | A documented panel contract and a small approved test dataset. |
| Agent runtime and modeling | Live backend, Docker environment, role instructions, coding loop, and model dependencies. | One real agent workflow that submits an executable pipeline. |
| Evaluation and reliability | Baselines, scoring, calibration, stress scenarios, aggregation, and ablations. | Reproducible reports and a frozen experimental protocol. |

Update this log when a milestone is verified. Record the date, configuration,
command, outcome, and remaining limitations. Update the design record when an
experimental decision changes and the README when execution instructions change.
