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
- Per-horizon MAE/MSE/RMSE, direction cross entropy/accuracy, zero-return comparison, abstention coverage, zero-return
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

For a DeepSeek connection check, set `DEEPSEEK_API_KEY` or keep the key in the
git-ignored local `key.txt`, then run:

```bash
python scripts/check_deepseek.py          # model listing only; no generation
python scripts/check_deepseek.py --probe  # two small billable tool-call requests
```

The check sends only a fictional addition task and never reads project datasets.
It reports authentication, available model IDs, and optional tool continuation
without printing the credential. It does not change experiment configuration.

The dedicated [DeepSeek configuration](configs/deepseek.toml) uses the official
OpenAI-compatible base URL `https://api.deepseek.com` and model `deepseek-flash`.
It uses synthetic data and explicitly disables thinking for the initial test.
After the connection check passes and the Docker image is available, run:

```bash
python -m forecast_agents run --config configs/deepseek.toml --key-file key.txt --output runs/deepseek-synthetic-01
```

Alternatively, set `DEEPSEEK_API_KEY` and omit `--key-file`. An explicit key file
temporarily overrides that environment variable in the controller and is never
recorded in the run configuration. Generated-code environments do not inherit it.
`llm.request_options` supports explicit `thinking` and `reasoning_effort` controls
for the OpenAI-compatible adapter; unsupported options are rejected. Reasoning
settings are part of the recorded experiment configuration and must match across
single-agent/multi-agent comparisons. This configuration makes billable calls.

Agents receive a refreshed budget notice on every call. `[scheduling]` enables
cumulative role limits by default: 15% diagnosis, 15% features, 45% modeling, and
25% auditing. These shares partition tokens, model calls, tool calls, and active
role time. Roles cannot borrow each other's allocations; revisions reuse their
previous counters. Time spent in other roles does not consume a role's own clock.
Single-agent mode receives the full shared allowance with the same closing policy.

At 70% of role tokens the notice requests wrap-up. By default, at 80% of tokens or
active time, or with only two calls/tools left, the runner enters **finish-only**
mode. It also closes early if recent prompt usage suggests two final calls need
the remaining tokens. This transition is permanent for that role, including
revisions. Only `finish_stage` remains available; modeling, auditing, and single
mode can also use `validate_candidate`; auditing/single can additionally run the
fixed `audit_candidate` checks. Writes, script execution, and new searches
are blocked, including attempts to call hidden tools. A blocked batch executes
nothing and receives structured tool-error feedback, with matching results for
every call ID. By default, `max_policy_corrections = 1` permits one corrective
request within the same role/global allowances; a second violation fails the run.
Rejected tool attempts still count toward the tool budget. An explicit valid
handoff is required; incomplete checks are never automatically accepted. A revision
target that is already finish-only is rejected immediately: revisiting it cannot
reopen code editing or reset its counters.

For endpoints that support it, `scheduling.force_tool_choice = true` also sets the
provider's API-level tool choice. Closing diagnosis/features must call
`finish_stage`; other closing roles must call an offered tool. After a correction,
their essential validation/audit tools remain available. The default is false
for provider portability.
The DeepSeek profile explicitly enables it with thinking disabled, as required by
the [DeepSeek tool-choice contract](https://api-docs.deepseek.com/api/create-chat-completion/).
Incompatible DeepSeek thinking settings are rejected locally; the controller never
silently changes the reasoning mode.

The feature role implements compact causal helpers, not a full trained predictor.
If its latest recorded Python check failed, `completed` is blocked. The agent must
repair and rerun its check while work is allowed, or explicitly hand off as
`revise` to modeling with the failure details (or reject). This consumes the normal
revision allowance. Modeling must investigate unresolved feature failures before
using those helpers, and the normal audit stage remains mandatory.

Before a request, the runner estimates its input cost using recent provider usage,
context growth, and `input_token_margin`, then limits output to the remaining role
allowance. This is an admission estimate, not an exact tokenizer. Full input
(including cached input) and output are charged from the provider's usage report.
Unexpected response overshoots are recorded and stop the run before executing its
tools. Unknown usage also stops the run. Model/tool call limits are checked before
admission; execution deadlines respect both role and global time limits.

The CLI and `run.json` include both shared usage and `stage_budgets`; events record
per-request finish-only status and per-stage counters, including failed stages.
Failed runs cannot resume; use a new output directory for another attempt.
`scheduling.enabled = false` restores the earlier advisory-only behavior and is
recorded as an experimental change.

`llm.max_context_chars` (default 32,000) sets a character target for the conversation
view sent to the model. Older complete assistant/tool exchanges are omitted with a
deterministic working memory: successfully written paths, cached task/contract
previews, the two most recent Python-source pages, recent execution outcomes, and
the latest validation result. Observed edits invalidate cached pages for that file;
recorded code/candidate-write revisions distinguish old checks from later edits.
Report writes outside `solution/` do not invalidate source-check evidence. Cached
pages are observations, so agents must re-read if a script itself changed source.
The current handoff and latest exchange, including native reasoning state, remain
intact; a large latest exchange can exceed the target, which is logged explicitly.
This is not an exact provider-token limit. Full controller histories and original
events are retained; this policy also applies to single-agent mode.

For synthetic experiments, `read_file` previews at most 8,000 characters of source
text or 1,000 characters under `data/`. Optional `offset` and `max_chars` allow deliberate pagination; full
datasets remain available to generated Python scripts. Script observations contain
at most 4,000 stdout and 2,000 stderr characters, with truncation flags. The existing
executor-captured output is retained in `execution_output` events. Agents should
save detailed reports to files and print compact summaries.

The current DeepSeek development pilot allows 600,000 observed tokens, 80 model
calls, 240 tool calls, and 1,800 seconds across the whole workflow. Its allocations
are:

| Role | Token allowance | Model calls |
| --- | ---: | ---: |
| Diagnosis | 90,000 | 12 |
| Features | 90,000 | 12 |
| Modeling | 270,000 | 36 |
| Auditing | 150,000 | 20 |

Earlier attempts used 100,000/40 and 300,000/60 without enforced partitions. The
current allowance is for integration testing, not a completed research benchmark. Paired
single-agent/multi-agent experiments must use identical budgets, context settings,
model, and execution resources. Changing the character target changes how much
working information can be retained; making it too small can cause repeated reads
and checks even though individual requests become smaller.

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

The adapters have fixture-based protocol tests. Real four-role DeepSeek runs have
completed in Docker on synthetic data (`runs/deepseek-synthetic-06/`) and the local
CRSP development panel (`runs/deepseek-crsp-dev-01/`, summaries-only feedback).
Results and limitations are summarized in [DEVELOPMENT_LOG.md](DEVELOPMENT_LOG.md).
Real-provider single-agent comparisons and the other live backends remain unverified.

## Frozen Single/Multi Comparison

Use the paired driver for comparisons after protocol changes, rather than reusing
an earlier integration run as one arm. Preparation snapshots framework sources,
the normalized configuration, and a private copy of the JSON panel, and records
their SHA-256 fingerprints. It does not read credentials or access Docker/API:

```bash
.venv/bin/python eval/run_comparison.py --config configs/deepseek_crsp_dev.toml --output-root runs/crsp-paired-next --prepare-only
```

Execute that unstarted plan once from a Docker-capable terminal:

```bash
.venv/bin/python eval/run_comparison.py --execute-prepared runs/crsp-paired-next --key-file key.txt
```

Both arms run in fresh controller processes using the frozen framework and data.
Only `mode` differs; the model, summaries-only policy, executor and global limits
are the same. The Docker image is resolved once and its image ID is passed to
both arms. Input fingerprints are checked before and after conditions. Conditions
run sequentially to avoid competing for local resources; `--repetitions N` during
preparation alternates single/multi order across repetitions. Dataset seeds remain
fixed; the provider's model alias does not guarantee immutable model weights or
deterministic LLM sampling.

`protocol.json` contains the fingerprints/order; `comparison.json` records every
attempt's status, resources, and metrics, with reports under `repeat-01/single/`
and `repeat-01/multi/`. Individual failures are retained and the other arm still
runs; protocol tampering invalidates the comparison and stops execution. Started
or interrupted comparisons cannot be resumed. Unused prepared plans may be
executed later if their driver, sources, configuration and data remain unchanged.
Keep the plan directory private: its `private/` snapshot contains real data.
Choose an unused output root for new preparation. `runs/crsp-paired-04/` has now
completed both arms under the package-aware worker and controlled diagnostic
protocol. Both passed fixed audits; test MAE was slightly worse than zero-return
forecasts at all horizons. Multi used 2.74 times the observed tokens; this single
pair establishes no general advantage. See [the development log](DEVELOPMENT_LOG.md)
and the local aggregate `runs/crsp-paired-04/results_review.json` for details.
Plans 02 and 03 retain their failed multi outcomes; plan 03 exposed a worker
package-loading incompatibility and insufficient validation diagnostics. None
can be resumed, and arms from different protocol versions must not be pooled.

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

`solution/features.py` must also export
`features_at(history, origin) -> {symbol: {feature_name: finite_number_or_null}}`.
It receives an explicit origin and must ignore later dates even if they are
present in `history`. Feature names must be consistent across symbols.

Before final `auditing`/`single` completion, the current validated source must pass
`audit_candidate`. The fixed audit chooses interior development origins, actually
changes future prices, checks feature equality against truncated and perturbed
histories, and verifies repeated predictions/schema in fresh workspaces. It returns
only aggregate evidence, not feature values. Both validation and audit are tied to
the source hash; edits require both to be repeated. These are sampled development
checks, not an exhaustive proof of causality or predictive superiority. Put authored
audit scripts outside `solution/` to preserve the validated candidate hash.

Rows contain `symbol`, `horizon`, `prediction`, `risk`, `abstain`, `fallback`,
and `prob_up` (the probability of a strictly positive horizon return).
The runtime writes a detailed contract into each workspace. Fit receives only
history through its cutoff, so any constructed forward labels must mature by that
cutoff; under walk-forward it is called once per fold and per test block. Every
predict call gets a fresh workspace and only history through the current origin;
future prices and final labels are not mounted. Persist model state in returned
JSON, not mutable files or Python globals.

Point-forecast metrics are MAE, MSE and RMSE; `MSE = RMSE**2`, so adding MSE
does not change the ranking of a fixed set of forecasts. MSE is also included
in risk-coverage curves and matched zero-baseline comparisons.

For direction, the controller derives `y = int(target > 0)`; negative and exactly
zero returns both have label 0. New candidate validation, audit and final scoring
require finite `prob_up` values in `[0,1]` for every stock/horizon. Learn/calibrate
these probabilities using only matured historical labels and development
validation. A separate classifier is allowed. Neither the return forecast nor
the generic risk score is automatically a probability.

`direction.raw` and `direction.served` report binary `cross_entropy` (natural
logs, probabilities clipped to `[1e-15, 1-1e-15]`) and accuracy (up if `p >= 0.5`).
The no-information direction baseline predicts `p=0.5`, giving cross entropy
`ln(2)`, approximately 0.693147. It is separate from predicting a zero return.
`half_probability_baseline` and `half_probability_baseline_served` use exactly
the corresponding raw/served rows. Fallback uses probability 0.5 for served
direction scores; raw direction scores retain the candidate probability.
Abstentions are excluded only from served scores.

Offline rescoring accepts legacy saved rows without probabilities and reports
unavailable cross entropy/accuracy (`null`, with `probability_count`). It never
infers probabilities from signs, returns or risk, and never silently scores only
a subset of rows with probabilities. Recompute metrics without API/Docker calls:

```bash
.venv/bin/python eval/rescore_reports.py \
  --input-root runs/crsp-paired-04 \
  --output runs/crsp-paired-04/metrics_v2.json
```

`--input-root` accepts a completed run or a finished comparison. Use a new output
file: existing files cannot be overwritten. Source reports and frozen protocols
stay unchanged, and failed conditions retain missing metrics. This workspace's
`metrics_v2.json` already contains the MSE backfill; original pair 04 has no saved
probabilities, so its candidate cross entropy remains unavailable. New live pairs
need a new frozen plan under the updated probability-output contract.
The next local plan is prepared at `runs/crsp-paired-05/` (one single/multi pair;
zero API calls during preparation). Execute it with
`.venv/bin/python eval/run_comparison.py --execute-prepared runs/crsp-paired-05 --key-file key.txt`.
Execution makes paid requests; the plan has not yet been run.

Abstention excludes a row from served-error metrics. Fallback replaces the served
prediction with zero. Compare `raw` against `zero_baseline` on all requested rows,
or `served` against `zero_baseline_served` on the exact same non-abstained rows.
`served_count` and `abstained_count` expose the selection sizes. Never compare a
selected-sample MAE with the all-row zero baseline: dropping volatile rows can make
the apparent improvement entirely a selection effect. `raw` means all predictions,
including abstained ones; it does not mean a model with shrinkage disabled.
The zero-return baseline (`baseline_report.json`) is fixed code, so it
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

Both windows use a 70/30 walk-forward protocol. Local agent workspaces contain the
first seven years; the LLM receives controlled summaries. Validation uses four
yearly folds, each trained on all earlier years. The last three
years are the hidden test, with the pipeline refitted each January. Validation folds
are 2008–2011 and 2018–2021; test periods are 2012–2014 and 2022–2024.

For a real DeepSeek development run, use the existing local 2005–2014 panel:

```bash
PYTHONPATH=src .venv/bin/python -m forecast_agents run --config configs/deepseek_crsp_dev.toml --key-file key.txt --output runs/deepseek-crsp-dev-01
```

Every JSON-panel experiment automatically enables **summaries-only LLM feedback**
and records `llm_data_access = "summaries_only"` in `run.json`. The raw development
panel remains in local Docker workspaces; final-test data remain outside agent
mounts. Agents still write code that loads local data and trains models. The LLM
receives the following controlled feedback:

- `describe_data`: dimensions, schema, development dates and rounded pooled return
  statistics. It accepts no row filters and returns no prices or observation series.
- `run_python`: success/failure, exit status, timeout, and sanitized error type and
  known source line locations. Exact builtin TypeError templates may also return
  fixed type-only hints (for example, numeric checks on None during feature warmup).
  KeyError hints disclose a key only if it was already an unchanged agent-authored
  literal or a declared horizon. Arbitrary values/messages and stdout/stderr remain
  in local audit logs.
- `validate_candidate` / `read_validation_summary`: rounded aggregate and fold
  metrics. The full `validation.json`, including prediction/target rows, stays local.
  Worker failures from validation/audit return the same controlled execution
  diagnostics, including the failed operation; raw tracebacks stay local. Compacted
  memory replaces earlier successful validation/audit records with the latest failure.
- `read_file`: original task/contract snapshots or unchanged agent-written files.
  Raw data, runtime-produced reports, and files modified by executed scripts are
  blocked. Apply source edits through `write_file`, not patch scripts. File lists
  include only known inputs and agent-written paths, not arbitrary runtime filenames.

Evaluation workers load `solution` as a package, supporting both local absolute
imports (`from features import ...`) and relative imports (`from .features import ...`),
with or without `solution/__init__.py`. Model state is opaque JSON: test its use by
`predict` rather than assuming internal model keys exist for empty histories.

This is an enforced tool-output boundary, not just a prompt instruction. Metadata,
aggregate statistics, code, and scores still reach the provider; vendor licensing
for those summaries must be respected. Full local logs/reports may contain raw
data and must not be uploaded wholesale. This run uses 600,000 tokens/80 calls,
with a longer 7,200-second development budget and 3,600-second final-evaluation
limit for yearly refits and per-origin Docker execution. It does not use the
reserved 2015–2024 final research window.

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
- Raw extracts, panels, and local run directories contain CRSP data. JSON-panel
  runs withhold raw tool output from the LLM and return controlled summaries as
  described above. This technical boundary does not itself determine licensing
  permission; check the applicable terms before sharing results or files.

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
