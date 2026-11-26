# Development Progress and Next Steps

Updated November 26, 2026.

The initial experimental framework is implemented and runs end to end with
synthetic data and a deterministic MockBackend. Both single-agent and four-role
workflows complete forecasting and independent scoring. The runtime now runs on
macOS, Linux, and Windows, and a CRSP/WRDS data layer builds the panel from CRSP.
Real DeepSeek four-role runs now complete on both synthetic data and the local
CRSP development panel, with generated code executed in Docker, a frozen
submission, and independent scoring. Real-data runs return controlled summaries
to the LLM. The first CRSP run is approximately tied with, slightly worse than,
zero-return forecasts on matched-sample MAE. A discovered selection-comparison
error has been corrected in the evaluator feedback. The first fully completed
frozen real-data single/multi pair is now available at `runs/crsp-paired-04/`.
Both arms are slightly worse than zero-return forecasts on test MAE; no
multi-agent advantage has been established.

## Current Handoff Summary

The framework supports single-agent and sequential four-role AutoML, generic LLM
backends, shared/role budgets, local code generation, Docker execution, immutable
candidate submissions, and chronological independent scoring. Real-data feedback
is restricted to controller-produced summaries; raw rows and arbitrary script
output remain local. The current regression suite passes 126 tests.

Two real DeepSeek integration pilots completed: a synthetic run and the 30-stock
CRSP development run `runs/deepseek-crsp-dev-01/`. The latter used 44 model calls,
322,057 tokens and about 17.3 minutes including final scoring. Its same-sample MAE
was approximately tied with, slightly worse than, zero-return forecasts. Claims
of a 2–3% gain were invalid because served forecasts and the baseline used
different rows. The evaluator now exposes the matched served baseline, and the
original run is preserved with a separate corrected review artifact.

Two executed frozen pairs (02 and 03) completed with a single-agent success and a
multi-agent failure. Pair 02 involved invalid agent-authored tests; pair 03 also
exposed an evaluator package-loading incompatibility and opaque validation errors.
Both are preserved, including failed-run costs. Pair 04 completed both arms under
the corrected worker protocol, passing fixed audits and independent scoring on
identical 2012–2014 origins. Single used 147,710 tokens; multi used 404,682.
Multi's MAE was slightly worse at 1/3 sessions and slightly better at 5, while
both arms were slightly worse than the zero-return baseline at all horizons.
Neither abstained or used fallback on test. The next work is prespecified manual
ML baselines, model-fit/compute accounting, repeated paired trials with uncertainty
estimates, and controlled reliability scenarios. One pair does not establish a
general advantage. The 2015–2024 final research window remains reserved.

MSE and direction binary cross entropy/accuracy are now implemented. Future
candidates must provide `prob_up`; pair 04's saved forecasts have no probabilities,
so its MSE was backfilled but direction scores remain unavailable. No new live
experiment was run for this addition.

### Completed in This Continuation

- Added the explicit `features_at(history, origin)` feature contract and mandatory
  fixed `audit_candidate` check. The audit uses interior development origins,
  verifies that future rows actually change, compares full/truncated/perturbed
  feature outputs, and checks prediction schema/repeatability in fresh workers.
  Final completion requires the audited hash to match the validated/frozen source.
  Checks remain sampled development evidence, not exhaustive reliability proof.
- Added `eval/run_comparison.py` with preparation separate from execution. It
  snapshots framework sources, configuration and private data, verifies input
  fingerprints, pins one Docker image for both arms, and runs counterbalanced,
  sequential single/multi pairs in separate controller processes. Failures and
  invalidated attempts are retained; no started comparison is resumable.
- The 107-test suite passed. Added negative tests for future-dependent features, stale audit
  evidence, altered snapshots, mid-run protocol changes, and failed conditions.
  A persisted mock comparison completed both arms at
  `runs/frozen-pair-smoke-01/`; it validates the new driver and audit path only.
- Prepared the initial real-data plan at `runs/crsp-paired-02/` using the 2005–2014 panel.
  This preparation used no credentials, Docker or paid API requests. It contains
  private raw data and must stay local. That plan has now been executed and is
  retained. Plans 03 and 04 were subsequently prepared and executed; plan 04 is the
  first to complete both arms. Started plans cannot be executed again.

Pair 04 ran one single-agent and one multi-agent condition, each with the frozen
600,000-token/80-call global limit. More repetitions and statistical analysis remain
subsequent research work. The earlier CRSP run used different feedback/audit rules
and remains an integration pilot; it is not a valid substitute for the new pair's
multi-agent arm. The reserved 2015–2024 final window is unchanged.

Two additional guards were verified before final preparation: closing corrections
leave mandatory validation/audit tools available, and failed revalidation revokes
earlier approval instead of accepting stale evidence. The unstarted
`crsp-paired-01` preparation was marked superseded; neither preparation made paid
requests. Plan 02 was subsequently executed by the user. Its framework SHA-256 is
`970a19f535b935067feba4943a0b7abb188dcb12dbecbd562aee06977c252178`,
and its private dataset matches the existing CRSP fingerprint
`ff8d078c3cff1748e17753f527934bce83f3fbc7faed3476e152ccd047cc8a4d`.

### November 26, 2026: First Frozen Pair and Controlled Debugging Feedback

`runs/crsp-paired-02/` ended `completed_with_failures`. Single completed validation,
fixed audit, and final scoring using 7 calls and 45,067 tokens. Multi failed after
55 calls and 382,125 tokens, without creating `solution/pipeline.py` or reaching
the audit role. Neither condition exhausted the global token ceiling. Preserve
both records, including resources spent on the failed condition; no multi forecast
score exists and it must not be replaced with a successful zero-baseline result.

The multi failure was an agent development error: its feature helper correctly
returned None for 20-session ranks on a three-session fixture. Authored edge tests
then called `math.isfinite(None)` and compared two None values, raising TypeError.
Features requested revision to modeling; modeling continued debugging the invalid
fixture instead of producing a pipeline and requested a second revision, exceeding
the existing limit of one. Full-pipeline audit calls during feature-only work also
failed because no pipeline existed. The local-only feedback had exposed exception
class/line but insufficient safe type information to identify this mistake easily.

Added exact-template, builtin-type diagnostics for numeric None, incompatible
comparisons, and non-subscriptable values. No raw values or arbitrary error text
are released. Unknown/type-name/value-bearing messages remain withheld. Missing
pipeline validation/audit attempts now return fixed actionable preconditions, and
the feature contract/prompt explicitly describes nullable warmup and compatible
fixtures. Working memory retains the safe hints. Budget and revision limits are
unchanged; this does not convert the failed run into a success.

All **110 tests pass**, including exact safe-type recognition and secret-sentinel
suppression for nonwhitelisted messages. `failure_review.json` adds an aggregate
review beside the original plan; frozen sources, data, protocol and run reports
are unchanged. The next plan, `runs/crsp-paired-03/`, is prepared with the new
feedback rules and zero API calls. Execute both arms under that protocol; do not
rerun only multi and pool it with the older single result. Both plans are
development pilots, and a single pair is insufficient for a general superiority
claim. The reserved final research window remains untouched.

### November 26, 2026: Second Frozen Pair and Package-Aware Evaluation

`runs/crsp-paired-03/` ended `completed_with_failures`. Single completed using
53 calls and 472,620 tokens. Its raw MAE was about 0.005%, 0.005%, and 0.020%
below the same-row zero-return baseline for 1/3/5 sessions, respectively; these
tiny differences have no established statistical significance. Multi failed
after 48 calls and 382,288 tokens, before auditing. Modeling consumed 267,579 of
its 270,000-token role allowance; the next input was rejected even though the
global budget had room. This was not a global token exhaustion.

Unlike pair 02, a pipeline existed. It used `from .features import ...`, which
worked in its package-based local smoke script. The evaluator instead loaded it
as the bare module `candidate_module`, causing ImportError on four validation
attempts. Protected feedback exposed only generic RuntimeError, obscuring the
cause. A separate smoke-test KeyError came from assuming all horizon entries
existed inside the opaque state returned for empty training history. Modeling
also requested revision to itself after entering permanent finish-only mode,
then spent more calls validating unchanged code. Classify this run as affected
by a runtime compatibility defect alongside agent test/closing behavior; it
does not establish that the multi-agent design is intrinsically inferior.

The worker now loads a proper `solution` package and registers modules through
Python's normal importer, supporting relative and absolute local imports with
or without `__init__.py`. Candidate execution failures retain their full output
only in local logs and expose controlled exception class, authored source
locations, failed operation and fixed hints. KeyError may identify only a key
already present in unchanged authored source or public horizons. Failed checks
replace previous successes in compacted memory. Revisions to a finish-only role
now fail immediately instead of reopening a loop that cannot edit code. The
contract clarifies package imports and opaque model-state testing.

All **117 tests pass**, including package-relative fit/predict/features with
namespace and regular packages, dataclasses, safe failure propagation, unknown
key suppression, memory invalidation and the closing revision guard. Tests use
hand-written synthetic fixtures; no live generated code was executed on the host
and no paid API requests were made. Original pair 03 artifacts remain unchanged,
with an aggregate `failure_review.json` beside them. The new protocol is prepared
at `runs/crsp-paired-04/`, with unchanged global/role budgets and zero API calls.
It was unexecuted at preparation; its subsequent results are recorded below.
Run both arms under each snapshot; preserve earlier
failures and do not pool arms across protocol versions. The reserved final
2015–2024 research window remains untouched.

### November 26, 2026: First Fully Completed Real-Data Pair

`runs/crsp-paired-04/` completed both conditions using DeepSeek Flash and Docker
on the local 30-stock CRSP panel. Both submitted candidates passed the fixed
development audit. Review verified frozen input fingerprints, identical configs
except mode, a shared pinned Docker image, identical scoring rows/targets, and
matching validated/audited/frozen source hashes. Recomputing metrics from local
prediction rows reproduced the saved reports. Original artifacts remain unchanged;
aggregate checks and results are in `runs/crsp-paired-04/results_review.json`.
The review made no API calls and executed no candidate code.

The development task's independent test covers 749 origins from January 3, 2012
through December 23, 2014: 22,470 stock/origin forecasts per horizon. MAE is in
decimal-return units; lower is better, and all three columns use identical rows.

| Horizon | Zero-return MAE | Single MAE | Multi MAE |
| --- | ---: | ---: | ---: |
| 1 session | 0.008541989 | 0.008546701 | 0.008548486 |
| 3 sessions | 0.015129702 | 0.015145159 | 0.015159630 |
| 5 sessions | 0.019842965 | 0.019880721 | 0.019872942 |

Both conditions are slightly worse than zero forecasts: single by approximately
0.055%/0.102%/0.190% MAE and multi by 0.076%/0.198%/0.151%. Relative to single,
multi has 0.021%/0.096% higher MAE at 1/3 sessions and 0.039% lower MAE at 5.
These are descriptive differences from one pair, without significance estimates.
Both candidates use authored ridge regression with prediction shrinkage, rather
than a broad ML/DL search. Test coverage is 100% with zero abstentions/fallbacks;
the operational selective-prediction behavior was not exercised in this window.

| Resource | Single | Multi |
| --- | ---: | ---: |
| LLM calls | 17 | 52 |
| Observed tokens | 147,710 | 404,682 |
| Tool calls | 20 | 62 |
| Agent workflow seconds | 285.6 | 461.8 |
| Independent final scoring seconds | 376.9 | 398.0 |
| Controller start to completion minutes | 11.05 | 14.34 |

The full comparison took about 25.4 minutes. Identical resource ceilings did not
produce equal actual use: multi consumed 2.74 times the tokens. Token counts are
not a dollar-cost estimate because provider cache/input/output pricing differs.

Next: add prespecified non-agent ML baselines and explicit model-fit/compute
accounting, freeze the protocol/repetition count, then run counterbalanced pairs
and estimate uncertainty while accounting for temporal/cross-stock dependence.
Define reliability scenarios and validation-only abstention calibration before
evaluating them. Keep the balanced-universe limitation explicit, preserve failed
pairs 02/03 as development history, and leave the 2015–2024 final window untouched.
No new experiment was prepared or launched during that review. At that point,
production code and the last verified 117-test regression result were unchanged.

### November 26, 2026: MSE and Return-Direction Classification

Added MSE alongside MAE/RMSE for raw forecasts, served forecasts, both matched
zero baselines, per-fold reports and risk-coverage curves. MSE is RMSE squared;
its addition cannot improve or reorder the same fixed forecasts relative to RMSE.

Added direction labels `y = int(return > 0)` (zero returns are class 0) and the
`prob_up` prediction field. New live candidates must provide a finite probability
in `[0,1]` for every stock/horizon. Validation, fixed audit and final scoring all
enforce the field; controlled contract errors identify missing/invalid output
without exposing raw data. Probabilities must be trained/calibrated on matured
historical labels and development validation, not final test targets.

Direction scores include binary cross entropy (natural logs; clipping at
`1e-15` and `1-1e-15`) and accuracy at `p >= 0.5`. The no-information direction
baseline predicts 0.5 (cross entropy approximately 0.693147), separate from the
zero-return regression baseline. Raw scores retain candidate probabilities;
served scores exclude abstentions and substitute probability 0.5 for fallback.
Matched baseline rows are reported for each. Legacy rows with missing
probabilities have unavailable scores, with availability counts; neither returns
nor generic risk values are silently converted into probabilities. Partial
availability does not produce a selectively scored cross entropy.

The mock fixture now supplies a constant direction probability estimated from
its matured training labels with Laplace smoothing. It exercises the interface,
not a sophisticated or validated directional strategy. The fixed zero fixture
supplies probability 0.5. No feature-window restriction or frozen model was changed.

Added `eval/rescore_reports.py` for local aggregate rescoring of a saved run or
finished comparison. It records source hashes, refuses existing output files,
does not run candidate code or APIs, and preserves failed-condition status.
Backfilled pair 04 into `runs/crsp-paired-04/metrics_v2.json`:

| Horizon | Zero-return MSE | Single MSE | Multi MSE |
| --- | ---: | ---: | ---: |
| 1 session | 0.000145162554 | 0.000145310403 | 0.000145330118 |
| 3 sessions | 0.000435916346 | 0.000436577613 | 0.000436678537 |
| 5 sessions | 0.000716229690 | 0.000718366962 | 0.000717253653 |

Both remain slightly worse than zero forecasts. Pair 04 contains no `prob_up`
values, so its candidate cross entropy is unavailable; only the explicit 0.5
direction baseline can be scored retrospectively. Existing reports, protocol,
candidate code and accuracy claims remain unchanged. Future paired experiments
need a new frozen protocol containing the probability-output contract.

All **126 tests pass**, including known MSE/BCE values, zero-return labels,
probability bounds/completeness, endpoint clipping, missing/partial availability,
abstention/fallback samples, end-to-end mock probability output, safe contract
feedback, and non-destructive rescoring with failed conditions. This work made no
paid requests and executed no live generated candidate code.

Interpretation of the small predictive differences remains limited: both agents
used price-derived features, authored Ridge models and shrinkage toward zero.
The default Python image has no configured third-party ML/DL stack; this pilot
has not tested a broad model-family search. Those facts motivate richer fixed
baselines/model environments, but do not quantify the cause of weak performance
from a single pair. Preserve the 2015–2024 final window and avoid tuning against
the already inspected development task's 2012–2014 test scores.

### November 26, 2026: Probability-Metric Pair Prepared

Prepared `runs/crsp-paired-05/` with the updated MSE/probability-output protocol.
The plan contains one single/multi pair, the same CRSP development data and
unchanged resource ceilings. Preparation verified frozen inputs and made zero
API requests; neither arm has been executed. To obtain direction cross entropy,
run `.venv/bin/python eval/run_comparison.py --execute-prepared runs/crsp-paired-05 --key-file key.txt`.
Execution makes paid API requests. Pair 04 remains unchanged; do not substitute
one arm across the two protocol versions. Its MSE already exists in the separate
backfill, so rerunning is needed for new candidate probability forecasts, not MSE.

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

The initial local verification passed **27 tests** through the standard-library
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
below. This initial snapshot predates the completed live run described next.

### November 25, 2026: First Completed CRSP Development Run

The user completed `runs/deepseek-crsp-dev-01/` with
`configs/deepseek_crsp_dev.toml`: 30 stocks from 2005–2014, four yearly validation
folds in 2008–2011, then yearly refits and daily test origins in 2012–2014. All four
roles returned `completed`. This is a real-data integration pilot, not a controlled
single/multi-agent comparison.

- **Resources:** 44 model calls, 57 charged tool attempts, 322,057 tokens (53.7%
  of 600,000). Agent development took 545.5 seconds and final scoring 494.8 seconds,
  about 17.3 minutes overall. Diagnosis used one bounded policy correction.
- **Data boundary:** `llm_data_access` is `summaries_only`. All seven recorded
  script-result messages withheld stdout/stderr; validation responses contained
  aggregates rather than prediction/target rows. No direct raw-data reads were
  requested. This is a log/implementation check, not a license certification.
- **Submission:** reviewed source hash matches
  `3967a271d8927b434804042732e15d7c85a722e6353271b2624032c7d9386e0d`.
  No model requests followed submission freeze. The candidate uses five causal
  features and ridge regression with cross-sectionally demeaned labels, lambda 60,
  shrinkage 0.7, and a relative-volatility abstention threshold of 2.5.
- **Test coverage:** 749 forecast origins, 22,470 rows per horizon, 67,410 total.
  Per horizon, 21,210 rows were served and 1,260 abstained: coverage 94.39%, no
  fallback. Every abstained row has risk above 3.5, consistent with the code's
  volatility threshold. The audit's explanation that abstention came from early
  fold warmup was not supported: prediction prefixes contain the earlier history.

| Horizon | All-row model MAE | All-row zero MAE | Model MAE increase |
| --- | ---: | ---: | ---: |
| 1 session | 0.008543423 | 0.008541989 | 0.0168% |
| 3 sessions | 0.015133811 | 0.015129702 | 0.0272% |
| 5 sessions | 0.019855924 | 0.019842965 | 0.0653% |

On the identical served subset, MAE increases versus zero are 0.0315%, 0.0639%,
and 0.1130%. These small differences do not establish statistical significance;
no MAE advantage was observed in this test window. Comparing served model MAE with
the all-row zero baseline incorrectly suggests 2–3% gains because different rows
are compared. The generated audit made this mistake and also misinterpreted `raw`
as a model without shrinkage. Its claim of 12/12 better validation cells becomes
3/12 under the matched served-sample comparison.

Added `zero_baseline_served`, `served_count`, and `abstained_count` to future
evaluator reports, and explicit same-sample instructions to tool/agent feedback.
All **95 offline tests pass**, including a case where selecting easier rows creates
a false apparent gain. Historical reports and candidate sources remain unchanged;
`matched_baseline_review.json` derives corrected test/validation comparisons from
existing local rows without model execution or API calls. This change affects
future agent feedback: freeze it and rerun both conditions before claiming a
matched comparison. The 2015–2024 final research window remains untouched.

### November 25, 2026: First Completed Live DeepSeek Synthetic Run

The user completed `runs/deepseek-synthetic-06/` with `configs/deepseek.toml`:
`deepseek-flash`, thinking disabled, four stocks, 120 synthetic sessions, seed 7,
and horizons 1/3/5. Diagnosis, features, modeling, and auditing all returned
`completed`; no revision or policy correction was used. Modeling and auditing
entered enforced finish-only mode and completed within their allocations.

- **Resources:** 63 model calls, 74 tool calls, 461,344 tokens (442,957 input and
  18,387 output), 76.9% of the 600,000-token allowance. The agent phase took 182.1
  seconds; final evaluation took 9.0 seconds, about 191.1 seconds overall.
- **Execution:** Docker image ID
  `sha256:7249d5587b78209a006dc4a3b5340678f84ea729239193aa559962cd8c1fce50`.
  The reviewed submission hash matches `run.json`:
  `4051d7360d0517e88ac7fc060440030beb8a5323baaf83195a2ee7425e27d753`.
  Logs contain four historical validation events and no model requests after
  submission freeze.
- **Candidate:** agent-written pooled ridge regression (`lambda=20`) using six
  causal return/volatility features and per-horizon shrinkage chosen on validation.
  The standardized final test contains 20 forecast origins, 80 predictions per
  horizon (240 total), with coverage 1.0 and fallback rate 0.0.

| Horizon | Test MAE | Zero-return MAE | Relative MAE reduction |
| --- | ---: | ---: | ---: |
| 1 session | 0.008856568 | 0.008983907 | 1.42% |
| 3 sessions | 0.020176281 | 0.020226350 | 0.25% |
| 5 sessions | 0.028842788 | 0.029733954 | 3.00% |

These are single-run synthetic integration results, not evidence of a robust
predictive edge or an advantage over a single agent. No abstention/fallback cases
were exercised. The generated audit also has two limitations: its independent
MAE calculation used 96/88/80 validation examples by horizon while the evaluator
uses a common 80, and its future-poisoning check chose the final visible date,
leaving no future rows to perturb. That check is vacuous despite printing success;
it is not valid evidence of pipeline causality. The evaluator still supplies only
causal prefixes in fresh workspaces. Align audit origin sets and require an
interior-origin perturbation that actually changes rows before stronger claims.

Artifacts are local and ignored by Git: `run.json`, `handoffs.json`,
`test_report.json`, `baseline_report.json`, `events.jsonl`, and `submission/` under
the run directory. Preserve this run; use a new directory for the next condition.

### November 25, 2026: Real-Data Configuration and Summaries-Only Feedback

The user clarified the intended boundary: the LLM calls local tools for
descriptive analysis, while raw observations remain local. This is now enforced
automatically for `data.source = "json"`, rather than relying on model instructions.

- Added `describe_data` for fixed-schema, rounded development-panel statistics
  and `read_validation_summary` for controller-cached aggregate metrics.
- Raw data and runtime-created reports cannot be read into the LLM. Only original
  task/contract snapshots and unchanged agent-written files are readable; hashes
  prevent raw-data copies or patched source from being returned as code. Runtime
  filenames are excluded from agent-visible file inventories.
- Python still loads data and trains models locally. Arbitrary stdout/stderr and
  detailed candidate exceptions stay in local logs. Model feedback includes only
  status, sanitized exception types/known source line locations, and approved
  aggregates. Validation target/prediction rows remain local. Source edits must
  use `write_file`; scripts cannot modify a file and then read that modified text
  back into the LLM.
- Added `configs/deepseek_crsp_dev.toml`: the existing 30-stock, 75,510-row CRSP
  2005–2014 panel, validated fingerprint, four 2008–2011 yearly validation folds
  (51/51/51/50 origins) and three 2012–2014 test blocks (250/252/247 origins).
  Tokens/calls remain 600,000/80. Longer development/evaluation limits account
  for the larger dataset and per-origin Docker jobs. The 2015–2024 final window
  remains untouched.
- All **93 offline tests pass**, including JSON-mode activation, sentinel values
  absent from captured model requests, copied reports/modified source blocked,
  withheld stdout/error values, immutable task snapshots, aggregate-only validation
  feedback, and no post-development data in descriptive summaries.
- The real-data DeepSeek run has not been executed here: this agent session cannot
  access `/var/run/docker.sock`. The user must launch the new config in the terminal
  where the synthetic Docker run succeeded. No raw CRSP observations were sent to
  DeepSeek during this preparation. Technical filtering does not by itself establish
  a vendor-license interpretation for aggregate results.

### November 25, 2026: DeepSeek Integration and Budget Scheduling

- Added `configs/deepseek.toml` for `deepseek-flash` using the OpenAI-compatible
  API, synthetic data, Docker execution, and explicitly disabled thinking.
  The controller can read a git-ignored `key.txt` through `--key-file`; generated
  code does not inherit credentials. The user verified authentication and a
  two-request tool-call continuation using `scripts/check_deepseek.py --probe`.
- The first live synthetic multi-agent attempt reached Docker execution but failed
  during diagnosis: 19 model calls, 103,194 input tokens and 9,061 output tokens,
  exceeding the 100,000 shared observed-token limit. Diagnosis repeatedly expanded
  its probes and never handed off. Input history was charged on each call,
  including cached tokens; this was not a successful forecasting experiment.
- Added refreshed budget notices to every request and focused role instructions.
  Advisory shares are 15% diagnosis, 15% features, 45% modeling, and 25% auditing;
  single-agent mode sees the full shared allowance. Wrap-up reminders preserve
  essential checks and explicit handoffs. These are advisory targets, not forced
  stage acceptance or new allowances on revisions. Global limits remain unchanged.
- The CLI now prints recorded budget usage on success or failure. All **59 offline
  tests pass**, including notice refresh, shared-ledger continuity, and tool-message
  pairing. A new live attempt is still needed to verify end-to-end completion with
  these changes; failed output directories are preserved and cannot be resumed.
- The second attempt completed diagnosis but exhausted 101,520 tokens in 15 calls
  during feature engineering. A direct data read inserted about 32,000 characters
  into the conversation; subsequent requests each used about 20,000 input tokens.
  Budget reminders alone did not address repeated large observations.
- Added paginated file previews (1,000 characters for data, up to 8,000 for other
  text), bounded script observations, and a configurable 12,000-character target
  for model conversation views. Older exchanges are dropped as complete groups;
  current handoffs, recent native tool state, and a deterministic artifact summary
  remain. Full event records and controller histories are retained. The latest
  exchange can exceed the character target, and such cases are logged explicitly.
- All **65 offline tests pass**. Replaying the final request of the failed second
  attempt reduced the conservative context character measure from 77,360 to 10,001;
  this is not a measured live token reduction. The shared 100,000-token limit
  remains unchanged. This environment still cannot resolve the DeepSeek API host,
  so a new live run must verify whether these controls permit completion.
- The third attempt reached modeling but failed after 33 calls and 103,953 tokens:
  diagnosis used 10 calls/28,959 tokens, features 15/48,363, and modeling 8/26,631.
  Modeling repeatedly read task/contract/source without writing a pipeline.
  The first compaction policy discarded operational knowledge: file names alone
  did not tell the model what it had read or which checks had already passed.
- Replaced that summary with deterministic working memory containing observed
  protocol/source previews, recent execution outcomes, and validation evidence.
  Successful edits invalidate old cached pages; observed Python/candidate writes
  carry revisions so old checks remain distinguishable from checks of edited code.
  Report writes outside `solution/` do not invalidate source-check evidence.
  Cached previews describe the last observed read/write, so source changed by a
  script must be read again. The character target is now 32,000. Replaying the
  third trace retains the task, contract, and feature source during modeling.
- The DeepSeek development configuration now allows **300,000 observed tokens and
  60 model calls**, up from 100,000/40. This is a recorded integration-budget change,
  not a matched-budget research result; both comparison conditions must use the
  same updated settings. Completion can omit `next_role`; revisions still require
  an explicit target, and normal role transitions still include auditing.
- All **68 offline tests pass**, including preservation of protocol/check evidence
  through compaction and invalidation of stale source pages. End-to-end live
  completion remains unverified; the local network restriction is unchanged.
- The fourth attempt completed diagnosis, features, and modeling but exhausted
  301,317 tokens after 42 calls during auditing. Usage by role was 37,208, 51,094,
  176,188, and 36,827 tokens respectively. Advisory shares did not reserve enough
  of the original 300,000-token budget for the final audit.
- Added enforced scheduling in `runtime/scheduling.py`. Role allowances partition
  tokens, model calls, tool calls, and active time; revisions reuse the same role
  ledger. Other roles cannot spend the audit allocation. At 80% of tokens/time,
  two remaining calls/tools, or an estimated final-call token reserve, a role
  permanently enters finish-only mode. Only explicit handoff and, for modeling,
  auditing, and single mode, candidate validation remain available. Hidden-tool
  attempts fail before executing any tools from that response; stages are never
  accepted automatically.
- Admission estimates reserve input as well as output tokens. Actual provider
  usage remains authoritative: an unexpected overshoot is charged and fails the
  stage before tool execution. This is an observed limit, not a guaranteed billing
  ceiling. `run.json`, CLI output, and events now include cumulative stage budgets
  for both successful and failed stages. Execution deadlines use the smaller of
  the active role's remaining time and the shared remaining time.
- With user approval, the current integration profile raises the shared limits to
  **600,000 tokens, 80 model calls, 240 tool calls, and 1,800 seconds**. Role token
  allowances are 90,000 diagnosis, 90,000 features, 270,000 modeling, and 150,000
  auditing. Single-agent mode gets the same overall limits and closing policy.
  `[scheduling]` and all derived usage records make this change reviewable.
- All **78 offline tests pass**. Added checks cover audit allocation, cumulative
  revision budgets and paused role clocks, admission without an API call when
  allowance is insufficient, response overshoots, tool restrictions, and permanent
  finish-only mode. Existing single/multi mock experiments still complete and
  produce identical forecasting sources. A new live synthetic run is needed to
  verify complete DeepSeek execution under this policy.
- The fifth attempt stopped at 87,401 tokens/16 calls, rather than exhausting the
  global allowance. The feature agent had written a full predictor instead of a
  compact helper; its smoke test raised `AssertionError`. At 10 of 12 feature
  calls, finish-only activated. The next response nevertheless requested two
  unavailable reads, and the initial strict policy aborted immediately.
- Added explicit provider `tool_choice` support and enabled it only in the current
  non-thinking DeepSeek profile. Closing feature/diagnosis requests name
  `finish_stage`; other closing roles require an offered tool. Thinking-mode
  incompatibility is checked locally for DeepSeek. Other providers retain default
  tool selection unless explicitly configured for this capability.
- A blocked closing batch now receives matching tool-error feedback without any
  tools executing. One correction is permitted by default, charged to existing
  model/token/tool allowances; repeated violations still fail. Plain-text endings
  and invalid handoffs are similarly bounded, and the next correction forces
  handoff when supported. Corrections persist across revisions.
- Feature prompts now exclude trained predictors/model search. A latest observed
  failed Python check prevents a `completed` feature handoff. Unresolved work must
  be repaired or explicitly routed to modeling as `revise`, consuming the normal
  revision limit; it is never silently treated as a passed check. The shared
  600,000-token budget and per-role allocations are unchanged.
- All **85 offline tests pass**, including provider payload mapping, rejection of
  incompatible thinking settings, matching error feedback for the rejected batch,
  bounded correction, and failed-check revision handoff. Real-model end-to-end
  completion is still unverified in this environment.

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

- **Real execution:** four-role DeepSeek synthetic and CRSP runs have completed in Docker.
  Other real providers, the real single-agent condition, and cross-platform Docker
  operation are not yet verified. Local execution is for trusted mock scripts.
- **Data:** the loader requires a balanced panel, so CRSP panels keep only stocks
  with every session, which biases them toward survivors; final windows need entries
  and exits. Local model code receives total-return indices, while the LLM receives
  summaries; volume, market-cap, and market-index fields remain unused. The legacy SIZ tables end on
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

### Priority 0 Verify the Real Execution Path (Initial Multi-Agent Run Complete)

Select one accessible model and a Docker-capable machine. Start with synthetic
data, the existing adapter, and the same model for all four roles. Check tool-call
compatibility, code execution, historical validation, stage handoffs, and submission.

**Completion criteria:** a real-model run completes without scripted mock actions;
generated code executes in Docker; image identity and resource usage are recorded;
final evaluation finishes without exposing test feedback to agents. A timeout or
invalid submission must produce an explicit failed-run record.

**Status (November 25):** completed for DeepSeek synthetic and CRSP development
four-role runs. Next, freeze the corrected comparison feedback and run both
single-agent and multi-agent conditions with identical data, model, budgets, and
executor settings. Successful integration does not establish comparative performance.

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
