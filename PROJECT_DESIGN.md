# Project Design and Experimental Plan

This document records the design agreed during the project discussion on
November 16, 2026, with engineering refinements from November 17, 2026. It develops the direction in
[the project proposal](<CS7652 Project Proposal.docx>) into an implementation and
evaluation plan. The project is being designed for the final research experiments.
The architecture below is agreed; numerical settings and candidate experiments
identified as provisional still require decisions. No implementation or results
are implied by this planning record. The initial implementation added on
November 21, 2026 is described in [README.md](README.md), including its tested
scope and remaining limitations.

## 1. Research Questions

Under the same total resource constraints, does assigning specialized agents to
data diagnosis, feature engineering, modeling, and auditing improve short-term
stock return forecasting compared with a single agent performing all those tasks?

Evaluate two outcomes separately:

1. **Forecasting performance:** whether specialization produces better predictive
   pipelines and model-selection decisions.
2. **Reliability:** whether the system identifies likely failures and makes useful
   abstention or fallback decisions.

An improvement in reliability can be meaningful even without an improvement in
unconditional prediction accuracy.

## 2. Forecasting Task

The task is to predict cumulative returns over the next **1, 3, and 5 trading
days**, rather than a 135-day horizon. Each stock and forecast origin has three
targets. Conceptually, for a consistently defined price series:

```text
y[i, t, h] = P[i, t+h] / P[i, t] - 1
h in {1, 3, 5}
```

The final implementation must specify the source return or price fields and how
corporate actions and delistings are handled. A provided daily return field may
instead be compounded over each horizon. These choices must be consistent across
all methods.

Each submitted forecast should identify the stock, forecast origin, horizon,
predicted return, reliability or risk score, and abstention/fallback decision.
Reliability is evaluated separately for each horizon.

### Shared Modeling Across Stocks

The main experiments will use a pool of stocks to train shared models, with
predictions produced separately for each stock. A candidate solution may train
one shared model per horizon or a model with three outputs. Shared modeling does
not require every stock to have identical behavior or features.

This setting gives the agents opportunities to investigate cross-stock patterns,
stock-specific differences, and heterogeneous data quality. It also allows one
agent workflow to develop a pipeline for an entire stock pool. Whether sharing
improves accuracy must be established empirically.

Per-stock modeling is a supplementary baseline or sensitivity experiment. The
main research comparison remains single-agent versus specialized multi-agent
development under a common modeling setting.

## 3. Specialized Agents

| Role | Responsibilities | Typical artifacts |
| --- | --- | --- |
| Data diagnosis | Analyze missingness, outliers, available history, distribution changes, and temporal structure; recommend further investigation and treatment. | Diagnostic scripts, reports, evidence, and recommendations. |
| Feature engineering | Implement transformations, lagged and rolling features, eligible external variables, and sequence representations appropriate to candidate models. | Executable feature pipelines and feature documentation. |
| Modeling | Develop and compare multiple ML and DL approaches; train, tune, validate, and revise models; request changes to input representations. | Training and prediction code, candidate models, validation results, and forecasts. |
| Auditing | Review leakage, validation design, robustness, and reliability claims; independently check supporting evidence. | Check scripts, audit findings, revision requests, and risk assessments. |

All roles can call shared tools and write and execute code. Their specialization
comes from their responsibilities, prompts, and separate contexts. The modeling
role initially covers both ML and DL; splitting it into separate ML and DL agents
is a possible later extension.

Feature engineering and modeling must support feedback. For example, a modeling
agent may request different lag windows or a sequence representation after an
unsuccessful validation experiment. DL candidates need not use the same manually
engineered inputs as tabular ML candidates.

### Initial Coordination Pattern

```text
Data diagnosis -> Feature engineering <-> Modeling -> Auditing
                        ^                   ^           |
                        +---- revision requests --------+
                                                        |
                                           accepted -> submission
```

A programmatic scheduler will coordinate the initial workflow. Revision loops
must have explicit resource and termination limits. The rule for submission or
fallback when the budget expires or an audit remains unresolved is still to be
specified.

## 4. Shared Tools and Autonomous Coding

Provide a stable experimental environment with:

- Data access, metadata, and time-splitting utilities.
- Metric functions and simple baseline examples.
- File operations, Python execution, logs, and process management.
- Resource accounting and submission-format validation.

Agents remain responsible for writing and revising analysis, feature, training,
and evaluation code. Tools should support this work without predetermining the
entire pipeline. Freeze dependencies and access rules for each experiment.

Keep the runtime, task definitions, and final evaluator separate from agent code.
The auditor is an agent within the system and can make mistakes; the evaluator is
external infrastructure that computes the final results.

Agent development frequency and prediction frequency are separate. An agent team
develops a pipeline for an experimental window; the runtime then executes daily
predictions and any permitted retraining. Do not launch a new team for every
stock on every trading day by default. The refit and agent revision schedules
must be fixed before the final experiments.

## 5. Evaluation Isolation and Leakage Prevention

- Split all stocks using common chronological boundaries; do not randomly split
  stock-date rows for the main evaluation.
- At each fitting or prediction time, use only features and training labels that
  would already be available. A five-day label matures only after its full horizon.
- Fit learned transformations on the corresponding training partition. Historical
  observations may supply causal lookback context without exposing future labels.
- Separate tasks/windows used to develop prompts and coordination strategies from
  tasks/windows reserved for final evaluation.
- Let agents inspect historical validation results. Keep final test labels and
  final test scores inaccessible to all agents, including the auditor.
- Permit submission-format checks without returning final test performance.
- Define stock-universe selection using information available at the relevant
  time, rather than selecting stocks based on later survival or performance.

The evaluator must enforce the agreed rolling or fixed evaluation protocol. Any
updates using newly matured labels require an explicit, identical rule for all
compared methods.

## 6. Fair Resource Budgets and Run Records

Single-agent and multi-agent conditions must use the same underlying LLM,
available data, dependencies, tools, execution hardware, and submission protocol.
The single agent may diagnose, engineer features, develop models, and review its
own work. Multi-agent communication and auditing consume the same shared budget.

Proposed primary controls are total LLM tokens, an LLM-call ceiling, execution
resource quotas, and elapsed runtime. Record actual consumption as well as limits.
Parallel execution must not receive additional aggregate compute implicitly.

With autonomous code execution, a single script can fit many models. Therefore,
do not claim an exactly matched model-fit count unless the runtime can enforce
and audit it. Record fits and candidate counts where observable, document their
measurement limits, and treat a mandatory training interface as a possible
separate controlled experiment. This refines the proposal's initial model-fit
budget assumption.

Each run should preserve task and method configurations, model identifiers,
prompts and messages, tool calls, generated code, validation artifacts, resource
usage, submissions, failures, and the evaluator's final report. Specify timeout,
retry, and failed-submission handling before evaluation.

## 7. Candidate Comparisons and Ablations

The following experiment matrix is proposed; its final size depends on resources.

| Condition | Purpose |
| --- | --- |
| Fixed forecasting pipeline | Establish performance without agent development. |
| Random search over a defined candidate space | Compare with a non-agent search procedure. |
| Single agent with self-review | Main control for specialized collaboration. |
| Diagnosis and modeling agents, with feature engineering inside modeling | Reference for testing a separate feature role. |
| Diagnosis, feature engineering, and modeling agents | Measure the effect of feature specialization. |
| Full four-role system | Measure the additional effect of auditing. |

Use paired tasks and repeated runs to compare methods. Three or more independent
agent runs per configuration are a provisional starting point, not a finalized
sample-size decision. Non-agent baselines should share applicable data and
training-resource constraints; unused LLM budget need not be spent.

Any oracle selector uses test outcomes only after submissions are frozen and
serves as a retrospective reference, never as information available to agents.

## 8. Reliability and Metrics

Distinguish two behaviors:

- **Abstention:** withhold a forecast; evaluate retained predictions using
  risk-coverage curves.
- **Fallback:** substitute a predefined baseline; evaluate the resulting complete
  prediction stream and fallback frequency.

Calibrate operational thresholds on development/validation data. Compute
post-hoc test risk-coverage curves without using them to tune the submitted
system. Define catastrophic-error thresholds and invalid-run treatment in advance.

Proposed prediction metrics are MAE and RMSE for each horizon, with a zero-return
baseline. Additional reporting may include model-selection regret, catastrophic
error rate, and resource cost. Inspect stock-level and time-period breakdowns as
well as aggregates. Account for overlapping return horizons and dependence among
stocks when estimating uncertainty; individual stock-date errors are not
independent replications.

Candidate reliability signals include self-reported confidence, rolling-validation
stability, disagreement among candidate predictive models, and disagreements
between the modeling and auditing roles. Specialized agents do not automatically
produce separate numerical forecasts: **agent disagreement and model disagreement
are different signals** and need separate definitions.

Evaluate clean data and controlled stress conditions. For input corruption,
retain an uncorrupted evaluation target. Distribution-shift scenarios require
their own definitions. The corruption types, severity levels, and calibration
protocol remain open.

## 9. Implementation Sequence

1. Implement a task loader, chronological splits, hidden targets, submission
   schema, simple baseline, and independent scoring.
2. Add isolated code execution, enforceable budgets, and complete run artifacts.
3. Implement a single agent that writes, runs, and revises forecasting code.
4. Implement the four specialized roles using the same environment and budgets.
5. Add reliability outputs, stress scenarios, ablations, and experiment aggregation.

The first implementation milestone is an end-to-end run with comparable outputs
and auditable resource usage. It should use the same interfaces intended for the
final experiments.

## 10. Open Decisions

- Data access and source: confirm CRSP/WRDS availability and permitted storage.
- Stock universe, date range, eligibility rules, and required fields.
- Return construction and exact forecast issuance time/information cutoff.
- Training, validation, development, and final-test windows; retraining schedule.
- Candidate model families, allowed dependencies, and initial feature inputs.
- Initial LLM provider/model, resource budgets, repetitions, and experimental
  sample size. The backend architecture will support interchangeable providers.
- Coordination details, audit acceptance rules, and fallback selection.
- Reliability score definitions, calibration, corruption scenarios, and statistical
  comparison procedures.

## 11. Engineering Decisions: Interchangeable LLM Backends

The LLM layer will expose a provider-independent internal protocol with adapters
for individual API families. Agent roles and orchestration will depend on this
protocol rather than a specific provider SDK.

```text
AgentRunner -> LLMBackend -> provider adapter -> model API
     |
     +-> budget checks -> tool executor -> execution environment
```

The adapter returns tool-call requests; the shared agent runtime executes tools
and controls the interaction loop. This keeps execution and resource rules
consistent across providers.

### Internal Contract

- `ModelRequest`: message history, tool definitions, generation options, and any
  provider-specific continuation state required for the conversation.
- `ModelResponse`: ordered content blocks, tool calls, stop reason, usage, and
  continuation state. Preserve required native response items rather than reducing
  every response to text.
- `ToolCall`: call identifier, tool name, and arguments, matched to a tool result.
- `Usage`: input/output tokens and available cache/reasoning details, together with
  the original provider usage and explicit indication of unknown values.

Adapters translate messages, tool schemas, results, errors, and stop reasons.
Declare supported capabilities and reject unsupported requirements or apply an
explicitly configured policy. Do not silently discard experiment settings or
change models during a run.

### Configuration and Experimental Controls

Select the adapter, model, endpoint, credential environment-variable name, and
request limits through configuration. Keep credentials in the controller, outside
generated code and saved request logs. Exact providers and models remain open.

Use the same backend and model settings across the single-agent and multi-agent
conditions within each comparison. Preserve provider-specific usage details;
token counts from different tokenizers are not interchangeable compute measures.
Retries must be visible to the shared budget manager, and requests with unknown
usage must not be recorded as consuming zero tokens.

### Initial LLM-Layer Milestone

1. Define the internal protocol and a deterministic `MockBackend` for development.
2. Implement one real adapter and complete a tool-call/result continuation cycle.
3. Verify tool-call matching, stop/truncation handling, usage accounting, and
   failure/retry behavior.
4. Add a second backend to validate the abstraction before expanding provider
   coverage.

An OpenAI-compatible adapter is a candidate starting point; native Responses,
Anthropic, and Gemini adapters can be added as needed. Compatibility claims must
be checked against the same behavioral tests before formal experiments.

## 12. References and Design Context

- [Original project proposal](<CS7652 Project Proposal.docx>).
- [MLE-bench paper](https://arxiv.org/abs/2410.07095) and
  [official repository](https://github.com/openai/mle-bench): references for task
  packaging, agent/evaluator separation, isolated execution, and run artifacts.
  MLE-bench is a benchmark; AIDE and OpenHands are examples of agents evaluated
  within it. We are adapting these design ideas, not committing to running the
  complete MLE-bench suite.
- [Montero-Manso and Hyndman, 2021: Locality and globality](https://fpp.robjhyndman.com/publications/global-forecasting/):
  background on shared versus per-series forecasting. It does not establish that
  shared models will outperform local models for this project's stock returns.
- Provider tool-call protocols:
  [OpenAI](https://developers.openai.com/api/docs/guides/function-calling),
  [Anthropic](https://platform.claude.com/docs/en/agents-and-tools/tool-use/handle-tool-calls),
  and [Gemini](https://ai.google.dev/gemini-api/docs/function-calling).
