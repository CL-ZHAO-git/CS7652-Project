"""Deterministic smoke-test fixtures, not an autonomous research strategy."""

FEATURES_CODE = '''from collections import defaultdict


def group_history(history):
    groups = defaultdict(list)
    for row in history:
        groups[row["symbol"]].append(row)
    return {symbol: sorted(rows, key=lambda row: row["date"])
            for symbol, rows in groups.items()}


def features(rows, index):
    if index < 5:
        return None
    recent = [rows[j]["close"] / rows[j - 1]["close"] - 1
              for j in range(index - 4, index + 1)]
    return [1.0, recent[-1], sum(recent) / 5]


def features_at(history, origin):
    result = {}
    for symbol, rows in group_history([r for r in history if r["date"] <= origin]).items():
        vector = features(rows, len(rows) - 1)
        result[symbol] = dict(zip(("intercept", "ret_1", "mom_5"), vector or (None, None, None)))
    return result
'''

PIPELINE_CODE = '''import math
from features import features, group_history


def solve(matrix, vector):
    augmented = [list(row) + [value] for row, value in zip(matrix, vector)]
    size = len(vector)
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(augmented[row][col]))
        augmented[col], augmented[pivot] = augmented[pivot], augmented[col]
        scale = augmented[col][col]
        if abs(scale) < 1e-14:
            raise ValueError("Singular regression system")
        augmented[col] = [value / scale for value in augmented[col]]
        for row in range(size):
            if row != col:
                factor = augmented[row][col]
                augmented[row] = [a - factor * b for a, b in zip(augmented[row], augmented[col])]
    return [row[-1] for row in augmented]


def fit(history, horizons):
    groups = group_history(history)
    state = {}
    for horizon in horizons:
        samples = []
        for rows in groups.values():
            for index in range(5, len(rows) - horizon):
                x = features(rows, index)
                y = rows[index + horizon]["close"] / rows[index]["close"] - 1
                samples.append((x, y))
        if not samples:
            raise ValueError("Insufficient matured training labels")
        matrix = [[sum(x[i] * x[j] for x, y in samples) for j in range(3)] for i in range(3)]
        for j in (1, 2):
            matrix[j][j] += 0.001
        vector = [sum(x[j] * y for x, y in samples) for j in range(3)]
        coefficients = solve(matrix, vector)
        variance = sum((sum(a * b for a, b in zip(x, coefficients)) - y) ** 2
                       for x, y in samples) / len(samples)
        # A constant classifier learned only from matured training labels.
        prob_up = (sum(y > 0 for x, y in samples) + 1) / (len(samples) + 2)
        state[str(horizon)] = {"coefficients": coefficients, "residual_scale": math.sqrt(variance),
                               "prob_up": prob_up}
    return state


def predict(history, state, horizons):
    result = []
    for symbol, rows in group_history(history).items():
        x = features(rows, len(rows) - 1)
        if x is None:
            raise ValueError("Insufficient feature history")
        for horizon in horizons:
            model = state[str(horizon)]
            prediction = sum(a * b for a, b in zip(x, model["coefficients"]))
            # A smoke-test heuristic; this is not a calibrated failure probability.
            risk = model["residual_scale"] * (1 + abs(x[1]) * 10)
            result.append({"symbol": symbol, "horizon": horizon, "prediction": prediction,
                           "prob_up": model["prob_up"], "risk": risk, "abstain": False, "fallback": False})
    return result
'''

DIAGNOSIS_CODE = '''import json
from collections import Counter
from pathlib import Path

rows = json.loads(Path("data/history.json").read_text())
report = {"rows": len(rows), "observations_per_stock": dict(Counter(row["symbol"] for row in rows)),
          "latest_visible_date": max(row["date"] for row in rows),
          "scope": "Scripted smoke test; no automated data-quality conclusions"}
Path("diagnostics.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
'''

AUDIT_CODE = '''import ast
import json
from pathlib import Path

files = sorted(Path("solution").glob("*.py"))
for source in files:
    ast.parse(source.read_text())
report = {"parseable_sources": [str(source) for source in files],
          "scope": "Syntax smoke checks only; not a leakage or reliability certification"}
Path("audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report))
'''

ZERO_CODE = '''def fit(history, horizons):
    return {}


def predict(history, state, horizons):
    return [{"symbol": symbol, "horizon": horizon, "prediction": 0.0, "prob_up": 0.5,
             "risk": 0.0, "abstain": False, "fallback": False}
            for symbol in sorted({row["symbol"] for row in history}) for horizon in horizons]
'''

PIPELINE_CONTRACT = '''# Pipeline contract

Write solution/pipeline.py and any local helpers under solution/.

- fit(history, horizons) -> JSON-serializable model state.
- predict(history, state, horizons) -> a list of prediction rows.
- features.py must export features_at(history, origin) -> {symbol: {feature_name:
  finite_number_or_null}}. Include every symbol observed through origin. This
  function must ignore later rows even when history includes future dates. Keep
  feature names consistent across symbols. Modeling should use the audited helper.
- null means a feature is unavailable, not necessarily an implementation error.
  Respect its lookback in tests: a 20-session return needs at least 21 closes.
  Do not call numeric operations on unavailable values or require a long-window
  rank from a 3-row fixture. Correct tests that contradict the nullable contract.
- history is a list of {symbol: str, date: ISO date, close: positive float} rows.
  close is the level of a total-return index (splits and dividends included), so
  close ratios are returns; the level itself is not a dollar price.
- fit receives observations through a cutoff, including only labels that can be
  constructed from that history. task.json gives the protocol. With refit "none",
  one fit at train_end serves validation and the final test. With walk-forward
  refits ("year", "quarter", or "month"), fit is called again at each period end
  with all history to date, in every validation fold and during the final test,
  so keep fitting fast enough to run several times.
- predict receives a causal prefix through the current forecast origin.
- Return exactly one row per observed stock and horizon, with these fields:
  symbol, horizon, prediction (finite number), risk (nonnegative finite number;
  higher means riskier), abstain (bool), fallback (bool), and prob_up (finite
  probability in [0,1] of the horizon return being strictly positive).
- Scores include MAE, MSE, RMSE and direction binary cross entropy/accuracy.
  Direction label is 1 if return > 0, otherwise 0 (including exactly zero).
  Learn probabilities using only matured fit labels and development validation;
  a separate classifier is allowed. Never treat the return forecast or risk as
  a probability. Declare any distributional mapping/calibration and validate it.
  The direction baseline predicts prob_up=0.5; BCE uses natural logs and clips
  probabilities to [1e-15, 1-1e-15]. Accuracy predicts up when prob_up >= 0.5.
  Offline rescoring accepts legacy rows without prob_up and marks direction
  scores unavailable. New candidate validation/audit/final scoring require it
  for every stock/horizon, including abstentions.
- Abstain and fallback cannot both be true. Fallback uses the evaluator's fixed
  zero-return baseline and a direction probability of 0.5. Emit finite candidate
  predictions/probabilities even when abstaining; raw scores retain them.
- No final test labels or final test prices beyond the current origin are visible.
- Use historical validation to choose models and thresholds. validate_candidate
  returns validation scores only, overall and per fold; folds cover different
  market regimes, so check the worst fold as well as the average. freeze/submit
  does not reveal final test scores.
- Before final completion, call audit_candidate on the unchanged validated source.
  It checks sampled feature-prefix/real future-perturbation invariance and repeated
  prediction/schema checks. It is not a statistical guarantee or exhaustive proof.
  Keep audit scripts/reports outside solution/ so evidence does not alter candidate
  sources after validation. Changes require revalidation and reauditing.
- Each fit/predict call runs in a fresh workspace. Persist learned state through
  the returned JSON state. Do not depend on fit-time files or Python globals.
- Local helpers support both absolute imports (from features import ...) and
  package-relative imports (from .features import ...). Model state is opaque
  JSON; tests should exercise predict instead of assuming internal state keys,
  especially for empty/short-history cases.
- The default runtime image has Python's standard library. Additional libraries
  require a prebuilt, configured image shared across comparison conditions.
'''
