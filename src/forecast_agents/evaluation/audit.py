"""Fixed development-only audit with nonempty future perturbations."""

import math
import time

from forecast_agents.evaluation.evaluator import invoke_candidate, validate_predictions


class AuditFailure(ValueError):
    """Messages are controlled reason codes, without candidate values or raw rows."""


def _same(a, b):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if type(a) in (int, float) and type(b) in (int, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return type(a) is type(b) and a == b


def _feature_schema(value, symbols):
    if not isinstance(value, dict) or set(value) != set(symbols):
        raise AuditFailure("audit_feature_symbol_coverage_failed")
    names = None
    for row in value.values():
        if not isinstance(row, dict) or not row or any(not isinstance(k, str) or not k for k in row):
            raise AuditFailure("audit_feature_schema_failed")
        if names is not None and row.keys() != names:
            raise AuditFailure("audit_feature_names_inconsistent")
        names = row.keys()
        if any(v is not None and (type(v) not in (int, float) or not math.isfinite(v)) for v in row.values()):
            raise AuditFailure("audit_feature_nonfinite_values")


def audit_candidate(panel, task, solution, executor, timeout_seconds=30, wall_seconds=300):
    started = time.monotonic()
    blocks = panel.blocks(task, "validation")
    choices = [(blocks[0], blocks[0].origins[len(blocks[0].origins) // 3]),
               (blocks[-1], blocks[-1].origins[2 * len(blocks[-1].origins) // 3])]
    probes = []
    full = panel.history(task.validation_end)

    def invoke(op, history, state=None, origin=None):
        remaining = wall_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("Audit wall-time limit exhausted")
        return invoke_candidate(solution, executor, op, history, task.horizons,
                                state=state, origin=origin, timeout_seconds=min(timeout_seconds, remaining))

    seen = set()
    for block, origin in choices:
        if origin in seen:
            continue
        seen.add(origin)
        prefix = panel.history(origin)
        poisoned = [{**r, "close": r["close"] * (2 + (i % 7) / 10)} if r["date"] > origin else dict(r)
                    for i, r in enumerate(full)]
        changed = sum(a["close"] != b["close"] for a, b in zip(full, poisoned))
        if changed == 0 or origin >= task.validation_end:
            raise AuditFailure("audit_future_perturbation_was_empty")
        before = invoke("features", full, origin=origin)
        truncated = invoke("features", prefix, origin=origin)
        after = invoke("features", poisoned, origin=origin)
        for value in (before, truncated, after):
            _feature_schema(value, panel.symbols)
        if not _same(before, truncated) or not _same(before, after):
            raise AuditFailure("audit_features_depend_on_future_rows")
        state = invoke("fit", panel.history(block.fit_cutoff))
        first = invoke("predict", prefix, state=state)
        repeated = invoke("predict", prefix, state=state)
        for predictions in (first, repeated):
            validate_predictions(predictions, panel.symbols, task.horizons, require_prob_up=True)
        order = lambda row: (row["symbol"], row["horizon"])
        if not _same(sorted(first, key=order), sorted(repeated, key=order)):
            raise AuditFailure("audit_prediction_repeatability_failed")
        probes.append({"origin": origin, "fit_cutoff": block.fit_cutoff,
                       "future_rows_changed": changed, "feature_prefix_invariant": True,
                       "feature_future_perturbation_invariant": True,
                       "prediction_schema_valid": True, "prediction_repeatable": True})
    return {"passed": True, "scope": "Sampled development feature causality and prediction contract; not exhaustive proof or statistical superiority",
            "probes": probes, "elapsed_seconds": time.monotonic() - started}
