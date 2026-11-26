"""Controller-produced summaries; arbitrary execution output stays local."""

import math
import re


class DataAccessError(ValueError):
    """Fixed controller message safe to return without runtime data."""


def rounded_summary(value):
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: rounded_summary(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded_summary(v) for v in value]
    return value


def describe_panel(panel, task):
    """Fixed pooled statistics over development history, without row-level output."""
    if panel is None or task is None:
        raise DataAccessError("No development panel is loaded")
    end = panel.dates.index(task.validation_end) + 1
    returns = [values[i] / values[i - 1] - 1 for values in panel.prices.values()
               for i in range(1, end)]
    result = {"scope": "development_only", "schema": ["symbol", "date", "close"],
              "close_semantics": "total_return_index", "stocks": len(panel.symbols),
              "sessions": end, "rows": end * len(panel.symbols),
              "first_date": panel.dates[0], "last_date": task.validation_end,
              "train_end": task.train_end, "horizons": list(task.horizons),
              "protocol": task.protocol, "balanced_panel": True,
              "quality": "Panel loader rejects missing cells, duplicate keys and invalid levels",
              "pooled_daily_returns": {"count": len(returns)}}
    if len(returns) >= 20:
        mean = sum(returns) / len(returns)
        ordered = sorted(returns)
        stats = {"mean": mean, "std": math.sqrt(sum((x - mean) ** 2 for x in returns) / (len(returns) - 1))}
        for label, fraction in (("p01", 0.01), ("median", 0.5), ("p99", 0.99)):
            stats[label] = ordered[int((len(ordered) - 1) * fraction)]
        result["pooled_daily_returns"].update(stats)
    else:
        result["pooled_daily_returns"]["summary_suppressed"] = "Fewer than 20 returns"
    return rounded_summary(result)


def safe_type_diagnostic(message):
    """Recognize only exact builtin type-error templates, never echo values."""
    if message == "must be real number, not NoneType":
        return {"diagnostic_code": "numeric_operation_on_none", "operand_types": ["NoneType"],
                "hint": "Handle None before numeric checks. A feature may be unavailable during its documented lookback/warmup."}
    allowed_types = {"NoneType", "int", "float", "str", "bool", "list", "tuple", "dict", "set"}
    match = re.fullmatch(r"'([<>=!]+)' not supported between instances of '([A-Za-z]+)' and '([A-Za-z]+)'", message)
    if match and match[2] in allowed_types and match[3] in allowed_types:
        return {"diagnostic_code": "incompatible_comparison_types", "operand_types": [match[2], match[3]],
                "hint": "Check nullable feature availability before ordering or comparing numeric values."}
    match = re.fullmatch(r"'([A-Za-z]+)' object is not subscriptable", message)
    if match and match[1] in allowed_types:
        return {"diagnostic_code": "not_subscriptable", "operand_types": [match[1]],
                "hint": "Check the documented return structure before indexing; None may indicate unavailable history."}
    return {}


def execution_summary(recorded, owned_paths, known_keys=()):
    result = {k: recorded[k] for k in ("returncode", "timed_out", "elapsed_seconds")}
    result.update(stdout="", stderr="", output_withheld=True)
    if recorded["returncode"] or recorded["timed_out"]:
        result["is_error"] = True
        result["error_type"] = "ExecutionError"
        allowed = {"AssertionError", "TypeError", "ValueError", "KeyError", "IndexError",
                   "NameError", "AttributeError", "ImportError", "ModuleNotFoundError",
                   "FileNotFoundError", "PermissionError", "RuntimeError", "MemoryError",
                   "OverflowError", "ZeroDivisionError", "SyntaxError", "IndentationError",
                   "TimeoutError", "StopIteration"}
        for line in reversed(recorded["stderr"].splitlines()):
            kind = line.partition(":")[0].strip()
            if kind in allowed:
                result["error_type"] = kind
                if kind == "TypeError":
                    result.update(safe_type_diagnostic(line.partition(":")[2].strip()))
                message = line.partition(":")[2].strip()
                if kind == "ImportError" and message == "attempted relative import with no known parent package":
                    result.update(diagnostic_code="relative_import_without_package",
                                  hint="The module needs package-aware loading or a compatible local absolute import.")
                if kind == "KeyError":
                    for key in sorted(known_keys):
                        if message == repr(key) or (key.isdecimal() and message == key):
                            result.update(diagnostic_code="missing_known_key", missing_key=key,
                                          hint="This key is already present in authored code or declared horizons. Check state shape and empty/short-history assumptions.")
                            break
                break
        locations = []
        for path in owned_paths:
            for match in re.finditer(r'File "[^"\n]*/' + re.escape(path) + r'", line ([0-9]+)', recorded["stderr"]):
                locations.append({"path": path, "line": int(match.group(1))})
        result["locations"] = locations[-5:]
    return result
