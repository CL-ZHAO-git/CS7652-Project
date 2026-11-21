"""Standalone worker copied into an isolated, per-origin execution workspace.

Only the payload's history and model state are available through this protocol.
The worker never loads the evaluator's full panel or final labels.
"""

import importlib.util
import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent
    sys.path.insert(0, str(root / "solution"))
    spec = importlib.util.spec_from_file_location("candidate_pipeline", root / "solution/pipeline.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.load(sys.stdin)
    if payload["operation"] == "fit":
        result = module.fit(payload["history"], payload["horizons"])
    elif payload["operation"] == "predict":
        result = module.predict(payload["history"], payload["state"], payload["horizons"])
    else:
        raise ValueError("Unknown worker operation")
    with (root / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False)


if __name__ == "__main__":
    main()
