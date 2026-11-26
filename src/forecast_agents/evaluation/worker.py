"""Standalone worker copied into an isolated, per-origin execution workspace.

Only the payload's history and model state are available through this protocol.
The worker never loads the evaluator's full panel or final labels.
"""

import importlib
import importlib.util
import types
import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).parent
    sys.path.insert(0, str(root / "solution"))
    payload = json.load(sys.stdin)
    filename = "features.py" if payload["operation"] == "features" else "pipeline.py"
    solution = root / "solution"
    init = solution / "__init__.py"
    if init.is_file():
        package_spec = importlib.util.spec_from_file_location("solution", init,
                                                            submodule_search_locations=[str(solution)])
        package = importlib.util.module_from_spec(package_spec)
        sys.modules["solution"] = package
        package_spec.loader.exec_module(package)
    else:
        package = types.ModuleType("solution")
        package.__path__ = [str(solution)]
        package.__package__ = "solution"
        sys.modules["solution"] = package
    module = importlib.import_module("solution." + Path(filename).stem)
    if payload["operation"] == "features":
        result = module.features_at(payload["history"], payload["origin"])
    elif payload["operation"] == "fit":
        result = module.fit(payload["history"], payload["horizons"])
    elif payload["operation"] == "predict":
        result = module.predict(payload["history"], payload["state"], payload["horizons"])
    else:
        raise ValueError("Unknown worker operation")
    with (root / "result.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, allow_nan=False)


if __name__ == "__main__":
    main()
