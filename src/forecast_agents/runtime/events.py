import json
from datetime import datetime, timezone
from pathlib import Path


class EventLog:
    def __init__(self, path: Path):
        self.path = path
        self.sequence = 0

    def emit(self, kind, **fields):
        self.sequence += 1
        record = {"sequence": self.sequence, "time": datetime.now(timezone.utc).isoformat(),
                  "event": kind, **fields}
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
