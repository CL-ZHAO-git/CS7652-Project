"""Controller-only credential loading; values never enter experiment configs."""

import os
import re
from contextlib import contextmanager
from pathlib import Path


def read_key_file(path: Path) -> str:
    if not path.is_file():
        raise ValueError("API key file does not exist")
    keys = set(re.findall(r"\bsk-[A-Za-z0-9_-]{16,}\b", path.read_text(encoding="utf-8")))
    if len(keys) != 1:
        raise ValueError("Key file must contain exactly one recognizable sk- API key")
    return keys.pop()


@contextmanager
def key_environment(name: str, path: Path | None):
    if path is None:
        yield
        return
    previous = os.environ.get(name)
    os.environ[name] = read_key_file(path)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
