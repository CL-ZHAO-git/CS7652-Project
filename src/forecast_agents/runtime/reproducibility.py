"""Fingerprints for frozen comparison inputs, excluding credentials and data rows."""

import hashlib
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_hash(root):
    root = Path(root)
    files = sorted(root.rglob("*.py"), key=lambda p: p.relative_to(root).as_posix())
    digest = hashlib.sha256()
    for path in files:
        if path.is_symlink():
            raise ValueError("Protocol source cannot contain symbolic links")
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
