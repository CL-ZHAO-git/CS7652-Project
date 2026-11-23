"""Checks shared by tests that touch OS-specific behavior."""

import os
import tempfile
from functools import cache
from pathlib import Path


@cache
def symlinks_supported():
    # Windows allows symlinks only with Developer Mode or administrator rights.
    with tempfile.TemporaryDirectory() as directory:
        try:
            (Path(directory) / "link").symlink_to(directory, target_is_directory=True)
        except (OSError, NotImplementedError):
            return False
    return True


def pid_alive(pid):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        kernel32.WaitForSingleObject.restype = wintypes.DWORD
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        handle = kernel32.OpenProcess(0x00100000 | 0x1000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x102  # WAIT_TIMEOUT: still running
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:  # Linux: a killed process nobody has reaped yet (common in containers) is a zombie.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in {"Z", "X"}
    except (OSError, IndexError):
        return True
