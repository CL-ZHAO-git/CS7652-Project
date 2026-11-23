"""Windows job objects for the local executor; imported only on Windows.

A job object gives Windows what POSIX local execution gets from rlimits and a
process group: an optional per-process memory cap, and termination of every
descendant process, even after the immediate child has exited. It does not make
local execution a security sandbox.
"""

import ctypes
from ctypes import wintypes

_EXTENDED_LIMIT_INFORMATION = 9
_LIMIT_PROCESS_MEMORY = 0x00000100
_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400  # no Windows Error Reporting dialogs
_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


class _BasicLimits(ctypes.Structure):
    _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD)]


class _IoCounters(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [("BasicLimitInformation", _BasicLimits),
                ("IoInfo", _IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t)]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_kernel32.CreateJobObjectW.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
_kernel32.CreateJobObjectW.restype = wintypes.HANDLE
_kernel32.SetInformationJobObject.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
_kernel32.SetInformationJobObject.restype = wintypes.BOOL
_kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
_kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
_kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
_kernel32.TerminateJobObject.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
_kernel32.CloseHandle.restype = wintypes.BOOL


def _check(result):
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
    return result


class JobObject:
    def __init__(self, memory_bytes=None):
        self._handle = _check(_kernel32.CreateJobObjectW(None, None))
        try:
            limits = _ExtendedLimits()
            flags = _LIMIT_KILL_ON_JOB_CLOSE | _LIMIT_DIE_ON_UNHANDLED_EXCEPTION
            if memory_bytes:
                flags |= _LIMIT_PROCESS_MEMORY
                limits.ProcessMemoryLimit = memory_bytes
            limits.BasicLimitInformation.LimitFlags = flags
            _check(_kernel32.SetInformationJobObject(
                self._handle, _EXTENDED_LIMIT_INFORMATION, ctypes.byref(limits), ctypes.sizeof(limits)))
        except OSError:
            self.close()
            raise

    @classmethod
    def for_process(cls, pid, memory_bytes=None):
        job = cls(memory_bytes)
        try:
            job.assign(pid)
        except OSError:
            job.close()
            raise
        return job

    def assign(self, pid):
        # Descendants created after assignment join the job automatically.
        process = _check(_kernel32.OpenProcess(_PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid))
        try:
            _check(_kernel32.AssignProcessToJobObject(self._handle, process))
        finally:
            _kernel32.CloseHandle(process)

    def terminate(self):
        """Kill every process in the job. Returns True on success."""
        return bool(self._handle) and bool(_kernel32.TerminateJobObject(self._handle, 1))

    def close(self):
        if self._handle:
            _kernel32.CloseHandle(self._handle)
            self._handle = None
