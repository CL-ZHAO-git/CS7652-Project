"""Bounded processes for trusted smoke tests, or explicit Docker execution."""

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


# Isolate the interpreter from host PYTHONPATH/user-site while allowing candidate
# helpers beside the requested script and within its own workspace.
LAUNCHER = (
    "import runpy,sys; from pathlib import Path; "
    "sys.path[:0]=[str(Path(sys.argv[1]).parent),str(Path.cwd())]; "
    "runpy.run_path(sys.argv[1],run_name='__main__')"
)


@dataclass(frozen=True)
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_seconds: float


class Executor:
    def __init__(self, mode="local", image="python:3.13-slim", cpus=1.0, memory_mb=1024,
                 output_limit=32_000):
        if mode not in {"local", "docker"}:
            raise ValueError("Execution mode must be local or docker")
        if cpus <= 0 or memory_mb < 64 or output_limit < 100:
            raise ValueError("Invalid execution resource limits")
        self.mode, self.image = mode, image
        self.cpus, self.memory_mb, self.output_limit = cpus, memory_mb, output_limit
        if mode == "docker":
            if shutil.which("docker") is None:
                raise RuntimeError("Docker is unavailable; no automatic local fallback")
            inspected = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", image],
                                       capture_output=True, text=True, timeout=15)
            if inspected.returncode:
                raise RuntimeError("Docker image is unavailable; pull/build the configured image first")
            self.image = inspected.stdout.strip()

    def _limits(self):
        size = self.memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (size, size))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def _log_limit(self):
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))

    def execute(self, workspace: Path, script: str, stdin="", timeout=30.0):
        workspace = workspace.resolve()
        source = (workspace / script).resolve()
        if not source.is_relative_to(workspace) or not source.is_file():
            raise ValueError("Executable script must be inside the workspace")
        if timeout <= 0:
            raise ValueError("Execution deadline expired")
        started = time.monotonic()
        container = "forecast-" + uuid.uuid4().hex
        env = {"PATH": os.defpath, "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        if self.mode == "local":
            command = [sys.executable, "-I", "-u", "-c", LAUNCHER, str(source)]
        else:
            if "," in str(workspace):
                raise ValueError("Docker workspace paths cannot contain commas")
            command = [
                "docker", "run", "--rm", "--name", container, "--network", "none",
                "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "128", "--cpus", str(self.cpus),
                "--ulimit", "fsize=16777216:16777216",
                "--memory", f"{self.memory_mb}m", "--memory-swap", f"{self.memory_mb}m",
                "--user", f"{os.getuid()}:{os.getgid()}",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                "--mount", f"type=bind,source={workspace},target=/workspace",
            ]
            if (workspace / "data").is_dir():
                command += ["--mount", f"type=bind,source={workspace / 'data'},target=/workspace/data,readonly"]
            command += ["--workdir", "/workspace", "--env", "PYTHONHASHSEED=0", "-i",
                        self.image, "python", "-I", "-u", "-c", LAUNCHER,
                        "/workspace/" + str(source.relative_to(workspace))]
        # Use files rather than unbounded in-memory stdout pipes.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            process = subprocess.Popen(command, cwd=workspace, env=env, stdin=subprocess.PIPE,
                                       stdout=out, stderr=err, start_new_session=True,
                                       preexec_fn=self._limits if self.mode == "local" else self._log_limit)
            timed_out = False
            try:
                process.communicate(stdin.encode(), timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Kill descendants even when the immediate process exited successfully.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                if self.mode == "docker":
                    subprocess.run(["docker", "rm", "-f", container], capture_output=True,
                                   timeout=10, check=False)
            out.seek(0)
            err.seek(0)
            stdout = out.read(self.output_limit).decode(errors="replace")
            stderr = err.read(self.output_limit).decode(errors="replace")
        return ExecutionResult(process.returncode, stdout, stderr, timed_out,
                               time.monotonic() - started)
