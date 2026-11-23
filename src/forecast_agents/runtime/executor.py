"""Bounded processes for trusted smoke tests, or explicit Docker execution.

Runs on Linux, macOS, and Windows. Local mode applies best-effort limits with each
platform's own mechanism (rlimits and a process group on POSIX, a job object on
Windows). It is not a security sandbox on any platform.
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

try:  # POSIX only; Windows uses a job object instead.
    import resource
except ImportError:
    resource = None

WINDOWS = os.name == "nt"
FILE_LIMIT_BYTES = 16 * 1024 * 1024

# Isolate the interpreter from host PYTHONPATH/user-site while allowing candidate
# helpers beside the requested script and within its own workspace.
LAUNCHER = (
    "import runpy,sys; from pathlib import Path; "
    "sys.path[:0]=[str(Path(sys.argv[1]).parent),str(Path.cwd())]; "
    "runpy.run_path(sys.argv[1],run_name='__main__')"
)
# -X utf8 makes generated code read, write, and print UTF-8 on every OS; Windows
# would otherwise use its ANSI code page (for example cp1252 or GBK).
PYTHON_FLAGS = ("-I", "-X", "utf8", "-u")

# Windows processes need these to start reliably and to find a temp directory.
WINDOWS_PROCESS_ENV = ("SYSTEMROOT", "TEMP", "TMP")
# The Docker CLI uses these to find its daemon, context, and config. They reach the
# controller-side client only; containers receive nothing but the --env values below.
DOCKER_CLIENT_ENV = ("HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
                     "SYSTEMROOT", "TEMP", "TMP", "DOCKER_HOST", "DOCKER_CONTEXT",
                     "DOCKER_CONFIG", "DOCKER_CERT_PATH", "DOCKER_TLS_VERIFY")
# Windows hosts have no POSIX uid/gid to map; containers run as a fixed non-root user.
WINDOWS_CONTAINER_USER = "1000:1000"


@dataclass(frozen=True)
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    elapsed_seconds: float


def _default_path():
    if WINDOWS:
        root = os.environ.get("SYSTEMROOT", r"C:\Windows")
        return os.pathsep.join((os.path.join(root, "System32"), root))
    return os.defpath


def _inherited(names):
    return {name: os.environ[name] for name in names if name in os.environ}


def _set_rlimits(limits):
    # Best effort: some platforms reject or ignore particular rlimits.
    for kind, value in limits:
        try:
            resource.setrlimit(kind, (value, value))
        except (ValueError, OSError):
            pass


class Executor:
    def __init__(self, mode="local", image="python:3.13-slim", cpus=1.0, memory_mb=1024,
                 output_limit=32_000):
        if mode not in {"local", "docker"}:
            raise ValueError("Execution mode must be local or docker")
        if cpus <= 0 or memory_mb < 64 or output_limit < 100:
            raise ValueError("Invalid execution resource limits")
        self.mode, self.image = mode, image
        self.cpus, self.memory_mb, self.output_limit = cpus, memory_mb, output_limit
        self.docker = None
        if mode == "docker":
            # Resolve once: the restricted child PATH may not contain Docker
            # (for example /usr/local/bin on macOS).
            self.docker = shutil.which("docker")
            if self.docker is None:
                raise RuntimeError("Docker is unavailable; no automatic local fallback")
            inspected = subprocess.run([self.docker, "image", "inspect", "--format", "{{.Id}}", image],
                                       capture_output=True, text=True, timeout=15)
            if inspected.returncode:
                raise RuntimeError("Docker image is unavailable; pull/build the configured image first")
            self.image = inspected.stdout.strip()

    def _limits(self):
        _set_rlimits(((resource.RLIMIT_AS, self.memory_mb * 1024 * 1024),
                      (resource.RLIMIT_FSIZE, FILE_LIMIT_BYTES), (resource.RLIMIT_CORE, 0)))

    def _log_limit(self):
        _set_rlimits(((resource.RLIMIT_FSIZE, FILE_LIMIT_BYTES),))

    def _spawn_options(self):
        if WINDOWS:
            # Like the POSIX new session, a new process group keeps console Ctrl+C
            # events away from the child. No console window is opened.
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
        return {"start_new_session": True,
                "preexec_fn": self._limits if self.mode == "local" else self._log_limit}

    def _windows_job(self, process):
        if not WINDOWS:
            return None
        from forecast_agents.runtime.windows_job import JobObject
        memory = self.memory_mb * 1024 * 1024 if self.mode == "local" else None
        try:
            # Assigned right after launch, so only a child started in that instant escapes.
            return JobObject.for_process(process.pid, memory)
        except OSError:
            return None  # best effort, like POSIX rlimits; the process itself is still killed

    def execute(self, workspace: Path, script: str, stdin="", timeout=30.0):
        workspace = workspace.resolve()
        source = (workspace / script).resolve()
        if not source.is_relative_to(workspace) or not source.is_file():
            raise ValueError("Executable script must be inside the workspace")
        if timeout <= 0:
            raise ValueError("Execution deadline expired")
        started = time.monotonic()
        container = "forecast-" + uuid.uuid4().hex
        env = {"PATH": _default_path(), "LANG": "C.UTF-8", "PYTHONHASHSEED": "0",
               "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        if self.mode == "local":
            command = [sys.executable, *PYTHON_FLAGS, "-c", LAUNCHER, str(source)]
            if WINDOWS:
                env.update(_inherited(WINDOWS_PROCESS_ENV))
        else:
            if "," in str(workspace):
                raise ValueError("Docker workspace paths cannot contain commas")
            env.update(_inherited(DOCKER_CLIENT_ENV))
            user = WINDOWS_CONTAINER_USER if WINDOWS else f"{os.getuid()}:{os.getgid()}"
            command = [
                self.docker, "run", "--rm", "--name", container, "--network", "none",
                "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "128", "--cpus", str(self.cpus),
                "--ulimit", f"fsize={FILE_LIMIT_BYTES}:{FILE_LIMIT_BYTES}",
                "--memory", f"{self.memory_mb}m", "--memory-swap", f"{self.memory_mb}m",
                "--user", user,
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
                "--mount", f"type=bind,source={workspace},target=/workspace",
            ]
            if (workspace / "data").is_dir():
                command += ["--mount", f"type=bind,source={workspace / 'data'},target=/workspace/data,readonly"]
            command += ["--workdir", "/workspace", "--env", "PYTHONHASHSEED=0", "-i",
                        self.image, "python", *PYTHON_FLAGS, "-c", LAUNCHER,
                        "/workspace/" + source.relative_to(workspace).as_posix()]
        # Use files rather than unbounded in-memory stdout pipes.
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            process = subprocess.Popen(command, cwd=workspace, env=env, stdin=subprocess.PIPE,
                                       stdout=out, stderr=err, **self._spawn_options())
            job, timed_out = None, False
            try:
                job = self._windows_job(process)
                process.communicate(stdin.encode(), timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
            finally:
                # Kill descendants even when the immediate process exited successfully.
                _kill_tree(process, job)
                process.wait()
                if self.mode == "docker":
                    subprocess.run([self.docker, "rm", "-f", container], capture_output=True,
                                   timeout=10, check=False)
            out.seek(0)
            err.seek(0)
            stdout = out.read(self.output_limit).decode(errors="replace")
            stderr = err.read(self.output_limit).decode(errors="replace")
        return ExecutionResult(process.returncode, stdout, stderr, timed_out,
                               time.monotonic() - started)


def _kill_tree(process, job):
    if job is not None:
        terminated = job.terminate()
        job.close()
        if terminated:
            return
    if WINDOWS:
        if process.poll() is None:
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
