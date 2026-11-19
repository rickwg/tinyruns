import json
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from typing import Any

import pytest
import yaml

from tinyruns.cluster.main import main

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIRECTORY = Path(__file__).resolve().parent
FAKE_CLUSTER_SCRIPT = TESTS_DIRECTORY / "fake_cluster.py"
FIXTURE_PROJECT = TESTS_DIRECTORY / "fixture_project"
FAKE_COMMANDS = ("sbatch", "squeue", "sacct", "scancel", "apptainer")
FIXTURE_PROJECT_NAME = "fixture-project"
STAGES_PACKAGE = "fixture_project.stages"
STAGES_DIRECTORY = Path("src") / "fixture_project" / "stages"
RUNS_DIRECTORY = Path("artifacts") / "runs"
RUN_COMMAND_PREFIX = "uv run python -m tinyruns.run "
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
PROXY_VARIABLES = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
ON_CPU = ("--resources", "cpu-small")
MARKER_CONFIG = {
    "stages": [f"{STAGES_PACKAGE}.marker"],
    "marker": {"text": "marker report"},
}


@pytest.fixture(autouse=True)
def block_internet(monkeypatch: pytest.MonkeyPatch) -> None:
    connect = socket.socket.connect

    def connect_to_this_machine_only(self: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address[0] not in LOOPBACK_HOSTS:
            raise RuntimeError(f"Tests must not reach the internet: {address}")
        connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect_to_this_machine_only)
    for variable in PROXY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def git(repository: Path, *arguments: str) -> str:
    identity = ("-c", "user.name=Test", "-c", "user.email=test@example.invalid")
    completed = subprocess.run(
        ["git", "-C", str(repository), *identity, *arguments],
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip()


def commit_all(repository: Path, message: str) -> None:
    git(repository, "add", "--all")
    git(repository, "commit", "--message", message)


def write_config(repository: Path, name: str, config: dict[str, Any]) -> Path:
    config_path = repository / "configs" / f"{name}.yaml"
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def local_run(repository: Path, run_id: str) -> Path:
    return repository / RUNS_DIRECTORY / run_id


def metrics(run_directory: Path) -> list[dict[str, Any]]:
    lines = (run_directory / "metrics.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def run_locally(repository: Path, run_with: str) -> None:
    assert run_with.startswith(RUN_COMMAND_PREFIX)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tinyruns.run",
            run_with.removeprefix(RUN_COMMAND_PREFIX),
        ],
        cwd=repository,
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        check=True,
    )


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    shutil.copytree(
        FIXTURE_PROJECT, repository, ignore=shutil.ignore_patterns("__pycache__")
    )
    (repository / ".gitignore").write_text("__pycache__/\nartifacts/\n")
    write_config(repository, "marker", MARKER_CONFIG)
    git(repository, "init", "--quiet")
    commit_all(repository, "Project under test")
    return repository


@dataclass(frozen=True)
class FakeCluster:
    root: Path
    profile_path: Path
    state_directory: Path

    @property
    def scratch_root(self) -> Path:
        return self.root / "scratch" / FIXTURE_PROJECT_NAME

    @property
    def code_root(self) -> Path:
        return self.root / "home" / "code"

    def run_directory(self, run_id: str) -> Path:
        return self.scratch_root / "runs" / run_id

    def hold_new_jobs(self) -> None:
        (self.state_directory / "hold").touch()

    def release(self, job_id: str) -> None:
        subprocess.run(
            [sys.executable, str(FAKE_CLUSTER_SCRIPT), "release", job_id], check=True
        )

    def forget(self, job_id: str) -> None:
        (self.state_directory / f"{job_id}.json").unlink()

    def job_ids(self) -> list[str]:
        return sorted(path.stem for path in self.state_directory.glob("*.json"))


def fake_cluster_profile(root: Path) -> dict[str, Any]:
    return {
        "name": "fake-cluster",
        "ssh_host": None,
        "code_root": str(root / "home" / "code"),
        "scratch_root": str(root / "scratch" / "{project}"),
        "node_local_scratch": str(root / "local"),
        "gpu_hours_without_confirmation": 2,
        "environment": {"FAKE_PROXY": "http://proxy.invalid:3128"},
        "image_build": {
            "partition": "small",
            "time": "01:00:00",
            "cpus": 2,
            "memory": "4G",
        },
        "smoke_resources": "cpu-small",
        "resources": {
            "cpu-small": {
                "partition": "small",
                "time": "00:10:00",
                "cpus": 1,
                "memory": "1G",
            },
            "gpu-short": {
                "partition": "gpu",
                "time": "01:00:00",
                "cpus": 2,
                "memory": "4G",
                "gpus": 1,
                "constraint": "GPU-80GB",
            },
            "gpu-long": {
                "partition": "gpu",
                "time": "1-00:00:00",
                "cpus": 2,
                "memory": "4G",
                "gpus": 2,
            },
        },
    }


def install_fake_commands(bin_directory: Path) -> None:
    for command in FAKE_COMMANDS:
        wrapper = bin_directory / command
        wrapper.write_text(
            "#!/bin/bash\n"
            f"exec {quote(sys.executable)} {quote(str(FAKE_CLUSTER_SCRIPT))} "
            f'{command} "$@"\n'
        )
        wrapper.chmod(0o755)


@pytest.fixture
def cluster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCluster:
    root = tmp_path / "cluster"
    for directory in ("bin", "slurm", "local"):
        (root / directory).mkdir(parents=True)
    install_fake_commands(root / "bin")
    monkeypatch.setenv("PATH", f"{root / 'bin'}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CLUSTER_STATE", str(root / "slurm"))
    monkeypatch.setenv("FAKE_CLUSTER_PYTHON", sys.executable)
    profile_path = root / "profile.yaml"
    profile_path.write_text(yaml.safe_dump(fake_cluster_profile(root)))
    return FakeCluster(
        root=root, profile_path=profile_path, state_directory=root / "slurm"
    )


@dataclass(frozen=True)
class CommandResult:
    exit_code: int
    output: Any
    error: str


Command = Callable[..., CommandResult]


@pytest.fixture
def tinyruns(
    repository: Path,
    cluster: FakeCluster,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> Command:
    monkeypatch.chdir(repository)

    def run_command(*arguments: str) -> CommandResult:
        if arguments[0] == "submit":
            arguments = (*arguments, "--cluster", str(cluster.profile_path))
        exit_code = main(list(arguments))
        captured = capsys.readouterr()
        output = json.loads(captured.out) if captured.out.strip() else None
        return CommandResult(exit_code=exit_code, output=output, error=captured.err)

    return run_command
