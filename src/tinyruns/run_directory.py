import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

CREATED = "created"
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
TERMINAL_STATES = frozenset({DONE, FAILED})

HEAVY_DIRECTORY_NAME = "heavy"
JOB_SCRIPT_NAME = "job.sbatch"
SLURM_LOG_NAME = "slurm.out"


@dataclass(frozen=True)
class RunDirectory:
    path: Path

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def status_path(self) -> Path:
        return self.path / "status.json"

    @property
    def job_script_path(self) -> Path:
        return self.path / JOB_SCRIPT_NAME

    @property
    def code_patch_path(self) -> Path:
        return self.path / "code.patch"

    @property
    def metrics_path(self) -> Path:
        return self.path / "metrics.jsonl"

    @property
    def report_path(self) -> Path:
        return self.path / "report.md"

    @property
    def figures_directory(self) -> Path:
        return self.path / "figures"

    @property
    def heavy_directory(self) -> Path:
        return self.path / HEAVY_DIRECTORY_NAME


def current_time() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, content: dict[str, Any]) -> None:
    partial_path = path.with_name(f"{path.name}.partial")
    partial_path.write_text(json.dumps(content, indent=2) + "\n")
    os.replace(partial_path, path)


def read_status(run_directory: RunDirectory) -> dict[str, Any]:
    return read_json(run_directory.status_path)


def write_status(run_directory: RunDirectory, status: dict[str, Any]) -> None:
    write_json(run_directory.status_path, status)


def append_metrics(run_directory: RunDirectory, record: dict[str, Any]) -> None:
    with run_directory.metrics_path.open("a") as metrics_file:
        metrics_file.write(json.dumps(record) + "\n")
