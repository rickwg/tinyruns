import math
import time
from collections.abc import Iterator
from typing import Any

from tinyruns.cluster.errors import ExperimentError
from tinyruns.cluster.remote import RemoteCommandError
from tinyruns.cluster.slurm import cancel_job
from tinyruns.cluster.status import (
    current_job_id,
    log_tail,
    refresh_status,
    remote_of,
)
from tinyruns.run_directory import (
    HEAVY_DIRECTORY_NAME,
    TERMINAL_STATES,
    RunDirectory,
    read_json,
    read_status,
)

MAX_CONSECUTIVE_FAILED_POLLS = 5
CANCEL_POLL_SECONDS = 2.0
CANCEL_TIMEOUT_SECONDS = 120.0


def run_summary(run_directory: RunDirectory) -> dict[str, Any]:
    manifest = read_json(run_directory.manifest_path)
    return {
        "run_id": manifest["run_id"],
        "state": read_status(run_directory)["state"],
        "location": "local" if manifest["cluster"] is None else "cluster",
        "smoke": manifest["smoke"],
        "config_name": manifest["config_name"],
        "created_at": manifest["created_at"],
        "job_id": current_job_id(manifest),
    }


def polled_statuses(
    run_directory: RunDirectory, interval_seconds: float
) -> Iterator[dict[str, Any]]:
    failed_polls = 0
    while True:
        try:
            status = refresh_status(run_directory)
        except RemoteCommandError:
            failed_polls += 1
            if failed_polls >= MAX_CONSECUTIVE_FAILED_POLLS:
                raise
        else:
            failed_polls = 0
            yield status
        time.sleep(interval_seconds)


def wait_for_run(
    run_directory: RunDirectory, interval_seconds: float, timeout_seconds: float | None
) -> dict[str, Any]:
    deadline = (
        math.inf if timeout_seconds is None else time.monotonic() + timeout_seconds
    )
    return next(
        status
        for status in polled_statuses(run_directory, interval_seconds)
        if status["state"] in TERMINAL_STATES or time.monotonic() >= deadline
    )


def run_log(run_directory: RunDirectory, lines: int) -> str:
    return log_tail(read_json(run_directory.manifest_path), lines)


def pull_run(run_directory: RunDirectory, include_heavy: bool) -> dict[str, Any]:
    manifest = read_json(run_directory.manifest_path)
    excludes = () if include_heavy else (f"/{HEAVY_DIRECTORY_NAME}/",)
    remote_of(manifest).download_directory(
        manifest["cluster"]["run_directory"], run_directory.path, excludes
    )
    return refresh_status(run_directory)


def cancel_run(run_directory: RunDirectory) -> dict[str, Any]:
    manifest = read_json(run_directory.manifest_path)
    job_id = current_job_id(manifest)
    if job_id is None:
        raise ExperimentError(
            "This run never reached Slurm; there is no job to cancel."
        )
    cancel_job(remote_of(manifest), job_id)
    status = wait_for_run(run_directory, CANCEL_POLL_SECONDS, CANCEL_TIMEOUT_SECONDS)
    if status["state"] not in TERMINAL_STATES:
        raise ExperimentError(
            f"Slurm was asked to cancel job {job_id} and has not ended it within "
            f"{CANCEL_TIMEOUT_SECONDS:g} seconds. `tinyruns status "
            f"{manifest['run_id']}` shows when it has."
        )
    return status
