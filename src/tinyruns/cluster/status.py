import json
from pathlib import Path
from shlex import quote
from typing import Any

from tinyruns.cluster.remote import Remote
from tinyruns.cluster.slurm import slurm_state
from tinyruns.run_directory import (
    FAILED,
    QUEUED,
    RUNNING,
    SLURM_LOG_NAME,
    TERMINAL_STATES,
    RunDirectory,
    current_time,
    read_json,
    read_status,
    write_status,
)

QUEUED_SLURM_STATES = frozenset(
    {"PENDING", "CONFIGURING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED"}
)
RUNNING_SLURM_STATES = frozenset(
    {"RUNNING", "COMPLETING", "SUSPENDED", "STOPPED", "RESIZING", "SIGNALING"}
)
FAILURE_LOG_LINES = 30


def local_runs(runs_root: Path) -> list[RunDirectory]:
    if not runs_root.is_dir():
        return []
    candidates = (RunDirectory(path) for path in sorted(runs_root.iterdir()))
    return [run for run in candidates if run.manifest_path.is_file()]


def is_cluster_run(run_directory: RunDirectory) -> bool:
    return read_json(run_directory.manifest_path)["cluster"] is not None


def remote_of(manifest: dict[str, Any]) -> Remote:
    return Remote(manifest["cluster"]["ssh_host"])


def current_job_id(manifest: dict[str, Any]) -> str | None:
    job_ids = manifest["slurm"]["job_ids"]
    return job_ids[-1] if job_ids else None


def log_tail(manifest: dict[str, Any], lines: int) -> str:
    log_path = f"{manifest['cluster']['run_directory']}/{SLURM_LOG_NAME}"
    return remote_of(manifest).run(
        f"tail -n {lines} {quote(log_path)} 2>/dev/null || true"
    )


def read_remote_json(remote: Remote, path: str) -> dict[str, Any]:
    return json.loads(remote.run(f"cat {quote(path)}"))


def reconciled_status(
    reported: dict[str, Any], job_state: str | None, job_id: str
) -> dict[str, Any]:
    if reported["state"] in TERMINAL_STATES:
        return reported
    if job_state in QUEUED_SLURM_STATES:
        return {**reported, "state": QUEUED, "slurm_state": job_state}
    if job_state in RUNNING_SLURM_STATES:
        return {**reported, "state": RUNNING, "slurm_state": job_state}
    failure = (
        f"Slurm no longer knows job {job_id}"
        if job_state is None
        else f"Slurm ended job {job_id} as {job_state} before the run recorded a result"
    )
    return {
        **reported,
        "state": FAILED,
        "slurm_state": job_state,
        "failure": failure,
        "finished_at": current_time(),
    }


def cluster_status(manifest: dict[str, Any], job_id: str) -> dict[str, Any]:
    remote = remote_of(manifest)
    status_path = f"{manifest['cluster']['run_directory']}/status.json"
    # Slurm is asked before the status file is read: read the other way round, a
    # job that finishes between the two calls would look like it died silently.
    job_state = slurm_state(remote, job_id)
    reported = read_remote_json(remote, status_path)
    status = reconciled_status(reported, job_state, job_id)
    if status["state"] == FAILED:
        status["log_tail"] = log_tail(manifest, FAILURE_LOG_LINES)
    return status


def refresh_status(run_directory: RunDirectory) -> dict[str, Any]:
    manifest = read_json(run_directory.manifest_path)
    job_id = current_job_id(manifest)
    if job_id is None:
        return read_status(run_directory)
    status = cluster_status(manifest, job_id)
    write_status(run_directory, status)
    return status
