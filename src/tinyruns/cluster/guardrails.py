from pathlib import Path
from shlex import quote
from typing import Any

from tinyruns.cluster.errors import ExperimentError
from tinyruns.cluster.profile import (
    ClusterProfile,
    Resources,
    time_limit_hours,
)
from tinyruns.cluster.remote import Remote
from tinyruns.cluster.snapshot import SNAPSHOT_PATHS, CodeSnapshot
from tinyruns.cluster.status import (
    local_runs,
    read_remote_json,
    refresh_status,
)
from tinyruns.run_config import stage_names
from tinyruns.run_directory import DONE, RunDirectory, read_json

STAGES_DIRECTORY = Path("src") / "tinyruns" / "stages"


def require_known_stages(repository: Path, config: dict[str, Any]) -> None:
    for stage_name in stage_names(config):
        stage_file = repository / STAGES_DIRECTORY / f"{stage_name}.py"
        if not stage_file.is_file():
            raise ExperimentError(
                f"Stage '{stage_name}' does not exist: {stage_file} is missing."
            )


def require_committed_code(snapshot: CodeSnapshot, smoke: bool) -> None:
    if snapshot.is_dirty and not smoke:
        raise ExperimentError(
            f"Uncommitted changes in {', '.join(SNAPSHOT_PATHS)}. Commit them before "
            "a full run, or pass --smoke to test the working tree."
        )


def gpu_hours(resources: Resources) -> float:
    return resources.gpus * time_limit_hours(resources.time)


def require_confirmed_gpu_hours(
    resources: Resources, profile: ClusterProfile, confirmed: bool
) -> None:
    requested = gpu_hours(resources)
    if requested > profile.gpu_hours_without_confirmation and not confirmed:
        raise ExperimentError(
            f"Resources '{resources.name}' reserve {requested:g} GPU hours, above the "
            f"limit of {profile.gpu_hours_without_confirmation:g}. Pass --confirm "
            "once a human has approved this."
        )


def is_smoke_run_of(
    run_directory: RunDirectory, code_tree: str, config_hash: str
) -> bool:
    manifest = read_json(run_directory.manifest_path)
    return (
        manifest["smoke"]
        and manifest["cluster"] is not None
        and manifest["code"]["tree"] == code_tree
        and manifest["config_hash"] == config_hash
    )


def require_passed_smoke_run(runs_root: Path, code_tree: str, config_hash: str) -> None:
    smoke_runs = [
        run_directory
        for run_directory in local_runs(runs_root)
        if is_smoke_run_of(run_directory, code_tree, config_hash)
    ]
    if not any(refresh_status(run)["state"] == DONE for run in smoke_runs):
        raise ExperimentError(
            "No smoke run of this code and config has finished successfully on the "
            "cluster. Submit with --smoke first, wait for it, then submit the full run."
        )


def require_usable_input_run(
    remote: Remote, input_run_directory: str, name: str, smoke: bool
) -> None:
    described = f"Input '{name}' is run '{input_run_directory.rsplit('/', 1)[-1]}'"
    status_path = f"{input_run_directory}/status.json"
    if not remote.test(f"test -f {quote(status_path)}"):
        raise ExperimentError(
            f"{described}, which is not on the cluster at {input_run_directory}."
        )
    state = read_remote_json(remote, status_path)["state"]
    if state != DONE:
        raise ExperimentError(
            f"{described}, which is {state} on the cluster. An input run must be done."
        )
    manifest = read_remote_json(remote, f"{input_run_directory}/manifest.json")
    if manifest["smoke"] and not smoke:
        raise ExperimentError(
            f"{described}, a smoke run. A full run must read full runs; pass --smoke "
            "to test against a smoke run."
        )


def require_usable_input_runs(
    remote: Remote, profile: ClusterProfile, input_run_ids: dict[str, str], smoke: bool
) -> None:
    for name, run_id in input_run_ids.items():
        require_usable_input_run(remote, f"{profile.runs_root}/{run_id}", name, smoke)
