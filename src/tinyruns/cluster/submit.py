import dataclasses
from dataclasses import dataclass
from pathlib import Path
from shlex import quote
from typing import Any

from tinyruns.cluster.errors import ExperimentError
from tinyruns.cluster.guardrails import (
    require_committed_code,
    require_confirmed_gpu_hours,
    require_passed_smoke_run,
    require_usable_input_runs,
)
from tinyruns.cluster.image import ensure_image
from tinyruns.cluster.job_script import RunJob, run_job_script
from tinyruns.cluster.new_run import (
    PlannedRun,
    create_run_directory,
    local_manifest,
    planned_run,
    runs_root,
)
from tinyruns.cluster.profile import (
    ClusterProfile,
    Resources,
    load_cluster_profile,
)
from tinyruns.cluster.project import project_name
from tinyruns.cluster.remote import Remote
from tinyruns.cluster.slurm import submit_job
from tinyruns.cluster.snapshot import upload_code_snapshot
from tinyruns.cluster.status import refresh_status, remote_of
from tinyruns.run_config import input_run_ids
from tinyruns.run_directory import (
    FAILED,
    JOB_SCRIPT_NAME,
    QUEUED,
    RunDirectory,
    current_time,
    read_json,
    write_json,
    write_status,
)


@dataclass(frozen=True)
class SubmitRequest:
    repository: Path
    config_path: Path
    cluster_profile_path: Path
    resources: str | None
    smoke: bool
    confirmed: bool
    skip_smoke_check: bool
    input_run_ids: dict[str, str]


@dataclass(frozen=True)
class ClusterPlacement:
    profile: ClusterProfile
    resources: Resources
    run_directory: str


def default_resources(profile: ClusterProfile, smoke: bool) -> str:
    if not smoke:
        raise ExperimentError(
            f"A full run needs --resources, one of {sorted(profile.resources)}."
        )
    if profile.smoke_resources is None:
        raise ExperimentError(
            f"Pass --resources, one of {sorted(profile.resources)}, or name the "
            "resources of smoke runs as `smoke_resources` in the cluster profile."
        )
    return profile.smoke_resources


def selected_resources(
    requested: str | None, profile: ClusterProfile, smoke: bool
) -> Resources:
    name = requested or default_resources(profile, smoke)
    if name not in profile.resources:
        raise ExperimentError(
            f"The cluster profile has no resources '{name}'. It has "
            f"{sorted(profile.resources)}."
        )
    return profile.resources[name]


def cluster_placement(request: SubmitRequest, plan: PlannedRun) -> ClusterPlacement:
    profile = load_cluster_profile(
        request.cluster_profile_path, project_name(request.repository)
    )
    return ClusterPlacement(
        profile=profile,
        resources=selected_resources(request.resources, profile, plan.smoke),
        run_directory=f"{profile.runs_root}/{plan.run_id}",
    )


def check_guardrails(
    request: SubmitRequest, plan: PlannedRun, placement: ClusterPlacement
) -> None:
    require_committed_code(plan.snapshot, plan.smoke)
    require_confirmed_gpu_hours(
        placement.resources, placement.profile, request.confirmed
    )
    if not plan.smoke and not request.skip_smoke_check:
        require_passed_smoke_run(
            runs_root(request.repository), plan.snapshot.tree, plan.config_hash
        )
    require_usable_input_runs(
        Remote(placement.profile.ssh_host),
        placement.profile,
        input_run_ids(plan.config),
        plan.smoke,
    )


def cluster_manifest(plan: PlannedRun, placement: ClusterPlacement) -> dict[str, Any]:
    return {
        **local_manifest(plan),
        "resources": dataclasses.asdict(placement.resources),
        "cluster": {
            "ssh_host": placement.profile.ssh_host,
            "run_directory": placement.run_directory,
        },
    }


def queued_status() -> dict[str, Any]:
    return {"state": QUEUED, "queued_at": current_time()}


def create_remote_roots(remote: Remote, profile: ClusterProfile) -> None:
    roots = (
        profile.code_root,
        profile.runs_root,
        profile.images_root,
        profile.hugging_face_cache,
    )
    remote.run("mkdir -p " + " ".join(quote(root) for root in roots))


def record_jobs(
    run_directory: RunDirectory, manifest: dict[str, Any], remote: Remote
) -> None:
    write_json(run_directory.manifest_path, manifest)
    remote.upload_file(
        run_directory.manifest_path,
        f"{manifest['cluster']['run_directory']}/{run_directory.manifest_path.name}",
    )


def launch(
    repository: Path,
    plan: PlannedRun,
    placement: ClusterPlacement,
    run_directory: RunDirectory,
) -> dict[str, Any]:
    profile = placement.profile
    remote = Remote(profile.ssh_host)
    create_remote_roots(remote, profile)
    code_directory = upload_code_snapshot(
        repository, plan.snapshot, remote, profile.code_root
    )
    image = ensure_image(repository, profile, remote)
    job = RunJob(
        run_id=plan.run_id,
        smoke=plan.smoke,
        resources=placement.resources,
        profile=profile,
        code_directory=code_directory,
        run_directory=placement.run_directory,
        image_path=image.path,
        input_run_ids=tuple(sorted(set(input_run_ids(plan.config).values()))),
    )
    run_directory.job_script_path.write_text(run_job_script(job))
    remote.upload_directory(run_directory.path, placement.run_directory)
    job_id = submit_job(
        remote,
        f"{placement.run_directory}/{JOB_SCRIPT_NAME}",
        after_job_id=image.build_job_id,
    )
    manifest = {
        **read_json(run_directory.manifest_path),
        "image": image.path,
        "slurm": {"job_ids": [job_id], "image_build_job_id": image.build_job_id},
    }
    record_jobs(run_directory, manifest, remote)
    return manifest


def submit_run(request: SubmitRequest) -> dict[str, Any]:
    plan = planned_run(
        request.repository, request.config_path, request.smoke, request.input_run_ids
    )
    placement = cluster_placement(request, plan)
    check_guardrails(request, plan, placement)
    run_directory = create_run_directory(
        request.repository, plan, cluster_manifest(plan, placement), queued_status()
    )
    try:
        return launch(request.repository, plan, placement, run_directory)
    except Exception as error:
        write_status(
            run_directory,
            {
                "state": FAILED,
                "failure": f"Submission failed: {error}",
                "finished_at": current_time(),
            },
        )
        raise


def resume_run(run_directory: RunDirectory) -> dict[str, Any]:
    manifest = read_json(run_directory.manifest_path)
    if not manifest["slurm"]["job_ids"]:
        raise ExperimentError("This run never reached Slurm. Submit its config again.")
    state = refresh_status(run_directory)["state"]
    if state != FAILED:
        raise ExperimentError(f"Only a failed run can be resumed; this run is {state}.")
    remote = remote_of(manifest)
    remote_run_directory = manifest["cluster"]["run_directory"]
    write_status(run_directory, queued_status())
    remote.upload_file(
        run_directory.status_path,
        f"{remote_run_directory}/{run_directory.status_path.name}",
    )
    job_id = submit_job(remote, f"{remote_run_directory}/{JOB_SCRIPT_NAME}")
    manifest["slurm"]["job_ids"].append(job_id)
    record_jobs(run_directory, manifest, remote)
    return manifest
