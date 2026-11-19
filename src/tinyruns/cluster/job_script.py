from dataclasses import dataclass
from shlex import quote

from tinyruns.cluster.profile import ClusterProfile, Resources
from tinyruns.run_directory import SLURM_LOG_NAME

CONTAINER_CODE = "/workdir"
CONTAINER_RUNS = "/mnt/runs"
CONTAINER_DATA = "/mnt/data"
CONTAINER_HUGGING_FACE_CACHE = "/mnt/cache/hf"
IMAGE_DEFINITION_NAME = "image.def"


@dataclass(frozen=True)
class RunJob:
    run_id: str
    smoke: bool
    resources: Resources
    profile: ClusterProfile
    code_directory: str
    run_directory: str
    image_path: str
    input_run_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageBuildJob:
    image_stem: str
    profile: ClusterProfile
    build_directory: str
    image_path: str


def sbatch_directives(job_name: str, resources: Resources, log_path: str) -> list[str]:
    directives = [
        f"--job-name={job_name}",
        f"--partition={resources.partition}",
        f"--time={resources.time}",
        f"--cpus-per-task={resources.cpus}",
        f"--mem={resources.memory}",
        f"--output={log_path}",
    ]
    if resources.gpus > 0:
        directives.append(f"--gpus={resources.gpus}")
    if resources.constraint is not None:
        directives.append(f"--constraint={resources.constraint}")
    return [f"#SBATCH {directive}" for directive in directives]


def container_environment(profile: ClusterProfile, smoke: bool) -> dict[str, str]:
    environment = {
        **profile.environment,
        "PYTHONPATH": f"{CONTAINER_CODE}/src",
        "PYTHONNOUSERSITE": "1",
        "HF_HOME": CONTAINER_HUGGING_FACE_CACHE,
    }
    if not smoke:
        environment["HF_HUB_OFFLINE"] = "1"
    return environment


def container_run_directory(run_id: str) -> str:
    return f"{CONTAINER_RUNS}/{run_id}"


def input_run_bind(profile: ClusterProfile, input_run_id: str) -> str:
    return (
        f"{profile.runs_root}/{input_run_id}:"
        f"{container_run_directory(input_run_id)}:ro"
    )


def apptainer_exec_arguments(job: RunJob) -> list[str]:
    binds = [
        f"{job.code_directory}:{CONTAINER_CODE}:ro",
        f"{job.run_directory}:{container_run_directory(job.run_id)}",
        *(input_run_bind(job.profile, run_id) for run_id in job.input_run_ids),
        f"{job.profile.data_root}:{CONTAINER_DATA}:ro",
        f"{job.profile.hugging_face_cache}:{CONTAINER_HUGGING_FACE_CACHE}",
    ]
    environment = container_environment(job.profile, job.smoke)
    return [
        "exec apptainer exec",
        *(["--nv"] if job.resources.gpus > 0 else []),
        *(f"--bind {quote(bind)}" for bind in binds),
        *(f"--env {quote(f'{name}={value}')}" for name, value in environment.items()),
        f"--pwd {CONTAINER_CODE}",
        quote(job.image_path),
        f"python -m tinyruns.run {container_run_directory(job.run_id)}",
    ]


def run_job_script(job: RunJob) -> str:
    lines = [
        "#!/usr/bin/env bash",
        *sbatch_directives(
            job.run_id, job.resources, f"{job.run_directory}/{SLURM_LOG_NAME}"
        ),
        "#SBATCH --open-mode=append",
        "#SBATCH --kill-on-invalid-dep=yes",
        "set -euo pipefail",
        f"mkdir -p {quote(job.profile.hugging_face_cache)} "
        f"{quote(job.profile.data_root)}",
        " \\\n    ".join(apptainer_exec_arguments(job)),
    ]
    return "\n".join(lines) + "\n"


def build_scratch_command(profile: ClusterProfile) -> str:
    if profile.node_local_scratch is None:
        return "mktemp -d"
    return f"mktemp -d {quote(profile.node_local_scratch)}/image-build.XXXXXX"


def image_build_script(job: ImageBuildJob) -> str:
    partial_image = f'{quote(job.image_path)}.partial-"$SLURM_JOB_ID"'
    lines = [
        "#!/usr/bin/env bash",
        *sbatch_directives(
            f"image-{job.image_stem}",
            job.profile.image_build,
            f"{job.build_directory}/{SLURM_LOG_NAME}",
        ),
        "set -euo pipefail",
        *(
            f"export {name}={quote(value)}"
            for name, value in job.profile.environment.items()
        ),
        f"export APPTAINER_CACHEDIR={quote(job.profile.apptainer_cache)}",
        'mkdir -p "$APPTAINER_CACHEDIR"',
        f"build_scratch=$({build_scratch_command(job.profile)})",
        "trap 'rm -rf \"$build_scratch\"' EXIT",
        'export APPTAINER_TMPDIR="$build_scratch"',
        f"cd {quote(job.build_directory)}",
        f'apptainer build --fakeroot "$build_scratch/image.sif" {IMAGE_DEFINITION_NAME}',
        f'cp "$build_scratch/image.sif" {partial_image}',
        f"mv {partial_image} {quote(job.image_path)}",
    ]
    return "\n".join(lines) + "\n"
