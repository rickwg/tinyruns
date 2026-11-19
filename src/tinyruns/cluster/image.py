import hashlib
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

from tinyruns.cluster.job_script import (
    IMAGE_DEFINITION_NAME,
    ImageBuildJob,
    image_build_script,
)
from tinyruns.cluster.profile import ClusterProfile
from tinyruns.cluster.remote import Remote
from tinyruns.cluster.slurm import submit_job

IMAGE_DEFINITION = Path("apptainer") / IMAGE_DEFINITION_NAME
LOCK_FILE = Path("uv.lock")
PROJECT_FILE = Path("pyproject.toml")
BUILD_SCRIPT_NAME = "build.sbatch"


@dataclass(frozen=True)
class Image:
    path: str
    build_job_id: str | None


def image_stem(repository: Path) -> str:
    digest = hashlib.sha256()
    for environment_input in (IMAGE_DEFINITION, LOCK_FILE):
        digest.update((repository / environment_input).read_bytes())
    return digest.hexdigest()[:16]


def stage_build_inputs(repository: Path, staging_directory: Path, script: str) -> None:
    for build_input in (IMAGE_DEFINITION, LOCK_FILE, PROJECT_FILE):
        shutil.copy(repository / build_input, staging_directory / build_input.name)
    (staging_directory / BUILD_SCRIPT_NAME).write_text(script)


def submit_image_build(repository: Path, job: ImageBuildJob, remote: Remote) -> str:
    with tempfile.TemporaryDirectory() as staging:
        stage_build_inputs(repository, Path(staging), image_build_script(job))
        remote.upload_directory(Path(staging), job.build_directory)
    return submit_job(remote, f"{job.build_directory}/{BUILD_SCRIPT_NAME}")


def ensure_image(repository: Path, profile: ClusterProfile, remote: Remote) -> Image:
    stem = image_stem(repository)
    image_path = f"{profile.images_root}/{stem}.sif"
    if remote.test(f"test -f {quote(image_path)}"):
        return Image(path=image_path, build_job_id=None)
    job = ImageBuildJob(
        image_stem=stem,
        profile=profile,
        build_directory=f"{profile.images_root}/{stem}.build",
        image_path=image_path,
    )
    return Image(
        path=image_path, build_job_id=submit_image_build(repository, job, remote)
    )
