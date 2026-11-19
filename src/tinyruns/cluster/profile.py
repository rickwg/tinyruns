import getpass
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tinyruns.cluster.errors import ExperimentError

TIME_LIMIT_PATTERN = re.compile(r"(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})")
REMOTE_PATH_PATTERN = re.compile(r"/[A-Za-z0-9_./-]+")
ENVIRONMENT_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class Resources:
    name: str
    partition: str
    time: str
    cpus: int
    memory: str
    gpus: int = 0
    constraint: str | None = None


@dataclass(frozen=True)
class ClusterProfile:
    ssh_host: str | None
    code_root: str
    scratch_root: str
    node_local_scratch: str | None
    environment: dict[str, str]
    gpu_hours_without_confirmation: float
    image_build: Resources
    resources: dict[str, Resources]

    @property
    def runs_root(self) -> str:
        return f"{self.scratch_root}/runs"

    @property
    def images_root(self) -> str:
        return f"{self.scratch_root}/images"

    @property
    def hugging_face_cache(self) -> str:
        return f"{self.scratch_root}/cache/hf"

    @property
    def apptainer_cache(self) -> str:
        return f"{self.scratch_root}/cache/apptainer"


def time_limit_hours(time_limit: str) -> float:
    match = TIME_LIMIT_PATTERN.fullmatch(time_limit)
    if match is None:
        raise ExperimentError(
            f"Time limit '{time_limit}' must look like HH:MM:SS or D-HH:MM:SS."
        )
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    return days * 24 + hours + minutes / 60 + seconds / 3600


def ssh_setting(ssh_host: str, name: str) -> str:
    completed = subprocess.run(["ssh", "-G", ssh_host], capture_output=True, text=True)
    for line in completed.stdout.splitlines():
        key, _, value = line.partition(" ")
        if key == name:
            return value
    raise ExperimentError(
        f"ssh has no '{name}' setting for host '{ssh_host}': "
        f"{completed.stderr.strip()}"
    )


def login_name(ssh_host: str | None) -> str:
    if ssh_host is None:
        return getpass.getuser()
    return ssh_setting(ssh_host, "user")


def remote_path(template: str, user: str) -> str:
    path = template.format(user=user).rstrip("/")
    if REMOTE_PATH_PATTERN.fullmatch(path) is None:
        raise ExperimentError(
            f"Cluster path '{path}' must be absolute and use only letters, digits "
            "and the characters _ . / -"
        )
    return path


def optional_remote_path(template: str | None, user: str) -> str | None:
    return None if template is None else remote_path(template, user)


def resources_from(name: str, settings: dict[str, Any]) -> Resources:
    if "time" not in settings:
        raise ExperimentError(
            f"Resources '{name}' need a `time` limit. Without one, Slurm applies the "
            "partition default and may end the job early."
        )
    resources = Resources(name=name, **settings)
    time_limit_hours(resources.time)
    return resources


def environment_from(settings: dict[str, Any]) -> dict[str, str]:
    for name in settings:
        if ENVIRONMENT_NAME_PATTERN.fullmatch(name) is None:
            raise ExperimentError(f"'{name}' is not an environment variable name.")
    return {name: str(value) for name, value in settings.items()}


def load_cluster_profile(path: Path) -> ClusterProfile:
    if not path.is_file():
        raise ExperimentError(f"Cluster profile {path} does not exist.")
    content = yaml.safe_load(path.read_text())
    ssh_host = content.get("ssh_host")
    user = login_name(ssh_host)
    return ClusterProfile(
        ssh_host=ssh_host,
        code_root=remote_path(content["code_root"], user),
        scratch_root=remote_path(content["scratch_root"], user),
        node_local_scratch=optional_remote_path(
            content.get("node_local_scratch"), user
        ),
        environment=environment_from(content.get("environment", {})),
        gpu_hours_without_confirmation=float(content["gpu_hours_without_confirmation"]),
        image_build=resources_from("image_build", content["image_build"]),
        resources={
            name: resources_from(name, settings)
            for name, settings in content["resources"].items()
        },
    )
