import getpass
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from tinyruns.cluster.errors import ExperimentError
from tinyruns.run_data import DATA_DIRECTORY_NAME

DEFAULT_CLUSTER_PROFILE = Path(".cluster.yaml")
TIME_LIMIT_PATTERN = re.compile(r"(?:(\d+)-)?(\d{1,2}):(\d{2}):(\d{2})")
REMOTE_PATH_PATTERN = re.compile(r"/[A-Za-z0-9_./-]+")
ENVIRONMENT_NAME_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
REQUIRED_SETTINGS = (
    "name",
    "code_root",
    "scratch_root",
    "gpu_hours_without_confirmation",
    "image_build",
    "resources",
)


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
    name: str
    ssh_host: str | None
    code_root: str
    scratch_root: str
    node_local_scratch: str | None
    environment: dict[str, str]
    gpu_hours_without_confirmation: float
    image_build: Resources
    resources: dict[str, Resources]
    smoke_resources: str | None

    @property
    def runs_root(self) -> str:
        return f"{self.scratch_root}/runs"

    @property
    def data_root(self) -> str:
        return f"{self.scratch_root}/{DATA_DIRECTORY_NAME}"

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


def remote_path(template: str, placeholders: dict[str, str]) -> str:
    try:
        path = template.format(**placeholders).rstrip("/")
    except KeyError, IndexError, ValueError:
        known_placeholders = " and ".join(f"{{{name}}}" for name in placeholders)
        raise ExperimentError(
            f"Cluster path '{template}' may contain only the placeholders "
            f"{known_placeholders}."
        ) from None
    if REMOTE_PATH_PATTERN.fullmatch(path) is None:
        raise ExperimentError(
            f"Cluster path '{path}' must be absolute and use only letters, digits "
            "and the characters _ . / -"
        )
    return path


def optional_remote_path(
    template: str | None, placeholders: dict[str, str]
) -> str | None:
    return None if template is None else remote_path(template, placeholders)


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


def smoke_resources_from(
    content: dict[str, Any], resources: dict[str, Resources]
) -> str | None:
    name = content.get("smoke_resources")
    if name is not None and name not in resources:
        raise ExperimentError(
            f"`smoke_resources` is '{name}', which is not one of the profile's "
            f"resources: {sorted(resources)}."
        )
    return name


def profile_content(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentError(
            f"Cluster profile {path} does not exist. `tinyruns init cluster` writes "
            f"{DEFAULT_CLUSTER_PROFILE} from a template."
        )
    content = yaml.safe_load(path.read_text())
    missing_settings = [
        setting
        for setting in REQUIRED_SETTINGS
        if not isinstance(content, dict) or setting not in content
    ]
    if missing_settings:
        raise ExperimentError(
            f"Cluster profile {path} lacks these settings: {missing_settings}."
        )
    return content


def load_cluster_profile(path: Path, project: str) -> ClusterProfile:
    content = profile_content(path)
    ssh_host = content.get("ssh_host")
    placeholders = {"user": login_name(ssh_host), "project": project}
    resources = {
        name: resources_from(name, settings)
        for name, settings in content["resources"].items()
    }
    return ClusterProfile(
        name=str(content["name"]),
        ssh_host=ssh_host,
        code_root=remote_path(content["code_root"], placeholders),
        scratch_root=remote_path(content["scratch_root"], placeholders),
        node_local_scratch=optional_remote_path(
            content.get("node_local_scratch"), placeholders
        ),
        environment=environment_from(content.get("environment", {})),
        gpu_hours_without_confirmation=float(content["gpu_hours_without_confirmation"]),
        image_build=resources_from("image_build", content["image_build"]),
        resources=resources,
        smoke_resources=smoke_resources_from(content, resources),
    )
