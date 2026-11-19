import tomllib
from pathlib import Path
from typing import Any

from tinyruns.cluster.errors import ExperimentError

PROJECT_FILE = Path("pyproject.toml")
LOCK_FILE = Path("uv.lock")
LOCAL_SOURCE_KINDS = ("path", "directory", "editable", "virtual")
THE_PROJECT_ITSELF = "."


def project_name(repository: Path) -> str:
    project_file = repository / PROJECT_FILE
    if not project_file.is_file():
        raise ExperimentError(f"{project_file} does not exist.")
    name = tomllib.loads(project_file.read_text()).get("project", {}).get("name")
    if not isinstance(name, str):
        raise ExperimentError(
            f"{project_file} needs a `name` under `[project]`. It names this "
            "project's directories on the cluster."
        )
    return name


def is_from_a_local_path(source: dict[str, Any]) -> bool:
    return any(
        source.get(kind) not in (None, THE_PROJECT_ITSELF)
        for kind in LOCAL_SOURCE_KINDS
    )


def packages_locked_to_local_paths(repository: Path) -> list[str]:
    lock = tomllib.loads((repository / LOCK_FILE).read_text())
    return [
        package["name"]
        for package in lock.get("package", [])
        if is_from_a_local_path(package.get("source", {}))
    ]
