import tomllib
from pathlib import Path

from tinyruns.cluster.errors import ExperimentError

PROJECT_FILE = Path("pyproject.toml")


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
