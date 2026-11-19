import re
from pathlib import Path
from typing import Any

from tinyruns.cluster.errors import ExperimentError
from tinyruns.cluster.image import IMAGE_DEFINITION

TEMPLATES_DIRECTORY = Path(__file__).parent.parent / "templates"
PYTHON_VERSION_FILE = Path(".python-version")
PYTHON_VERSION_PLACEHOLDER = "{python_version}"
PLAIN_VERSION_PATTERN = re.compile(r"\d+\.\d+(\.\d+)?")


def template(name: str) -> str:
    return (TEMPLATES_DIRECTORY / name).read_text()


def require_absent(path: Path) -> None:
    if path.exists():
        raise ExperimentError(
            f"{path} exists already and is not overwritten. Delete it to start from "
            "the template again."
        )


def write_project_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def project_python_version(repository: Path) -> str:
    version_file = repository / PYTHON_VERSION_FILE
    if not version_file.is_file():
        raise ExperimentError(
            f"{version_file} does not exist. The image takes its Python version "
            "from it; `uv python pin <version>` writes it."
        )
    version = version_file.read_text().strip()
    if PLAIN_VERSION_PATTERN.fullmatch(version) is None:
        raise ExperimentError(
            f"{version_file} holds '{version}'. The image needs a plain version "
            "such as 3.14."
        )
    return version


def create_image_definition(repository: Path) -> dict[str, Any]:
    require_absent(repository / IMAGE_DEFINITION)
    python_version = project_python_version(repository)
    write_project_file(
        repository / IMAGE_DEFINITION,
        template(IMAGE_DEFINITION.name).replace(
            PYTHON_VERSION_PLACEHOLDER, python_version
        ),
    )
    return {
        "written": str(IMAGE_DEFINITION),
        "python": python_version,
        "next": "Commit it. The file is this project's from now on, and the image "
        "is built again whenever it or uv.lock changes.",
    }
