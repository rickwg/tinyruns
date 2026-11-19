import json
from pathlib import Path

import pytest

from conftest import Command

IMAGE_DEFINITION = Path("apptainer") / "image.def"
PYTHON_VERSION_FILE = ".python-version"


def test_init_image_writes_a_definition_for_the_python_of_the_project(
    tinyruns: Command, repository: Path
) -> None:
    (repository / IMAGE_DEFINITION).unlink()
    (repository / PYTHON_VERSION_FILE).write_text("3.13\n")

    written = tinyruns("init", "image")

    assert written.exit_code == 0, written.error
    assert written.output["written"] == str(IMAGE_DEFINITION)
    assert written.output["python"] == "3.13"
    definition = (repository / IMAGE_DEFINITION).read_text()
    assert "From: python:3.13-slim" in definition
    assert "{python_version}" not in definition
    assert "git ca-certificates" in definition
    assert "uv sync --frozen" in definition


def test_init_image_keeps_an_existing_definition(
    tinyruns: Command, repository: Path
) -> None:
    (repository / PYTHON_VERSION_FILE).write_text("3.13\n")
    existing_definition = (repository / IMAGE_DEFINITION).read_text()

    refused = tinyruns("init", "image")

    assert refused.exit_code == 1
    assert "exists already" in json.loads(refused.error)["error"]
    assert (repository / IMAGE_DEFINITION).read_text() == existing_definition


@pytest.mark.parametrize(
    ("pinned_version", "complaint"),
    [(None, "does not exist"), ("cpython@3.13", "a plain version")],
)
def test_init_image_needs_a_plain_python_version(
    tinyruns: Command, repository: Path, pinned_version: str | None, complaint: str
) -> None:
    (repository / IMAGE_DEFINITION).unlink()
    if pinned_version is not None:
        (repository / PYTHON_VERSION_FILE).write_text(pinned_version)

    refused = tinyruns("init", "image")

    assert refused.exit_code == 1
    assert complaint in json.loads(refused.error)["error"]
    assert not (repository / IMAGE_DEFINITION).exists()
