import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import FIXTURE_PROJECT_NAME, Command, FakeCluster
from tinyruns.cluster.guide import SKILL_FILE, guide_text
from tinyruns.cluster.main import argument_parser, main
from tinyruns.cluster.profile import load_cluster_profile

CLUSTER_PROFILE = ".cluster.yaml"
IMAGE_DEFINITION = Path("apptainer") / "image.def"
PYTHON_VERSION_FILE = ".python-version"


def test_init_cluster_writes_a_profile_that_loads(
    tinyruns: Command, repository: Path
) -> None:
    written = tinyruns("init", "cluster")

    assert written.exit_code == 0, written.error
    assert written.output["written"] == CLUSTER_PROFILE
    assert written.output["ignored_by_git"] is False
    profile = load_cluster_profile(repository / CLUSTER_PROFILE, FIXTURE_PROJECT_NAME)
    assert profile.name == "my-cluster"
    assert profile.scratch_root.endswith(f"/{FIXTURE_PROJECT_NAME}")
    assert profile.smoke_resources in profile.resources


def test_init_cluster_reports_a_profile_that_git_ignores(
    tinyruns: Command, repository: Path
) -> None:
    (repository / ".git" / "info" / "exclude").write_text(f"{CLUSTER_PROFILE}\n")

    written = tinyruns("init", "cluster")

    assert written.output["ignored_by_git"] is True


def test_init_cluster_keeps_an_existing_profile(
    tinyruns: Command, repository: Path
) -> None:
    (repository / CLUSTER_PROFILE).write_text("name: mine\n")

    refused = tinyruns("init", "cluster")

    assert refused.exit_code == 1
    assert "exists already" in json.loads(refused.error)["error"]
    assert (repository / CLUSTER_PROFILE).read_text() == "name: mine\n"


def test_missing_profile_points_to_init_cluster(
    tinyruns: Command, cluster: FakeCluster
) -> None:
    cluster.profile_path.unlink()

    refused = tinyruns("submit", "configs/marker.yaml", "--smoke")

    assert "tinyruns init cluster" in json.loads(refused.error)["error"]


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


def frozen_config(cluster: FakeCluster, run_id: str) -> dict[str, Any]:
    return yaml.safe_load((cluster.run_directory(run_id) / "config.yaml").read_text())


def test_check_queues_the_environment_check_as_a_smoke_run(
    tinyruns: Command, cluster: FakeCluster
) -> None:
    cluster.hold_new_jobs()

    submission = tinyruns("check")

    assert submission.exit_code == 0, submission.error
    assert submission.output["smoke"] is True
    assert submission.output["config_name"] == "environment_check"
    assert submission.output["resources"]["name"] == "cpu-small"
    config = frozen_config(cluster, submission.output["run_id"])
    assert config["stages"] == ["tinyruns.stages.environment_check"]
    assert config["environment_check"]["require_gpu"] is False
    assert "https://github.com" in config["environment_check"]["urls"]


def test_gpu_check_requires_a_gpu_on_the_named_resources(
    tinyruns: Command, cluster: FakeCluster
) -> None:
    cluster.hold_new_jobs()

    unnamed = tinyruns("check", "--gpu")
    submission = tinyruns("check", "--gpu", "--resources", "gpu-short")

    assert "needs --resources" in json.loads(unnamed.error)["error"]
    assert submission.exit_code == 0, submission.error
    run_id = submission.output["run_id"]
    assert frozen_config(cluster, run_id)["environment_check"]["require_gpu"] is True
    assert "--nv" in (cluster.run_directory(run_id) / "job.sbatch").read_text()


def command_names() -> list[str]:
    listed_commands = re.search(r"\{(.+?)\}", argument_parser().format_usage())
    return listed_commands.group(1).split(",")


def test_guide_prints_the_skill_without_its_front_matter(
    repository: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(repository)

    exit_code = main(["guide"])

    printed = capsys.readouterr().out
    assert exit_code == 0
    assert printed == guide_text()
    assert printed.startswith("# tinyruns\n")
    assert SKILL_FILE.read_text().startswith("---\nname: tinyruns\ndescription: ")
    assert SKILL_FILE.read_text().endswith(printed)


def test_guide_names_every_command() -> None:
    commands = command_names()

    assert "submit" in commands and "guide" in commands
    for command in commands:
        assert f"tinyruns {command}" in guide_text()
