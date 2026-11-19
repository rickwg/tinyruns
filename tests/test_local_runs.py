import importlib.metadata
import json
from pathlib import Path

import pytest
import yaml

from conftest import (
    MARKER_CONFIG,
    ON_CPU,
    STAGES_PACKAGE,
    Command,
    FakeCluster,
    run_locally,
    write_config,
)

TINYRUNS_VERSION = importlib.metadata.version("tinyruns")


def test_new_run_is_created_and_run_without_the_cluster(
    tinyruns: Command, repository: Path, cluster: FakeCluster
) -> None:
    created = tinyruns("new", "configs/marker.yaml", "--smoke")

    assert created.exit_code == 0, created.error
    run_id = created.output["run_id"]
    run_directory = repository / created.output["directory"]
    assert run_id.endswith("-smoke")
    assert created.output["directory"] == f"artifacts/runs/{run_id}"
    assert created.output["cluster"] is None
    assert created.output["tinyruns_version"] == TINYRUNS_VERSION
    frozen_config = yaml.safe_load((run_directory / "config.yaml").read_text())
    assert frozen_config["smoke_test"]["enabled"] is True
    assert tinyruns("status", run_id).output["state"] == "created"
    assert cluster.job_ids() == []
    assert not cluster.scratch_root.exists()

    run_locally(repository, created.output["run_with"])

    assert (run_directory / "report.md").read_text() == "marker report"
    listed = tinyruns("ls").output
    assert [(run["state"], run["location"]) for run in listed] == [("done", "local")]


def test_new_run_records_uncommitted_code(tinyruns: Command, repository: Path) -> None:
    (repository / "src" / "fixture_project" / "draft.py").write_text("DRAFT = 1\n")

    created = tinyruns("new", "configs/marker.yaml").output

    assert created["code"]["dirty"] is True
    assert not created["run_id"].endswith("-smoke")
    code_patch = repository / created["directory"] / "code.patch"
    assert "DRAFT = 1" in code_patch.read_text()


def test_local_smoke_run_does_not_unlock_a_full_cluster_run(
    tinyruns: Command, repository: Path
) -> None:
    created = tinyruns("new", "configs/marker.yaml", "--smoke").output
    run_locally(repository, created["run_with"])

    refused = tinyruns("submit", "configs/marker.yaml", *ON_CPU)
    cluster_smoke = tinyruns("submit", "configs/marker.yaml", "--smoke")
    allowed = tinyruns("submit", "configs/marker.yaml", *ON_CPU)

    assert refused.exit_code == 1
    assert "on the cluster" in json.loads(refused.error)["error"]
    assert allowed.exit_code == 0, allowed.error
    locations = {run["run_id"]: run["location"] for run in tinyruns("ls").output}
    assert locations[created["run_id"]] == "local"
    assert locations[cluster_smoke.output["run_id"]] == "cluster"


@pytest.mark.parametrize("command", ["wait", "logs", "pull", "cancel", "resume"])
def test_cluster_commands_explain_how_to_run_a_local_run(
    tinyruns: Command, command: str
) -> None:
    created = tinyruns("new", "configs/marker.yaml").output

    rejected = tinyruns(command, created["run_id"])

    assert rejected.exit_code == 1
    assert created["run_with"] in json.loads(rejected.error)["error"]
    assert tinyruns("status").output == []


def test_new_rejects_configs_that_cannot_run(
    tinyruns: Command, repository: Path
) -> None:
    write_config(repository, "located", {**MARKER_CONFIG, "output": "/tmp/results"})
    write_config(
        repository,
        "no_stage",
        {**MARKER_CONFIG, "stages": [f"{STAGES_PACKAGE}.absent"]},
    )

    assert "does not exist" in tinyruns("new", "configs/absent.yaml").error
    assert "output=/tmp/results" in tinyruns("new", "configs/located.yaml").error
    assert "absent.py is missing" in tinyruns("new", "configs/no_stage.yaml").error
    assert tinyruns("ls").output == []
