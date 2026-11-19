import json
from pathlib import Path

import pytest
import yaml

from conftest import MARKER_CONFIG, Exp, FakeCluster, run_locally, write_config


def test_new_run_is_created_and_run_without_the_cluster(
    exp: Exp, repository: Path, cluster: FakeCluster
) -> None:
    created = exp("new", "configs/marker.yaml", "--smoke")

    assert created.exit_code == 0, created.error
    run_id = created.output["run_id"]
    run_directory = repository / created.output["directory"]
    assert run_id.endswith("-smoke")
    assert created.output["directory"] == f"artifacts/runs/{run_id}"
    assert created.output["cluster"] is None
    frozen_config = yaml.safe_load((run_directory / "config.yaml").read_text())
    assert frozen_config["smoke_test"]["enabled"] is True
    assert exp("status", run_id).output["state"] == "created"
    assert cluster.job_ids() == []
    assert not cluster.scratch_root.exists()

    run_locally(repository, created.output["run_with"])

    assert (run_directory / "report.md").read_text() == "marker report"
    listed = exp("ls").output
    assert [(run["state"], run["location"]) for run in listed] == [("done", "local")]


def test_new_run_records_uncommitted_code(exp: Exp, repository: Path) -> None:
    (repository / "src" / "tinyruns" / "draft.py").write_text("DRAFT = 1\n")

    created = exp("new", "configs/marker.yaml").output

    assert created["code"]["dirty"] is True
    assert not created["run_id"].endswith("-smoke")
    code_patch = repository / created["directory"] / "code.patch"
    assert "DRAFT = 1" in code_patch.read_text()


def test_local_smoke_run_does_not_unlock_a_full_cluster_run(
    exp: Exp, repository: Path
) -> None:
    created = exp("new", "configs/marker.yaml", "--smoke").output
    run_locally(repository, created["run_with"])

    refused = exp("submit", "configs/marker.yaml")
    cluster_smoke = exp("submit", "configs/marker.yaml", "--smoke")
    allowed = exp("submit", "configs/marker.yaml")

    assert refused.exit_code == 1
    assert "on the cluster" in json.loads(refused.error)["error"]
    assert allowed.exit_code == 0, allowed.error
    locations = {run["run_id"]: run["location"] for run in exp("ls").output}
    assert locations[created["run_id"]] == "local"
    assert locations[cluster_smoke.output["run_id"]] == "cluster"


@pytest.mark.parametrize("command", ["wait", "logs", "pull", "cancel", "resume"])
def test_cluster_commands_explain_how_to_run_a_local_run(
    exp: Exp, command: str
) -> None:
    created = exp("new", "configs/marker.yaml").output

    rejected = exp(command, created["run_id"])

    assert rejected.exit_code == 1
    assert created["run_with"] in json.loads(rejected.error)["error"]
    assert exp("status").output == []


def test_new_rejects_configs_that_cannot_run(exp: Exp, repository: Path) -> None:
    write_config(repository, "located", {**MARKER_CONFIG, "output": "/tmp/results"})
    write_config(repository, "no_stage", {**MARKER_CONFIG, "stages": ["missing"]})

    assert "does not exist" in exp("new", "configs/absent.yaml").error
    assert "output=/tmp/results" in exp("new", "configs/located.yaml").error
    assert "missing" in exp("new", "configs/no_stage.yaml").error
    assert exp("ls").output == []
