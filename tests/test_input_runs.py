import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import (
    MARKER_CONFIG,
    RUN_COMMAND_PREFIX,
    RUNS_DIRECTORY,
    Command,
    FakeCluster,
    local_run,
    metrics,
    run_locally,
    write_config,
)
from tinyruns.run_config import RunConfigError, validate_run_config
from tinyruns.run_directory import RunDirectory
from tinyruns.run_inputs import input_run

ABSENT_RUN = "20260101-000000-absent-abc1234"
EVALUATION_CONFIG = {
    **MARKER_CONFIG,
    "marker": {"text": "evaluation report", "read_input": "training"},
}


@pytest.fixture
def evaluation_config(repository: Path) -> str:
    write_config(repository, "evaluation", EVALUATION_CONFIG)
    return "configs/evaluation.yaml"


def finished_cluster_run(tinyruns: Command, *submit_arguments: str) -> str:
    submission = tinyruns("submit", *submit_arguments)
    assert submission.exit_code == 0, submission.error
    run_id = submission.output["run_id"]
    assert tinyruns("wait", run_id, "--interval", "0").output["state"] == "done"
    return run_id


def refusal(tinyruns: Command, *arguments: str) -> str:
    refused = tinyruns(*arguments)
    assert refused.exit_code == 1
    return json.loads(refused.error)["error"]


def test_several_runs_read_one_finished_run_on_the_cluster(
    tinyruns: Command, repository: Path, cluster: FakeCluster, evaluation_config: str
) -> None:
    training = finished_cluster_run(tinyruns, "configs/marker.yaml", "--smoke")
    write_config(
        repository, "pinned", {**EVALUATION_CONFIG, "inputs": {"training": training}}
    )

    named_on_command_line = finished_cluster_run(
        tinyruns, evaluation_config, "--smoke", "--input", f"training={training}"
    )
    named_in_config = finished_cluster_run(tinyruns, "configs/pinned.yaml", "--smoke")

    for evaluation in (named_on_command_line, named_in_config):
        recorded = metrics(cluster.run_directory(evaluation))[-1]
        assert recorded["input_report"] == "marker report"
        assert recorded["input_weights"] == "heavy"
        frozen_config = yaml.safe_load(
            (local_run(repository, evaluation) / "config.yaml").read_text()
        )
        assert frozen_config["inputs"] == {"training": training}
        job_script = (local_run(repository, evaluation) / "job.sbatch").read_text()
        assert f"/runs/{training}:/mnt/runs/{training}:ro" in job_script


def test_pulled_run_reads_its_pulled_input_run_locally(
    tinyruns: Command, repository: Path, evaluation_config: str
) -> None:
    training = finished_cluster_run(tinyruns, "configs/marker.yaml", "--smoke")
    evaluation = finished_cluster_run(
        tinyruns, evaluation_config, "--smoke", "--input", f"training={training}"
    )
    tinyruns("pull", evaluation)
    tinyruns("pull", training, "--heavy")
    evaluation_directory = local_run(repository, evaluation)
    (evaluation_directory / "metrics.jsonl").unlink()

    run_locally(repository, f"{RUN_COMMAND_PREFIX}{RUNS_DIRECTORY / evaluation}")

    assert metrics(evaluation_directory)[-1]["input_weights"] == "heavy"


def test_input_run_must_be_done_on_the_cluster(
    tinyruns: Command, cluster: FakeCluster, evaluation_config: str
) -> None:
    submit_evaluation = ("submit", evaluation_config, "--smoke", "--input")
    cluster.hold_new_jobs()
    queued_training = tinyruns("submit", "configs/marker.yaml", "--smoke").output[
        "run_id"
    ]

    absent = refusal(tinyruns, *submit_evaluation, f"training={ABSENT_RUN}")
    unfinished = refusal(tinyruns, *submit_evaluation, f"training={queued_training}")

    assert "not on the cluster" in absent
    assert "is queued" in unfinished
    assert [run["run_id"] for run in tinyruns("ls").output] == [queued_training]


def test_full_run_reads_only_full_runs_and_needs_its_own_smoke_run(
    tinyruns: Command, evaluation_config: str
) -> None:
    smoke_training = finished_cluster_run(tinyruns, "configs/marker.yaml", "--smoke")
    full_training = finished_cluster_run(tinyruns, "configs/marker.yaml")
    reads_smoke = ("--input", f"training={smoke_training}")
    reads_full = ("--input", f"training={full_training}")
    finished_cluster_run(tinyruns, evaluation_config, "--smoke", *reads_smoke)

    smoke_input_refused = refusal(tinyruns, "submit", evaluation_config, *reads_smoke)
    other_input_refused = refusal(tinyruns, "submit", evaluation_config, *reads_full)
    finished_cluster_run(tinyruns, evaluation_config, "--smoke", *reads_full)
    finished_cluster_run(tinyruns, evaluation_config, *reads_full)

    assert "a smoke run" in smoke_input_refused
    assert "No smoke run" in other_input_refused


def test_local_run_reads_a_finished_local_run(
    tinyruns: Command, repository: Path, cluster: FakeCluster, evaluation_config: str
) -> None:
    training = tinyruns("new", "configs/marker.yaml").output
    new_evaluation = ("new", evaluation_config, "--input")

    before_training_ran = refusal(
        tinyruns, *new_evaluation, f"training={training['run_id']}"
    )
    run_locally(repository, training["run_with"])
    evaluation = tinyruns(*new_evaluation, f"training={training['run_id']}").output
    run_locally(repository, evaluation["run_with"])

    assert "is created here" in before_training_ran
    recorded = metrics(repository / evaluation["directory"])[-1]
    assert recorded["input_report"] == "marker report"
    absent = refusal(tinyruns, *new_evaluation, f"training={ABSENT_RUN}")
    assert f"uv run tinyruns pull {ABSENT_RUN} --heavy" in absent
    assert cluster.job_ids() == []


def test_environment_check_reports_its_input_runs(
    tinyruns: Command, repository: Path, cluster: FakeCluster
) -> None:
    write_config(
        repository,
        "environment",
        {
            "stages": ["tinyruns.stages.environment_check"],
            "resources": "cpu-small",
            "environment_check": {"urls": []},
        },
    )
    previous = finished_cluster_run(tinyruns, "configs/marker.yaml", "--smoke")

    check = finished_cluster_run(
        tinyruns,
        "configs/environment.yaml",
        "--smoke",
        "--input",
        f"previous={previous}",
    )

    report = (cluster.run_directory(check) / "report.md").read_text()
    assert f"| previous | {previous}: done, " in report


@pytest.mark.parametrize(
    "inputs",
    [
        {"training": "../elsewhere"},
        {"training": "runs/20261119-train"},
        {"not a name": "20261119-train"},
        {"training": 20261119},
        ["20261119-train"],
        None,
    ],
)
def test_inputs_must_map_names_to_run_ids(inputs: Any) -> None:
    with pytest.raises(RunConfigError, match="inputs"):
        validate_run_config({"stages": ["a"], "inputs": inputs}, source="config.yaml")


@pytest.mark.parametrize(
    ("assignment", "explanation"),
    [("training", "NAME=RUN_ID"), ("training=../elsewhere", "not a path")],
)
def test_input_option_must_name_a_run_id(
    tinyruns: Command, evaluation_config: str, assignment: str, explanation: str
) -> None:
    assert explanation in refusal(
        tinyruns, "new", evaluation_config, "--input", assignment
    )
    assert tinyruns("ls").output == []


def test_stage_asking_for_an_undeclared_input_is_told_how_to_declare_it(
    tmp_path: Path,
) -> None:
    with pytest.raises(RunConfigError, match="--input training=RUN_ID"):
        input_run(RunDirectory(tmp_path / "run"), {"stages": ["a"]}, "training")
