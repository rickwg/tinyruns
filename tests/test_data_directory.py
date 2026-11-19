import json
from pathlib import Path

import pytest

from conftest import (
    MARKER_CONFIG,
    Command,
    FakeCluster,
    metrics,
    run_locally,
    write_config,
)

DATASET_FILE = Path("answers") / "train.txt"
DATASET_CONFIG = {
    **MARKER_CONFIG,
    "marker": {"text": "dataset report", "read_dataset": str(DATASET_FILE)},
}


@pytest.fixture
def dataset_config(repository: Path) -> str:
    write_config(repository, "dataset", DATASET_CONFIG)
    return "configs/dataset.yaml"


def write_local_dataset(repository: Path, content: str) -> None:
    dataset_file = repository / "artifacts" / "data" / DATASET_FILE
    dataset_file.parent.mkdir(parents=True, exist_ok=True)
    dataset_file.write_text(content)


def finished_smoke_run(tinyruns: Command, config: str) -> str:
    submission = tinyruns("submit", config, "--smoke")
    assert submission.exit_code == 0, submission.error
    run_id = submission.output["run_id"]
    assert tinyruns("wait", run_id, "--interval", "0").output["state"] == "done"
    return run_id


def test_local_run_reads_a_dataset_beside_the_runs(
    tinyruns: Command, repository: Path, cluster: FakeCluster, dataset_config: str
) -> None:
    write_local_dataset(repository, "local rows")

    created = tinyruns("new", dataset_config).output
    run_locally(repository, created["run_with"])

    recorded = metrics(repository / created["directory"])[-1]
    assert recorded["dataset_content"] == "local rows"
    assert created["code"]["dirty"] is False
    assert cluster.job_ids() == []


def test_cluster_run_reads_a_pushed_dataset(
    tinyruns: Command, repository: Path, cluster: FakeCluster, dataset_config: str
) -> None:
    write_local_dataset(repository, "pushed rows")
    cluster_data = cluster.scratch_root / "data"

    pushed = tinyruns("push-data")
    run_id = finished_smoke_run(tinyruns, dataset_config)

    assert pushed.output == {
        "pushed": "artifacts/data",
        "to": str(cluster_data),
        "cluster": "fake-cluster",
        "datasets": ["answers"],
    }
    recorded = metrics(cluster.run_directory(run_id))[-1]
    assert recorded["dataset_content"] == "pushed rows"
    job_script = (cluster.run_directory(run_id) / "job.sbatch").read_text()
    assert f"--bind {cluster_data}:/mnt/data:ro " in job_script


def test_push_data_replaces_files_and_keeps_what_is_only_on_the_cluster(
    tinyruns: Command, repository: Path, cluster: FakeCluster
) -> None:
    only_on_the_cluster = cluster.scratch_root / "data" / "large" / "rows.bin"
    only_on_the_cluster.parent.mkdir(parents=True)
    only_on_the_cluster.write_text("large rows")
    write_local_dataset(repository, "first rows")
    tinyruns("push-data")
    write_local_dataset(repository, "second rows, longer")

    pushed_again = tinyruns("push-data")

    assert pushed_again.exit_code == 0, pushed_again.error
    assert (cluster.scratch_root / "data" / DATASET_FILE).read_text() == (
        "second rows, longer"
    )
    assert only_on_the_cluster.read_text() == "large rows"


def test_push_data_needs_a_data_directory(
    tinyruns: Command, cluster: FakeCluster
) -> None:
    refused = tinyruns("push-data")

    assert refused.exit_code == 1
    assert "artifacts/data does not exist" in json.loads(refused.error)["error"]
    assert not cluster.scratch_root.exists()


def test_environment_check_reports_the_datasets_a_job_sees(
    tinyruns: Command, repository: Path, cluster: FakeCluster
) -> None:
    write_config(
        repository,
        "environment",
        {
            "stages": ["tinyruns.stages.environment_check"],
            "environment_check": {"urls": []},
        },
    )
    write_local_dataset(repository, "rows")
    tinyruns("push-data")

    run_id = finished_smoke_run(tinyruns, "configs/environment.yaml")

    report = (cluster.run_directory(run_id) / "report.md").read_text()
    assert "holds ['answers'] |" in report
