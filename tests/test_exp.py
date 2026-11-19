import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from conftest import (
    MARKER_CONFIG,
    RUNS_DIRECTORY,
    STAGES_DIRECTORY,
    Exp,
    FakeCluster,
    commit_all,
    local_run,
    metrics,
    write_config,
)


def submitted_smoke_run(exp: Exp, config: str = "configs/marker.yaml") -> str:
    submission = exp("submit", config, "--smoke")
    assert submission.exit_code == 0, submission.error
    return submission.output["run_id"]


def test_smoke_run_of_uncommitted_code_runs_and_is_pulled(
    exp: Exp, repository: Path, cluster: FakeCluster
) -> None:
    uncommitted_stage = repository / STAGES_DIRECTORY / "uncommitted.py"
    uncommitted_stage.write_text(
        (repository / STAGES_DIRECTORY / "marker.py").read_text()
    )
    write_config(
        repository, "uncommitted", {**MARKER_CONFIG, "stages": ["uncommitted"]}
    )

    submission = exp("submit", "configs/uncommitted.yaml", "--smoke")

    assert submission.exit_code == 0, submission.error
    manifest = submission.output
    run_id = manifest["run_id"]
    assert run_id.endswith("-smoke")
    assert manifest["code"]["dirty"] is True
    assert exp("wait", run_id, "--interval", "0").output["state"] == "done"

    pulled = exp("pull", run_id)

    run_directory = local_run(repository, run_id)
    assert pulled.output["directory"] == str(RUNS_DIRECTORY / run_id)
    assert (run_directory / "report.md").read_text() == "marker report"
    assert "marker stage runs" in (run_directory / "slurm.out").read_text()
    assert "uncommitted.py" in (run_directory / "code.patch").read_text()
    assert not (run_directory / "heavy").exists()
    assert (cluster.run_directory(run_id) / "heavy" / "weights.bin").is_file()


def test_job_runs_the_snapshot_with_the_cluster_environment(
    exp: Exp, repository: Path, cluster: FakeCluster
) -> None:
    run_id = submitted_smoke_run(exp)
    exp("pull", run_id)

    manifest = json.loads((local_run(repository, run_id) / "manifest.json").read_text())
    snapshot_directory = cluster.code_root / manifest["code"]["tree"]
    recorded = metrics(local_run(repository, run_id))[0]
    assert recorded["PYTHONPATH"] == str(snapshot_directory / "src")
    assert recorded["HF_HOME"] == str(cluster.scratch_root / "cache" / "hf")
    assert recorded["FAKE_PROXY"] == "http://proxy.invalid:3128"
    assert recorded["HF_HUB_OFFLINE"] is None
    assert recorded["smoke_test"] is True
    assert recorded["SLURM_JOB_ID"] == manifest["slurm"]["job_ids"][0]
    assert not (snapshot_directory / "configs").exists()


def test_pulled_run_reruns_locally_without_edits(exp: Exp, repository: Path) -> None:
    run_id = submitted_smoke_run(exp)
    exp("pull", run_id, "--heavy")
    run_directory = local_run(repository, run_id)
    assert (run_directory / "heavy" / "weights.bin").read_bytes() == b"heavy"
    (run_directory / "report.md").unlink()

    subprocess.run(
        [sys.executable, "-m", "tinyruns.run", str(run_directory)],
        env={**os.environ, "PYTHONPATH": str(repository / "src")},
        check=True,
    )

    assert (run_directory / "report.md").read_text() == "marker report"
    assert len(metrics(run_directory)) == 2


def test_full_run_needs_committed_code(exp: Exp, repository: Path) -> None:
    marker_stage = repository / STAGES_DIRECTORY / "marker.py"
    marker_stage.write_text(marker_stage.read_text() + "\nEDITED = True\n")

    submission = exp("submit", "configs/marker.yaml")

    assert submission.exit_code == 1
    assert "Commit" in json.loads(submission.error)["error"]
    assert not (repository / RUNS_DIRECTORY).exists()


def test_full_run_needs_a_passed_smoke_run_of_the_same_code_and_config(
    exp: Exp, repository: Path, cluster: FakeCluster
) -> None:
    refused = exp("submit", "configs/marker.yaml")
    assert refused.exit_code == 1
    assert "--smoke" in json.loads(refused.error)["error"]

    smoke_run_id = submitted_smoke_run(exp)
    (repository / "notes.md").write_text("Files outside the snapshot do not count.")
    commit_all(repository, "Notes")
    full = exp("submit", "configs/marker.yaml")

    assert full.exit_code == 0, full.error
    full_run_id = full.output["run_id"]
    assert full.output["code"]["dirty"] is False
    smoke_script = (cluster.run_directory(smoke_run_id) / "job.sbatch").read_text()
    full_script = (cluster.run_directory(full_run_id) / "job.sbatch").read_text()
    assert "HF_HUB_OFFLINE=1" not in smoke_script
    assert "HF_HUB_OFFLINE=1" in full_script
    full_config = yaml.safe_load(
        (local_run(repository, full_run_id) / "config.yaml").read_text()
    )
    assert full_config["smoke_test"]["enabled"] is False


def test_smoke_run_of_other_code_does_not_unlock_a_full_run(
    exp: Exp, repository: Path
) -> None:
    submitted_smoke_run(exp)
    marker_stage = repository / STAGES_DIRECTORY / "marker.py"
    marker_stage.write_text(marker_stage.read_text() + "\nEDITED = True\n")
    commit_all(repository, "Change the stage")

    assert exp("submit", "configs/marker.yaml").exit_code == 1
    assert exp("submit", "configs/marker.yaml", "--skip-smoke-check").exit_code == 0


def test_image_is_built_once_and_the_first_run_waits_for_it(
    exp: Exp, cluster: FakeCluster
) -> None:
    first = exp("submit", "configs/marker.yaml", "--smoke").output
    second = exp("submit", "configs/marker.yaml", "--smoke").output

    assert first["slurm"]["image_build_job_id"] is not None
    assert second["slurm"]["image_build_job_id"] is None
    assert first["image"] == second["image"]
    assert Path(first["image"]).read_text() == "fake image"
    assert list((cluster.root / "local").iterdir()) == []


def test_failed_image_build_fails_the_run(
    exp: Exp, monkeypatch: Any, cluster: FakeCluster
) -> None:
    monkeypatch.setenv("FAKE_CLUSTER_FAIL_BUILD", "1")
    run_id = exp("submit", "configs/marker.yaml", "--smoke").output["run_id"]

    status = exp("status", run_id).output

    assert status["state"] == "failed"
    assert status["slurm_state"] == "CANCELLED"


def test_gpu_hours_above_the_limit_need_confirmation(
    exp: Exp, repository: Path
) -> None:
    write_config(repository, "long", {**MARKER_CONFIG, "resources": "gpu-long"})

    refused = exp("submit", "configs/long.yaml", "--smoke")
    confirmed = exp("submit", "configs/long.yaml", "--smoke", "--confirm")

    assert refused.exit_code == 1
    assert "48 GPU hours" in json.loads(refused.error)["error"]
    assert confirmed.exit_code == 0, confirmed.error


def test_smoke_run_uses_the_smoke_resources(
    exp: Exp, repository: Path, cluster: FakeCluster
) -> None:
    write_config(
        repository,
        "gpu",
        {
            **MARKER_CONFIG,
            "resources": "gpu-long",
            "smoke_test": {"resources": "gpu-short"},
        },
    )

    run_id = submitted_smoke_run(exp, "configs/gpu.yaml")

    job_script = (cluster.run_directory(run_id) / "job.sbatch").read_text()
    assert "#SBATCH --time=01:00:00" in job_script
    assert "#SBATCH --gpus=1" in job_script
    assert "#SBATCH --constraint=GPU-80GB" in job_script
    assert "--nv" in job_script


def test_queued_run_waits_then_finishes(exp: Exp, cluster: FakeCluster) -> None:
    cluster.hold_new_jobs()
    submission = exp("submit", "configs/marker.yaml", "--smoke").output
    run_id = submission["run_id"]

    assert exp("status", run_id).output["state"] == "queued"
    still_waiting = exp("wait", run_id, "--interval", "0", "--timeout", "0")
    assert still_waiting.exit_code == 2

    cluster.release(submission["slurm"]["image_build_job_id"])
    cluster.release(submission["slurm"]["job_ids"][0])
    assert exp("wait", run_id, "--interval", "0").exit_code == 0
    assert exp("status").output == []


def test_cancelled_run_is_reported_failed(exp: Exp, cluster: FakeCluster) -> None:
    cluster.hold_new_jobs()
    run_id = exp("submit", "configs/marker.yaml", "--smoke").output["run_id"]

    cancelled = exp("cancel", run_id).output

    assert cancelled["state"] == "failed"
    assert cancelled["slurm_state"] == "CANCELLED"
    assert exp("ls").output[0]["state"] == "failed"


def test_job_that_slurm_forgot_is_reported_failed(
    exp: Exp, cluster: FakeCluster
) -> None:
    cluster.hold_new_jobs()
    submission = exp("submit", "configs/marker.yaml", "--smoke").output
    job_id = submission["slurm"]["job_ids"][0]
    cluster.forget(job_id)

    status = exp("status", submission["run_id"]).output

    assert status["state"] == "failed"
    assert status["failure"] == f"Slurm no longer knows job {job_id}"


def test_failed_run_shows_its_log_and_resumes_in_place(
    exp: Exp, repository: Path, cluster: FakeCluster
) -> None:
    write_config(
        repository,
        "flaky",
        {**MARKER_CONFIG, "marker": {"text": "second try", "fail_first_attempt": True}},
    )
    run_id = exp("submit", "configs/flaky.yaml", "--smoke").output["run_id"]

    failed = exp("wait", run_id, "--interval", "0")

    assert failed.exit_code == 1
    assert failed.output["failure"] == "RuntimeError: the first attempt fails"
    assert "Traceback" in failed.output["log_tail"]
    assert "Traceback" in exp("logs", run_id, "--lines", "40").output["log"]

    resumed = exp("resume", run_id)

    assert len(resumed.output["slurm"]["job_ids"]) == 2
    assert exp("wait", run_id, "--interval", "0").exit_code == 0
    cluster_directory = cluster.run_directory(run_id)
    assert (cluster_directory / "report.md").read_text() == "second try"
    assert (cluster_directory / "slurm.out").read_text().count("marker stage runs") == 2
    assert exp("resume", run_id).exit_code == 1


def test_config_naming_a_location_is_rejected(exp: Exp, repository: Path) -> None:
    write_config(
        repository, "located", {**MARKER_CONFIG, "data": {"directory": "/mnt/data"}}
    )

    submission = exp("submit", "configs/located.yaml", "--smoke")

    assert submission.exit_code == 1
    assert "data.directory=/mnt/data" in json.loads(submission.error)["error"]


def test_unknown_stage_resources_and_run_are_rejected(
    exp: Exp, repository: Path
) -> None:
    write_config(repository, "no_stage", {**MARKER_CONFIG, "stages": ["missing"]})
    write_config(repository, "no_resources", {**MARKER_CONFIG, "resources": "huge"})

    assert "missing" in exp("submit", "configs/no_stage.yaml", "--smoke").error
    assert "cpu-small" in exp("submit", "configs/no_resources.yaml", "--smoke").error
    assert "exp ls" in exp("status", "20260101-000000-nothing").error
    assert exp("ls").output == []


def test_environment_check_runs_as_a_cluster_job(
    exp: Exp, repository: Path, cluster: FakeCluster
) -> None:
    write_config(
        repository,
        "environment",
        {
            "stages": ["environment_check"],
            "resources": "cpu-small",
            "environment_check": {"urls": []},
        },
    )

    run_id = submitted_smoke_run(exp, "configs/environment.yaml")

    assert exp("wait", run_id, "--interval", "0").output["state"] == "done"
    report = (cluster.run_directory(run_id) / "report.md").read_text()
    assert "Result: passed" in report
