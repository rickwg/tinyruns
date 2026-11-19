import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import (
    MARKER_CONFIG,
    ON_CPU,
    RUNS_DIRECTORY,
    STAGES_DIRECTORY,
    STAGES_PACKAGE,
    Command,
    FakeCluster,
    commit_all,
    local_run,
    metrics,
    write_config,
)
from tinyruns.cluster import operations


def submitted_smoke_run(tinyruns: Command, config: str = "configs/marker.yaml") -> str:
    submission = tinyruns("submit", config, "--smoke")
    assert submission.exit_code == 0, submission.error
    return submission.output["run_id"]


def test_smoke_run_of_uncommitted_code_runs_and_is_pulled(
    tinyruns: Command, repository: Path, cluster: FakeCluster
) -> None:
    uncommitted_stage = repository / STAGES_DIRECTORY / "uncommitted.py"
    uncommitted_stage.write_text(
        (repository / STAGES_DIRECTORY / "marker.py").read_text()
    )
    write_config(
        repository,
        "uncommitted",
        {**MARKER_CONFIG, "stages": [f"{STAGES_PACKAGE}.uncommitted"]},
    )

    submission = tinyruns("submit", "configs/uncommitted.yaml", "--smoke")

    assert submission.exit_code == 0, submission.error
    manifest = submission.output
    run_id = manifest["run_id"]
    assert run_id.endswith("-smoke")
    assert manifest["code"]["dirty"] is True
    assert tinyruns("wait", run_id, "--interval", "0").output["state"] == "done"

    pulled = tinyruns("pull", run_id)

    run_directory = local_run(repository, run_id)
    assert pulled.output["directory"] == str(RUNS_DIRECTORY / run_id)
    assert (run_directory / "report.md").read_text() == "marker report"
    assert "marker stage runs" in (run_directory / "slurm.out").read_text()
    assert "uncommitted.py" in (run_directory / "code.patch").read_text()
    assert not (run_directory / "heavy").exists()
    assert (cluster.run_directory(run_id) / "heavy" / "weights.bin").is_file()


def test_job_runs_the_snapshot_with_the_cluster_environment(
    tinyruns: Command, repository: Path, cluster: FakeCluster
) -> None:
    run_id = submitted_smoke_run(tinyruns)
    tinyruns("pull", run_id)

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


def test_manifest_names_the_cluster_and_the_resources(tinyruns: Command) -> None:
    manifest = tinyruns("submit", "configs/marker.yaml", "--smoke").output

    assert manifest["cluster"]["name"] == "fake-cluster"
    assert manifest["resources"]["name"] == "cpu-small"
    assert manifest["tinyruns_version"]


def test_pulled_run_reruns_locally_without_edits(
    tinyruns: Command, repository: Path
) -> None:
    run_id = submitted_smoke_run(tinyruns)
    tinyruns("pull", run_id, "--heavy")
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


def test_full_run_needs_committed_code(tinyruns: Command, repository: Path) -> None:
    marker_stage = repository / STAGES_DIRECTORY / "marker.py"
    marker_stage.write_text(marker_stage.read_text() + "\nEDITED = True\n")

    submission = tinyruns("submit", "configs/marker.yaml", *ON_CPU)

    assert submission.exit_code == 1
    assert "Commit" in json.loads(submission.error)["error"]
    assert not (repository / RUNS_DIRECTORY).exists()


def test_full_run_needs_a_passed_smoke_run_of_the_same_code_and_config(
    tinyruns: Command, repository: Path, cluster: FakeCluster
) -> None:
    refused = tinyruns("submit", "configs/marker.yaml", *ON_CPU)
    assert refused.exit_code == 1
    assert "--smoke" in json.loads(refused.error)["error"]

    smoke_run_id = submitted_smoke_run(tinyruns)
    (repository / "notes.md").write_text("Files outside the snapshot do not count.")
    commit_all(repository, "Notes")
    full = tinyruns("submit", "configs/marker.yaml", *ON_CPU)

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
    tinyruns: Command, repository: Path
) -> None:
    submitted_smoke_run(tinyruns)
    marker_stage = repository / STAGES_DIRECTORY / "marker.py"
    marker_stage.write_text(marker_stage.read_text() + "\nEDITED = True\n")
    commit_all(repository, "Change the stage")

    unchecked = ("configs/marker.yaml", *ON_CPU, "--skip-smoke-check")

    assert tinyruns("submit", "configs/marker.yaml", *ON_CPU).exit_code == 1
    assert tinyruns("submit", *unchecked).exit_code == 0


def test_lock_that_takes_a_package_from_a_local_path_is_refused(
    tinyruns: Command, repository: Path
) -> None:
    lock_file = repository / "uv.lock"
    lock_file.write_text(
        lock_file.read_text()
        + '\n[[package]]\nname = "neighbour"\nversion = "0.1.0"\n'
        + 'source = { editable = "../neighbour" }\n'
    )

    refused = tinyruns("submit", "configs/marker.yaml", "--smoke")

    assert refused.exit_code == 1
    assert "['neighbour'] from a path" in json.loads(refused.error)["error"]
    assert tinyruns("ls").output == []


def test_image_is_built_once_and_the_first_run_waits_for_it(
    tinyruns: Command, cluster: FakeCluster
) -> None:
    first = tinyruns("submit", "configs/marker.yaml", "--smoke").output
    second = tinyruns("submit", "configs/marker.yaml", "--smoke").output

    assert first["slurm"]["image_build_job_id"] is not None
    assert second["slurm"]["image_build_job_id"] is None
    assert first["image"] == second["image"]
    assert Path(first["image"]).read_text() == "fake image"
    assert list((cluster.root / "local").iterdir()) == []


def test_failed_image_build_fails_the_run(
    tinyruns: Command, monkeypatch: Any, cluster: FakeCluster
) -> None:
    monkeypatch.setenv("FAKE_CLUSTER_FAIL_BUILD", "1")
    run_id = tinyruns("submit", "configs/marker.yaml", "--smoke").output["run_id"]

    status = tinyruns("status", run_id).output

    assert status["state"] == "failed"
    assert status["slurm_state"] == "CANCELLED"


def test_gpu_hours_above_the_limit_need_confirmation(tinyruns: Command) -> None:
    long_smoke_run = ("configs/marker.yaml", "--smoke", "--resources", "gpu-long")

    refused = tinyruns("submit", *long_smoke_run)
    confirmed = tinyruns("submit", *long_smoke_run, "--confirm")

    assert refused.exit_code == 1
    assert "48 GPU hours" in json.loads(refused.error)["error"]
    assert confirmed.exit_code == 0, confirmed.error


def test_smoke_run_uses_the_smoke_resources_of_the_profile(
    tinyruns: Command, cluster: FakeCluster
) -> None:
    submission = tinyruns("submit", "configs/marker.yaml", "--smoke").output

    job_script = cluster.run_directory(submission["run_id"]) / "job.sbatch"
    assert submission["resources"]["name"] == "cpu-small"
    assert "#SBATCH --time=00:10:00" in job_script.read_text()
    assert "--nv" not in job_script.read_text()


def test_run_uses_the_resources_named_on_the_command_line(
    tinyruns: Command, cluster: FakeCluster
) -> None:
    submission = tinyruns(
        "submit", "configs/marker.yaml", "--smoke", "--resources", "gpu-short"
    ).output

    job_script = (
        cluster.run_directory(submission["run_id"]) / "job.sbatch"
    ).read_text()
    assert submission["resources"]["name"] == "gpu-short"
    assert "#SBATCH --time=01:00:00" in job_script
    assert "#SBATCH --gpus=1" in job_script
    assert "#SBATCH --constraint=GPU-80GB" in job_script
    assert "--nv" in job_script


def test_run_without_resources_is_refused_unless_the_profile_covers_smoke_runs(
    tinyruns: Command, cluster: FakeCluster
) -> None:
    full = tinyruns("submit", "configs/marker.yaml")
    profile = yaml.safe_load(cluster.profile_path.read_text())
    cluster.profile_path.write_text(
        yaml.safe_dump({**profile, "smoke_resources": None})
    )
    smoke = tinyruns("submit", "configs/marker.yaml", "--smoke")

    assert "A full run needs --resources" in json.loads(full.error)["error"]
    assert "`smoke_resources`" in json.loads(smoke.error)["error"]
    assert tinyruns("ls").output == []


def test_queued_run_waits_then_finishes(
    tinyruns: Command, cluster: FakeCluster
) -> None:
    cluster.hold_new_jobs()
    submission = tinyruns("submit", "configs/marker.yaml", "--smoke").output
    run_id = submission["run_id"]

    assert tinyruns("status", run_id).output["state"] == "queued"
    still_waiting = tinyruns("wait", run_id, "--interval", "0", "--timeout", "0")
    assert still_waiting.exit_code == 2

    cluster.release(submission["slurm"]["image_build_job_id"])
    cluster.release(submission["slurm"]["job_ids"][0])
    assert tinyruns("wait", run_id, "--interval", "0").exit_code == 0
    assert tinyruns("status").output == []


def test_cancelled_run_is_reported_failed(
    tinyruns: Command, cluster: FakeCluster
) -> None:
    cluster.hold_new_jobs()
    run_id = tinyruns("submit", "configs/marker.yaml", "--smoke").output["run_id"]

    cancelled = tinyruns("cancel", run_id).output

    assert cancelled["state"] == "failed"
    assert cancelled["slurm_state"] == "CANCELLED"
    assert tinyruns("ls").output[0]["state"] == "failed"


def test_cancel_returns_once_slurm_has_ended_the_job(
    tinyruns: Command, cluster: FakeCluster, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(operations, "CANCEL_POLL_SECONDS", 0)
    cluster.hold_new_jobs()
    cluster.keep_cancelled_jobs_completing(polls=2)
    run_id = tinyruns("submit", "configs/marker.yaml", "--smoke").output["run_id"]

    cancelled = tinyruns("cancel", run_id).output

    assert cancelled["state"] == "failed"
    assert cancelled["slurm_state"] == "CANCELLED"
    assert "as CANCELLED" in cancelled["failure"]


def test_cancel_says_so_when_slurm_does_not_end_the_job_in_time(
    tinyruns: Command, cluster: FakeCluster, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(operations, "CANCEL_POLL_SECONDS", 0)
    monkeypatch.setattr(operations, "CANCEL_TIMEOUT_SECONDS", 0)
    cluster.hold_new_jobs()
    cluster.keep_cancelled_jobs_completing(polls=100)
    submission = tinyruns("submit", "configs/marker.yaml", "--smoke").output

    cancelled = tinyruns("cancel", submission["run_id"])

    assert cancelled.exit_code == 1
    job_id = submission["slurm"]["job_ids"][0]
    assert "has not ended it" in json.loads(cancelled.error)["error"]
    assert f"cancel job {job_id}" in json.loads(cancelled.error)["error"]
    still_ending = tinyruns("status", submission["run_id"]).output
    assert (still_ending["state"], still_ending["slurm_state"]) == (
        "running",
        "COMPLETING",
    )


def test_job_that_slurm_forgot_is_reported_failed(
    tinyruns: Command, cluster: FakeCluster
) -> None:
    cluster.hold_new_jobs()
    submission = tinyruns("submit", "configs/marker.yaml", "--smoke").output
    job_id = submission["slurm"]["job_ids"][0]
    cluster.forget(job_id)

    status = tinyruns("status", submission["run_id"]).output

    assert status["state"] == "failed"
    assert status["failure"] == f"Slurm no longer knows job {job_id}"


def test_failed_run_shows_its_log_and_resumes_in_place(
    tinyruns: Command, repository: Path, cluster: FakeCluster
) -> None:
    write_config(
        repository,
        "flaky",
        {**MARKER_CONFIG, "marker": {"text": "second try", "fail_first_attempt": True}},
    )
    run_id = tinyruns("submit", "configs/flaky.yaml", "--smoke").output["run_id"]

    failed = tinyruns("wait", run_id, "--interval", "0")

    assert failed.exit_code == 1
    assert failed.output["failure"] == "RuntimeError: the first attempt fails"
    assert "Traceback" in failed.output["log_tail"]
    assert "Traceback" in tinyruns("logs", run_id, "--lines", "40").output["log"]

    resumed = tinyruns("resume", run_id)

    assert len(resumed.output["slurm"]["job_ids"]) == 2
    assert tinyruns("wait", run_id, "--interval", "0").exit_code == 0
    cluster_directory = cluster.run_directory(run_id)
    assert (cluster_directory / "report.md").read_text() == "second try"
    assert (cluster_directory / "slurm.out").read_text().count("marker stage runs") == 2
    assert tinyruns("resume", run_id).exit_code == 1


def test_config_naming_a_location_is_rejected(
    tinyruns: Command, repository: Path
) -> None:
    write_config(
        repository, "located", {**MARKER_CONFIG, "data": {"directory": "/mnt/data"}}
    )

    submission = tinyruns("submit", "configs/located.yaml", "--smoke")

    assert submission.exit_code == 1
    assert "data.directory=/mnt/data" in json.loads(submission.error)["error"]


def test_unknown_stage_resources_and_run_are_rejected(
    tinyruns: Command, repository: Path
) -> None:
    write_config(
        repository,
        "no_stage",
        {**MARKER_CONFIG, "stages": [f"{STAGES_PACKAGE}.absent"]},
    )
    write_config(
        repository,
        "no_packaged_stage",
        {**MARKER_CONFIG, "stages": ["tinyruns.stages.absent"]},
    )
    huge_smoke_run = ("configs/marker.yaml", "--smoke", "--resources", "huge")

    for config in ("configs/no_stage.yaml", "configs/no_packaged_stage.yaml"):
        assert "absent.py is missing" in tinyruns("submit", config, "--smoke").error
    assert "no resources 'huge'" in tinyruns("submit", *huge_smoke_run).error
    assert "tinyruns ls" in tinyruns("status", "20260101-000000-nothing").error
    assert tinyruns("ls").output == []


def test_environment_check_runs_as_a_cluster_job(
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

    run_id = submitted_smoke_run(tinyruns, "configs/environment.yaml")

    assert tinyruns("wait", run_id, "--interval", "0").output["state"] == "done"
    report = (cluster.run_directory(run_id) / "report.md").read_text()
    assert "Result: passed" in report
