import getpass
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from conftest import PROJECT_ROOT, STAGES_DIRECTORY, commit_all, git
from tinyruns.cluster.errors import ExperimentError
from tinyruns.cluster.job_script import (
    ImageBuildJob,
    RunJob,
    image_build_script,
    run_job_script,
)
from tinyruns.cluster.profile import (
    ClusterProfile,
    Resources,
    load_cluster_profile,
    time_limit_hours,
)
from tinyruns.cluster.remote import Remote, RemoteCommandError
from tinyruns.cluster.slurm import parsed_job_id
from tinyruns.cluster.snapshot import take_code_snapshot
from tinyruns.cluster.status import reconciled_status

GPU_RESOURCES = Resources(
    name="gpu",
    partition="gpu",
    time="04:00:00",
    cpus=8,
    memory="48G",
    gpus=1,
    constraint="GPU-80GB",
)
PROFILE_TEMPLATE = PROJECT_ROOT / "src" / "tinyruns" / "templates" / "cluster.yaml"
CPU_RESOURCES = Resources(
    name="cpu", partition="small", time="01:00:00", cpus=4, memory="8G"
)


def profile(node_local_scratch: str | None = "/scratch/local") -> ClusterProfile:
    return ClusterProfile(
        ssh_host="cluster",
        code_root="/home/someone/project/code",
        scratch_root="/scratch/someone/project",
        node_local_scratch=node_local_scratch,
        environment={"https_proxy": "http://proxy.example:3128"},
        gpu_hours_without_confirmation=24,
        image_build=CPU_RESOURCES,
        resources={"cpu": CPU_RESOURCES, "gpu": GPU_RESOURCES},
    )


def run_job(
    resources: Resources, smoke: bool, input_run_ids: tuple[str, ...] = ()
) -> RunJob:
    return RunJob(
        run_id="20261118-120000-example-abc1234",
        smoke=smoke,
        resources=resources,
        profile=profile(),
        code_directory="/home/someone/project/code/abc1234",
        run_directory="/scratch/someone/project/runs/20261118-120000-example-abc1234",
        image_path="/scratch/someone/project/images/0123456789abcdef.sif",
        input_run_ids=input_run_ids,
    )


def assert_bash_parses(script: str) -> None:
    completed = subprocess.run(
        ["bash", "-n"], input=script, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def write_profile(tmp_path: Path, **changes: Any) -> Path:
    content = yaml.safe_load(PROFILE_TEMPLATE.read_text())
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(yaml.safe_dump({**content, "ssh_host": None, **changes}))
    return profile_path


def test_full_gpu_job_script_reserves_a_gpu_and_stays_offline() -> None:
    script = run_job_script(run_job(GPU_RESOURCES, smoke=False))

    assert_bash_parses(script)
    for directive in ("--time=04:00:00", "--gpus=1", "--constraint=GPU-80GB"):
        assert f"#SBATCH {directive}" in script
    assert "--nv" in script
    assert "--env HF_HUB_OFFLINE=1" in script
    assert "--env https_proxy=http://proxy.example:3128" in script
    assert "/code/abc1234:/workdir:ro" in script
    assert script.rstrip().endswith(
        "python -m tinyruns.run /mnt/runs/20261118-120000-example-abc1234"
    )


def test_job_script_binds_its_run_writable_and_input_runs_read_only() -> None:
    runs = "/scratch/someone/project/runs"
    own_run, input_run = (
        "20261118-120000-example-abc1234",
        "20261117-090000-train-abc1234",
    )

    script = run_job_script(run_job(CPU_RESOURCES, True, input_run_ids=(input_run,)))

    assert_bash_parses(script)
    assert f"--bind {runs}/{own_run}:/mnt/runs/{own_run} " in script
    assert f"--bind {runs}/{input_run}:/mnt/runs/{input_run}:ro " in script
    assert script.count(":/mnt/runs/") == 2


def test_smoke_cpu_job_script_has_no_gpu_and_may_download() -> None:
    script = run_job_script(run_job(CPU_RESOURCES, smoke=True))

    assert_bash_parses(script)
    assert "--nv" not in script
    assert "--gpus" not in script
    assert "HF_HUB_OFFLINE" not in script


@pytest.mark.parametrize(
    ("node_local_scratch", "scratch_command"),
    [
        ("/scratch/local", "mktemp -d /scratch/local/image-build.XXXXXX"),
        (None, "build_scratch=$(mktemp -d)"),
    ],
)
def test_image_build_script_builds_off_the_shared_filesystem(
    node_local_scratch: str | None, scratch_command: str
) -> None:
    job = ImageBuildJob(
        image_stem="0123456789abcdef",
        profile=profile(node_local_scratch),
        build_directory="/scratch/someone/project/images/0123456789abcdef.build",
        image_path="/scratch/someone/project/images/0123456789abcdef.sif",
    )

    script = image_build_script(job)

    assert_bash_parses(script)
    assert scratch_command in script
    assert "export https_proxy=http://proxy.example:3128" in script
    assert "apptainer build --fakeroot" in script
    assert "export APPTAINER_CACHEDIR=/scratch/someone/project/cache/apptainer" in (
        script
    )


@pytest.mark.parametrize(
    ("time_limit", "hours"),
    [("00:30:00", 0.5), ("04:00:00", 4), ("1-12:30:00", 36.5)],
)
def test_time_limit_hours(time_limit: str, hours: float) -> None:
    assert time_limit_hours(time_limit) == hours


def test_time_limit_must_be_complete() -> None:
    with pytest.raises(ExperimentError, match="HH:MM:SS"):
        time_limit_hours("90")


def test_profile_template_loads_with_the_login_name_filled_in(tmp_path: Path) -> None:
    template = load_cluster_profile(write_profile(tmp_path))

    user = getpass.getuser()
    assert template.code_root == f"/home/{user}/experiments/code"
    assert template.runs_root == f"/scratch/{user}/experiments/runs"
    assert template.environment["https_proxy"] == "http://proxy.example.org:3128"
    assert all(resources.time for resources in template.resources.values())


def test_profile_reads_the_login_name_from_the_ssh_configuration(
    tmp_path: Path,
) -> None:
    profile_path = write_profile(tmp_path, ssh_host="host-without-ssh-config.invalid")

    assert getpass.getuser() in load_cluster_profile(profile_path).code_root


def test_profile_resources_need_a_time_limit(tmp_path: Path) -> None:
    untimed = {"untimed": {"partition": "small", "cpus": 1, "memory": "1G"}}

    with pytest.raises(ExperimentError, match="`time` limit"):
        load_cluster_profile(write_profile(tmp_path, resources=untimed))


def test_profile_paths_must_be_plain_absolute_paths(tmp_path: Path) -> None:
    with pytest.raises(ExperimentError, match="must be absolute"):
        load_cluster_profile(write_profile(tmp_path, code_root="code; rm -rf $HOME"))


def test_ssh_command_runs_in_a_login_shell_without_prompting() -> None:
    command = "echo \"$HOME\" 'quoted'"

    arguments = Remote("cluster").command_arguments(command)

    assert arguments[0] == "ssh"
    assert "BatchMode=yes" in arguments
    assert arguments[-2] == "cluster"
    assert shlex.split(arguments[-1]) == ["bash", "-lc", command]
    assert Remote("cluster").location("/scratch/run") == "cluster:/scratch/run"


def test_local_remote_runs_tests_and_reports_failures() -> None:
    this_machine = Remote(None)

    assert this_machine.run("echo reached") == "reached\n"
    assert this_machine.test("test -d /") is True
    assert this_machine.test("test -d /no/such/directory") is False
    with pytest.raises(RemoteCommandError, match="no-such-command"):
        this_machine.run("no-such-command")
    with pytest.raises(RemoteCommandError, match="exit code 3"):
        this_machine.test("exit 3")


def test_transfers_copy_directories_files_and_skip_excluded_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    (source / "heavy").mkdir(parents=True)
    (source / "nested" / "heavy").mkdir(parents=True)
    (source / "report.md").write_text("report")
    (source / "heavy" / "weights.bin").write_text("weights")
    (source / "nested" / "heavy" / "kept.txt").write_text("kept")
    this_machine = Remote(None)

    this_machine.upload_directory(source, str(tmp_path / "remote"))
    this_machine.upload_file(source / "report.md", str(tmp_path / "remote" / "copy.md"))
    this_machine.download_directory(
        str(tmp_path / "remote"), tmp_path / "pulled", excludes=("/heavy/",)
    )

    assert (tmp_path / "remote" / "heavy" / "weights.bin").is_file()
    assert (tmp_path / "pulled" / "copy.md").read_text() == "report"
    assert (tmp_path / "pulled" / "nested" / "heavy" / "kept.txt").is_file()
    assert not (tmp_path / "pulled" / "heavy").exists()
    with pytest.raises(RemoteCommandError, match="Copying"):
        this_machine.upload_file(tmp_path / "absent", str(tmp_path / "remote" / "x"))


@pytest.mark.parametrize(
    ("sbatch_output", "job_id"),
    [("4711\n", "4711"), ("Welcome to the cluster\n4712;cluster\n", "4712")],
)
def test_job_id_is_read_from_sbatch_output(sbatch_output: str, job_id: str) -> None:
    assert parsed_job_id(sbatch_output) == job_id


def test_sbatch_output_without_a_job_id_is_an_error() -> None:
    with pytest.raises(ExperimentError, match="job ID"):
        parsed_job_id("sbatch: error: invalid partition specified\n")


@pytest.mark.parametrize(
    ("reported_state", "slurm_state", "state"),
    [
        ("queued", "PENDING", "queued"),
        ("queued", "RUNNING", "running"),
        ("running", "PENDING", "queued"),
        ("running", "COMPLETING", "running"),
        ("running", "TIMEOUT", "failed"),
        ("running", "COMPLETED", "failed"),
        ("queued", None, "failed"),
        ("done", None, "done"),
        ("failed", "RUNNING", "failed"),
    ],
)
def test_status_is_reconciled_with_slurm(
    reported_state: str, slurm_state: str | None, state: str
) -> None:
    status = reconciled_status({"state": reported_state}, slurm_state, job_id="7")

    assert status["state"] == state


def test_timeout_is_named_as_the_failure() -> None:
    status = reconciled_status({"state": "running"}, "TIMEOUT", job_id="7")

    assert "TIMEOUT" in status["failure"]


def test_clean_repository_has_a_clean_snapshot(repository: Path) -> None:
    snapshot = take_code_snapshot(repository)

    assert not snapshot.is_dirty
    assert snapshot.commit == git(repository, "rev-parse", "HEAD")
    snapshot_files = git(repository, "ls-tree", "-r", "--name-only", snapshot.tree)
    assert "src/fixture_project/stages/marker.py" in snapshot_files
    assert "configs/marker.yaml" not in snapshot_files


def test_snapshot_holds_uncommitted_and_untracked_code(repository: Path) -> None:
    clean_tree = take_code_snapshot(repository).tree
    (repository / STAGES_DIRECTORY / "untracked.py").write_text("UNTRACKED = True\n")
    (repository / STAGES_DIRECTORY / "__pycache__").mkdir()
    (repository / STAGES_DIRECTORY / "__pycache__" / "ignored.pyc").write_text("")

    dirty_snapshot = take_code_snapshot(repository)

    assert dirty_snapshot.is_dirty
    assert dirty_snapshot.tree != clean_tree
    assert "UNTRACKED = True" in dirty_snapshot.patch
    snapshot_files = git(
        repository, "ls-tree", "-r", "--name-only", dirty_snapshot.tree
    )
    assert "untracked.py" in snapshot_files
    assert "ignored.pyc" not in snapshot_files
    assert git(repository, "status", "--porcelain") == (
        "?? src/fixture_project/stages/untracked.py"
    )


def test_committing_keeps_the_tree_and_other_files_do_not_change_it(
    repository: Path,
) -> None:
    (repository / STAGES_DIRECTORY / "untracked.py").write_text("UNTRACKED = True\n")
    dirty_snapshot = take_code_snapshot(repository)

    (repository / "notes.md").write_text("Not part of the snapshot.")
    commit_all(repository, "Add a stage and notes")
    committed_snapshot = take_code_snapshot(repository)

    assert committed_snapshot.tree == dirty_snapshot.tree
    assert not committed_snapshot.is_dirty
