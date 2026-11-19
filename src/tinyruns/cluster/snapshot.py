import os
import shlex
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from tinyruns.cluster.errors import ExperimentError
from tinyruns.cluster.remote import Remote

SNAPSHOT_PATHS = ("src", "pyproject.toml", "uv.lock", "apptainer")


@dataclass(frozen=True)
class CodeSnapshot:
    tree: str
    commit: str
    patch: str

    @property
    def is_dirty(self) -> bool:
        return self.patch != ""


def git_output(
    repository: Path, *arguments: str, index_file: Path | None = None
) -> str:
    environment = dict(os.environ)
    if index_file is not None:
        environment["GIT_INDEX_FILE"] = str(index_file)
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0:
        raise ExperimentError(
            f"git {' '.join(arguments)} failed: {completed.stderr.strip()}"
        )
    return completed.stdout


def repository_root(directory: Path) -> Path:
    return Path(git_output(directory, "rev-parse", "--show-toplevel").strip())


def require_snapshot_paths(repository: Path) -> None:
    missing_paths = [
        path for path in SNAPSHOT_PATHS if not (repository / path).exists()
    ]
    if missing_paths:
        raise ExperimentError(
            f"{repository} lacks {missing_paths}. Every run records the state of "
            f"{', '.join(SNAPSHOT_PATHS)}."
        )


def take_code_snapshot(repository: Path) -> CodeSnapshot:
    require_snapshot_paths(repository)
    with tempfile.TemporaryDirectory() as scratch_directory:
        scratch_index = Path(scratch_directory) / "index"
        git_output(
            repository, "add", "--all", "--", *SNAPSHOT_PATHS, index_file=scratch_index
        )
        tree = git_output(repository, "write-tree", index_file=scratch_index).strip()
        patch = git_output(
            repository,
            "diff",
            "--cached",
            "--binary",
            "HEAD",
            "--",
            *SNAPSHOT_PATHS,
            index_file=scratch_index,
        )
    commit = git_output(repository, "rev-parse", "HEAD").strip()
    return CodeSnapshot(tree=tree, commit=commit, patch=patch)


def unpack_command(snapshot_directory: str) -> str:
    target = shlex.quote(snapshot_directory)
    staging = shlex.quote(f"{snapshot_directory}.partial-{uuid.uuid4().hex[:8]}")
    return (
        f"mkdir -p {staging} && tar -x -f - -C {staging} && "
        f"if [ -d {target} ]; then rm -rf {staging}; else mv {staging} {target}; fi"
    )


def upload_code_snapshot(
    repository: Path, snapshot: CodeSnapshot, remote: Remote, code_root: str
) -> str:
    snapshot_directory = f"{code_root}/{snapshot.tree}"
    if remote.test(f"test -d {shlex.quote(snapshot_directory)}"):
        return snapshot_directory
    archive = subprocess.Popen(
        ["git", "-C", str(repository), "archive", "--format=tar", snapshot.tree],
        stdout=subprocess.PIPE,
    )
    remote.run(unpack_command(snapshot_directory), stdin=archive.stdout)
    if archive.wait() != 0:
        raise ExperimentError(f"git archive of tree {snapshot.tree} failed.")
    return snapshot_directory
