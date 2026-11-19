from pathlib import Path
from shlex import quote
from typing import Any

from tinyruns.cluster.errors import ExperimentError
from tinyruns.cluster.new_run import runs_root
from tinyruns.cluster.profile import ClusterProfile
from tinyruns.cluster.remote import Remote
from tinyruns.run_data import DATA_DIRECTORY_NAME


def local_data_root(repository: Path) -> Path:
    return runs_root(repository).parent / DATA_DIRECTORY_NAME


def push_data_directory(repository: Path, profile: ClusterProfile) -> dict[str, Any]:
    data_root = local_data_root(repository)
    if not data_root.is_dir():
        raise ExperimentError(
            f"{data_root} does not exist. It is where a project keeps the datasets "
            "its stages read; create it and put a dataset in."
        )
    remote = Remote(profile.ssh_host)
    remote.run(f"mkdir -p {quote(profile.data_root)}")
    remote.upload_directory(data_root, profile.data_root)
    return {
        "pushed": str(data_root.relative_to(repository)),
        "to": remote.location(profile.data_root),
        "cluster": profile.name,
        "datasets": sorted(entry.name for entry in data_root.iterdir()),
    }
