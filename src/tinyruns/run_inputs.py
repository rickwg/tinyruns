from pathlib import Path
from typing import Any

from tinyruns.run_config import RunConfigError, input_run_ids
from tinyruns.run_directory import DONE, RunDirectory, read_status


class UnusableInputRunError(RuntimeError):
    pass


def usable_input_run(runs_directory: Path, name: str, run_id: str) -> RunDirectory:
    input_run_directory = RunDirectory(runs_directory / run_id)
    fetch_hint = (
        f"If it ran on the cluster, fetch it with: uv run exp pull {run_id} --heavy"
    )
    if not input_run_directory.status_path.is_file():
        raise UnusableInputRunError(
            f"Input '{name}' is run '{run_id}', which is not in {runs_directory}. "
            f"{fetch_hint}"
        )
    state = read_status(input_run_directory)["state"]
    if state != DONE:
        raise UnusableInputRunError(
            f"Input '{name}' is run '{run_id}', which is {state} here, not done. "
            f"{fetch_hint}"
        )
    return input_run_directory


def input_run(
    run_directory: RunDirectory, config: dict[str, Any], name: str
) -> RunDirectory:
    run_ids = input_run_ids(config)
    if name not in run_ids:
        raise RunConfigError(
            f"This run has no input named '{name}'; its inputs are {sorted(run_ids)}. "
            f"Name the run in the config under `inputs`, or pass --input {name}=RUN_ID."
        )
    return usable_input_run(run_directory.path.absolute().parent, name, run_ids[name])
