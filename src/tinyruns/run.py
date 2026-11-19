import argparse
import importlib
import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tinyruns.run_config import (
    RunConfigError,
    load_run_config,
    stage_names,
)
from tinyruns.run_directory import (
    DONE,
    FAILED,
    RUNNING,
    RunDirectory,
    current_time,
    write_status,
)

Stage = Callable[[RunDirectory, dict[str, Any]], None]


def stage_function(stage_name: str) -> Stage:
    return importlib.import_module(stage_name).run_stage


def last_part(stage_name: str) -> str:
    return stage_name.rpartition(".")[2]


def selected_stage_names(config: dict[str, Any], only_stage: str | None) -> list[str]:
    configured_stages = stage_names(config)
    if only_stage is None:
        return configured_stages
    if only_stage in configured_stages:
        return [only_stage]
    abbreviated_stages = [
        stage_name
        for stage_name in dict.fromkeys(configured_stages)
        if last_part(stage_name) == only_stage
    ]
    if len(abbreviated_stages) != 1:
        raise RunConfigError(
            f"Stage '{only_stage}' names {len(abbreviated_stages)} of this run's "
            f"stages: {configured_stages}. Give a stage's module path, or its last "
            "part when no other stage ends the same."
        )
    return abbreviated_stages


def running_status(stages: list[str]) -> dict[str, Any]:
    return {
        "state": RUNNING,
        "stages": stages,
        "started_at": current_time(),
        "host": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def failed_status(status: dict[str, Any], error: BaseException) -> dict[str, Any]:
    return {
        **status,
        "state": FAILED,
        "finished_at": current_time(),
        "failure": f"{type(error).__name__}: {error}",
    }


def done_status(status: dict[str, Any]) -> dict[str, Any]:
    return {**status, "state": DONE, "finished_at": current_time()}


def run_stages(run_directory: RunDirectory, only_stage: str | None = None) -> None:
    config = load_run_config(run_directory.config_path)
    status = running_status(selected_stage_names(config, only_stage))
    write_status(run_directory, status)
    try:
        for stage_name in status["stages"]:
            stage_function(stage_name)(run_directory, config)
    except BaseException as error:
        write_status(run_directory, failed_status(status, error))
        raise
    write_status(run_directory, done_status(status))


def parsed_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the stages of a run directory, on the cluster or locally."
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--stage", help="Run only this stage of the run.")
    return parser.parse_args()


def main() -> None:
    arguments = parsed_arguments()
    run_stages(RunDirectory(arguments.run_directory), only_stage=arguments.stage)


if __name__ == "__main__":
    main()
