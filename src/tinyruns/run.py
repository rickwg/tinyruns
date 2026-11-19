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
    module = importlib.import_module(f"tinyruns.stages.{stage_name}")
    return module.run_stage


def selected_stage_names(config: dict[str, Any], only_stage: str | None) -> list[str]:
    configured_stages = stage_names(config)
    if only_stage is None:
        return configured_stages
    if only_stage not in configured_stages:
        raise RunConfigError(
            f"Stage '{only_stage}' is not in this run's stages: {configured_stages}."
        )
    return [only_stage]


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
