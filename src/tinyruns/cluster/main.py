"""Create experiment runs, and submit, watch and fetch them on the cluster.

Every command prints JSON on stdout and exits non-zero on failure, with the reason
as JSON on stderr. A run made by `tinyruns new` or fetched by `tinyruns pull` runs
locally with:
uv run python -m tinyruns.run artifacts/runs/<run_id>
"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tinyruns.cluster.errors import ExperimentError
from tinyruns.cluster.new_run import create_local_run, runs_root
from tinyruns.cluster.operations import (
    cancel_run,
    pull_run,
    run_log,
    run_summary,
    wait_for_run,
)
from tinyruns.cluster.snapshot import repository_root
from tinyruns.cluster.status import (
    is_cluster_run,
    local_runs,
    refresh_status,
)
from tinyruns.cluster.submit import (
    SubmitRequest,
    resume_run,
    submit_run,
)
from tinyruns.run_config import RunConfigError
from tinyruns.run_directory import (
    DONE,
    FAILED,
    TERMINAL_STATES,
    RunDirectory,
    read_json,
    read_status,
)
from tinyruns.run_inputs import UnusableInputRunError

DEFAULT_CLUSTER_PROFILE = Path(".cluster.yaml")
EXIT_CODE_BY_FINAL_STATE = {DONE: 0, FAILED: 1}
STILL_RUNNING_EXIT_CODE = 2


def print_json(content: Any) -> None:
    print(json.dumps(content, indent=2))


def known_run(repository: Path, run_id: str) -> RunDirectory:
    run_directory = RunDirectory(runs_root(repository) / run_id)
    if not run_directory.manifest_path.is_file():
        raise ExperimentError(
            f"No run '{run_id}' under {runs_root(repository)}. "
            "`tinyruns ls` lists the runs."
        )
    return run_directory


def run_command(repository: Path, run_directory: RunDirectory) -> str:
    relative_directory = run_directory.path.relative_to(repository)
    return f"uv run python -m tinyruns.run {relative_directory}"


def cluster_run(repository: Path, run_id: str) -> RunDirectory:
    run_directory = known_run(repository, run_id)
    if not is_cluster_run(run_directory):
        raise ExperimentError(
            f"Run '{run_id}' was made by `tinyruns new` and exists only on this "
            f"machine. Run it with: {run_command(repository, run_directory)}"
        )
    return run_directory


def with_run_id(run_directory: RunDirectory, status: dict[str, Any]) -> dict[str, Any]:
    return {"run_id": run_directory.path.name, **status}


def input_runs_on_command_line(assignments: list[str]) -> dict[str, str]:
    malformed = [assignment for assignment in assignments if "=" not in assignment]
    if malformed:
        raise ExperimentError(f"--input takes NAME=RUN_ID; it got {malformed}.")
    return dict(assignment.split("=", 1) for assignment in assignments)


def new(arguments: argparse.Namespace, repository: Path) -> int:
    run_directory = create_local_run(
        repository,
        arguments.config,
        arguments.smoke,
        input_runs_on_command_line(arguments.input),
    )
    print_json(
        {
            **read_json(run_directory.manifest_path),
            "directory": str(run_directory.path.relative_to(repository)),
            "run_with": run_command(repository, run_directory),
        }
    )
    return 0


def submit(arguments: argparse.Namespace, repository: Path) -> int:
    request = SubmitRequest(
        repository=repository,
        config_path=arguments.config,
        cluster_profile_path=arguments.cluster or repository / DEFAULT_CLUSTER_PROFILE,
        resources=arguments.resources,
        smoke=arguments.smoke,
        confirmed=arguments.confirm,
        skip_smoke_check=arguments.skip_smoke_check,
        input_run_ids=input_runs_on_command_line(arguments.input),
    )
    print_json(submit_run(request))
    return 0


def unfinished_runs(repository: Path) -> list[RunDirectory]:
    return [
        run_directory
        for run_directory in local_runs(runs_root(repository))
        if is_cluster_run(run_directory)
        and read_status(run_directory)["state"] not in TERMINAL_STATES
    ]


def status(arguments: argparse.Namespace, repository: Path) -> int:
    if arguments.run_id is None:
        print_json(
            [
                with_run_id(run_directory, refresh_status(run_directory))
                for run_directory in unfinished_runs(repository)
            ]
        )
    else:
        run_directory = known_run(repository, arguments.run_id)
        print_json(with_run_id(run_directory, refresh_status(run_directory)))
    return 0


def wait(arguments: argparse.Namespace, repository: Path) -> int:
    run_directory = cluster_run(repository, arguments.run_id)
    final_status = wait_for_run(run_directory, arguments.interval, arguments.timeout)
    print_json(with_run_id(run_directory, final_status))
    return EXIT_CODE_BY_FINAL_STATE.get(final_status["state"], STILL_RUNNING_EXIT_CODE)


def logs(arguments: argparse.Namespace, repository: Path) -> int:
    run_directory = cluster_run(repository, arguments.run_id)
    print_json(
        {
            "run_id": arguments.run_id,
            "log": run_log(run_directory, arguments.lines),
        }
    )
    return 0


def pull(arguments: argparse.Namespace, repository: Path) -> int:
    run_directory = cluster_run(repository, arguments.run_id)
    pulled_status = pull_run(run_directory, include_heavy=arguments.heavy)
    print_json(
        {
            **with_run_id(run_directory, pulled_status),
            "directory": str(run_directory.path.relative_to(repository)),
        }
    )
    return 0


def list_runs(arguments: argparse.Namespace, repository: Path) -> int:
    print_json([run_summary(run) for run in local_runs(runs_root(repository))])
    return 0


def cancel(arguments: argparse.Namespace, repository: Path) -> int:
    run_directory = cluster_run(repository, arguments.run_id)
    print_json(with_run_id(run_directory, cancel_run(run_directory)))
    return 0


def resume(arguments: argparse.Namespace, repository: Path) -> int:
    print_json(resume_run(cluster_run(repository, arguments.run_id)))
    return 0


Handler = Callable[[argparse.Namespace, Path], int]


def add_input_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="NAME=RUN_ID",
        help="A finished run whose outputs this run reads; sets `inputs.NAME`.",
    )


def add_new_command(commands: Any) -> None:
    parser = commands.add_parser(
        "new", help="Create a run directory on this machine, without the cluster."
    )
    parser.add_argument("config", type=Path, help="Experiment config (YAML).")
    parser.add_argument(
        "--smoke", action="store_true", help="Freeze the config as a short test run."
    )
    add_input_option(parser)
    parser.set_defaults(handler=new)


def add_submit_command(commands: Any) -> None:
    parser = commands.add_parser(
        "submit", help="Snapshot the code, create a run and queue it."
    )
    parser.add_argument("config", type=Path, help="Experiment config (YAML).")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Short test run; may use uncommitted code and may download models.",
    )
    parser.add_argument(
        "--resources",
        metavar="NAME",
        help="Resources from the cluster profile. Needed for a full run; a smoke "
        "run without it uses the profile's `smoke_resources`.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="A human approved exceeding the cluster profile's GPU-hour limit.",
    )
    parser.add_argument(
        "--skip-smoke-check",
        action="store_true",
        help="Submit a full run without a passed smoke run of this code and config.",
    )
    parser.add_argument(
        "--cluster",
        type=Path,
        help=f"Cluster profile (YAML). Default: {DEFAULT_CLUSTER_PROFILE}.",
    )
    add_input_option(parser)
    parser.set_defaults(handler=submit)


def add_run_command(
    commands: Any, name: str, handler: Handler, help_text: str
) -> argparse.ArgumentParser:
    parser = commands.add_parser(name, help=help_text)
    parser.add_argument("run_id")
    parser.set_defaults(handler=handler)
    return parser


def add_status_command(commands: Any) -> None:
    parser = commands.add_parser(
        "status", help="Ask the cluster for the state of one run, or of all unfinished."
    )
    parser.add_argument("run_id", nargs="?")
    parser.set_defaults(handler=status)


def add_wait_command(commands: Any) -> None:
    parser = add_run_command(
        commands,
        "wait",
        wait,
        "Block until the run ends. Exit code: 0 done, 1 failed, 2 still running.",
    )
    parser.add_argument("--interval", type=float, default=30.0, help="Seconds.")
    parser.add_argument("--timeout", type=float, help="Seconds. Default: none.")


def add_logs_command(commands: Any) -> None:
    parser = add_run_command(
        commands, "logs", logs, "Show the end of the run's Slurm log."
    )
    parser.add_argument("--lines", type=int, default=50)


def add_pull_command(commands: Any) -> None:
    parser = add_run_command(
        commands, "pull", pull, "Copy the run directory from the cluster."
    )
    parser.add_argument(
        "--heavy", action="store_true", help="Include the run's heavy/ directory."
    )


def add_list_command(commands: Any) -> None:
    parser = commands.add_parser(
        "ls", help="List the runs known locally, without asking the cluster."
    )
    parser.set_defaults(handler=list_runs)


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tinyruns",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    add_new_command(commands)
    add_submit_command(commands)
    add_status_command(commands)
    add_wait_command(commands)
    add_logs_command(commands)
    add_pull_command(commands)
    add_list_command(commands)
    add_run_command(commands, "cancel", cancel, "Cancel the run's Slurm job.")
    add_run_command(
        commands, "resume", resume, "Queue a failed run again, in its directory."
    )
    return parser


def main(command_line: list[str] | None = None) -> int:
    arguments = argument_parser().parse_args(command_line)
    try:
        return arguments.handler(arguments, repository_root(Path.cwd()))
    except (ExperimentError, RunConfigError, UnusableInputRunError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
