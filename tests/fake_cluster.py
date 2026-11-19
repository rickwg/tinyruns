import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ACTIVE_STATES = ("PENDING", "RUNNING")
LOG_PATH_PATTERN = re.compile(r"^#SBATCH --output=(.+)$", re.MULTILINE)
DEPENDENCY_PREFIX = "--dependency=afterok:"
OPTIONS_WITHOUT_VALUE = ("--nv",)


def state_directory() -> Path:
    return Path(os.environ["FAKE_CLUSTER_STATE"])


def job_path(job_id: str) -> Path:
    return state_directory() / f"{job_id}.json"


def load_job(job_id: str) -> dict[str, Any] | None:
    if not job_path(job_id).is_file():
        return None
    return json.loads(job_path(job_id).read_text())


def save_job(job_id: str, job: dict[str, Any]) -> None:
    job_path(job_id).write_text(json.dumps(job))


def set_job_state(job_id: str, state: str) -> None:
    save_job(job_id, {**load_job(job_id), "state": state})


def job_state(job_id: str) -> str | None:
    job = load_job(job_id)
    return None if job is None else job["state"]


def allocate_job_id() -> str:
    counter_path = state_directory() / "last_job_id"
    job_id = int(counter_path.read_text()) + 1 if counter_path.is_file() else 1000
    counter_path.write_text(str(job_id))
    return str(job_id)


def run_job(job_id: str) -> None:
    script = Path(load_job(job_id)["script"])
    log_path = LOG_PATH_PATTERN.search(script.read_text()).group(1)
    set_job_state(job_id, "RUNNING")
    with open(log_path, "a") as log_file:
        completed = subprocess.run(
            ["bash", str(script)],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "SLURM_JOB_ID": job_id},
        )
    set_job_state(job_id, "COMPLETED" if completed.returncode == 0 else "FAILED")


def sbatch(arguments: list[str]) -> int:
    dependencies = [
        argument.removeprefix(DEPENDENCY_PREFIX)
        for argument in arguments
        if argument.startswith(DEPENDENCY_PREFIX)
    ]
    dependency_states = [job_state(dependency) for dependency in dependencies]
    job_id = allocate_job_id()
    save_job(job_id, {"script": arguments[-1], "state": "PENDING"})
    if any(state not in (*ACTIVE_STATES, "COMPLETED") for state in dependency_states):
        set_job_state(job_id, "CANCELLED")
    elif not is_held(dependency_states):
        run_job(job_id)
    print(job_id)
    return 0


def is_held(dependency_states: list[str | None]) -> bool:
    waits_for_dependency = any(state in ACTIVE_STATES for state in dependency_states)
    return waits_for_dependency or (state_directory() / "hold").exists()


def release(arguments: list[str]) -> int:
    run_job(arguments[0])
    return 0


def job_id_argument(arguments: list[str]) -> str:
    return arguments[arguments.index("-j") + 1]


def squeue(arguments: list[str]) -> int:
    state = job_state(job_id_argument(arguments))
    if state is None:
        print("slurm_load_jobs error: Invalid job id specified", file=sys.stderr)
        return 1
    if state in ACTIVE_STATES:
        print(state)
    return 0


def sacct(arguments: list[str]) -> int:
    state = job_state(job_id_argument(arguments))
    if state == "CANCELLED":
        print("CANCELLED by 501")
    elif state is not None:
        print(state)
    return 0


def scancel(arguments: list[str]) -> int:
    if job_state(arguments[0]) in ACTIVE_STATES:
        set_job_state(arguments[0], "CANCELLED")
    return 0


def apptainer_build(arguments: list[str]) -> int:
    image_path, definition = Path(arguments[-2]), Path(arguments[-1])
    build_inputs = (definition, Path("uv.lock"), Path("pyproject.toml"))
    missing_inputs = [str(path) for path in build_inputs if not path.is_file()]
    if missing_inputs or "FAKE_CLUSTER_FAIL_BUILD" in os.environ:
        print(f"FATAL: build failed, missing inputs: {missing_inputs}")
        return 255
    image_path.write_text("fake image")
    return 0


def host_path_translator(binds: list[str]) -> Any:
    host_by_container_path = {
        bind.split(":")[1]: bind.split(":")[0]
        for bind in sorted(binds, key=lambda bind: -len(bind.split(":")[1]))
    }

    def translated(text: str) -> str:
        for container_path, host_path in host_by_container_path.items():
            text = text.replace(container_path, host_path)
        return text

    return translated


def split_exec_arguments(
    arguments: list[str],
) -> tuple[dict[str, list[str]], str, list[str]]:
    options: dict[str, list[str]] = {"--bind": [], "--env": [], "--pwd": []}
    position = 0
    while arguments[position].startswith("--"):
        if arguments[position] in OPTIONS_WITHOUT_VALUE:
            position += 1
        else:
            options[arguments[position]].append(arguments[position + 1])
            position += 2
    return options, arguments[position], arguments[position + 1 :]


def apptainer_exec(arguments: list[str]) -> int:
    options, image, command = split_exec_arguments(arguments)
    if not Path(image).is_file():
        print(f"FATAL: could not open image {image}")
        return 255
    translated = host_path_translator(options["--bind"])
    container_variables = dict(setting.split("=", 1) for setting in options["--env"])
    interpreter = os.environ["FAKE_CLUSTER_PYTHON"]
    completed = subprocess.run(
        [interpreter if word == "python" else translated(word) for word in command],
        cwd=translated(options["--pwd"][0]),
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            **{name: translated(value) for name, value in container_variables.items()},
        },
    )
    return completed.returncode


def apptainer(arguments: list[str]) -> int:
    subcommands = {"build": apptainer_build, "exec": apptainer_exec}
    return subcommands[arguments[0]](arguments[1:])


COMMANDS = {
    "sbatch": sbatch,
    "squeue": squeue,
    "sacct": sacct,
    "scancel": scancel,
    "apptainer": apptainer,
    "release": release,
}

if __name__ == "__main__":
    sys.exit(COMMANDS[sys.argv[1]](sys.argv[2:]))
