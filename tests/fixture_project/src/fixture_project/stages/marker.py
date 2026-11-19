import os
from typing import Any

from tinyruns.run_config import is_smoke_test
from tinyruns.run_directory import RunDirectory, append_metrics
from tinyruns.run_inputs import input_run

RECORDED_ENVIRONMENT_VARIABLES = (
    "SLURM_JOB_ID",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "FAKE_PROXY",
    "PYTHONPATH",
)


def fail_first_attempt(run_directory: RunDirectory) -> None:
    attempt_marker = run_directory.path / "first_attempt"
    if not attempt_marker.exists():
        attempt_marker.touch()
        raise RuntimeError("the first attempt fails")


def input_run_outputs(
    run_directory: RunDirectory, config: dict[str, Any], name: str
) -> dict[str, str]:
    source = input_run(run_directory, config, name)
    return {
        "input_report": source.report_path.read_text(),
        "input_weights": (source.heavy_directory / "weights.bin").read_text(),
    }


def run_stage(run_directory: RunDirectory, config: dict[str, Any]) -> None:
    settings = config["marker"]
    read_input = settings.get("read_input")
    print(f"marker stage runs: {settings['text']}", flush=True)
    append_metrics(
        run_directory,
        {
            "smoke_test": is_smoke_test(config),
            **{name: os.environ.get(name) for name in RECORDED_ENVIRONMENT_VARIABLES},
            **(
                input_run_outputs(run_directory, config, read_input)
                if read_input
                else {}
            ),
        },
    )
    run_directory.heavy_directory.mkdir(exist_ok=True)
    (run_directory.heavy_directory / "weights.bin").write_bytes(b"heavy")
    if settings.get("fail_first_attempt", False):
        fail_first_attempt(run_directory)
    run_directory.report_path.write_text(settings["text"])
