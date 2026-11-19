import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class RunConfigError(ValueError):
    pass


def load_run_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text())
    validate_run_config(config, source=str(path))
    return config


def validate_run_config(config: Any, source: str) -> None:
    if not isinstance(config, dict):
        raise RunConfigError(f"{source} must hold a YAML mapping.")
    located_values = list(location_bound_values(config))
    if located_values:
        raise RunConfigError(
            f"{source} names a location: {', '.join(located_values)}. "
            "Write paths relative to the run directory, so the run works wherever "
            "its directory is copied."
        )
    stage_names(config)
    input_run_ids(config)


def location_bound_values(value: Any, key: str = "") -> Iterator[str]:
    if isinstance(value, dict):
        for name, nested in value.items():
            yield from location_bound_values(nested, f"{key}.{name}" if key else name)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from location_bound_values(nested, f"{key}[{index}]")
    elif isinstance(value, str) and value.startswith(("/", "~")):
        yield f"{key}={value}"


def stage_names(config: dict[str, Any]) -> list[str]:
    stages = config.get("stages")
    if (
        not isinstance(stages, list)
        or not stages
        or not all(map(is_module_path, stages))
    ):
        raise RunConfigError(
            "The config needs `stages`: a non-empty list of module paths, for "
            "example `my_project.stages.train`."
        )
    return stages


def is_module_name(value: Any) -> bool:
    return isinstance(value, str) and value.isidentifier()


def is_module_path(value: Any) -> bool:
    return isinstance(value, str) and all(map(is_module_name, value.split(".")))


def input_run_ids(config: dict[str, Any]) -> dict[str, str]:
    inputs = config.get("inputs", {})
    if not isinstance(inputs, dict) or not all(
        is_module_name(name) and is_run_id(run_id) for name, run_id in inputs.items()
    ):
        raise RunConfigError(
            "`inputs` must map names to run IDs, for example "
            "`inputs: {training: 20261119-081855-train-407ab79}`. A run ID is the "
            "name of a run directory, not a path."
        )
    return inputs


def is_run_id(value: Any) -> bool:
    return isinstance(value, str) and RUN_ID_PATTERN.fullmatch(value) is not None


def is_smoke_test(config: dict[str, Any]) -> bool:
    return bool(config.get("smoke_test", {}).get("enabled", False))
