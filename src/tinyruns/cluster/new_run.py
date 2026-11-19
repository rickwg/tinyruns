import copy
import hashlib
import itertools
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from tinyruns.cluster.errors import ExperimentError
from tinyruns.cluster.guardrails import require_known_stages
from tinyruns.cluster.snapshot import CodeSnapshot, take_code_snapshot
from tinyruns.run_config import (
    input_run_ids,
    is_smoke_test,
    load_run_config,
    validate_run_config,
)
from tinyruns.run_directory import (
    CREATED,
    RunDirectory,
    current_time,
    write_json,
    write_status,
)
from tinyruns.run_inputs import usable_input_run

RUNS_DIRECTORY = Path("artifacts") / "runs"
UNSAFE_NAME_CHARACTERS = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    config_name: str
    config: dict[str, Any]
    config_hash: str
    smoke: bool
    snapshot: CodeSnapshot


def runs_root(repository: Path) -> Path:
    return repository / RUNS_DIRECTORY


def config_hash(source_config: dict[str, Any]) -> str:
    comparable = copy.deepcopy(source_config)
    comparable.get("smoke_test", {}).pop("enabled", None)
    canonical = json.dumps(comparable, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def frozen_config(source_config: dict[str, Any], smoke: bool) -> dict[str, Any]:
    config = copy.deepcopy(source_config)
    config["smoke_test"] = {**config.get("smoke_test", {}), "enabled": smoke}
    return config


def new_run_id(
    existing_runs: Path, config_name: str, code_tree: str, smoke: bool
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    kind = "-smoke" if smoke else ""
    repeats = itertools.chain([""], (f"-{repeat}" for repeat in itertools.count(2)))
    run_ids = (
        f"{timestamp}-{config_name}-{code_tree[:7]}{repeat}{kind}" for repeat in repeats
    )
    return next(run_id for run_id in run_ids if not (existing_runs / run_id).exists())


def with_input_runs(
    source_config: dict[str, Any], named_on_command_line: dict[str, str]
) -> dict[str, Any]:
    if not named_on_command_line:
        return source_config
    inputs = {**source_config.get("inputs", {}), **named_on_command_line}
    return {**source_config, "inputs": inputs}


def planned_run(
    repository: Path,
    config_path: Path,
    smoke_requested: bool,
    input_runs_on_command_line: dict[str, str],
) -> PlannedRun:
    if not config_path.is_file():
        raise ExperimentError(f"Config {config_path} does not exist.")
    source_config = with_input_runs(
        load_run_config(config_path), input_runs_on_command_line
    )
    validate_run_config(source_config, source=f"{config_path} with --input")
    require_known_stages(repository, source_config)
    smoke = smoke_requested or is_smoke_test(source_config)
    snapshot = take_code_snapshot(repository)
    config_name = UNSAFE_NAME_CHARACTERS.sub("_", config_path.stem)
    return PlannedRun(
        run_id=new_run_id(runs_root(repository), config_name, snapshot.tree, smoke),
        config_name=config_name,
        config=frozen_config(source_config, smoke),
        config_hash=config_hash(source_config),
        smoke=smoke,
        snapshot=snapshot,
    )


def local_manifest(plan: PlannedRun) -> dict[str, Any]:
    return {
        "run_id": plan.run_id,
        "created_at": current_time(),
        "config_name": plan.config_name,
        "config_hash": plan.config_hash,
        "smoke": plan.smoke,
        "code": {
            "commit": plan.snapshot.commit,
            "tree": plan.snapshot.tree,
            "dirty": plan.snapshot.is_dirty,
        },
        "resources": None,
        "cluster": None,
        "image": None,
        "slurm": {"job_ids": [], "image_build_job_id": None},
    }


def created_status() -> dict[str, Any]:
    return {"state": CREATED, "created_at": current_time()}


def create_run_directory(
    repository: Path,
    plan: PlannedRun,
    manifest: dict[str, Any],
    status: dict[str, Any],
) -> RunDirectory:
    run_directory = RunDirectory(runs_root(repository) / plan.run_id)
    run_directory.path.mkdir(parents=True)
    run_directory.config_path.write_text(yaml.safe_dump(plan.config, sort_keys=False))
    write_json(run_directory.manifest_path, manifest)
    write_status(run_directory, status)
    if plan.snapshot.is_dirty:
        run_directory.code_patch_path.write_text(plan.snapshot.patch)
    return run_directory


def create_local_run(
    repository: Path,
    config_path: Path,
    smoke_requested: bool,
    input_runs_on_command_line: dict[str, str],
) -> RunDirectory:
    plan = planned_run(
        repository, config_path, smoke_requested, input_runs_on_command_line
    )
    for name, run_id in input_run_ids(plan.config).items():
        usable_input_run(runs_root(repository), name, run_id)
    return create_run_directory(
        repository, plan, local_manifest(plan), created_status()
    )
