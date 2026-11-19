import functools
import http.server
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from tinyruns.run import run_stages, selected_stage_names
from tinyruns.run_config import RunConfigError, validate_run_config
from tinyruns.run_directory import RunDirectory, read_status
from tinyruns.stages.environment_check import url_response

CLOSED_PORT_URL = "http://127.0.0.1:1"
ENVIRONMENT_CHECK_STAGE = "tinyruns.stages.environment_check"


def run_directory_with(tmp_path: Path, config: dict[str, Any]) -> RunDirectory:
    run_directory = RunDirectory(tmp_path / "run")
    run_directory.path.mkdir()
    run_directory.config_path.write_text(yaml.safe_dump(config))
    return run_directory


def environment_check_config(urls: list[str]) -> dict[str, Any]:
    return {
        "stages": [ENVIRONMENT_CHECK_STAGE],
        "environment_check": {"urls": urls, "timeout_seconds": 5},
    }


@pytest.fixture
def served_directory_url(tmp_path: Path) -> Iterator[str]:
    (tmp_path / "present.txt").write_text("present")
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(tmp_path)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize(
    ("config", "named_location"),
    [
        ({"stages": ["a"], "data": {"directory": "/mnt/data"}}, "data.directory"),
        ({"stages": ["a"], "sources": ["relative", "~/home"]}, "sources[1]"),
    ],
)
def test_config_must_not_name_a_location(
    config: dict[str, Any], named_location: str
) -> None:
    with pytest.raises(RunConfigError, match=named_location.replace("[", r"\[")):
        validate_run_config(config, source="config.yaml")


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"stages": []},
        {"stages": ["not a module"]},
        {"stages": ["package..module"]},
        {"stages": ["package.module."]},
        ["stages"],
    ],
)
def test_config_needs_stages(config: Any) -> None:
    with pytest.raises(RunConfigError):
        validate_run_config(config, source="config.yaml")


def test_config_with_relative_paths_is_valid() -> None:
    validate_run_config(
        {"stages": ["a"], "output": "figures/plot.png", "url": "https://example.org"},
        source="config.yaml",
    )


def test_finished_run_records_done_and_its_outputs(tmp_path: Path) -> None:
    run_directory = run_directory_with(tmp_path, environment_check_config(urls=[]))

    run_stages(run_directory)

    status = read_status(run_directory)
    assert status["state"] == "done"
    assert status["stages"] == [ENVIRONMENT_CHECK_STAGE]
    assert "Result: passed" in run_directory.report_path.read_text()
    recorded = json.loads(run_directory.metrics_path.read_text())
    assert recorded["smoke_test"] is False
    assert recorded["urls"] == {}


def test_failing_stage_records_failed_and_raises(tmp_path: Path) -> None:
    run_directory = run_directory_with(
        tmp_path, environment_check_config(urls=[CLOSED_PORT_URL])
    )

    with pytest.raises(RuntimeError, match="Environment check failed"):
        run_stages(run_directory)

    status = read_status(run_directory)
    assert status["state"] == "failed"
    assert CLOSED_PORT_URL in status["failure"]
    assert "Result: failed" in run_directory.report_path.read_text()


def test_only_a_configured_stage_can_be_selected(tmp_path: Path) -> None:
    run_directory = run_directory_with(tmp_path, environment_check_config(urls=[]))

    with pytest.raises(RunConfigError, match="other_stage"):
        run_stages(run_directory, only_stage="other_stage")
    run_stages(run_directory, only_stage=ENVIRONMENT_CHECK_STAGE)

    assert read_status(run_directory)["state"] == "done"


def test_stage_is_selected_by_the_last_part_of_its_module_path(tmp_path: Path) -> None:
    run_directory = run_directory_with(tmp_path, environment_check_config(urls=[]))

    run_stages(run_directory, only_stage="environment_check")

    assert read_status(run_directory)["stages"] == [ENVIRONMENT_CHECK_STAGE]


def test_last_part_shared_by_two_stages_selects_neither() -> None:
    config = {"stages": ["training.run", "evaluation.run"]}

    with pytest.raises(RunConfigError, match="names 2 of this run's stages"):
        selected_stage_names(config, only_stage="run")
    assert selected_stage_names(config, only_stage="training.run") == ["training.run"]


def test_url_response_tells_reachable_from_unreachable(
    served_directory_url: str,
) -> None:
    assert url_response(f"{served_directory_url}/present.txt", 5) == "HTTP 200"
    assert url_response(f"{served_directory_url}/absent.txt", 5) == "HTTP 404"
    assert url_response(CLOSED_PORT_URL, 5).startswith("unreachable")
