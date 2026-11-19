import os
import platform
import socket
import urllib.error
import urllib.request
from typing import Any

import torch

from tinyruns.run_config import input_run_ids, is_smoke_test
from tinyruns.run_directory import (
    RunDirectory,
    append_metrics,
    current_time,
    read_status,
)
from tinyruns.run_inputs import input_run

REPORTED_ENVIRONMENT_VARIABLES = (
    "SLURM_JOB_ID",
    "HF_HOME",
    "HF_HUB_OFFLINE",
    "http_proxy",
    "https_proxy",
)
DEFAULT_TIMEOUT_SECONDS = 20.0


def environment_facts() -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpus": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        **{name: os.environ.get(name) for name in REPORTED_ENVIRONMENT_VARIABLES},
    }


def url_response(url: str, timeout_seconds: float) -> str:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return f"HTTP {response.status}"
    except urllib.error.HTTPError as error:
        return f"HTTP {error.code}"
    except OSError as error:
        return f"unreachable: {error}"


def input_run_description(
    run_directory: RunDirectory, config: dict[str, Any], name: str
) -> str:
    source = input_run(run_directory, config, name)
    access = "writable" if os.access(source.path, os.W_OK) else "read-only"
    return f"{source.path.name}: {read_status(source)['state']}, {access}"


def failed_checks(
    settings: dict[str, Any], facts: dict[str, Any], url_responses: dict[str, str]
) -> list[str]:
    failures = [
        f"{url} is {response}"
        for url, response in url_responses.items()
        if response.startswith("unreachable")
    ]
    if settings.get("require_gpu", False) and not facts["cuda_available"]:
        failures.append("no CUDA device is available")
    return failures


def markdown_table(heading: str, rows: dict[str, Any]) -> str:
    lines = [f"| {heading} | Value |", "|---|---|"]
    lines.extend(f"| {name} | {value} |" for name, value in rows.items())
    return "\n".join(lines)


def report_markdown(
    facts: dict[str, Any],
    url_responses: dict[str, str],
    input_runs: dict[str, str],
    failures: list[str],
) -> str:
    verdict = "passed" if not failures else "failed: " + "; ".join(failures)
    sections = [
        "# Environment check",
        f"Result: {verdict}",
        markdown_table("Fact", facts),
    ]
    if url_responses:
        sections.append(markdown_table("URL", url_responses))
    if input_runs:
        sections.append(markdown_table("Input", input_runs))
    return "\n\n".join(sections) + "\n"


def run_stage(run_directory: RunDirectory, config: dict[str, Any]) -> None:
    settings = config.get("environment_check", {})
    timeout_seconds = settings.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    facts = {**environment_facts(), "smoke_test": is_smoke_test(config)}
    url_responses = {
        url: url_response(url, timeout_seconds) for url in settings.get("urls", [])
    }
    input_runs = {
        name: input_run_description(run_directory, config, name)
        for name in input_run_ids(config)
    }
    failures = failed_checks(settings, facts, url_responses)
    append_metrics(
        run_directory,
        {
            "recorded_at": current_time(),
            **facts,
            "urls": url_responses,
            "inputs": input_runs,
        },
    )
    run_directory.report_path.write_text(
        report_markdown(facts, url_responses, input_runs, failures)
    )
    if failures:
        raise RuntimeError("Environment check failed: " + "; ".join(failures))
