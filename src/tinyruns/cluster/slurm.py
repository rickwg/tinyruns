import re
import shlex

from tinyruns.cluster.errors import ExperimentError
from tinyruns.cluster.remote import Remote

SLURM_STATE_PATTERN = re.compile(r"^slurm_state=(\w*)", re.MULTILINE)


def parsed_job_id(sbatch_output: str) -> str:
    lines = [line.strip() for line in sbatch_output.splitlines() if line.strip()]
    job_id = lines[-1].split(";")[0] if lines else ""
    if not job_id.isdigit():
        raise ExperimentError(f"sbatch did not print a job ID: {sbatch_output!r}")
    return job_id


def submit_job(
    remote: Remote, script_path: str, after_job_id: str | None = None
) -> str:
    dependency = "" if after_job_id is None else f"--dependency=afterok:{after_job_id} "
    output = remote.run(f"sbatch --parsable {dependency}{shlex.quote(script_path)}")
    return parsed_job_id(output)


def slurm_state_command(job_id: str) -> str:
    queued_or_running = f"squeue -h -j {job_id} -o %T 2>/dev/null | head -n 1"
    finished = f"sacct -n -X -P -j {job_id} -o State 2>/dev/null | head -n 1"
    return (
        f"state=$({queued_or_running}); "
        f'[ -n "$state" ] || state=$({finished}); '
        'echo "slurm_state=$state"'
    )


def slurm_state(remote: Remote, job_id: str) -> str | None:
    match = SLURM_STATE_PATTERN.search(remote.run(slurm_state_command(job_id)))
    if match is None or match.group(1) == "":
        return None
    return match.group(1).upper()


def cancel_job(remote: Remote, job_id: str) -> None:
    remote.run(f"scancel {job_id}")
