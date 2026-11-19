import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from tinyruns.cluster.errors import ExperimentError

SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ControlMaster=auto",
    "-o",
    "ControlPath=~/.ssh/control-%C",
    "-o",
    "ControlPersist=600",
)
TEST_IS_FALSE_EXIT_CODE = 1


class RemoteCommandError(ExperimentError):
    pass


@dataclass(frozen=True)
class Remote:
    ssh_host: str | None

    def command_arguments(self, command: str) -> list[str]:
        if self.ssh_host is None:
            return ["bash", "-c", command]
        # A login shell, because non-interactive SSH sessions skip /etc/profile,
        # which is where clusters commonly put Slurm and Apptainer on PATH.
        return ["ssh", *SSH_OPTIONS, self.ssh_host, f"bash -lc {shlex.quote(command)}"]

    def completed(
        self, command: str, stdin: IO[bytes] | None = None
    ) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(
            self.command_arguments(command),
            stdin=subprocess.DEVNULL if stdin is None else stdin,
            capture_output=True,
        )

    def run(self, command: str, stdin: IO[bytes] | None = None) -> str:
        completed = self.completed(command, stdin)
        if completed.returncode != 0:
            raise RemoteCommandError(self.failure_message(command, completed))
        return completed.stdout.decode()

    def test(self, test_command: str) -> bool:
        completed = self.completed(test_command)
        if completed.returncode not in (0, TEST_IS_FALSE_EXIT_CODE):
            raise RemoteCommandError(self.failure_message(test_command, completed))
        return completed.returncode == 0

    def failure_message(
        self, command: str, completed: subprocess.CompletedProcess[bytes]
    ) -> str:
        where = self.ssh_host or "this machine"
        return (
            f"Command failed on {where} with exit code {completed.returncode}: "
            f"{command}\n{completed.stderr.decode().strip()}"
        )

    def location(self, path: str) -> str:
        return path if self.ssh_host is None else f"{self.ssh_host}:{path}"

    def transfer(
        self, source: str, destination: str, excludes: tuple[str, ...]
    ) -> None:
        arguments = ["rsync", "-a", *(f"--exclude={exclude}" for exclude in excludes)]
        if self.ssh_host is not None:
            arguments += ["-e", shlex.join(["ssh", *SSH_OPTIONS])]
        completed = subprocess.run(
            [*arguments, source, destination], capture_output=True, text=True
        )
        if completed.returncode != 0:
            raise RemoteCommandError(
                f"Copying {source} to {destination} failed: {completed.stderr.strip()}"
            )

    def upload_directory(self, local_directory: Path, remote_directory: str) -> None:
        self.transfer(f"{local_directory}/", self.location(f"{remote_directory}/"), ())

    def upload_file(self, local_file: Path, remote_file: str) -> None:
        self.transfer(str(local_file), self.location(remote_file), ())

    def download_directory(
        self, remote_directory: str, local_directory: Path, excludes: tuple[str, ...]
    ) -> None:
        self.transfer(
            self.location(f"{remote_directory}/"), f"{local_directory}/", excludes
        )
