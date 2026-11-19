from pathlib import Path

from tinyruns.run_directory import RunDirectory

DATA_DIRECTORY_NAME = "data"


def data_directory(run_directory: RunDirectory) -> Path:
    return run_directory.path.absolute().parent.parent / DATA_DIRECTORY_NAME
