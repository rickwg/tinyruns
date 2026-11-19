# tinyruns

Run directories for experiments, and `exp`, a command that runs them on a Slurm cluster inside an Apptainer image.

An experiment is a config, executed by one or more stages, with everything it produces kept in one run directory. The same run directory runs on a laptop and in a cluster job, and can be copied between the two.

## Status

Early, and not yet usable from another project. The code was written inside one research project and is being generalised:

- Stage modules are still looked up inside this package (`tinyruns.stages`), not in the project that uses it.
- The environment-check stage needs PyTorch, which is only a development dependency here.
- There are no commands yet to create a cluster profile or an image definition from a template.

## Development

```bash
uv sync
uv run pytest
uv run black .
```

The tests need no cluster. They run `exp` against fake `sbatch`, `squeue`, `sacct`, `scancel` and `apptainer` commands, and they block internet access.

## Layout

| Path | Content |
|---|---|
| `src/tinyruns/run_directory.py`, `run_config.py`, `run_inputs.py` | the run directory, its frozen config and its input runs |
| `src/tinyruns/run.py` | the stage runner: `python -m tinyruns.run <run_directory>` |
| `src/tinyruns/cluster/` | the `exp` command: snapshot, image build, submit, status, pull |
| `src/tinyruns/stages/` | stages shipped with the package |
| `src/tinyruns/templates/cluster.yaml` | a cluster profile with placeholder values |
| `apptainer/image.def` | the image definition: dependencies from `uv.lock`, no project code |
