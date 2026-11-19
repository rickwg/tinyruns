# tinyruns

A run is one directory, `artifacts/runs/<run_id>/`. `tinyruns` creates it from a config, executes it as a Slurm job inside the project's Apptainer image, and copies it back. The same directory runs unchanged on this machine.

Run the commands from inside the project, as `uv run tinyruns <command>`. Every command prints JSON. A failure exits non-zero with `{"error": "..."}` on stderr. Only `tinyruns guide` prints text: this guide.

## The usual sequence

```bash
uv run tinyruns submit configs/train.yaml --smoke          # prints the manifest with the run_id
uv run tinyruns wait <run_id>                              # in the background, not in a polling loop
uv run tinyruns pull <run_id>
git commit                                                 # a full run needs committed code
uv run tinyruns submit configs/train.yaml --resources gpu-1
uv run tinyruns wait <run_id>
uv run tinyruns pull <run_id>
```

- **A smoke run** tests the working tree, committed or not, with a short job. It is also the run that downloads models into the shared cache.
- **A full run** needs `--resources`, committed code, and a smoke run of the same code and config that finished on the cluster. Committing does not invalidate the smoke run. Changing `src`, `pyproject.toml`, `uv.lock`, `apptainer`, the config or its inputs does.
- **`--confirm` and `--skip-smoke-check` override guardrails.** Pass them only when the human has said so.

## Commands

| Command | Effect |
|---|---|
| `tinyruns new <config> [--smoke]` | Creates a run directory on this machine without contacting the cluster, and prints the command that runs it as `run_with`. Such a run does not count as the smoke run that a full run needs. |
| `tinyruns submit <config> --smoke` | Copies the working tree's code to the cluster, builds the image if `uv.lock` or `apptainer/image.def` changed, and queues a short run on the profile's `smoke_resources`. |
| `tinyruns submit <config> --resources <name>` | Queues a full run on the named resources of the cluster profile. |
| `tinyruns wait <run_id>` | Blocks until the run ends. Exit code 0 for done, 1 for failed, 2 for still running after `--timeout <seconds>`. |
| `tinyruns pull <run_id>` | Copies the run directory back, without `heavy/` unless `--heavy` is given. |
| `tinyruns status [<run_id>]` | Asks the cluster for the state of one run, or of every unfinished run. |
| `tinyruns logs <run_id>` | Shows the end of the run's Slurm log. |
| `tinyruns ls` | Lists the runs known on this machine, without asking the cluster. |
| `tinyruns cancel <run_id>` | Cancels the run's Slurm job. |
| `tinyruns resume <run_id>` | Queues a failed run again, in its directory. |
| `tinyruns check` | Queues a smoke run that reports what a job on the cluster sees and which URLs it reaches. |
| `tinyruns init cluster`, `tinyruns init image` | Write `.cluster.yaml` and `apptainer/image.def` from templates. Neither overwrites a file. |
| `tinyruns guide` | Prints this guide. |

`uv run python -m tinyruns.run artifacts/runs/<run_id>` runs a run directory on this machine, whether `tinyruns new` created it or `tinyruns pull` fetched it. `--stage <name>` runs one of its stages.

## A run directory

| Path | Content |
|---|---|
| `config.yaml` | the config as frozen when the run was created |
| `manifest.json` | the code's commit and tree, the tinyruns version, the resources, the cluster, the image and the Slurm job IDs |
| `status.json` | the state: `created`, `queued`, `running`, `done` or `failed`, with the failure if there is one |
| `slurm.out` | the job's log |
| `code.patch` | the uncommitted changes the run included, if any |
| `metrics.jsonl`, `report.md`, `figures/` | what the stages write |
| `heavy/` | large outputs such as checkpoints, left on the cluster unless pulled with `--heavy` |

## Stages

A stage is a module under `src/` with a function `run_stage(run_directory, config)`:

```python
from typing import Any

from tinyruns.run_config import is_smoke_test
from tinyruns.run_directory import RunDirectory, append_metrics


def run_stage(run_directory: RunDirectory, config: dict[str, Any]) -> None:
    settings = config["train"]
    steps = 10 if is_smoke_test(config) else settings["steps"]
    append_metrics(run_directory, {"steps": steps})
    run_directory.report_path.write_text(f"# Training\n\n{steps} steps.\n")
```

- **A stage writes only inside the run directory** and records paths relative to it, so the run works wherever its directory is copied.
- **`RunDirectory`** has `path`, `metrics_path`, `report_path`, `figures_directory` and `heavy_directory`. The two directories are not created for the stage.
- **`is_smoke_test(config)`** is true in a smoke run. Use it to shorten the work, not to change what is tested.
- **Everything a job executes must be under `src/`.** Only `src`, `pyproject.toml`, `uv.lock` and `apptainer` reach the cluster.
- **Keep a stage thin.** It reads its section of the config, calls library code and writes into the run directory.

## Configs

```yaml
stages:
  - my_project.stages.train
  - my_project.stages.evaluate
train:
  steps: 1000
```

- **`stages`** lists full module paths, which run in order. Every other top-level key is the project's own.
- **A config names no location.** A value that starts with `/` or `~` is rejected.
- **A config names no cluster resources.** They are chosen with `--resources` and recorded in the manifest.

## Reading another run

A run reads the outputs of a finished run through a named input, so one training run can be evaluated many times.

- **Name the run** with `--input <name>=<run_id>` on `tinyruns submit` or `tinyruns new`, or under `inputs:` in the config.
- **In the stage,** `input_run(run_directory, config, "<name>")` from `tinyruns.run_inputs` returns that run's `RunDirectory`.
- **The input run must be done.** The job gets it read-only. A full run cannot read a smoke run.
- **For a local rerun, pull the input run too:** `tinyruns pull <input_run_id> --heavy`.
- **Runs do not cross projects.** An input is a sibling directory of the run that reads it.

## Setting up a project

The project is a git repository managed by uv, with its code as an installable package under `src/` and a `.python-version` file.

1. `uv add --raw "tinyruns @ git+https://github.com/rickwg/tinyruns@<tag>"`. `--raw` writes the reference into `dependencies`, where pip reads it too.
2. `uv run tinyruns init cluster`, then replace the values in `.cluster.yaml` with the cluster's. The file may stay out of git: a line `.cluster.yaml` in `.git/info/exclude` ignores it without a trace in the repository.
3. `uv run tinyruns init image`, then commit `apptainer/image.def`.
4. Add `artifacts/` to `.gitignore`.
5. `uv run tinyruns check`, then `wait` and `pull` the run and read its `report.md`. On a cluster with GPUs, `uv run tinyruns check --gpu --resources <name>` fails unless the job sees a CUDA device; it needs PyTorch among the project's dependencies.

The machine that submits needs `git`, `ssh` and `rsync`, and a login to the cluster that works without a prompt. The cluster needs Slurm, `rsync` and Apptainer with `apptainer build --fakeroot`.

## The cluster profile

`.cluster.yaml` at the root of the project describes one cluster. `--cluster <path>` on `submit` and `check` reads another file. The profile is not part of a run's recorded code, so editing it invalidates no smoke run.

| Setting | Meaning |
|---|---|
| `name` | recorded in each run's manifest |
| `ssh_host` | anything `ssh` accepts. Without it, the commands run on the machine `tinyruns` is on, for use on a login node. |
| `code_root`, `scratch_root` | where the code of each run goes, and where runs, images and caches go. `{user}` is the login name, `{project}` the name in `pyproject.toml`. |
| `node_local_scratch` | optional directory on a node's own disk for building images |
| `environment` | optional variables, such as a proxy, set in every job and image build |
| `image_build` | partition, time, CPUs and memory of the job that builds an image |
| `resources` | named sets of `partition`, `time`, `cpus` and `memory`, optionally `gpus` and `constraint` |
| `smoke_resources` | optional name of the resources a smoke run gets without `--resources` |
| `gpu_hours_without_confirmation` | a run reserving more, GPUs times time limit, needs `--confirm` |

## The image

`apptainer/image.def` belongs to the project. `tinyruns` never edits it, and builds it with `apptainer build --fakeroot` in a Slurm job, next to copies of `uv.lock` and `pyproject.toml`.

- **The image is built again** when `apptainer/image.def` or `uv.lock` changes, and reused otherwise.
- **Any definition works if** `python` on its `PATH` has the locked dependencies, `tinyruns` among them, and the directories `/workdir`, `/mnt/runs` and `/mnt/cache/hf` exist.
- **In a job,** the code is at `/workdir` read-only with `PYTHONPATH=/workdir/src`, the run directory is under `/mnt/runs`, and `HF_HOME` is `/mnt/cache/hf`, shared by the project's runs. A full run sets `HF_HUB_OFFLINE=1`.
- **A lock that takes a package from a path on this machine is refused,** because the build on the cluster cannot reach it.
