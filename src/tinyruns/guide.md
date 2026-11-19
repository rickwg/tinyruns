---
name: tinyruns
description: How to run experiments with the `tinyruns` command, as run directories on this machine or as Slurm jobs inside an Apptainer image. Walks from setting up a project to a full run, and covers every command, stages, configs, checkpoints, input runs, datasets, the cluster profile, the image and the guardrails. Use in a project that depends on the `tinyruns` package.
---

# Getting started with tinyruns

A run is one directory, `artifacts/runs/<run_id>/`. `tinyruns` creates it from a config, executes it as a Slurm job inside your project's Apptainer image, and copies it back. The same directory runs unchanged on your own machine.

This guide goes from an empty project to a full run on the cluster, in the order you would do it. The tables at the end list every command and every file.

Run the commands from inside the project, as `uv run tinyruns <command>`. Every command prints JSON. A failure exits non-zero with `{"error": "..."}` on stderr. Only `tinyruns guide` prints text: this guide.

## 1. Set up the project

The project is a git repository managed by uv, with its code as an installable package under `src/` and a `.python-version` file. `uv init --package` creates one.

1. **Depend on tinyruns:** `uv add --raw "tinyruns @ git+https://github.com/rickwg/tinyruns@<tag>"`. `--raw` writes the reference into `dependencies`, where pip reads it too.
2. **Describe the cluster:** `uv run tinyruns init cluster` writes `.cluster.yaml`. Replace its values with your cluster's; [the cluster profile](#the-cluster-profile) explains them. The file may stay out of git: a line `.cluster.yaml` in `.git/info/exclude` ignores it and leaves no trace in the repository.
3. **Define the image:** `uv run tinyruns init image` writes `apptainer/image.def` for the project's Python version. The file is yours to change from then on.
4. **Ignore the runs:** add `artifacts/` to `.gitignore`.
5. **Commit.** Every run records the commit it was made from, so the repository needs one.
6. **Check the cluster:** `uv run tinyruns check` queues a short job that reports what a job sees and which URLs it reaches. Fetch its `report.md` with `wait` and `pull`, as in [step 5](#5-run-it-on-the-cluster). On a cluster with GPUs, `uv run tinyruns check --gpu --resources <name>` fails unless the job sees a CUDA device; it needs PyTorch among the project's dependencies.

Your machine needs `git`, `ssh` and `rsync`, and a login to the cluster that works without a prompt. The cluster needs Slurm, `rsync` and Apptainer with `apptainer build --fakeroot`.

## 2. Write a stage

A stage is one step of a run, such as training or evaluation. It is a module under `src/` with a function `run_stage(run_directory, config)`:

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

## 3. Write a config

```yaml
stages:
  - my_project.stages.train
  - my_project.stages.evaluate
train:
  steps: 1000
```

- **`stages`** lists full module paths, which run in order. Every other top-level key is the project's own.
- **A config names no location.** A value that starts with `/` or `~` is rejected.
- **A config names no cluster resources.** They are chosen with `--resources` and recorded in the run's manifest.

## 4. Try it on your machine

```bash
uv run tinyruns new configs/train.yaml --smoke        # prints the run_id, and the next command as run_with
uv run python -m tinyruns.run artifacts/runs/<run_id>
```

`tinyruns new` creates the run directory without contacting the cluster. `--stage <name>` on the second command runs one stage. A run made this way does not count as the smoke run that a full run needs.

## 5. Run it on the cluster

```bash
uv run tinyruns submit configs/train.yaml --smoke          # prints the manifest with the run_id
uv run tinyruns wait <run_id>
uv run tinyruns pull <run_id>
git commit
uv run tinyruns submit configs/train.yaml --resources gpu-1
uv run tinyruns wait <run_id>
uv run tinyruns pull <run_id>
```

- **A smoke run** tests the working tree, committed or not, with a short job on the profile's `smoke_resources`. It builds the image if `uv.lock` or `apptainer/image.def` changed, and it is the run that downloads models into the shared cache.
- **A full run** needs `--resources`, committed code, and a smoke run of the same code and config that finished on the cluster. Committing does not invalidate the smoke run. Changing `src`, `pyproject.toml`, `uv.lock`, `apptainer`, the config or its inputs does.
- **`tinyruns wait`** blocks until the run ends, with exit code 0 for done, 1 for failed and 2 for still running after `--timeout <seconds>`. Run it in the background; do not poll `status` in a loop.
- **`tinyruns pull`** copies the run directory back, without `heavy/` unless `--heavy` is given. A pulled run reruns on your machine with the command from step 4.
- **`--confirm` and `--skip-smoke-check` override guardrails:** the GPU-hour limit of the profile, and the smoke run before a full run. They are a human's decision. An agent passes them only when the human has said so.

## 6. Watch and repair runs

- **`tinyruns status <run_id>`** asks the cluster for the run's state. Without a run ID it reports every unfinished run. A failed run's status holds the reason and the end of its log.
- **`tinyruns logs <run_id>`** shows the end of the Slurm log, `--lines <n>` of it.
- **`tinyruns ls`** lists the runs known on your machine, without asking the cluster.
- **`tinyruns cancel <run_id>`** cancels the job and returns when Slurm has ended it. The run is then failed.
- **`tinyruns resume <run_id>`** queues a failed run again, in its directory, with the same code and config. Its stages start again from the first; [step 7](#7-continue-a-run-from-a-checkpoint) lets a stage continue where it stopped.

## 7. Continue a run from a checkpoint

tinyruns saves no checkpoints. A stage that should continue where it stopped saves its own into `heavy/`, and loads the last one when it starts:

```python
import os
from typing import Any

import torch

from my_project.model import new_model, training_loss
from tinyruns.run_directory import RunDirectory, append_metrics


def run_stage(run_directory: RunDirectory, config: dict[str, Any]) -> None:
    settings = config["train"]
    run_directory.heavy_directory.mkdir(exist_ok=True)
    checkpoint_path = run_directory.heavy_directory / "checkpoint.pt"
    partial_path = run_directory.heavy_directory / "checkpoint.pt.partial"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = new_model(settings).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    steps_done = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        steps_done = checkpoint["steps_done"]
    while steps_done < settings["steps"]:
        loss = training_loss(model, steps_done)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps_done += 1
        if steps_done % settings["checkpoint_every"] == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "steps_done": steps_done,
            }
            torch.save(checkpoint, partial_path)
            os.replace(partial_path, checkpoint_path)
            append_metrics(run_directory, {"step": steps_done, "loss": loss.item()})
```

`new_model` and `training_loss` stand for the project's own code.

- **A job that Slurm ended is a failed run.** A run that reached its time limit or was cancelled resumes like one that raised an error.
- **A resumed run keeps its directory,** with what the stages wrote into `heavy/`, and runs the same code and config in the same image.
- **Every stage runs again.** tinyruns does not record which stages finished, so each stage skips the work it finds done. The stage above finds its last checkpoint and trains no further.
- **Replace a checkpoint in one step.** Slurm can end the job in the middle of a write, and `os.replace` leaves the old checkpoint or the new one, never half of one.
- **Save all that the loop needs.** Beyond the model, the optimizer and the step count, that can be a learning-rate schedule, and the random state if the continued run must match an uninterrupted one.
- **Load onto the device at hand.** With `map_location`, a checkpoint saved on a GPU also loads where there is none, such as in a pulled run on your machine.
- **Record metrics when a checkpoint is saved.** `metrics.jsonl` is only appended to, so steps recorded after the last checkpoint are recorded again by the next attempt.
- **A resumed job gets the same resources,** the time limit and the memory among them. A training longer than the time limit is resumed until it is done; nothing queues it again by itself. A run that needs more memory is submitted anew with other `--resources`, and starts from the beginning.
- **Try it on your machine first.** `tinyruns resume` takes only runs on the cluster. For a run made by `tinyruns new`, interrupt the command of [step 4](#4-try-it-on-your-machine) and give it again: the stage must continue from its checkpoint. A smoke run tests this only if it is long enough to save one.

## 8. Read another run

A run reads the outputs of a finished run through a named input, so one training run can be evaluated many times.

- **Name the run** with `--input <name>=<run_id>` on `tinyruns submit` or `tinyruns new`, or under `inputs:` in the config.
- **In the stage,** `input_run(run_directory, config, "<name>")` from `tinyruns.run_inputs` returns that run's `RunDirectory`.
- **The input run must be done.** The job gets it read-only. A full run cannot read a smoke run.
- **For a local rerun, pull the input run too:** `tinyruns pull <input_run_id> --heavy`.
- **Runs do not cross projects.** An input is a sibling directory of the run that reads it.

## 9. Read a dataset

A model or a dataset on the Hugging Face Hub needs no place of its own: `HF_HOME` is a cache that the project's runs share, a smoke run downloads into it, and a full run reads it offline. A dataset from anywhere else goes into the data directory.

- **On your machine** the data directory is `artifacts/data/`, beside `artifacts/runs/`. Give each dataset a directory of its own in it.
- **In the stage,** `data_directory(run_directory)` from `tinyruns.run_data` returns the data directory, on your machine and in a job alike: `data_directory(run_directory) / "my_dataset"`.
- **`tinyruns push-data`** copies `artifacts/data/` to the cluster. It adds and replaces files and deletes none, so a dataset that is only on the cluster stays there.
- **On the cluster** the data directory is `data/` under the profile's `scratch_root`. A dataset too large for your machine can be put there directly.
- **A job gets the data directory read-only.** What a run makes of a dataset goes into its own `heavy/`, and other runs read it as an [input](#8-read-another-run).
- **A config names a dataset by its directory name,** never by a path.
- **A run does not record its datasets.** The manifest holds the code and the config, not the content of the data directory. Put a version into the directory name of a dataset that can change.

## Every command

| Command | Effect |
|---|---|
| `tinyruns init cluster` | Writes the cluster profile `.cluster.yaml` from a template. Does not overwrite. |
| `tinyruns init image` | Writes the image definition `apptainer/image.def` from a template. Does not overwrite. |
| `tinyruns check` | Queues a smoke run that reports what a job on the cluster sees and which URLs it reaches. `--gpu` fails it unless a CUDA device is visible. |
| `tinyruns push-data` | Copies the datasets in `artifacts/data/` to the cluster. Deletes nothing there. |
| `tinyruns new <config>` | Creates a run directory on this machine, without the cluster. `--smoke` freezes it as a short test run. |
| `tinyruns submit <config>` | Copies the code to the cluster, builds the image if needed, and queues the run. `--smoke` for a smoke run, `--resources <name>` for a full run. |
| `tinyruns wait <run_id>` | Blocks until the run ends. |
| `tinyruns pull <run_id>` | Copies the run directory back. |
| `tinyruns status [<run_id>]` | Asks the cluster for the state of one run, or of every unfinished run. |
| `tinyruns logs <run_id>` | Shows the end of the run's Slurm log. |
| `tinyruns ls` | Lists the runs known on this machine. |
| `tinyruns cancel <run_id>` | Cancels the run's Slurm job and waits until it has ended. |
| `tinyruns resume <run_id>` | Queues a failed run again, in its directory. |
| `tinyruns guide` | Prints this guide. |
| `python -m tinyruns.run <run_directory>` | Runs the stages of a run directory, on this machine or in the job. |

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

## The cluster profile

`.cluster.yaml` at the root of the project describes one cluster. `--cluster <path>` on `submit`, `check` and `push-data` reads another file. The profile is not part of a run's recorded code, so editing it invalidates no smoke run.

| Setting | Meaning |
|---|---|
| `name` | recorded in each run's manifest |
| `ssh_host` | anything `ssh` accepts. Without it, the commands run on the machine `tinyruns` is on, for use on a login node. |
| `code_root`, `scratch_root` | where the code of each run goes, and where runs, images, datasets and caches go. `{user}` is the login name, `{project}` the name in `pyproject.toml`. |
| `node_local_scratch` | optional directory on a node's own disk for building images |
| `environment` | optional variables, such as a proxy, set in every job and image build |
| `image_build` | partition, time, CPUs and memory of the job that builds an image |
| `resources` | named sets of `partition`, `time`, `cpus` and `memory`, optionally `gpus` and `constraint` |
| `smoke_resources` | optional name of the resources a smoke run gets without `--resources` |
| `gpu_hours_without_confirmation` | a run reserving more, GPUs times time limit, needs `--confirm` |

## The image

`apptainer/image.def` belongs to the project. `tinyruns` never edits it, and builds it with `apptainer build --fakeroot` in a Slurm job, next to copies of `uv.lock` and `pyproject.toml`.

- **The image is built again** when `apptainer/image.def` or `uv.lock` changes, and reused otherwise.
- **Any definition works if** `python` on its `PATH` has the locked dependencies, `tinyruns` among them, and the directories `/workdir`, `/mnt/runs`, `/mnt/data` and `/mnt/cache/hf` exist.
- **In a job,** the code is at `/workdir` read-only with `PYTHONPATH=/workdir/src`, the run directory is under `/mnt/runs`, the data directory is `/mnt/data` read-only, and `HF_HOME` is `/mnt/cache/hf`, shared by the project's runs. A full run sets `HF_HUB_OFFLINE=1`.
- **A lock that takes a package from a path on your machine is refused,** because the build on the cluster cannot reach it.
