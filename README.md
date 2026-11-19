# tinyruns

Run directories for experiments on a Slurm cluster inside an Apptainer image.

Without tinyruns, one experiment on the cluster is a series of manual steps on two machines, for example:

```bash
rsync -a ./ my-cluster:project/                 # copy the code
ssh my-cluster
cd project
apptainer build image.sif image.def             # again whenever dependencies change
nano train.sbatch                               # partition, time limit, GPUs, bind mounts, paths
sbatch train.sbatch
squeue --me                                     # repeat until the job has ended
exit
rsync -a my-cluster:project/outputs/ outputs/   # copy the results back
```

With tinyruns, it is three commands on your own machine:

```bash
uv run tinyruns submit configs/train.yaml --resources gpu-1
uv run tinyruns wait <run_id>
uv run tinyruns pull <run_id>
```

Before a full run, tinyruns asks for a short smoke run of the same code and config, `submit --smoke`, so a typo fails in a minute and not at the end of a queue.

A run is one directory. It holds the frozen config, the commit of the code, the job's log and everything the experiment wrote, and it runs again unchanged on your own machine:

```bash
uv run python -m tinyruns.run artifacts/runs/<run_id>
```

## Start

Your project is a git repository managed by [uv](https://docs.astral.sh/uv/), with its code as a package under `src/`. The cluster runs Slurm and Apptainer.

```bash
uv add --raw "tinyruns @ git+https://github.com/rickwg/tinyruns@<tag>"
uv run tinyruns init cluster    # writes .cluster.yaml: describe your cluster in it
uv run tinyruns init image      # writes apptainer/image.def, the image your jobs run in
uv run tinyruns check           # queues a job that reports what the cluster provides
```

`uv run tinyruns guide` prints the rest: every command, the run directory, and how to write a stage and a config.

The guide is also a [Claude Code skill](src/tinyruns/skill/SKILL.md). From a checkout of this repository, link it with:

```bash
ln -s "$PWD/src/tinyruns/skill" ~/.claude/skills/tinyruns
```

## Status

Early. An earlier form of this code ran on one Slurm cluster. This version is tested against a simulated cluster only, and its interface may change before `v0.1.0`.

## Licence

MIT, see [LICENSE](LICENSE).
