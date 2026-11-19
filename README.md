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

## Guide

The [guide](src/tinyruns/guide.md) goes from an empty project to a full run on the cluster.

## Status

Early. It runs the experiments of one project on one Slurm cluster, and its own tests use a simulated cluster. Its interface may still change.

## Licence

MIT, see [LICENSE](LICENSE).
