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
uv run tinyruns submit configs/train.yaml
uv run tinyruns wait <run_id>
uv run tinyruns pull <run_id>
```


## Status

Early, and not yet usable from another project. The code was written inside one research project and is being generalised:

- Stage modules are still looked up inside this package (`tinyruns.stages`), not in the project that uses it.
- The environment-check stage needs PyTorch, which is only a development dependency here.
- There are no commands yet to create a cluster profile or an image definition from a template.
