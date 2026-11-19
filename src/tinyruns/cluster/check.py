from pathlib import Path

from tinyruns.cluster.errors import ExperimentError

CONFIGS_DIRECTORY = Path(__file__).parent.parent / "configs"
ENVIRONMENT_CHECK = CONFIGS_DIRECTORY / "environment_check.yaml"
GPU_ENVIRONMENT_CHECK = CONFIGS_DIRECTORY / "environment_check_gpu.yaml"


def environment_check_config(gpu: bool, resources: str | None) -> Path:
    if not gpu:
        return ENVIRONMENT_CHECK
    if resources is None:
        raise ExperimentError(
            "`tinyruns check --gpu` needs --resources: the name of resources with a "
            "GPU in the cluster profile."
        )
    return GPU_ENVIRONMENT_CHECK
