from ultralytics import YOLO
from clearml import Task, Dataset
from hydra import main
from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
import os


@main(config_path="./conf", config_name="config.yaml")
def train(cfg: DictConfig):
    # Initialize ClearML Task
    task = Task.init(
        project_name=cfg.task.project_name,
        task_name=cfg.task.task_name,
        reuse_last_task_id=cfg.task.reuse_last_task_id,
    )

    # Download dataset
    dataset = Dataset.get(
        dataset_project=cfg.dataset.project,
        dataset_name=cfg.dataset.name,
        dataset_version=cfg.dataset.version,
    )
    local_path = dataset.get_mutable_local_copy(cfg.dataset.output_dir, overwrite=True)

    # Log model name as parameter
    task.set_parameter("model", cfg.training.model_name)

    # Load YOLO model
    model = YOLO(cfg.training.model_weights)
    training_args = OmegaConf.to_container(cfg.yolo_args, resolve=True)
    training_args["data"] = os.path.join(get_original_cwd(), training_args["data"])

    # Log args to ClearML
    task.connect(training_args)

    # Train
    _ = model.train(**training_args)


if __name__ == "__main__":
    train()
