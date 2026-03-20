"""
Job 3 — Train YOLO using a versioned ClearML Dataset (S3-backed).

Steps:
  1. Pull the ClearML Dataset (downloads files from S3 to local ./data/)
  2. Generate yolo.yaml pointing to the local data directory
  3. Train YOLO — all metrics/checkpoints are auto-logged to ClearML
  4. Register the best model weights as a ClearML artifact stored in S3

Run:
    python train.py
    python train.py yolo_args.epochs=100 dataset.version=1.1.0
"""

import os

import yaml
from clearml import Dataset, OutputModel, Task
from hydra import main as hydra_main
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from ultralytics import YOLO


def _generate_yolo_yaml(data_dir: str, allowed_classes: list[str]) -> str:
    """Write a yolo.yaml into data_dir and return its absolute path."""
    config = {
        "path": os.path.abspath(data_dir),
        "train": "train/images",
        "val": "val/images",
        "names": {idx: name for idx, name in enumerate(allowed_classes)},
    }
    yaml_path = os.path.join(data_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return yaml_path


@hydra_main(config_path="./conf", config_name="config.yaml", version_base=None)
def train(cfg: DictConfig):
    task = Task.init(
        project_name=cfg.task.project_name,
        task_name=cfg.task.task_name,
        reuse_last_task_id=cfg.task.reuse_last_task_id,
        output_uri=cfg.training.output_uri,
    )

    # Download dataset from ClearML (files pulled from S3)
    dataset = Dataset.get(
        dataset_project=cfg.dataset.project,
        dataset_name=cfg.dataset.name,
        dataset_version=cfg.dataset.version,
    )
    local_data_dir = dataset.get_mutable_local_copy(cfg.dataset.output_dir, overwrite=True)

    # Build yolo.yaml dynamically so it always points to the right local paths
    allowed_classes = list(cfg.sample.allowed_classes)
    yaml_path = _generate_yolo_yaml(local_data_dir, allowed_classes)

    # Assemble YOLO training args
    training_args = OmegaConf.to_container(cfg.yolo_args, resolve=True)
    training_args["data"] = yaml_path
    training_args["project"] = os.path.join(get_original_cwd(), "runs")
    training_args["name"] = cfg.training.model_name

    task.set_parameter("model", cfg.training.model_name)
    task.set_parameter("dataset_version", cfg.dataset.version)
    task.connect(training_args)

    # Train
    model = YOLO(cfg.training.model_weights)
    result = model.train(**training_args)

    # Register best weights as a ClearML artifact (stored in S3 via output_uri)
    best_weights = result.save_dir / "weights" / "best.pt"
    if best_weights.exists():
        output_model = OutputModel(task=task, framework="PyTorch")
        output_model.update_weights(str(best_weights), auto_delete_file=False)
        output_model.update_design(config_dict=training_args)
        print(f"Model registered: {best_weights}")

    task.close()


if __name__ == "__main__":
    train()
