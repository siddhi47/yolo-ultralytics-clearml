from ultralytics import YOLO
from clearml import Task
from clearml import Dataset
import os
import shutil

task = Task.init(
    project_name="YOLO11n",
    task_name="Train on COCO8",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False,
)


dataset = Dataset.get(
    dataset_project="YOLO11", dataset_name="sample", dataset_version="1.0.0"
)

local_path = dataset.get_mutable_local_copy("./data", overwrite=True)

print(f"Dataset downloaded to {local_path}")

# move the dataset to the data directory

model = "yolo11n"
task.set_parameter("model", model)

# Load a pretrained YOLO11n model
model = YOLO(f"{model}.pt")

args = dict(
    data="./yolo.yaml",
    epochs=1,
    imgsz=416,
)
# Train the model on COCO8
task.connect(args)

results = model.train(**args)
