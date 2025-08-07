from ultralytics import YOLO
from clearml import Task
from clearml import Dataset

task = Task.init(
    project_name="YOLO11n",
    task_name="Train on COCO8",
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False,
)


dataset = Dataset.get(
    dataset_project="YOLO11", dataset_name="sample", dataset_version="1.0.0"
)
local_path = dataset.get_local_copy()
model = "yolo11n"
task.set_parameter("model", model)

# Load a pretrained YOLO11n model
model = YOLO(f"{model}.pt")

args = dict(
    data="./yolo.yaml",
    epochs=1,
    imgsz=416,
    device="mps",
)
# Train the model on COCO8
task.connect(args)

results = model.train(**args)
