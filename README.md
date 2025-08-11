# YOLO Ultralytics + ClearML

This project integrates **YOLO (Ultralytics)** object detection with **ClearML** for experiment tracking, dataset management, and remote training orchestration. It provides a configurable and reproducible pipeline for training, evaluating, and managing YOLO models.

---

## Features
- 🚀 **YOLOv8-based object detection** powered by [Ultralytics](https://github.com/ultralytics/ultralytics)
- 📊 **Experiment tracking** using [ClearML](https://clear.ml)
- 🛠 **Config-driven setup** using [Hydra](https://hydra.cc)
- 📁 **Dataset versioning** with ClearML Dataset
- 🖥 **Remote execution** on ClearML Agents
- 🧪 Easy-to-reproduce training & evaluation runs

---

## Project Structure
```
yolo-ultralytics-clearml/
│
├── main.py                 # Entry point for running YOLO with ClearML integration
├── train.py                # Training script with ClearML and Hydra config
├── yolo.yaml               # YOLO configuration file (Ultralytics format)
├── conf/config.yaml        # Hydra config for dataset, model, and training parameters
├── ALLOWED_CLASS.txt       # List of allowed classes for detection
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata & dependency management
├── runs/                   # YOLO training outputs
└── LICENSE
```

---

## Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/yourusername/yolo-ultralytics-clearml.git
cd yolo-ultralytics-clearml
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## Configuration
- Modify `conf/config.yaml` to set:
  - Dataset ID or path
  - Model parameters
  - Training hyperparameters
- Update `ALLOWED_CLASS.txt` if you want to restrict detection to certain classes.

---

## Usage

### Local Training
```bash
python train.py
```

### Run with Hydra overrides
```bash
python train.py dataset.name=my_dataset model.name=yolov8n
```

### Remote Execution on ClearML Agent
```bash
clearml-agent daemon --queue default
```
Then enqueue the task from your ClearML dashboard or via:
```python
from clearml import Task
task = Task.init(project_name="YOLO", task_name="remote_train", task_type=Task.TaskTypes.training)
```

---

## Tracking with ClearML
1. Sign up at [https://clear.ml](https://clear.ml)
2. Install ClearML and run:
```bash
clearml-init
```
3. All training runs, metrics, and artifacts will be automatically logged.

---

## Dataset Management
- Upload a dataset to ClearML:
```bash
clearml-data create --project "YOLO Dataset" --name "v1"
clearml-data add --files /path/to/data
clearml-data close
```
- Use dataset in `config.yaml` with its ClearML dataset ID.

---

## Results
YOLO training results (precision-recall curves, confusion matrix, sample predictions) are stored in:
```
runs/detect/trainX/
```
and also logged to the ClearML dashboard.

---

## License
This project is licensed under the MIT License.

