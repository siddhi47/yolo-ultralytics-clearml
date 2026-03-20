# YOLO Ultralytics + ClearML

End-to-end YOLO training pipeline for CVAT-annotated video data. Annotations and videos are exported from CVAT as project ZIPs, stored in S3, sampled into a versioned dataset, and used to train YOLO — with all experiments, datasets, and model artifacts tracked in ClearML (backed by S3, nothing stored in ClearML's file server).

---

## Pipeline Overview

```
CVAT export ZIP → S3
       │
       ▼
  ingest.py          Download ZIPs from S3, extract, re-upload organised structure
       │
       ▼
  sample.py          Sample every Nth frame, convert CVAT JSON → YOLO format,
       │              upload images + labels to S3, create versioned ClearML Dataset
       ▼
  train.py           Pull dataset from ClearML (S3), train YOLO, register model artifact
```

---

## Project Structure

```
yolo-ultralytics-clearml/
├── ingest.py                        # Job 1: ingest CVAT export ZIPs from S3
├── sample.py                        # Job 2: sample frames, build ClearML Dataset
├── train.py                         # Job 3: train YOLO, log to ClearML
├── conf/config.yaml                 # Hydra config for all three jobs
├── src/yolo_training/
│   ├── cvat_parser.py               # Parse CVAT native JSON → per-frame annotations
│   └── s3_ops.py                    # S3 upload/download/list helpers
├── ALLOWED_CLASS.txt                # Classes to keep (others are dropped)
├── pyproject.toml                   # Project dependencies
└── LICENSE
```

---

## S3 Layout

```
s3://gt-cvat-annotations/
  ├── <project>.zip                  # CVAT project export ZIPs (uploaded manually)
  │
  ├── raw/
  │   ├── .markers/<zip>.done        # Processed ZIP markers (contains S3 ETag)
  │   └── <project-name>/
  │       ├── project.json
  │       └── task_N/
  │           ├── task.json
  │           ├── annotations.json
  │           └── video.mp4
  │
  ├── datasets/
  │   └── yolo-cvat/
  │       └── 1.0.0/
  │           ├── train/
  │           │   ├── images/
  │           │   └── labels/
  │           └── val/
  │               ├── images/
  │               └── labels/
  │
  └── clearml/                       # ClearML metadata + model artifacts
```

---

## Installation

```bash
git clone https://github.com/yourusername/yolo-ultralytics-clearml.git
cd yolo-ultralytics-clearml

# PyTorch CUDA (not on PyPI — install separately)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Everything else
pip install -e .
```

---

## Credentials

**AWS** — boto3 uses the standard credential chain:
- IAM instance role (recommended on EC2 — no credentials needed)
- `~/.aws/credentials` via `aws configure`
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`

**ClearML** — run once on each machine:
```bash
clearml-init
```

---

## Configuration

All settings are in `conf/config.yaml`. Key sections:

| Section | Purpose |
|---|---|
| `ingest` | S3 bucket, ZIP prefix, markers prefix, optional temp dir |
| `sample` | Sample rate, val ratio, allowed classes, dataset version |
| `dataset` | ClearML dataset to pull for training |
| `training` | Model weights, S3 output URI for artifacts |
| `yolo_args` | YOLO hyperparameters (epochs, batch, imgsz, device, …) |

Any key can be overridden on the command line via Hydra.

---

## Usage

### Job 1 — Ingest

Upload one or more CVAT project export ZIPs to `s3://gt-cvat-annotations/`, then run:

```bash
python ingest.py
```

Already-ingested ZIPs are skipped automatically (ETag-based). If a ZIP is re-uploaded with new content, it is re-ingested.

```bash
# Use /dev/shm as temp dir if root partition is tight
python ingest.py ingest.tmp_dir=/dev/shm
```

### Job 2 — Sample

```bash
python sample.py

# Tune sampling rate or cut a new dataset version
python sample.py sample.sample_every_n=5 sample.dataset_version=1.1.0
```

Creates a versioned ClearML Dataset whose files are S3 references — no data is copied into ClearML.

### Job 3 — Train

```bash
python train.py

# Use a specific dataset version
python train.py dataset.version=1.1.0 yolo_args.epochs=100
```

YOLO metrics are auto-logged to ClearML. The best model weights (`best.pt`) are registered as a ClearML OutputModel artifact stored in S3.

---

## Allowed Classes

Edit `ALLOWED_CLASS.txt` (one class per line) to control which CVAT labels are kept. All other labels are silently dropped during sampling.

Current defaults: `person`, `bicycle`, `car`.

---

## License

MIT License
