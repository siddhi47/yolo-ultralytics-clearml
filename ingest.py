"""
Job 1 — Ingest CVAT project export ZIPs from S3.

Downloads each ZIP from s3://<bucket>/<zip_prefix>, extracts it, and re-uploads
the organised structure to s3://<bucket>/raw/<project_name>/task_N/.

S3 output layout:
  raw/
    <project-name>/
      project.json
      task_0/
        task.json
        annotations.json
        video.mp4
      task_1/
        ...

Run:
    python ingest.py
    python ingest.py ingest.zip_prefix=exports/  # override any hydra key
"""

import json
import os
import tempfile
import zipfile

import boto3
from clearml import Task
from hydra import main as hydra_main
from omegaconf import DictConfig
from tqdm import tqdm

from yolo_training.s3_ops import download_file, list_zip_keys, upload_file


def _safe_project_name(name: str) -> str:
    return name.replace(" ", "-").replace("/", "_").replace("\\", "_")


def _ingest_zip(s3, bucket: str, zip_key: str, raw_prefix: str):
    """Download one ZIP, extract it, upload the organised structure to S3."""
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "export.zip")
        print(f"  Downloading s3://{bucket}/{zip_key}")
        download_file(s3, bucket, zip_key, zip_path)

        extract_dir = os.path.join(tmpdir, "extracted")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        # CVAT may wrap everything in a top-level folder — unwrap if needed
        entries = [e for e in os.listdir(extract_dir) if not e.startswith(".")]
        if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
            extract_dir = os.path.join(extract_dir, entries[0])

        # Read project name
        project_json_path = os.path.join(extract_dir, "project.json")
        with open(project_json_path) as f:
            project_meta = json.load(f)
        project_name = _safe_project_name(project_meta["name"])
        dest_prefix = f"{raw_prefix}/{project_name}"

        upload_file(s3, bucket, f"{dest_prefix}/project.json", project_json_path)

        task_dirs = sorted(
            d for d in os.listdir(extract_dir)
            if d.startswith("task_") and os.path.isdir(os.path.join(extract_dir, d))
        )
        for task_dir_name in task_dirs:
            task_dir = os.path.join(extract_dir, task_dir_name)
            task_s3_prefix = f"{dest_prefix}/{task_dir_name}"

            for fname in ("task.json", "annotations.json"):
                fpath = os.path.join(task_dir, fname)
                if os.path.exists(fpath):
                    upload_file(s3, bucket, f"{task_s3_prefix}/{fname}", fpath)

            data_dir = os.path.join(task_dir, "data")
            if os.path.exists(data_dir):
                for fname in os.listdir(data_dir):
                    if fname.endswith(".mp4"):
                        upload_file(
                            s3, bucket,
                            f"{task_s3_prefix}/video.mp4",
                            os.path.join(data_dir, fname),
                        )
                        break  # one video per task

        print(f"  Uploaded project '{project_name}' ({len(task_dirs)} tasks)")
        return project_name, len(task_dirs)


@hydra_main(config_path="./conf", config_name="config.yaml", version_base=None)
def ingest(cfg: DictConfig):
    clearml_task = Task.init(
        project_name=cfg.task.project_name,
        task_name="Ingest CVAT exports",
        task_type=Task.TaskTypes.data_processing,
        reuse_last_task_id=False,
    )

    s3 = boto3.client("s3")
    bucket = cfg.ingest.bucket
    zip_keys = list_zip_keys(s3, bucket, cfg.ingest.zip_prefix)

    if not zip_keys:
        print(f"No ZIP files found in s3://{bucket}/{cfg.ingest.zip_prefix}")
        clearml_task.close()
        return

    print(f"Found {len(zip_keys)} ZIP file(s) in s3://{bucket}/{cfg.ingest.zip_prefix}")

    results = []
    for zip_key in tqdm(zip_keys, desc="Ingesting ZIPs"):
        project_name, n_tasks = _ingest_zip(s3, bucket, zip_key, cfg.ingest.raw_prefix)
        results.append({"zip": zip_key, "project": project_name, "tasks": n_tasks})

    clearml_task.logger.report_text(
        f"Ingested {len(results)} project(s):\n" +
        "\n".join(f"  {r['project']}: {r['tasks']} tasks" for r in results)
    )
    clearml_task.close()


if __name__ == "__main__":
    ingest()
