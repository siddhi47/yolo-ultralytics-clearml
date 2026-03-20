"""
Job 1 — Ingest CVAT project export ZIPs from S3.

Downloads each ZIP from s3://<bucket>/<zip_prefix>, extracts it, and re-uploads
the organised structure to s3://<bucket>/raw/<project_name>/task_N/.

Processed ZIPs are tracked via marker files at:
  <markers_prefix>/<zip_key>.done   (content = S3 ETag of the ZIP)

On re-run, a ZIP is skipped if its marker exists AND the ETag matches.
If the same ZIP filename is re-uploaded with new content, the ETag changes
and the ZIP is re-ingested automatically.

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
    python ingest.py ingest.zip_prefix=exports/
    python ingest.py ingest.tmp_dir=/dev/shm
"""

import io
import json
import os
import tempfile
import zipfile

import boto3
from clearml import Task
from hydra import main as hydra_main
from omegaconf import DictConfig
from tqdm import tqdm

from yolo_training.log import get_logger
from yolo_training.s3_ops import download_file, list_zip_keys, upload_file, upload_fileobj

log = get_logger("ingest")


def _safe_project_name(name: str) -> str:
    return name.replace(" ", "-").replace("/", "_").replace("\\", "_")


def _marker_key(markers_prefix: str, zip_key: str) -> str:
    safe = zip_key.replace("/", "__")
    return f"{markers_prefix}/{safe}.done"


def _get_zip_etag(s3, bucket: str, zip_key: str) -> str:
    resp = s3.head_object(Bucket=bucket, Key=zip_key)
    return resp["ETag"].strip('"')


def _is_already_ingested(s3, bucket: str, markers_prefix: str, zip_key: str, etag: str) -> bool:
    marker = _marker_key(markers_prefix, zip_key)
    try:
        resp = s3.get_object(Bucket=bucket, Key=marker)
        stored_etag = resp["Body"].read().decode().strip()
        return stored_etag == etag
    except s3.exceptions.NoSuchKey:
        return False
    except Exception:
        return False


def _write_marker(s3, bucket: str, markers_prefix: str, zip_key: str, etag: str):
    marker = _marker_key(markers_prefix, zip_key)
    upload_fileobj(s3, bucket, marker, io.BytesIO(etag.encode()))


def _ingest_zip(s3, bucket: str, zip_key: str, raw_prefix: str, tmp_dir: str | None):
    """Download one ZIP, extract it, upload the organised structure to S3."""
    with tempfile.TemporaryDirectory(dir=tmp_dir) as tmpdir:
        zip_path = os.path.join(tmpdir, "export.zip")
        log.info("Downloading s3://%s/%s", bucket, zip_key)
        download_file(s3, bucket, zip_key, zip_path)

        extract_dir = os.path.join(tmpdir, "extracted")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        # CVAT may wrap everything in a top-level folder — unwrap if needed
        entries = [e for e in os.listdir(extract_dir) if not e.startswith(".")]
        if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
            extract_dir = os.path.join(extract_dir, entries[0])

        project_json_path = os.path.join(extract_dir, "project.json")
        if not os.path.exists(project_json_path):
            log.warning("Skipping %s — not a CVAT project export (no project.json)", zip_key)
            return None, 0
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
                        break

        log.info("Uploaded project '%s' (%d tasks)", project_name, len(task_dirs))
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
    markers_prefix = cfg.ingest.markers_prefix
    tmp_dir = cfg.ingest.get("tmp_dir", None)

    zip_keys = list_zip_keys(s3, bucket, cfg.ingest.zip_prefix)
    if not zip_keys:
        log.warning("No ZIP files found in s3://%s/%s", bucket, cfg.ingest.zip_prefix)
        clearml_task.close()
        return

    log.info("Found %d ZIP file(s) in s3://%s/%s", len(zip_keys), bucket, cfg.ingest.zip_prefix)

    ingested, skipped = [], []
    for zip_key in tqdm(zip_keys, desc="Ingesting ZIPs"):
        etag = _get_zip_etag(s3, bucket, zip_key)
        if _is_already_ingested(s3, bucket, markers_prefix, zip_key, etag):
            log.debug("Skipping (already ingested): %s", zip_key)
            skipped.append(zip_key)
            continue

        project_name, n_tasks = _ingest_zip(s3, bucket, zip_key, cfg.ingest.raw_prefix, tmp_dir)
        if project_name is None:
            skipped.append(zip_key)
            continue
        _write_marker(s3, bucket, markers_prefix, zip_key, etag)
        ingested.append({"zip": zip_key, "project": project_name, "tasks": n_tasks})

    summary = (
        f"Ingested: {len(ingested)} | Skipped (already done): {len(skipped)}\n" +
        "\n".join(f"  {r['project']}: {r['tasks']} tasks" for r in ingested)
    )
    log.info(summary)
    clearml_task.logger.report_text(summary)
    clearml_task.close()


if __name__ == "__main__":
    ingest()
