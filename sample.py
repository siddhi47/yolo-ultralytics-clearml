"""
Job 2 — Sample frames from ingested CVAT tasks, convert to YOLO format,
upload to S3, and register a versioned ClearML Dataset (S3-backed, no files
stored in ClearML itself).

Algorithm per task:
  1. Download task.json + annotations.json from S3 raw/
  2. Parse CVAT JSON → {frame_num: [(label, x1, y1, x2, y2)]}
  3. Sample every Nth frame across the full video range (0 … stop_frame)
     → naturally includes background frames (empty label files)
  4. Download video, extract sampled frames with OpenCV
  5. Filter annotations to allowed classes, write YOLO label files
  6. Upload images + labels to s3://<bucket>/datasets/<name>/<version>/<split>/

Train/val split: last val_ratio fraction of tasks → val (split by task, not frame).

S3 dataset layout:
  datasets/<dataset_name>/<version>/
    train/
      images/<project>_<task>_frame_000010.jpg
      labels/<project>_<task>_frame_000010.txt
    val/
      images/...
      labels/...

ClearML Dataset: created with output_uri=s3://... so all metadata also lives in S3.

Run:
    python sample.py
    python sample.py sample.sample_every_n=5 sample.dataset_version=1.1.0
"""

import io
import json
import math
import os
import tempfile

import boto3
import cv2
from clearml import Dataset, Task
from hydra import main as hydra_main
from omegaconf import DictConfig
from tqdm import tqdm

from yolo_training.cvat_parser import parse_annotations, to_yolo_line
from yolo_training.log import get_logger
from yolo_training.s3_ops import (
    download_file,
    list_immediate_prefixes,
    upload_fileobj,
)

log = get_logger("sample")


def _build_class_map(allowed_classes: list[str]) -> dict[str, int]:
    """Map class name → YOLO index for allowed classes only."""
    return {name: idx for idx, name in enumerate(allowed_classes)}


def _collect_tasks(s3, bucket: str, raw_prefix: str) -> list[tuple[str, str, str]]:
    """
    Walk raw/<project>/task_N/ and return list of (project_name, task_id, task_s3_prefix).
    """
    tasks = []
    for project_prefix in list_immediate_prefixes(s3, bucket, raw_prefix):
        project_name = project_prefix.rstrip("/").split("/")[-1]
        for task_prefix in list_immediate_prefixes(s3, bucket, project_prefix):
            task_id = task_prefix.rstrip("/").split("/")[-1]
            if task_id.startswith("task_"):
                tasks.append((project_name, task_id, task_prefix))
    return tasks


def _process_task(
    s3,
    bucket: str,
    project_name: str,
    task_id: str,
    task_s3_prefix: str,
    split: str,
    dataset_s3_prefix: str,
    class_map: dict[str, int],
    sample_every_n: int,
) -> int:
    """Download video + annotations for one task, extract sampled frames, upload to S3.

    Returns number of frames uploaded.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        ann_path = os.path.join(tmpdir, "annotations.json")
        task_json_path = os.path.join(tmpdir, "task.json")
        video_path = os.path.join(tmpdir, "video.mp4")

        log.debug("Downloading metadata for %s/%s", task_s3_prefix, task_id)
        download_file(s3, bucket, f"{task_s3_prefix}annotations.json", ann_path)
        download_file(s3, bucket, f"{task_s3_prefix}task.json", task_json_path)

        with open(task_json_path) as f:
            task_meta = json.load(f)
        stop_frame: int = task_meta["data"]["stop_frame"]

        frame_annotations = parse_annotations(ann_path)
        sampled_frames = set(range(0, stop_frame + 1, sample_every_n))
        log.debug(
            "%s/%s: stop_frame=%d, annotated=%d, sampling every %d → %d frames",
            project_name, task_id, stop_frame, len(frame_annotations),
            sample_every_n, len(sampled_frames),
        )

        log.info("Downloading video for %s/%s", project_name, task_id)
        download_file(s3, bucket, f"{task_s3_prefix}video.mp4", video_path)

        cap = cv2.VideoCapture(video_path)
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.debug("Video dimensions: %dx%d", img_w, img_h)

        s3_img_prefix = f"{dataset_s3_prefix}/{split}/images"
        s3_lbl_prefix = f"{dataset_s3_prefix}/{split}/labels"
        frame_prefix = f"{project_name}_{task_id}"

        uploaded = 0
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx in sampled_frames:
                frame_name = f"{frame_prefix}_frame_{frame_idx:06d}"

                # Upload JPEG image
                _, img_buf = cv2.imencode(".jpg", frame)
                upload_fileobj(
                    s3, bucket,
                    f"{s3_img_prefix}/{frame_name}.jpg",
                    io.BytesIO(img_buf.tobytes()),
                )

                # Build YOLO label (empty string = background frame)
                raw_annots = frame_annotations.get(frame_idx, [])
                label_lines = []
                for label, x1, y1, x2, y2 in raw_annots:
                    if label not in class_map:
                        continue
                    label_lines.append(
                        to_yolo_line(class_map[label], x1, y1, x2, y2, img_w, img_h)
                    )

                # Only upload label file if there are annotations — YOLO treats
                # a missing label file as a background image, and ClearML rejects 0-byte files.
                if label_lines:
                    upload_fileobj(
                        s3, bucket,
                        f"{s3_lbl_prefix}/{frame_name}.txt",
                        io.BytesIO("\n".join(label_lines).encode()),
                    )
                uploaded += 1

            frame_idx += 1

        cap.release()
        return uploaded


@hydra_main(config_path="./conf", config_name="config.yaml", version_base=None)
def sample(cfg: DictConfig):
    clearml_task = Task.init(
        project_name=cfg.task.project_name,
        task_name=f"Sample frames — {cfg.sample.dataset_name} v{cfg.sample.dataset_version}",
        task_type=Task.TaskTypes.data_processing,
        reuse_last_task_id=False,
        output_uri=cfg.sample.output_uri,
    )
    clearml_task.connect(dict(cfg.sample))

    s3 = boto3.client("s3")
    bucket = cfg.sample.bucket

    all_tasks = _collect_tasks(s3, bucket, cfg.sample.raw_prefix)
    if not all_tasks:
        log.warning("No tasks found under s3://%s/%s", bucket, cfg.sample.raw_prefix)
        clearml_task.close()
        return

    # Split by task: last val_ratio fraction → val
    n_val = max(1, math.ceil(len(all_tasks) * cfg.sample.val_ratio))
    val_set = {t[2] for t in all_tasks[-n_val:]}  # set of task_s3_prefix strings
    log.info("Tasks: %d total — %d train / %d val", len(all_tasks), len(all_tasks) - n_val, n_val)

    allowed_classes: list[str] = list(cfg.sample.allowed_classes)
    class_map = _build_class_map(allowed_classes)

    dataset_s3_prefix = (
        f"{cfg.sample.datasets_prefix}/{cfg.sample.dataset_name}/{cfg.sample.dataset_version}"
    )

    stats = {"train": 0, "val": 0}

    for project_name, task_id, task_prefix in tqdm(all_tasks, desc="Processing tasks"):
        split = "val" if task_prefix in val_set else "train"
        n_frames = _process_task(
            s3=s3,
            bucket=bucket,
            project_name=project_name,
            task_id=task_id,
            task_s3_prefix=task_prefix,
            split=split,
            dataset_s3_prefix=dataset_s3_prefix,
            class_map=class_map,
            sample_every_n=cfg.sample.sample_every_n,
        )
        stats[split] += n_frames
        log.info("[%s] %s/%s: %d frames uploaded", split, project_name, task_id, n_frames)

    log.info("Total — train: %d frames | val: %d frames", stats["train"], stats["val"])

    # Create versioned ClearML Dataset referencing S3 files (no upload to ClearML)
    dataset = Dataset.create(
        dataset_name=cfg.sample.dataset_name,
        dataset_project=cfg.task.project_name,
        dataset_version=cfg.sample.dataset_version,
        output_uri=cfg.sample.output_uri,
    )

    for split in ("train", "val"):
        dataset.add_external_files(
            source_url=f"s3://{bucket}/{dataset_s3_prefix}/{split}/",
            dataset_path=split,
            recursive=True,
        )

    dataset.upload()
    dataset.finalize()

    clearml_task.logger.report_text(
        f"Dataset '{cfg.sample.dataset_name}' v{cfg.sample.dataset_version} created.\n"
        f"train: {stats['train']} frames | val: {stats['val']} frames\n"
        f"S3 path: s3://{bucket}/{dataset_s3_prefix}"
    )
    clearml_task.close()


if __name__ == "__main__":
    sample()
