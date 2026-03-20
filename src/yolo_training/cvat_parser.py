import json
from collections import defaultdict


def parse_annotations(annotations_path: str) -> dict:
    """
    Parse CVAT native JSON export (annotations.json) into a per-frame dict.

    CVAT exports tracks with interpolated shapes for every frame. Each shape has:
      - points: [x1, y1, x2, y2] in absolute pixel coordinates
      - outside: True if object is not visible in this frame (skip)

    Returns:
        {frame_num: [(label_name, x1, y1, x2, y2), ...]}
        Frames with no visible objects are NOT included (caller handles empty label files).
    """
    with open(annotations_path) as f:
        jobs = json.load(f)

    frame_annotations = defaultdict(list)

    for job in jobs:
        # Tracks: interpolated object trajectories across frames
        for track in job.get("tracks", []):
            label = track["label"]
            for shape in track.get("shapes", []):
                if shape.get("outside", False):
                    continue
                if shape.get("type") != "rectangle":
                    continue
                pts = shape["points"]  # [x1, y1, x2, y2]
                frame_num = shape["frame"]
                frame_annotations[frame_num].append((label, pts[0], pts[1], pts[2], pts[3]))

        # Standalone shapes (not part of a track)
        for shape in job.get("shapes", []):
            if shape.get("outside", False):
                continue
            if shape.get("type") != "rectangle":
                continue
            label = shape.get("label", "")
            pts = shape["points"]
            frame_num = shape["frame"]
            frame_annotations[frame_num].append((label, pts[0], pts[1], pts[2], pts[3]))

    return dict(frame_annotations)


def to_yolo_line(label_idx: int, x1: float, y1: float, x2: float, y2: float,
                 img_w: int, img_h: int) -> str:
    """Convert absolute bbox to a YOLO label line (normalized cx cy w h)."""
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    cx, cy, w, h = [max(0.0, min(1.0, v)) for v in (cx, cy, w, h)]
    return f"{label_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
