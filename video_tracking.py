"""Sample continuous video and associate pole detections across its frames."""

import math
from collections import Counter, defaultdict

import cv2
import numpy as np


VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm"}
VIDEO_MAX_SECONDS = 60
VIDEO_TARGET_FPS = 5
EQUIPMENT_THRESHOLD = {"antenna": 0.25, "rru": 0.25, "microwave": 0.35}


def extract_video_frames(path, folder, *, max_frames=100, target_fps=VIDEO_TARGET_FPS, start=0.0, end=None):
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or frame_count < 1:
            raise ValueError("Не удалось прочитать видеодорожку")
        duration = frame_count / fps
        end = duration if end is None else min(end, duration)
        if not 0 <= start < end or end - start > VIDEO_MAX_SECONDS + 0.01:
            raise ValueError(f"Выберите видео длительностью до {VIDEO_MAX_SECONDS} секунд")
        stride = max(1, math.ceil(fps / target_fps), math.ceil((end - start) * fps / max_frames))
        first = math.ceil(start * fps)
        last = min(frame_count, math.ceil(end * fps))
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        manifest = {}
        for index in range(first, last):
            if not capture.grab():
                if index < last - 1:
                    raise ValueError("Видео обрывается до конца записи")
                break
            if (index - first) % stride:
                continue
            ok, image = capture.retrieve()
            if not ok:
                raise ValueError("Не удалось декодировать кадр видео")
            height, width = image.shape[:2]
            if max(width, height) > 1280:
                scale = 1280 / max(width, height)
                image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
            timestamp_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            second = timestamp_ms / 1000 if timestamp_ms > 0 else index / fps
            name = f"frame-{index:07d}.jpg"
            if not cv2.imwrite(str(folder / name), image):
                raise OSError(f"Не удалось сохранить {name}")
            manifest[name] = {"label": f"Кадр {second:.2f} с", "video_time": round(second, 4)}
        if not manifest:
            raise ValueError("В видео нет кадров для анализа")
        return manifest, fps / stride
    finally:
        capture.release()


def track_poles(frames, input_dir, detections, *, algorithm="botsort", sample_fps=None):
    poles = [item for item in detections if item["class"] == "pole"]
    return track_boxes(frames, input_dir, poles, algorithm=algorithm, sample_fps=sample_fps)


def track_boxes(frames, input_dir, detections, *, algorithm="botsort", sample_fps=None):
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers import BOTSORT, BYTETracker
    from ultralytics.utils import ROOT, YAML, IterableSimpleNamespace

    trackers = {"botsort": BOTSORT, "bytetrack": BYTETracker}
    config = YAML.load(ROOT / f"cfg/trackers/{algorithm}.yaml")
    if sample_fps is not None:
        # The upstream buffer is expressed in frames. At five samples per second,
        # the default 30-frame buffer can revive a track from six seconds ago.
        config["track_buffer"] = max(1, round(sample_fps))
    tracker = trackers[algorithm](IterableSimpleNamespace(**config))
    by_frame = defaultdict(list)
    for item in detections:
        by_frame[item["frame"]].append(item)
    assignments = {}
    counts = Counter()
    for name in frames:
        image = cv2.imread(str(input_dir / name))
        if image is None:
            raise ValueError(f"Не удалось повторно прочитать {name}")
        poles = by_frame[name]
        boxes = [item["bbox"] + [item["confidence"], 0] for item in poles]
        values = np.asarray(boxes, dtype=np.float32).reshape(-1, 6)
        for row in tracker.update(Boxes(values, image.shape[:2]), image):
            track_id = int(row[-4])
            detection = poles[int(row[-1])]
            assignments[name, tuple(detection["bbox"])] = track_id
            counts[track_id] += 1
    return {key: value for key, value in assignments.items() if counts[value] >= 2}


def nearby_equipment(first, second):
    ax1, ay1, ax2, ay2 = first["bbox"]
    bx1, by1, bx2, by2 = second["bbox"]
    horizontal = max(ax1 - bx2, bx1 - ax2, 0)
    vertical = max(ay1 - by2, by1 - ay2, 0)
    return (horizontal <= 1.5 * max(ax2 - ax1, bx2 - bx1)
            and vertical <= 0.5 * max(ay2 - ay1, by2 - by1))


def equipment_clusters(detections):
    by_frame = defaultdict(list)
    for item in detections:
        if item["confidence"] >= EQUIPMENT_THRESHOLD.get(item["class"], math.inf):
            by_frame[item["frame"]].append(item)
    clusters = []
    for frame, items in by_frame.items():
        pending = items.copy()
        while pending:
            group = [pending.pop()]
            for member in group:
                adjacent = [item for item in pending if nearby_equipment(member, item)]
                for item in adjacent:
                    pending.remove(item)
                group.extend(adjacent)
            classes = {item["class"] for item in group}
            if len(group) < 2 or "antenna" not in classes:
                continue
            x1, y1, x2, y2 = zip(*(item["bbox"] for item in group))
            clusters.append({"frame": frame, "bbox": [min(x1), min(y1), max(x2), max(y2)],
                             "confidence": max(item["confidence"] for item in group),
                             "equipment": sorted(classes), "members": group})
    return clusters


def cluster_evidence(frames, input_dir, detections, candidates, video_fps, pole_tracks):
    clusters = equipment_clusters(detections)
    matches = track_boxes(frames, input_dir, clusters, sample_fps=video_fps) if clusters else {}
    by_frame = defaultdict(list)
    for row in candidates:
        box = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        by_frame[row["frame"]].append((row, box))
    absorbed = set()
    associated_poles = set()
    extra = []
    object_ids = {key: f"video-pole-{value:04d}" for key, value in pole_tracks.items()}
    for cluster in sorted(clusters, key=lambda item: item["confidence"], reverse=True):
        frame, box = cluster["frame"], tuple(cluster["bbox"])
        track_id = matches.get((frame, box))
        if track_id is None:
            continue
        member_boxes = {tuple(item["bbox"]) for item in cluster["members"]}
        # Preserve pole-based evidence whenever a unique pole occupies the cluster center.
        center = (box[0] + box[2]) / 2
        poles = [pole for row, pole in by_frame[frame] if row["association"] == "pole+equipment"
                 and pole[0] <= center <= pole[2] and pole[1] <= box[3] and pole[3] >= box[1]]
        if len(poles) > 1:
            continue
        if poles:
            key = (frame, poles[0])
            if key in associated_poles:
                continue
            associated_poles.add(key)
            object_ids[key] = f"video-equipment-{track_id:04d}"
        else:
            object_ids[frame, box] = f"video-equipment-{track_id:04d}"
            extra.append({"frame": frame, "association": "equipment_cluster", "pole_conf": None,
                          "equipment_classes": "|".join(cluster["equipment"]),
                          "max_equipment_conf": cluster["confidence"],
                          **dict(zip(("x1", "y1", "x2", "y2"), box))})
        absorbed.update((frame, member) for member in member_boxes)
    selected = [row for name, pairs in by_frame.items() for row, box in pairs
                if row["association"] != "equipment_only" or (name, box) not in absorbed]
    return selected + extra, object_ids
