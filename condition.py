import json
import math
import os
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path


@dataclass(frozen=True)
class ConditionConfig:
    classifier: Path
    segmenter: Path
    normal_labels: frozenset[str]
    repair_labels: frozenset[str]
    classification_confidence: float
    segmentation_confidence: float

    def available(self):
        return {"classifier": self.classifier.is_file(), "segmenter": self.segmenter.is_file()}


def load_condition_config(root):
    config_path = root / os.getenv("MGTS_CONDITION_CONFIG", "models/condition.json")
    if "MGTS_CONDITION_CONFIG" in os.environ and not config_path.is_file():
        raise ValueError(f"Файл модели состояния не найден: {config_path}")
    raw = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.is_file() else {}
    if not isinstance(raw, dict):
        raise ValueError("Настройки модели состояния должны быть объектом JSON")

    def labels(key, default):
        values = raw.get(key, default)
        if not isinstance(values, list) or any(not isinstance(label, str) or not label.strip() for label in values):
            raise ValueError(f"{key} должен быть списком непустых названий классов")
        return frozenset(label.strip().casefold() for label in values)

    config = ConditionConfig(
        root / raw.get("classifier", "models/condition-cls.pt"),
        root / raw.get("segmenter", "models/condition-seg.pt"),
        labels("normal_labels", ["normal", "ok", "норма", "исправно"]),
        labels("repair_labels", []),
        float(raw.get("classification_confidence", 0.65)),
        float(raw.get("segmentation_confidence", 0.30)),
    )
    if config.normal_labels & config.repair_labels:
        raise ValueError("normal_labels и repair_labels пересекаются")
    if not all(0 < value <= 1 for value in (config.classification_confidence, config.segmentation_confidence)):
        raise ValueError("Пороги уверенности должны быть больше 0 и не выше 1")
    return config


def condition_result(config, classification, masks):
    defects = [(mask["label"], mask["confidence"]) for mask in masks]
    if classification is not None:
        label, score = classification
        if score >= config.classification_confidence and label.strip().casefold() not in config.normal_labels:
            defects.append(classification)
    if defects:
        label, score = max(defects, key=lambda item: (item[0].strip().casefold() in config.repair_labels, item[1]))
        status = "repair" if label.strip().casefold() in config.repair_labels else "review"
        return {"status": status, "defect": label, "condition_confidence": score, "masks": masks}
    if classification is None:
        return {"status": "unassessed", "defect": None, "condition_confidence": None, "masks": []}

    _, score = classification
    return {"status": "review" if score < config.classification_confidence else "normal",
            "defect": None, "condition_confidence": score, "masks": []}


def crop_regions(image, objects):
    height, width = image.shape[:2]
    for item in objects:
        if item["bbox"] is None:
            yield item, image, 0, 0
            continue
        x1, y1, x2, y2 = item["bbox"]
        margin = max(4, 0.12 * max(x2 - x1, y2 - y1))
        left = max(0, math.floor(x1 - margin))
        top = max(0, math.floor(y1 - margin))
        right = min(width, math.ceil(x2 + margin))
        bottom = min(height, math.ceil(y2 + margin))
        if right > left and bottom > top:
            yield item, image[top:bottom, left:right], left, top


def square_image(image):
    import cv2
    height, width = image.shape[:2]
    size = max(height, width)
    top, left = (size - height) // 2, (size - width) // 2
    return cv2.copyMakeBorder(image, top, size - height - top, left, size - width - left,
                              cv2.BORDER_CONSTANT, value=(114, 114, 114))


def assess_objects(objects, input_dir, config, device, *, classify_only=False):
    for item in objects:
        item.update(condition_result(config, None, []))
    available = config.available()
    if not objects or not any(available.values()):
        return

    import cv2
    import numpy as np
    from ultralytics import YOLO

    classifier = YOLO(str(config.classifier)) if available["classifier"] else None
    segmenter = YOLO(str(config.segmenter)) if available["segmenter"] and not classify_only else None
    if classifier is not None and classifier.task != "classify":
        raise ValueError("Веса classifier должны быть моделью YOLO classify")
    if segmenter is not None and segmenter.task != "segment":
        raise ValueError("Веса segmenter должны быть моделью YOLO segment")

    for frame, items in groupby(sorted(objects, key=lambda item: item["frame"]), key=lambda item: item["frame"]):
        # imdecode also handles Unicode project paths on Windows.
        image = cv2.imdecode(np.frombuffer((input_dir / frame).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        regions = list(crop_regions(image, items))
        for start in range(0, len(regions), 8):
            batch = regions[start:start + 8]
            crops = [crop for _, crop, _, _ in batch]
            options = {"imgsz": 640} if classify_only else {}
            classifications = classifier.predict([square_image(crop) for crop in crops], device=device,
                                                  verbose=False, **options) if classifier else [None] * len(batch)
            segments = segmenter.predict(crops, conf=config.segmentation_confidence, device=device,
                                         max_det=40, verbose=False) if segmenter else [None] * len(batch)
            for (item, _, left, top), clas, seg in zip(batch, classifications, segments):
                prediction = None
                if clas is not None and clas.probs is not None:
                    prediction = (clas.names[clas.probs.top1], round(float(clas.probs.top1conf), 4))
                    item["condition_prediction"] = prediction[0]
                masks = []
                if seg is not None and seg.masks is not None and seg.boxes is not None:
                    for box, polygon in zip(seg.boxes, seg.masks.xy):
                        label = seg.names[int(box.cls)]
                        if label.strip().casefold() in config.normal_labels or len(polygon) < 3:
                            continue
                        stride = max(1, math.ceil(len(polygon) / 160))
                        points = [[round(float(x) + left, 1), round(float(y) + top, 1)] for x, y in polygon[::stride]]
                        masks.append({"label": label, "confidence": round(float(box.conf), 4), "polygon": points})
                item.update(condition_result(config, prediction, masks))
