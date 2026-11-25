#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import batched_nms
from ultralytics import YOLO

from odn import active_detector

EQUIPMENT_CLASSES = {"antenna", "rru", "microwave"}
DEFAULT_CLASS_CONF = {"pole": 0.60, "antenna": 0.10, "rru": 0.05, "microwave": 0.25}


def parse_class_conf(values, names, fallback):
    thresholds = {name: fallback for name in names.values()}
    for value in values:
        name, separator, raw_conf = value.partition("=")
        if not separator or name not in thresholds:
            raise ValueError(f"Expected CLASS=CONF with a known class, got: {value}")
        try:
            confidence = float(raw_conf)
        except ValueError as exc:
            raise ValueError(f"Invalid confidence in {value}") from exc
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got: {value}")
        thresholds[name] = confidence
    return thresholds


def result_detections(result):
    if result.boxes is None:
        return []
    return [
        [float(value) for value in box.xyxy[0]] + [float(box.conf), int(box.cls)]
        for box in result.boxes
    ]


def crop_pole(image, box, top_fraction=0.55, width_scale=1.6):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    pole_width = x2 - x1
    pole_height = y2 - y1
    center_x = (x1 + x2) / 2
    crop_x1 = max(0, int(center_x - pole_width * width_scale / 2))
    crop_x2 = min(width, int(center_x + pole_width * width_scale / 2))
    crop_y1 = max(0, int(y1 - 0.50 * pole_height))
    crop_y2 = min(height, int(y1 + top_fraction * pole_height))
    if crop_x2 - crop_x1 < 8 or crop_y2 - crop_y1 < 8:
        return None
    return image[crop_y1:crop_y2, crop_x1:crop_x2], crop_x1, crop_y1


def merge_nms(detections, iou_threshold):
    if not detections:
        return []
    boxes = torch.tensor([item[:4] for item in detections], dtype=torch.float32)
    scores = torch.tensor([item[4] for item in detections], dtype=torch.float32)
    classes = torch.tensor([item[5] for item in detections], dtype=torch.int64)
    keep = batched_nms(boxes, scores, classes, iou_threshold).tolist()
    return [detections[index] for index in keep]


def tile_regions(width, height, tile_size, overlap):
    if tile_size < 64:
        raise ValueError("Tile size must be at least 64 pixels")
    if not 0 <= overlap < 0.9:
        raise ValueError("Tile overlap must be in [0, 0.9)")

    def axis_starts(length):
        if length <= tile_size:
            return [0]
        step = max(1, int(tile_size * (1 - overlap)))
        count = int(np.ceil((length - tile_size) / step)) + 1
        return [round(index * (length - tile_size) / (count - 1))
                for index in range(count)]

    return [
        (x, y, min(x + tile_size, width), min(y + tile_size, height))
        for y in axis_starts(height)
        for x in axis_starts(width)
    ]


def is_equipment_on_pole(equipment_box, pole_box):
    ex = (equipment_box[0] + equipment_box[2]) / 2
    ey = (equipment_box[1] + equipment_box[3]) / 2
    x1, y1, x2, y2 = pole_box
    width, height = x2 - x1, y2 - y1
    return (x1 - 0.35 * width <= ex <= x2 + 0.35 * width
            and y1 - 0.60 * height <= ey <= y1 + 0.60 * height)


def suppress_nested_detections(detections, class_ids):
    kept = []
    for index, item in enumerate(detections):
        if item[5] not in class_ids:
            kept.append(item)
            continue
        x1, y1, x2, y2 = item[:4]
        area = max(0, x2 - x1) * max(0, y2 - y1)
        nested = False
        for other_index, other in enumerate(detections):
            if (index == other_index or item[5] != other[5]
                    or item[4] > other[4]
                    or (item[4] == other[4] and index < other_index)):
                continue
            ox1, oy1, ox2, oy2 = other[:4]
            other_area = max(0, ox2 - ox1) * max(0, oy2 - oy1)
            intersection = max(0, min(x2, ox2) - max(x1, ox1)) * max(0, min(y2, oy2) - max(y1, oy1))
            if min(area, other_area) > 0 and intersection / min(area, other_area) >= 0.85:
                nested = True
                break
        if not nested:
            kept.append(item)
    return kept


def draw_detections(image, detections, names):
    colors = {"pole": (60, 220, 60), "antenna": (40, 180, 255),
              "rru": (180, 80, 255), "microwave": (60, 220, 255), "manhole": (105, 85, 255)}
    for x1, y1, x2, y2, conf, class_id in detections:
        name = names[class_id]
        color = colors.get(name, (255, 255, 255))
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
        cv2.putText(image, f"{name} {conf:.2f}", (int(x1), max(20, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    return image


def predict_manhole_boxes(model, images, *, confidence, device, batch):
    """Find cover candidates on overlapping road crops of wide street frames."""
    crops, locations = [], []
    for index, image in enumerate(images):
        if image is None:
            continue
        height, width = image.shape[:2]
        if width >= 1.4 * height:
            top = round(height * .50)
            regions = [(0, top, round(width * .60), height),
                       (round(width * .40), top, width, height)]
        else:
            regions = [(0, 0, width, height)]
        for x1, y1, x2, y2 in regions:
            crops.append(image[y1:y2, x1:x2])
            locations.append((index, x1, y1, x2 - x1, y2 - y1))

    found = [[] for _ in images]
    for offset in range(0, len(crops), batch):
        predictions = model.predict(crops[offset:offset + batch], conf=confidence, imgsz=640,
                                    batch=batch, device=device, verbose=False)
        for result, (index, x, y, width, height) in zip(predictions, locations[offset:offset + batch], strict=True):
            for x1, y1, x2, y2, score, category in result_detections(result):
                # Boxes covering most of a road crop are usually entire scenes.
                if (x2 - x1) > .65 * width or (y2 - y1) > .70 * height:
                    continue
                found[index].append([x1+x, y1+y, x2+x, y2+y, score, category])
    return [merge_nms(items, .5) if items else [] for items in found]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--weights", default=str(active_detector()[1]))
    parser.add_argument("--manhole-weights", type=Path, help="Trained YOLO26s weights for manhole detection")
    parser.add_argument("--manhole-conf", type=float, default=0.30)
    parser.add_argument("--out", type=Path, default=Path("detector_output"))
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--class-conf", nargs="+",
                        default=[f"{name}={value}" for name, value in DEFAULT_CLASS_CONF.items()],
                        help="Final per-class thresholds, for example pole=0.6 antenna=0.1")
    parser.add_argument("--crop-conf", type=float, default=0.05)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--crop-imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-top-crop", action="store_true")
    parser.add_argument("--tile", action="store_true")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=float, default=0.2)
    parser.add_argument("--tile-conf", type=float, default=None)
    parser.add_argument("--profile", action="store_true", help="Record PyTorch peak GPU memory")
    args = parser.parse_args()

    image_paths = sorted(path for path in args.input.iterdir()
                         if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        image_paths = image_paths[:args.limit]
    if not image_paths:
        raise SystemExit(f"No images found in {args.input}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "labeled").mkdir(exist_ok=True)
    model = YOLO(args.weights)
    names = model.names
    equipment_ids = {index for index, name in names.items() if name in EQUIPMENT_CLASSES}
    pole_ids = {index for index, name in names.items() if name == "pole"}
    try:
        class_conf = parse_class_conf(args.class_conf, names, args.conf)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    candidate_conf = min(args.conf, *class_conf.values())
    manhole_model = None
    if args.manhole_weights:
        if not args.manhole_weights.is_file():
            raise SystemExit(f"Manhole weights not found: {args.manhole_weights}")
        manhole_model = YOLO(str(args.manhole_weights))
        manhole_classes = [idx for idx, name in manhole_model.names.items()
                           if name in {"manhole", "manhole cover", "open manhole"}]
        if not manhole_classes:
            raise SystemExit("Manhole checkpoint has no supported manhole classes")
        manhole_id = max(names) + 1
        names = {**names, manhole_id: "manhole"}
    crop_candidate_conf = min([args.crop_conf,
                               *(class_conf[name] for name in EQUIPMENT_CLASSES if name in class_conf)])
    site_evidence = {}

    with (args.out / "detections.csv").open("w", newline="", encoding="utf-8") as detections_file, \
            (args.out / "odn_candidates.csv").open("w", newline="", encoding="utf-8") as odn_file:
        detection_writer = csv.writer(detections_file)
        detection_writer.writerow(["frame", "class", "conf", "x1", "y1", "x2", "y2"])
        odn_writer = csv.writer(odn_file)
        odn_writer.writerow(["frame", "pole_conf", "equipment_classes",
                             "max_equipment_conf", "x1", "y1", "x2", "y2",
                             "association"])

        for start in range(0, len(image_paths), args.batch):
            chunk = image_paths[start:start + args.batch]
            images = [cv2.imread(str(path)) for path in chunk]
            results = model.predict([str(path) for path in chunk], conf=candidate_conf,
                                    imgsz=args.imgsz, batch=args.batch,
                                    device=args.device, verbose=False)
            manholes = [[] for _ in chunk]
            if manhole_model:
                manholes = predict_manhole_boxes(manhole_model, images, confidence=args.manhole_conf,
                                                 device=args.device, batch=args.batch)
            for path, result, image, manhole_result in zip(chunk, results, images, manholes, strict=True):
                detections = result_detections(result)
                if image is None:
                    continue
                if args.tile:
                    height, width = image.shape[:2]
                    regions = [region for region in tile_regions(
                        width, height, args.tile_size, args.tile_overlap)
                        if region != (0, 0, width, height)]
                    tiles = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in regions]
                    if tiles:
                        tile_results = model.predict(
                            tiles, conf=candidate_conf if args.tile_conf is None else min(args.tile_conf, candidate_conf),
                            imgsz=args.imgsz, batch=args.batch,
                            device=args.device, verbose=False)
                        for (offset_x, offset_y, _, _), tile_result in zip(regions, tile_results):
                            for x1, y1, x2, y2, conf, class_id in result_detections(tile_result):
                                detections.append([x1 + offset_x, y1 + offset_y,
                                                   x2 + offset_x, y2 + offset_y,
                                                   conf, class_id])
                if not args.no_top_crop and pole_ids:
                    poles = [item for item in detections if item[5] in pole_ids]
                    for pole in poles:
                        cropped = crop_pole(image, pole[:4])
                        if cropped is None:
                            continue
                        crop, offset_x, offset_y = cropped
                        crop_result = model.predict(crop, conf=crop_candidate_conf,
                                                    imgsz=args.crop_imgsz,
                                                    device=args.device, verbose=False)[0]
                        for x1, y1, x2, y2, conf, class_id in result_detections(crop_result):
                            if class_id in equipment_ids:
                                detections.append([x1 + offset_x, y1 + offset_y,
                                                   x2 + offset_x, y2 + offset_y,
                                                   conf, class_id])
                detections = merge_nms(detections, args.iou)
                detections = suppress_nested_detections(detections, equipment_ids | pole_ids)
                detections = [item for item in detections
                              if item[4] >= class_conf[names[item[5]]]]
                if manhole_result:
                    detections.extend(item[:5] + [manhole_id] for item in manhole_result if item[5] in manhole_classes)
                if not detections:
                    continue
                for x1, y1, x2, y2, conf, class_id in detections:
                    detection_writer.writerow([path.name, names[class_id], round(conf, 4),
                                               round(x1, 1), round(y1, 1),
                                               round(x2, 1), round(y2, 1)])
                poles = [item for item in detections if item[5] in pole_ids]
                equipment = [item for item in detections if item[5] in equipment_ids]
                if equipment:
                    match = re.search(r"site_\d+", path.stem)
                    site = match.group() if match else path.stem
                    evidence = site_evidence.setdefault(site, {
                        "frames": set(), "equipment_classes": set(),
                        "max_equipment_conf": 0.0, "max_pole_conf": 0.0,
                    })
                    evidence["frames"].add(path.name)
                    evidence["equipment_classes"].update(names[item[5]] for item in equipment)
                    evidence["max_equipment_conf"] = max(
                        evidence["max_equipment_conf"], max(item[4] for item in equipment))
                    if poles:
                        evidence["max_pole_conf"] = max(
                            evidence["max_pole_conf"], max(item[4] for item in poles))
                attached_equipment = set()
                for pole in poles:
                    attached = [(index, item) for index, item in enumerate(equipment)
                                if is_equipment_on_pole(item[:4], pole[:4])]
                    if attached:
                        attached_equipment.update(index for index, _ in attached)
                        attached_items = [item for _, item in attached]
                        odn_writer.writerow([
                            path.name, round(pole[4], 4),
                            "|".join(sorted({names[item[5]] for item in attached_items})),
                            round(max(item[4] for item in attached_items), 4),
                            *[round(value, 1) for value in pole[:4]],
                            "pole+equipment",
                        ])
                for index, item in enumerate(equipment):
                    if index not in attached_equipment:
                        odn_writer.writerow([
                            path.name, "", names[item[5]], round(item[4], 4),
                            *[round(value, 1) for value in item[:4]],
                            "equipment_only",
                        ])
                rendered = draw_detections(image.copy(), detections, names)
                cv2.imwrite(str(args.out / "labeled" / path.name), rendered,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
            print(f"{min(start + args.batch, len(image_paths))}/{len(image_paths)}", flush=True)

    with (args.out / "odn_sites.csv").open("w", newline="", encoding="utf-8") as site_file:
        writer = csv.writer(site_file)
        writer.writerow(["site", "frames_with_equipment", "evidence_frames",
                         "equipment_classes", "max_equipment_conf", "max_pole_conf"])
        for site in sorted(site_evidence):
            evidence = site_evidence[site]
            writer.writerow([
                site, len(evidence["frames"]), "|".join(sorted(evidence["frames"])),
                "|".join(sorted(evidence["equipment_classes"])),
                round(evidence["max_equipment_conf"], 4),
                round(evidence["max_pole_conf"], 4),
            ])

    print(f"Results: {args.out / 'detections.csv'}")
    print(f"ODN candidates: {args.out / 'odn_candidates.csv'}")
    print(f"ODN sites: {args.out / 'odn_sites.csv'}")
    if args.profile:
        device = next(model.model.parameters()).device
        memory = {}
        if device.type == "cuda":
            memory = {"torch_allocated_bytes": torch.cuda.max_memory_allocated(device),
                      "torch_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        (args.out / "profile.json").write_text(json.dumps(memory), encoding="utf-8")


if __name__ == "__main__":
    main()
