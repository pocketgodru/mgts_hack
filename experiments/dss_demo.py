"""Run the local DSS classifier and write its inspection report as CSV."""

import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO

from mistral_vision import DEFAULT_WEIGHTS, predict_local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--locations", type=Path, help="CSV with file, object_id, address, latitude, longitude")
    args = parser.parse_args()
    locations = {}
    if args.locations:
        with args.locations.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "file" not in reader.fieldnames:
                parser.error("--locations requires a file column")
            for row in reader:
                name = row["file"]
                if not name or name in locations:
                    parser.error("--locations contains an empty or duplicate file name")
                locations[name] = row
    detector = YOLO(str(DEFAULT_WEIGHTS.resolve(strict=True)))
    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    rows = []
    for image in args.images:
        prediction, confidence = predict_local(detector, image, device)
        status = prediction if confidence >= 0.65 else "review"
        recommendation = {
            "normal": "Действий не требуется",
            "damaged": "Проверить повреждение по кадру и назначить осмотр",
            "review": "Проверить кадр вручную",
        }[status]
        rows.append({
            "photo_id": image.stem,
            "file": image.name,
            "object_id": locations.get(image.name, {}).get("object_id", ""),
            "address": locations.get(image.name, {}).get("address", ""),
            "latitude": locations.get(image.name, {}).get("latitude", ""),
            "longitude": locations.get(image.name, {}).get("longitude", ""),
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "status": status,
            "recommendation": recommendation,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} фото → {args.output}")


if __name__ == "__main__":
    main()
