import csv
import io
import math
from bisect import bisect_left
from collections import Counter, defaultdict


TYPE_NAMES = {"odn": "Опора двойного назначения", "manhole": "Кабельный колодец"}
STATUS = {
    "repair": ("Ремонт", "Передать в ремонт после подтверждения"),
    "review": ("Проверка", "Проверить по кадру или на месте"),
    "unassessed": ("Не оценивалось", "Оценить состояние вручную"),
    "normal": ("Норма", "Действий не требуется"),
}
STATUS_RANK = {name: len(STATUS) - index for index, name in enumerate(STATUS)}


def valid_coordinates(latitude, longitude):
    return (latitude is not None and longitude is not None
            and math.isfinite(latitude) and -90 <= latitude <= 90
            and math.isfinite(longitude) and -180 <= longitude <= 180)


def _csv_reader(data):
    text = data.decode("utf-8-sig")
    first_line = text.partition("\n")[0]
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter)
    if reader.fieldnames and len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("Имена колонок CSV должны быть уникальны")
    return reader


def read_frame_csv(data, frame_names):
    reader = _csv_reader(data)
    if not reader.fieldnames or "frame" not in reader.fieldnames:
        raise ValueError("CSV должен содержать колонку frame с именем изображения")

    rows = {}
    for row in reader:
        if None in row:
            raise ValueError(f"Строка {reader.line_num}: число значений превышает число колонок CSV")
        def value(key):
            return (row.get(key) or "").strip()

        frame, asset_id, kind = value("frame"), value("object_id"), value("object_type").lower()
        if frame not in frame_names or frame in rows or frame_names[frame] != 1:
            raise ValueError(f"Строка {reader.line_num}: имя кадра не найдено или неоднозначно: {frame or 'пусто'}")
        if kind and kind not in TYPE_NAMES:
            raise ValueError(f"Строка {reader.line_num}: object_type должен быть odn или manhole")
        if len(asset_id) > 160:
            raise ValueError(f"Строка {reader.line_num}: object_id слишком длинный")
        latitude_text, longitude_text = value("latitude"), value("longitude")
        coords = (None, None)
        if latitude_text or longitude_text:
            try:
                coords = (float(latitude_text.replace(",", ".")), float(longitude_text.replace(",", ".")))
            except ValueError as exc:
                raise ValueError(f"Строка {reader.line_num}: нужны обе координаты в градусах") from exc
            if not valid_coordinates(*coords):
                raise ValueError(f"Строка {reader.line_num}: координаты вне допустимого диапазона")
        rows[frame] = {
            "object_id": asset_id or None, "object_type": kind or None,
            "latitude": coords[0], "longitude": coords[1],
            "coordinate_source": "csv" if coords[0] is not None else None,
        }
    return rows


def read_video_gps(data):
    reader = _csv_reader(data)
    columns = ("time_s", "latitude", "longitude")
    if not reader.fieldnames or not set(columns).issubset(reader.fieldnames):
        raise ValueError("Нужны колонки time_s, latitude, longitude")
    samples = {}
    for row in reader:
        try:
            if None in row:
                raise ValueError("Лишние значения в строке")
            second, latitude, longitude = (float((row[key] or "").strip().replace(",", ".")) for key in columns)
            if not math.isfinite(second) or second < 0:
                raise ValueError("time_s — неотрицательное число секунд от начала ролика")
            if not valid_coordinates(latitude, longitude):
                raise ValueError("Координаты вне допустимого диапазона")
            if second in samples:
                raise ValueError("Повторяется значение time_s")
        except ValueError as exc:
            raise ValueError(f"Строка {reader.line_num}: {exc}") from exc
        samples[second] = (latitude, longitude)
    if not samples:
        raise ValueError("В GPS-треке нет точек")
    return samples


def locate_video_frames(manifest, samples, *, max_gap=10.0):
    """Locate the camera at sampled frame times, without extrapolating."""
    times = sorted(samples)
    located = 0
    for frame in manifest.values():
        second = frame["video_time"]
        index = bisect_left(times, second)
        if index < len(times) and times[index] == second:
            latitude, longitude = samples[second]
        elif 0 < index < len(times) and times[index] - times[index - 1] <= max_gap:
            start, end = times[index - 1:index + 1]
            weight = (second - start) / (end - start)
            start_lat, start_lon = samples[start]
            end_lat, end_lon = samples[end]
            latitude = start_lat + (end_lat - start_lat) * weight
            delta_lon = (end_lon - start_lon + 180) % 360 - 180
            longitude = (start_lon + delta_lon * weight + 180) % 360 - 180
        else:
            continue
        frame.update(latitude=round(latitude, 7), longitude=round(longitude, 7), coordinate_source="video_gps")
        located += 1
    if not located:
        raise ValueError("GPS-трек не покрывает ни одного кадра. Проверьте время от начала ролика и интервалы между точками")


def find_objects(frames, detections, candidates, video_ids=None):
    video_ids = video_ids or {}
    by_frame = defaultdict(list)
    for row in candidates:
        by_frame[row["frame"]].append({
            "kind": "odn", "source": row["association"],
            "bbox": [float(row[key]) for key in ("x1", "y1", "x2", "y2")],
            "detection_confidence": row["max_equipment_conf"],
            "equipment": row["equipment_classes"].split("|"),
        })
    for row in detections:
        if row["class"] == "manhole":
            by_frame[row["frame"]].append({
                "kind": "manhole", "source": "detector", "bbox": row["bbox"],
                "detection_confidence": row["confidence"], "equipment": [],
            })

    objects = []
    for frame in frames:
        matches = by_frame[frame["name"]]
        declared_type = frame.get("object_type")
        if declared_type and not any(item["kind"] == declared_type for item in matches):
            matches.append({"kind": declared_type, "source": "metadata", "bbox": None,
                            "detection_confidence": None, "equipment": []})
        eligible = [item for item in matches if not declared_type or item["kind"] == declared_type]
        identified = eligible[0] if frame.get("object_id") and len(eligible) == 1 else None
        for index, item in enumerate(matches, start=1):
            object_id = frame["object_id"] if item is identified else None
            if object_id is None and item["source"] in {"pole+equipment", "equipment_cluster"}:
                object_id = video_ids.get((frame["name"], tuple(item["bbox"])))
            objects.append({
                **item,
                "object_id": object_id,
                "observation_id": f"{frame['name']}:{index}",
                "frame": frame["name"],
                "frame_label": frame["label"],
                "latitude": frame["latitude"], "longitude": frame["longitude"],
                "coordinate_source": frame["coordinate_source"],
                "evidence_url": frame["original_url"],
            })
    return objects


def group_observations(objects):
    groups = defaultdict(list)
    for item in objects:
        groups[(item["kind"], bool(item["object_id"]), item["object_id"] or item["observation_id"])].append(item)
    rows = []
    for items in groups.values():
        best = max(items, key=lambda item: (STATUS_RANK[item["status"]], item["defect"] is not None,
                                            item["condition_confidence"] or 0, item["detection_confidence"] or 0))
        geo = best if best["latitude"] is not None else next((item for item in items if item["latitude"] is not None), best)
        rows.append({**best, "latitude": geo["latitude"], "longitude": geo["longitude"],
                     "coordinate_source": geo["coordinate_source"],
                     "coordinate_frame": geo["frame"] if geo["latitude"] is not None else None,
                     "observations": len(items),
                     "evidence_frames": list(dict.fromkeys(item["frame"] for item in items))})
    return sorted(rows, key=lambda row: (-STATUS_RANK[row["status"]], row["kind"], row["observation_id"]))


def write_registry(path, rows):
    def cell(value):
        if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
            return "'" + value
        return value

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle, delimiter=";")
        writer.writerow(["object_id", "object_type", "status", "defect", "recommendation", "latitude", "longitude",
                         "coordinate_source", "frame", "evidence_url", "detection_confidence",
                         "condition_confidence", "observations", "coordinate_frame"])
        for row in rows:
            status, recommendation = STATUS[row["status"]]
            writer.writerow([cell(row["object_id"] or row["observation_id"]), TYPE_NAMES[row["kind"]], status,
                             cell(row["defect"]), recommendation, row["latitude"], row["longitude"],
                             row["coordinate_source"], cell(row["frame_label"]), row["evidence_url"],
                             row["detection_confidence"], row["condition_confidence"], row["observations"],
                             cell(row.get("coordinate_frame"))])


def status_counts(rows):
    counts = Counter(row["status"] for row in rows)
    return {status: counts[status] for status in STATUS}
