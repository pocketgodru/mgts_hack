import argparse
import csv
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from condition import condition_result, load_condition_config, square_image
from final_annotations import IMAGE_TYPES, import_excel, inspect_excel, manifest_images


ROOT = Path(__file__).resolve().parent
START_WEIGHTS = {"classify": "yolo11s-cls.pt", "segment": "yolo11s-seg.pt"}


def folder_images(source, groups_file):
    groups = {}
    if groups_file:
        with groups_file.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not {"path", "group"}.issubset(reader.fieldnames or []):
                raise ValueError("CSV групп должен содержать path,group.")
            for row in reader:
                relative = row["path"].replace("\\", "/")
                group = row["group"].strip()
                if relative in groups or not group:
                    raise ValueError(f"Некорректная или повторяющаяся группа: {relative}")
                groups[relative] = group

    records = []
    matched = set()
    for folder in sorted(path for path in source.iterdir() if path.is_dir() and not path.name.startswith(".")):
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_TYPES:
                continue
            relative = path.relative_to(source).as_posix()
            if groups_file and relative not in groups:
                raise ValueError(f"Нет группы для {relative}")
            matched.add(relative)
            records.append((path, folder.name, groups.get(relative)))
    if set(groups) - matched:
        raise ValueError("В CSV групп есть пути, которых нет во входной папке.")
    return records


def prepare_classification(args):
    import cv2
    import numpy as np
    from sklearn.model_selection import GroupShuffleSplit

    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir() or output.exists() or output.is_relative_to(source):
        raise ValueError("Укажите папку с классами и новый выходной каталог вне исходной папки.")
    if not 0 < args.val_fraction < 1:
        raise ValueError("Доля валидации должна быть между 0 и 1.")
    if args.size < 32:
        raise ValueError("Размер подготовленного кадра должен быть не меньше 32 пикселей.")

    if args.manifest:
        inputs = manifest_images(source, args.manifest)
    else:
        inputs = folder_images(source, args.groups)
    records, by_hash = [], {}
    for path, label, supplied_group in inputs:
        with path.open("rb") as image_file:
            sha = hashlib.file_digest(image_file, "sha256").hexdigest()
        group = supplied_group or sha
        if sha in by_hash and by_hash[sha] != (label, group):
            raise ValueError(f"У копии {path.relative_to(source)} расходится класс или группа.")
        by_hash[sha] = (label, group)
        records.append((path, label, group))

    classes = {label for _, label, _ in records}
    if len(classes) < 2:
        raise ValueError("Нужны минимум два класса изображений.")
    if len({label.casefold() for label in classes}) != len(classes):
        raise ValueError("Названия классов отличаются только регистром. Приведите их к одному написанию.")
    class_groups = defaultdict(set)
    for _, label, group in records:
        class_groups[label].add(group)
    limited = sorted(label for label, groups in class_groups.items() if len(groups) < 2)
    if limited:
        raise ValueError(f"Для раздельных train/val нужны две независимые группы каждого класса: {', '.join(limited)}")

    splitter = GroupShuffleSplit(n_splits=256, test_size=args.val_fraction, random_state=args.seed)
    labels = [label for _, label, _ in records]
    groups = [group for _, _, group in records]
    class_counts = Counter(labels)
    best_split = None
    for train, val in splitter.split(records, labels, groups):
        val_counts = Counter(labels[index] for index in val)
        if any(not 0 < val_counts[label] < class_counts[label] for label in classes):
            continue
        score = sum(abs(val_counts[label] / class_counts[label] - args.val_fraction) for label in classes)
        if best_split is None or score < best_split[0]:
            best_split = (score, train, val)
    if best_split is None:
        raise ValueError("Не удалось сохранить все классы в train и val. Проверьте группы или измените --val-fraction.")
    _, train, val = best_split

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="mgts-dataset-") as temporary:
        stage = Path(temporary) / "ready"
        stage.mkdir()
        with (stage / "split.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source", "class", "group", "split", "image"])
            for split, indices in (("train", train), ("val", val)):
                for index in indices:
                    path, label, group = records[index]
                    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f"Повреждённое изображение: {path}")
                    target = stage / split / label / f"{index:06d}.jpg"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    image = square_image(image)
                    if image.shape[0] > args.size:
                        image = cv2.resize(image, (args.size, args.size), interpolation=cv2.INTER_AREA)
                    encoded, binary = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 96])
                    if not encoded:
                        raise OSError(f"Ошибка кодирования: {path}")
                    target.write_bytes(binary.tobytes())
                    writer.writerow([path.relative_to(source).as_posix(), label, group, split, target.relative_to(stage).as_posix()])
        stage.rename(output)
    print(json.dumps({"dataset": str(output), "train": len(train), "val": len(val),
                      "classes": sorted(classes), "groups_supplied": any(group for _, _, group in inputs)}, ensure_ascii=False))


def classification_report(model, validation, device, batch, imgsz):
    config = load_condition_config(ROOT)
    images = [(path, folder.name) for folder in sorted(validation.iterdir()) if folder.is_dir()
              for path in sorted(folder.rglob("*")) if path.is_file() and path.suffix.lower() in IMAGE_TYPES]
    names = set(model.names.values())
    if not images or any(label not in names for _, label in images):
        raise ValueError("Классы папки val отсутствуют в модели или папка пуста.")
    correct, counts = Counter(), Counter()
    total_defects = found_defects = flagged_defects = total_normal = flagged_normal = 0
    has_normal_class = any(name.casefold() in config.normal_labels for name in names)
    for start in range(0, len(images), batch):
        part = images[start:start + batch]
        predictions = model.predict([str(path) for path, _ in part], imgsz=imgsz, device=device, verbose=False)
        for (_, actual), prediction in zip(part, predictions):
            predicted = prediction.names[prediction.probs.top1]
            counts[actual] += 1
            correct[actual] += predicted == actual
            if has_normal_class:
                confidence = float(prediction.probs.top1conf)
                outcome = condition_result(config, (predicted, round(confidence, 4)), [])
                flagged = outcome["status"] in {"review", "repair"}
                if actual.strip().casefold() in config.normal_labels:
                    total_normal += 1
                    flagged_normal += flagged
                else:
                    total_defects += 1
                    found_defects += outcome["defect"] is not None
                    flagged_defects += flagged
    return {"images": len(images), "per_class_recall": {label: round(correct[label] / counts[label], 4) for label in counts},
            "defect_recall_at_threshold": round(found_defects / total_defects, 4) if total_defects else None,
            "defect_review_or_repair_recall": round(flagged_defects / total_defects, 4) if total_defects else None,
            "normal_review_or_repair_rate": round(flagged_normal / total_normal, 4) if total_normal else None,
            "confidence_threshold": config.classification_confidence}


def train(args):
    import torch
    from ultralytics import YOLO

    if args.epochs < 1 or args.batch < 1:
        raise ValueError("--epochs и --batch должны быть положительными.")
    if args.hours is not None and (not math.isfinite(args.hours) or args.hours <= 0):
        raise ValueError("--hours должен быть положительным числом.")
    device = args.device
    if device == "auto":
        device = "0" if torch.cuda.is_available() else "cpu"
    dataset = args.data.resolve(strict=True)
    checkpoint = (args.weights or ROOT / "models" / START_WEIGHTS[args.task]).resolve(strict=True)
    if args.task == "classify" and not all((dataset / split).is_dir() for split in ("train", "val")):
        raise ValueError("Для классификации подготовьте каталог с train/ и val/.")
    if args.task == "segment" and dataset.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError("Для сегментации нужен YOLO data.yaml с полигональной разметкой.")
    model = YOLO(str(checkpoint))
    if model.task != args.task:
        raise ValueError(f"Весам {checkpoint.name} соответствует задача {model.task}, а выбрана {args.task}.")
    imgsz = args.imgsz or (224 if args.task == "classify" else 640)
    model.train(data=str(dataset), epochs=args.epochs, imgsz=imgsz, batch=args.batch,
                device=device, project=str(args.output.resolve()), name=args.task, workers=0,
                seed=args.seed, amp=False, plots=False, time=args.hours)
    best = Path(model.trainer.best).resolve(strict=True)
    validator = YOLO(str(best))
    if args.task == "classify":
        metrics = classification_report(validator, dataset / "val", device, args.batch, imgsz)
    else:
        results = validator.val(data=str(dataset), split="val", imgsz=imgsz, batch=args.batch,
                                device=device, workers=0, plots=False)
        metrics = {name: round(float(value), 4) for name, value in results.results_dict.items()}
    report = best.parent.parent / "final-metrics.json"
    report.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.install:
        config = load_condition_config(ROOT)
        target = config.classifier if args.task == "classify" else config.segmenter
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=target.parent, prefix="mgts-weights-") as staging:
            copy = Path(staging) / target.name
            shutil.copy2(best, copy)
            copy.replace(target)
        print(f"Модель подключена: {target}")
    print(f"Веса: {best}\nМетрики: {report}")


def main():
    parser = argparse.ArgumentParser(description="Подготовка и обучение модели состояния инфраструктуры")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect-excel", help="Показать листы и заголовки Excel")
    inspect.add_argument("workbooks", type=Path, nargs="+")
    inspect.add_argument("--header-row", type=int, default=1)
    inspect.set_defaults(action=inspect_excel)

    excel = commands.add_parser("import-excel", help="Проверить Excel и собрать CSV-манифест")
    excel.add_argument("source", type=Path, help="Общая папка изображений")
    excel.add_argument("output", type=Path, help="Новый CSV-манифест")
    excel.add_argument("workbooks", type=Path, nargs="+")
    excel.add_argument("--sheet", help="Имя листа, если в книге их несколько")
    excel.add_argument("--header-row", type=int, default=1)
    excel.add_argument("--image-column", required=True, help="Столбец с именем файла или относительным путём")
    excel.add_argument("--class-column", help="Столбец класса. По умолчанию используется имя папки верхнего уровня")
    excel.add_argument("--group-column", help="Столбец с ID объекта или локации для разделения выборок")
    excel.add_argument("--status-column", help="Столбец со статусами норма, проверка или ремонт")
    excel.add_argument("--description-column", help="Столбец с описанием дефекта")
    excel.set_defaults(action=import_excel)

    prepare = commands.add_parser("prepare-cls")
    prepare.add_argument("source", type=Path)
    prepare.add_argument("output", type=Path)
    annotations = prepare.add_mutually_exclusive_group()
    annotations.add_argument("--groups", type=Path, help="CSV path,group для одной локации или объекта")
    annotations.add_argument("--manifest", type=Path, help="CSV path,class,group из import-excel")
    prepare.add_argument("--val-fraction", type=float, default=0.2)
    prepare.add_argument("--size", type=int, default=448, help="Максимальный размер подготовленного кадра")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.set_defaults(action=prepare_classification)

    fit = commands.add_parser("train")
    fit.add_argument("task", choices=tuple(START_WEIGHTS))
    fit.add_argument("--data", type=Path, required=True)
    fit.add_argument("--weights", type=Path)
    fit.add_argument("--output", type=Path, default=ROOT / "runs/final")
    fit.add_argument("--device", default="auto", help="auto выбирает GPU, если он доступен, или CPU")
    fit.add_argument("--epochs", type=int, default=30)
    fit.add_argument("--hours", type=float, help="Бюджет времени обучения в часах вместо числа эпох")
    fit.add_argument("--batch", type=int, default=8)
    fit.add_argument("--imgsz", type=int)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--install", action="store_true", help="Подключить лучшие веса в веб-приложении")
    fit.set_defaults(action=train)
    args = parser.parse_args()
    try:
        args.action(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    main()
