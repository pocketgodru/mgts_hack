
"""Import organizer Excel annotations without assuming their sheet or column names."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import PurePosixPath, PureWindowsPath
from zipfile import BadZipFile


IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
STATUS_NAMES = {
    "normal": "normal", "норма": "normal", "исправно": "normal",
    "review": "review", "проверка": "review",
    "repair": "repair", "ремонт": "repair",
}


def cell_text(value):
    return "" if value is None else str(value).strip()


def class_name(value):
    label = cell_text(value)
    if (not label or label in {".", ".."} or any(ch in '\\/:*?"<>|' or ord(ch) < 32 for ch in label)
            or label.endswith(".") or PureWindowsPath(label).is_reserved()):
        raise ValueError(f"Неподходящее название класса: {label!r}")
    return label


def relative_path(value):
    path = PurePosixPath(cell_text(value).replace("\\", "/"))
    if not path.parts or PureWindowsPath(path.as_posix()).drive or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Нужен относительный путь к изображению: {value!r}")
    return path


def open_workbook(path):
    from openpyxl import load_workbook
    from openpyxl.utils.exceptions import InvalidFileException

    if path.suffix.lower() not in {".xlsx", ".xlsm"}:
        raise ValueError(f"{path.name}: сохраните книгу в формате .xlsx или .xlsm.")
    try:
        return load_workbook(path, read_only=True, data_only=True)
    except (BadZipFile, InvalidFileException) as exc:
        raise ValueError(f"Не удалось прочитать Excel: {path}") from exc


def sheet_headers(sheet, row_number):
    if row_number < 1:
        raise ValueError("Номер строки заголовков должен быть не меньше 1.")
    values = next(sheet.iter_rows(min_row=row_number, max_row=row_number, values_only=True), ())
    return [cell_text(value) for value in values]


def inspect_excel(args):
    for path in args.workbooks:
        workbook = open_workbook(path)
        try:
            sheets = {sheet.title: [name for name in sheet_headers(sheet, args.header_row) if name]
                      for sheet in workbook.worksheets}
            print(json.dumps({"книга": str(path), "листы": sheets}, ensure_ascii=False))
        finally:
            workbook.close()


def image_catalog(source):
    paths, names = {}, defaultdict(list)
    for image in sorted(source.rglob("*")):
        if image.suffix.lower() not in IMAGE_TYPES or not image.is_file() or not image.resolve().is_relative_to(source):
            continue
        relative = image.relative_to(source).as_posix()
        key = relative.casefold()
        if key in paths:
            raise ValueError(f"Совпадающие пути без учёта регистра: {paths[key]} и {relative}")
        paths[key] = relative
        names[image.name.casefold()].append(relative)
    return paths, names


def locate_image(value, workbook_folder, paths, names):
    path = relative_path(value)
    searches = [workbook_folder / path, path] if workbook_folder else [path]
    for candidate in searches:
        match = paths.get(candidate.as_posix().casefold())
        if match:
            return match
    matches = names.get(path.name.casefold(), []) if len(path.parts) == 1 else []
    if len(matches) > 1:
        raise ValueError(f"Укажите относительный путь для неоднозначного имени {value!r}.")
    if not matches:
        raise ValueError(f"Изображение не найдено: {value!r}.")
    return matches[0]


def select_columns(headers, names):
    indices = {}
    for purpose, column in names.items():
        if column is None:
            continue
        matches = [i for i, header in enumerate(headers) if header.casefold() == column.strip().casefold()]
        if len(matches) != 1:
            raise ValueError(f"Столбец {column!r} отсутствует или повторяется. Заголовки: {', '.join(filter(None, headers))}")
        indices[purpose] = matches[0]
    return indices


def workbook_records(path, args, source, paths, names):
    workbook = open_workbook(path)
    try:
        if args.sheet and args.sheet not in workbook.sheetnames:
            raise ValueError(f"{path.name}: лист {args.sheet!r} не найден. Доступны: {', '.join(workbook.sheetnames)}")
        if not args.sheet and len(workbook.sheetnames) != 1:
            raise ValueError(f"{path.name}: укажите --sheet. Доступны: {', '.join(workbook.sheetnames)}")
        sheet = workbook[args.sheet or workbook.sheetnames[0]]
        columns = select_columns(sheet_headers(sheet, args.header_row), {
            "image": args.image_column, "label": args.class_column, "group": args.group_column,
            "status": args.status_column, "description": args.description_column,
        })
        parent = path.resolve().parent
        folder = PurePosixPath(parent.relative_to(source).as_posix()) if parent.is_relative_to(source) else None
        for number, values in enumerate(sheet.iter_rows(min_row=args.header_row + 1, values_only=True), args.header_row + 1):
            row = {field: cell_text(values[index]) if index < len(values) else "" for field, index in columns.items()}
            if not any(row.values()):
                continue
            context = f"{path.name}, {sheet.title}, строка {number}"
            try:
                image = locate_image(row["image"], folder, paths, names)
                parts = PurePosixPath(image).parts
                label = class_name(row.get("label", parts[0] if len(parts) > 1 else ""))
                group = row.get("group", "")
                if args.group_column and not group:
                    raise ValueError("Группа объекта или локации не заполнена.")
                raw_status = row.get("status", "").casefold()
                status = STATUS_NAMES.get(raw_status, "")
                if raw_status and not status:
                    raise ValueError(f"Неизвестный статус {row['status']!r}.")
                yield context, {"path": image, "class": label, "group": group, "status": status,
                                "description": row.get("description", "")}, None
            except ValueError as exc:
                yield context, None, str(exc)
    finally:
        workbook.close()


def import_excel(args):
    source, output = args.source.resolve(), args.output.resolve()
    if not source.is_dir():
        raise ValueError(f"Нет папки с изображениями: {source}")
    if output.suffix.lower() != ".csv" or output.exists():
        raise ValueError("Укажите новый файл манифеста с расширением .csv.")
    paths, names = image_catalog(source)
    rows, states, errors, errors_count = {}, {}, [], 0
    for workbook in args.workbooks:
        for context, row, problem in workbook_records(workbook, args, source, paths, names):
            if problem is None:
                image, label, status = row["path"], row["class"], row["status"]
                if image in rows and rows[image] != row:
                    problem = f"Противоречивые записи для {image}."
                elif status and states.setdefault(label, status) != status:
                    problem = f"Разные статусы у класса {label!r}."
                else:
                    rows[image] = row
            if problem:
                errors_count += 1
                if len(errors) < 20:
                    errors.append(f"{context}: {problem}")
    if errors:
        raise ValueError(f"Ошибок в Excel: {errors_count}.\n" + "\n".join(errors))
    if not rows:
        raise ValueError("В выбранных листах нет строк с изображениями.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("path", "class", "group", "status", "description"))
        writer.writeheader()
        writer.writerows(rows[image] for image in sorted(rows))
    print(json.dumps({"manifest": str(output), "images": len(rows), "images_without_row": len(paths) - len(rows),
                      "classes": Counter(row["class"] for row in rows.values()), "class_statuses": states},
                     ensure_ascii=False))


def manifest_images(source, manifest):
    records, known = [], set()
    with manifest.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"path", "class", "group"}.issubset(reader.fieldnames or []):
            raise ValueError("Манифест должен содержать колонки path,class,group.")
        for row in reader:
            relative = relative_path(row["path"])
            image = (source / relative).resolve()
            if (not image.is_relative_to(source) or image.suffix.lower() not in IMAGE_TYPES
                    or not image.is_file() or image in known):
                raise ValueError(f"Некорректный или повторяющийся путь в манифесте: {relative}")
            known.add(image)
            records.append((image, class_name(row["class"]), cell_text(row["group"])))
    return records
