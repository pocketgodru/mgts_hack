#!/usr/bin/env python3
import asyncio
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import ExifTags, Image

from condition import assess_objects, load_condition_config
from odn import active_detector
from registry import (find_objects, group_observations, locate_video_frames, read_frame_csv, read_video_gps,
                      status_counts, valid_coordinates, write_registry)
from video_tracking import VIDEO_EXTENSIONS, cluster_evidence, extract_video_frames, track_poles

ROOT = Path(__file__).resolve().parent
WEB_DIR = ROOT / "web"
RUNS_DIR = ROOT / (os.getenv("MGTS_RUNS_DIR") or "web_runs")
DEMO_DIR = ROOT / "demo"
DEMO_ROUTE_DATA = DEMO_DIR / "routes.json"
DEMO_FRAMES = DEMO_DIR / "frames"
ARCHIVE_CATALOG = ROOT / "runs/archive_panorama_scan/catalog.json"
PANORAMA_FRAMES = ROOT / "panos_out/frames"
ARCHIVE_FRAME_ROOTS = {"archive": PANORAMA_FRAMES, "near": ROOT / "panos_out/near_towers_25m/frames"}
MANHOLE_WEIGHTS = ROOT / "models/manhole-yolo26s.pt"
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILES = 100
MAX_IMAGE_SIZE = 30 * 1024 * 1024
MAX_VIDEO_SIZE = 150 * 1024 * 1024
MAX_CSV_SIZE = 2 * 1024 * 1024
RUN_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
GPS_IFD = next(key for key, value in ExifTags.TAGS.items() if value == "GPSInfo")

RUNS_DIR.mkdir(parents=True, exist_ok=True)
app = FastAPI(title="МГТС · Мониторинг инфраструктуры", version="0.2.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")
inference_lock = asyncio.Lock()


def gps_decimal(values, reference):
    degrees, minutes, seconds = (float(value) for value in values)
    result = degrees + minutes / 60 + seconds / 3600
    return -result if reference in {"S", "W"} else result


def read_image_metadata(path):
    metadata = {"latitude": None, "longitude": None, "width": None, "height": None, "coordinate_source": None}
    try:
        with Image.open(path) as image:
            metadata["width"], metadata["height"] = image.size
            exif = image.getexif()
            if exif.get(274) in {5, 6, 7, 8}:
                metadata["width"], metadata["height"] = image.height, image.width
            gps = exif.get_ifd(GPS_IFD)
            if all(key in gps for key in (1, 2, 3, 4)):
                latitude = round(gps_decimal(gps[2], gps[1]), 7)
                longitude = round(gps_decimal(gps[4], gps[3]), 7)
                if valid_coordinates(latitude, longitude):
                    metadata.update(latitude=latitude, longitude=longitude, coordinate_source="exif")
    except (OSError, ValueError, TypeError, ZeroDivisionError, SyntaxError):
        pass
    return metadata


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run_directory(run_id):
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=404, detail="Прогон не найден")
    directory = RUNS_DIR / run_id
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Прогон не найден")
    return directory


def build_result(run_id, directory, start, device, condition, model, fallback=False, video_fps=None, timings=None):
    input_dir = directory / "input"
    output_dir = directory / "output"
    frames = []
    frame_gps = {}
    manifest = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    for path in sorted(input_dir.iterdir()):
        if path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        metadata = read_image_metadata(path)
        source = manifest.get(path.name, {})
        if source.get("latitude") is not None:
            metadata.update(latitude=source["latitude"], longitude=source["longitude"],
                            coordinate_source=source["coordinate_source"])
        frame_gps[path.name] = metadata
        original = f"/runs/{run_id}/input/{path.name}"
        rendered = output_dir / "labeled" / path.name
        frames.append({
            "name": path.name,
            "label": source.get("label", path.name),
            "original_url": original,
            "labeled_url": f"/runs/{run_id}/output/labeled/{path.name}" if rendered.is_file() else original,
            "object_id": source.get("object_id"),
            "object_type": source.get("object_type"),
            **metadata,
        })

    detections = []
    for row in read_csv(output_dir / "detections.csv"):
        detections.append({
            "frame": row["frame"],
            "class": row["class"],
            "confidence": float(row["conf"]),
            "bbox": [float(row[key]) for key in ("x1", "y1", "x2", "y2")],
        })

    candidates = []
    for row in read_csv(output_dir / "odn_candidates.csv"):
        candidates.append({
            **row,
            "pole_conf": float(row["pole_conf"]) if row["pole_conf"] else None,
            "max_equipment_conf": float(row["max_equipment_conf"]),
        })

    sites = []
    for row in read_csv(output_dir / "odn_sites.csv"):
        evidence_frames = row["evidence_frames"].split("|") if row["evidence_frames"] else []
        coordinates = next((frame_gps[name] for name in evidence_frames
                            if name in frame_gps and frame_gps[name]["latitude"] is not None), None)
        sites.append({
            "site": row["site"],
            "frames_with_equipment": int(row["frames_with_equipment"]),
            "evidence_frames": evidence_frames,
            "equipment_classes": row["equipment_classes"].split("|") if row["equipment_classes"] else [],
            "max_equipment_conf": float(row["max_equipment_conf"]),
            "max_pole_conf": float(row["max_pole_conf"]),
            "latitude": coordinates["latitude"] if coordinates else None,
            "longitude": coordinates["longitude"] if coordinates else None,
            "condition": "Не оценивалось",
        })

    video_ids = None
    candidate_csv = "odn_candidates.csv"
    if video_fps is not None:
        tracking_start = time.perf_counter()
        names = [frame["name"] for frame in frames]
        pole_tracks = track_poles(names, input_dir, detections, sample_fps=video_fps)
        candidates, video_ids = cluster_evidence(names, input_dir, detections, candidates, video_fps, pole_tracks)
        candidate_csv = "video_candidates.csv"
        with (output_dir / candidate_csv).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=["frame", "pole_conf", "equipment_classes",
                                                       "max_equipment_conf", "x1", "y1", "x2", "y2", "association"])
            writer.writeheader()
            writer.writerows(candidates)
        for item in detections:
            if item["class"] == "pole":
                item["track_id"] = pole_tracks.get((item["frame"], tuple(item["bbox"])))
        for frame in frames:
            frame["video_time"] = manifest[frame["name"]]["video_time"]
        if timings is not None:
            timings["tracking_seconds"] = time.perf_counter() - tracking_start
    registry_candidates = candidates if video_fps is None else [
        row for row in candidates if row["association"] != "equipment_only"
    ]
    observations = find_objects(frames, detections, registry_candidates, video_ids)
    devices = condition.available()
    assess_objects(observations, input_dir, condition, device)
    registry = group_observations(observations)
    write_registry(output_dir / "registry.csv", registry)
    action_rows = [row for row in registry if row["status"] in {"repair", "review"}]
    write_registry(output_dir / "inspection.csv", action_rows)
    counts = status_counts(registry)
    result = {
        "run_id": run_id,
        "frames": frames,
        "detections": detections,
        "candidates": candidates,
        "sites": sites,
        "observations": observations,
        "registry": registry,
        "stats": {
            "frames": len(frames),
            "detections": len(detections),
            "sites": len(sites),
            "equipment": sum(item["class"] in {"antenna", "rru", "microwave"} for item in detections),
            "georeferenced_sites": sum(site["latitude"] is not None for site in sites),
            "objects": len(registry),
            "manholes": sum(row["kind"] == "manhole" for row in registry),
            "georeferenced_objects": sum(row["latitude"] is not None for row in registry),
            "actions": len(action_rows),
            **counts,
        },
        "downloads": {
            "detections": f"/runs/{run_id}/output/detections.csv",
            "candidates": f"/runs/{run_id}/output/{candidate_csv}",
            "sites": f"/runs/{run_id}/output/odn_sites.csv",
            "registry": f"/runs/{run_id}/output/registry.csv",
            "inspection": f"/runs/{run_id}/output/inspection.csv",
        },
        "elapsed_seconds": round(time.perf_counter() - start, 2),
        "model": model,
        "condition_model": {name: getattr(condition, name).stem for name, active in devices.items() if active} or None,
        "device": device,
        "backend": inference_backend(device),
        "cpu_fallback": fallback,
    }
    pending = directory / "result.tmp"
    pending.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(directory / "result.json")
    return result


def inference_device():
    configured = os.getenv("MGTS_DEVICE", "auto").lower()
    if configured == "auto":
        if torch.cuda.is_available():
            return "0"
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if configured in {"mps", "mps:0"}:
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return configured


def inference_backend(device):
    if device in {"cpu", "mps"}:
        return device
    return "rocm" if torch.version.hip else "cuda"


def device_failure(error):
    return any(code in error.lower() for code in ("miopenstatus", "cuda error", "hip error",
                                                   "invalid cuda", "out of memory", "device-side assert",
                                                   "invalid mps", "mps device", "mps backend", "mpsgraph",
                                                   "not implemented for 'mps'"))


def detector_command(input_dir, output_dir, weights, *, tile=False, video=False):
    command = [
        sys.executable, str(ROOT / "infer_detector.py"), str(input_dir),
        "--out", str(output_dir), "--weights", str(weights), "--batch", "4",
    ]
    if MANHOLE_WEIGHTS.is_file():
        command.extend(["--manhole-weights", str(MANHOLE_WEIGHTS)])
    if tile:
        command.append("--tile")
    if video:
        command.extend(["--class-conf", "pole=0.10", "antenna=0.10", "rru=0.05", "microwave=0.25"])
    return command


async def execute_inference(run_id, tile=False, video_fps=None):
    directory = run_directory(run_id)
    input_dir = directory / "input"
    if not any(input_dir.iterdir()):
        raise HTTPException(status_code=400, detail="Нет изображений для анализа")
    device = inference_device()
    fallback = False
    try:
        condition = load_condition_config(ROOT)
    except (OSError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=f"Настройки модели состояния: {exc}") from exc
    model, weights = active_detector()
    command = detector_command(input_dir, directory / "output", weights, tile=tile, video=video_fps is not None)
    start = time.perf_counter()
    async with inference_lock:
        log_path = directory / "inference.log"
        require_rocm = os.getenv("MGTS_REQUIRE_ROCM") == "1"
        devices = (device,) if require_rocm or device == "cpu" else (device, "cpu")
        for selected_device in devices:
            fallback = selected_device != device
            try:
                process = await asyncio.to_thread(
                    subprocess.run, [*command, "--device", selected_device], cwd=ROOT, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                    timeout=1800, check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise HTTPException(status_code=504, detail="Превышено время анализа") from exc
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n[{selected_device}]\n{process.stdout}\n")
            if process.returncode:
                if not require_rocm and not fallback and selected_device != "cpu" and device_failure(process.stdout):
                    shutil.rmtree(directory / "output", ignore_errors=True)
                    continue
                raise HTTPException(status_code=500, detail=f"Ошибка детектора. Журнал анализа: {run_id}.")
            try:
                return await asyncio.to_thread(
                    build_result, run_id, directory, start, selected_device, condition, model, fallback, video_fps,
                )
            except (OSError, RuntimeError, ValueError) as exc:
                with log_path.open("a", encoding="utf-8") as log:
                    log.write(f"\n{type(exc).__name__}: {exc}\n")
                if not require_rocm and not fallback and selected_device != "cpu" and device_failure(str(exc)):
                    shutil.rmtree(directory / "output", ignore_errors=True)
                    continue
                raise HTTPException(status_code=500, detail=f"Ошибка оценки состояния или экспорта. Журнал анализа: {run_id}.") from exc


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/routes")
def routes_page():
    return FileResponse(WEB_DIR / "routes.html")


@app.get("/api/health")
def health():
    try:
        condition = load_condition_config(ROOT).available()
        config_error = None
    except (OSError, TypeError, ValueError) as exc:
        condition = {"classifier": False, "segmenter": False}
        config_error = str(exc)
    device = inference_device()
    model, weights = active_detector()
    return {
        "status": "ok",
        "model_available": weights.is_file(),
        "model": model,
        "gpu": torch.cuda.get_device_name(0) if device != "cpu" and torch.cuda.is_available() else None,
        "device": device,
        "backend": inference_backend(device),
        "manhole_model": "YOLO26s · люки" if MANHOLE_WEIGHTS.is_file() else None,
        "manhole_model_available": MANHOLE_WEIGHTS.is_file(),
        "condition_model_available": any(condition.values()),
        "condition_models": condition,
        "condition_config_error": config_error,
    }


@app.get("/api/routes")
def demonstration_routes():
    return json.loads(DEMO_ROUTE_DATA.read_text(encoding="utf-8"))


@app.get("/api/routes/archive")
def panorama_archive():
    if not ARCHIVE_CATALOG.is_file():
        raise HTTPException(status_code=404, detail="Архив панорам пока недоступен")
    return FileResponse(ARCHIVE_CATALOG, media_type="application/json")


@app.get("/api/routes/archive/frame/{source}/{frame}")
def panorama_archive_frame(source: str, frame: str):
    directory = ARCHIVE_FRAME_ROOTS.get(source)
    if directory is not None and re.fullmatch(r"site_\d{5,}_f[0-3]\.jpg", frame):
        path = directory / frame
        if path.is_file():
            return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Кадр архива не найден")


@app.get("/api/routes/frame/{frame}")
def demonstration_frame(frame: str):
    if Path(frame).name == frame and any(
        site["frame"] == frame for route in demonstration_routes()["routes"] for site in route["sites"]
    ):
        path = DEMO_FRAMES / frame
        if path.is_file():
            return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="Кадр маршрута не найден")


@app.post("/api/analyze")
async def analyze(files: list[UploadFile] = File(...), metadata: UploadFile | None = File(None), tile: bool = False):
    if not files or len(files) > MAX_FILES:
        raise HTTPException(status_code=400, detail=f"Загрузите от 1 до {MAX_FILES} изображений")
    run_id = uuid.uuid4().hex
    directory = RUNS_DIR / run_id
    input_dir = directory / "input"
    input_dir.mkdir(parents=True)
    used_names = set()
    uploads = [(upload, (upload.filename or "image.jpg").replace("\\", "/").rsplit("/", 1)[-1]) for upload in files]
    try:
        rows = {}
        if metadata is not None:
            data = await metadata.read(MAX_CSV_SIZE + 1)
            if len(data) > MAX_CSV_SIZE:
                raise HTTPException(status_code=413, detail="CSV должен занимать не более 2 МБ")
            try:
                rows = read_frame_csv(data, Counter(name for _, name in uploads))
            except (UnicodeError, ValueError, csv.Error) as exc:
                raise HTTPException(status_code=400, detail=f"CSV метаданных: {exc}") from exc
        manifest = {}
        for upload, original_name in uploads:
            extension = Path(original_name).suffix.lower()
            if extension not in ALLOWED_EXTENSIONS:
                raise HTTPException(status_code=400, detail=f"Неподдерживаемый файл: {original_name}")
            stem = re.sub(r"[^A-Za-z0-9_-]", "_", Path(original_name).stem)[:90].strip("_") or "image"
            if stem.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
                stem = f"image_{stem}"
            name = f"{stem}{extension}"
            suffix = 2
            while name.casefold() in used_names:
                name = f"{stem}_{suffix}{extension}"
                suffix += 1
            used_names.add(name.casefold())
            image_path = input_dir / name
            length = 0
            with image_path.open("wb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    length += len(chunk)
                    if length > MAX_IMAGE_SIZE:
                        raise HTTPException(status_code=413, detail=f"Изображение превышает 30 МБ: {original_name}")
                    handle.write(chunk)
            try:
                with Image.open(image_path) as image:
                    if image.format not in {"JPEG", "PNG"}:
                        raise ValueError("Неподдерживаемый тип изображения")
                    image.verify()
            except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
                raise HTTPException(status_code=400, detail=f"Не удалось прочитать изображение: {original_name}") from exc
            manifest[name] = {"label": original_name, **rows.get(original_name, {})}
        (directory / "source.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        for upload, _ in uploads:
            await upload.close()
        if metadata is not None:
            await metadata.close()
    return await execute_inference(run_id, tile)


@app.post("/api/analyze/video")
async def analyze_video(file: UploadFile = File(...), tile: bool = False, gps: UploadFile | None = File(None)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in VIDEO_EXTENSIONS:
        await file.close()
        if gps is not None:
            await gps.close()
        raise HTTPException(status_code=400, detail="Поддерживаются MP4, MOV и WebM")
    run_id = uuid.uuid4().hex
    directory = RUNS_DIR / run_id
    input_dir = directory / "input"
    input_dir.mkdir(parents=True)
    upload_path = directory / f"upload{extension}"
    try:
        samples = None
        if gps is not None:
            data = await gps.read(MAX_CSV_SIZE + 1)
            if len(data) > MAX_CSV_SIZE:
                raise HTTPException(status_code=413, detail="GPS CSV должен занимать не более 2 МБ")
            try:
                samples = read_video_gps(data)
            except (ValueError, csv.Error) as exc:
                raise HTTPException(status_code=400, detail=f"GPS CSV: {exc}") from exc
        length = 0
        with upload_path.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                length += len(chunk)
                if length > MAX_VIDEO_SIZE:
                    raise HTTPException(status_code=413, detail="Видео превышает 150 МБ")
                handle.write(chunk)
        try:
            manifest, fps = await asyncio.to_thread(extract_video_frames, upload_path, input_dir, max_frames=MAX_FILES)
            if samples is not None:
                locate_video_frames(manifest, samples)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        (directory / "source.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        await file.close()
        if gps is not None:
            await gps.close()
        upload_path.unlink(missing_ok=True)
    return await execute_inference(run_id, tile, video_fps=fps)


@app.post("/api/demo")
async def demo(tile: bool = False):
    manifest = json.loads((DEMO_DIR / "inputs.json").read_text(encoding="utf-8"))
    examples = [DEMO_FRAMES / name for name in manifest]
    if not all(path.is_file() for path in examples):
        raise HTTPException(status_code=404, detail="Демонстрационные кадры не найдены")
    run_id = uuid.uuid4().hex
    input_dir = RUNS_DIR / run_id / "input"
    input_dir.mkdir(parents=True)
    for path in examples:
        shutil.copy2(path, input_dir / path.name)
    (input_dir.parent / "source.json").write_text(json.dumps(manifest), encoding="utf-8")
    return await execute_inference(run_id, tile)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    directory = run_directory(run_id)
    result_path = directory / "result.json"
    if not result_path.is_file():
        raise HTTPException(status_code=404, detail="Результат ещё не готов")
    return json.loads(result_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_app:app", host="127.0.0.1", port=8000, reload=False)
