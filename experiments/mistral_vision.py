"""Classify photos locally and request a second opinion only for uncertain frames."""

import argparse
import base64
import io
import json
import os
from pathlib import Path

from PIL import Image, ImageOps


DEFAULT_MODEL = "mistral-small-latest"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = ROOT / "models/dss-yolo26x-cls.pt"
IMAGE_SIZE = 640
STATES = {"normal", "damaged", "uncertain"}
PROMPT = (
    "Оцени только видимые признаки состояния шкафа уличного оборудования. "
    "Проверь, видны ли деформация корпуса или двери, повреждение креплений, "
    "отслоение краски или коррозия. Если объект плохо виден или деталей недостаточно, "
    "верни uncertain. Не делай выводов о работе оборудования по его внешнему виду. "
    "Ответь JSON с полями status (normal, damaged или uncertain), "
    "visible_signs (массив кратких строк на русском, пустой при отсутствии признаков) "
    "и reason (одно короткое предложение на русском)."
)


def image_url(path):
    with Image.open(path) as source:
        photo = ImageOps.exif_transpose(source).convert("RGB")
    photo.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    photo.save(buffer, format="JPEG", quality=93)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def second_opinion(client, path, model):
    answer = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": image_url(path)},
        ]}],
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=300,
        timeout_ms=20_000,
    )
    if not answer.choices or not isinstance(answer.choices[0].message.content, str):
        raise ValueError("Mistral не вернула текстовый результат")
    result = json.loads(answer.choices[0].message.content)
    if not isinstance(result, dict):
        raise ValueError("Ожидался объект JSON")
    status, signs, reason = result.get("status"), result.get("visible_signs"), result.get("reason")
    if (
        not isinstance(status, str) or status not in STATES
        or not isinstance(signs, list) or any(not isinstance(sign, str) for sign in signs)
        or not isinstance(reason, str)
    ):
        raise ValueError("Mistral вернула ответ вне ожидаемой схемы")
    return {
        "source": "mistral_vision",
        "model": model,
        "status": status,
        "visible_signs": signs,
        "reason": reason,
    }


def predict_local(detector, path, device):
    with Image.open(path) as source:
        frame = ImageOps.pad(
            ImageOps.exif_transpose(source).convert("RGB"),
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS, color=(114, 114, 114),
        )
    result = detector.predict(frame, imgsz=IMAGE_SIZE, device=device, verbose=False)[0]
    return result.names[result.probs.top1], float(result.probs.top1conf)


def review_photo(path, prediction, threshold, get_client, model):
    label, confidence = prediction
    uncertain = confidence < threshold
    row = {
        "frame": path.name,
        "prediction": label,
        "confidence": round(confidence, 4),
        "status": "review" if uncertain else label,
        "second_opinion": None,
    }
    if uncertain:
        try:
            row["second_opinion"] = second_opinion(get_client(), path, model)
        except Exception as error:
            # SDK exceptions can include the request payload. Record only the type.
            row["second_opinion_error"] = type(error).__name__
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Анализирует фото YOLO локально. Отправляет в Mistral только кадры ниже порога."
    )
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.65)
    args = parser.parse_args()
    if not 0 < args.threshold <= 1:
        parser.error("--threshold должен быть больше 0 и не выше 1")

    import torch
    from ultralytics import YOLO

    device = ("0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    detector = YOLO(str(args.weights.resolve(strict=True)))
    if detector.task != "classify" or set(detector.names.values()) != {"normal", "damaged"}:
        parser.error("Нужны веса классификатора с классами normal и damaged")
    client = None

    def get_client():
        nonlocal client
        if client is None:
            from mistralai.client import Mistral

            client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        return client

    results = [
        review_photo(path, predict_local(detector, path, device), args.threshold, get_client, args.model)
        for path in args.images
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
