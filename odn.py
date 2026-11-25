"""Weights for the detector used by the application and the CLI."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
YOLO26_WEIGHTS = ROOT / "models/odn-yolo26s.pt"


def active_detector():
    return "YOLO26s", YOLO26_WEIGHTS
