"""Validate the selected PyTorch backend before starting the prototype."""

import os
import sys
from pathlib import Path

import torch


def configure_backend():
    backend = os.environ.get("MGTS_TORCH_BACKEND", "rocm")
    if backend == "cpu":
        if torch.version.hip or torch.version.cuda:
            raise RuntimeError("CPU image must contain the PyTorch CPU build")
        os.environ["MGTS_DEVICE"] = "cpu"
        print(f"Backend: CPU, PyTorch {torch.__version__}", flush=True)
        return

    if backend != "rocm" or not torch.version.hip:
        raise RuntimeError("PyTorch with ROCm support is required")
    if not torch.cuda.is_available():
        raise RuntimeError("ROCm cannot see a GPU. On Linux, pass /dev/kfd and /dev/dri and their group IDs.")

    from torchvision.ops import nms

    device = torch.device("cuda:0")  # PyTorch exposes ROCm devices through torch.cuda.
    with torch.inference_mode():
        image = torch.ones((1, 3, 32, 32), device=device)
        kernel = torch.ones((8, 3, 3, 3), device=device)
        result = torch.nn.functional.conv2d(image, kernel)
        boxes = torch.tensor([[0, 0, 6, 6], [1, 1, 6, 6]], dtype=torch.float32, device=device)
        scores = torch.tensor([0.9, 0.5], device=device)
        indices = nms(boxes, scores, 0.5)
        torch.cuda.synchronize(device)
        if result[0, 0, 0, 0].item() != 27 or indices.tolist() != [0]:
            raise RuntimeError("Unexpected result from the GPU convolution or NMS")
    os.environ["MGTS_DEVICE"] = "0"
    os.environ["MGTS_REQUIRE_ROCM"] = "1"
    print(f"Backend: ROCm, GPU: {torch.cuda.get_device_name(device)}, "
          f"PyTorch {torch.__version__}, HIP {torch.version.hip}", flush=True)


def main():
    if os.environ.get("MGTS_TORCH_BACKEND", "rocm") == "rocm":
        cache = Path(os.environ.get("MGTS_RUNS_DIR", "/data/runs")).parent / "miopen"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MIOPEN_USER_DB_PATH", str(cache))
        os.environ.setdefault("MIOPEN_CUSTOM_CACHE_DIR", str(cache))
        os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    configure_backend()
    if len(sys.argv) > 1 and sys.argv[1:] != ["--check"]:
        os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        sys.exit(f"GPU startup check: {error}")
