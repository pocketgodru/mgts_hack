FROM python:3.12-slim-trixie@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ARG MGTS_TORCH_BACKEND=rocm
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MGTS_TORCH_BACKEND=${MGTS_TORCH_BACKEND} \
    MGTS_RUNS_DIR=/data/runs \
    YOLO_CONFIG_DIR=/data/config \
    YOLO_OFFLINE=true \
    YOLO_AUTOINSTALL=false \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

RUN apt-get update \
    && apt-get install -y --no-install-recommends libdrm-amdgpu1 libelf1t64 libglib2.0-0t64 libgomp1 libnuma1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app \
    && install -d -o app -g app /data /data/runs /data/config

WORKDIR /app
COPY requirements.runtime.txt ./
RUN --mount=type=cache,target=/root/.cache/pip case "$MGTS_TORCH_BACKEND" in \
        rocm) python -m pip install --timeout 90 --retries 8 --constraint requirements.runtime.txt --index-url https://stable.repo.amd.com/rocm/whl-next/ \
                  --extra-index-url https://pypi.org/simple \
                  'torch[device-gfx1201]==2.15.0+rocm10.2.0' 'torchvision[device-gfx1201]==0.29.2+rocm10.2.0' ;; \
        cpu) python -m pip install --timeout 90 --retries 8 --constraint requirements.runtime.txt --index-url https://download.pytorch.org/whl/cpu \
                  'torch==2.12.0+cpu' 'torchvision==0.27.0+cpu' ;; \
        *) echo "MGTS_TORCH_BACKEND must be rocm or cpu" >&2; exit 2 ;; \
    esac
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --timeout 90 --retries 8 --requirement requirements.runtime.txt \
    && python -m pip check \
    && python -c "import cv2, os, torch, torchvision; assert (torch.version.hip is not None) == (os.environ['MGTS_TORCH_BACKEND'] == 'rocm'); assert torchvision.extension._has_ops()"

COPY docker_entrypoint.py web_app.py condition.py infer_detector.py odn.py registry.py video_tracking.py ./
COPY web/ ./web/
COPY models/odn-yolo26s.pt models/manhole-yolo26s.pt models/condition.example.json models/dss-yolo26x-cls.pt ./models/
COPY demo/ ./demo/

USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD ["python", "-c", "import json, os, urllib.request; d = json.load(urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)); assert d['model_available'] and d['manhole_model_available'] and (os.environ['MGTS_TORCH_BACKEND'] == 'cpu' or d['gpu'])"]
ENTRYPOINT ["python", "/app/docker_entrypoint.py"]
CMD ["python", "-m", "uvicorn", "web_app:app", "--host", "0.0.0.0", "--port", "8000"]
